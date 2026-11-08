"""Multi-model routing for using different LLMs for different purposes.

A ``ModelRouter`` selects which ``LLMBackend`` to use based on the current
purpose (e.g. 'reasoning', 'recovery', 'compaction').  ``CostTracker`` wraps
any backend and accumulates estimated spend.  ``RoutingLLMBackend`` exposes
the whole routing system as a single drop-in ``LLMBackend``.

Typical usage::

    profiles = {
        "reasoning": ModelProfile("gpt-4o", gpt4_backend, cost_per_1k_input=0.005),
        "recovery": ModelProfile("gpt-4o-mini", mini_backend, cost_per_1k_input=0.00015),
    }
    router = SimpleRouter(profiles, default_profile=profiles["reasoning"])
    llm = RoutingLLMBackend(router)
    # Use llm as a drop-in LLMBackend - it routes internally.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Iterator, Protocol, cast, runtime_checkable

from looplet.backend_contract import (
    FORWARDED_CAPABILITIES,
    MUTABLE_CAPABILITIES,
    BackendRequestPolicy,
    TerminalLLMError,
    backend_attribute,
    backend_request_policy,
    invoke_backend,
    single_model_request_active,
)
from looplet.types import LLMBackend

logger = logging.getLogger(__name__)


# ── ModelProfile ─────────────────────────────────────────────────


@dataclass
class ModelProfile:
    """Metadata about a specific model and its backend.

    Args:
        name: Human-readable identifier (e.g. 'gpt-4o', 'claude-3-haiku').
        backend: The ``LLMBackend`` to use for this model.
        cost_per_1k_input: USD cost per 1,000 input tokens (default 0.0).
        cost_per_1k_output: USD cost per 1,000 output tokens (default 0.0).
        context_window: Maximum context length in tokens (default 100,000).
        strengths: List of purpose tags this model excels at, e.g.
            ``['reasoning', 'code', 'fast']``.
    """

    name: str
    backend: LLMBackend
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    context_window: int = 100_000
    strengths: list[str] = field(default_factory=list)


# ── ModelRouter Protocol ─────────────────────────────────────────


@runtime_checkable
class ModelRouter(Protocol):
    """Protocol for purpose-based model selection.

    Implementations map purpose strings to ``LLMBackend`` instances so
    the loop engine can call different models for different stages.
    """

    def select(self, purpose: str, **kwargs: Any) -> LLMBackend:
        """Return the best ``LLMBackend`` for the given purpose."""
        ...


# ── SimpleRouter ──────────────────────────────────────────────────


class SimpleRouter:
    """Routes by purpose string with a fallback default.

    Args:
        profiles: Mapping of purpose string → ``ModelProfile``.
            Standard purposes: 'reasoning', 'recovery', 'compaction', 'sub_agent'.
        default_profile: Used when the requested purpose has no mapping.
    """

    def __init__(
        self,
        profiles: dict[str, ModelProfile],
        *,
        default_profile: ModelProfile,
    ) -> None:
        self._profiles = profiles
        self._default = default_profile

    def select(self, purpose: str, **kwargs: Any) -> LLMBackend:
        profile = self._profiles.get(purpose, self._default)
        return profile.backend


# ── FallbackRouter ───────────────────────────────────────────────


class _FallbackLLM:
    """Transparent LLMBackend that retries with a fallback on any exception."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active = primary
        # Expose native tools only when either backend can satisfy the full
        # fallback contract; otherwise the loop uses text generation safely.
        if hasattr(primary, "generate_with_tools") and hasattr(fallback, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl

    _looplet_manages_retries = True

    @property
    def request_policy(self) -> BackendRequestPolicy:
        primary_policy = backend_request_policy(self._primary)
        return BackendRequestPolicy(
            manages_retries=True,
            supports_single_model_request=(
                not primary_policy.manages_retries or primary_policy.supports_single_model_request
            ),
        )

    @property
    def _supports_single_model_request(self) -> bool:
        return self.request_policy.supports_single_model_request

    def _check_admission(self) -> None:
        if single_model_request_active() and not self.request_policy.supports_single_model_request:
            raise TerminalLLMError(
                "Retry-owning backend does not support single-request admission",
                termination_reason="unsupported_context",
            )

    def __getattr__(self, name: str) -> Any:
        if name == "stream":
            if hasattr(self._primary, name) and hasattr(self._fallback, name):
                return self._stream_impl
            raise AttributeError(name)
        if name == "stream_in_loop":
            return getattr(self._primary, name, True) and getattr(self._fallback, name, True)
        backend = self._primary if name == "request_budgets" else self._active
        return backend_attribute(backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in MUTABLE_CAPABILITIES and "_active" in self.__dict__:
            setattr(self._active, name, value)
        else:
            object.__setattr__(self, name, value)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        options = {
            **options,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        self._check_admission()
        self._active = self._primary
        try:
            response = invoke_backend(self._primary.generate, prompt, options)
            if inspect.isawaitable(response):
                return self._await_primary(response, "generate", prompt, options)
            return response
        except Exception as exc:
            if single_model_request_active():
                raise
            logger.warning("Primary LLM failed (%s); switching to fallback", exc)
            self._active = self._fallback
            return invoke_backend(self._fallback.generate, prompt, options)

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        options = {
            **options,
            "tools": tools,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        self._check_admission()
        self._active = self._primary
        try:
            if hasattr(self._primary, "generate_with_tools"):
                response = invoke_backend(self._primary.generate_with_tools, prompt, options)  # pyright: ignore[reportAttributeAccessIssue]
                if inspect.isawaitable(response):
                    return self._await_primary(response, "generate_with_tools", prompt, options)
                return response
        except Exception as exc:
            if single_model_request_active():
                raise
            logger.warning(
                "Primary LLM generate_with_tools failed (%s); switching to fallback", exc
            )
        if hasattr(self._fallback, "generate_with_tools"):
            self._active = self._fallback
            return invoke_backend(self._fallback.generate_with_tools, prompt, options)  # pyright: ignore[reportAttributeAccessIssue]
        raise AttributeError("Neither primary nor fallback supports generate_with_tools")

    async def _await_primary(
        self, response: Any, method_name: str, prompt: str, options: dict[str, Any]
    ) -> Any:
        try:
            return await response
        except Exception as exc:
            if single_model_request_active():
                raise
            logger.warning("Primary LLM %s failed (%s); switching to fallback", method_name, exc)
            self._active = self._fallback
            method = getattr(self._fallback, method_name)
            if inspect.iscoroutinefunction(method):
                pending = invoke_backend(method, prompt, options)
            else:
                from looplet.checkpoint import _run_persistence

                pending = await _run_persistence(invoke_backend, method, prompt, options)
            return await pending if inspect.isawaitable(pending) else pending

    def _stream_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        self._check_admission()
        options = {
            **options,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        self._active = self._primary
        return _FallbackStream(self, prompt, options)

    def _stream_primary(self, prompt: str, options: dict[str, Any]) -> Iterator[str]:
        delivered = False
        stream = None
        try:
            try:
                stream = invoke_backend(getattr(self._primary, "stream"), prompt, options)
                for chunk in stream:
                    delivered = True
                    yield chunk
            except Exception:
                if delivered or single_model_request_active():
                    raise
                primary_stream = stream
                stream = None
                close = getattr(primary_stream, "close", None)
                if close is not None:
                    close()
                self._active = self._fallback
                stream = invoke_backend(getattr(self._fallback, "stream"), prompt, options)
                for chunk in stream:
                    yield chunk
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    async def _async_stream_chunks(
        self, backend: Any, prompt: str, options: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        from looplet.checkpoint import _run_persistence

        context = contextvars.copy_context()
        stream: Any = None

        def create_stream() -> None:
            nonlocal stream
            stream = invoke_backend(getattr(backend, "stream"), prompt, options)

        try:
            await _run_persistence(context.run, create_stream)
            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    yield chunk
            else:
                iterator = iter(stream)
                end = object()
                while True:
                    chunk = await _run_persistence(context.run, next, iterator, end)
                    if chunk is end:
                        break
                    yield chunk
        finally:
            if hasattr(stream, "__aiter__"):
                await self._close_async_stream(stream)
            else:
                close = getattr(stream, "close", None)
                if close is not None:
                    await _run_persistence(context.run, close)

    async def _close_async_stream(self, stream: Any) -> None:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
        else:
            close = getattr(stream, "close", None)
            if close is not None:
                if inspect.iscoroutinefunction(close):
                    await close()
                else:
                    from looplet.checkpoint import _run_persistence

                    await _run_persistence(close)

    async def _async_stream_primary(
        self, prompt: str, options: dict[str, Any]
    ) -> AsyncIterator[str]:
        delivered = False
        chunks = self._async_stream_chunks(self._primary, prompt, options)
        try:
            try:
                async for chunk in chunks:
                    delivered = True
                    yield chunk
            except Exception:
                if delivered or single_model_request_active():
                    raise
                await chunks.aclose()
                self._active = self._fallback
                chunks = self._async_stream_chunks(self._fallback, prompt, options)
                async for chunk in chunks:
                    yield chunk
        finally:
            await chunks.aclose()


class _FallbackStream:
    def __init__(self, owner: _FallbackLLM, prompt: str, options: dict[str, Any]) -> None:
        self._owner = owner
        self._prompt = prompt
        self._options = options
        self._iterator: Iterator[str] | AsyncIterator[str] | None = None

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._iterator is None:
            self._iterator = self._owner._stream_primary(self._prompt, self._options)
        return next(cast(Iterator[str], self._iterator))

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if self._iterator is None:
            self._iterator = self._owner._async_stream_primary(self._prompt, self._options)
        return await anext(cast(AsyncIterator[str], self._iterator))

    def close(self) -> None:
        if self._iterator is None:
            self._iterator = self._owner._stream_primary(self._prompt, self._options)
        target = self._iterator
        close = getattr(target, "close", None)
        if close is not None:
            if inspect.iscoroutinefunction(close):
                raise TypeError("Use aclose for an async stream")
            close()
        elif hasattr(target, "aclose"):
            raise TypeError("Use aclose for an async stream")

    async def aclose(self) -> None:
        if self._iterator is None:
            self._iterator = self._owner._async_stream_primary(self._prompt, self._options)
        await self._owner._close_async_stream(self._iterator)


class FallbackRouter:
    """Always returns a backend that tries primary first, fallback on error.

    Args:
        primary: The preferred ``LLMBackend`` to use.
        fallback: The backup ``LLMBackend`` used when primary raises.
    """

    def __init__(self, *, primary: LLMBackend, fallback: LLMBackend) -> None:
        self._backend = _FallbackLLM(primary, fallback)

    def select(self, purpose: str, **kwargs: Any) -> LLMBackend:
        return self._backend


# ── CostTracker ──────────────────────────────────────────────────


class CostTracker:
    """Wraps an ``LLMBackend`` and accumulates estimated token costs.

    .. note::
       Prefer :class:`looplet.cost.CostTracker` + :class:`looplet.cost.CostHook`
       for new code. That pair reads **real** provider token counts from
       ``backend.last_usage`` (Anthropic ``input_tokens``, OpenAI
       ``prompt_tokens``) and supports cache-read pricing. This wrapper-
       based tracker remains for backwards compatibility with code that
       cannot install a hook, but its word-count heuristic systematically
       under-reports token usage by 15-25% versus a real BPE tokenizer.

    Token estimates use a simple word-count heuristic (words ≈ tokens).
    This is intentionally approximate; production deployments should use
    a proper tokenizer or migrate to :mod:`looplet.cost`.

    Args:
        backend: The ``LLMBackend`` to wrap.
        cost_per_1k_input: USD cost per 1,000 input tokens.
        cost_per_1k_output: USD cost per 1,000 output tokens.
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        *,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ) -> None:
        if backend is None:
            raise TypeError("CostTracker(backend=..., ...) - backend is required")
        self._backend = backend
        self._cost_per_1k_input = cost_per_1k_input
        self._cost_per_1k_output = cost_per_1k_output
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self._call_count: int = 0
        # Proxy generate_with_tools if the wrapped backend supports it
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl

    def __getattr__(self, name: str) -> Any:
        return backend_attribute(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in MUTABLE_CAPABILITIES and "_backend" in self.__dict__:
            setattr(self._backend, name, value)
        else:
            object.__setattr__(self, name, value)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Word-count heuristic: each word ≈ 1 token (minimum 1)."""
        return max(1, len(text.split()))

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> str:
        response = invoke_backend(
            self._backend.generate,
            prompt,
            {
                **options,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
            },
        )
        input_tokens = self._estimate_tokens(prompt)
        if system_prompt:
            input_tokens += self._estimate_tokens(system_prompt)
        output_tokens = self._estimate_tokens(response)

        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += (
            input_tokens * self._cost_per_1k_input / 1000
            + output_tokens * self._cost_per_1k_output / 1000
        )
        self._call_count += 1
        return response

    @property
    def total_cost(self) -> float:
        """Accumulated USD cost across all calls."""
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        """Total estimated input tokens across all calls."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total estimated output tokens across all calls."""
        return self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Number of generate() calls made."""
        return self._call_count

    def report(self) -> str:
        """Return a human-readable summary of accumulated cost and usage."""
        return (
            f"CostTracker: {self._call_count} calls | "
            f"input={self._total_input_tokens} tokens | "
            f"output={self._total_output_tokens} tokens | "
            f"cost=${self._total_cost:.6f}"
        )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        response = invoke_backend(
            getattr(self._backend, "generate_with_tools"),
            prompt,
            {
                **options,
                "tools": tools,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
            },
        )
        input_tokens = self._estimate_tokens(prompt)
        if system_prompt:
            input_tokens += self._estimate_tokens(system_prompt)
        # Tool schemas ARE serialized into the request body and ARE
        # billed as input tokens by every major provider. Count them.
        # Use compact JSON so the heuristic isn't inflated by
        # whitespace formatting choices.
        if tools:
            import json as _json

            input_tokens += self._estimate_tokens(_json.dumps(tools, separators=(",", ":")))
        # Native tool responses are lists of blocks; estimate from str repr
        output_tokens = self._estimate_tokens(str(response))
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += (
            input_tokens * self._cost_per_1k_input / 1000
            + output_tokens * self._cost_per_1k_output / 1000
        )
        self._call_count += 1
        return response


# ── RoutingLLMBackend ─────────────────────────────────────────────


class RoutingLLMBackend:
    """Drop-in ``LLMBackend`` that delegates to a ``ModelRouter``.

    Use this to swap the active model mid-loop without changing loop code::

        backend = RoutingLLMBackend(router)
        backend.set_purpose("recovery")   # next generate() uses recovery model
        backend.set_purpose("reasoning")  # switch back

    Args:
        router: The ``ModelRouter`` used to select backends.
        default_purpose: Initial purpose string (default ``'reasoning'``).
    """

    def __init__(
        self,
        router: ModelRouter,
        default_purpose: str = "reasoning",
    ) -> None:
        self._router = router
        self._purpose = default_purpose
        self._active: Any = None

    @property
    def purpose(self) -> str:
        """Current routing purpose."""
        return self._purpose

    def set_purpose(self, purpose: str) -> None:
        """Change the routing purpose for subsequent generate() calls."""
        self._purpose = purpose

    def _stream_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        backend = self._router.select(self._purpose)
        self._active = backend
        return invoke_backend(
            getattr(backend, "stream"),
            prompt,
            {
                **options,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
            },
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> str:
        backend = self._router.select(self._purpose)
        self._active = backend
        return invoke_backend(
            backend.generate,
            prompt,
            {
                **options,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
            },
        )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        """Proxy to the selected backend's native tool calling."""
        backend = self._router.select(self._purpose)
        self._active = backend
        return invoke_backend(
            getattr(backend, "generate_with_tools"),
            prompt,
            {
                **options,
                "tools": tools,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
            },
        )

    def __getattr__(self, name: str) -> Any:
        # Dynamically surface ``generate_with_tools`` only when the
        # currently-selected backend supports it, so callers using
        # ``hasattr(llm, "generate_with_tools")`` get a truthful answer
        # under a dynamic routing regime. Matches the gating pattern
        # used by ``_FallbackLLM`` and ``CostTracker``.
        if name in {"generate_with_tools", "stream"}:
            backend = self._router.select(self._purpose)
            if hasattr(backend, name):
                return (
                    self._generate_with_tools_impl
                    if name == "generate_with_tools"
                    else self._stream_impl
                )
            raise AttributeError(name)
        if name not in FORWARDED_CAPABILITIES:
            raise AttributeError(name)
        backend = (
            self._router.select(self._purpose)
            if name
            in {
                "request_policy",
                "request_budgets",
                "stream_in_loop",
                "_looplet_manages_retries",
                "_supports_single_model_request",
            }
            or self._active is None
            else self._active
        )
        return backend_attribute(backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in MUTABLE_CAPABILITIES and "_router" in self.__dict__:
            backend = self._router.select(self._purpose) if self._active is None else self._active
            setattr(backend, name, value)
        else:
            object.__setattr__(self, name, value)

    @property
    def _supports_native_tools(self) -> bool:
        """Check if the current backend supports generate_with_tools."""
        backend = self._router.select(self._purpose)
        return hasattr(backend, "generate_with_tools")
