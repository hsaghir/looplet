"""Async composable loop - ``async for step in async_composable_loop(...)``.

Async mirror of :func:`looplet.loop.composable_loop` for use with
async LLM backends (:class:`AsyncOpenAIBackend`, etc.). Tool execution,
parsing, and state management remain synchronous; LLM calls and
awaitable hook results are awaited.

Usage::

    from looplet.async_loop import async_composable_loop
    from looplet.backends import AsyncOpenAIBackend

    llm = AsyncOpenAIBackend(base_url="...", api_key="...", model="gpt-4o")

    async for step in async_composable_loop(
        llm=llm, tools=tools, state=state, config=config, task=task,
    ):
        print(step.pretty())

The function accepts the same arguments as ``composable_loop`` and
yields the same :class:`Step` objects. Hooks may be synchronous or
``async def`` and run sequentially between awaited LLM calls.

When to use this instead of ``composable_loop``:
- Your LLM backend has ``async def generate()``
- You're inside an async context (FastAPI, aiohttp, Discord bot)
- You want to run multiple agent loops concurrently via asyncio.gather

When NOT to use this:
- Your LLM backend is synchronous - use ``composable_loop`` instead
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from contextlib import aclosing
from functools import wraps
from threading import RLock
from typing import Any, AsyncGenerator, Callable, Sequence

from looplet._backend_contract import TerminalLLMError
from looplet._backend_contract import attempt_limit as _attempt_limit
from looplet.checkpoint import _run_persistence
from looplet.context_projection import ContextProjection as ContextProjection
from looplet.loop import (
    ContextBudgetSnapshot as ContextBudgetSnapshot,
)
from looplet.loop import (
    LoopConfig,
    _build_tool_ctx,
    _mark_failed_state,
    _normalize_loop_inputs,
    _reset_context_overrides,
    _resolve_loop_config,
    _resolve_loop_state,
    _set_context_overrides,
    _validate_hooks,
    _validate_loop_inputs,
)
from looplet.loop import (
    _render_projection as _render_projection,
)
from looplet.native_tools import NativeToolPolicy, NativeToolUnsupportedError
from looplet.scaffolding import (
    LLMResult,
    _is_prompt_too_long,
    _is_terminal_llm_error,
    _native_llm_result,
    _NativeTurnRecovery,
)
from looplet.scaffolding import _accepts_kwarg as _sync_accepts_kwarg
from looplet.scaffolding import (
    estimate_prompt_tokens as estimate_prompt_tokens,
)
from looplet.session import SessionLog
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import AgentState, RunResult, Step, ToolCall

logger = logging.getLogger(__name__)

__all__ = [
    "run_result_async",
    "async_composable_loop",
    "async_llm_call",
]


# ── Retry constants (mirror scaffolding.py) ──────────────────────
MAX_LLM_RETRIES = 2
RETRY_BACKOFF_BASE = 1.0


async def _maybe_await(value: Any) -> Any:
    """Resolve sync or async hook results in registration order."""
    if inspect.isawaitable(value):
        return await value
    return value


class _NestedModelRequest:
    def __init__(self, owner: _NestedModelRequests, response: Any) -> None:
        self.owner = owner
        self.response = response
        self.future: concurrent.futures.Future[str] | None = None
        self.task: asyncio.Task[str] | None = None
        self.completed: concurrent.futures.Future[None] = concurrent.futures.Future()
        self.started = False
        self.aborted = False

    def _settle(self, _task: Any = None) -> None:
        with self.owner.lock:
            if not self.completed.done():
                self.completed.set_result(None)
            self.owner.pending.discard(self)

    def close_unstarted(
        self, future: concurrent.futures.Future[str] | asyncio.Future[str] | None = None
    ) -> None:
        if future is not None and not future.cancelled():
            return
        with self.owner.lock:
            if self.started:
                return
            self.aborted = True
            if isinstance(self.response, asyncio.Future):

                def cancel_response() -> None:
                    self.response.add_done_callback(self._settle)
                    self.response.cancel()

                self.owner.loop.call_soon_threadsafe(cancel_response)
            else:
                if inspect.iscoroutine(self.response):
                    self.response.close()
                self._settle()

    async def resolve(self) -> str:
        with self.owner.lock:
            if self.aborted:
                raise asyncio.CancelledError
            self.started = True
        task = asyncio.current_task()
        assert task is not None
        task.add_done_callback(self._settle)
        return await self.response


class _NestedModelRequests:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.lock = RLock()
        self.pending: set[_NestedModelRequest] = set()
        self.closed = False

    def invoke(self, generate: Callable[[], Any]) -> Any:
        with self.lock:
            if self.closed:
                raise concurrent.futures.CancelledError
            request = _NestedModelRequest(self, None)
            self.pending.add(request)
        try:
            response = generate()
        except BaseException:
            request._settle()
            raise
        request.response = response
        if not inspect.isawaitable(response):
            with self.lock:
                request._settle()
                closed = self.closed
            if closed:
                raise concurrent.futures.CancelledError
            return response
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.loop:
            request.close_unstarted()
            raise RuntimeError(
                "Synchronous model calls cannot block their owner event loop. "
                "Use await ctx.async_llm.generate() instead."
            )
        return self.run(response, request=request)

    async def invoke_async(self, generate: Callable[[], Any]) -> Any:
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("Async helper model calls must use their owner event loop.")
        with self.lock:
            if self.closed:
                raise asyncio.CancelledError
            request = _NestedModelRequest(self, None)
            self.pending.add(request)
        try:
            response = generate()
        except BaseException:
            request._settle()
            raise
        request.response = response
        with self.lock:
            if self.closed:
                request.close_unstarted()
                raise asyncio.CancelledError
            if not inspect.isawaitable(response):
                request._settle()
                return response
            resolver = request.resolve()
            try:
                request.task = self.loop.create_task(resolver)
            except BaseException:
                resolver.close()
                request.close_unstarted()
                raise
            request.task.add_done_callback(request.close_unstarted)
        return await request.task

    def run(self, response: Any, *, request: _NestedModelRequest | None = None) -> str:
        with self.lock:
            request = request if request is not None else _NestedModelRequest(self, response)
            if self.closed:
                request.close_unstarted()
                raise concurrent.futures.CancelledError
            self.pending.add(request)
            resolver = request.resolve()
            try:
                future = asyncio.run_coroutine_threadsafe(resolver, self.loop)
            except BaseException:
                resolver.close()
                request.close_unstarted()
                raise
            request.future = future
            future.add_done_callback(request.close_unstarted)
        return future.result()

    def _drain(self, requests: tuple[_NestedModelRequest, ...]) -> None:
        for request in requests:
            request.completed.result()

    async def cancel_pending(self) -> None:
        with self.lock:
            self.closed = True
            requests = tuple(self.pending)
            for request in requests:
                if request.future is not None:
                    request.future.cancel()
                if request.task is not None:
                    request.task.cancel()
        if requests:
            await _run_persistence(self._drain, requests)


class _AsyncBridgeLLM:
    """Keep asynchronous tool model requests inside their host run."""

    def __init__(self, backend: Any, requests: _NestedModelRequests) -> None:
        self._backend = backend
        self._requests = requests

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._backend, name)
        if name not in {"generate", "generate_with_tools"} or not callable(method):
            return method

        @wraps(method)
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            return await self._requests.invoke_async(lambda: method(*args, **kwargs))

        return invoke


class _SyncBridgeLLM:
    """Provide synchronous model access to tools running in worker threads.

    The async loop supplies its running loop so async backend calls stay
    on that loop. Async tools use an owned ``ctx.async_llm`` proxy.
    """

    def __init__(
        self,
        backend: Any,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        requests: _NestedModelRequests | None = None,
    ) -> None:
        self._backend = backend
        self._loop = loop
        self._requests = requests
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _run_call(self, call: Callable[[], Any]) -> Any:
        if self._requests is not None:
            return self._requests.invoke(call)
        result = call()
        if inspect.isawaitable(result):
            if self._loop is not None:
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is self._loop:
                    if inspect.iscoroutine(result):
                        result.close()
                    raise RuntimeError(
                        "Synchronous model calls cannot block their owner event loop. "
                        "Use await ctx.async_llm.generate() instead."
                    )

                async def resolve() -> Any:
                    return await result

                return asyncio.run_coroutine_threadsafe(resolve(), self._loop).result()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, result)  # pyright: ignore[reportArgumentType]
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(result)
        return result

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str = "",
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        return self._run_call(
            lambda: self._backend.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
        )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        system_prompt: str = "",
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._run_call(
            lambda: self._backend.generate_with_tools(
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )
        )


# ── Async LLM call with retry ───────────────────────────────────


async def async_llm_call(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int | None = None,
    system_prompt: str = "",
    temperature: float | None = None,
    max_retries: int = MAX_LLM_RETRIES,
    max_continuations: int = 0,
    tools: list[dict[str, Any]] | None = None,
    native_policy: NativeToolPolicy | None = None,
    cancel_token: Any | None = None,
    cache_breakpoints: list[Any] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
    should_continue: Callable[[], bool] | None = None,
    on_chunk: Callable[[str, int], None] | None = None,
    on_tool_call: Callable[[dict[str, Any]], None] | None = None,
    on_stream_reset: Callable[[], bool | None] | None = None,
) -> LLMResult:
    """Async version of :func:`looplet.scaffolding.llm_call_with_retry`.

    Awaits ``llm.generate()`` or ``llm.generate_with_tools()`` when
    they are coroutines; calls them synchronously otherwise (supporting
    sync backends used from async context). Native tool support is required
    unless ``NativeToolPolicy(enabled=False)`` selects JSON-text mode.
    """
    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
        return LLMResult(None, RuntimeError("cancelled before LLM call"))

    policy = native_policy or NativeToolPolicy()
    if (
        tools is not None
        and policy.enabled
        and not policy.demoted
        and not policy.should_use(llm, tools)
    ):
        return LLMResult(
            None,
            NativeToolUnsupportedError(
                "Backend does not support native tool calls; set use_native_tools=False "
                "to explicitly use JSON-text tool calls"
            ),
            native_requested=True,
        )
    use_native = policy.should_use(llm, tools)
    _gk = generate_kwargs or {}

    def _method_accepts(method_name: str, name: str) -> bool:
        fn = getattr(llm, method_name, None)
        if fn is None:
            return False
        try:
            return _sync_accepts_kwarg(fn, name)
        except (TypeError, ValueError):
            return False

    def _method_accepts_kwarg(method_name: str, name: str) -> bool:
        fn = getattr(llm, method_name, None)
        return _sync_accepts_kwarg(fn, name) if fn is not None else False

    # Filter generate_kwargs to only keys the backend accepts
    def _filtered_kwargs(method_name: str) -> dict[str, Any]:
        fn = getattr(llm, method_name, None)
        if fn is None or not _gk:
            return {}
        out: dict[str, Any] = {}
        for k, v in _gk.items():
            try:
                sig = inspect.signature(fn)
                if k in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    out[k] = v
            except (TypeError, ValueError):
                pass
        return out

    last_error: Exception | None = None
    native_attempted = False
    attempt_limit = _attempt_limit(llm, max_retries)
    recovery: _NativeTurnRecovery | None = None

    for attempt in range(attempt_limit):
        if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
            return LLMResult(None, RuntimeError("cancelled during retry"))
        if attempt > 0 and on_stream_reset is not None:
            on_stream_reset()
        try:
            if use_native:
                native_attempted = True
                call_kwargs: dict[str, Any] = {
                    "tools": tools,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    **_filtered_kwargs("generate_with_tools"),
                }
                if cache_breakpoints and _method_accepts(
                    "generate_with_tools", "cache_breakpoints"
                ):
                    call_kwargs["cache_breakpoints"] = cache_breakpoints
                if cancel_token is not None and _method_accepts_kwarg(
                    "generate_with_tools", "cancel_token"
                ):
                    call_kwargs["cancel_token"] = cancel_token
                if on_chunk is not None and _method_accepts("generate_with_tools", "on_chunk"):
                    call_kwargs["on_chunk"] = on_chunk
                if on_tool_call is not None and _method_accepts(
                    "generate_with_tools", "on_tool_call"
                ):
                    call_kwargs["on_tool_call"] = on_tool_call
                if recovery is None:
                    recovery = _NativeTurnRecovery(llm, max_continuations)
                recovery.begin_call(prompt)
                result = llm.generate_with_tools(prompt, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result
                recovery.observe_usage(llm, result)
                stop = getattr(llm, "last_stop_reason", None)
                while stop == "max_tokens":
                    if on_stream_reset is not None and on_stream_reset() is False:
                        break
                    if should_continue is not None and not should_continue():
                        break
                    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
                        raise TerminalLLMError(
                            "Native recovery cancelled", termination_reason="cancelled"
                        )
                    next_options = recovery.restart_options(call_kwargs)
                    if next_options is None:
                        break
                    call_kwargs = next_options
                    recovery.begin_call(prompt)
                    result = llm.generate_with_tools(prompt, **call_kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                    recovery.observe_usage(llm, result)
                    stop = getattr(llm, "last_stop_reason", None)
                return recovery.finish(_native_llm_result(result, stop, recovery.continuations))

            call_kwargs = {
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
                **_filtered_kwargs("generate"),
            }
            if cache_breakpoints and _method_accepts("generate", "cache_breakpoints"):
                call_kwargs["cache_breakpoints"] = cache_breakpoints
            if cancel_token is not None and _method_accepts_kwarg("generate", "cancel_token"):
                call_kwargs["cancel_token"] = cancel_token
            stream_fn: Any = (
                getattr(llm, "stream", None)
                if on_chunk is not None and getattr(llm, "stream_in_loop", True)
                else None
            )
            if callable(stream_fn) and on_chunk is not None:
                stream_kwargs = {
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    **_filtered_kwargs("stream"),
                }
                if cache_breakpoints and _method_accepts("stream", "cache_breakpoints"):
                    stream_kwargs["cache_breakpoints"] = cache_breakpoints
                if cancel_token is not None and _method_accepts_kwarg("stream", "cancel_token"):
                    stream_kwargs["cancel_token"] = cancel_token
                stream_out: Any = stream_fn(prompt, **stream_kwargs)
                if inspect.isawaitable(stream_out):
                    stream_out = await stream_out
                chunks: list[str] = []
                if hasattr(stream_out, "__aiter__"):
                    idx = 0
                    async for chunk in stream_out:
                        if isinstance(chunk, str) and chunk:
                            on_chunk(chunk, idx)
                            chunks.append(chunk)
                            idx += 1
                else:
                    for idx, chunk in enumerate(stream_out):
                        if isinstance(chunk, str) and chunk:
                            on_chunk(chunk, idx)
                            chunks.append(chunk)
                result = "".join(chunks)
            else:
                result = llm.generate(prompt, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result
            raw_usage = getattr(llm, "last_usage", None)
            first_usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
            text_calls = [{"usage": first_usage}]
            text_usage: dict[str, Any] = {
                key: value
                for key, value in first_usage.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            stop = getattr(llm, "last_stop_reason", None)
            continuation_count = 0
            accumulated = result if isinstance(result, str) else ""
            while (
                max_continuations > 0
                and continuation_count < max_continuations
                and stop == "max_tokens"
                and isinstance(accumulated, str)
            ):
                if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
                    break
                continuation_prompt = (
                    prompt
                    + "\n\n[assistant partial output so far]\n"
                    + accumulated
                    + "\n\n[continue from exactly where you left off; "
                    "do not repeat any prior text]"
                )
                more = llm.generate(continuation_prompt, **call_kwargs)
                if inspect.isawaitable(more):
                    more = await more
                if not isinstance(more, str):
                    break
                cont_raw = getattr(llm, "last_usage", None)
                cont_usage = dict(cont_raw) if isinstance(cont_raw, dict) else {}
                text_calls.append({"usage": cont_usage})
                for key, value in cont_usage.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        text_usage[key] = text_usage.get(key, 0) + value
                accumulated += more
                stop = getattr(llm, "last_stop_reason", None)
                continuation_count += 1
            result = accumulated if continuation_count else result
            return LLMResult(
                result,
                stop_reason=stop,
                continuations=continuation_count,
                native_requested=bool(tools is not None and policy.enabled),
                native_attempted=native_attempted,
                native_succeeded=False,
                usage=text_usage or None,
                model_calls=text_calls,
            )

        except Exception as e:
            last_error = e
            if getattr(e, "request_started", None) is False:
                if recovery is not None:
                    recovery.calls -= 1
                    recovery.model_calls.pop()
                    return recovery.finish(LLMResult(None, e))
                return LLMResult(None, e, calls=0)
            if recovery is not None and (
                _is_terminal_llm_error(e)
                or attempt >= attempt_limit - 1
                or (on_stream_reset is not None and on_stream_reset() is False)
            ):
                return recovery.finish(
                    LLMResult(None, e, native_requested=True, native_attempted=True)
                )
            if isinstance(e, TerminalLLMError):
                logger.warning("Backend stopped (not retrying): %s", e)
                return LLMResult(
                    None,
                    e,
                    native_requested=bool(tools is not None and policy.enabled),
                    native_attempted=native_attempted,
                )
            if _is_prompt_too_long(e):
                return LLMResult(None, e)
            if use_native and isinstance(e, NativeToolUnsupportedError):
                logger.error("Native tool call unsupported; set use_native_tools=False: %s", e)
                return LLMResult(None, e, native_requested=True, native_attempted=True)
            if type(e).__name__ == "LLMResponsesExhausted":
                logger.warning("Mock LLM exhausted (not retrying): %s", e)
                return LLMResult(None, e)
            if use_native:
                if attempt < max_retries:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Async native LLM call attempt %d/%d failed: %s - retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                return LLMResult(
                    None,
                    e,
                    native_requested=bool(tools is not None and policy.enabled),
                    native_attempted=native_attempted,
                )
            if attempt < attempt_limit - 1:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "Async LLM call attempt %d/%d failed: %s - retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                return LLMResult(None, last_error)

    return LLMResult(None, last_error)


# ── Async composable loop ───────────────────────────────────────


async def async_composable_loop(
    llm: Any,
    task: Any = None,
    tools: BaseToolRegistry | Sequence[ToolSpec | Callable[..., Any]] | None = None,
    context: Any = None,
    hooks: list[Any] | None = None,
    config: LoopConfig | None = None,
    state: AgentState | None = None,
    session_log: SessionLog | None = None,
    stream: Any | None = None,
    conversation: Any | None = None,
    *,
    max_steps: int | None = None,
    system_prompt: str | None = None,
) -> AsyncGenerator[Step, None]:
    """Run the asynchronous tool-calling loop and yield recorded steps.

    Args:
        llm: Backend whose ``generate()`` result can be awaited.
        task: A task string or a dict of task fields. Strings become
            ``{'description': task}``. Defaults to an empty dict.
        tools: Python functions, tool specs, or an explicit registry.
            Sequences get argument validation and a standard completion tool
            unless completion is already configured. Explicit registries stay unchanged.
        context: Optional application object passed unchanged to hook methods.
        hooks: Explicit policies composed with the configured harness defaults.
        config: ``LoopConfig`` with budgets, prompt builders, and runtime options.
            Defaults to a new ``LoopConfig``.
        state: Mutable ``AgentState`` for this run. If omitted, creates ``DefaultState``
            with the effective ``config.max_steps`` limit.
        session_log: Memory log for this run. Defaults to a new ``SessionLog``.
        stream: Optional event emitter for structured loop lifecycle events.
        conversation: Optional conversation that records prompts, responses, and tool messages.
        max_steps: Optional override for ``config.max_steps``.
            An omitted state also receives this limit. A supplied config is updated in place.
        system_prompt: Optional override for ``config.system_prompt``.
            A supplied config is updated in place; live host controls retain their identity.

    Yields:
        A ``Step`` for each recorded loop step, including failed tool calls.

    Consume the iterator with ``async for``. If stopping early, await its ``aclose()`` method
    or use ``contextlib.aclosing``. Cleanup runs on exhaustion, closure, and exceptions.
    Early closure marks an unfinished run as stopped, not completed.
    """
    from looplet.harness import prepare_harness_run_async

    effective_config = _resolve_loop_config(
        config, max_steps=max_steps, system_prompt=system_prompt
    )
    task, tools = _normalize_loop_inputs(task, tools, effective_config)
    prepared = await prepare_harness_run_async(
        effective_config, hooks or [], llm, tools, state, session_log, conversation
    )
    tokens: list[tuple[Any, Any]] = []
    from looplet.loop import _LoopFinalizer, _mark_terminal_state

    finalizer = _LoopFinalizer()
    nested_requests = _NestedModelRequests(asyncio.get_running_loop())
    iterator = None
    original_cancellation: asyncio.CancelledError | None = None
    try:
        prepared.enable_async_persistence()
        tokens = _set_context_overrides(effective_config)
        iterator = _async_composable_loop_impl(
            llm=prepared.llm,
            task=task,
            tools=tools,
            context=context,
            hooks=prepared.hooks,
            config=prepared.config,
            state=prepared.state,
            session_log=prepared.session_log,
            stream=stream,
            conversation=conversation,
            _finalizer=finalizer,
            _checkpoint_store=prepared.checkpoint_store,
            _initial_checkpoint=prepared.initial_checkpoint,
            _compact_service=prepared.compact_service,
            _nested_requests=nested_requests,
        )
        async for step in iterator:
            yield step
    except GeneratorExit:
        if getattr(finalizer.state, "run_phase", None) != "terminal":
            _mark_terminal_state(finalizer.state, status="stopped", reason="stream_closed")
        raise
    except asyncio.CancelledError as exc:
        original_cancellation = exc
        _mark_terminal_state(
            finalizer.state if finalizer.state is not None else prepared.state,
            status="cancelled",
            reason="cancelled",
        )
        raise
    except BaseException:
        _mark_failed_state(finalizer.state if finalizer.state is not None else prepared.state)
        raise
    finally:
        try:
            try:
                if iterator is not None:
                    await iterator.aclose()
            finally:
                try:
                    await nested_requests.cancel_pending()
                except asyncio.CancelledError:
                    if original_cancellation is None:
                        raise
                finally:
                    await finalizer.finish_async(suppress_errors=True)
        finally:
            try:
                try:
                    await _run_persistence(prepared.persist)
                except (Exception, asyncio.CancelledError) as exc:
                    if original_cancellation is None:
                        raise
                    logger.warning(
                        "Persistence failed during async cancellation (%s)", type(exc).__name__
                    )
            finally:
                try:
                    prepared.release()
                finally:
                    _reset_context_overrides(tokens)


async def run_result_async(
    llm: Any,
    task: Any = None,
    tools: BaseToolRegistry | Sequence[ToolSpec | Callable[..., Any]] | None = None,
    *,
    config: LoopConfig | None = None,
    state: AgentState | None = None,
    max_steps: int | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> RunResult:
    """Await and close the ordinary async loop, returning its accepted output and status.

    Functions and task strings use the same defaults as :func:`async_composable_loop`.
    Omitted state is created for this invocation. Supplied state uses ordinary budget rules.
    Additional keyword arguments pass unchanged to :func:`async_composable_loop`.
    Host-supplied backends, tools, hooks, and resources are not closed.
    """
    config = _resolve_loop_config(config, max_steps=max_steps, system_prompt=system_prompt)
    if state is None:
        state = _resolve_loop_state(None, config)
    async with aclosing(
        async_composable_loop(llm, task=task, tools=tools, config=config, state=state, **kwargs)
    ) as steps:
        async for _step in steps:
            pass
    return RunResult.from_state(state)


async def _async_composable_loop_impl(
    llm: Any,
    task: Any = None,
    tools: BaseToolRegistry | None = None,
    context: Any = None,
    hooks: list[Any] | None = None,
    config: LoopConfig | None = None,
    state: AgentState | None = None,
    session_log: SessionLog | None = None,
    stream: Any | None = None,
    conversation: Any | None = None,
    *,
    max_steps: int | None = None,
    system_prompt: str | None = None,
    _finalizer: Any = None,
    _checkpoint_store: Any = None,
    _initial_checkpoint: Any = None,
    _compact_service: Any = None,
    _nested_requests: _NestedModelRequests | None = None,
) -> AsyncGenerator[Step, None]:
    """Async version of :func:`looplet.loop.composable_loop`.

    Yields the same :class:`Step` objects. LLM calls, async hooks, and
    async tools are awaited; synchronous tools run via worker threads
    through :meth:`BaseToolRegistry.async_dispatch`.

    Usage::

        async for step in async_composable_loop(llm=llm, tools=tools, ...):
            print(step.pretty())

    The ``max_steps`` and ``system_prompt`` keyword shorthands mirror
    :func:`looplet.loop.composable_loop` - when set they override the
    matching fields on ``config`` (a fresh ``LoopConfig`` is created
    if none is passed) so simple async agents don't need to construct
    a config explicitly.
    """
    # ── Defaults ────────────────────────────────────────────────
    if task is None:
        task = {}
    if tools is None:
        raise ValueError("tools is required")
    if config is None or max_steps is not None or system_prompt is not None:
        config = _resolve_loop_config(config, max_steps=max_steps, system_prompt=system_prompt)
    if hooks is None:
        hooks = []
    _validate_loop_inputs(task, tools, config)

    if not callable(getattr(llm, "generate", None)):
        raise TypeError(
            f"llm must implement generate() (got {type(llm).__name__}). "
            "Use AsyncOpenAIBackend(base_url=...) or similar."
        )

    state = _resolve_loop_state(state, config)
    _validate_hooks(hooks)
    if session_log is None:
        session_log = SessionLog()
    from looplet.loop import _LoopFinalizer

    finalizer = _finalizer or _LoopFinalizer()
    finalizer.bind(hooks, state, session_log, context, llm)

    from looplet.streaming import AsyncStreamingToolExecutor
    from looplet.turn_engine import drive_turn_machine_async, run_turn_machine

    running_loop = asyncio.get_running_loop()
    sync_bridge_box: list[Any] = [_SyncBridgeLLM(llm, loop=running_loop, requests=_nested_requests)]

    def _on_turn_llm(eff_llm: Any) -> None:
        sync_bridge_box[0] = _SyncBridgeLLM(eff_llm, loop=running_loop, requests=_nested_requests)

    def _make_async_tool_ctx(tc: ToolCall, sn: int, eff_llm: Any) -> Any:
        async_tool_llm = (
            _AsyncBridgeLLM(eff_llm, _nested_requests) if _nested_requests is not None else eff_llm
        )
        return _build_tool_ctx(
            config,
            tools=tools,
            hooks=hooks,
            tool_call=tc,
            step_num=sn,
            state=state,
            session_log=session_log,
            llm=sync_bridge_box[0],
            async_llm=async_tool_llm,
        )

    async for step in drive_turn_machine_async(
        run_turn_machine(
            llm=llm,
            task=task,
            tools=tools,
            context=context,
            hooks=hooks,
            config=config,
            state=state,
            session_log=session_log,
            stream=stream,
            explicit_conversation=conversation,
            finalizer=finalizer,
            checkpoint_store=_checkpoint_store,
            initial_checkpoint=_initial_checkpoint,
            compact_service=_compact_service,
            make_tool_ctx=_make_async_tool_ctx,
            make_stream_executor=lambda: AsyncStreamingToolExecutor(
                tools, max_concurrent=max(1, config.max_parallel_calls)
            ),
            on_turn_llm_selected=_on_turn_llm,
            build_final_trace=False,
        )
    ):
        yield step
