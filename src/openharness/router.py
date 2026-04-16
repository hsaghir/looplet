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
    # Use llm as a drop-in LLMBackend — it routes internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openharness.types import LLMBackend

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

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        try:
            return self._primary.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("Primary LLM failed (%s); switching to fallback", exc)
            return self._fallback.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )


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

    Token estimates use a simple word-count heuristic (words ≈ tokens).
    This is intentionally approximate; production deployments should use
    a proper tokenizer.

    Args:
        backend: The ``LLMBackend`` to wrap.
        cost_per_1k_input: USD cost per 1,000 input tokens.
        cost_per_1k_output: USD cost per 1,000 output tokens.
    """

    def __init__(
        self,
        *,
        backend: LLMBackend,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ) -> None:
        self._backend = backend
        self._cost_per_1k_input = cost_per_1k_input
        self._cost_per_1k_output = cost_per_1k_output
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self._call_count: int = 0

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
        temperature: float = 0.2,
    ) -> str:
        response = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
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

    @property
    def purpose(self) -> str:
        """Current routing purpose."""
        return self._purpose

    def set_purpose(self, purpose: str) -> None:
        """Change the routing purpose for subsequent generate() calls."""
        self._purpose = purpose

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        backend = self._router.select(self._purpose)
        return backend.generate(
            prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
        )
