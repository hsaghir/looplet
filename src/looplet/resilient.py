"""Resilient wrapper for any :class:`LLMBackend`.

Transient failures (network blips, rate limits, provider timeouts) are
common in production and every agent builder writes a variant of the
same retry-with-backoff wrapper.  :class:`ResilientBackend` folds the
pattern into looplet as a drop-in backend.

Features:
- Configurable retry count with exponential backoff + jitter.
- Per-call timeout (wraps the synchronous ``generate`` call in a
  worker thread; the *call* is cancelled from the caller's
  perspective, though provider SDKs may continue in the background
  until their own socket timeouts fire).
- ``retry_on`` predicate lets the caller decide which exceptions are
  retriable — default retries every ``Exception`` that is not
  :class:`KeyboardInterrupt` / :class:`SystemExit`.
- Preserves ``generate_with_tools`` when the wrapped backend
  provides it.

Typical use::

    from looplet.resilient import ResilientBackend
    from looplet.router import FallbackRouter

    primary = ResilientBackend(OpenAIBackend(...), retries=3, timeout_s=30)
    fallback = ResilientBackend(AnthropicBackend(...), retries=2, timeout_s=30)
    llm = FallbackRouter(primary=primary, fallback=fallback).select("default")

The module has zero third-party dependencies.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Any, Callable

from looplet.types import LLMBackend

__all__ = ["ResilientBackend", "RetryExhausted"]

logger = logging.getLogger("looplet.resilient")


class RetryExhausted(RuntimeError):
    """Raised when every retry attempt failed.

    Carries the last exception in ``__cause__`` and a list of every
    exception observed in ``attempts``.
    """

    def __init__(self, attempts: list[BaseException]) -> None:
        last = attempts[-1] if attempts else None
        super().__init__(
            f"ResilientBackend exhausted {len(attempts)} attempts; "
            f"last error: {type(last).__name__}: {last}"
        )
        self.attempts = attempts


def _default_retry_on(exc: BaseException) -> bool:
    """Retry every exception except user-interrupts / system-exits."""
    return not isinstance(exc, (KeyboardInterrupt, SystemExit))


def _run_with_timeout(
    fn: Callable[[], Any],
    timeout_s: float | None,
) -> Any:
    """Run ``fn`` in a worker thread and wait up to ``timeout_s`` seconds.

    Raises :class:`TimeoutError` if the call does not complete in time.
    The worker thread is daemonic — if the provider SDK ignores
    interruption we abandon it rather than hang the agent loop.
    """
    if timeout_s is None or timeout_s <= 0:
        return fn()

    result: list[Any] = []
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 — forward to caller
            error.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout_s}s timeout")
    if error:
        raise error[0]
    return result[0]


class ResilientBackend:
    """Wrap an :class:`LLMBackend` with retry + timeout.

    Args:
        inner: The backend to wrap.
        retries: Total number of attempts (``retries=1`` means no retry).
        timeout_s: Optional per-call timeout.  ``None`` or ``0`` disables.
        base_delay_s: Base delay for exponential backoff between retries.
        max_delay_s: Cap for the computed backoff delay.
        jitter: Multiplicative jitter applied to each backoff delay.
            ``0.1`` means ±10 %.  Set to ``0`` to disable.
        retry_on: Callable ``(exc) -> bool`` deciding whether an
            exception is retriable.  Non-retriable exceptions propagate
            immediately.
        sleep: Injectable sleep function (exposed for tests).
    """

    def __init__(
        self,
        inner: LLMBackend,
        *,
        retries: int = 3,
        timeout_s: float | None = None,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
        jitter: float = 0.2,
        retry_on: Callable[[BaseException], bool] = _default_retry_on,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._jitter = jitter
        self._retry_on = retry_on
        if sleep is None:
            import time

            sleep = time.sleep
        self._sleep = sleep

        # Expose generate_with_tools only if the wrapped backend has it.
        if hasattr(inner, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl

    # ── public API ────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        def _call() -> str:
            return self._inner.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        return self._attempt(_call, op="generate")

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Any:
        def _call() -> Any:
            return self._inner.generate_with_tools(  # pyright: ignore[reportAttributeAccessIssue]
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )

        return self._attempt(_call, op="generate_with_tools")

    # ── internals ─────────────────────────────────────────────

    def _attempt(self, fn: Callable[[], Any], *, op: str) -> Any:
        errors: list[BaseException] = []
        for attempt in range(1, self._retries + 1):
            try:
                return _run_with_timeout(fn, self._timeout_s)
            except BaseException as exc:  # noqa: BLE001 — decide below
                if not self._retry_on(exc):
                    raise
                errors.append(exc)
                if attempt >= self._retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "ResilientBackend %s attempt %d/%d failed (%s: %s); retrying in %.2fs",
                    op,
                    attempt,
                    self._retries,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                self._sleep(delay)
        raise RetryExhausted(errors) from errors[-1]

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with optional jitter."""
        raw = self._base_delay_s * (2 ** (attempt - 1))
        capped = min(raw, self._max_delay_s)
        if self._jitter <= 0:
            return capped
        # Multiplicative jitter in [1 - jitter, 1 + jitter].
        factor = 1.0 + random.uniform(-self._jitter, self._jitter)
        return max(0.0, capped * factor)
