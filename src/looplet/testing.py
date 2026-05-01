"""Test utilities for users of ``looplet``.

This module exposes scripted mock backends and helpers so downstream
packages can write unit tests for their own hooks, tools, and agent
configurations without depending on a real LLM provider.

Example::

    from looplet import composable_loop, LoopConfig, BaseToolRegistry
    from looplet.testing import MockLLMBackend

    def test_my_hook() -> None:
        llm = MockLLMBackend(responses=[
            '{"tool": "add", "args": {"a": 1, "b": 2}, "reasoning": "sum"}',
            '{"tool": "done", "args": {}, "reasoning": "finished"}',
        ])
        tools = BaseToolRegistry()
        # ... register tools and run the loop

This module intentionally has zero third-party dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AsyncMockLLMBackend",
    "LLMResponsesExhausted",
    "MockLLMBackend",
]


class LLMResponsesExhausted(RuntimeError):
    """Raised by ``MockLLMBackend(cycle=False)`` (and the async variant)
    when ``generate`` is called more times than there are scripted
    responses. Surfaces "the loop made N+1 calls but only N were
    scripted" as a clear test failure instead of silently returning
    the first response again.
    """


class MockLLMBackend:
    """Scripted synchronous LLM backend for tests.

    Accepts a list of responses at construction; each call to ``generate``
    returns the next response.

    Args:
        responses: List of strings to return in order. If ``None`` or empty,
            returns ``"mock response"`` on every call.
        cycle: When ``True`` (the default, for backward compatibility),
            wraps around to ``responses[0]`` once exhausted. When ``False``,
            raises :class:`LLMResponsesExhausted` past the last response
            so test authors notice the loop made more LLM calls than they
            scripted (the silent-cycle behaviour produces confusing test
            failures: every "extra" call returns the first response
            again, making the loop look like it's stuck on step 1).

    Attributes:
        calls: Number of times ``generate`` has been called. Useful for
            asserting that a hook short-circuits the loop or that retries
            are counted correctly.
        last_prompt: The prompt passed to the most recent call.
        last_system_prompt: The ``system_prompt`` kwarg passed most recently.

    Example::

        llm = MockLLMBackend(responses=["step 1", "step 2"])
        assert llm.generate("hi") == "step 1"
        assert llm.calls == 1

        # In tests, prefer cycle=False so an over-run loop fails loudly:
        strict = MockLLMBackend(responses=["only one"], cycle=False)
        strict.generate("x")
        with pytest.raises(LLMResponsesExhausted):
            strict.generate("x")
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        cycle: bool = True,
    ) -> None:
        self._responses: list[str] = list(responses) if responses else ["mock response"]
        self._index: int = 0
        self._cycle: bool = cycle
        self.calls: int = 0
        self.last_prompt: str = ""
        self.last_system_prompt: str = ""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if not self._cycle and self._index >= len(self._responses):
            raise LLMResponsesExhausted(
                f"MockLLMBackend(cycle=False): generate() called "
                f"{self.calls} times but only {len(self._responses)} "
                f"responses were scripted. The loop made more LLM calls "
                f"than your test expected — either script more responses "
                f"or arrange for the loop to terminate sooner."
            )
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response

    def reset(self) -> None:
        """Reset the response cursor and call counters."""
        self._index = 0
        self.calls = 0
        self.last_prompt = ""
        self.last_system_prompt = ""


class AsyncMockLLMBackend:
    """Scripted asynchronous LLM backend for tests.

    Mirrors :class:`MockLLMBackend` but implements the ``AsyncLLMBackend``
    protocol — ``generate`` is a coroutine.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        cycle: bool = True,
    ) -> None:
        self._responses: list[str] = list(responses) if responses else ["mock response"]
        self._index: int = 0
        self._cycle: bool = cycle
        self.calls: int = 0
        self.last_prompt: str = ""
        self.last_system_prompt: str = ""

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if not self._cycle and self._index >= len(self._responses):
            raise LLMResponsesExhausted(
                f"AsyncMockLLMBackend(cycle=False): generate() called "
                f"{self.calls} times but only {len(self._responses)} "
                f"responses were scripted. The loop made more LLM calls "
                f"than your test expected — either script more responses "
                f"or arrange for the loop to terminate sooner."
            )
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response

    def reset(self) -> None:
        self._index = 0
        self.calls = 0
        self.last_prompt = ""
        self.last_system_prompt = ""


def _is_llm_backend_subclass(obj: Any) -> bool:
    """Internal smoke-test helper — verifies Mock backends satisfy the protocol."""
    from looplet.types import LLMBackend  # noqa: PLC0415

    return isinstance(obj, LLMBackend)
