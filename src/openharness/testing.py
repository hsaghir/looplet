"""Test utilities for users of ``openharness``.

This module exposes scripted mock backends and helpers so downstream
packages can write unit tests for their own hooks, tools, and agent
configurations without depending on a real LLM provider.

Example::

    from openharness import composable_loop, LoopConfig, BaseToolRegistry
    from openharness.testing import MockLLMBackend

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
    "MockLLMBackend",
    "AsyncMockLLMBackend",
]


class MockLLMBackend:
    """Scripted synchronous LLM backend for tests.

    Accepts a list of responses at construction; each call to ``generate``
    returns the next response, cycling once the list is exhausted.

    Args:
        responses: List of strings to return in order. If ``None`` or empty,
            returns ``"mock response"`` on every call.

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
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(responses) if responses else ["mock response"]
        self._index: int = 0
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

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(responses) if responses else ["mock response"]
        self._index: int = 0
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
    from openharness.types import LLMBackend  # noqa: PLC0415
    return isinstance(obj, LLMBackend)
