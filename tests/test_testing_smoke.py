"""Smoke tests for the ``looplet.testing`` helpers."""

from __future__ import annotations

import asyncio

import pytest

from looplet.testing import MockLLMBackend
from looplet.types import LLMBackend

pytestmark = pytest.mark.smoke


class TestMockLLMBackendSmoke:
    def test_satisfies_protocol(self) -> None:
        llm = MockLLMBackend()
        assert isinstance(llm, LLMBackend)

    def test_default_response(self) -> None:
        llm = MockLLMBackend()
        assert llm.generate("hello") == "mock response"
        assert llm.calls == 1
        assert llm.last_prompt == "hello"

    def test_scripted_cycle(self) -> None:
        llm = MockLLMBackend(responses=["a", "b"])
        assert llm.generate("x") == "a"
        assert llm.generate("y") == "b"
        assert llm.generate("z") == "a"  # cycles
        assert llm.calls == 3

    def test_reset(self) -> None:
        llm = MockLLMBackend(responses=["a", "b"])
        llm.generate("x")
        llm.reset()
        assert llm.calls == 0
        assert llm.last_prompt == ""
        assert llm.generate("y") == "a"

    def test_captures_system_prompt(self) -> None:
        llm = MockLLMBackend()
        llm.generate("hi", system_prompt="sys")
        assert llm.last_system_prompt == "sys"


# ── MockLLMBackend(cycle=False) — outcome-side ergonomics ──


def test_mock_backend_default_still_cycles() -> None:
    """Backward compat: cycle=True is the default and existing tests
    that rely on cycling continue to work."""
    from looplet.testing import MockLLMBackend

    llm = MockLLMBackend(responses=["a", "b"])
    assert [llm.generate("x") for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_mock_backend_cycle_false_raises_when_exhausted() -> None:
    """``cycle=False`` raises ``LLMResponsesExhausted`` past the last
    response. Lets test authors notice when the loop made more LLM
    calls than they scripted (which otherwise silently re-uses
    response[0] and looks like "stuck on step 1")."""
    import pytest

    from looplet.testing import LLMResponsesExhausted, MockLLMBackend

    llm = MockLLMBackend(responses=["only one"], cycle=False)
    assert llm.generate("x") == "only one"
    with pytest.raises(LLMResponsesExhausted, match="2 times"):
        llm.generate("y")


def test_async_mock_backend_cycle_false_raises_when_exhausted() -> None:
    """Same contract on the async variant."""
    import asyncio

    import pytest

    from looplet.testing import AsyncMockLLMBackend, LLMResponsesExhausted

    llm = AsyncMockLLMBackend(responses=["only one"], cycle=False)

    async def _drive() -> None:
        assert await llm.generate("x") == "only one"
        with pytest.raises(LLMResponsesExhausted):
            await llm.generate("y")

    asyncio.run(_drive())
