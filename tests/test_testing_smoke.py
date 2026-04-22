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
