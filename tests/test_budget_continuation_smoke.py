"""Smoke tests for budget-aware turn continuation."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.scaffolding import LLMResult, llm_call_with_retry
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


class _TruncatingBackend:
    """Mock backend that exposes ``last_stop_reason`` and emits
    partial chunks for the first N calls before returning a terminal
    'stop' response."""

    def __init__(self, chunks: list[tuple[str, str]]):
        # chunks: list of (text, stop_reason)
        self._chunks = list(chunks)
        self.last_stop_reason: str | None = None
        self.calls = 0

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.calls += 1
        text, stop = self._chunks.pop(0)
        self.last_stop_reason = stop
        return text


class TestBudgetContinuation:
    def test_concatenates_on_max_tokens(self):
        b = _TruncatingBackend(
            [
                ("part-A ", "max_tokens"),
                ("part-B ", "max_tokens"),
                ("part-C", "stop"),
            ]
        )
        result = llm_call_with_retry(b, "prompt", max_continuations=3)
        assert result.ok
        assert result.text == "part-A part-B part-C"
        assert result.continuations == 2
        assert result.stop_reason == "stop"
        assert b.calls == 3

    def test_respects_continuation_cap(self):
        b = _TruncatingBackend(
            [
                ("A", "max_tokens"),
                ("B", "max_tokens"),
                ("C", "max_tokens"),
                ("D", "stop"),
            ]
        )
        result = llm_call_with_retry(b, "prompt", max_continuations=1)
        assert result.text == "AB"
        assert result.continuations == 1
        assert result.stop_reason == "max_tokens"
        assert b.calls == 2

    def test_disabled_by_default(self):
        b = _TruncatingBackend([("only", "max_tokens")])
        result = llm_call_with_retry(b, "prompt")
        assert result.text == "only"
        assert result.continuations == 0

    def test_backend_without_stop_reason_attribute_still_works(self):
        class _PlainBackend:
            def generate(self, prompt, **kw):
                return "text"

        result = llm_call_with_retry(_PlainBackend(), "prompt")
        assert result.ok
        assert result.stop_reason is None
        assert result.continuations == 0

    def test_loop_passes_config_through(self):
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )
        b = _TruncatingBackend(
            [
                ('{"tool":"done"', "max_tokens"),
                (', "args":{"answer":"ok"},"reasoning":"r"}', "stop"),
            ]
        )
        steps = list(
            composable_loop(
                llm=b,
                tools=reg,
                state=DefaultState(max_steps=2),
                hooks=[],
                config=LoopConfig(max_steps=2, max_turn_continuations=2),
            )
        )
        # Continuation stitched the truncated JSON back together -
        # the parser succeeds, producing exactly one done step.
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "done"
        assert b.calls == 2
