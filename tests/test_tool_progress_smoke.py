"""Smoke tests for :attr:`LifecycleEvent.TOOL_PROGRESS` emission."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.events import LifecycleEvent
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools():
    reg = BaseToolRegistry()

    def long_running(msg: str, ctx=None):
        if ctx is not None:
            ctx.report_progress("started", {"msg": msg})
            ctx.report_progress("half", {"pct": 50})
            ctx.report_progress("done", {"pct": 100})
        return {"ok": True}

    reg.register(
        ToolSpec(
            name="work",
            description="long op",
            parameters={"msg": "str"},
            execute=long_running,
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


class _Observer:
    def __init__(self):
        self.progress_events: list[dict] = []

    def on_event(self, payload):
        if payload.event is LifecycleEvent.TOOL_PROGRESS:
            self.progress_events.append(payload)

    def pre_loop(self, *a, **k):
        return None

    def pre_prompt(self, *a, **k):
        return None

    def post_dispatch(self, *a, **k):
        return None

    def check_done(self, *a, **k):
        return None

    def should_stop(self, *a, **k):
        return False


class TestToolProgressEvents:
    def test_progress_emitted_and_observed(self):
        obs = _Observer()
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"work","args":{"msg":"go"},"reasoning":"r"}',
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                    ]
                ),
                tools=_tools(),
                state=DefaultState(max_steps=3),
                hooks=[obs],
                config=LoopConfig(max_steps=3),
            )
        )
        assert len(obs.progress_events) == 3
        stages = [p.extra["stage"] for p in obs.progress_events]
        assert stages == ["started", "half", "done"]
        # Each event carries the tool_call that emitted it.
        for p in obs.progress_events:
            assert p.tool_call.tool == "work"

    def test_no_subscribers_no_context_overhead(self):
        # With zero hooks and no cancel/approval, ctx is None — tool's
        # progress calls become no-ops (ctx is None in the tool).
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"work","args":{"msg":"go"},"reasoning":"r"}',
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                    ]
                ),
                tools=_tools(),
                state=DefaultState(max_steps=3),
                hooks=[],
                config=LoopConfig(max_steps=3),
            )
        )
