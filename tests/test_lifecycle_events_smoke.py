"""Smoke tests for lifecycle event dispatch (:meth:`LoopHook.on_event`)."""
from __future__ import annotations

import pytest

from openharness import (
    BaseToolRegistry,
    DefaultState,
    HookDecision,
    LifecycleEvent,
    LoopConfig,
    composable_loop,
)
from openharness.events import EventPayload
from openharness.testing import MockLLMBackend
from openharness.tools import ToolSpec

pytestmark = pytest.mark.smoke


class _Recorder:
    """Captures every on_event payload for assertions."""

    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []
        self.payloads: list[EventPayload] = []

    def on_event(self, payload: EventPayload) -> HookDecision | None:
        self.events.append(payload.event)
        self.payloads.append(payload)
        return None


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="add",
        description="Add",
        parameters={"a": "int", "b": "int"},
        execute=lambda *, a, b: {"sum": a + b},
    ))
    reg.register(ToolSpec(
        name="done",
        description="Finish",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


def _run_simple(hook, *, responses=None):
    responses = responses or [
        '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
    ]
    state = DefaultState(max_steps=5)
    return list(composable_loop(
        llm=MockLLMBackend(responses=responses),
        tools=_tools(),
        state=state,
        hooks=[hook],
        config=LoopConfig(max_steps=5),
    ))


class TestLifecycleEventDispatch:
    def test_session_start_fires_once(self):
        r = _Recorder()
        _run_simple(r)
        assert r.events.count(LifecycleEvent.SESSION_START) == 1

    def test_pre_and_post_llm_call_fire_per_step(self):
        r = _Recorder()
        _run_simple(r)
        # 2 steps × 2 events each
        assert r.events.count(LifecycleEvent.PRE_LLM_CALL) == 2
        assert r.events.count(LifecycleEvent.POST_LLM_RESPONSE) == 2

    def test_pre_and_post_tool_use_fire_per_call(self):
        r = _Recorder()
        _run_simple(r)
        assert r.events.count(LifecycleEvent.PRE_TOOL_USE) == 1  # non-done
        assert r.events.count(LifecycleEvent.POST_TOOL_USE) == 1

    def test_stop_fires_once_with_reason(self):
        r = _Recorder()
        _run_simple(r)
        stop_payloads = [p for p in r.payloads if p.event == LifecycleEvent.STOP]
        assert len(stop_payloads) == 1
        assert stop_payloads[0].termination_reason == "done"

    def test_post_llm_response_payload_carries_raw_response(self):
        r = _Recorder()
        _run_simple(r)
        post = [p for p in r.payloads if p.event == LifecycleEvent.POST_LLM_RESPONSE]
        assert all(p.raw_response for p in post)
        assert all(p.prompt for p in post)


class TestLifecycleEventDecisions:
    def test_stop_from_post_llm_response_halts_loop(self):
        class HaltAfterFirst:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.POST_LLM_RESPONSE:
                    return HookDecision(stop="halt_by_event")
                return None

        steps = _run_simple(HaltAfterFirst())
        # First tool call executes then loop exits; done is never reached.
        assert len(steps) == 1

    def test_pre_tool_use_rewrites_args(self):
        class Rewriter:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.PRE_TOOL_USE:
                    if payload.tool_call.tool == "add":
                        return HookDecision(updated_args={"a": 10, "b": 20})
                return None

        steps = _run_simple(Rewriter())
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_call.args == {"a": 10, "b": 20}
        assert add_step.tool_result.data == {"sum": 30}

    def test_pre_tool_use_short_circuits_with_result(self):
        from openharness.types import ToolResult

        class Cacher:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.PRE_TOOL_USE:
                    if payload.tool_call.tool == "add":
                        return HookDecision(updated_result=ToolResult(
                            tool="add", args_summary="cached",
                            data={"sum": 999}, error=None,
                        ))
                return None

        steps = _run_simple(Cacher())
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": 999}

    def test_post_tool_use_rewrites_result(self):
        from openharness.types import ToolResult

        class Scrubber:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.POST_TOOL_USE:
                    return HookDecision(updated_result=ToolResult(
                        tool=payload.tool_result.tool,
                        args_summary=payload.tool_result.args_summary,
                        data={"sum": "SCRUBBED"},
                        error=None,
                    ))
                return None

        steps = _run_simple(Scrubber())
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": "SCRUBBED"}

    def test_pre_tool_use_denies_with_permission(self):
        class Denier:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.PRE_TOOL_USE:
                    if payload.tool_call.tool == "add":
                        return HookDecision(permission="deny", block="no adds")
                return None

        steps = _run_simple(Denier())
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.error
        assert "no adds" in add_step.tool_result.error


class TestLifecycleEventSafety:
    def test_exceptions_in_on_event_do_not_break_loop(self):
        class Angry:
            def on_event(self, payload):
                raise RuntimeError("boom")

        # Loop completes normally even though every event dispatch raises.
        steps = _run_simple(Angry())
        assert len(steps) == 2

    def test_hook_without_on_event_is_ignored(self):
        class NoOnEvent:
            def pre_loop(self, state, session_log, context):
                return None

        steps = _run_simple(NoOnEvent())
        assert len(steps) == 2
