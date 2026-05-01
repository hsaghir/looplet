"""Smoke tests for SUBAGENT_START / SUBAGENT_STOP lifecycle events."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    LifecycleEvent,
)
from looplet.subagent import run_sub_loop
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="add",
            description="Add",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="Finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


class _Recorder:
    def __init__(self):
        self.events: list[LifecycleEvent] = []
        self.payloads = []

    def on_event(self, payload):
        self.events.append(payload.event)
        self.payloads.append(payload)
        return None


class TestSubagentLifecycleEvents:
    def test_subagent_start_and_stop_fire(self):
        r = _Recorder()
        run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
        )
        assert LifecycleEvent.SUBAGENT_START in r.events
        assert LifecycleEvent.SUBAGENT_STOP in r.events

    def test_subagent_id_correlates_events(self):
        r = _Recorder()
        run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
            subagent_id="custom-id",
        )
        ids = [
            p.subagent_id
            for p in r.payloads
            if p.event in (LifecycleEvent.SUBAGENT_START, LifecycleEvent.SUBAGENT_STOP)
        ]
        assert ids == ["custom-id", "custom-id"]

    def test_result_carries_subagent_id(self):
        result = run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            subagent_id="correlation-7",
        )
        assert result["subagent_id"] == "correlation-7"

    def test_auto_generated_id_when_not_supplied(self):
        r = _Recorder()
        result = run_sub_loop(
            llm=MockLLMBackend(
                responses=[
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            ),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
        )
        assert result["subagent_id"]  # non-empty
        assert len(result["subagent_id"]) == 12


# ── parent_hooks=... forwards sub-loop events to parent observers ──


def test_parent_hooks_receive_sub_loop_events() -> None:
    """``run_sub_loop(parent_hooks=...)`` forwards every lifecycle
    event the sub-loop emits onto the parent's hooks via their
    ``on_event`` method, tagged with ``subagent_id`` in
    ``payload.extra``. Lets parent observability stacks (MetricsHook,
    StreamingHook, TrajectoryRecorder) see sub-activity without the
    user manually plumbing it."""
    import json

    from looplet import register_done_tool
    from looplet.subagent import run_sub_loop
    from looplet.testing import MockLLMBackend
    from looplet.tools import BaseToolRegistry

    received: list[tuple[str, dict | None]] = []

    class _Spy:
        def on_event(self, payload) -> None:
            received.append((str(payload.event), dict(payload.extra or {})))

    parent_hook = _Spy()

    tools = BaseToolRegistry()
    register_done_tool(tools)

    sub_llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "done", "args": {"summary": "sub-done"}, "reasoning": "r"}),
        ]
    )

    result = run_sub_loop(
        llm=sub_llm,
        task={"goal": "do nothing"},
        tools=tools,
        max_steps=2,
        parent_hooks=[parent_hook],
    )
    assert result["subagent_id"]
    # The parent's spy received SUBAGENT_START + SUBAGENT_STOP at minimum,
    # both tagged with the same subagent_id.
    event_names = [name for name, _ in received]
    assert any("SUBAGENT_START" in n for n in event_names), event_names
    assert any("SUBAGENT_STOP" in n for n in event_names), event_names
    # And every event payload carries the subagent_id.
    sub_ids = {extra.get("subagent_id") for _, extra in received if extra}
    sub_ids.discard(None)
    assert sub_ids == {result["subagent_id"]}, sub_ids


def test_no_parent_hooks_means_no_forwarding() -> None:
    """Without ``parent_hooks``, the sub-loop's events stay isolated
    (default opt-in behaviour)."""
    import json

    from looplet import register_done_tool
    from looplet.subagent import run_sub_loop
    from looplet.testing import MockLLMBackend
    from looplet.tools import BaseToolRegistry

    received: list[str] = []

    class _Spy:
        def on_event(self, payload) -> None:
            received.append(str(payload.event))

    parent_hook = _Spy()

    tools = BaseToolRegistry()
    register_done_tool(tools)
    sub_llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "done", "args": {"summary": "x"}, "reasoning": "r"}),
        ]
    )

    # parent_hooks omitted → spy must see nothing.
    run_sub_loop(
        llm=sub_llm,
        task={"goal": "x"},
        tools=tools,
        max_steps=2,
        # hooks= is the SUB-loop's hooks — does not implicitly include
        # the parent observer.
    )
    assert received == []
    # And: even if you pass [parent_hook] as hooks=, it gets the
    # sub-loop's full LoopHook interface (which it doesn't implement),
    # not just on_event forwarding. The on_event method is still
    # called for events, so we'd see them — that's the OLD behaviour.
    # The new behaviour is that you can have parent observers see
    # events WITHOUT sharing the sub-loop's full hook interface.
