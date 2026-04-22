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
            llm=MockLLMBackend(responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
        )
        assert LifecycleEvent.SUBAGENT_START in r.events
        assert LifecycleEvent.SUBAGENT_STOP in r.events

    def test_subagent_id_correlates_events(self):
        r = _Recorder()
        run_sub_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
            subagent_id="custom-id",
        )
        ids = [
            p.subagent_id for p in r.payloads
            if p.event in (LifecycleEvent.SUBAGENT_START, LifecycleEvent.SUBAGENT_STOP)
        ]
        assert ids == ["custom-id", "custom-id"]

    def test_result_carries_subagent_id(self):
        result = run_sub_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(),
            max_steps=3,
            subagent_id="correlation-7",
        )
        assert result["subagent_id"] == "correlation-7"

    def test_auto_generated_id_when_not_supplied(self):
        r = _Recorder()
        result = run_sub_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(),
            max_steps=3,
            hooks=[r],
        )
        assert result["subagent_id"]  # non-empty
        assert len(result["subagent_id"]) == 12
