"""Tests for looplet.streaming — structured event emission for agent observability."""

from __future__ import annotations

import queue
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from looplet.streaming import (
    CallbackEmitter,
    CompositeEmitter,
    Event,
    EventEmitter,
    HookEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LoopEndEvent,
    LoopStartEvent,
    QueueEmitter,
    RecoveryEvent,
    StepEndEvent,
    StepStartEvent,
    StreamingHook,
    ToolDispatchEvent,
    ToolResultEvent,
)
from looplet.types import ToolCall, ToolResult

# ── Base Event ──────────────────────────────────────────────────


def test_event_has_event_type_and_timestamp():
    e = Event(event_type="test_event")
    assert e.event_type == "test_event"
    assert isinstance(e.timestamp, float)
    assert e.timestamp > 0


def test_event_timestamp_defaults_to_now():
    before = time.time()
    e = Event(event_type="x")
    after = time.time()
    assert before <= e.timestamp <= after


def test_event_custom_timestamp():
    e = Event(event_type="x", timestamp=1234.5)
    assert e.timestamp == 1234.5


# ── Concrete Events auto event_type ────────────────────────────


def test_loop_start_event_creation():
    e = LoopStartEvent(task_summary="Find vulnerabilities", max_steps=15)
    assert e.task_summary == "Find vulnerabilities"
    assert e.max_steps == 15
    assert e.event_type == "LoopStartEvent"
    assert isinstance(e.timestamp, float)


def test_step_start_event_creation():
    e = StepStartEvent(step_num=3)
    assert e.step_num == 3
    assert e.event_type == "StepStartEvent"


def test_llm_call_start_event_creation():
    e = LLMCallStartEvent(step_num=2, prompt_tokens_est=500)
    assert e.step_num == 2
    assert e.prompt_tokens_est == 500
    assert e.event_type == "LLMCallStartEvent"


def test_llm_call_end_event_creation():
    e = LLMCallEndEvent(step_num=2, response_length=300, duration_ms=150.5)
    assert e.step_num == 2
    assert e.response_length == 300
    assert e.duration_ms == 150.5
    assert e.event_type == "LLMCallEndEvent"


def test_tool_dispatch_event_creation():
    e = ToolDispatchEvent(step_num=1, tool_name="search", args_summary="query=foo")
    assert e.step_num == 1
    assert e.tool_name == "search"
    assert e.args_summary == "query=foo"
    assert e.event_type == "ToolDispatchEvent"


def test_tool_result_event_creation():
    e = ToolResultEvent(step_num=1, tool_name="search", duration_ms=50.0, has_error=False)
    assert e.step_num == 1
    assert e.tool_name == "search"
    assert e.duration_ms == 50.0
    assert e.has_error is False
    assert e.event_type == "ToolResultEvent"


def test_tool_result_event_with_error():
    e = ToolResultEvent(step_num=2, tool_name="broken", duration_ms=10.0, has_error=True)
    assert e.has_error is True


def test_step_end_event_creation():
    e = StepEndEvent(step_num=5, classification="continue", new_entities_count=3)
    assert e.step_num == 5
    assert e.classification == "continue"
    assert e.new_entities_count == 3
    assert e.event_type == "StepEndEvent"


def test_loop_end_event_creation():
    e = LoopEndEvent(total_steps=10, total_llm_calls=12, reason="done")
    assert e.total_steps == 10
    assert e.total_llm_calls == 12
    assert e.reason == "done"
    assert e.event_type == "LoopEndEvent"


def test_hook_event_creation():
    e = HookEvent(hook_name="MyHook", method="post_dispatch", message="entities found: 3")
    assert e.hook_name == "MyHook"
    assert e.method == "post_dispatch"
    assert e.message == "entities found: 3"
    assert e.event_type == "HookEvent"


def test_recovery_event_creation():
    e = RecoveryEvent(strategy="retry", success=True)
    assert e.strategy == "retry"
    assert e.success is True
    assert e.event_type == "RecoveryEvent"


# ── EventEmitter Protocol ───────────────────────────────────────


def test_event_emitter_is_protocol():
    # Protocol classes cannot be instantiated directly
    assert hasattr(EventEmitter, "emit")


def test_callback_emitter_satisfies_protocol():
    cb = CallbackEmitter(lambda e: None)
    assert isinstance(cb, EventEmitter)


def test_queue_emitter_satisfies_protocol():
    q = queue.Queue()
    qe = QueueEmitter(q)
    assert isinstance(qe, EventEmitter)


# ── CallbackEmitter ─────────────────────────────────────────────


def test_callback_emitter_calls_callback():
    received: list[Event] = []
    emitter = CallbackEmitter(received.append)
    ev = LoopStartEvent(task_summary="test", max_steps=5)
    emitter.emit(ev)
    assert len(received) == 1
    assert received[0] is ev


def test_callback_emitter_multiple_events():
    received: list[Event] = []
    emitter = CallbackEmitter(received.append)
    emitter.emit(StepStartEvent(step_num=1))
    emitter.emit(StepStartEvent(step_num=2))
    emitter.emit(LoopEndEvent(total_steps=2, total_llm_calls=3, reason="done"))
    assert len(received) == 3
    assert received[1].step_num == 2


# ── QueueEmitter ────────────────────────────────────────────────


def test_queue_emitter_enqueues_event():
    q: queue.Queue[Event] = queue.Queue()
    emitter = QueueEmitter(q)
    ev = StepStartEvent(step_num=1)
    emitter.emit(ev)
    assert not q.empty()
    got = q.get_nowait()
    assert got is ev


def test_queue_emitter_multiple_events_in_order():
    q: queue.Queue[Event] = queue.Queue()
    emitter = QueueEmitter(q)
    events = [StepStartEvent(step_num=i) for i in range(5)]
    for e in events:
        emitter.emit(e)
    for expected in events:
        got = q.get_nowait()
        assert got is expected


# ── CompositeEmitter ────────────────────────────────────────────


def test_composite_emitter_fans_out():
    a: list[Event] = []
    b: list[Event] = []
    composite = CompositeEmitter([CallbackEmitter(a.append), CallbackEmitter(b.append)])
    ev = LoopStartEvent(task_summary="x", max_steps=1)
    composite.emit(ev)
    assert len(a) == 1 and a[0] is ev
    assert len(b) == 1 and b[0] is ev


def test_composite_emitter_empty_list():
    composite = CompositeEmitter([])
    # Should not raise
    composite.emit(Event(event_type="test"))


def test_composite_emitter_three_emitters():
    received = [[], [], []]
    composite = CompositeEmitter(
        [
            CallbackEmitter(received[0].append),
            CallbackEmitter(received[1].append),
            CallbackEmitter(received[2].append),
        ]
    )
    ev = StepStartEvent(step_num=7)
    composite.emit(ev)
    for r in received:
        assert len(r) == 1 and r[0] is ev


def test_composite_emitter_satisfies_protocol():
    composite = CompositeEmitter([])
    assert isinstance(composite, EventEmitter)


# ── StreamingHook implements LoopHook ───────────────────────────


def test_streaming_hook_is_loop_hook():
    from looplet.loop import LoopHook

    received: list[Event] = []
    hook = StreamingHook(CallbackEmitter(received.append))
    assert isinstance(hook, LoopHook)


def test_streaming_hook_pre_prompt_emits_step_start():
    received: list[Event] = []
    hook = StreamingHook(CallbackEmitter(received.append))
    state = MagicMock()
    session_log = MagicMock()
    result = hook.pre_prompt(state, session_log, context=None, step_num=3)
    assert result is None  # hook should not inject text
    step_events = [e for e in received if isinstance(e, StepStartEvent)]
    assert len(step_events) == 1
    assert step_events[0].step_num == 3


def test_streaming_hook_post_dispatch_emits_tool_result_and_step_end():
    received: list[Event] = []
    hook = StreamingHook(CallbackEmitter(received.append))

    state = MagicMock()
    session_log = MagicMock()
    tool_call = ToolCall(tool="search", args={"query": "foo"})
    tool_result = ToolResult(tool="search", args_summary="query=foo", data={"results": []})

    result = hook.post_dispatch(state, session_log, tool_call, tool_result, step_num=2)
    # May return None or str (briefing injection)

    event_types = [type(e).__name__ for e in received]
    assert "ToolResultEvent" in event_types
    assert "StepEndEvent" in event_types


def test_streaming_hook_post_dispatch_has_error_flag():
    received: list[Event] = []
    hook = StreamingHook(CallbackEmitter(received.append))

    state = MagicMock()
    session_log = MagicMock()
    tool_call = ToolCall(tool="bad_tool", args={})
    tool_result = ToolResult(tool="bad_tool", args_summary="", data=None, error="boom")

    hook.post_dispatch(state, session_log, tool_call, tool_result, step_num=5)

    tr_events = [e for e in received if isinstance(e, ToolResultEvent)]
    assert tr_events[0].has_error is True
    assert tr_events[0].tool_name == "bad_tool"


def test_streaming_hook_on_loop_end_emits_loop_end():
    received: list[Event] = []
    hook = StreamingHook(CallbackEmitter(received.append))

    state = MagicMock()
    session_log = MagicMock()
    result = hook.on_loop_end(state, session_log, context=None, llm=None)
    assert isinstance(result, int)  # returns extra LLM call count

    le_events = [e for e in received if isinstance(e, LoopEndEvent)]
    assert len(le_events) == 1


def test_streaming_hook_loop_end_accumulates_step_count():
    received: list[Event] = []
    hook = StreamingHook(CallbackEmitter(received.append))

    state = MagicMock()
    state.step_count = 7
    session_log = MagicMock()

    hook.on_loop_end(state, session_log, context=None, llm=None)

    le_events = [e for e in received if isinstance(e, LoopEndEvent)]
    assert le_events[0].total_steps == 7


def test_streaming_hook_pre_dispatch_returns_none():
    hook = StreamingHook(CallbackEmitter(lambda e: None))
    tc = ToolCall(tool="x", args={})
    state, session_log = MagicMock(), MagicMock()
    result = hook.pre_dispatch(state, session_log, tc, step_num=1)
    assert result is None  # never intercepts


def test_streaming_hook_check_done_returns_none():
    hook = StreamingHook(CallbackEmitter(lambda e: None))
    state, session_log = MagicMock(), MagicMock()
    result = hook.check_done(state, session_log, context=None, step_num=1)
    assert result is None


def test_streaming_hook_should_stop_returns_false():
    hook = StreamingHook(CallbackEmitter(lambda e: None))
    state = MagicMock()
    result = hook.should_stop(state, step_num=1, new_entities=0)
    assert result is False
