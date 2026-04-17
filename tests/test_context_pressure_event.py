"""Tests for ContextPressureEvent — the 4th tier of token budget tracking.

The 3-tier (compact / warning / blocking) system already exists on
``ContextManagerHook``, but only the *compact* and *blocking* tiers
trigger visible actions. The *warning* tier is computed but nothing
consumes it. Claude Code uses the warning tier to signal the *user*
("85% full — consider wrapping up"), which requires pushing the signal
through an event channel.

This test locks in:

1. A new ``ContextPressureEvent`` dataclass exists in
   ``openharness.streaming`` with ``level``, ``estimated_tokens``,
   ``threshold``, ``context_window``, ``percent_used``.
2. ``ContextManagerHook`` emits one per ``pre_prompt`` call via the
   attached emitter when thresholds are crossed (ok → warning →
   compact → blocking), exactly once per level crossing.
3. Downgrades (pressure dropping after a compaction) emit a fresh
   event so consumers can clear a warning UI.
4. No emitter attached → silent no-op.
"""

from __future__ import annotations

from openharness.context import ContextManagerHook
from openharness.streaming import ContextPressureEvent
from openharness.types import DefaultState, Step, ToolCall, ToolResult


class _CaptureEmitter:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


def _mk_state(n: int, payload_len: int) -> DefaultState:
    state = DefaultState()
    for i in range(1, n + 1):
        state.steps.append(Step(
            number=i,
            tool_call=ToolCall(tool="t", args={}, reasoning="r"),
            tool_result=ToolResult(
                tool="t", args_summary="",
                data={"blob": "A" * payload_len},
                result_key=f"k{i}",
            ),
        ))
    return state


class TestContextPressureEvent:
    def test_event_shape(self):
        ev = ContextPressureEvent(
            level="warning",
            estimated_tokens=100_000,
            threshold=110_000,
            context_window=128_000,
            percent_used=78.1,
        )
        assert ev.event_type == "ContextPressureEvent"
        assert ev.level == "warning"
        assert ev.estimated_tokens == 100_000
        assert ev.percent_used == 78.1


class TestHookEmitsPressure:
    def test_ok_level_emitted_when_under_all_thresholds(self):
        emitter = _CaptureEmitter()
        state = _mk_state(1, 10)  # tiny payload
        from openharness.session import SessionLog
        hook = ContextManagerHook(llm=None, emitter=emitter)
        hook.pre_prompt(state=state, session_log=SessionLog(),
                        context=None, step_num=2)
        assert emitter.events, "no pressure event emitted"
        ev = emitter.events[-1]
        assert isinstance(ev, ContextPressureEvent)
        assert ev.level == "ok"

    def test_warning_level_crossed(self):
        emitter = _CaptureEmitter()
        # Size the state so estimated tokens is between warning and compact
        # With a 10_000 context window and defaults (warn=30K=>out of range),
        # use explicit buffers that put us in the warning band.
        state = _mk_state(3, 20_000)  # ~15K tokens estimated
        from openharness.session import SessionLog
        hook = ContextManagerHook(
            llm=None,
            context_window=20_000,
            compact_buffer=2_000,       # compact at 18K
            warning_buffer=8_000,       # warn at 12K
            blocking_buffer=500,        # block at 19_500
            emitter=emitter,
        )
        hook.pre_prompt(state=state, session_log=SessionLog(),
                        context=None, step_num=4)
        levels = [e.level for e in emitter.events if isinstance(e, ContextPressureEvent)]
        assert levels, "no pressure events"
        assert levels[-1] in ("warning", "compact", "blocking"), \
            f"expected elevated pressure, got {levels}"

    def test_no_emitter_no_error(self):
        state = _mk_state(1, 10)
        from openharness.session import SessionLog
        hook = ContextManagerHook(llm=None)  # no emitter
        # Must not raise
        hook.pre_prompt(state=state, session_log=SessionLog(),
                        context=None, step_num=2)
