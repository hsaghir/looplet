"""Integration tests: ContextPressureHook emits compaction boundaries."""

from __future__ import annotations

from looplet.context import ContextPressureHook
from looplet.conversation import Conversation
from looplet.history import HistoryRecorder
from looplet.session import SessionLog
from looplet.types import DefaultState, Step, ToolCall, ToolResult


def _mk_state_with_huge_log(n_steps: int = 3) -> DefaultState:
    state = DefaultState()
    for i in range(1, n_steps + 1):
        state.steps.append(Step(
            number=i,
            tool_call=ToolCall(tool="t", args={}, reasoning="r" * 50),
            tool_result=ToolResult(
                tool="t", args_summary="",
                # Huge payload so blocking threshold trips
                data={"rows": [{"x": i}] * 2000, "blob": "A" * 200_000},
                result_key=f"k{i}",
            ),
        ))
    return state


class TestContextManagerEmitsBoundary:
    def test_emergency_compact_records_boundary(self):
        conv = Conversation()
        log = SessionLog()
        log.record(step=1, theory="t", tool="noop", reasoning="r",
                   entities=["e"], findings=[], highlights=[], recall_key="k")
        state = _mk_state_with_huge_log(3)

        recorder = HistoryRecorder(state=state, session_log=log, conversation=conv)
        hook = ContextPressureHook(
            llm=None,
            # Small context window so huge state blows past blocking
            context_window=1_000,
            compact_buffer=100,
            warning_buffer=200,
            blocking_buffer=50,
            recorder=recorder,
        )
        msg = hook.pre_prompt(state=state, session_log=log, context=None, step_num=4)
        # Emergency message returned
        assert msg is not None and "CONTEXT LIMIT" in msg
        # Boundary was emitted on the conversation
        boundaries = conv.find_compaction_boundaries()
        assert len(boundaries) == 1
        b = boundaries[0]
        assert b.metadata["kind"] == "compaction_boundary"
        assert b.metadata["dropped_step_range"] == (1, 3)
        assert "Emergency compaction" in b.metadata["summary"]

    def test_hook_without_recorder_does_not_raise(self):
        state = _mk_state_with_huge_log(2)
        log = SessionLog()
        hook = ContextPressureHook(
            llm=None,
            context_window=1_000,
            compact_buffer=100,
            warning_buffer=200,
            blocking_buffer=50,
        )  # no recorder
        # Should succeed silently
        msg = hook.pre_prompt(state=state, session_log=log, context=None, step_num=3)
        assert msg is not None
