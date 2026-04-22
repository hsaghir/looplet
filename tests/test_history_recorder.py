"""Tests for HistoryRecorder — single write path for agent loop history.

HistoryRecorder unifies the three parallel representations
(Conversation messages / SessionLog entries / state.steps) behind a
single API: ``record_step()`` and ``record_llm_turn()``. Each call
updates all attached surfaces so they stay in lockstep — removing the
"write three times" duplication in the loop.
"""

from __future__ import annotations

from looplet.conversation import Conversation, MessageRole
from looplet.history import HistoryRecorder
from looplet.session import SessionLog
from looplet.types import DefaultState, Step, ToolCall, ToolResult


def _mk_step(num: int, tool: str = "t", data: object | None = None) -> Step:
    return Step(
        number=num,
        tool_call=ToolCall(tool=tool, args={"a": 1}, reasoning=f"r{num}"),
        tool_result=ToolResult(
            tool=tool,
            args_summary=f"a={num}",
            data=data or {"x": num},
            result_key=f"k{num}",
        ),
    )


class TestHistoryRecorderRecordStep:
    def test_writes_to_state_steps(self):
        state = DefaultState()
        rec = HistoryRecorder(state=state)
        step = _mk_step(1)
        rec.record_step(step, theory="t1", entities=["e1"], findings=["f1"])
        assert state.steps == [step]

    def test_writes_to_session_log(self):
        log = SessionLog()
        rec = HistoryRecorder(session_log=log)
        step = _mk_step(2, tool="search")
        rec.record_step(
            step,
            theory="theory-A",
            entities=["host1"],
            findings=["finding1"],
            highlights=["h"],
            recall_key="k2",
        )
        assert len(log.entries) == 1
        e = log.entries[0]
        assert e.step == 2
        assert e.tool == "search"
        assert e.theory == "theory-A"
        assert e.reasoning == "r2"
        assert e.entities_seen == ["host1"]
        assert e.findings == ["finding1"]
        assert e.highlights == ["h"]
        assert e.recall_key == "k2"

    def test_writes_to_conversation(self):
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv)
        step = _mk_step(3)
        rec.record_step(step, theory="", entities=[])
        # One assistant msg with tool_call + one tool msg with tool_result
        assert len(conv.messages) == 2
        m_assistant, m_tool = conv.messages
        assert m_assistant.role == MessageRole.ASSISTANT
        assert m_assistant.tool_call is step.tool_call
        assert m_tool.role == MessageRole.TOOL
        assert m_tool.tool_result is step.tool_result

    def test_writes_to_all_three_in_one_call(self):
        state = DefaultState()
        log = SessionLog()
        conv = Conversation()
        rec = HistoryRecorder(state=state, session_log=log, conversation=conv)
        step = _mk_step(4)
        rec.record_step(
            step, theory="Tx", entities=["e"], findings=["f"], highlights=["h"], recall_key="k4"
        )
        assert state.steps == [step]
        assert len(log.entries) == 1 and log.entries[0].step == 4
        assert len(conv.messages) == 2
        assert log.current_theory == "Tx"

    def test_does_not_append_step_twice_if_already_in_state(self):
        """When the caller has already appended (migration scenario), recorder
        must detect and not duplicate."""
        state = DefaultState()
        log = SessionLog()
        rec = HistoryRecorder(state=state, session_log=log)
        step = _mk_step(5)
        state.steps.append(step)  # caller pre-appended
        rec.record_step(step, theory="", entities=[])
        assert state.steps == [step]  # no duplicate
        assert len(log.entries) == 1  # log still recorded

    def test_record_step_preserves_theory_across_calls_when_blank(self):
        log = SessionLog()
        rec = HistoryRecorder(session_log=log)
        rec.record_step(_mk_step(1), theory="theory-1", entities=[])
        rec.record_step(_mk_step(2), theory="", entities=[])
        assert log.entries[1].theory == "theory-1"


class TestHistoryRecorderRecordLLMTurn:
    def test_appends_user_then_assistant_message(self):
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv)
        rec.record_llm_turn(prompt="what now?", response="thinking…")
        assert [m.role for m in conv.messages] == [MessageRole.USER, MessageRole.ASSISTANT]
        assert conv.messages[0].content == "what now?"
        assert conv.messages[1].content == "thinking…"

    def test_truncates_large_prompt_and_response(self):
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv, max_message_chars=100)
        rec.record_llm_turn(prompt="x" * 500, response="y" * 500)
        assert len(conv.messages[0].content) == 100
        assert len(conv.messages[1].content) == 100

    def test_no_op_when_no_conversation(self):
        rec = HistoryRecorder()
        rec.record_llm_turn(prompt="p", response="r")  # must not raise

    def test_record_non_string_response_coerces(self):
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv)
        rec.record_llm_turn(prompt="p", response=[{"type": "tool_use"}])
        # still gets recorded as string
        assert isinstance(conv.messages[1].content, str)
        assert "tool_use" in conv.messages[1].content
