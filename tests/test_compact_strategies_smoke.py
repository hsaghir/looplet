"""Smoke tests for PruneToolResults, compact_chain, cleanup callback,
and the TruncateCompact / SummarizeCompact renames."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.compact import (
    CompactOutcome,
    DefaultCompactService,
    PruneToolResults,
    SummarizeCompact,
    TruncateCompact,
    compact_chain,
    run_compact,
)
from looplet.conversation import Conversation, Message, MessageRole
from looplet.session import SessionLog
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import ToolResult

pytestmark = pytest.mark.smoke


# ── Helpers ──────────────────────────────────────────────────────


def _conv_with_tool_results(n: int) -> Conversation:
    """Build a conversation with n TOOL messages carrying result data."""
    conv = Conversation()
    for i in range(n):
        conv.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=f"Calling tool_{i}",
            )
        )
        conv.append(
            Message(
                role=MessageRole.TOOL,
                content=f'{{"rows": [{{"col": "val_{i}"}}]}}',
                tool_result=ToolResult(tool=f"tool_{i}", args_summary="", data={"i": i}),
            )
        )
    return conv


def _tools():
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="done",
            description="d",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


def _log(n: int = 5) -> SessionLog:
    log = SessionLog()
    for i in range(1, n + 1):
        log.record(step=i, theory="t", tool=f"tool_{i}", reasoning=f"r{i}", findings=[f"f{i}"])
    return log


# ── Renames / aliases ────────────────────────────────────────────


class TestRenames:
    def test_truncate_compact_is_canonical(self):
        assert TruncateCompact is not None
        svc = TruncateCompact(keep_recent=1)
        assert hasattr(svc, "compact")

    def test_summarize_compact_is_canonical(self):
        assert SummarizeCompact is not None
        svc = SummarizeCompact(keep_recent=1)
        assert hasattr(svc, "compact")


# ── PruneToolResults ─────────────────────────────────────────────


class TestPruneToolResults:
    def test_clears_old_tool_results(self):
        conv = _conv_with_tool_results(6)
        svc = PruneToolResults(keep_recent=2)
        out = svc.compact(
            state=None,
            session_log=None,
            llm=None,
            conversation=conv,
            step_num=6,
            reason="test",
        )
        # 6 tool messages, keep 2 → 4 cleared
        assert out.extra["cleared"] == 4
        assert out.extra["mode"] == "prune"
        assert out.messages_after == out.messages_before  # structure unchanged

        tool_msgs = [m for m in conv.messages if m.role == MessageRole.TOOL]
        cleared = [m for m in tool_msgs if m.content == "[tool result cleared by compact]"]
        kept = [m for m in tool_msgs if m.content != "[tool result cleared by compact]"]
        assert len(cleared) == 4
        assert len(kept) == 2
        # The kept ones are the LAST two
        assert kept[0].tool_result.data == {"i": 4}
        assert kept[1].tool_result.data == {"i": 5}

    def test_no_conversation_noop(self):
        svc = PruneToolResults()
        out = svc.compact(
            state=None,
            session_log=None,
            llm=None,
            conversation=None,
            step_num=0,
            reason="test",
        )
        assert out.extra["mode"] == "no_conversation"

    def test_fewer_than_keep_recent_noop(self):
        conv = _conv_with_tool_results(3)
        svc = PruneToolResults(keep_recent=5)
        out = svc.compact(
            state=None,
            session_log=None,
            llm=None,
            conversation=conv,
            step_num=3,
            reason="test",
        )
        assert out.extra["cleared"] == 0

    def test_compactable_tools_filter(self):
        conv = _conv_with_tool_results(4)
        # Only prune tool_0 and tool_1
        svc = PruneToolResults(
            keep_recent=1,
            compactable_tools=frozenset({"tool_0", "tool_1"}),
        )
        out = svc.compact(
            state=None,
            session_log=None,
            llm=None,
            conversation=conv,
            step_num=4,
            reason="test",
        )
        # 2 matching tools, keep 1 → 1 cleared
        assert out.extra["cleared"] == 1

    def test_idempotent(self):
        conv = _conv_with_tool_results(4)
        svc = PruneToolResults(keep_recent=2)
        svc.compact(
            state=None, session_log=None, llm=None, conversation=conv, step_num=4, reason="1st"
        )
        out2 = svc.compact(
            state=None, session_log=None, llm=None, conversation=conv, step_num=4, reason="2nd"
        )
        # Already cleared - nothing new to clear
        assert out2.extra["cleared"] == 0


# ── compact_chain ────────────────────────────────────────────────


class TestCompactChain:
    def test_first_effective_stage_wins(self):
        conv = _conv_with_tool_results(6)
        chain = compact_chain(
            PruneToolResults(keep_recent=2),
            TruncateCompact(keep_recent=1),
        )
        out = chain.compact(
            state=type("S", (), {"steps": []})(),
            session_log=_log(),
            llm=None,
            conversation=conv,
            step_num=6,
            reason="test",
        )
        # Prune cleared 4, so chain stopped at stage 0
        assert out.extra["chain_stage"] == 0
        assert out.extra["cleared"] == 4
        assert out.llm_calls_spent == 0

    def test_falls_through_when_no_effect(self):
        conv = _conv_with_tool_results(2)  # only 2 → nothing to prune
        chain = compact_chain(
            PruneToolResults(keep_recent=5),  # won't fire
            TruncateCompact(keep_recent=1),  # will fire
        )
        out = chain.compact(
            state=type("S", (), {"steps": []})(),
            session_log=_log(),
            llm=None,
            conversation=conv,
            step_num=2,
            reason="test",
        )
        # Fell through to stage 1
        assert out.extra["chain_stage"] == 1

    def test_requires_at_least_one_service(self):
        with pytest.raises(ValueError, match="at least one"):
            compact_chain()

    def test_single_service_passthrough(self):
        chain = compact_chain(TruncateCompact(keep_recent=1))
        out = chain.compact(
            state=type("S", (), {"steps": []})(),
            session_log=_log(),
            llm=None,
            conversation=None,
            step_num=5,
            reason="test",
        )
        assert out.extra["chain_stage"] == 0

    def test_session_log_only_compaction_counts_as_effective(self):
        class SessionOnlyCompact:
            def compact(self, *, session_log, reason, **kwargs):
                before = len(session_log.entries)
                del session_log.entries[:-1]
                return CompactOutcome(
                    reason=reason,
                    session_entries_before=before,
                    session_entries_after=len(session_log.entries),
                    compacted_step_range=(1, before - 1),
                    extra={"mode": "session_only"},
                )

        chain = compact_chain(SessionOnlyCompact(), TruncateCompact(keep_recent=1))
        log = _log(5)
        out = chain.compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=None,
            conversation=None,
            step_num=5,
            reason="test",
        )
        assert out.extra["chain_stage"] == 0
        assert len(log.entries) == 1


# ── CompactOutcome.cleanup ───────────────────────────────────────


class TestCleanupCallback:
    def test_cleanup_called_by_run_compact(self):
        called = []

        class _Svc:
            def compact(self, **kw):
                return CompactOutcome(
                    reason="test",
                    cleanup=lambda: called.append("cleaned"),
                )

        run_compact(
            _Svc(),
            hooks=[],
            state=None,
            session_log=None,
            llm=None,
            conversation=None,
            step_num=0,
            reason="test",
        )
        assert called == ["cleaned"]

    def test_cleanup_exception_swallowed(self):
        class _Svc:
            def compact(self, **kw):
                return CompactOutcome(
                    reason="test",
                    cleanup=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                )

        # Must not raise
        run_compact(
            _Svc(),
            hooks=[],
            state=None,
            session_log=None,
            llm=None,
            conversation=None,
            step_num=0,
            reason="test",
        )

    def test_no_cleanup_noop(self):
        out = CompactOutcome(reason="test")
        assert out.cleanup is None


# ── Loop integration ─────────────────────────────────────────────


class TestLoopIntegration:
    def test_chain_in_loop(self):
        """compact_chain plugs into LoopConfig.compact_service."""
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="echo",
                description="e",
                parameters={"msg": "str"},
                execute=lambda *, msg: {"msg": msg},
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )

        chain = compact_chain(
            PruneToolResults(keep_recent=2),
            TruncateCompact(keep_recent=1),
        )
        cfg = LoopConfig(max_steps=3, compact_service=chain)

        steps = list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"echo","args":{"msg":"a"},"reasoning":"r"}',
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                    ]
                ),
                tools=reg,
                state=DefaultState(max_steps=3),
                hooks=[],
                config=cfg,
            )
        )
        assert len(steps) == 2


class _SummaryLLM:
    def __init__(self, text: str = "LLM summary kept") -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.calls += 1
        return self.text


class TestSummarizeCompact:
    def test_reuses_llm_summary_for_conversation_boundary(self):
        conv = _conv_with_tool_results(5)
        log = _log(5)
        llm = _SummaryLLM("TASK: preserve this summary")
        out = SummarizeCompact(keep_recent=2).compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=llm,
            conversation=conv,
            step_num=5,
            reason="test",
        )
        assert out.summary == "TASK: preserve this summary"
        boundaries = conv.find_compaction_boundaries()
        assert boundaries
        assert boundaries[-1].metadata["summary"] == "TASK: preserve this summary"
        assert out.session_entries_after < out.session_entries_before

    def test_short_session_log_does_not_grow(self):
        conv = _conv_with_tool_results(4)
        log = _log(2)
        llm = _SummaryLLM("summary for conversation only")
        out = SummarizeCompact(keep_recent=1).compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=llm,
            conversation=conv,
            step_num=4,
            reason="test",
        )
        assert out.summary == "summary for conversation only"
        assert out.session_entries_after == out.session_entries_before
        assert out.compacted_step_range is None


class TestDefaultCompactService:
    def test_prunes_and_summarizes_in_one_service(self):
        conv = _conv_with_tool_results(8)
        log = _log(8)
        llm = _SummaryLLM("default summary")
        out = DefaultCompactService(keep_recent=2, keep_recent_tool_results=3).compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=llm,
            conversation=conv,
            step_num=8,
            reason="proactive",
        )
        assert out.compacted is True
        assert out.llm_calls_spent == 1
        assert out.summary == "default summary"
        assert out.extra["mode"] == "default"
        stage_names = [stage["name"] for stage in out.extra["stages"]]
        assert stage_names == ["prune_tool_results", "summarize_context"]
        assert out.extra["cleared"] > 0

    def test_can_run_without_llm_summary(self):
        log = _log(6)
        out = DefaultCompactService(keep_recent=2, use_llm_summary=False).compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=None,
            conversation=None,
            step_num=6,
            reason="offline",
        )
        assert out.llm_calls_spent == 0
        assert out.session_entries_after < out.session_entries_before
        assert out.extra["use_llm_summary"] is False

    def test_repeat_compaction_does_not_expand_conversation(self):
        conv = Conversation()
        for step in range(1, 8):
            conv.append(Message(role=MessageRole.USER, content=f"user {step}"))
            conv.append(Message(role=MessageRole.ASSISTANT, content=f"assistant {step}"))
            conv.append(
                Message(
                    role=MessageRole.TOOL,
                    content="large result" * 200,
                    tool_result=ToolResult(tool="tool", args_summary="", data={"step": step}),
                )
            )
        log = _log(7)
        service = DefaultCompactService(keep_recent=2, keep_recent_tool_results=2)

        first = service.compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=_SummaryLLM("first summary"),
            conversation=conv,
            step_num=8,
            reason="first",
        )
        messages_after_first = len(conv.messages)
        entries_after_first = len(log.entries)
        second = service.compact(
            state=type("S", (), {"steps": []})(),
            session_log=log,
            llm=_SummaryLLM("second summary"),
            conversation=conv,
            step_num=9,
            reason="second",
        )

        assert first.compacted is True
        assert len(conv.messages) <= messages_after_first
        assert len(log.entries) <= entries_after_first
        assert second.messages_after <= second.messages_before
        assert second.session_entries_after <= second.session_entries_before
