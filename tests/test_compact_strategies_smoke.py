"""Smoke tests for PruneToolResults, compact_chain, cleanup callback,
and the TruncateCompact / SummarizeCompact renames."""
from __future__ import annotations

import pytest

from openharness import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from openharness.compact import (
    CompactOutcome,
    PruneToolResults,
    SummarizeCompact,
    TruncateCompact,
    compact_chain,
    run_compact,
)
from openharness.conversation import Conversation, Message, MessageRole
from openharness.session import SessionLog
from openharness.testing import MockLLMBackend
from openharness.tools import ToolSpec
from openharness.types import ToolResult

pytestmark = pytest.mark.smoke


# ── Helpers ──────────────────────────────────────────────────────


def _conv_with_tool_results(n: int) -> Conversation:
    """Build a conversation with n TOOL messages carrying result data."""
    conv = Conversation()
    for i in range(n):
        conv.append(Message(
            role=MessageRole.ASSISTANT,
            content=f"Calling tool_{i}",
        ))
        conv.append(Message(
            role=MessageRole.TOOL,
            content=f'{{"rows": [{{"col": "val_{i}"}}]}}',
            tool_result=ToolResult(tool=f"tool_{i}", args_summary="", data={"i": i}),
        ))
    return conv


def _tools():
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="done", description="d",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


def _log(n: int = 5) -> SessionLog:
    log = SessionLog()
    for i in range(1, n + 1):
        log.record(step=i, theory="t", tool=f"tool_{i}",
                   reasoning=f"r{i}", findings=[f"f{i}"])
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
            state=None, session_log=None, llm=None,
            conversation=conv, step_num=6, reason="test",
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
            state=None, session_log=None, llm=None,
            conversation=None, step_num=0, reason="test",
        )
        assert out.extra["mode"] == "no_conversation"

    def test_fewer_than_keep_recent_noop(self):
        conv = _conv_with_tool_results(3)
        svc = PruneToolResults(keep_recent=5)
        out = svc.compact(
            state=None, session_log=None, llm=None,
            conversation=conv, step_num=3, reason="test",
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
            state=None, session_log=None, llm=None,
            conversation=conv, step_num=4, reason="test",
        )
        # 2 matching tools, keep 1 → 1 cleared
        assert out.extra["cleared"] == 1

    def test_idempotent(self):
        conv = _conv_with_tool_results(4)
        svc = PruneToolResults(keep_recent=2)
        svc.compact(state=None, session_log=None, llm=None,
                    conversation=conv, step_num=4, reason="1st")
        out2 = svc.compact(state=None, session_log=None, llm=None,
                           conversation=conv, step_num=4, reason="2nd")
        # Already cleared — nothing new to clear
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
            TruncateCompact(keep_recent=1),   # will fire
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
            _Svc(), hooks=[], state=None, session_log=None,
            llm=None, conversation=None, step_num=0, reason="test",
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
            _Svc(), hooks=[], state=None, session_log=None,
            llm=None, conversation=None, step_num=0, reason="test",
        )

    def test_no_cleanup_noop(self):
        out = CompactOutcome(reason="test")
        assert out.cleanup is None


# ── Loop integration ─────────────────────────────────────────────


class TestLoopIntegration:
    def test_chain_in_loop(self):
        """compact_chain plugs into LoopConfig.compact_service."""
        reg = BaseToolRegistry()
        reg.register(ToolSpec(
            name="echo", description="e",
            parameters={"msg": "str"},
            execute=lambda *, msg: {"msg": msg},
        ))
        reg.register(ToolSpec(
            name="done", description="d",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        ))

        chain = compact_chain(
            PruneToolResults(keep_recent=2),
            TruncateCompact(keep_recent=1),
        )
        cfg = LoopConfig(max_steps=3, compact_service=chain)

        steps = list(composable_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"echo","args":{"msg":"a"},"reasoning":"r"}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=reg, state=DefaultState(max_steps=3),
            hooks=[], config=cfg,
        ))
        assert len(steps) == 2
