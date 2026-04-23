"""Round-8 friction fixes: MCP leak + state.conversation + BudgetTelemetry."""

from __future__ import annotations

import pytest

from looplet import BaseToolRegistry, DefaultState, LoopConfig, composable_loop
from looplet.budget import BudgetTelemetry, ContextBudget
from looplet.conversation import Conversation, Message
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


class TestStateStashesConversation:
    def test_conversation_visible_on_state(self):
        captured: dict = {}

        class _Spy:
            def pre_loop(self, state, session_log, context):
                captured["conv"] = getattr(state, "conversation", None)

        conv = Conversation()
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']
                ),
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
                task={},
                conversation=conv,
                hooks=[_Spy()],
            )
        )
        assert captured["conv"] is conv


class TestBudgetTelemetryUsesConversation:
    def test_non_zero_estimate_with_conversation(self):
        budget = ContextBudget(context_window=1000, warning_at=600, error_at=800, compact_buffer=50)
        telem = BudgetTelemetry(budget)
        conv = Conversation()
        # Seed with enough content to register a measurable estimate.
        conv.append(Message(role="user", content="x" * 2000))
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']
                ),
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
                task={},
                conversation=conv,
                hooks=[telem],
            )
        )
        # Without the fix, estimate would be 1 (session_log empty).
        # With conversation access: ~500 tokens from 2000 chars.
        assert telem.samples, "expected at least one sample"
        assert telem.samples[0][2] > 100, (
            f"expected >100 tokens from seeded conv, got {telem.samples[0][2]}"
        )


class TestMCPStartupCleanup:
    def test_bad_command_does_not_leak_subprocess(self, tmp_path):
        from looplet.mcp import MCPToolAdapter

        # Spawn a shell that exits immediately — init will fail since
        # the server never replies. The adapter must clean up the proc.
        adapter = MCPToolAdapter("true", timeout=1.0)
        with pytest.raises(RuntimeError, match="failed to initialize"):
            adapter._ensure_started()
        # After the failed start, _proc is None (was cleaned up).
        assert adapter._proc is None
