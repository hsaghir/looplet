"""step_context: per-step ephemeral hook-to-hook communication."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    ToolSpec,
    composable_loop,
    register_done_tool,
)
from looplet.testing import MockLLMBackend
from looplet.types import ToolCall, ToolResult

pytestmark = pytest.mark.smoke


class TestStepContextLifecycle:
    def test_step_context_cleared_each_step(self):
        """step_context is reset to {} at the start of every step."""
        contexts_seen: list[dict] = []

        class SpyHook:
            def pre_prompt(self, state, session_log, context, step_num):
                # Record what step_context looks like at the start of each step
                contexts_seen.append(dict(getattr(state, "step_context", {})))
                return None

            def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                # Write something so it would leak if not cleared
                state.step_context["written_at"] = step_num
                return None

        mock = MockLLMBackend(
            responses=[
                '{"tool": "ping", "args": {}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec(name="ping", description="p", parameters={}, execute=lambda: {}))

        state = DefaultState(max_steps=5)
        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=state,
                config=LoopConfig(max_steps=5),
                hooks=[SpyHook()],
                task={},
            )
        )

        # Step 1: pre_prompt sees empty (just cleared)
        assert contexts_seen[0] == {}
        # Step 2: pre_prompt sees empty (cleared again, not leaked from step 1)
        assert contexts_seen[1] == {}

    def test_hooks_can_share_data_within_step(self):
        """One hook writes to step_context, another reads it in the same step."""
        read_values: list = []

        class WriterHook:
            def pre_prompt(self, state, session_log, context, step_num):
                state.step_context["from_writer"] = f"step_{step_num}"
                return None

        class ReaderHook:
            def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                read_values.append(state.step_context.get("from_writer"))
                return None

        mock = MockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=5)

        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=state,
                config=LoopConfig(max_steps=5),
                hooks=[WriterHook(), ReaderHook()],
                task={},
            )
        )

        assert read_values[0] == "step_1"


class TestDefaultStateStepContext:
    def test_default_state_has_step_context(self):
        state = DefaultState(max_steps=5)
        assert hasattr(state, "step_context")
        assert state.step_context == {}

    def test_step_context_not_in_snapshot(self):
        """step_context is ephemeral - should NOT leak into snapshot."""
        state = DefaultState(max_steps=5)
        state.step_context["temp"] = "data"
        snap = state.snapshot()
        assert "temp" not in snap
        assert "step_context" not in snap
