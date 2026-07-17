"""ToolContext always created + metadata from state."""

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
from looplet.types import ToolContext

pytestmark = pytest.mark.smoke


class TestToolContextAlwaysCreated:
    def test_ctx_is_never_none(self):
        """Tools that accept ctx= should always receive a ToolContext, never None."""
        received_ctx = []

        def my_tool(*, value: str, ctx: ToolContext) -> dict:
            received_ctx.append(ctx)
            return {"value": value}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "my_tool", "args": {"value": "x"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="my_tool", description="t", parameters={"value": "str"}, execute=my_tool)
        )

        # No cancel_token, no approval_handler, no hooks - previously ctx was None
        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        assert len(received_ctx) == 1
        assert received_ctx[0] is not None
        assert isinstance(received_ctx[0], ToolContext)

    def test_ctx_has_llm(self):
        """ctx.llm should be populated even without cancel_token/approval."""
        received_llm = []

        def my_tool(*, x: str, ctx: ToolContext) -> dict:
            received_llm.append(ctx.llm)
            return {}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "t", "args": {"x": "1"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="t", description="t", parameters={"x": "str"}, execute=my_tool)
        )

        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        assert len(received_llm) == 1
        assert received_llm[0] is not None


class TestToolContextMetadataFromState:
    def test_metadata_populated_from_state(self):
        """ctx.metadata should contain state.metadata values."""
        received_meta = []

        def my_tool(*, x: str, ctx: ToolContext) -> dict:
            received_meta.append(dict(ctx.metadata))
            return {}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "t", "args": {"x": "1"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="t", description="t", parameters={"x": "str"}, execute=my_tool)
        )

        state = DefaultState(max_steps=5)
        state.metadata["db_path"] = "/tmp/test.db"
        state.metadata["mode"] = "read-only"

        list(
            composable_loop(
                llm=mock, tools=tools, state=state, config=LoopConfig(max_steps=5), task={}
            )
        )

        assert len(received_meta) == 1
        assert received_meta[0]["db_path"] == "/tmp/test.db"
        assert received_meta[0]["mode"] == "read-only"

    def test_metadata_is_copy_not_reference(self):
        """Modifying ctx.metadata should not affect state.metadata."""

        def mutating_tool(*, x: str, ctx: ToolContext) -> dict:
            ctx.metadata["injected"] = "bad"
            return {}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "t", "args": {"x": "1"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="t", description="t", parameters={"x": "str"}, execute=mutating_tool)
        )

        state = DefaultState(max_steps=5)
        state.metadata["safe"] = True

        list(
            composable_loop(
                llm=mock, tools=tools, state=state, config=LoopConfig(max_steps=5), task={}
            )
        )

        # Tool's mutation should not leak back to state
        assert "injected" not in state.metadata
        assert state.metadata["safe"] is True
