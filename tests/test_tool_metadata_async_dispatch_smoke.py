"""tool_metadata + async tool dispatch tests."""

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


class TestToolMetadata:
    def test_tool_metadata_flows_to_ctx(self):
        received = []

        def t(*, x: str, ctx: ToolContext) -> dict:
            received.append(dict(ctx.metadata))
            return {}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "t", "args": {"x": "1"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec(name="t", description="t", parameters={"x": "str"}, execute=t))

        config = LoopConfig(
            max_steps=5,
            tool_metadata={"db_path": "/data/prod.db", "read_only": True},
        )
        list(
            composable_loop(
                llm=mock, tools=tools, state=DefaultState(max_steps=5), config=config, task={}
            )
        )

        assert received[0]["db_path"] == "/data/prod.db"
        assert received[0]["read_only"] is True

    def test_state_metadata_overrides_tool_metadata(self):
        received = []

        def t(*, x: str, ctx: ToolContext) -> dict:
            received.append(dict(ctx.metadata))
            return {}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "t", "args": {"x": "1"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec(name="t", description="t", parameters={"x": "str"}, execute=t))

        config = LoopConfig(
            max_steps=5,
            tool_metadata={"db_path": "/default", "shared": "from_config"},
        )
        state = DefaultState(max_steps=5)
        state.metadata["db_path"] = "/override"  # should win
        state.metadata["unique"] = "from_state"

        list(composable_loop(llm=mock, tools=tools, state=state, config=config, task={}))

        assert received[0]["db_path"] == "/override"  # state wins
        assert received[0]["shared"] == "from_config"  # config preserved
        assert received[0]["unique"] == "from_state"  # state-only key

    def test_empty_tool_metadata_ok(self):
        """Default empty tool_metadata doesn't break anything."""
        received = []

        def t(*, x: str, ctx: ToolContext) -> dict:
            received.append(dict(ctx.metadata))
            return {}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "t", "args": {"x": "1"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec(name="t", description="t", parameters={"x": "str"}, execute=t))

        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        assert received[0] == {}  # no metadata from either source


class TestAsyncToolDispatch:
    def test_async_tool_in_sync_loop(self):
        """Async tool functions should work in the sync loop."""

        async def async_tool(*, query: str) -> dict:
            return {"result": f"async result for {query}"}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "search", "args": {"query": "test"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="search", description="s", parameters={"query": "str"}, execute=async_tool
            )
        )

        steps = list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        assert len(steps) == 2
        assert steps[0].tool_result.data == {"result": "async result for test"}
        assert steps[0].tool_result.error is None
