"""async_composable_loop — async agent loop tests."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    ToolSpec,
    register_done_tool,
)
from looplet.async_loop import async_composable_loop, async_llm_call
from looplet.testing import AsyncMockLLMBackend
from looplet.types import ToolContext

pytestmark = [pytest.mark.smoke, pytest.mark.asyncio]


class TestAsyncLlmCall:
    async def test_awaits_async_backend(self):
        mock = AsyncMockLLMBackend(responses=["hello"])
        result = await async_llm_call(mock, "test")
        assert result.ok
        assert result.text == "hello"
        assert mock.calls == 1

    async def test_retries_on_error(self):
        call_count = 0

        class FailOnceMock:
            calls = 0

            async def generate(self, prompt, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("transient")
                return "recovered"

        mock = FailOnceMock()
        result = await async_llm_call(mock, "test", max_retries=2)
        assert result.ok
        assert result.text == "recovered"
        assert mock.calls == 2


class TestAsyncComposableLoop:
    async def test_basic_loop(self):
        mock = AsyncMockLLMBackend(
            responses=[
                '{"tool": "greet", "args": {"name": "Alice"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="greet",
                description="Greet",
                parameters={"name": "str"},
                execute=lambda *, name: {"greeting": f"Hi {name}!"},
            )
        )

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5),
            task={"goal": "greet Alice"},
        ):
            steps.append(step)

        assert len(steps) == 2
        assert steps[0].tool_call.tool == "greet"
        assert steps[1].tool_call.tool == "done"
        assert mock.calls == 2

    async def test_ctx_available_in_async_loop(self):
        """Tools should receive ctx with llm in async loop."""
        received_ctx = []

        def my_tool(*, x: str, ctx: ToolContext) -> dict:
            received_ctx.append(ctx)
            return {"x": x}

        mock = AsyncMockLLMBackend(
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

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5),
            task={},
        ):
            pass

        assert len(received_ctx) == 1
        assert received_ctx[0] is not None
        assert received_ctx[0].llm is not None

    async def test_hooks_fire_in_async_loop(self):
        """Sync hooks should still fire in the async loop."""
        pre_prompts = []

        class SpyHook:
            def pre_prompt(self, state, session_log, context, step_num):
                pre_prompts.append(step_num)
                return None

            def should_stop(self, state, step_num, new_entities):
                return False

        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5),
            hooks=[SpyHook()],
            task={},
        ):
            pass

        assert len(pre_prompts) >= 1

    async def test_step_context_cleared_per_step(self):
        """step_context should be cleared at each step in async loop."""
        ctx_values = []

        class CtxHook:
            def pre_prompt(self, state, session_log, context, step_num):
                ctx_values.append(dict(getattr(state, "step_context", {})))
                state.step_context["set_by_hook"] = step_num
                return None

            def should_stop(self, state, step_num, new_entities):
                return False

        mock = AsyncMockLLMBackend(
            responses=[
                '{"tool": "ping", "args": {}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec(name="ping", description="p", parameters={}, execute=lambda: {}))

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5),
            hooks=[CtxHook()],
            task={},
        ):
            pass

        # Both steps should start with empty step_context
        assert ctx_values[0] == {}
        assert ctx_values[1] == {}
