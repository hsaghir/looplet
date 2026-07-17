"""generate_kwargs passthrough + async hook support tests."""

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

pytestmark = pytest.mark.smoke


class TestGenerateKwargs:
    def test_generate_kwargs_passed_to_backend(self):
        """Extra kwargs should be forwarded when the backend accepts them."""
        received_kwargs: list[dict] = []

        class KwargsCapturingBackend:
            calls = 0

            def generate(
                self,
                prompt,
                *,
                max_tokens=2000,
                system_prompt="",
                temperature=0.2,
                top_p=None,
                custom_param=None,
            ):
                self.calls += 1
                received_kwargs.append({"top_p": top_p, "custom_param": custom_param})
                if self.calls == 1:
                    return '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}'
                return '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}'

        tools = BaseToolRegistry()
        register_done_tool(tools)

        config = LoopConfig(
            max_steps=3,
            generate_kwargs={"top_p": 0.9, "custom_param": "hello"},
        )

        list(
            composable_loop(
                llm=KwargsCapturingBackend(),
                tools=tools,
                state=DefaultState(max_steps=3),
                config=config,
                task={},
            )
        )

        assert len(received_kwargs) >= 1
        assert received_kwargs[0]["top_p"] == 0.9
        assert received_kwargs[0]["custom_param"] == "hello"

    def test_generate_kwargs_overrides_positional_params(self):
        """generate_kwargs should be able to override temperature, max_tokens."""
        received: list[dict] = []

        class CapturingBackend:
            calls = 0

            def generate(
                self,
                prompt,
                *,
                max_tokens=2000,
                system_prompt="",
                temperature=0.2,
            ):
                self.calls += 1
                received.append({"max_tokens": max_tokens, "temperature": temperature})
                return '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}'

        tools = BaseToolRegistry()
        register_done_tool(tools)

        # LoopConfig sets temperature=0.3 and max_tokens=500,
        # but generate_kwargs overrides both
        config = LoopConfig(
            max_steps=3,
            temperature=0.3,
            max_tokens=500,
            generate_kwargs={"temperature": 0.0, "max_tokens": 8000},
        )

        list(
            composable_loop(
                llm=CapturingBackend(),
                tools=tools,
                state=DefaultState(max_steps=3),
                config=config,
                task={},
            )
        )

        assert len(received) >= 1
        assert received[0]["temperature"] == 0.0  # overridden
        assert received[0]["max_tokens"] == 8000  # overridden

    def test_unknown_kwargs_silently_skipped(self):
        """Keys the backend doesn't accept should be silently dropped."""
        mock = MockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        # MockLLMBackend.generate doesn't accept 'top_p' - should not crash
        config = LoopConfig(
            max_steps=3,
            generate_kwargs={"top_p": 0.9, "nonexistent_kwarg": True},
        )

        steps = list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=3),
                config=config,
                task={},
            )
        )
        assert len(steps) == 1  # done step

    def test_empty_generate_kwargs_ok(self):
        """Default empty generate_kwargs doesn't break anything."""
        mock = MockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        steps = list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
                task={},
            )
        )
        assert len(steps) == 1

    def test_kwargs_forwarded_to_backend_with_var_keyword(self):
        """Regression: backends written as ``def generate(self, prompt,
        **kw)`` should receive forwarded ``generate_kwargs``. Previously
        ``_accepts_kwarg`` only checked named parameters and silently
        dropped every kwarg on permissive backend signatures, so
        ``LoopConfig.generate_kwargs`` no-opped on a common pattern.
        """

        class CapturingLLM:
            def __init__(self, response):
                self.response = response
                self.captured: list[dict] = []

            def generate(self, prompt, **kw):
                self.captured.append(dict(kw))
                return self.response

        llm = CapturingLLM('{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}')
        tools = BaseToolRegistry()
        register_done_tool(tools)
        config = LoopConfig(
            max_steps=2,
            generate_kwargs={"top_p": 0.9, "response_format": {"type": "json"}},
        )

        list(
            composable_loop(
                llm=llm,
                tools=tools,
                state=DefaultState(max_steps=2),
                config=config,
                task={},
            )
        )
        assert llm.captured[0]["top_p"] == 0.9
        assert llm.captured[0]["response_format"] == {"type": "json"}


@pytest.mark.asyncio
class TestAsyncHooks:
    async def test_async_on_loop_end(self):
        """Async on_loop_end hooks should be awaited."""
        from looplet.async_loop import async_composable_loop
        from looplet.testing import AsyncMockLLMBackend

        end_called = []

        class AsyncEndHook:
            async def on_loop_end(self, state, session_log, context, llm):
                end_called.append(True)
                return 0

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
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
            hooks=[AsyncEndHook()],
            task={},
        ):
            pass

        assert end_called == [True]

    async def test_async_pre_loop(self):
        """Async pre_loop hooks should be awaited."""
        from looplet.async_loop import async_composable_loop
        from looplet.testing import AsyncMockLLMBackend

        pre_called = []

        class AsyncPreHook:
            async def pre_loop(self, state, session_log, context):
                pre_called.append(True)

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
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
            hooks=[AsyncPreHook()],
            task={},
        ):
            pass

        assert pre_called == [True]
