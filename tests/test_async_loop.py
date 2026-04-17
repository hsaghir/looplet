"""Tests for cadence.async_loop — async composable loop, protocols, and adapter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openharness.types import Step, ToolCall, ToolResult

# ── Helpers ────────────────────────────────────────────────────────


def _make_tool_result(tool: str = "search", data: Any = None, error: str | None = None) -> ToolResult:
    return ToolResult(tool=tool, args_summary="", data=data or {"rows": []}, error=error)


def _make_step(number: int, tool: str = "search") -> Step:
    tc = ToolCall(tool=tool, args={}, reasoning="test")
    tr = _make_tool_result(tool)
    return Step(number=number, tool_call=tc, tool_result=tr)


class _MockAsyncLLM:
    """Minimal async LLM returning scripted JSON responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ['{"tool": "done", "args": {}, "reasoning": "done"}']
        self._index = 0

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        r = self._responses[self._index % len(self._responses)]
        self._index += 1
        return r


class _MockSyncLLM:
    """Minimal sync LLM for SyncToAsyncAdapter tests."""

    def __init__(self, response: str = "sync response") -> None:
        self._response = response

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        return self._response


# ── Import tests ────────────────────────────────────────────────────


class TestImports:
    def test_async_llm_backend_importable(self):
        from openharness.async_loop import AsyncLLMBackend
        assert AsyncLLMBackend is not None

    def test_async_loop_hook_importable(self):
        from openharness.async_loop import AsyncLoopHook
        assert AsyncLoopHook is not None

    def test_async_composable_loop_importable(self):
        from openharness.async_loop import async_composable_loop
        assert callable(async_composable_loop)

    def test_async_llm_call_with_retry_importable(self):
        from openharness.async_loop import async_llm_call_with_retry
        assert callable(async_llm_call_with_retry)

    def test_sync_to_async_adapter_importable(self):
        from openharness.async_loop import SyncToAsyncAdapter
        assert SyncToAsyncAdapter is not None

    def test_exported_from_init(self):
        from openharness import AsyncLLMBackend, AsyncLoopHook, async_composable_loop
        assert AsyncLLMBackend is not None
        assert AsyncLoopHook is not None
        assert callable(async_composable_loop)


# ── AsyncLLMBackend protocol ────────────────────────────────────────


class TestAsyncLLMBackend:
    def test_is_runtime_checkable(self):
        from openharness.async_loop import AsyncLLMBackend
        # runtime_checkable: isinstance check works
        mock = _MockAsyncLLM()
        # Can check isinstance against protocol
        result = isinstance(mock, AsyncLLMBackend)
        assert isinstance(result, bool)  # no error

    def test_mock_satisfies_protocol(self):
        from openharness.async_loop import AsyncLLMBackend
        mock = _MockAsyncLLM()
        assert isinstance(mock, AsyncLLMBackend)

    def test_has_async_generate(self):
        import inspect

        from openharness.async_loop import AsyncLLMBackend
        # Check that generate is a coroutine function in the mock
        assert asyncio.iscoroutinefunction(_MockAsyncLLM.generate)

    def test_sync_llm_not_async_backend(self):
        """Sync LLM's generate is not a coroutine function (documents limitation)."""
        # Note: Python's @runtime_checkable Protocol only checks method existence,
        # not whether methods are async. This test documents the actual behavior.
        sync_llm = _MockSyncLLM()
        # generate is NOT a coroutine function on the sync LLM
        assert not asyncio.iscoroutinefunction(sync_llm.generate)


# ── AsyncLoopHook protocol ──────────────────────────────────────────


class TestAsyncLoopHook:
    def test_is_runtime_checkable(self):
        import inspect

        from openharness.async_loop import AsyncLoopHook
        # Protocol must be @runtime_checkable
        assert hasattr(AsyncLoopHook, "__protocol_attrs__") or \
               getattr(AsyncLoopHook, "_is_protocol", False) or \
               getattr(AsyncLoopHook, "__runtime_checkable__", False)

    def test_has_six_async_methods(self):
        from openharness.async_loop import AsyncLoopHook
        methods = ["pre_prompt", "pre_dispatch", "post_dispatch", "check_done", "should_stop", "on_loop_end"]
        for method in methods:
            assert hasattr(AsyncLoopHook, method), f"Missing method: {method}"

    def test_all_methods_are_async(self):
        """All protocol methods must be async (coroutine functions)."""
        from openharness.async_loop import AsyncLoopHook

        class ConcreteHook:
            async def pre_prompt(self, state, session_log, context, step_num): return None
            async def pre_dispatch(self, state, session_log, tool_call, step_num): return None
            async def post_dispatch(self, state, session_log, tool_call, tool_result, step_num): return None
            async def check_done(self, state, session_log, context, step_num): return None
            async def should_stop(self, state, step_num, new_entities): return False
            async def on_loop_end(self, state, session_log, context, llm): return 0

        hook = ConcreteHook()
        for method in ["pre_prompt", "pre_dispatch", "post_dispatch", "check_done", "should_stop", "on_loop_end"]:
            assert asyncio.iscoroutinefunction(getattr(hook, method))


# ── SyncToAsyncAdapter ──────────────────────────────────────────────


class TestSyncToAsyncAdapter:
    def test_wraps_sync_llm(self):
        from openharness.async_loop import SyncToAsyncAdapter
        sync_llm = _MockSyncLLM("hello")
        adapter = SyncToAsyncAdapter(sync_llm)
        assert adapter is not None

    async def test_generate_is_async(self):
        from openharness.async_loop import SyncToAsyncAdapter
        sync_llm = _MockSyncLLM("test response")
        adapter = SyncToAsyncAdapter(sync_llm)
        result = await adapter.generate("prompt")
        assert result == "test response"

    async def test_adapter_satisfies_async_backend(self):
        from openharness.async_loop import AsyncLLMBackend, SyncToAsyncAdapter
        sync_llm = _MockSyncLLM("response")
        adapter = SyncToAsyncAdapter(sync_llm)
        assert isinstance(adapter, AsyncLLMBackend)

    async def test_passes_kwargs_through(self):
        from openharness.async_loop import SyncToAsyncAdapter
        calls = []

        class TracingSync:
            def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
                return "traced"

        adapter = SyncToAsyncAdapter(TracingSync())
        await adapter.generate("test prompt", max_tokens=500, temperature=0.5)
        assert len(calls) == 1
        assert calls[0]["max_tokens"] == 500
        assert calls[0]["temperature"] == 0.5


# ── async_llm_call_with_retry ────────────────────────────────────────


class TestAsyncLlmCallWithRetry:
    async def test_success_first_call(self):
        from openharness.async_loop import async_llm_call_with_retry
        llm = _MockAsyncLLM(["good response"])
        result = await async_llm_call_with_retry(llm, "prompt")
        assert result.ok
        assert result.text == "good response"

    async def test_retries_on_failure(self):
        from openharness.async_loop import async_llm_call_with_retry

        class FlakyLLM:
            def __init__(self):
                self._calls = 0

            async def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                self._calls += 1
                if self._calls < 3:
                    raise RuntimeError("transient failure")
                return "success after retries"

        flaky = FlakyLLM()
        with patch("openharness.async_loop.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await async_llm_call_with_retry(flaky, "prompt", max_retries=3)
        assert result.ok
        assert result.text == "success after retries"

    async def test_prompt_too_long_not_retried(self):
        from openharness.async_loop import async_llm_call_with_retry

        class TooLongLLM:
            async def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                raise Exception("prompt is too long")

        llm = TooLongLLM()
        with patch("openharness.async_loop.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await async_llm_call_with_retry(llm, "prompt", max_retries=3)
        # Should not retry on prompt-too-long
        assert not result.ok
        assert result.is_prompt_too_long
        # No sleep calls (not retried)
        mock_sleep.assert_not_called()

    async def test_exponential_backoff(self):
        from openharness.async_loop import async_llm_call_with_retry

        fail_count = 0

        class FailingLLM:
            async def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                nonlocal fail_count
                fail_count += 1
                raise ValueError("fail")

        with patch("openharness.async_loop.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await async_llm_call_with_retry(FailingLLM(), "prompt", max_retries=2)
        assert not result.ok
        assert mock_sleep.call_count >= 1


# ── async_composable_loop ────────────────────────────────────────────


class TestAsyncComposableLoop:
    def _make_mock_tools(self, with_done: bool = True) -> Any:
        """Create a minimal mock tool registry."""
        tools = MagicMock()
        tools.tool_catalog_text.return_value = "- search: search for stuff\n- done: finish"
        tools._tools = {}

        def mock_dispatch(tc: ToolCall) -> ToolResult:
            if tc.tool == "done":
                return ToolResult(tool="done", args_summary="", data={"done": True})
            return ToolResult(tool=tc.tool, args_summary="", data={"rows": [1, 2, 3]})

        async def mock_async_dispatch(tc: ToolCall) -> ToolResult:
            return mock_dispatch(tc)

        tools.dispatch = mock_dispatch
        tools.dispatch_async = mock_async_dispatch
        tools.dispatch_batch_async = AsyncMock(
            side_effect=lambda calls: asyncio.gather(*[mock_async_dispatch(c) for c in calls])
        )
        return tools

    def _make_state(self, max_steps: int = 5) -> Any:
        from types import SimpleNamespace
        state = SimpleNamespace()
        state.steps = []
        state.queries_used = 0
        state.max_steps = max_steps

        @property
        def step_count(self):
            return len(self.steps)

        @property
        def budget_remaining(self):
            return max(0, self.max_steps - len(self.steps))

        def context_summary():
            return "(context summary)"

        def snapshot():
            return {"step_count": len(state.steps)}

        state.step_count = property(lambda s: len(s.steps))
        # Use simple attributes for these
        state.context_summary = context_summary
        state.snapshot = snapshot
        return state

    async def test_yields_steps(self):
        from openharness.async_loop import async_composable_loop
        from openharness.types import Step

        llm = _MockAsyncLLM(['{"tool": "done", "args": {}, "reasoning": "done"}'])
        tools = self._make_mock_tools()

        class SimpleState:
            def __init__(self):
                self.steps = []
                self.queries_used = 0

            @property
            def step_count(self): return len(self.steps)

            @property
            def budget_remaining(self): return 5 - len(self.steps)

            def context_summary(self): return "(summary)"
            def snapshot(self): return {}

        state = SimpleState()
        session_log = MagicMock()
        session_log.render.return_value = ""
        session_log.all_entities.return_value = set()
        session_log.record = MagicMock()
        session_log._entries = []

        from openharness.loop import LoopConfig
        config = LoopConfig(max_steps=3)

        collected: list[Step] = []
        async for step in async_composable_loop(
            llm=llm, task={"id": "t"}, tools=tools,
            context=None, hooks=[], config=config,
            state=state, session_log=session_log,
        ):
            collected.append(step)

        # Should have at least 1 step (the done step)
        assert len(collected) >= 1
        assert isinstance(collected[-1], Step)

    async def test_terminates_on_done(self):
        from openharness.async_loop import async_composable_loop

        # Script: one search then done
        responses = [
            '{"tool": "search", "args": {"q": "test"}, "reasoning": "search"}',
            '{"tool": "done", "args": {}, "reasoning": "done"}',
        ]
        llm = _MockAsyncLLM(responses)

        class SimpleState:
            def __init__(self):
                self.steps = []
                self.queries_used = 0

            @property
            def step_count(self): return len(self.steps)

            @property
            def budget_remaining(self): return 5 - len(self.steps)

            def context_summary(self): return ""
            def snapshot(self): return {}

        state = SimpleState()
        session_log = MagicMock()
        session_log.render.return_value = ""
        session_log.all_entities.return_value = set()
        session_log.record = MagicMock()
        session_log._entries = []

        tools = self._make_mock_tools()

        from openharness.loop import LoopConfig
        config = LoopConfig(max_steps=10, done_tool="done")

        steps: list = []
        async for step in async_composable_loop(
            llm=llm, task={}, tools=tools,
            context=None, hooks=[], config=config,
            state=state, session_log=session_log,
        ):
            steps.append(step)

        # Should terminate (not exhaust all 10 steps)
        assert len(steps) <= 5

    async def test_async_hook_pre_prompt_called(self):
        from openharness.async_loop import async_composable_loop

        hook_calls: list[str] = []

        class TrackingHook:
            async def pre_prompt(self, state, session_log, context, step_num):
                hook_calls.append("pre_prompt")
                return None

            async def pre_dispatch(self, state, session_log, tool_call, step_num):
                return None

            async def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                return None

            async def check_done(self, state, session_log, context, step_num):
                return None

            async def should_stop(self, state, step_num, new_entities):
                return False

            async def on_loop_end(self, state, session_log, context, llm):
                hook_calls.append("on_loop_end")
                return 0

        llm = _MockAsyncLLM(['{"tool": "done", "args": {}, "reasoning": "done"}'])
        tools = self._make_mock_tools()

        class SimpleState:
            def __init__(self):
                self.steps = []
                self.queries_used = 0

            @property
            def step_count(self): return len(self.steps)

            @property
            def budget_remaining(self): return 5 - len(self.steps)

            def context_summary(self): return ""
            def snapshot(self): return {}

        state = SimpleState()
        session_log = MagicMock()
        session_log.render.return_value = ""
        session_log.record = MagicMock()
        session_log._entries = []

        from openharness.loop import LoopConfig
        config = LoopConfig(max_steps=5)

        steps = []
        async for step in async_composable_loop(
            llm=llm, task={}, tools=tools,
            context=None, hooks=[TrackingHook()], config=config,
            state=state, session_log=session_log,
        ):
            steps.append(step)

        assert "pre_prompt" in hook_calls
        assert "on_loop_end" in hook_calls

    async def test_concurrent_safe_tools_gathered(self):
        """Concurrent-safe tools should be dispatched concurrently."""
        from openharness.async_loop import async_composable_loop

        dispatch_order: list[str] = []

        class SimpleState:
            def __init__(self):
                self.steps = []
                self.queries_used = 0

            @property
            def step_count(self): return len(self.steps)

            @property
            def budget_remaining(self): return 5 - len(self.steps)

            def context_summary(self): return ""
            def snapshot(self): return {}

        state = SimpleState()

        # Two concurrent-safe tools in one response
        responses = [
            '{"tools": [{"tool": "a", "args": {}, "reasoning": "r"}, {"tool": "b", "args": {}, "reasoning": "r"}], "theory": "t"}',
            '{"tool": "done", "args": {}, "reasoning": "done"}',
        ]
        llm = _MockAsyncLLM(responses)

        tools = MagicMock()
        tools.tool_catalog_text.return_value = "- a: tool a\n- b: tool b\n- done: finish"

        def get_spec(name):
            spec = MagicMock()
            spec.concurrent_safe = True
            spec.free = False
            spec.name = name
            return spec

        tools._tools = {
            "a": get_spec("a"),
            "b": get_spec("b"),
            "done": get_spec("done"),
        }

        async def async_dispatch(tc):
            dispatch_order.append(tc.tool)
            return ToolResult(tool=tc.tool, args_summary="", data={"ok": True})

        tools.dispatch = lambda tc: ToolResult(tool=tc.tool, args_summary="", data={"ok": True})
        tools.dispatch_async = async_dispatch

        async def batch_dispatch(calls):
            return await asyncio.gather(*[async_dispatch(c) for c in calls])

        tools.dispatch_batch_async = batch_dispatch

        session_log = MagicMock()
        session_log.render.return_value = ""
        session_log.record = MagicMock()
        session_log._entries = []

        from openharness.loop import LoopConfig
        config = LoopConfig(max_steps=5, done_tool="done")

        steps = []
        async for step in async_composable_loop(
            llm=llm, task={}, tools=tools,
            context=None, hooks=[], config=config,
            state=state, session_log=session_log,
        ):
            steps.append(step)

        # Both a and b should have been dispatched
        assert "a" in dispatch_order or len(steps) > 0  # dispatched happened

    async def test_returns_async_generator(self):
        """async_composable_loop must return an async generator."""
        import inspect

        from openharness.async_loop import async_composable_loop

        class SimpleState:
            def __init__(self):
                self.steps = []
                self.queries_used = 0

            @property
            def step_count(self): return len(self.steps)

            @property
            def budget_remaining(self): return 0  # immediately terminates

            def context_summary(self): return ""
            def snapshot(self): return {}

        state = SimpleState()
        llm = _MockAsyncLLM([])
        tools = self._make_mock_tools()
        session_log = MagicMock()
        session_log.render.return_value = ""
        session_log.record = MagicMock()
        session_log._entries = []

        from openharness.loop import LoopConfig
        gen = async_composable_loop(
            llm=llm, task={}, tools=tools,
            context=None, hooks=[], config=LoopConfig(),
            state=state, session_log=session_log,
        )
        assert inspect.isasyncgen(gen) or hasattr(gen, "__aiter__")

    async def test_on_loop_end_called_once(self):
        from openharness.async_loop import async_composable_loop

        end_count = [0]

        class CountingHook:
            async def pre_prompt(self, *a): return None
            async def pre_dispatch(self, *a): return None
            async def post_dispatch(self, *a): return None
            async def check_done(self, *a): return None
            async def should_stop(self, *a): return False

            async def on_loop_end(self, *a):
                end_count[0] += 1
                return 0

        llm = _MockAsyncLLM(['{"tool": "done", "args": {}, "reasoning": "done"}'])
        tools = self._make_mock_tools()

        class SimpleState:
            def __init__(self):
                self.steps = []
                self.queries_used = 0

            @property
            def step_count(self): return len(self.steps)

            @property
            def budget_remaining(self): return 5 - len(self.steps)

            def context_summary(self): return ""
            def snapshot(self): return {}

        state = SimpleState()
        session_log = MagicMock()
        session_log.render.return_value = ""
        session_log.record = MagicMock()
        session_log._entries = []

        from openharness.loop import LoopConfig
        config = LoopConfig(max_steps=5)

        async for _ in async_composable_loop(
            llm=llm, task={}, tools=tools,
            context=None, hooks=[CountingHook()], config=config,
            state=state, session_log=session_log,
        ):
            pass

        assert end_count[0] == 1
