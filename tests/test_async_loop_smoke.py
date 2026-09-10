"""async_composable_loop - async agent loop tests."""

from __future__ import annotations

import asyncio

import pytest

from looplet import (
    BaseToolRegistry,
    ContextProjection,
    DefaultState,
    InjectContext,
    LoopConfig,
    ToolSpec,
    register_done_tool,
)
from looplet.async_loop import async_composable_loop, async_llm_call
from looplet.cache import CacheControl, CachePolicy
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

    async def test_native_failure_falls_back_to_regular_generation(self):
        calls = []

        class NativeFailureBackend:
            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                calls.append("native")
                raise RuntimeError("tools endpoint unsupported")

            async def generate(self, prompt, **kwargs):
                calls.append("regular")
                return '{"tool": "done", "args": {}}'

        result = await async_llm_call(
            NativeFailureBackend(),
            "finish",
            tools=[{"name": "done"}],
            max_retries=0,
        )

        assert result.ok
        assert result.text == '{"tool": "done", "args": {}}'
        assert calls == ["native", "regular"]

    async def test_native_fallback_stats_are_exposed_on_state(self):
        class TrackingBackend:
            def __init__(self):
                self.calls = []

            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                self.calls.append("native")
                raise RuntimeError("unsupported native endpoint")

            async def generate(self, prompt, **kwargs):
                self.calls.append("regular")
                return '{"tool": "done", "args": {}}'

        from looplet import BaseToolRegistry, DefaultState, LoopConfig, register_done_tool

        backend = TrackingBackend()
        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=2)
        async for _ in async_composable_loop(
            llm=backend,
            tools=tools,
            state=state,
            config=LoopConfig(max_steps=2),
            task={},
        ):
            pass

        assert state.metadata["native_tool_stats"]["fallbacks"] == 1


class TestAsyncComposableLoop:
    async def test_async_hook_slots_are_awaited_and_effects_apply(self):
        seen: list[str] = []

        class AsyncPolicy:
            async def pre_prompt(self, state, session_log, context, step_num):
                seen.append(f"pre:{step_num}")
                return "async briefing"

            async def pre_dispatch(self, state, session_log, tool_call, step_num):
                seen.append(f"pre_dispatch:{tool_call.tool}")
                if tool_call.tool == "ping":
                    return InjectContext("ping inspected")
                return None

            async def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                seen.append(f"post:{tool_call.tool}")
                return InjectContext("post inspected")

            async def check_done(self, state, session_log, context, step_num, tool_call=None):
                seen.append(f"done:{tool_call.tool if tool_call else 'none'}")
                return None

            async def should_stop(self, state, step_num, new_entities):
                seen.append(f"stop:{step_num}")
                return False

            async def on_event(self, payload):
                if payload.event.value == "session_start":
                    seen.append("event:session_start")

        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec("ping", "Ping", {}, lambda: {"pong": True}))
        backend = AsyncMockLLMBackend(
            responses=[
                '{"tool":"ping","args":{}}',
                '{"tool":"done","args":{"summary":"ok"}}',
            ]
        )

        async for _ in async_composable_loop(
            llm=backend,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
            hooks=[AsyncPolicy()],
            task={},
        ):
            pass

        assert "event:session_start" in seen
        assert "pre:1" in seen
        assert "pre_dispatch:ping" in seen
        assert "post:ping" in seen
        assert "done:done" in seen

    async def test_late_llm_deadline_does_not_dispatch_response(self):
        import time

        class SlowBackend:
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.02)
                return '{"tool":"done","args":{"summary":"late"}}'

        from looplet import RunEnvelope

        dispatched = []
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                "side_effect",
                "Side effect",
                {},
                lambda: dispatched.append(True) or {"ok": True},
            )
        )
        state = DefaultState(max_steps=2)

        async for _ in async_composable_loop(
            llm=SlowBackend(),
            tools=tools,
            state=state,
            config=LoopConfig(
                max_steps=2,
                run_envelope=RunEnvelope(
                    run_id="late-async",
                    deadline_at=time.time() + 0.005,
                ),
            ),
            task={},
        ):
            pass

        assert dispatched == []
        assert state.termination_reason == "deadline_exceeded"

    async def test_prompt_exception_restores_context_override_and_terminal_state(self):
        from looplet.context_budget import get_context_window_steps

        class BrokenRenderer:
            def __call__(self, *, projection):
                raise RuntimeError("renderer failed")

        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=2)
        before = get_context_window_steps()

        with pytest.raises(RuntimeError, match="renderer failed"):
            async for _ in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=[]),
                tools=tools,
                state=state,
                config=LoopConfig(
                    max_steps=2,
                    context_window_steps=77,
                    render_messages_override=BrokenRenderer(),
                ),
                task={},
            ):
                pass

        assert get_context_window_steps() == before
        assert state.run_status == "failed"
        assert state.run_phase == "terminal"

    async def test_router_selected_llm_reaches_async_tool_context(self):
        from looplet.router import ModelProfile, SimpleRouter

        selected = AsyncMockLLMBackend(
            responses=[
                '{"tool":"inspect","args":{}}',
                '{"tool":"done","args":{"summary":"ok"}}',
            ]
        )
        original = AsyncMockLLMBackend(responses=[])
        seen = []

        def inspect_llm(*, ctx):
            seen.append(ctx.llm)
            return {"ok": True}

        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(ToolSpec("inspect", "Inspect", {}, inspect_llm))
        profile = ModelProfile("selected", selected)
        async for _ in async_composable_loop(
            llm=original,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(
                max_steps=3,
                router=SimpleRouter({"reasoning": profile}, default_profile=profile),
                use_native_tools=False,
            ),
            task={},
        ):
            pass

        assert seen
        assert getattr(seen[0], "_backend", None) is selected

    async def test_async_compaction_events_are_awaited(self):
        from looplet import DefaultCompactService, LifecycleEvent

        events = []

        class CompactVote:
            async def should_compact(self, state, session_log, conversation, step_num):
                return step_num == 1

            async def on_event(self, payload):
                if payload.event in (LifecycleEvent.PRE_COMPACT, LifecycleEvent.POST_COMPACT):
                    events.append(payload.event)

        tools = BaseToolRegistry()
        register_done_tool(tools)
        async for _ in async_composable_loop(
            llm=AsyncMockLLMBackend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
            tools=tools,
            state=DefaultState(max_steps=2),
            config=LoopConfig(max_steps=2, compact_service=DefaultCompactService()),
            hooks=[CompactVote()],
            task={},
        ):
            pass

        assert events == [LifecycleEvent.PRE_COMPACT, LifecycleEvent.POST_COMPACT]

    async def test_context_projection_override_is_shared_with_sync_loop(self):
        projections: list[ContextProjection] = []

        def render(*, projection):
            projections.append(projection)
            return projection.default_prompt

        tools = BaseToolRegistry()
        register_done_tool(tools)
        async for _ in async_composable_loop(
            llm=AsyncMockLLMBackend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
            tools=tools,
            state=DefaultState(max_steps=2),
            config=LoopConfig(max_steps=2, render_messages_override=render),
            task={},
        ):
            pass

        assert projections
        assert projections[0].messages == ()
        assert projections[0].step_num == 1

    async def test_expired_deadline_has_cancelled_terminal_state(self):
        import time

        from looplet import RunEnvelope

        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=2)
        async for _ in async_composable_loop(
            llm=AsyncMockLLMBackend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
            tools=tools,
            state=state,
            config=LoopConfig(
                max_steps=2,
                run_envelope=RunEnvelope(run_id="expired", deadline_at=time.time() - 1),
            ),
            task={},
        ):
            pass

        assert state.run_status == "cancelled"
        assert state.run_phase == "terminal"
        assert state.termination_reason == "deadline_exceeded"

    async def test_llm_exhaustion_is_failed_terminal_state(self):
        class FailingBackend:
            async def generate(self, prompt, **kwargs):
                raise RuntimeError("provider unavailable")

        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=2)

        async for _ in async_composable_loop(
            llm=FailingBackend(),
            tools=tools,
            state=state,
            config=LoopConfig(max_steps=2),
            task={},
        ):
            pass

        assert state.run_status == "failed"
        assert state.run_phase == "terminal"
        assert state.termination_reason == "llm_error"

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

    async def test_max_steps_and_system_prompt_shorthand(self):
        """Regression: ``async_composable_loop`` accepts the same
        ``max_steps`` / ``system_prompt`` keyword shorthands as
        ``composable_loop`` so callers don't need to construct a
        ``LoopConfig`` for one-liner agents."""
        mock = AsyncMockLLMBackend(
            responses=[
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            task={"goal": "x"},
            max_steps=3,
            system_prompt="be brief",
        ):
            steps.append(step)

        assert len(steps) == 1
        assert steps[0].tool_call.tool == "done"

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

    async def test_async_loop_context_exposes_registry_resources(self):
        resource = object()
        hook_resources = []
        tool_resources = []

        class ResourceHook:
            def bind(self, ctx):
                hook_resources.append(ctx.resources["shared"])

        def inspect_resource(*, ctx):
            tool_resources.append(ctx.resources["shared"])
            return {"same": True}

        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="inspect_resource",
                description="inspect",
                parameters={},
                execute=inspect_resource,
                requires=["shared"],
            )
        )
        tools.set_resources({"shared": resource})
        mock = AsyncMockLLMBackend(
            responses=[
                '{"tool": "inspect_resource", "args": {}}',
                '{"tool": "done", "args": {"summary": "ok"}}',
            ]
        )

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
            hooks=[ResourceHook()],
            task={},
        ):
            pass

        assert hook_resources == [resource]
        assert tool_resources == [resource]

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

    async def test_initial_checkpoint_restores_step_offset_and_session_log(self):
        from looplet.checkpoint import Checkpoint
        from looplet.session import SessionLog

        checkpoint = Checkpoint(
            step_number=3,
            session_log_data={"entries": [], "current_theory": "resumed theory"},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={},
        )
        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "resumed"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        session_log = SessionLog()

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            session_log=session_log,
            config=LoopConfig(max_steps=5, initial_checkpoint=checkpoint),
            task={},
        ):
            steps.append(step)

        assert steps[0].number == 4
        assert session_log.current_theory == "resumed theory"

    async def test_done_checkpoint_includes_recorded_session_log_entry(self, tmp_path):
        from looplet.checkpoint import FileCheckpointStore

        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3, checkpoint_dir=tmp_path),
            task={},
        ):
            pass

        checkpoint = FileCheckpointStore(tmp_path).load("step_1_done")

        assert checkpoint is not None
        assert checkpoint.metadata["status"] == "done"
        assert len(checkpoint.session_log_data["entries"]) == 1

    async def test_domain_checkpoint_state_round_trips(self, tmp_path):
        from looplet.checkpoint import Checkpoint, FileCheckpointStore
        from looplet.loop import LoopConfig

        restored = []

        def snapshot(ctx):
            return {"phase": ctx.phase.value, "step": ctx.step_num}

        def restore(ctx, data):
            restored.append((ctx.step_num, data))

        checkpoint = FileCheckpointStore(tmp_path)
        checkpoint.save(
            Checkpoint(
                step_number=1,
                session_log_data={"entries": [], "current_theory": ""},
                conversation_data=None,
                config_snapshot={"max_steps": 3},
                tool_results_store={},
                metadata={},
                domain_state={"saved": True},
            ),
            "step_1",
        )
        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(
                max_steps=3,
                checkpoint_dir=tmp_path,
                checkpoint_state=snapshot,
                restore_checkpoint_state=restore,
            ),
            task={},
        ):
            pass

        saved = checkpoint.load("step_2_done")
        assert restored == [(1, {"saved": True})]
        assert saved is not None
        assert saved.domain_state == {"phase": "terminal", "step": 2}

    async def test_context_budget_is_exposed_on_async_event(self):
        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=3)
        budgets = []

        class BudgetObserver:
            def on_event(self, payload):
                if payload.context_budget is not None:
                    budgets.append(payload.context_budget)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=state,
            config=LoopConfig(max_steps=3),
            hooks=[BudgetObserver()],
            task={},
        ):
            pass

        assert budgets
        assert budgets[0]["prompt_chars"] > 0
        assert budgets[0]["estimated_tokens"] > 0
        assert state.metadata.get("context_budget") is None

    async def test_regular_checkpoint_includes_recorded_session_log_entry(self, tmp_path):
        from looplet.checkpoint import FileCheckpointStore

        mock = AsyncMockLLMBackend(
            responses=[
                '{"tool": "ping", "args": {}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="ping", description="ping", parameters={}, execute=lambda: {"pong": True})
        )

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3, checkpoint_dir=tmp_path),
            task={},
        ):
            pass

        checkpoint = FileCheckpointStore(tmp_path).load("step_1")

        assert checkpoint is not None
        assert len(checkpoint.session_log_data["entries"]) == 1

    async def test_async_stop_event_precedes_loop_end_with_stop_reason(self):
        from looplet import LifecycleEvent

        observations = []

        class StopOrderHook:
            def on_event(self, payload):
                if payload.event == LifecycleEvent.STOP:
                    observations.append(
                        (
                            "stop",
                            payload.termination_reason,
                            getattr(payload.state, "_stop_reason", None),
                        )
                    )

            async def on_loop_end(self, state, session_log, context, llm):
                observations.append(("end", getattr(state, "_stop_reason", None)))
                return 0

        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
            hooks=[StopOrderHook()],
            task={},
        ):
            pass

        assert observations == [("stop", "done", "done"), ("end", "done")]

    async def test_cache_policy_threads_breakpoints_into_async_backend(self):
        class CacheAwareAsyncBackend:
            def __init__(self) -> None:
                self.received = []

            async def generate(
                self,
                prompt,
                *,
                max_tokens=2000,
                system_prompt="",
                temperature=0.2,
                cache_breakpoints=None,
            ):
                self.received.append(cache_breakpoints)
                return '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}'

        backend = CacheAwareAsyncBackend()
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=backend,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(
                max_steps=3,
                system_prompt="SYS",
                cache_policy=CachePolicy(system_prompt=CacheControl()),
            ),
            task={},
        ):
            pass

        assert backend.received[0] is not None
        assert backend.received[0][0].label == "system_prompt"

    async def test_recovery_registry_consulted_on_parse_error(self):
        from looplet.recovery import (
            FailureScenario,
            RecoveryAction,
            RecoveryRecipe,
            RecoveryRegistry,
        )

        recovery_called = []

        def handler(ctx):
            recovery_called.append(ctx)
            return RecoveryAction(action_type="log_and_continue", message="retry as JSON")

        registry = RecoveryRegistry()
        registry.register(
            RecoveryRecipe(
                scenario=FailureScenario.PARSE_ERROR,
                handler=handler,
                max_attempts=3,
            )
        )
        mock = AsyncMockLLMBackend(
            responses=[
                "not json",
                '{"tool": "done", "args": {"summary": "recovered"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5, recovery_registry=registry),
            task={},
        ):
            steps.append(step)

        assert recovery_called
        assert steps[-1].tool_call.tool == "done"

    async def test_concurrent_dispatch_uses_batch_path(self):
        class SpyRegistry(BaseToolRegistry):
            def __init__(self) -> None:
                super().__init__()
                self.batch_calls = 0

            def dispatch_batch(self, calls, *, ctx=None):
                self.batch_calls += 1
                return super().dispatch_batch(calls, ctx=ctx)

        tools = SpyRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="a",
                description="a",
                parameters={},
                execute=lambda: {"a": True},
                concurrent_safe=True,
            )
        )
        tools.register(
            ToolSpec(
                name="b",
                description="b",
                parameters={},
                execute=lambda: {"b": True},
                concurrent_safe=True,
            )
        )
        mock = AsyncMockLLMBackend(
            responses=[
                '{"tools": ['
                '{"tool": "a", "args": {}, "reasoning": "r"},'
                '{"tool": "b", "args": {}, "reasoning": "r"},'
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}'
                "]}"
            ]
        )

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5, concurrent_dispatch=True),
            task={},
        ):
            pass

        assert tools.batch_calls == 1
