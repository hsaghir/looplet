"""async_composable_loop - async agent loop tests."""

from __future__ import annotations

import asyncio

import pytest

from looplet import (
    AssistantTurnDecision,
    BaseToolRegistry,
    ContextProjection,
    DefaultState,
    InjectContext,
    LoopConfig,
    RunStatus,
    ToolSpec,
    register_done_tool,
    run_result_async,
)
from looplet.async_loop import async_composable_loop, async_llm_call
from looplet.cache import CacheControl, CachePolicy
from looplet.testing import AsyncMockLLMBackend
from looplet.types import ToolContext

pytestmark = [pytest.mark.smoke, pytest.mark.asyncio]


@pytest.mark.parametrize("backend_kind", ["future", "coroutine"])
async def test_sync_model_bridge_refuses_owner_loop_wait(monkeypatch, backend_kind):
    from looplet.async_loop import _SyncBridgeLLM

    owner_loop = asyncio.get_running_loop()

    async def nested_response():
        return "nested summary"

    if backend_kind == "coroutine":
        response = nested_response()
    else:
        response = owner_loop.create_future()
        response.set_result("nested summary")

    class Backend:
        def generate(self, prompt, **kwargs):
            return response

    def refuse_owner_loop_schedule(coroutine, loop):
        coroutine.close()
        raise AssertionError("Synchronous bridge scheduled work onto its owner loop")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", refuse_owner_loop_schedule)
    with pytest.raises(RuntimeError, match=r"ctx\.async_llm"):
        _SyncBridgeLLM(Backend(), loop=owner_loop).generate("Summarize")
    if asyncio.iscoroutine(response):
        assert response.cr_frame is None


async def test_sync_model_bridge_preserves_sync_backend_on_owner_loop():
    from looplet.async_loop import _SyncBridgeLLM
    from looplet.testing import MockLLMBackend

    backend = MockLLMBackend(["nested summary"])
    bridge = _SyncBridgeLLM(backend, loop=asyncio.get_running_loop())
    assert bridge.generate("Summarize") == "nested summary"
    assert backend.calls == 1


@pytest.mark.parametrize("native", [False, True])
async def test_async_tool_sync_model_access_fails_without_deadlock(tmp_path, native):
    from looplet import tools_from

    async def summarize(ctx: ToolContext):
        return {"summary": ctx.llm.generate("Summarize")}

    backend = AsyncMockLLMBackend(
        ['{"tool":"summarize","args":{}}', '{"tool":"done","args":{"summary":"ok"}}']
    )
    result = await run_result_async(
        backend,
        tools=tools_from([summarize], include_done=True, done_parameters={"summary": "Outcome"}),
        config=LoopConfig(
            max_steps=2,
            max_model_calls=3,
            use_native_tools=native,
            run_dir=str(tmp_path),
        ),
    )
    assert result.completed
    assert backend.calls == 2
    assert len(result.steps) == 2
    assert "ctx.async_llm" in result.steps[0].tool_result.error
    assert result.steps[1].tool_call.tool == "done"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("model_limit", [None, 6])
async def test_default_persistence_keeps_io_off_the_loop_and_callbacks_on_it(
    tmp_path, monkeypatch, native, model_limit
):
    from functools import wraps
    from threading import get_ident

    from looplet import tools_from
    from looplet.checkpoint import FileCheckpointStore
    from looplet.provenance import ProvenanceSink, TrajectoryRecorder
    from looplet.session import SessionLog
    from looplet.session_tree import SessionTreeRecorder

    owner = get_ident()
    writes = []
    flushes = []
    captures = []
    recorder_callbacks = []
    original_save = FileCheckpointStore.save
    original_flush = ProvenanceSink.flush

    def save(store, checkpoint, key):
        writes.append((key, get_ident()))
        return original_save(store, checkpoint, key)

    def flush(sink):
        flushes.append(get_ident())
        return original_flush(sink)

    def capture(context):
        captures.append(get_ident())
        return {"cursor": context.step_num}

    def recorded(callback):
        @wraps(callback)
        def record(*args, **kwargs):
            assert get_ident() != owner
            recorder_callbacks.append(callback.__name__)
            return callback(*args, **kwargs)

        return record

    class ThreadBoundLog(SessionLog):
        def to_list(self):
            assert get_ident() == owner
            return super().to_list()

        def render(self, *args, **kwargs):
            assert get_ident() == owner
            return super().render(*args, **kwargs)

    class ThreadBoundConfig(LoopConfig):
        def __setattr__(self, name, value):
            assert get_ident() == owner
            super().__setattr__(name, value)

    def source():
        return {"answer": 42}

    monkeypatch.setattr(FileCheckpointStore, "save", save)
    monkeypatch.setattr(ProvenanceSink, "flush", flush)
    for recorder_type, method in (
        (TrajectoryRecorder, "post_dispatch"),
        (TrajectoryRecorder, "on_loop_end"),
        (SessionTreeRecorder, "on_event"),
    ):
        monkeypatch.setattr(recorder_type, method, recorded(getattr(recorder_type, method)))
    backend = AsyncMockLLMBackend(
        ['{"tool":"source","args":{}}', '{"tool":"done","args":{"summary":"ok"}}'],
        cycle=False,
    )
    state = DefaultState(max_steps=2)
    async for _step in async_composable_loop(
        backend,
        tools=tools_from([source], include_done=True, done_parameters={"summary": "Outcome"}),
        state=state,
        session_log=ThreadBoundLog(),
        config=ThreadBoundConfig(
            max_steps=2,
            max_model_calls=model_limit,
            use_native_tools=native,
            run_dir=str(tmp_path),
            checkpoint_state=capture,
        ),
    ):
        pass
    assert state.run_status == "completed"
    assert state.step_count == 2
    assert backend.calls == 2
    assert writes and all(thread != owner for _, thread in writes)
    assert flushes and all(thread != owner for thread in flushes)
    assert captures and all(thread == owner for thread in captures)
    assert set(recorder_callbacks) == {"post_dispatch", "on_event", "on_loop_end"}
    assert any("pending" in key for key, _ in writes)
    assert any(key == "model_turn" for key, _ in writes) == (model_limit is not None)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("action", ["continue", "complete", "stop"])
async def test_assistant_decision_persistence_keeps_io_off_owner_loop(
    tmp_path, monkeypatch, native, action
):
    from threading import get_ident

    from looplet import tools_from
    from looplet.checkpoint import FileCheckpointStore

    owner = get_ident()
    writes = []
    captures = []
    original = FileCheckpointStore.save

    def save(store, checkpoint, key):
        writes.append((key, get_ident()))
        return original(store, checkpoint, key)

    def capture(context):
        captures.append(get_ident())
        return {"cursor": context.step_num}

    def source():
        return {"value": 17}

    monkeypatch.setattr(FileCheckpointStore, "save", save)
    backend = AsyncMockLLMBackend(
        ["A short progress note.", '{"tool":"done","args":{"summary":"ok"}}'],
        cycle=False,
    )
    result = await run_result_async(
        backend,
        tools=tools_from([source], include_done=True, done_parameters={"summary": "Outcome"}),
        config=LoopConfig(
            max_steps=2,
            max_model_calls=4,
            use_native_tools=native,
            allow_text_completion=True,
            assistant_turn_policy=lambda context: AssistantTurnDecision(action),
            checkpoint_state=capture,
            run_dir=str(tmp_path),
        ),
    )
    assert result.status is (RunStatus.STOPPED if action == "stop" else RunStatus.COMPLETED)
    assert backend.calls == (2 if action == "continue" else 1)
    assert writes and all(thread != owner for _, thread in writes)
    assert captures and all(thread == owner for thread in captures)


@pytest.mark.parametrize("fault", ["none", "oserror", "cancelled"])
@pytest.mark.parametrize("cancel_helpers", [False, True])
async def test_persistence_drains_inflight_write_before_propagating_cancellation(
    fault, cancel_helpers
):
    from contextvars import ContextVar
    from threading import Event

    from looplet.checkpoint import _run_persistence

    marker = ContextVar("persistence_context", default="missing")
    marker.set("caller")
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    events = []

    def write():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        assert marker.get() == "caller"
        events.append("written")
        if fault == "oserror":
            raise OSError("storage failed")
        if fault == "cancelled":
            raise asyncio.CancelledError("worker cancellation")

    async def drive():
        try:
            await _run_persistence(write)
        finally:
            events.append("finalized")

    existing_tasks = asyncio.all_tasks()
    pending = asyncio.create_task(drive())
    try:
        await started.wait()
        pending.cancel()
        if cancel_helpers:
            helpers = asyncio.all_tasks() - existing_tasks - {pending}
            for helper in helpers:
                helper.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)
        advanced = loop.create_future()
        loop.call_soon(advanced.set_result, None)
        await advanced
        pending.cancel()
        assert not pending.done()
    finally:
        release.set()
    with pytest.raises(OSError if fault == "oserror" else asyncio.CancelledError):
        await pending
    assert events == ["written", "finalized"]


@pytest.mark.timeout(5)
async def test_persistence_callback_cancellation_propagates_without_retry():
    from looplet.checkpoint import _run_persistence

    calls = []

    def write():
        calls.append("written")
        raise asyncio.CancelledError("worker cancellation")

    with pytest.raises(asyncio.CancelledError) as caught:
        await _run_persistence(write)
    assert caught.value.args == ("worker cancellation",)
    assert calls == ["written"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("profile", ["minimal", "execution-v1"])
@pytest.mark.parametrize("model_limit", [None, 3])
@pytest.mark.parametrize("tool_mode", ["async", "sync"])
async def test_nested_tool_model_calls_stay_on_owner_loop(
    tmp_path, native, profile, model_limit, tool_mode
):
    owner = asyncio.get_running_loop()
    nested_calls = []

    class LoopBoundBackend(AsyncMockLLMBackend):
        async def generate(self, prompt, **kwargs):
            assert asyncio.get_running_loop() is owner
            if prompt == "nested summary":
                nested_calls.append(prompt)
            return await super().generate(prompt, **kwargs)

    async def async_nested(ctx: ToolContext):
        return {"summary": await ctx.async_llm.generate("nested summary")}

    def sync_nested(ctx: ToolContext):
        return {"summary": ctx.llm.generate("nested summary")}

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="nested",
            description="Summarize using the supplied model",
            parameters={},
            execute=async_nested if tool_mode == "async" else sync_nested,
        )
    )
    register_done_tool(tools)
    backend = LoopBoundBackend(
        [
            '{"tool":"nested","args":{}}',
            "nested value",
            '{"tool":"done","args":{"summary":"ok"}}',
        ],
        cycle=False,
    )
    result = await run_result_async(
        backend,
        tools=tools,
        config=LoopConfig(
            max_steps=2,
            max_model_calls=model_limit,
            use_native_tools=native,
            harness_profile=profile,
            run_dir=str(tmp_path),
        ),
    )
    assert result.steps[0].tool_result.error is None
    assert result.steps[0].tool_result.data == {"summary": "nested value"}
    assert nested_calls == ["nested summary"]
    assert backend.calls == 3
    assert result.status is RunStatus.COMPLETED


class TestAsyncLlmCall:
    async def test_forwards_cancel_token_and_kwargs_to_permissive_backend(self):
        class Backend:
            def __init__(self):
                self.calls = []

            async def generate(self, prompt, **kwargs):
                self.calls.append(kwargs)
                return '{"tool":"done","args":{}}'

        from looplet.types import CancelToken

        backend = Backend()
        token = CancelToken()
        await async_llm_call(
            backend,
            "prompt",
            cancel_token=token,
            generate_kwargs={"provider_flag": 7},
            max_retries=0,
        )

        assert backend.calls == [
            {
                "max_tokens": None,
                "system_prompt": "",
                "temperature": None,
                "provider_flag": 7,
                "cancel_token": token,
            }
        ]

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

    async def test_native_failure_stays_error_without_regular_generation(self):
        calls = []

        class NativeFailureBackend:
            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                calls.append("native")
                from looplet.native_tools import NativeToolUnsupportedError

                raise NativeToolUnsupportedError("tools endpoint unsupported")

            async def generate(self, prompt, **kwargs):
                calls.append("regular")
                return '{"tool": "done", "args": {}}'

        result = await async_llm_call(
            NativeFailureBackend(),
            "finish",
            tools=[{"name": "done"}],
            max_retries=0,
        )

        from looplet.native_tools import NativeToolUnsupportedError

        assert not result.ok
        assert isinstance(result.error, NativeToolUnsupportedError)
        assert calls == ["native"]

    async def test_non_native_backend_requires_explicit_text_mode(self):
        from looplet.native_tools import NativeToolPolicy, NativeToolUnsupportedError

        class TextOnlyBackend:
            async def generate(self, prompt, **kwargs):
                return '{"tool": "done", "args": {}}'

        backend = TextOnlyBackend()
        result = await async_llm_call(backend, "finish", tools=[{"name": "done"}])
        assert not result.ok
        assert isinstance(result.error, NativeToolUnsupportedError)

        result = await async_llm_call(
            backend,
            "finish",
            tools=[{"name": "done"}],
            native_policy=NativeToolPolicy(enabled=False),
        )
        assert result.ok
        assert result.text == '{"tool": "done", "args": {}}'

    async def test_provider_failure_does_not_fallback_to_text(self):
        calls = []

        class ProviderFailureBackend:
            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                calls.append("native")
                raise RuntimeError("authentication failed")

            async def generate(self, prompt, **kwargs):
                calls.append("regular")
                return '{"tool": "done", "args": {}}'

        result = await async_llm_call(
            ProviderFailureBackend(),
            "finish",
            tools=[{"name": "done"}],
            max_retries=0,
        )

        assert not result.ok
        assert isinstance(result.error, RuntimeError)
        assert "authentication failed" in str(result.error)
        assert calls == ["native"]

    async def test_async_loop_awaits_async_tools(self):
        seen = []

        async def lookup(*, query: str) -> dict:
            await asyncio.sleep(0)
            seen.append(query)
            return {"query": query}

        mock = AsyncMockLLMBackend(
            responses=[
                '{"tool":"lookup","args":{"query":"hello"}}',
                '{"tool":"done","args":{"summary":"ok"}}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="lookup",
                description="lookup",
                parameters={"query": "str"},
                execute=lookup,
            )
        )

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3, use_native_tools=False),
            task={},
        ):
            steps.append(step)

        assert seen == ["hello"]
        assert steps[0].tool_result.data == {"query": "hello"}
        assert steps[0].tool_result.error is None

    async def test_native_unsupported_stats_are_exposed_on_state(self):
        class TrackingBackend:
            def __init__(self):
                self.calls = []

            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                self.calls.append("native")
                from looplet.native_tools import NativeToolUnsupportedError

                raise NativeToolUnsupportedError("unsupported native endpoint")

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

        assert state.metadata["native_tool_stats"] == {
            "requested": 1,
            "attempted": 1,
            "succeeded": 0,
            "fallbacks": 0,
            "last_fallback_reason": None,
        }
        assert backend.calls == ["native"]

    async def test_check_done_hook_failure_rejects_completion(self):
        class BrokenGate:
            async def check_done(self, state, session_log, context, step_num, tool_call=None):
                raise RuntimeError("gate unavailable")

        llm = AsyncMockLLMBackend(
            responses=[
                '{"tool":"done","args":{"summary":"first"}}',
                '{"tool":"done","args":{"summary":"second"}}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        state = DefaultState(max_steps=2)

        steps = []
        async for step in async_composable_loop(
            llm=llm,
            tools=tools,
            state=state,
            config=LoopConfig(max_steps=2, use_native_tools=False),
            hooks=[BrokenGate()],
            task={},
        ):
            steps.append(step)

        assert steps[0].tool_result.data["rejected"] is True
        assert "quality gate" in steps[0].tool_result.error.lower()


class TestAsyncComposableLoop:
    async def test_preflight_disabled_recovery_never_calls_backend(self):
        class Backend:
            async def generate(self, *_args, **_kwargs):
                raise AssertionError("oversized prompt must be blocked before backend")

        from looplet import BaseToolRegistry, DefaultState, LoopConfig, register_done_tool

        tools = BaseToolRegistry()
        register_done_tool(tools)
        steps = []
        async for step in async_composable_loop(
            llm=Backend(),
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1, context_window=1, reactive_recovery=False),
            task={},
        ):
            steps.append(step)

        assert steps[0].tool_call.tool == "__llm_error__"
        assert "context" in steps[0].tool_result.error.lower()

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
            config=LoopConfig(max_steps=3, use_native_tools=False),
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
                use_native_tools=False,
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
        backend = seen[0]
        for _ in range(5):
            if backend is selected:
                break
            backend = getattr(backend, "_backend", None)
        assert backend is selected
        assert selected.calls == 2
        assert original.calls == 0

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
            config=LoopConfig(max_steps=5, use_native_tools=False),
            task={"goal": "greet Alice"},
        ):
            steps.append(step)

        assert len(steps) == 2
        assert steps[0].tool_call.tool == "greet"
        assert steps[1].tool_call.tool == "done"
        assert mock.calls == 2

    async def test_extract_entities_receives_live_state(self):
        seen_states = []

        def extract_entities(data, *, state):
            seen_states.append(state)
            return ["entity"]

        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="inspect", description="Inspect", parameters={}, execute=lambda: {"value": 1}
            )
        )
        state = DefaultState(max_steps=3)
        async for _ in async_composable_loop(
            llm=AsyncMockLLMBackend(
                responses=[
                    '{"tool":"inspect","args":{}}',
                    '{"tool":"done","args":{"summary":"ok"}}',
                ]
            ),
            tools=tools,
            state=state,
            config=LoopConfig(
                max_steps=3, extract_entities=extract_entities, use_native_tools=False
            ),
            task={},
        ):
            pass

        assert seen_states == [state]

    async def test_max_steps_and_system_prompt_shorthand(self):
        """Regression: ``async_composable_loop`` accepts the same
        ``max_steps`` / ``system_prompt`` keyword shorthands as
        ``composable_loop`` so callers don't need to construct a
        ``LoopConfig`` for one-liner agents."""

        class NativeDoneBackend:
            async def generate(self, prompt, **kwargs):
                raise AssertionError("native shorthand must use generate_with_tools")

            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                return [
                    {"type": "tool_use", "id": "done_1", "name": "done", "input": {"summary": "ok"}}
                ]

        mock = NativeDoneBackend()
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
            config=LoopConfig(max_steps=5, use_native_tools=False),
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
            config=LoopConfig(max_steps=3, use_native_tools=False),
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
            config=LoopConfig(max_steps=5, use_native_tools=False),
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
            config=LoopConfig(max_steps=3, checkpoint_dir=tmp_path, use_native_tools=False),
            task={},
        ):
            pass

        checkpoint = FileCheckpointStore(tmp_path).load("step_1_done")

        assert checkpoint is not None
        assert checkpoint.metadata["status"] == "done"
        assert len(checkpoint.session_log_data["entries"]) == 1

    @pytest.mark.parametrize("cancelled", [False, True])
    async def test_interrupted_dispatch_blocks_implicit_retry(self, tmp_path, cancelled):
        from asyncio import CancelledError

        from looplet import RunResult, RunStatus
        from looplet.checkpoint import FileCheckpointStore

        effects = []
        tools = BaseToolRegistry()
        interruption = CancelledError if cancelled else KeyboardInterrupt

        def interrupt():
            effects.append("started")
            raise interruption

        tools.register(ToolSpec("effect", "Side effect", {}, interrupt))
        register_done_tool(tools)
        config = LoopConfig(max_steps=2, checkpoint_dir=tmp_path, use_native_tools=False)
        response = '{"tool":"effect","args":{},"call_id":"effect-1"}'
        state = DefaultState(max_steps=2)
        with pytest.raises(interruption):
            async for _ in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=[response]),
                tools=tools,
                state=state,
                config=config,
            ):
                pass

        assert RunResult.from_state(state).status is (
            RunStatus.CANCELLED if cancelled else RunStatus.FAILED
        )
        pending = FileCheckpointStore(tmp_path).load_latest_for_resume()
        assert pending is not None
        assert pending.metadata["pending_calls"][0]["execution_status"] == "effect_unknown"
        retry = AsyncMockLLMBackend(responses=[response])
        with pytest.raises(RuntimeError, match="Tool effects unknown"):
            async for _ in async_composable_loop(
                llm=retry, tools=tools, state=DefaultState(max_steps=2), config=config
            ):
                pass
        assert effects == ["started"]
        assert retry.calls == 0

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
                use_native_tools=False,
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
            config=LoopConfig(max_steps=3, checkpoint_dir=tmp_path, use_native_tools=False),
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
            config=LoopConfig(max_steps=3, use_native_tools=False),
            hooks=[StopOrderHook()],
            task={},
        ):
            pass

        assert observations == [("stop", "done", "done"), ("end", "done")]

    async def test_async_pre_loop_receives_tools_when_declared(self):
        seen = []

        class ToolAwareHook:
            async def pre_loop(self, state, session_log, context, tools):
                seen.append(tools)

        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1),
            hooks=[ToolAwareHook()],
            task={},
        ):
            pass

        assert seen == [tools]

    async def test_async_loop_uses_configured_build_briefing(self):
        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(
                max_steps=1,
                use_native_tools=False,
                build_briefing=lambda *_args: "configured briefing",
            ),
            task={},
        ):
            pass

        assert "configured briefing" in mock.last_prompt

    async def test_async_loop_applies_briefing_budget(self):
        class BriefingHook:
            def pre_prompt(self, state, session_log, context, step_num):
                return "x" * 100

        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1, max_briefing_tokens=5, use_native_tools=False),
            hooks=[BriefingHook()],
            task={},
        ):
            pass

        assert "briefing truncated - token budget exceeded" in mock.last_prompt

    async def test_async_batched_done_is_deferred_when_budget_is_exhausted(self):
        seen = []

        class Gate:
            def check_done(self, state, session_log, context, step_num, tool_call=None):
                seen.append(step_num)

        mock = AsyncMockLLMBackend(
            responses=[
                '{"tools": [{"tool": "search", "args": {}}, {"tool": "done", "args": {"summary": "ok"}}]}'
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="search", description="search", parameters={}, execute=lambda: {})
        )

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1, use_native_tools=False),
            hooks=[Gate()],
            task={},
        ):
            steps.append(step)

        assert seen == []
        assert [step.tool_call.tool for step in steps] == ["search"]

    async def test_async_multi_tool_turn_metadata_is_ordered(self):
        mock = AsyncMockLLMBackend(
            responses=[
                '{"tools": [{"tool": "a", "args": {}}, {"tool": "b", "args": {}}]}',
                '{"tool": "done", "args": {}}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="a", description="a", parameters={}, execute=lambda: {"a": True})
        )
        tools.register(
            ToolSpec(name="b", description="b", parameters={}, execute=lambda: {"b": True})
        )

        steps = []
        async for step in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5, use_native_tools=False),
            task={},
        ):
            steps.append(step)

        turns = [step.metadata["turn"] for step in steps[:2]]
        assert [turn["call_index"] for turn in turns] == [0, 1]
        assert {turn["turn_id"] for turn in turns} == {turns[0]["turn_id"]}
        assert {turn["batch_size"] for turn in turns} == {2}

    async def test_async_loop_end_event_follows_cleanup_and_counts_extra_calls(self):
        order = []

        class Stream:
            def __init__(self):
                self.loop_end = None

            def emit(self, event):
                if type(event).__name__ == "LoopEndEvent":
                    order.append("loop_end")
                    self.loop_end = event

        class CleanupHook:
            async def on_loop_end(self, state, session_log, context, llm):
                order.append("cleanup")
                return 1

        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        stream = Stream()

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1),
            hooks=[CleanupHook()],
            task={},
            stream=stream,
        ):
            pass

        assert order == ["cleanup", "loop_end"]
        assert stream.loop_end is not None
        assert stream.loop_end.total_llm_calls == 2

    async def test_async_streaming_hook_does_not_duplicate_loop_lifecycle_events(self):
        from looplet.streaming import CallbackEmitter, LoopEndEvent, LoopStartEvent, StreamingHook

        events = []
        stream = type("Stream", (), {"emit": lambda self, event: events.append(event)})()
        streaming_hook = StreamingHook(CallbackEmitter(events.append))
        mock = AsyncMockLLMBackend(
            responses=['{"tool": "done", "args": {"summary": "finished"}, "reasoning": "r"}']
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1),
            hooks=[streaming_hook],
            task={},
            stream=stream,
        ):
            pass

        assert sum(isinstance(event, LoopStartEvent) for event in events) == 1
        assert sum(isinstance(event, LoopEndEvent) for event in events) == 1

    async def test_async_direct_stream_exposes_only_public_usage(self):
        from looplet.streaming import CallbackEmitter, LLMCallEndEvent

        events = []
        mock = AsyncMockLLMBackend(responses=['{"tool": "done", "args": {"summary": "finished"}}'])
        setattr(mock, "last_usage", {"input": 12, "output": 3, "secret": "private"})
        tools = BaseToolRegistry()
        register_done_tool(tools)

        async for _ in async_composable_loop(
            llm=mock,
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1),
            task={},
            stream=CallbackEmitter(events.append),
        ):
            pass

        ended = [event for event in events if isinstance(event, LLMCallEndEvent)]
        assert len(ended) == 1
        assert ended[0].usage == {"input": 12, "output": 3}

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
                use_native_tools=False,
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
            config=LoopConfig(max_steps=5, recovery_registry=registry, use_native_tools=False),
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
            config=LoopConfig(max_steps=5, concurrent_dispatch=True, use_native_tools=False),
            task={},
        ):
            pass

        assert tools.batch_calls == 1
