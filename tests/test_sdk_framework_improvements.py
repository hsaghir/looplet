"""Tests for SDK and framework improvements across sync/async parity,
compaction projection, input-aware tool predicates, streaming execution,
result recall, parallel subagents, and hierarchical cancellation.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from looplet import (
    AsyncResilientBackend,
    AsyncStreamingToolExecutor,
    CancelToken,
    Conversation,
    DefaultCompactService,
    DefaultState,
    ErrorKind,
    FileToolResultStore,
    LoopConfig,
    MockLLMBackend,
    PermissionDecision,
    PermissionEngine,
    PruneToolResults,
    ResilientBackend,
    RetryExhausted,
    SessionLog,
    Step,
    StreamingHook,
    StreamingToolExecutor,
    ToolCall,
    ToolCancelledError,
    ToolContext,
    ToolResult,
    ToolSpec,
    Tracer,
    TrajectoryRecorder,
    async_composable_loop,
    async_run_sub_loop,
    composable_loop,
    run_result_async,
    run_sub_loops,
    run_sub_loops_parallel,
    tool,
    tools_from,
)
from looplet.builtin_tools import get_builtin_tool
from looplet.history import HistoryRecorder
from looplet.provenance import AsyncRecordingLLMBackend, RecordingLLMBackend
from looplet.streaming import CallbackEmitter, LLMChunkEvent

# ── 1. Sync / Async Parity & Async ResilientBackend ─────────────


async def test_async_loop_records_tracer_spans_and_surfaces_usage() -> None:
    class UsageAsyncBackend:
        last_stop_reason = "stop"

        def __init__(self) -> None:
            self.calls = 0
            self.last_usage: dict[str, int] = {}

        async def generate(self, prompt: str, **kwargs: Any) -> str:
            self.calls += 1
            self.last_usage = {"input": 100 * self.calls, "output": 25 * self.calls}
            if self.calls == 1:
                return json.dumps({"tool": "ping", "args": {"msg": "hi"}})
            return json.dumps({"tool": "done", "args": {"summary": "complete"}})

    @tool(concurrent_safe=True, read_only=True)
    def ping(msg: str) -> dict[str, str]:
        return {"pong": msg}

    tracer = Tracer()
    state = DefaultState(max_steps=4)
    config = LoopConfig(
        max_steps=4,
        use_native_tools=False,
        tracer=tracer,
        disabled_defaults=("persistence",),
    )
    res = await run_result_async(
        UsageAsyncBackend(),
        task="test tracer and usage",
        tools=tools_from([ping], include_done=True),
        config=config,
        state=state,
    )
    assert res.completed
    span_names = [s.name for s in tracer.root_spans]
    assert "llm.call.step_1" in span_names
    assert "llm.call.step_2" in span_names
    assert state.metadata["last_usage"] == {"input": 200, "output": 50}
    assert state.metadata["usage_total"] == {"input": 300, "output": 75}


async def test_async_resilient_backend_retries_and_times_out() -> None:
    attempts = 0

    class FlakyAsyncBackend:
        async def generate(self, prompt: str, **kwargs: Any) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("transient network drop")
            return '{"tool": "done", "args": {"summary": "recovered"}}'

    resilient = AsyncResilientBackend(
        FlakyAsyncBackend(),
        retries=3,
        base_delay_s=0.001,
        max_delay_s=0.005,
        jitter=0.0,
    )
    out = await resilient.generate("hello")
    assert "recovered" in out
    assert attempts == 3

    class SlowAsyncBackend:
        async def generate(self, prompt: str, **kwargs: Any) -> str:
            await asyncio.sleep(5.0)
            return "late"

    timed = ResilientBackend(SlowAsyncBackend(), retries=2, timeout_s=0.03)
    with pytest.raises(RetryExhausted):
        await timed.generate("hello")


# ── 2. Unified History, Compaction & context_summary ────────────


def test_context_summary_respects_prune_and_compaction_without_call_ids() -> None:
    state = DefaultState(max_steps=10)
    log = SessionLog()
    conv = Conversation()
    state.conversation = conv  # type: ignore[attr-defined]
    recorder = HistoryRecorder(state=state, session_log=log, conversation=conv)

    for idx in range(1, 5):
        step = Step(
            number=idx,
            tool_call=ToolCall(tool="fetch", args={"page": idx}),
            tool_result=ToolResult(
                tool="fetch",
                args_summary=f"page={idx}",
                data={"secret_payload": f"PAGE_DATA_{idx}"},
            ),
        )
        recorder.record_llm_turn(prompt=f"prompt {idx}", response=f"resp {idx}")

        recorder.record_step(step)

    assert "PAGE_DATA_1" in state.context_summary()
    assert "PAGE_DATA_4" in state.context_summary()

    # Prune older tool results (keep only the last 2)
    PruneToolResults(keep_recent=2).compact(
        state=state,
        session_log=log,
        llm=MockLLMBackend(responses=[]),
        conversation=conv,
        step_num=4,
        reason="test_prune",
    )
    summary_after_prune = state.context_summary()
    assert "PAGE_DATA_1" not in summary_after_prune
    assert "PAGE_DATA_2" not in summary_after_prune
    assert "[tool result cleared by compact]" in summary_after_prune
    assert "PAGE_DATA_3" in summary_after_prune
    assert "PAGE_DATA_4" in summary_after_prune

    # Now compact the conversation keeping only the last step's messages
    DefaultCompactService(keep_recent=2, keep_recent_tool_results=1, use_llm_summary=False).compact(
        state=state,
        session_log=log,
        llm=MockLLMBackend(responses=[]),
        conversation=conv,
        step_num=4,
        reason="test_compact",
    )
    summary_after_compact = state.context_summary()
    assert "S1" not in summary_after_compact
    assert "S2" not in summary_after_compact
    assert "S3" not in summary_after_compact
    assert "PAGE_DATA_4" in summary_after_compact


# ── 3. Input-Aware Tool Predicates & Permission Helpers ─────────


def test_input_aware_tool_predicates_partition_and_permissions() -> None:
    @tool(
        concurrent_safe=lambda args: str(args.get("cmd", "")).startswith(("ls", "cat")),
        read_only=lambda args: str(args.get("cmd", "")).startswith(("ls", "cat")),
        destructive=lambda args: "rm " in str(args.get("cmd", "")),
    )
    def shell(cmd: str) -> dict[str, str]:
        return {"ran": cmd}

    registry = tools_from([shell])
    calls = [
        ToolCall(tool="shell", args={"cmd": "ls -la"}),
        ToolCall(tool="shell", args={"cmd": "cat README.md"}),
        ToolCall(tool="shell", args={"cmd": "rm -rf build"}),
    ]
    batches = registry._partition_calls(calls)
    assert len(batches) == 2
    assert batches[0]["concurrent"] is True
    assert len(batches[0]["calls"]) == 2
    assert batches[1]["concurrent"] is False
    assert len(batches[1]["calls"]) == 1

    engine = (
        PermissionEngine(
            default=PermissionDecision.ASK,
            ask_handler=lambda c, r: PermissionDecision.ALLOW,
        )
        .deny_destructive(registry)
        .allow_read_only(registry)
    )
    assert engine.evaluate(calls[0]).decision == PermissionDecision.ALLOW
    assert engine.evaluate(calls[0]).reason == "read-only tool call"
    assert engine.evaluate(calls[2]).decision == PermissionDecision.DENY
    assert engine.evaluate(calls[2]).reason == "destructive tool call denied"


def test_buggy_tool_predicates_fail_closed() -> None:
    def boom(_args: dict[str, Any]) -> bool:
        raise RuntimeError("predicate bug")

    spec = ToolSpec(
        name="risky",
        description="risky",
        parameters={"x": "str"},
        execute=lambda x: x,
        concurrent_safe=boom,
        read_only=boom,
        destructive=boom,
    )
    assert spec.is_concurrent_safe({"x": "1"}) is False
    assert spec.is_read_only({"x": "1"}) is False
    assert spec.is_destructive({"x": "1"}) is True


# ── 4. Token-Level LLMChunkEvent & StreamingToolExecutor ────────


def test_sync_and_async_loops_emit_llm_chunk_events() -> None:
    class ChunkSyncBackend:
        stream_in_loop = True

        def generate(self, prompt: str, **kwargs: Any) -> str:
            return '{"tool": "done", "args": {"summary": "ok"}}'

        def stream(self, prompt: str, **kwargs: Any):
            for part in ['{"tool": ', '"done", ', '"args": {"summary": "ok"}}']:
                yield part

    class ChunkAsyncBackend:
        stream_in_loop = True

        async def generate(self, prompt: str, **kwargs: Any) -> str:
            return '{"tool": "done", "args": {"summary": "ok"}}'

        async def stream(self, prompt: str, **kwargs: Any):
            for part in ['{"tool": ', '"done", ', '"args": {"summary": "ok"}}']:
                yield part

    sync_events: list[Any] = []
    rec_sync = RecordingLLMBackend(ChunkSyncBackend())
    list(
        composable_loop(
            rec_sync,
            task="stream sync",
            tools=tools_from([], include_done=True),
            config=LoopConfig(
                max_steps=2,
                use_native_tools=False,
                disabled_defaults=("persistence",),
            ),
            stream=CallbackEmitter(sync_events.append),
        )
    )
    sync_chunks = [e for e in sync_events if isinstance(e, LLMChunkEvent)]
    assert [c.chunk for c in sync_chunks] == [
        '{"tool": ',
        '"done", ',
        '"args": {"summary": "ok"}}',
    ]
    assert len(rec_sync.calls) == 1
    assert rec_sync.calls[0].response == '{"tool": "done", "args": {"summary": "ok"}}'

    async def _run_async() -> None:
        async_events: list[Any] = []
        rec_async = AsyncRecordingLLMBackend(ChunkAsyncBackend())
        async for _ in async_composable_loop(
            rec_async,
            task="stream async",
            tools=tools_from([], include_done=True),
            config=LoopConfig(
                max_steps=2,
                use_native_tools=False,
                disabled_defaults=("persistence",),
            ),
            hooks=[StreamingHook(CallbackEmitter(async_events.append))],
        ):
            pass
        async_chunks = [e for e in async_events if isinstance(e, LLMChunkEvent)]
        assert [c.chunk for c in async_chunks] == [
            '{"tool": ',
            '"done", ',
            '"args": {"summary": "ok"}}',
        ]
        assert len(rec_async.calls) == 1

    asyncio.run(_run_async())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ending", ["complete", "close", "error"])
async def test_streaming_records_durable_intent_and_partial_outcomes(
    tmp_path: Path, monkeypatch, asynchronous: bool, ending: str
) -> None:
    from looplet.provenance import load_model_attempts

    class SyncBackend:
        closed = False

        def generate(self, prompt, **kwargs):
            raise AssertionError("Streaming must not buffer through generate")

        def parts(self):
            try:
                yield "first"
                if ending == "error":
                    raise ValueError("provider stream stopped")
                yield "second"
            finally:
                self.closed = True

        def stream(self, prompt, **kwargs):
            self.active_stream = self.parts()
            return self.active_stream

    class AsyncBackend(SyncBackend):
        async def generate(self, prompt, **kwargs):
            raise AssertionError("Streaming must not buffer through generate")

        async def parts(self):
            try:
                yield "first"
                if ending == "error":
                    raise ValueError("provider stream stopped")
                yield "second"
            finally:
                self.closed = True

    backend = AsyncBackend() if asynchronous else SyncBackend()
    wrapper = AsyncRecordingLLMBackend if asynchronous else RecordingLLMBackend
    recorder = wrapper(backend, attempt_directory=tmp_path / "model-attempts")
    writes = []
    original_write = recorder._write_attempt

    def write(*args, **kwargs):
        writes.append(threading.get_ident())
        return original_write(*args, **kwargs)

    monkeypatch.setattr(recorder, "_write_attempt", write)
    stream = recorder.stream("task", max_tokens=17, temperature=0.3)
    try:
        assert (await anext(stream) if asynchronous else next(stream)) == "first"
        attempts = load_model_attempts(tmp_path)
        assert len(attempts) == 1
        assert attempts[0]["outcome"] is None
        assert attempts[0]["intent"]["request"]["temperature"] == 0.3
        if ending == "error":
            with pytest.raises(ValueError, match="provider stream stopped"):
                if asynchronous:
                    await anext(stream)
                else:
                    next(stream)
        elif ending == "complete":
            remaining = [chunk async for chunk in stream] if asynchronous else list(stream)
            assert remaining == ["second"]
    finally:
        if asynchronous:
            await stream.aclose()
        else:
            stream.close()

    attempts = load_model_attempts(tmp_path)
    assert len(attempts) == len(recorder.calls) == 1
    outcome = attempts[0]["outcome"]
    assert outcome["response_complete"] is (ending == "complete")
    assert outcome["capture"]["response"] == ("firstsecond" if ending == "complete" else "first")
    assert backend.closed
    if asynchronous:
        assert writes and all(identifier != threading.get_ident() for identifier in writes)


async def test_streaming_tool_executors_overlap_safe_and_serialize_exclusive() -> None:
    barrier = threading.Barrier(2, timeout=2.0)
    order: list[str] = []

    @tool(concurrent_safe=True)
    def safe_read(label: str) -> str:
        barrier.wait()
        order.append(f"safe:{label}")
        return label

    @tool(concurrent_safe=False)
    def exclusive_write(label: str) -> str:
        order.append(f"write:{label}")
        return label

    registry = tools_from([safe_read, exclusive_write])
    executor = StreamingToolExecutor(registry, max_workers=4)
    executor.submit(ToolCall(tool="safe_read", args={"label": "a"}))
    executor.submit(ToolCall(tool="safe_read", args={"label": "b"}))
    executor.submit(ToolCall(tool="exclusive_write", args={"label": "c"}))
    results = executor.results(timeout=5.0)
    assert [r.data for r in results] == ["a", "b", "c"]
    assert order[-1] == "write:c"

    # AsyncStreamingToolExecutor + discard fallback
    async_exec = AsyncStreamingToolExecutor(registry, max_concurrent=4)
    async_exec.submit(ToolCall(tool="exclusive_write", args={"label": "first"}))
    async_exec.discard()
    async_exec.submit(ToolCall(tool="exclusive_write", args={"label": "second"}))
    discarded = await async_exec.results()
    assert len(discarded) == 2
    assert all(r.error_detail and r.error_detail.kind == ErrorKind.CANCELLED for r in discarded)


async def test_async_streaming_discard_settles_running_tools_before_return() -> None:
    started = asyncio.Event()
    closed = asyncio.Event()
    release = asyncio.Event()

    @tool(concurrent_safe=True)
    async def suspended_tool() -> str:
        started.set()
        try:
            await release.wait()
            return "released"
        finally:
            closed.set()

    executor = AsyncStreamingToolExecutor(tools_from([suspended_tool]), max_concurrent=1)
    executor.submit(ToolCall(tool="suspended_tool", args={}))
    await started.wait()
    try:
        executor.discard()
        results = await executor.results()
        assert closed.is_set(), "discarded tool cleanup must finish before results return"
        assert len(results) == 1
        assert results[0].error_detail is not None
        assert results[0].error_detail.kind == ErrorKind.CANCELLED
    finally:
        await asyncio.wait_for(closed.wait(), timeout=1.0)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streaming_discard_cancels_owned_token_without_cancelling_parent(
    asynchronous: bool,
) -> None:
    parent = CancelToken()
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    observed: list[CancelToken] = []

    @tool(concurrent_safe=True)
    def waiting_tool(ctx: ToolContext) -> str:
        assert ctx.cancel_token is not None
        observed.append(ctx.cancel_token)
        started.set()
        try:
            assert release.wait(timeout=2.0)
            return "released"
        finally:
            closed.set()

    registry = tools_from([waiting_tool])
    context = ToolContext(cancel_token=parent)
    executor = (
        AsyncStreamingToolExecutor(registry, ctx=context, max_concurrent=1)
        if asynchronous
        else StreamingToolExecutor(registry, ctx=context, max_workers=1)
    )
    executor.submit(ToolCall(tool="waiting_tool", args={}))
    assert await asyncio.to_thread(started.wait, 1.0)
    try:
        executor.discard()
        assert observed[0].is_cancelled, "discard must reach the running tool's owned token"
        assert not parent.is_cancelled
    finally:
        release.set()
        if asynchronous:
            await executor.results()
        else:
            await asyncio.to_thread(executor.results, 2.0)
        assert await asyncio.to_thread(closed.wait, 1.0)


def test_sync_streaming_discard_waits_for_worker_with_result_deadline() -> None:
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    @tool(concurrent_safe=True)
    def waiting_tool() -> str:
        started.set()
        try:
            assert release.wait(timeout=2.0)
            return "released"
        finally:
            closed.set()

    executor = StreamingToolExecutor(tools_from([waiting_tool]), max_workers=1)
    executor.submit(ToolCall(tool="waiting_tool", args={}))
    assert started.wait(timeout=1.0)
    try:
        executor.discard()
        with pytest.raises(TimeoutError, match="waiting for tool results"):
            executor.results(timeout=0.0)
    finally:
        release.set()
        results = executor.results(timeout=2.0)
        assert closed.is_set()
        assert len(results) == 1


# ── 5. Result Recall & Line-Range Slicing ───────────────────────


def test_read_result_and_recall_result_support_line_ranges(tmp_path: Path) -> None:
    lines_text = "\n".join(f"line-{i}: value={i * 10}" for i in range(1, 21))

    @tool
    def emit_log() -> dict[str, str]:
        return {"log": lines_text}

    registry = tools_from([emit_log], include_result_tools=True)
    recall_spec = get_builtin_tool("recall_result")
    assert recall_spec is not None
    registry.register(recall_spec)

    step_res = registry.dispatch(ToolCall(tool="emit_log", args={}))
    assert step_res.result_key is not None

    sliced = registry.dispatch(
        ToolCall(
            tool="recall_result",
            args={
                "result_key": step_res.result_key,
                "json_pointer": "/log",
                "start_line": 3,
                "end_line": 5,
            },
        )
    )
    assert sliced.error is None
    assert sliced.data["start_line"] == 3
    assert sliced.data["end_line"] == 5
    assert sliced.data["total_lines"] == 20
    assert sliced.data["content"].splitlines() == [
        "line-3: value=30",
        "line-4: value=40",
        "line-5: value=50",
    ]

    # Also verify FileToolResultStore line-range support
    file_store = FileToolResultStore(str(tmp_path / "results"))
    f_key = file_store.store_result({"log": lines_text})
    f_page = file_store.read_result(f_key, json_pointer="/log", start_line=10, end_line=12)
    assert f_page["content"].splitlines() == [
        "line-10: value=100",
        "line-11: value=110",
        "line-12: value=120",
    ]


# ── 6. Sync Parallel Subagents & Trajectory Provenance Linking ──


def _wrap_test_backend(backend, wrapper):
    from looplet.backend_contract import BackendWrapper
    from looplet.router import FallbackRouter, ModelProfile, RoutingLLMBackend, SimpleRouter

    if wrapper == "bare":
        return backend
    if wrapper == "transparent":
        return BackendWrapper(backend)
    if wrapper == "resilient":
        return ResilientBackend(backend, retries=1)
    if wrapper == "routing":
        return RoutingLLMBackend(SimpleRouter({}, default_profile=ModelProfile("default", backend)))
    return FallbackRouter(primary=backend, fallback=backend).select("reasoning")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("wrapper", ["bare", "transparent", "resilient", "routing", "fallback"])
@pytest.mark.parametrize("recording_order", ["inside", "outside"])
@pytest.mark.parametrize("streaming_allowed", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_admitted_backend_preserves_mode_metadata_and_accounting(
    tmp_path, asynchronous, wrapper, recording_order, streaming_allowed, native
):
    from looplet.assistant_turns import TURN_STATE_KEY

    response = json.dumps({"tool": "done", "args": {"summary": "ok"}})
    native_response = [
        {"type": "tool_use", "name": "done", "id": "completion", "input": {"summary": "ok"}}
    ]
    chunks = [response[:20], response[20:]]
    calls = []

    class Backend:
        stream_in_loop = streaming_allowed
        last_model = "contract-model"

        def __init__(self):
            self.last_usage = {}
            self.last_request = {}

        def record_request(self, mode, prompt, options):
            self.last_usage = {"input": 11, "output": 5}
            self.last_request = {"mode": mode, "prompt": prompt, "options": dict(options)}

        def generate(self, prompt, **kwargs):
            calls.append("generate")
            self.record_request("generate", prompt, kwargs)
            return response

        def generate_with_tools(self, prompt, **kwargs):
            calls.append("native")
            self.record_request("native", prompt, kwargs)
            return native_response

        def stream(self, prompt, **kwargs):
            calls.append("stream")
            self.record_request("stream", prompt, kwargs)
            yield from chunks

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

        async def stream(self, prompt, **kwargs):
            calls.append("stream")
            self.record_request("stream", prompt, kwargs)
            for chunk in chunks:
                yield chunk

    backend = AsyncBackend() if asynchronous else Backend()
    recorder_type = AsyncRecordingLLMBackend if asynchronous else RecordingLLMBackend
    if recording_order == "inside":
        recorder = recorder_type(backend, attempt_directory=tmp_path)
        wrapped = _wrap_test_backend(recorder, wrapper)
    else:
        recorder = recorder_type(_wrap_test_backend(backend, wrapper), attempt_directory=tmp_path)
        wrapped = recorder
    events = []
    state = DefaultState(max_steps=1)
    request = {
        "llm": wrapped,
        "tools": tools_from([], include_done=True),
        "state": state,
        "task": "Finish with the completion tool.",
        "config": LoopConfig(
            harness_profile="minimal",
            max_steps=1,
            max_model_calls=1,
            max_tokens=57,
            temperature=0.25,
            use_native_tools=native,
        ),
        "stream": CallbackEmitter(events.append),
    }
    if asynchronous:
        async for _step in async_composable_loop(**request):
            pass
    else:
        list(composable_loop(**request))

    assert calls == (["native"] if native else ["stream"] if streaming_allowed else ["generate"])
    assert [event.chunk for event in events if isinstance(event, LLMChunkEvent)] == (
        chunks if streaming_allowed and not native else []
    )
    assert state.metadata[TURN_STATE_KEY]["model_calls"] == 1
    assert len(recorder.calls) == 1
    if native:
        assert recorder.calls[0].response == native_response
        assert [schema["name"] for schema in wrapped.last_request["options"]["tools"]] == ["done"]
    else:
        assert recorder.calls[0].response == response
    assert wrapped.last_model == "contract-model"
    assert wrapped.last_usage == {"input": 11, "output": 5}
    assert wrapped.last_request == backend.last_request
    assert wrapped.last_request["options"]["max_tokens"] == 57
    assert wrapped.last_request["options"]["temperature"] == 0.25
    assert state.metadata["usage_total"] == {"input": 11, "output": 5}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("termination", ["complete", "close", "error", "cancel"])
@pytest.mark.parametrize("wrapper", ["bare", "transparent", "resilient", "routing", "fallback"])
@pytest.mark.parametrize("recording_order", ["inside", "outside"])
async def test_admitted_stream_settles_lifecycle_and_budget(
    tmp_path, asynchronous, termination, wrapper, recording_order
):
    from looplet.assistant_turns import TURN_STATE_KEY, _AssistantTurns
    from looplet.backend_contract import TerminalLLMError
    from looplet.provenance import load_model_attempts

    calls = []
    closed = []
    failures = {
        "error": RuntimeError("provider failed"),
        "cancel": asyncio.CancelledError("provider cancelled"),
    }

    class Backend:
        def stream(self, prompt, **kwargs):
            calls.append(prompt)
            try:
                yield "partial"
                if termination in failures:
                    raise failures[termination]
                yield "rest"
            finally:
                closed.append(True)

    class AsyncBackend(Backend):
        async def stream(self, prompt, **kwargs):
            calls.append(prompt)
            try:
                yield "partial"
                if termination in failures:
                    raise failures[termination]
                yield "rest"
            finally:
                closed.append(True)

    backend = AsyncBackend() if asynchronous else Backend()
    recorder_type = AsyncRecordingLLMBackend if asynchronous else RecordingLLMBackend
    if recording_order == "inside":
        recorder = recorder_type(backend, attempt_directory=tmp_path / "model-attempts")
        wrapped = _wrap_test_backend(recorder, wrapper)
    else:
        recorder = recorder_type(
            _wrap_test_backend(backend, wrapper), attempt_directory=tmp_path / "model-attempts"
        )
        wrapped = recorder
    state = DefaultState(max_steps=2)
    turns = _AssistantTurns(state, LoopConfig(harness_profile="minimal", max_model_calls=1))
    checkpoints = []
    if asynchronous:

        async def checkpoint():
            checkpoints.append("admit")

        turns.async_checkpoint = checkpoint
    turns.begin_turn()
    admitted = turns.wrap(wrapped)
    iterator = admitted.stream("first")
    if asynchronous:
        assert await anext(iterator) == "partial"
        if termination == "close":
            await iterator.aclose()
        elif termination in failures:
            with pytest.raises(type(failures[termination])) as caught:
                await anext(iterator)
            assert caught.value is failures[termination]
        else:
            assert [chunk async for chunk in iterator] == ["rest"]
    else:
        assert next(iterator) == "partial"
        if termination == "close":
            iterator.close()
        elif termination in failures:
            with pytest.raises(type(failures[termination])) as caught:
                next(iterator)
            assert caught.value is failures[termination]
        else:
            assert list(iterator) == ["rest"]

    response = "partialrest" if termination == "complete" else "partial"
    assert calls == ["first"] and closed == [True]
    assert checkpoints == (["admit"] if asynchronous else [])
    assert not turns._active_calls and not turns._pending_calls
    assert state.metadata[TURN_STATE_KEY]["model_calls"] == 1
    assert state.metadata[TURN_STATE_KEY]["requests"][0]["response_chars"] == len(response)
    assert len(recorder.calls) == 1 and recorder.calls[0].response == response
    attempts = load_model_attempts(tmp_path)
    assert len(attempts) == 1
    assert attempts[0]["outcome"]["response_complete"] is (termination == "complete")

    with pytest.raises(TerminalLLMError, match=r"model_call_budget"):
        refused = admitted.stream("second")
        if asynchronous:
            await anext(refused)
        else:
            next(refused)
    assert calls == ["first"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("max_model_calls", [0, 1])
async def test_default_child_state_honors_model_call_cap(asynchronous, max_model_calls):
    from looplet import run_sub_loop, run_sub_loop_async

    calls = []

    class Backend:
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            return json.dumps({"tool": "done", "args": {"summary": "ok"}})

    request = {
        "llm": Backend(),
        "tools": tools_from([], include_done=True),
        "task": "Finish within the caller's model allowance.",
        "max_steps": 1,
        "config": LoopConfig(
            harness_profile="minimal",
            max_steps=1,
            max_model_calls=max_model_calls,
            use_native_tools=False,
        ),
    }
    if max_model_calls == 0:
        with pytest.raises(ValueError, match=r"max_model_calls.*positive"):
            if asynchronous:
                await run_sub_loop_async(**request)
            else:
                run_sub_loop(**request)
        assert not calls
        return
    result = await run_sub_loop_async(**request) if asynchronous else run_sub_loop(**request)
    assert len(calls) == result["llm_calls"] == 1
    assert result["completed"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("max_concurrency", [2, 4])
async def test_child_fanout_bounds_scheduled_work(monkeypatch, asynchronous, max_concurrency):
    from concurrent.futures import ThreadPoolExecutor

    from looplet import run_sub_loops_async

    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    async_release = asyncio.Event()
    sync_release = threading.Event()
    lock = threading.Lock()
    started = 0
    scheduled = []

    def admitted():
        nonlocal started
        with lock:
            started += 1
            if started == max_concurrency:
                loop.call_soon_threadsafe(ready.set)

    class SyncBackend:
        def generate(self, prompt, **kwargs):
            raise AssertionError("Native children must not use text fallback")

        def generate_with_tools(self, prompt, **kwargs):
            admitted()
            sync_release.wait()
            return [
                {"type": "tool_use", "name": "done", "id": "finish", "input": {"summary": "ok"}}
            ]

    class AsyncBackend(SyncBackend):
        async def generate(self, prompt, **kwargs):
            raise AssertionError("Native children must not use text fallback")

        async def generate_with_tools(self, prompt, **kwargs):
            admitted()
            await async_release.wait()
            return [
                {"type": "tool_use", "name": "done", "id": "finish", "input": {"summary": "ok"}}
            ]

    if asynchronous:
        original = asyncio.TaskGroup.create_task

        def create_task(group, coroutine, *args, **kwargs):
            if coroutine.__qualname__.startswith("run_sub_loops_async.<locals>."):
                scheduled.append(coroutine.__qualname__)
            return original(group, coroutine, *args, **kwargs)

        monkeypatch.setattr(asyncio.TaskGroup, "create_task", create_task)
    else:
        original = ThreadPoolExecutor.submit

        def submit(pool, function, *args, **kwargs):
            if args and getattr(args[0], "__qualname__", "").startswith("run_sub_loops.<locals>."):
                scheduled.append(args[0].__qualname__)
            return original(pool, function, *args, **kwargs)

        monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)

    requests = [
        {
            "llm": AsyncBackend() if asynchronous else SyncBackend(),
            "tools": tools_from([], include_done=True),
            "max_steps": 1,
            "config": LoopConfig(harness_profile="minimal", max_steps=1),
            "task": "Complete one native child call.",
        }
        for _ in range(32)
    ]
    result_task = asyncio.create_task(
        run_sub_loops_async(requests, max_concurrency=max_concurrency)
        if asynchronous
        else asyncio.to_thread(run_sub_loops, requests, max_concurrency=max_concurrency)
    )
    try:
        await ready.wait()
        assert len(scheduled) == max_concurrency
    finally:
        async_release.set()
        sync_release.set()
        results = await result_task

    assert len(results) == len(requests)
    assert all(result["status"] == "completed" for result in results)


def test_sync_parallel_subagents_and_trajectory_subagent_metadata() -> None:
    rec = TrajectoryRecorder()
    requests = [
        {
            "llm": MockLLMBackend(
                responses=[json.dumps({"tool": "done", "args": {"summary": f"child-{i}"}})]
            ),
            "tools": tools_from([], include_done=True),
            "config": LoopConfig(
                max_steps=2,
                use_native_tools=False,
                disabled_defaults=("persistence",),
            ),
            "parent_hooks": [rec],
            "subagent_id": f"worker-{i}",
        }
        for i in range(3)
    ]
    results = run_sub_loops_parallel(requests, max_concurrency=2)
    assert [r["output"]["summary"] for r in results] == ["child-0", "child-1", "child-2"]
    recorded_subs = rec.trajectory.metadata.get("subagents", [])
    assert len(recorded_subs) == 3
    assert {entry["subagent_id"] for entry in recorded_subs} == {
        "worker-0",
        "worker-1",
        "worker-2",
    }
    assert async_run_sub_loop is not None
    assert run_sub_loops is run_sub_loops_parallel


# ── 7. Hierarchical CancelToken & Sibling Abort Policy ──────────


def test_hierarchical_cancel_token_and_sibling_abort_policy() -> None:
    parent = CancelToken()
    child = parent.create_child()
    assert not parent.is_cancelled
    assert not child.is_cancelled

    # Cancelling child does NOT cancel parent
    child.cancel()
    assert child.is_cancelled
    assert not parent.is_cancelled
    with pytest.raises(ToolCancelledError):
        child.raise_if_cancelled()

    # Cancelling parent DOES cancel another child
    child2 = parent.create_child()
    parent.cancel()
    assert child2.is_cancelled

    # Sibling abort in dispatch_batch without cancelling caller's parent token
    caller_token = CancelToken()

    @tool(concurrent_safe=False)
    def step_one() -> str:
        raise ValueError("step_one failed")

    @tool(concurrent_safe=False)
    def step_two(ctx: ToolContext) -> str:
        return "should not run"

    reg = tools_from([step_one, step_two])
    batch_results = reg.dispatch_batch(
        [ToolCall(tool="step_one", args={}), ToolCall(tool="step_two", args={})],
        ctx=ToolContext(cancel_token=caller_token),
        cancel_on_sibling_error=True,
    )
    assert batch_results[0].error_detail is not None
    assert batch_results[0].error_detail.kind == ErrorKind.VALIDATION
    assert batch_results[1].error_detail is not None
    assert batch_results[1].error_detail.kind == ErrorKind.CANCELLED
    assert not caller_token.is_cancelled
