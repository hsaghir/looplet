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
