"""Wave-2 first-principles tests for observation fidelity, O(N) evidence, module isolation, protocol bounds, and lazy exports."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from looplet import (
    AgentPreset,
    AgentRuntime,
    Conversation,
    DefaultState,
    ErrorKind,
    FileRunStore,
    FileToolResultStore,
    LoopConfig,
    Message,
    MessageRole,
    MockLLMBackend,
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    ProvenanceSink,
    StreamingToolExecutor,
    Tracer,
    blueprint_from_preset,
    cartridge_to_preset,
    minimal_preset,
    preset_to_cartridge,
    run_result,
    run_result_async,
    tool,
    tools_from,
)
from looplet.backends import (
    AnthropicStreamingBackend,
    AsyncAnthropicBackend,
    AsyncOpenAIStreamingBackend,
    OpenAIBackend,
    OpenAIStreamingBackend,
)
from looplet.events import EventPayload, LifecycleEvent
from looplet.lep import LEPHookAdapter
from looplet.mcp import MCPToolAdapter
from looplet.provenance import load_model_attempts
from looplet.state_service import StateServiceError, _LineReader
from looplet.streaming import AsyncStreamingToolExecutor
from looplet.types import Step, ToolCall, ToolError, ToolResult


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_speculative_streaming_reuses_partial_prefix_without_double_execution(
    asynchronous: bool,
) -> None:
    counts = {"safe_op": 0, "serial_op": 0}

    @tool(concurrent_safe=True)
    def safe_op(x: int) -> dict[str, int]:
        counts["safe_op"] += 1
        return {"x": x, "executions": counts["safe_op"]}

    @tool(concurrent_safe=False)
    def serial_op(y: int) -> dict[str, int]:
        counts["serial_op"] += 1
        return {"y": y, "executions": counts["serial_op"]}

    class PartialStreamBackend:
        turn = 0

        def generate_with_tools(
            self,
            prompt: str,
            *,
            tools: list[dict[str, Any]],
            on_chunk: Any = None,
            on_tool_call: Any = None,
            **kw: Any,
        ) -> list[dict[str, Any]]:
            del prompt, tools, on_chunk, kw
            self.turn += 1
            if self.turn == 1:
                b0 = {"type": "tool_use", "id": "c0", "name": "safe_op", "input": {"x": 10}}
                b1 = {"type": "tool_use", "id": "c1", "name": "serial_op", "input": {"y": 20}}
                b2 = {"type": "tool_use", "id": "c2", "name": "safe_op", "input": {"x": 30}}
                if on_tool_call is not None:
                    # Interleaved non-zero block index after a text block + serial barrier at c1
                    on_tool_call({**b0, "index": 1})
                    on_tool_call({**b1, "index": 2})
                    on_tool_call({**b2, "index": 3})
                return [{"type": "text", "text": "plan"}, b0, b1, b2]
            return [{"type": "tool_use", "id": "d0", "name": "done", "input": {"summary": "ok"}}]

        def generate(self, prompt: str, **kw: Any) -> str:
            return ""

    class AsyncPartialStreamBackend(PartialStreamBackend):
        async def generate_with_tools(self, prompt: str, **kw: Any) -> list[dict[str, Any]]:  # type: ignore[override]
            return super().generate_with_tools(prompt, **kw)

        async def generate(self, prompt: str, **kw: Any) -> str:  # type: ignore[override]
            return ""

    cfg = LoopConfig(
        max_steps=6,
        use_native_tools=True,
        concurrent_dispatch=True,
        streaming_tool_execution=True,
        harness_profile="minimal",
    )
    registry = tools_from([safe_op, serial_op], include_done=True)
    if asynchronous:
        res = await run_result_async(
            AsyncPartialStreamBackend(), task="go", tools=registry, config=cfg
        )
    else:
        res = run_result(PartialStreamBackend(), task="go", tools=registry, config=cfg)
    assert res.completed
    assert counts == {"safe_op": 2, "serial_op": 1}
    assert [s.tool_result.data["x"] for s in res.steps if s.tool_call.tool == "safe_op"] == [10, 30]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streaming_discard_distinguishes_queued_unknown_and_completed_status(
    asynchronous: bool,
) -> None:
    started = threading.Event()
    release = threading.Event()

    @tool(concurrent_safe=False)
    def blocking_op() -> dict[str, str]:
        started.set()
        release.wait(timeout=3.0)
        return {"committed": "yes"}

    @tool(concurrent_safe=False)
    def queued_op() -> dict[str, str]:
        return {"queued": "never"}

    reg = tools_from([blocking_op, queued_op])
    if asynchronous:
        exec_async = AsyncStreamingToolExecutor(reg, max_concurrent=1)

        @tool(concurrent_safe=False)
        async def async_blocking() -> dict[str, str]:
            started.set()
            await asyncio.to_thread(release.wait, 3.0)
            return {"committed": "yes"}

        reg_a = tools_from([async_blocking, queued_op])
        exec_async = AsyncStreamingToolExecutor(reg_a, max_concurrent=1)
        exec_async.submit(ToolCall(tool="async_blocking", args={}, call_id="a1"))
        exec_async.submit(ToolCall(tool="queued_op", args={}, call_id="a2"))
        await asyncio.to_thread(started.wait, 2.0)
        exec_async.discard()
        release.set()
        results = await exec_async.results()
    else:
        exec_sync = StreamingToolExecutor(reg, max_workers=1)
        exec_sync.submit(ToolCall(tool="blocking_op", args={}, call_id="s1"))
        exec_sync.submit(ToolCall(tool="queued_op", args={}, call_id="s2"))
        assert started.wait(timeout=2.0)
        exec_sync.discard()
        release.set()
        results = exec_sync.results(timeout=3.0)

    assert results[0].metadata["execution_status"] in {"effect_unknown", "completed_before_discard"}
    assert results[1].metadata["execution_status"] == "not_executed"


async def test_async_batch_cancellation_retains_settled_sibling_steps() -> None:
    mutated: list[str] = []
    fast_done = asyncio.Event()
    slow_waiting = asyncio.Event()

    @tool(concurrent_safe=True)
    async def fast_write() -> dict[str, str]:
        mutated.append("committed")
        fast_done.set()
        return {"status": "committed"}

    @tool(concurrent_safe=True)
    async def slow_op() -> dict[str, str]:
        await fast_done.wait()
        slow_waiting.set()
        await asyncio.sleep(10)
        return {"status": "slow"}

    class Stub:
        async def generate_with_tools(self, prompt: str, **kw: Any) -> list[dict[str, Any]]:
            return [
                {"type": "tool_use", "id": "c_fast", "name": "fast_write", "input": {}},
                {"type": "tool_use", "id": "c_slow", "name": "slow_op", "input": {}},
            ]

        async def generate(self, prompt: str, **kw: Any) -> str:
            return ""

    state = DefaultState(max_steps=4)
    conv = Conversation()
    task = asyncio.create_task(
        run_result_async(
            Stub(),
            task="batch cancel",
            tools=tools_from([fast_write, slow_op], include_done=True),
            state=state,
            conversation=conv,
            config=LoopConfig(
                max_steps=4,
                use_native_tools=True,
                concurrent_dispatch=True,
                max_parallel_calls=2,
                harness_profile="minimal",
            ),
        )
    )
    await asyncio.wait_for(slow_waiting.wait(), timeout=3.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mutated == ["committed"]
    assert len(state.steps) == 1
    assert state.steps[0].tool_call.tool == "fast_write"
    assert state.steps[0].tool_result.data == {"status": "committed"}
    assert any(
        m.tool_result is not None and m.tool_result.tool == "fast_write" for m in conv.messages
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("finish_reason", ["length", None])
async def test_openai_stream_never_emits_truncated_or_incomplete_tool_calls(
    asynchronous: bool, finish_reason: str | None
) -> None:
    raw_chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content="partial text",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(name="rm_file", arguments='{"path":"/tmp'),
                            )
                        ],
                    ),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    delta=SimpleNamespace(content=None, tool_calls=None),
                )
            ],
            usage=None,
        ),
    ]
    callbacks: list[dict[str, Any]] = []
    if asynchronous:

        async def create_async(**kw: Any) -> Any:
            async def gen():
                for ch in raw_chunks:
                    yield ch

            return gen()

        client = SimpleNamespace(
            max_retries=0, chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))
        )
        backend = AsyncOpenAIStreamingBackend(client, model="m")
        blocks = await backend.stream_with_tools(
            "go",
            tools=[{"name": "rm_file", "input_schema": {"type": "object"}}],
            on_tool_call=callbacks.append,
        )
    else:
        client = SimpleNamespace(
            max_retries=0,
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(raw_chunks))),
        )
        backend = OpenAIStreamingBackend(client, model="m")
        blocks = backend.stream_with_tools(
            "go",
            tools=[{"name": "rm_file", "input_schema": {"type": "object"}}],
            on_tool_call=callbacks.append,
        )

    assert callbacks == []
    assert [b for b in blocks if b["type"] == "tool_use"] == []
    assert backend.last_stop_reason == ("max_tokens" if finish_reason == "length" else "incomplete")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streaming_backends_preserve_conversation_input_in_stream(
    provider: str, asynchronous: bool
) -> None:
    captured_kwargs: list[dict[str, Any]] = []
    conv = Conversation(
        messages=[
            Message(role=MessageRole.USER, content="earlier context", metadata={"origin": "host"}),
        ]
    )
    snapshot = {"prompt": "Initial task", "conversation": conv.serialize()}

    if provider == "openai":
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        if asynchronous:

            async def oai_async(**kw: Any) -> Any:
                captured_kwargs.append(kw)

                async def gen():
                    yield chunk

                return gen()

            b_oa = AsyncOpenAIStreamingBackend(
                SimpleNamespace(
                    max_retries=0,
                    chat=SimpleNamespace(completions=SimpleNamespace(create=oai_async)),
                ),
                model="m",
            )
            b_oa.set_conversation_input(snapshot)
            out = "".join([c async for c in b_oa.stream("ignored fallback")])
            assert b_oa.last_stop_reason == "stop"
        else:
            b_os = OpenAIStreamingBackend(
                SimpleNamespace(
                    max_retries=0,
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(
                            create=lambda **kw: captured_kwargs.append(kw) or iter([chunk])
                        )
                    ),
                ),
                model="m",
            )
            b_os.set_conversation_input(snapshot)
            out = "".join(list(b_os.stream("ignored fallback")))
            assert b_os.last_stop_reason == "stop"
    else:
        if asynchronous:

            class AsyncAnthStream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

                @property
                async def text_stream(self):
                    yield "ok"

                async def get_final_message(self):
                    return SimpleNamespace(stop_reason="end_turn", model="claude", usage=None)

            b_aa = AsyncAnthropicBackend(
                SimpleNamespace(
                    max_retries=0,
                    messages=SimpleNamespace(
                        stream=lambda **kw: captured_kwargs.append(kw) or AsyncAnthStream()
                    ),
                ),
                model="claude",
            )
            b_aa.set_conversation_input(snapshot)
            out = "".join([c async for c in b_aa.stream("ignored fallback")])
        else:

            class SyncAnthStream:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                text_stream = iter(["ok"])

                def get_final_message(self):
                    return SimpleNamespace(stop_reason="end_turn", model="claude", usage=None)

            b_as = AnthropicStreamingBackend(
                SimpleNamespace(
                    max_retries=0,
                    messages=SimpleNamespace(
                        stream=lambda **kw: captured_kwargs.append(kw) or SyncAnthStream()
                    ),
                ),
                model="claude",
            )
            b_as.set_conversation_input(snapshot)
            out = "".join(list(b_as.stream("ignored fallback")))

    assert out == "ok"
    sent_json = json.dumps(captured_kwargs[0]["messages"])
    assert "earlier context" in sent_json


def test_error_projection_preserves_error_kind_retriable_and_execution_status() -> None:
    state = DefaultState()
    tr = ToolResult(
        tool="mutate_db",
        args_summary="id=42",
        data=None,
        error="timed out after 5s; effects unknown",
        error_detail=ToolError(kind=ErrorKind.TIMEOUT, message="timed out", retriable=False),
        result_key="sha256:" + "b" * 64,
        warnings=["connection pool degraded"],
        metadata={"execution_status": "effect_unknown"},
    )
    state.steps.append(
        Step(number=1, tool_call=ToolCall(tool="mutate_db", args={"id": 42}), tool_result=tr)
    )
    summary = state.context_summary()
    assert '"error_kind": "timeout"' in summary
    assert '"error_retriable": false' in summary
    assert '"execution_status": "effect_unknown"' in summary
    assert '"warnings": ["connection pool degraded"]' in summary


def test_file_run_store_and_provenance_deduplicate_prompts_and_message_blobs(
    tmp_path: Path,
) -> None:
    @tool(concurrent_safe=True)
    def fetch_item(idx: int) -> dict[str, Any]:
        return {"idx": idx, "body": "payload-" * 80}

    class TypedBackend:
        conversation_mode = "messages"
        step = 0

        def set_conversation_input(self, snap: Any) -> None:
            self._snap = snap

        def generate_with_tools(self, prompt: str, **kw: Any) -> list[dict[str, Any]]:
            self.step += 1
            if self.step < 15:
                return [
                    {
                        "type": "tool_use",
                        "id": f"c_{self.step}",
                        "name": "fetch_item",
                        "input": {"idx": self.step},
                    }
                ]
            return [
                {
                    "type": "tool_use",
                    "id": "done_1",
                    "name": "done",
                    "input": {"summary": "complete"},
                }
            ]

        def generate(self, prompt: str, **kw: Any) -> str:
            return ""

    trace_dir = tmp_path / "trace"
    store_dir = tmp_path / "store"
    sink = ProvenanceSink(trace_dir)
    preset = AgentPreset(
        state=DefaultState(max_steps=20),
        tools=tools_from([fetch_item], include_done=True),
        hooks=[sink.trajectory_hook()],
        config=LoopConfig(
            max_steps=20,
            use_native_tools=True,
            harness_profile="execution-v1",
            run_dir=str(tmp_path / "run"),
        ),
    )
    rt = AgentRuntime(preset, store=FileRunStore(store_dir))
    res = rt.run(sink.wrap_llm(TypedBackend()), task="fetch 14 items")
    assert res.completed
    sink.flush()

    # Stored RunEvents record prompt_chars + prompt_sha256 rather than raw prompt copies
    ev_file = next(store_dir.rglob("events.jsonl"))
    for line in ev_file.read_text(encoding="utf-8").splitlines():
        ev_obj = json.loads(line)
        if ev_obj["kind"] in {"pre_llm_call", "post_llm_response"}:
            assert "prompt" not in ev_obj["payload"]

    # Model attempts store content-addressed blobs and outcome request_ref="intent" on disk,
    # while load_model_attempts() rehydrates complete lossless dicts!
    raw_outcomes = list((trace_dir / "model-attempts").glob("*.outcome.json"))
    assert len(raw_outcomes) == 15
    disk_outcome = json.loads(raw_outcomes[0].read_text(encoding="utf-8"))
    assert disk_outcome["capture"]["request_ref"] == "intent"
    assert "request" not in disk_outcome["capture"]
    assert (trace_dir / "model-attempts" / "blobs").is_dir()

    attempts = load_model_attempts(trace_dir)
    assert len(attempts) == 15
    assert attempts[-1]["outcome"]["capture"]["request"] == attempts[-1]["intent"]["request"]
    assert (
        len(attempts[-1]["intent"]["request"]["conversation_input"]["conversation"]["messages"]) > 0
    )


def test_deep_nested_conversation_mutation_and_streaming_line_range_reader(
    tmp_path: Path,
) -> None:
    conv = Conversation(
        messages=[
            Message(
                role=MessageRole.TOOL,
                content="",
                tool_result=ToolResult(
                    tool="query",
                    args_summary="",
                    data={"outer": {"inner": {"val": 1}}},
                    call_id="q1",
                ),
            )
        ]
    )
    first = conv.serialize()
    assert first["messages"][0]["tool_result"]["data"]["outer"]["inner"]["val"] == 1
    # Mutate depth-3 nested dict value without changing container identity or length!
    assert conv.messages[0].tool_result is not None
    conv.messages[0].tool_result.data["outer"]["inner"]["val"] = 99
    second = conv.serialize()
    assert second["messages"][0]["tool_result"]["data"]["outer"]["inner"]["val"] == 99

    store = FileToolResultStore(tmp_path / "results")
    key = store.store_result("\n".join(f"row-{i}" for i in range(1, 51)))
    page = store.read_result(key, start_line=10, end_line=12, limit=500)
    assert page["content"] == "row-10\nrow-11\nrow-12\n"
    assert page["total_lines"] == 50
    assert page["start_line"] == 10 and page["end_line"] == 12


def test_cartridge_isolated_namespaces_concurrent_loads_and_roundtrip(
    tmp_path: Path,
) -> None:
    def write_cart(path: Path, name: str, tag: str) -> None:
        (path / "prompts").mkdir(parents=True)
        (path / "tools/ping").mkdir(parents=True)
        (path / "hooks/00_TagHook").mkdir(parents=True)
        (path / "cartridge.json").write_text(
            json.dumps({"name": name, "schema_version": 2}), encoding="utf-8"
        )
        (path / "config.yaml").write_text("max_steps: 2\ndone_tool: ping\n", encoding="utf-8")
        (path / "prompts/system.md").write_text("sys", encoding="utf-8")
        (path / "shared_helper.py").write_text(
            f"TAG = {tag!r}\nclass BaseTagHook:\n    def __init__(self, label='ok', tag=TAG):\n        self.label = label\n        self.tag = tag\n    def to_config(self):\n        return {{'label': self.label, 'tag': self.tag}}\n",
            encoding="utf-8",
        )
        (path / "hooks/00_TagHook/config.yaml").write_text("label: ok\n", encoding="utf-8")
        (path / "hooks/00_TagHook/hook.py").write_text(
            "from shared_helper import BaseTagHook\nclass TagHook(BaseTagHook):\n    pass\n",
            encoding="utf-8",
        )
        (path / "tools/ping/tool.yaml").write_text(
            "name: ping\ndescription: ping\nparameters: {}\n", encoding="utf-8"
        )
        (path / "tools/ping/execute.py").write_text(
            "import shared_helper as eager_mod\n"
            "def execute(ctx) -> dict:\n"
            "    import shared_helper as lazy_mod\n"
            "    return {'eager': eager_mod.TAG, 'lazy': lazy_mod.TAG}\n",
            encoding="utf-8",
        )

    ca = tmp_path / "a.cartridge"
    cb = tmp_path / "b.cartridge"
    write_cart(ca, "agent_a", "ALPHA")
    write_cart(cb, "agent_b", "BETA")

    orig_path = list(sys.path)
    orig_meta = list(sys.meta_path)
    loaded: dict[str, AgentPreset] = {}

    def load_worker(key: str, root: Path) -> None:
        loaded[key] = cartridge_to_preset(root, strict=True)

    t1 = threading.Thread(target=load_worker, args=("a", ca))
    t2 = threading.Thread(target=load_worker, args=("b", cb))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert sys.path == orig_path
    assert sys.meta_path == orig_meta
    assert "shared_helper" not in sys.modules
    assert loaded["a"].tools.dispatch(ToolCall(tool="ping", args={})).data == {
        "eager": "ALPHA",
        "lazy": "ALPHA",
    }
    assert loaded["b"].tools.dispatch(ToolCall(tool="ping", args={})).data == {
        "eager": "BETA",
        "lazy": "BETA",
    }

    # Round-tripping preset_a STILL succeeds even after preset_b loaded with the same helper name!
    exported = tmp_path / "a_roundtrip.cartridge"
    preset_to_cartridge(loaded["a"], exported, strict=True)
    reloaded_a = cartridge_to_preset(exported, strict=True)
    assert reloaded_a.tools.dispatch(ToolCall(tool="ping", args={})).data == {
        "eager": "ALPHA",
        "lazy": "ALPHA",
    }


def test_mcp_chatty_stderr_does_not_deadlock_and_timeout_invalidates_process(
    tmp_path: Path,
) -> None:
    server = tmp_path / "chatty_mcp.py"
    server.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    # Write 256 KB to stderr BEFORE replying on stdout!\n"
        "    sys.stderr.write('E' * (256 * 1024) + '\\n')\n"
        "    sys.stderr.flush()\n"
        "    m = msg.get('method')\n"
        "    if m == 'initialize':\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'capabilities': {}}}) + '\\n')\n"
        "    elif m == 'tools/list':\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'tools': [{'name': 'echo', 'description': 'd', 'inputSchema': {'type': 'object'}}]}}) + '\\n')\n"
        "    elif m == 'tools/call':\n"
        "        if msg['params']['arguments'].get('hang'):\n"
        "            import time; time.sleep(5)\n"
        "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'content': [{'type': 'text', 'text': 'pong'}], 'isError': False}}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    with MCPToolAdapter([sys.executable, "-I", "-B", str(server)], timeout=0.3) as adapter:
        specs = adapter.tools()
        assert len(specs) == 1
        assert specs[0].execute() == {"text": "pong"}
        with pytest.raises(TimeoutError):
            adapter._send_request("tools/call", {"name": "echo", "arguments": {"hang": True}})
        # Timed-out server was closed and invalidated, not left live!
        assert adapter._proc is None and not adapter._started


def test_lep_timeout_and_state_service_frame_bounds(tmp_path: Path) -> None:
    import socket

    hung_lep = tmp_path / "hung_lep.py"
    hung_lep.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    hook = LEPHookAdapter(
        [sys.executable, "-I", "-B", str(hung_lep)],
        on_failure="fail_closed",
        timeout=0.15,
    )
    t0 = time.monotonic()
    hook.pre_loop(DefaultState(), None, None)
    assert hook.check_permission(ToolCall(tool="write_file", args={}), DefaultState()) is False
    assert time.monotonic() - t0 < 2.0
    hook.close()

    a, b = socket.socketpair()
    try:
        reader = _LineReader(a, max_message_bytes=32)
        b.sendall(b"x" * 100 + b"\n")
        with pytest.raises(StateServiceError, match="max_message_bytes"):
            reader.readline()
    finally:
        a.close()
        b.close()


def test_concurrent_permission_approvals_tracer_spans_and_backend_call_locals() -> None:
    engine = PermissionEngine(
        default=PermissionDecision.ALLOW,
        ask_handler=lambda tc, reason: PermissionDecision.ALLOW,
    )
    engine.ask("write_file")
    perm_hook = PermissionHook(engine)
    c1 = ToolCall(tool="write_file", args={"path": "a.txt"}, call_id="call_a")
    c2 = ToolCall(tool="write_file", args={"path": "b.txt"}, call_id="call_b")

    # Interleave on_event for both calls BEFORE check_permission runs!
    perm_hook.on_event(EventPayload(event=LifecycleEvent.PRE_TOOL_USE, tool_call=c1))
    perm_hook.on_event(EventPayload(event=LifecycleEvent.PRE_TOOL_USE, tool_call=c2))
    assert perm_hook.check_permission(c1, DefaultState()) is True
    assert perm_hook.check_permission(c2, DefaultState()) is True

    # Concurrent async tasks using one Tracer keep independent span stacks
    tracer = Tracer()

    async def trace_branch(name: str) -> tuple[str, list[str]]:
        root = tracer.start_span(f"root.{name}")
        await asyncio.sleep(0.01)
        child = tracer.start_span(f"child.{name}")
        await asyncio.sleep(0.01)
        tracer.end_span(child)
        tracer.end_span(root)
        return root.name, [c.name for c in root.children]

    async def run_branches() -> list[tuple[str, list[str]]]:
        return list(await asyncio.gather(trace_branch("left"), trace_branch("right")))

    branches = asyncio.run(run_branches())
    assert branches == [("root.left", ["child.left"]), ("root.right", ["child.right"])]

    # Concurrent threads calling one OpenAIBackend see isolated last_usage
    barrier = threading.Barrier(2, timeout=5)

    def create_chat(**kw: Any) -> Any:
        barrier.wait()
        tok = 11 if kw["messages"][-1]["content"] == "p1" else 99
        return SimpleNamespace(
            model="m",
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="ok", tool_calls=None)
                )
            ],
            usage=SimpleNamespace(prompt_tokens=tok, completion_tokens=1, total_tokens=tok + 1),
        )

    backend = OpenAIBackend(
        SimpleNamespace(
            max_retries=0, chat=SimpleNamespace(completions=SimpleNamespace(create=create_chat))
        ),
        model="m",
    )
    observed: dict[str, int] = {}

    def worker(p: str) -> None:
        backend.generate(p)
        observed[p] = backend.last_usage["input"]

    th1 = threading.Thread(target=worker, args=("p1",))
    th2 = threading.Thread(target=worker, args=("p2",))
    th1.start()
    th2.start()
    th1.join(timeout=5)
    th2.join(timeout=5)
    assert observed == {"p1": 11, "p2": 99}


def test_sync_reactive_recovery_resets_flags_and_blueprints_redact_secrets() -> None:
    # Verify sync loop can recover from TWO separate context overflows in one run!
    overflow_turns = {1, 3}
    calls = 0

    class TwoOverflowBackend:
        def generate(self, prompt: str, **kw: Any) -> str:
            nonlocal calls
            calls += 1
            if calls in overflow_turns:
                raise RuntimeError("maximum context length exceeded: prompt is too long")
            if calls < 5:
                return json.dumps({"tool": "step_op", "args": {}})
            return json.dumps({"tool": "done", "args": {"summary": "recovered twice"}})

    @tool
    def step_op() -> dict[str, str]:
        return {"data": "x" * 200}

    res = run_result(
        TwoOverflowBackend(),
        task="survive two overflows",
        tools=tools_from([step_op], include_done=True),
        config=LoopConfig(
            max_steps=5,
            use_native_tools=False,
            reactive_recovery=True,
            harness_profile="minimal",
        ),
    )
    assert res.completed and res.output["summary"] == "recovered twice"

    # Blueprints use explicit provenance_config/to_config and omit secret-named fields
    class CustomSecretHook:
        def __init__(self) -> None:
            self.threshold = 5
            self.api_key = "sk-super-secret-value"
            self.webhook_token = "tok-secret"

        def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> None:
            return None

    preset = minimal_preset()
    preset.hooks.append(CustomSecretHook())
    bp = blueprint_from_preset(preset, name="safe")
    hook_cfg = bp.hooks[-1].config
    assert hook_cfg == {"threshold": 5}
    assert "sk-super-secret-value" not in json.dumps(bp.to_dict())


def test_lazy_root_import_loads_only_package_root() -> None:
    out = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import json, sys; import looplet; "
            "mods = [m for m in sys.modules if m == 'looplet' or m.startswith('looplet.')]; "
            "print(json.dumps({'mods': mods, 'version': looplet.__version__}))",
        ],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
    )
    parsed = json.loads(out)
    assert parsed["mods"] == ["looplet"]
    assert parsed["version"] == "0.5.0.dev0"
