"""Wave-3 tests for prefix stability, O(N) persistence, hot-window storage, zero-thread RPC, preset.fork(), observer-compatible speculative streaming, and Sans-I/O turn engine parity."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from contextlib import aclosing, closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from looplet import (
    AgentPreset,
    AgentRuntime,
    CachePolicy,
    Conversation,
    DefaultState,
    FileRunStore,
    FileToolResultStore,
    LoopConfig,
    Message,
    MessageRole,
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    ProvenanceSink,
    StreamingHook,
    ToolCall,
    ToolResult,
    ToolSpec,
    async_composable_loop,
    cartridge_to_preset,
    composable_loop,
    run_result,
    run_result_async,
    scaffold_cartridge,
    tool,
    tools_from,
)
from looplet._backend_contract import json_snapshot
from looplet._storage import DiskBackedDict, DiskBackedList
from looplet.backends import (
    AnthropicBackend,
    AsyncOpenAIStreamingBackend,
    OpenAIBackend,
    OpenAIStreamingBackend,
    _conversation_messages,
)
from looplet.cache import CacheBreakDetector
from looplet.lep import LEPHookAdapter
from looplet.mcp import MCPToolAdapter
from looplet.provenance import _redact_value, load_model_attempts
from looplet.streaming import CallbackEmitter, LLMChunkEvent

# ── Pillar 1: Default Prefix Stability & Clean OpenAI Tool Tail ─────


def test_default_typed_conversation_keeps_first_user_message_prefix_stable() -> None:
    sent_messages: list[list[dict[str, Any]]] = []

    def create(**kwargs: Any) -> Any:
        sent_messages.append(json_snapshot(kwargs["messages"]))
        turn = len(sent_messages)
        if turn < 4:
            tc = SimpleNamespace(
                id=f"call_{turn}",
                type="function",
                function=SimpleNamespace(name="step_op", arguments=json.dumps({"n": turn})),
            )
            return SimpleNamespace(
                model="gpt-test",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(content=None, tool_calls=[tc]),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
            )
        tc = SimpleNamespace(
            id="call_done",
            type="function",
            function=SimpleNamespace(name="done", arguments='{"summary": "stable"}'),
        )
        return SimpleNamespace(
            model="gpt-test",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(content=None, tool_calls=[tc]),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        )

    class CustomFactsState(DefaultState):
        def snapshot(self) -> dict[str, Any]:
            base = super().snapshot()
            if self.step_count >= 1:
                base["findings"] = [{"description": f"fact-at-{self.step_count}", "step": 1}]
            return base

    class BriefingHook:
        def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> str:
            return f"Guidance-{step_num}"

    @tool(concurrent_safe=True)
    def step_op(n: int) -> dict[str, int]:
        return {"n": n}

    llm = OpenAIBackend(
        SimpleNamespace(
            max_retries=0, chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="gpt-test",
    )
    # Note: cache_policy is None and harness_profile is default ("execution-v1")!
    state = CustomFactsState(max_steps=4)
    res = run_result(
        llm,
        task="Verify default prefix stability",
        tools=tools_from([step_op], include_done=True),
        state=state,
        hooks=[BriefingHook()],
        config=LoopConfig(
            max_steps=4,
            disabled_defaults=["persistence"],
        ),
    )
    assert res.completed and res.output is not None and res.output["summary"] == "stable"
    assert len(sent_messages) == 4
    first_msg = sent_messages[0][0]
    assert all(turn_msgs[0] == first_msg for turn_msgs in sent_messages)
    # Dynamic facts, step briefings, and step counters are delivered in the tail turn_update!
    tail_turn_3 = sent_messages[2][-1]["content"]
    assert "fact-at-2" in tail_turn_3
    assert "Guidance-3" in tail_turn_3
    assert "Step 3/4." in tail_turn_3

    # Explicit opt-out CachePolicy(stable_prefix=False) restores legacy inline step rendering
    sent_opt_out: list[list[dict[str, Any]]] = []
    llm_legacy = OpenAIBackend(
        SimpleNamespace(
            max_retries=0,
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: (
                        sent_opt_out.append(json_snapshot(kw["messages"])) or create(**kw)
                    )
                )
            ),
        ),
        model="gpt-test",
    )
    sent_messages.clear()
    res2 = run_result(
        llm_legacy,
        task="Opt-out prefix stability",
        tools=tools_from([step_op], include_done=True),
        config=LoopConfig(
            max_steps=4,
            cache_policy=CachePolicy(stable_prefix=False),
            disabled_defaults=["persistence"],
        ),
    )
    assert res2.completed
    assert sent_opt_out[0][0] != sent_opt_out[1][0]


def test_openai_tool_role_tail_omits_synthetic_user_message_while_assistant_tail_keeps_it() -> None:
    tool_tail_snap = {
        "prompt": "Task",
        "conversation": {
            "messages": [
                {
                    "role": "assistant",
                    "metadata": {"origin": "model"},
                    "content": [
                        {"kind": "tool_use", "data": {"id": "c1", "name": "op", "input": {}}}
                    ],
                },
                {
                    "role": "tool",
                    "tool_result": {"tool": "op", "call_id": "c1", "data": {"ok": 1}},
                },
            ]
        },
    }
    oai_msgs = _conversation_messages(
        SimpleNamespace(_conversation_input=tool_tail_snap), "Task", native_blocks=False
    )
    assert oai_msgs[-1]["role"] == "tool"
    assert not any(m.get("content") == "Continue the task." for m in oai_msgs)

    # Assistant-ending conversation still appends "Continue the task."
    asst_tail_snap = {
        "prompt": "Task",
        "conversation": {
            "messages": [
                {"role": "assistant", "metadata": {"origin": "model"}, "content": "Thinking..."}
            ]
        },
    }
    oai_asst = _conversation_messages(
        SimpleNamespace(_conversation_input=asst_tail_snap), "Task", native_blocks=False
    )
    assert oai_asst[-1] == {"role": "user", "content": "Continue the task."}


# ── Pillar 2 & 3: Provenance Tool/Provider Blobs, Fast Snapshot, & Incremental Caches ──


def test_provenance_deduplicates_tool_and_provider_blobs_and_rehydrates_losslessly(
    tmp_path: Path,
) -> None:
    @tool(concurrent_safe=True)
    def lookup(item_id: int) -> dict[str, Any]:
        """Look up a structured item record with full metadata."""
        return {"item_id": item_id, "tags": ["a", "b", "c"]}

    turn = 0

    def create(**kw: Any) -> Any:
        nonlocal turn
        turn += 1
        if turn < 6:
            tc = SimpleNamespace(
                id=f"tc_{turn}",
                type="function",
                function=SimpleNamespace(name="lookup", arguments=json.dumps({"item_id": turn})),
            )
            return SimpleNamespace(
                model="m",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(content=None, tool_calls=[tc]),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5, total_tokens=35),
            )
        tc = SimpleNamespace(
            id="tc_done",
            type="function",
            function=SimpleNamespace(name="done", arguments='{"summary": "all_fetched"}'),
        )
        return SimpleNamespace(
            model="m",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(content=None, tool_calls=[tc]),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5, total_tokens=35),
        )

    trace_dir = tmp_path / "trace"
    sink = ProvenanceSink(trace_dir)
    backend = OpenAIBackend(
        SimpleNamespace(
            max_retries=0, chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        model="m",
    )
    res = run_result(
        sink.wrap_llm(backend),
        task="Fetch 5 items",
        tools=tools_from([lookup], include_done=True),
        hooks=[sink.trajectory_hook()],
        config=LoopConfig(max_steps=8, disabled_defaults=["persistence"]),
    )
    assert res.completed
    sink.flush()

    # On disk: intent & outcome store compact SHA-256 refs rather than inline tool/message arrays!
    intent_files = sorted((trace_dir / "model-attempts").glob("*.intent.json"))
    outcome_files = sorted((trace_dir / "model-attempts").glob("*.outcome.json"))
    assert len(intent_files) == 6 and len(outcome_files) == 6
    raw_last_intent = json.loads(intent_files[-1].read_text(encoding="utf-8"))
    assert "tools_blob_ref" in raw_last_intent["request"]
    assert "tools" not in raw_last_intent["request"]
    raw_last_outcome = json.loads(outcome_files[-1].read_text(encoding="utf-8"))
    assert raw_last_outcome["capture"]["request_ref"] == "intent"
    assert "tools_blob_ref" in raw_last_outcome["capture"]["provider_request"]
    assert "message_blob_refs" in raw_last_outcome["capture"]["provider_request"]

    # Rehydrated attempts restore complete tools, conversation messages, and provider_request!
    attempts = load_model_attempts(trace_dir)
    assert len(attempts) == 6
    for idx, att in enumerate(attempts):
        assert isinstance(att["intent"]["request"]["tools"], list)
        assert len(att["intent"]["request"]["tools"]) == 2
        assert att["outcome"]["capture"]["request"] == att["intent"]["request"]
        preq = att["outcome"]["capture"]["provider_request"]
        assert isinstance(preq["tools"], list) and len(preq["tools"]) == 2
        assert isinstance(preq["messages"], list) and len(preq["messages"]) >= 1 + 2 * idx


def test_fast_json_snapshot_redact_noop_and_result_store_checkpoint_fast_path(
    tmp_path: Path,
) -> None:
    payload = {
        "a": (1, 2, {"nested": True, None: "null_key", 7: "seven"}),
        "ns": SimpleNamespace(x=[10, 20]),
    }
    snap = json_snapshot(payload)
    assert snap == {
        "a": [1, 2, {"nested": True, "null": "null_key", "7": "seven"}],
        "ns": {"x": [10, 20]},
    }
    assert _redact_value(snap, None) is snap

    # FileToolResultStore.checkpoint_results() reuses cached reference when catalog is unchanged,
    # yet still detects external file truncation!
    rs = FileToolResultStore(str(tmp_path / "results"))
    rs.store_result({"k": "v1"})
    ref1 = rs.checkpoint_results()
    ref2 = rs.checkpoint_results()
    assert ref1 == ref2 and ref1 is not ref2
    # Truncating the catalog log invalidates the size check and raises ValueError on next append
    log_path = (
        tmp_path / "results" / ".snapshots" / ref1["__looplet_file_result_store_v1__"]["catalog"]
    )
    log_path.write_bytes(b"x")
    with pytest.raises(ValueError, match="length changed"):
        rs.checkpoint_results()


# ── Pillar 4: Hot-Window Storage, Spill-to-SQLite, & Sharded Concurrent RunStore ──


def test_disk_backed_list_and_dict_hot_window_and_seamless_spill() -> None:
    # 1. Small items stay in-memory (_mem is not None) with zero SQLite queries
    dl: DiskBackedList[dict[str, Any]] = DiskBackedList(dict, dict, [{"i": i} for i in range(10)])
    assert dl._mem is not None
    dl[0]["mutated"] = True
    assert dl[1] == {"i": 1}
    assert dl[0] == {"i": 0, "mutated": True}
    cloned = dl.clone()
    assert cloned._mem is not None and cloned == dl
    dl.insert(0, {"i": -1})
    assert dl[0] == {"i": -1}
    del dl[0]
    dl.sort(key=lambda r: -r["i"])
    assert dl[0]["i"] == 9

    # 2. Exceeding 64 KiB spills seamlessly to SQLite spool while preserving all data!
    dl.append({"i": 999, "big": "x" * 70_000})
    assert dl._mem is None
    assert len(dl) == 11
    assert dl[-1]["i"] == 999 and len(dl[-1]["big"]) == 70_000
    spilled_clone = dl.clone()
    assert spilled_clone._mem is None and len(spilled_clone) == 11
    assert spilled_clone[0]["i"] == 9

    # 3. DiskBackedDict hot-window transaction rollback AND mid-transaction spill rollback
    dd: DiskBackedDict[dict[str, Any]] = DiskBackedDict(dict, dict)
    dd["keep"] = {"v": 1}
    assert dd._mem is not None
    with pytest.raises(RuntimeError, match="abort_hot"):
        with dd.transaction():
            dd["drop_hot"] = {"v": 2}
            raise RuntimeError("abort_hot")
    assert list(dd.keys()) == ["keep"]

    with pytest.raises(RuntimeError, match="abort_spill"):
        with dd.transaction():
            dd["huge"] = {"blob": "y" * 80_000}
            assert dd._mem is None
            raise RuntimeError("abort_spill")
    assert list(dd.keys()) == ["keep"]
    assert dd["keep"] == {"v": 1}


async def test_concurrent_async_runtimes_with_sharded_file_run_store_and_flush_async(
    tmp_path: Path,
) -> None:
    @tool(concurrent_safe=True)
    async def async_calc(v: int) -> dict[str, int]:
        await asyncio.sleep(0.001)
        return {"out": v * 2}

    class FastAsyncBackend:
        conversation_mode = "messages"

        def __init__(self, idx: int) -> None:
            self.idx = idx
            self.turn = 0

        def set_conversation_input(self, snap: Any) -> None:
            self._snap = snap

        async def generate_with_tools(self, prompt: str, **kw: Any) -> list[dict[str, Any]]:
            self.turn += 1
            if self.turn <= 3:
                return [
                    {
                        "type": "tool_use",
                        "id": f"c_{self.idx}_{self.turn}",
                        "name": "async_calc",
                        "input": {"v": self.idx + self.turn},
                    }
                ]
            return [
                {
                    "type": "tool_use",
                    "id": f"done_{self.idx}",
                    "name": "done",
                    "input": {"summary": f"done-{self.idx}"},
                }
            ]

        async def generate(self, prompt: str, **kw: Any) -> str:
            return ""

    template = AgentPreset(
        config=LoopConfig(max_steps=6, use_native_tools=True, harness_profile="execution-v1"),
        hooks=[],
        tools=tools_from([async_calc], include_done=True),
        state=DefaultState(max_steps=6),
    )
    store = FileRunStore(tmp_path / "store")

    async def worker(i: int) -> str:
        forked = template.fork()
        sink = ProvenanceSink(tmp_path / f"trace_{i}")
        forked.hooks.append(sink.trajectory_hook())
        forked.config.run_dir = str(tmp_path / f"run_{i}")
        rt = AgentRuntime(forked, store=store)
        res = await rt.run_async(sink.wrap_llm(FastAsyncBackend(i), async_=True), task=f"job-{i}")
        await sink.flush_async()
        assert res.completed and res.run_envelope is not None and res.output is not None
        rec = store.load(res.run_envelope.run_id)
        assert rec is not None and rec.status == "completed" and len(rec.events) > 0
        return res.output["summary"]

    summaries = await asyncio.gather(*(worker(i) for i in range(16)))
    assert summaries == [f"done-{i}" for i in range(16)]


# ── Pillar 5: Zero Per-Call Thread Spawning in MCPToolAdapter & LEPHookAdapter ──


def test_mcp_and_lep_multiplex_over_persistent_reader_with_zero_per_call_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_srv = tmp_path / "mcp_srv.py"
    mcp_srv.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    m = json.loads(line)\n"
        "    method = m.get('method')\n"
        "    if method == 'initialize': r = {'capabilities': {}}\n"
        "    elif method == 'tools/list': r = {'tools': [{'name': 'echo', 'description': 'e', 'inputSchema': {'type': 'object', 'properties': {'v': {'type': 'integer'}}}}]}\n"
        "    elif method == 'tools/call': r = {'content': [{'type': 'text', 'text': str(m['params']['arguments']['v'] * 3)}], 'isError': False}\n"
        "    else: continue\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': r}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )
    lep_srv = tmp_path / "lep_srv.py"
    lep_srv.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    m = json.loads(line)\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': {'effect': {'kind': 'Allow'}}}) + '\\n')\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8",
    )

    started_threads: list[str] = []
    orig_start = threading.Thread.start

    def track_start(self: threading.Thread) -> None:
        started_threads.append(self.name)
        orig_start(self)

    with MCPToolAdapter([sys.executable, "-I", "-B", str(mcp_srv)], timeout=5.0) as mcp:
        spec = mcp.tools()[0]
        lep = LEPHookAdapter([sys.executable, "-I", "-B", str(lep_srv)], timeout=5.0)
        lep.pre_loop(DefaultState(), None, None)
        try:
            monkeypatch.setattr(threading.Thread, "start", track_start)
            for idx in range(40):
                assert spec.execute(v=idx) == idx * 3
                assert lep.check_permission(ToolCall(tool="echo", args={"v": idx}), DefaultState())
            assert started_threads == []
        finally:
            monkeypatch.undo()
            lep.close()


# ── Pillar 6: AgentPreset.fork() Resource Ownership & Speculative Streaming with Observers ──


def test_preset_fork_shares_parent_resources_without_closing_parent_on_child_close(
    tmp_path: Path,
) -> None:
    cart_dir = tmp_path / "svc.cartridge"
    scaffold_cartridge(cart_dir, name="svc_agent", tools=["read_pool"])
    (cart_dir / "resources").mkdir(exist_ok=True)
    (cart_dir / "resources" / "pool.py").write_text(
        "class SharedPool:\n"
        "    def __init__(self):\n"
        "        self.closed = False\n"
        "    def close(self):\n"
        "        self.closed = True\n"
        "def build(runtime=None):\n"
        "    return SharedPool()\n",
        encoding="utf-8",
    )
    (cart_dir / "tools" / "read_pool" / "tool.yaml").write_text(
        "name: read_pool\ndescription: Check pool.\nparameters: {}\nrequires:\n  - pool\n",
        encoding="utf-8",
    )
    (cart_dir / "tools" / "read_pool" / "execute.py").write_text(
        "def execute(ctx) -> dict:\n    return {'closed': ctx.resources['pool'].closed}\n",
        encoding="utf-8",
    )

    parent = cartridge_to_preset(cart_dir, strict=True)
    try:
        for i in range(5):
            with parent.fork() as child:
                turns = [
                    [{"type": "tool_use", "id": f"r{i}", "name": "read_pool", "input": {}}],
                    [
                        {
                            "type": "tool_use",
                            "id": f"d{i}",
                            "name": "done",
                            "input": {"summary": "ok"},
                        }
                    ],
                ]
                idx = 0

                class Stub:
                    def generate(self, prompt: str, **_kw: Any) -> str:
                        return ""

                    def generate_with_tools(self, prompt: str, **_kw: Any) -> list[dict[str, Any]]:
                        nonlocal idx
                        out = turns[idx]
                        idx += 1
                        return out

                res = child.run_result(
                    Stub(),
                    task="check pool",
                )
                assert res.completed
                assert res.steps[0].tool_result.data == {"closed": False}
            # Child exited `with` (called child.close()), yet parent pool remains open!
            assert parent.resources["pool"].closed is False
    finally:
        parent.close()
    assert parent.resources["pool"].closed is True


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_speculative_streaming_overlaps_with_observer_hooks_and_blocks_on_permission_hook(
    asynchronous: bool, tmp_path: Path
) -> None:
    tool_started = threading.Event()
    turn_no = 0

    def make_chunks(t: int) -> list[Any]:
        if t == 1:
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content="Planning...",
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="c_fast_1",
                                        function=SimpleNamespace(
                                            name="fast_op", arguments='{"v": 5}'
                                        ),
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
                            finish_reason="tool_calls",
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=1,
                                        id="c_fast_2",
                                        function=SimpleNamespace(
                                            name="fast_op", arguments='{"v": 6}'
                                        ),
                                    )
                                ],
                            ),
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                ),
            ]
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="c_done",
                                    function=SimpleNamespace(
                                        name="done", arguments='{"summary": "overlapped"}'
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=25, completion_tokens=5, total_tokens=30),
            )
        ]

    observed_during_stream: list[bool] = []

    def create_sync(**_kw: Any) -> Any:
        nonlocal turn_no
        turn_no += 1
        chunks = make_chunks(turn_no)

        def gen():
            yield chunks[0]
            if turn_no == 1:
                yield chunks[1]
                observed_during_stream.append(tool_started.wait(timeout=0.3))

        return gen()

    async def create_async(**_kw: Any) -> Any:
        nonlocal turn_no
        turn_no += 1
        chunks = make_chunks(turn_no)

        async def agen():
            yield chunks[0]
            if turn_no == 1:
                yield chunks[1]
                await asyncio.sleep(0.05)
                observed_during_stream.append(tool_started.is_set())

        return agen()

    @tool(concurrent_safe=True)
    def fast_op(v: int) -> dict[str, int]:
        tool_started.set()
        return {"val": v * 10}

    # 1. With StreamingHook + TrajectoryRecorder + CacheBreakDetector: speculative overlap IS active!
    sink = ProvenanceSink(tmp_path / f"obs_{asynchronous}")
    emitted: list[Any] = []
    cache_pol = CachePolicy.stable()
    obs_hooks = [
        StreamingHook(CallbackEmitter(emitted.append)),
        sink.trajectory_hook(),
        CacheBreakDetector(cache_pol),
    ]
    cfg = LoopConfig(
        max_steps=5,
        concurrent_dispatch=True,
        streaming_tool_execution=True,
        cache_policy=cache_pol,
        disabled_defaults=["persistence"],
    )
    client = SimpleNamespace(
        max_retries=0,
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_async if asynchronous else create_sync)
        ),
    )
    llm = (
        AsyncOpenAIStreamingBackend(client, model="m")
        if asynchronous
        else OpenAIStreamingBackend(client, model="m")
    )
    if asynchronous:
        res = await run_result_async(
            sink.wrap_llm(llm, async_=True),
            task="overlap with observers",
            tools=tools_from([fast_op], include_done=True),
            hooks=obs_hooks,
            config=cfg,
        )
        await sink.flush_async()
    else:
        res = run_result(
            sink.wrap_llm(llm),
            task="overlap with observers",
            tools=tools_from([fast_op], include_done=True),
            hooks=obs_hooks,
            config=cfg,
        )
        sink.flush()

    assert res.completed and observed_during_stream == [True]
    assert [e.chunk for e in emitted if isinstance(e, LLMChunkEvent)] == ["Planning..."]
    traj = json.loads((tmp_path / f"obs_{asynchronous}" / "trajectory.json").read_text())
    assert len(traj["steps"]) == 3

    # 2. With StreamingHook + PermissionHook denying fast_op: wire streaming runs,
    #    yet speculative execution MUST NOT bypass PermissionHook!
    tool_started.clear()
    turn_no = 0
    observed_during_stream.clear()
    deny_engine = PermissionEngine(default=PermissionDecision.ALLOW).deny(
        "fast_op", reason="blocked"
    )
    denied_hooks = [
        StreamingHook(CallbackEmitter(lambda _ev: None)),
        PermissionHook(deny_engine),
    ]
    llm_denied = (
        AsyncOpenAIStreamingBackend(client, model="m")
        if asynchronous
        else OpenAIStreamingBackend(client, model="m")
    )
    if asynchronous:
        async with aclosing(
            async_composable_loop(
                llm_denied,
                task="deny fast_op",
                tools=tools_from([fast_op], include_done=True),
                hooks=denied_hooks,
                config=cfg,
            )
        ) as asteps:
            first_step = await anext(asteps)
    else:
        with closing(
            composable_loop(
                llm_denied,
                task="deny fast_op",
                tools=tools_from([fast_op], include_done=True),
                hooks=denied_hooks,
                config=cfg,
            )
        ) as ssteps:
            first_step = next(ssteps)

    assert not tool_started.is_set()
    assert observed_during_stream == [False]
    assert (
        first_step.tool_result.error is not None
        and "Permission denied" in first_step.tool_result.error
    )


# ── Pillar 7: Sans-I/O TurnEngine Sync / Async Behavioral Equivalence & Fork Hook Isolation ──


async def test_sans_io_turn_engine_sync_and_async_behavioral_equivalence_and_early_close() -> None:
    from looplet.stagnation import StagnationHook

    @tool
    def bump(n: int) -> dict[str, int]:
        return {"added": n}

    class ScoreGateHook:
        def check_done(
            self, state: Any, session_log: Any, context: Any, step_num: int
        ) -> str | None:
            total = sum(
                int(s.tool_result.data.get("added", 0))
                for s in state.steps
                if isinstance(s.tool_result.data, dict)
            )
            if total < 10:
                return f"Score ({total}) must reach at least 10 before finishing."
            return None

    script = [
        [{"type": "tool_use", "id": "t1", "name": "bump", "input": {"n": 4}}],
        # Premature done -> rejected by check_done -> loop must continue!
        [{"type": "tool_use", "id": "t2", "name": "done", "input": {"summary": "too early"}}],
        [{"type": "tool_use", "id": "t3", "name": "bump", "input": {"n": 7}}],
        [{"type": "tool_use", "id": "t4", "name": "done", "input": {"summary": "valid"}}],
    ]

    class SyncScriptLLM:
        def __init__(self) -> None:
            self.idx = 0

        def generate(self, prompt: str, **_kw: Any) -> str:
            return ""

        def generate_with_tools(self, prompt: str, **_kw: Any) -> list[dict[str, Any]]:
            out = script[self.idx]
            self.idx += 1
            return out

    class AsyncScriptLLM:
        def __init__(self) -> None:
            self.idx = 0

        async def generate(self, prompt: str, **_kw: Any) -> str:
            return ""

        async def generate_with_tools(self, prompt: str, **_kw: Any) -> list[dict[str, Any]]:
            out = script[self.idx]
            self.idx += 1
            return out

    cfg = LoopConfig(max_steps=6, disabled_defaults=["persistence"])
    sync_res = run_result(
        SyncScriptLLM(),
        task="reach 10",
        tools=tools_from([bump], include_done=True),
        hooks=[ScoreGateHook()],
        config=cfg,
    )
    async_res = await run_result_async(
        AsyncScriptLLM(),
        task="reach 10",
        tools=tools_from([bump], include_done=True),
        hooks=[ScoreGateHook()],
        config=cfg,
    )

    assert sync_res.completed and async_res.completed
    assert sync_res.output == async_res.output == {"status": "completed", "summary": "valid"}
    assert [
        (s.tool_call.tool, s.tool_result.data, s.tool_result.error) for s in sync_res.steps
    ] == [(s.tool_call.tool, s.tool_result.data, s.tool_result.error) for s in async_res.steps]
    assert "must reach at least 10" in (sync_res.steps[1].tool_result.error or "")

    # Verify early generator close / aclose fires on_loop_end deterministically on both drivers
    closed_events: list[str] = []

    class LifecycleProbe:
        def on_loop_end(self, state: Any, session_log: Any, context: Any, llm: Any) -> None:
            closed_events.append(f"closed:{len(state.steps)}")

    with closing(
        composable_loop(
            SyncScriptLLM(),
            task="early stop",
            tools=tools_from([bump], include_done=True),
            hooks=[LifecycleProbe()],
            config=cfg,
        )
    ) as s_gen:
        _ = next(s_gen)

    async with aclosing(
        async_composable_loop(
            AsyncScriptLLM(),
            task="early stop",
            tools=tools_from([bump], include_done=True),
            hooks=[LifecycleProbe()],
            config=cfg,
        )
    ) as a_gen:
        _ = await anext(a_gen)

    assert closed_events == ["closed:1", "closed:1"]

    # Also verify AgentPreset.fork() clones StagnationHook with isolated streak state
    from looplet.presets import AgentPreset

    stag = StagnationHook(threshold=2, fingerprint="tool_call", ignore_tools={"done"})
    parent_preset = AgentPreset(
        config=cfg,
        hooks=[stag],
        tools=tools_from([bump], include_done=True),
        state=DefaultState(),
    )
    child_preset = parent_preset.fork()
    child_stag = next(h for h in child_preset.hooks if isinstance(h, StagnationHook))
    assert child_stag is not stag
    assert child_stag._threshold == 2
    assert child_stag._ignore_tools == {"done"}


# ── Review Hardening Regressions ──────────────────────────────────────────────


async def test_async_speculative_streaming_cancellation_discards_and_drains_inflight_tasks() -> (
    None
):
    tool_entered = asyncio.Event()
    tool_cleaned_up = False
    late_effect_ran = False

    @tool(concurrent_safe=True)
    async def slow_speculative(v: int) -> dict[str, int]:
        nonlocal tool_cleaned_up, late_effect_ran
        tool_entered.set()
        try:
            await asyncio.sleep(10.0)
            late_effect_ran = True
            return {"v": v}
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)
            tool_cleaned_up = True
            raise

    async def create_async(**_kw: Any) -> Any:
        async def agen():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="spec_cancel_1",
                                    function=SimpleNamespace(
                                        name="slow_speculative", arguments='{"v": 1}'
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(content=None, tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return agen()

    client = SimpleNamespace(
        max_retries=0,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async)),
    )
    state = DefaultState(max_steps=4)
    cfg = LoopConfig(
        max_steps=4,
        concurrent_dispatch=True,
        streaming_tool_execution=True,
        disabled_defaults=["persistence"],
    )

    async def drive() -> None:
        async with aclosing(
            async_composable_loop(
                AsyncOpenAIStreamingBackend(client, model="m"),
                task="cancel during speculative drain",
                tools=tools_from([slow_speculative], include_done=True),
                state=state,
                config=cfg,
            )
        ) as gen:
            async for _ in gen:
                pass

    task = asyncio.create_task(drive())
    await tool_entered.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tool_cleaned_up is True
    assert late_effect_ran is False
    receipts = state.metadata.get("speculative_tool_receipts", [])
    assert len(receipts) == 1
    assert receipts[0]["metadata"]["execution_status"] == "effect_unknown"


def test_speculative_streaming_eligibility_for_post_dispatch_and_observability_hooks() -> None:
    from looplet.limits import BudgetWarningHook
    from looplet.telemetry import MetricsCollector, MetricsHook, Tracer, TracingHook
    from looplet.tools import StopOnUnknownEffectHook
    from looplet.turn_engine import _hooks_allow_speculative_streaming

    assert _hooks_allow_speculative_streaming(
        [
            BudgetWarningHook(),
            TracingHook(Tracer()),
            MetricsHook(MetricsCollector()),
        ]
    )
    assert not _hooks_allow_speculative_streaming([StopOnUnknownEffectHook()])

    class CustomPostDispatch:
        def post_dispatch(
            self, state: Any, session_log: Any, tc: Any, tr: Any, step_num: int
        ) -> str | None:
            return "custom"

    class CustomShouldStop:
        def should_stop(self, state: Any, step_num: int, new_entities: int) -> bool:
            return True

    assert not _hooks_allow_speculative_streaming([CustomPostDispatch()])
    assert not _hooks_allow_speculative_streaming([CustomShouldStop()])


def test_preset_fork_isolates_mutable_config_tool_schemas_stateful_hooks_and_rejects_gateway() -> (
    None
):
    from looplet.limits import BudgetWarningHook, PerToolLimitHook, ToolCallBudgetHook
    from looplet.presets import AgentPreset
    from looplet.telemetry import Tracer, TracingHook
    from looplet.tools import StopOnUnknownEffectHook

    @tool
    def calc(x: int) -> dict[str, int]:
        return {"x": x}

    parent_cfg = LoopConfig(
        max_steps=5,
        disabled_defaults=["persistence"],
        tool_metadata={"k": "v1"},
        generate_kwargs={"top_p": 0.9},
    )
    ptl = PerToolLimitHook(limits={"calc": 2})
    bwh = BudgetWarningHook(thresholds=(0.5,))
    tcb = ToolCallBudgetHook(max_calls=5)
    sou = StopOnUnknownEffectHook()
    trh = TracingHook(Tracer())

    parent = AgentPreset(
        config=parent_cfg,
        hooks=[ptl, bwh, tcb, sou, trh],
        tools=tools_from([calc], include_done=True, validate=True),
        state=DefaultState(max_steps=5),
    )
    child = parent.fork(max_steps=8)
    child.config.disabled_defaults.append("extra")
    child.config.tool_metadata["k"] = "mutated"
    child.config.generate_kwargs["top_p"] = 0.1
    calc_spec = child.tools.get("calc")
    assert calc_spec is not None
    calc_spec.parameters["properties"]["x"]["type"] = "string"

    assert parent.config.disabled_defaults == ["persistence"]
    assert parent.config.tool_metadata == {"k": "v1"}
    assert parent.config.generate_kwargs == {"top_p": 0.9}
    parent_calc = parent.tools.get("calc")
    assert parent_calc is not None
    assert parent_calc.parameters["properties"]["x"]["type"] == "integer"

    for h_cls, orig in (
        (PerToolLimitHook, ptl),
        (BudgetWarningHook, bwh),
        (ToolCallBudgetHook, tcb),
        (StopOnUnknownEffectHook, sou),
        (TracingHook, trh),
    ):
        cloned = next(h for h in child.hooks if isinstance(h, h_cls))
        assert cloned is not orig

    gw_preset = AgentPreset(
        config=LoopConfig(),
        hooks=[],
        tools=tools_from([calc], include_done=True),
        state=DefaultState(),
        llm_gateway_enabled=True,
    )
    with pytest.raises(RuntimeError, match="model_gateway"):
        gw_preset.fork()

    parent.close()
    with pytest.raises(RuntimeError, match="closed"):
        parent.fork()


def test_in_place_message_mutation_updates_incremental_checkpoint_and_tool_schema_detachment(
    tmp_path: Path,
) -> None:
    from looplet.checkpoint import Checkpoint, _IncrementalCheckpointStore
    from looplet.conversation import Conversation, Message, MessageRole

    conv = Conversation()
    conv.append(Message(role=MessageRole.USER, content="hello", metadata={"tag": "v1"}))
    store = _IncrementalCheckpointStore(tmp_path / "inc_cp")
    cp1 = Checkpoint(
        step_number=1,
        session_log_data={"entries": []},
        conversation_data=conv.serialize(),
        config_snapshot={},
        tool_results_store={},
        metadata={},
    )
    store.save(cp1, "step_1")

    # Mutate metadata dict IN PLACE without reassigning attributes on Message!
    conv.messages[0].metadata["tag"] = "v2_in_place"
    cp2 = Checkpoint(
        step_number=2,
        session_log_data={"entries": []},
        conversation_data=conv.serialize(),
        config_snapshot={},
        tool_results_store={},
        metadata={},
    )
    store.save(cp2, "step_2")
    loaded = store.load("step_2")
    assert loaded is not None and loaded.conversation_data is not None
    assert loaded.conversation_data["messages"][0]["metadata"]["tag"] == "v2_in_place"

    # Verify ToolView schemas and shorthand ToolSpec.to_api_schema() are detached from caller mutation
    @tool
    def typed_fn(a: int) -> dict[str, int]:
        return {"a": a}

    shorthand = ToolSpec(
        name="short_fn",
        description="short",
        parameters={"q": "str"},
        execute=lambda q: q,
    )
    reg = tools_from([typed_fn, shorthand], include_done=False)
    view_before = json_snapshot(reg.tool_view().schemas)

    assert typed_fn.to_api_schema()["input_schema"] is typed_fn.parameters
    s2 = shorthand.to_api_schema()
    s2["input_schema"]["properties"]["q"]["type"] = "CORRUPTED"

    assert shorthand.to_api_schema()["input_schema"]["properties"]["q"]["type"] == "string"
    assert json_snapshot(reg.tool_view().schemas) == view_before


def test_provenance_overwrite_replaces_prior_model_attempts_and_preserves_same_run_reflush(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "overwritable_trace"

    def make_backend(tag: str) -> OpenAIBackend:
        return OpenAIBackend(
            SimpleNamespace(
                max_retries=0,
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **_kw: SimpleNamespace(
                            model="gpt-test",
                            choices=[
                                SimpleNamespace(
                                    finish_reason="tool_calls",
                                    message=SimpleNamespace(
                                        content="",
                                        tool_calls=[
                                            SimpleNamespace(
                                                id=f"id_{tag}",
                                                function=SimpleNamespace(
                                                    name="done",
                                                    arguments=json.dumps({"summary": tag}),
                                                ),
                                            )
                                        ],
                                    ),
                                )
                            ],
                            usage=SimpleNamespace(
                                prompt_tokens=10, completion_tokens=5, total_tokens=15
                            ),
                        )
                    )
                ),
            ),
            model="gpt-test",
        )

    sink1 = ProvenanceSink(trace_dir)
    r1 = run_result(
        sink1.wrap_llm(make_backend("run1")),
        task="first",
        tools=tools_from([], include_done=True),
        hooks=[sink1.trajectory_hook()],
        config=LoopConfig(max_steps=3, disabled_defaults=["persistence"]),
    )
    assert r1.completed
    sink1.flush()
    assert len(load_model_attempts(trace_dir)) == 1

    # Overwrite with a new sink: old attempt & old blobs must be replaced cleanly!
    sink2 = ProvenanceSink(trace_dir, overwrite=True)
    r2 = run_result(
        sink2.wrap_llm(make_backend("run2")),
        task="second",
        tools=tools_from([], include_done=True),
        hooks=[sink2.trajectory_hook()],
        config=LoopConfig(max_steps=3, disabled_defaults=["persistence"]),
    )
    assert r2.completed
    sink2.flush()
    # Repeated flush of the same run must keep its own attempts intact
    sink2.flush()
    attempts = load_model_attempts(trace_dir)
    assert len(attempts) == 1
    assert "run2" in json.dumps(attempts[0]["outcome"])
    assert "run1" not in json.dumps(attempts[0]["outcome"])


def test_hot_window_spill_rollback_preserves_in_memory_records_and_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    from looplet._storage import DiskBackedDict, DiskBackedList

    lst: DiskBackedList[dict[str, Any]] = DiskBackedList(lambda x: x, lambda x: x)
    lst.extend([{"i": 0}, {"i": 1}, {"i": 2}])
    lst.insert(1, {"i": 99})
    assert [x["i"] for x in lst] == [0, 99, 1, 2]
    lst[1:3] = [{"i": 10}, {"i": 20}]
    assert [x["i"] for x in lst] == [0, 10, 20, 2]
    del lst[1]
    assert lst.pop()["i"] == 2
    cloned = lst.clone()
    assert [x["i"] for x in cloned] == [0, 20]

    dct: DiskBackedDict[dict[str, Any]] = DiskBackedDict(lambda x: x, lambda x: x)
    dct["a"] = {"v": 1}
    dct["b"] = {"v": 2}
    snap = dict(dct)
    assert snap == {"a": {"v": 1}, "b": {"v": 2}}
    del dct["a"]
    assert list(dct.keys()) == ["b"]

    # Simulate SQLite OperationalError during _spill_to_spool: in-memory records must survive!
    class FailingConn:
        def __init__(self, real: Any) -> None:
            self._real = real

        def execute(self, sql: str, *args: Any) -> Any:
            return self._real.execute(sql, *args)

        def executemany(self, sql: str, *args: Any) -> Any:
            raise sqlite3.OperationalError("simulated disk full")

    monkeypatch.setattr(lst._spool, "connection", FailingConn(lst._spool.connection))
    with pytest.raises(sqlite3.OperationalError, match="simulated disk full"):
        lst._spill_to_spool()
    assert lst._mem is not None
    assert [x["i"] for x in lst] == [0, 20]

    assert dct._spool is not None
    monkeypatch.setattr(dct._spool, "connection", FailingConn(dct._spool.connection))
    with pytest.raises(sqlite3.OperationalError, match="simulated disk full"):
        dct._spill_dict_to_spool()
    assert dct._mem is not None
    assert dct["b"] == {"v": 2}


def test_lep_close_unblocks_waiting_rpc_and_shared_backend_first_use_thread_race(
    tmp_path: Path,
) -> None:
    # 1. Hung LEP server after initialize: close() from another thread must return promptly!
    hung_lep = tmp_path / "hung_lep.py"
    hung_lep.write_text(
        "import json, sys, time\n"
        "line = sys.stdin.readline()\n"
        "m = json.loads(line)\n"
        "sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': m['id'], 'result': {'capabilities': {}}}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "for _ in sys.stdin:\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )
    lep = LEPHookAdapter(
        [sys.executable, "-I", "-B", str(hung_lep)], timeout=30.0, on_failure="fail_open"
    )
    lep.pre_loop(DefaultState(), None, None)

    entered = threading.Event()
    worker_done = threading.Event()

    def blocked_caller() -> None:
        entered.set()
        _ = lep.check_permission(ToolCall(tool="echo", args={}), DefaultState())
        worker_done.set()

    t = threading.Thread(target=blocked_caller, daemon=True)
    t.start()
    assert entered.wait(timeout=2.0)
    time.sleep(0.05)
    t0 = time.perf_counter()
    lep.close()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"lep.close() blocked for {elapsed:.2f}s"
    assert worker_done.wait(timeout=2.0)

    # 2. Simultaneous first-use _conv_state_var() across 12 threads sharing one OpenAIBackend
    captured_by_thread: dict[int, str] = {}
    cap_lock = threading.Lock()
    barrier = threading.Barrier(12)

    def create_shared(**kw: Any) -> Any:
        msgs = kw["messages"]
        with cap_lock:
            captured_by_thread[threading.get_ident()] = msgs[0]["content"]
        return SimpleNamespace(
            model="m",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="d1",
                                function=SimpleNamespace(
                                    name="done", arguments='{"summary": "ok"}'
                                ),
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    shared_backend = OpenAIBackend(
        SimpleNamespace(
            max_retries=0,
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_shared)),
        ),
        model="m",
    )
    errors: list[BaseException] = []

    def thread_worker(idx: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            res = run_result(
                shared_backend,
                task=f"unique-thread-task-{idx}",
                tools=tools_from([], include_done=True),
                config=LoopConfig(max_steps=2, disabled_defaults=["persistence"]),
            )
            assert res.completed
            with cap_lock:
                seen = captured_by_thread[threading.get_ident()]
            assert f"unique-thread-task-{idx}" in seen
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(12)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10.0)
    assert errors == []
