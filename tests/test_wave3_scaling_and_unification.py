"""Wave-3 tests for prefix stability, O(N) persistence, hot-window storage, zero-thread RPC, preset.fork(), observer-compatible speculative streaming, and Sans-I/O turn engine parity."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
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
