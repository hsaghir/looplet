"""First-principles scaling, persistence, turn-engine, cartridge, and eval regression tests."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from looplet import (
    Conversation,
    DefaultState,
    LoopConfig,
    Message,
    MessageRole,
    MockLLMBackend,
    cartridge_to_preset,
    run_cartridge_evals,
    run_result,
    run_result_async,
    tools_from,
)
from looplet._checkpoint_storage import CheckpointDocuments
from looplet._storage import DiskBackedDict, DiskBackedList, TrackedList
from looplet.backends import OpenAIBackend
from looplet.cartridge import Cartridge, CartridgeSerializationError, _load_yaml
from looplet.cartridge._load import _EXTENDS_TEMPDIRS
from looplet.lep import LEPHookAdapter, server_argv
from looplet.router import ModelProfile, SimpleRouter
from looplet.run_records import RunEnvelope, RunEvent
from looplet.run_store import MemoryRunStore
from looplet.session import SessionLog
from looplet.types import ToolResult


def test_tracked_list_snapshot_detaches_on_mutation_and_deepcopies() -> None:
    import copy

    items = TrackedList([{"n": 1}, {"n": 2}])
    snap = items.snapshot_list()
    assert isinstance(snap, list)
    assert snap._record_identity == items._record_identity
    snap.append({"n": 3})
    assert len(items) == 2
    assert len(snap) == 3
    assert snap._record_identity != items._record_identity
    copied = copy.deepcopy(items)
    assert copied == [{"n": 1}, {"n": 2}]


def test_incremental_checkpoints_encode_conversation_and_session_log_in_o_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = CheckpointDocuments(tmp_path)
    put_calls = 0
    original_put = documents._put

    def count_put(connection, value):
        nonlocal put_calls
        put_calls += 1
        return original_put(connection, value)

    monkeypatch.setattr(documents, "_put", count_put)
    conv = Conversation()
    log = SessionLog()
    roots = []
    for step in range(1, 51):
        conv.append(Message(role=MessageRole.USER, content=f"step {step} payload " + "x" * 200))
        log.record(step=step, theory="working", tool="read", reasoning=f"r-{step}")
        roots.append(
            documents.save(
                {"conversation": conv.serialize(), "session_log": {"entries": log.to_list()}},
                {"step_number": step},
            )
        )

    # Without O(delta) TrackedList encoding, 50 checkpoints of growing 50-item
    # dicts require >25,000 _put calls; O(delta) stays well under 2,000.
    assert put_calls < 2000
    loaded_mid = documents.load(roots[19])
    loaded_end = documents.load(roots[-1])
    assert len(loaded_mid["conversation"]["messages"]) == 20
    assert len(loaded_end["conversation"]["messages"]) == 50
    assert len(loaded_end["session_log"]["entries"]) == 50
    documents.verify(roots[-1], set())


def test_shared_temp_spool_reuses_single_connection_and_clones_in_sql() -> None:
    first = DiskBackedList(dict, dict, [{"i": i} for i in range(16)])
    second = DiskBackedList(dict, dict, [{"j": j} for j in range(8)])
    temp_map: DiskBackedDict[dict] = DiskBackedDict(dict, dict)
    with temp_map.transaction():
        temp_map["a"] = {"v": 1}
        temp_map["b"] = {"v": 2}
    assert first._spool is second._spool is temp_map._spool
    cloned = first.clone()
    first.clear()
    assert len(first) == 0
    assert len(cloned) == 16
    assert cloned[15] == {"i": 15}
    assert list(temp_map.keys()) == ["a", "b"]


def test_memory_run_store_append_event_is_linear() -> None:
    store = MemoryRunStore()
    env = RunEnvelope(run_id="run-linear")
    store.create(env)
    for seq in range(1, 101):
        store.append_event(
            RunEvent(
                sequence=seq,
                kind="step",
                timestamp=float(seq),
                envelope=env,
                payload={"seq": seq},
            )
        )
    loaded = store.load("run-linear")
    assert loaded is not None and len(loaded.events) == 100
    assert len(store.events("run-linear")) == 100


def test_conversation_incremental_projection_and_backend_translation_cache() -> None:
    conv = Conversation()
    conv.append(
        Message(
            role=MessageRole.ASSISTANT,
            content=[],
            metadata={"origin": "model"},
        )
    )
    conv.append(
        Message(
            role=MessageRole.TOOL,
            content="",
            tool_result=ToolResult(
                tool="lookup", args_summary="", data={"v": "alpha"}, call_id="c1"
            ),
            metadata={"step_number": 1, "tool_call_id": "c1"},
        )
    )
    by_call, by_step, _, _ = conv._projection_index()
    assert "c1" in by_call and 1 in by_step
    assert "[TOOL RESULT lookup]" in conv.continuation_context()

    created_payloads = []
    client = SimpleNamespace(
        max_retries=0,
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: (
                    created_payloads.append(kw)
                    or SimpleNamespace(
                        choices=[
                            SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))
                        ],
                        usage=None,
                    )
                )
            )
        ),
    )
    backend = OpenAIBackend(client, model="test-model")
    backend.set_conversation_input(
        {"prompt": "Initial task", "conversation": {"version": 1, "messages": []}}
    )
    assert backend._translated_conversation_messages is not None
    assert backend.generate("Initial task") == "ok"
    assert len(created_payloads) == 1


def test_turn_engine_preserves_scoped_llm_proxy_under_router(tmp_path: Path) -> None:
    from looplet import ToolContext
    from looplet.provenance import ProvenanceSink

    routed_backend = MockLLMBackend(
        [
            json.dumps({"tool": "delegate", "args": {}}),
            "sub_answer",
            json.dumps({"tool": "done", "args": {"summary": "sub_done"}}),
            json.dumps({"tool": "done", "args": {"summary": "outer_done"}}),
        ],
        cycle=False,
    )
    router = SimpleRouter({}, default_profile=ModelProfile("routed", routed_backend))

    def sub_helper(ctx: ToolContext) -> dict:
        return {"text": ctx.llm.generate("sub prompt")}

    def delegate(ctx: ToolContext) -> dict:
        inner = run_result(
            ctx.llm,
            task="inner",
            tools=tools_from([sub_helper], include_done=True),
            config=LoopConfig(
                max_steps=3,
                use_native_tools=False,
                harness_profile="minimal",
                router=router,
            ),
        )
        return {"inner": inner.completed}

    sink = ProvenanceSink(tmp_path / "routed-scope")
    recorded = sink.wrap_llm(routed_backend)
    outer = run_result(
        recorded,
        task="outer",
        tools=tools_from([delegate], include_done=True),
        hooks=[sink.trajectory_hook()],
        config=LoopConfig(
            max_steps=3,
            use_native_tools=False,
            harness_profile="minimal",
            router=router,
        ),
    )
    assert outer.completed
    assert any(call.scope == "tool:delegate" for call in recorded.calls)


async def test_async_loop_runs_lep_hook_off_event_loop_thread(tmp_path: Path) -> None:
    server = tmp_path / "lep_server.py"
    server.write_text(
        "import time\n"
        "from looplet.lep import LEPServerBase\n"
        "class S(LEPServerBase):\n"
        "    def decide(self, slot, view):\n"
        "        if slot == 'pre_dispatch':\n"
        "            time.sleep(0.08)\n"
        "        return None\n"
        "if __name__ == '__main__':\n"
        "    S().serve()\n",
        encoding="utf-8",
    )
    ticks = 0
    stop_ticker = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop_ticker.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    def noop() -> dict:
        return {"ok": True}

    from looplet.testing import AsyncMockLLMBackend

    hook = LEPHookAdapter(server_argv(str(server)))
    bg = asyncio.create_task(ticker())
    try:
        res = await run_result_async(
            AsyncMockLLMBackend(
                [
                    json.dumps({"tool": "noop", "args": {}}),
                    json.dumps({"tool": "done", "args": {"summary": "ok"}}),
                ]
            ),
            task="test async lep",
            tools=tools_from([noop], include_done=True),
            hooks=[hook],
            config=LoopConfig(max_steps=3, use_native_tools=False, harness_profile="minimal"),
        )
    finally:
        stop_ticker.set()
        await bg
        hook.close()
    assert res.completed
    # Ticker ran multiple 10ms ticks during the 80ms LEP pre_dispatch sleep.
    assert ticks >= 3


def test_cartridge_extends_caches_content_addressed_overlay_and_invalidates_on_edit(
    tmp_path: Path,
) -> None:
    def write_cart(root: Path, name: str, config_yaml: str, prompt: str) -> None:
        (root / "prompts").mkdir(parents=True, exist_ok=True)
        (root / "cartridge.json").write_text(
            json.dumps({"name": name, "schema_version": 2}), encoding="utf-8"
        )
        (root / "config.yaml").write_text(config_yaml, encoding="utf-8")
        (root / "prompts/system.md").write_text(prompt, encoding="utf-8")

    base = tmp_path / "base.cartridge"
    child = tmp_path / "child.cartridge"
    write_cart(base, "base_agent", "max_steps: 10\ndone_tool: done\n", "Base v1")
    write_cart(child, "child_agent", "extends: ../base.cartridge\nmax_steps: 4\n", "Child v1")

    before = len(_EXTENDS_TEMPDIRS)
    for _ in range(15):
        preset = cartridge_to_preset(child)
        assert preset.config.max_steps == 4
        assert preset.config.system_prompt == "Child v1"
    after_15 = len(_EXTENDS_TEMPDIRS)
    assert after_15 - before == 1

    # Editing parent or child invalidates the content-addressed cache immediately.
    (child / "prompts/system.md").write_text("Child v2", encoding="utf-8")
    updated = cartridge_to_preset(child)
    assert updated.config.system_prompt == "Child v2"
    assert len(_EXTENDS_TEMPDIRS) == after_15 + 1


@pytest.mark.parametrize(
    "bad_yaml",
    [
        "max_steps: 5\nmax_steps: 10\n",
        "items:\n  - name: a\n    name: b\n",
        "meta: {a: 1, a: 2}\n",
        'meta: {"a": 1, "a": 2}\n',
    ],
)
def test_yaml_rejects_duplicate_mapping_keys(bad_yaml: str) -> None:
    with pytest.raises(CartridgeSerializationError, match="duplicate workspace YAML key"):
        _load_yaml(bad_yaml, source_path="config.yaml")


@pytest.mark.parametrize(
    "compatibility",
    [
        {"unknown_key": True},
        {"looplet": "   "},
    ],
)
def test_manifest_rejects_schema_invalid_compatibility(tmp_path: Path, compatibility: dict) -> None:
    cart = tmp_path / "invalid.cartridge"
    cart.mkdir()
    (cart / "cartridge.json").write_text(
        json.dumps({"name": "bad", "schema_version": 2, "compatibility": compatibility}),
        encoding="utf-8",
    )
    with pytest.raises(
        CartridgeSerializationError, match="compatibility|invalid Looplet version constraint"
    ):
        Cartridge.from_directory(cart)


def test_run_cartridge_evals_max_concurrency_preserves_order_and_runs_in_parallel(
    tmp_path: Path,
) -> None:
    cart = tmp_path / "parallel.cartridge"
    (cart / "prompts").mkdir(parents=True)
    (cart / "tools/done").mkdir(parents=True)
    (cart / "evals/cases").mkdir(parents=True)
    (cart / "cartridge.json").write_text(
        json.dumps({"name": "parallel_eval", "schema_version": 2}), encoding="utf-8"
    )
    (cart / "config.yaml").write_text("max_steps: 2\ndone_tool: done\n", encoding="utf-8")
    (cart / "runtime.yaml").write_text("use_native_tools: false\n", encoding="utf-8")
    (cart / "prompts/system.md").write_text("Complete immediately.\n", encoding="utf-8")
    (cart / "tools/done/tool.yaml").write_text(
        "name: done\ndescription: Finish.\nparameters:\n  summary:\n    type: string\n",
        encoding="utf-8",
    )
    (cart / "tools/done/execute.py").write_text(
        "def execute(ctx, *, summary: str = '') -> dict:\n    return {'summary': summary}\n",
        encoding="utf-8",
    )
    for idx in range(4):
        (cart / f"evals/cases/case_{idx}.json").write_text(
            json.dumps(
                {
                    "id": f"case_{idx}",
                    "task": {"goal": f"goal_{idx}"},
                    "expected": {"goal": f"goal_{idx}"},
                }
            ),
            encoding="utf-8",
        )
    (cart / "evals/eval_check.py").write_text(
        "from looplet import eval_mark\n"
        "@eval_mark('required')\n"
        "def eval_goal(ctx):\n"
        "    return ctx.task['goal'] == ctx.expected['goal']\n",
        encoding="utf-8",
    )

    barrier = threading.Barrier(4, timeout=10)

    class ConcurrentBackend:
        def generate(self, prompt: str, **kwargs) -> str:
            barrier.wait()
            time.sleep(0.01)
            return json.dumps({"tool": "done", "args": {"summary": "ok"}})

    records = run_cartridge_evals(
        cart,
        llm=ConcurrentBackend(),
        output_dir=tmp_path / "eval_out",
        max_concurrency=4,
    )
    assert [r.case.id for r in records if r.case is not None] == [
        "case_0",
        "case_1",
        "case_2",
        "case_3",
    ]
    assert all(r.results[0].passed for r in records)


def test_tracked_list_all_mutations_and_cursor_boundaries() -> None:
    tl = TrackedList([3, 1, 2])
    tl.insert(0, 9)
    tl.insert(99, 8)
    assert list(tl) == [9, 3, 1, 2, 8]
    assert tl.pop() == 8
    assert tl.pop(0) == 9
    tl.sort()
    assert list(tl) == [1, 2, 3]
    tl.reverse()
    assert list(tl) == [3, 2, 1]
    tl[-1] = 10
    tl[0:2] = [30, 20]
    assert list(tl) == [30, 20, 10]
    gen_before = tl._generation
    tl[0:2] = [99]
    assert tl._generation > gen_before
    assert list(tl) == [99, 10]
    tl += [7, 8]
    del tl[1]
    assert list(tl) == [99, 7, 8]
    with pytest.raises(IndexError, match="out of range"):
        with tl.changes_since((0, 0, 0), 10):
            pass
    tl.clear()
    assert list(tl) == []


def test_shared_spool_concurrent_transactions_and_deferred_gc_deletes() -> None:
    import gc

    from looplet._storage import _cleanup_active_spool

    d1: DiskBackedDict[dict] = DiskBackedDict(dict, dict)
    d2: DiskBackedDict[dict] = DiskBackedDict(dict, dict)
    d1["seed"] = {"v": 0}
    d2["seed"] = {"v": 0}

    entered_tx = threading.Event()
    proceed_rollback = threading.Event()
    errors: list[BaseException] = []

    def failing_writer() -> None:
        try:
            with pytest.raises(RuntimeError, match="abort"):
                with d1.transaction():
                    d1["bad"] = {"v": -1}
                    entered_tx.set()
                    proceed_rollback.wait(timeout=5)
                    raise RuntimeError("abort")
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def committing_writer() -> None:
        try:
            entered_tx.wait(timeout=5)
            proceed_rollback.set()
            with d2.transaction():
                d2["good"] = {"v": 42}
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=failing_writer)
    t2 = threading.Thread(target=committing_writer)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not errors
    assert "bad" not in d1
    assert d1["seed"] == {"v": 0}
    assert d2["good"] == {"v": 42}
    assert [k for _, k, _ in d2.items_after(0)] == ["seed", "good"]
    del d2["seed"]
    with pytest.raises(KeyError):
        _ = d2["seed"]
    with pytest.raises(KeyError):
        del d2["missing"]
    d2.clear()
    assert len(d2) == 0

    # Verify GC finalizer queues onto pending_deletes and drains cleanly.
    ephemeral: DiskBackedList[dict] | None = DiskBackedList(dict, dict, [{"x": 1}])
    spool = ephemeral._spool
    cid = ephemeral._cid
    ephemeral = None
    gc.collect()
    _ = DiskBackedList(dict, dict, [{"y": 2}])
    with spool.lock:
        leftover = spool.connection.execute(
            "SELECT count(*) FROM list_records WHERE cid = ?", (cid,)
        ).fetchone()[0]
    assert leftover == 0
    _cleanup_active_spool()
    fresh = DiskBackedList(dict, dict, [{"z": 3}])
    assert fresh[0] == {"z": 3}


def test_in_place_mutation_at_oldest_index_invalidates_conversation_and_session_caches(
    tmp_path: Path,
) -> None:
    conv = Conversation()
    log = SessionLog()
    for idx in range(10):
        conv.append(
            Message(
                role=MessageRole.TOOL if idx == 1 else MessageRole.USER,
                content=f"msg-{idx}",
                tool_result=(
                    ToolResult(tool="lookup", args_summary="", data={"v": "v1"}, call_id="c1")
                    if idx == 1
                    else None
                ),
                metadata={"step_number": idx, **({"tool_call_id": "c1"} if idx == 1 else {})},
            )
        )
        log.record(step=idx + 1, theory=f"t-{idx}", tool="lookup", reasoning=f"r-{idx}")

    docs = CheckpointDocuments(tmp_path)
    docs.save({"conv": conv.serialize(), "log": log.to_list()}, {"step_number": 10})
    assert "msg-0" in conv.continuation_context()
    by_call, _, _, _ = conv._projection_index()
    assert by_call["c1"].data == {"v": "v1"}

    # Mutate oldest items (indices 0 and 1, well outside any 4-item tail) in-place!
    conv.messages[0].content = "msg-0-mutated"
    conv.messages[0].metadata["custom"] = "edited"
    assert conv.messages[1].tool_result is not None
    conv.messages[1].tool_result.data["v"] = "v2"
    log.entries[0].theory = "t-0-mutated"
    log.entries[0].findings.append("new-finding")

    updated_conv = conv.serialize()
    updated_log = log.to_list()
    assert updated_conv["messages"][0]["content"] == "msg-0-mutated"
    assert updated_conv["messages"][0]["metadata"]["custom"] == "edited"
    assert updated_conv["messages"][1]["tool_result"]["data"] == {"v": "v2"}
    assert "msg-0-mutated" in conv.continuation_context()
    assert '"v": "v2"' in conv.continuation_context()
    assert updated_log[0]["theory"] == "t-0-mutated"
    assert updated_log[0]["findings"] == ["new-finding"]

    root2 = docs.save({"conv": updated_conv, "log": updated_log}, {"step_number": 11})
    loaded = docs.load(root2)
    assert loaded["conv"]["messages"][0]["content"] == "msg-0-mutated"
    assert loaded["log"][0]["theory"] == "t-0-mutated"


def test_shared_stateless_backend_isolates_concurrent_conversations_and_guards_stateful(
    tmp_path: Path,
) -> None:
    from looplet.backends import OpenAIResponsesBackend

    cart = tmp_path / "multistep.cartridge"
    (cart / "prompts").mkdir(parents=True)
    (cart / "tools/echo").mkdir(parents=True)
    (cart / "tools/done").mkdir(parents=True)
    (cart / "evals/cases").mkdir(parents=True)
    (cart / "cartridge.json").write_text(
        json.dumps({"name": "multistep_eval", "schema_version": 2}), encoding="utf-8"
    )
    (cart / "config.yaml").write_text("max_steps: 4\ndone_tool: done\n", encoding="utf-8")
    (cart / "runtime.yaml").write_text("use_native_tools: true\n", encoding="utf-8")
    (cart / "prompts/system.md").write_text("Echo token then finish.\n", encoding="utf-8")
    (cart / "tools/echo/tool.yaml").write_text(
        "name: echo\ndescription: Echo.\nparameters:\n  token:\n    type: string\n",
        encoding="utf-8",
    )
    (cart / "tools/echo/execute.py").write_text(
        "def execute(ctx, *, token: str) -> dict:\n    return {'echoed': token}\n",
        encoding="utf-8",
    )
    (cart / "tools/done/tool.yaml").write_text(
        "name: done\ndescription: Done.\nparameters:\n  summary:\n    type: string\n",
        encoding="utf-8",
    )
    (cart / "tools/done/execute.py").write_text(
        "def execute(ctx, *, summary: str = '') -> dict:\n    return {'summary': summary}\n",
        encoding="utf-8",
    )
    for idx in range(5):
        (cart / f"evals/cases/case_{idx}.json").write_text(
            json.dumps(
                {
                    "id": f"case_{idx}",
                    "task": {"token": f"TOK_{idx}"},
                    "expected": {"token": f"TOK_{idx}"},
                }
            ),
            encoding="utf-8",
        )
    (cart / "evals/eval_check.py").write_text(
        "from looplet import eval_mark\n"
        "@eval_mark('required')\n"
        "def eval_tok(ctx):\n"
        "    return ctx.final_output.get('summary') == ctx.expected['token']\n",
        encoding="utf-8",
    )

    active = 0
    peak_active = 0
    lock = threading.Lock()

    def create_completion(**kwargs):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            time.sleep(0.015)
            msgs = kwargs["messages"]
            # Find which TOK_i belongs to this conversation's initial user message
            user_texts = [
                m["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], str)
            ]
            my_tok = next(f"TOK_{i}" for i in range(5) if any(f"TOK_{i}" in t for t in user_texts))
            # Verify NO message in this request leaked another case's token!
            serialized = json.dumps(msgs)
            for other in range(5):
                other_tok = f"TOK_{other}"
                if other_tok != my_tok:
                    assert other_tok not in serialized
            has_tool_reply = any(m["role"] == "tool" for m in msgs)
            if not has_tool_reply:
                tc = SimpleNamespace(
                    id=f"call_{my_tok}",
                    function=SimpleNamespace(name="echo", arguments=json.dumps({"token": my_tok})),
                )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))
                    ],
                    usage=None,
                )
            done_tc = SimpleNamespace(
                id=f"done_{my_tok}",
                function=SimpleNamespace(name="done", arguments=json.dumps({"summary": my_tok})),
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[done_tc]))
                ],
                usage=None,
            )
        finally:
            with lock:
                active -= 1

    client = SimpleNamespace(
        max_retries=0,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)),
    )
    shared_openai = OpenAIBackend(client, model="test-model")
    records = run_cartridge_evals(cart, llm=shared_openai, max_concurrency=2)
    assert len(records) == 5
    assert peak_active <= 2
    assert all(r.results[0].passed for r in records)
    for r in records:
        r.cleanup()

    # Reject sharing stateful previous_response_id backend across max_concurrency > 1
    stateful = OpenAIResponsesBackend(
        SimpleNamespace(responses=None), use_previous_response_id=True
    )
    with pytest.raises(ValueError, match="stateful backend"):
        run_cartridge_evals(cart, llm=stateful, max_concurrency=2)
    with pytest.raises(ValueError, match="max_concurrency"):
        run_cartridge_evals(cart, llm=shared_openai, max_concurrency=0)
    with pytest.raises(ValueError, match="max_concurrency"):
        run_cartridge_evals(cart, llm=shared_openai, max_concurrency=True)  # type: ignore[arg-type]

    # Factory callable creates and closes per-case backends cleanly
    closed_count = 0

    class PerCaseBackend:
        def generate_with_tools(self, prompt: str, **kwargs):
            my_tok = next(f"TOK_{i}" for i in range(5) if f"TOK_{i}" in prompt)
            return [{"type": "tool_use", "id": "d1", "name": "done", "input": {"summary": my_tok}}]

        def generate(self, prompt: str, **kwargs) -> str:
            return ""

        def close(self) -> None:
            nonlocal closed_count
            with lock:
                closed_count += 1

    factory_records = run_cartridge_evals(cart, llm=PerCaseBackend, max_concurrency=2)
    assert len(factory_records) == 5
    assert closed_count == 5
    for r in factory_records:
        r.cleanup()


def test_span_and_trajectory_span_serialization_roundtrip_and_replay_checkpoint_router(
    tmp_path: Path,
) -> None:
    from looplet.provenance import Trajectory, TrajectoryRecorder, _ReplayLLMBackend
    from looplet.telemetry import Span, Tracer
    from looplet.turn_engine import restore_checkpoint_into_loop

    tracer = Tracer()
    rec = TrajectoryRecorder(tracer=tracer)
    res = run_result(
        MockLLMBackend([json.dumps({"tool": "done", "args": {"summary": "ok"}})]),
        task="trace spans",
        tools=tools_from([], include_done=True),
        hooks=[rec],
        config=LoopConfig(max_steps=2, use_native_tools=False, harness_profile="minimal"),
    )
    assert res.completed
    rec.save(tmp_path / "span-trace")
    saved = json.loads((tmp_path / "span-trace/trajectory.json").read_text(encoding="utf-8"))
    assert len(saved["spans"]) == 1
    restored_span = Span.from_dict(saved["spans"][0])
    assert restored_span.name == "loop.run"
    assert restored_span.duration_ms is not None

    traj = Trajectory(run_id="t1", started_at=1.0, spans=[restored_span])
    assert [Span.from_dict(s).name for s in traj.to_dict()["spans"]] == ["loop.run"]

    # Verify restore_checkpoint_into_loop does NOT invoke router.select when replaying
    router_called = False

    class ForbiddenRouter:
        def select(self, **kwargs):
            nonlocal router_called
            router_called = True
            raise AssertionError("router should not be selected during replay")

    restored_payloads: list[dict] = []
    replay_llm = _ReplayLLMBackend(
        [{"index": 0, "method": "generate", "request_id": "loop/0", "response": "ok"}],
        source_trace="trace",
    )
    replay_llm.restore_checkpoint_state = restored_payloads.append  # type: ignore[attr-defined]
    fake_ckpt = SimpleNamespace(
        metadata={},
        step_number=1,
        is_terminal=False,
    )
    cfg = LoopConfig(max_steps=3, router=ForbiddenRouter())  # type: ignore[arg-type]
    restore_checkpoint_into_loop(
        fake_ckpt,
        config=cfg,
        llm=replay_llm,
        state=DefaultState(max_steps=3),
        session_log=SessionLog(),
        active_conv=Conversation(),
        explicit_conversation=None,
        validate_checkpoint_identity=lambda ckpt, rid: None,
        resume_loop_state=lambda ckpt, max_steps: {"domain_state": {"llm": {"cursor": 1}}},
    )
    assert not router_called
    assert restored_payloads == [{"cursor": 1}]
