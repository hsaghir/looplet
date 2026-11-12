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
