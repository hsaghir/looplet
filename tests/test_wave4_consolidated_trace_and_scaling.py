"""Wave 4 tests: consolidated trace.jsonl + CAS blobs and 7 scaling/fidelity pillars."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import looplet
import looplet.backends as backends_mod
from looplet import (
    AsyncOpenAIResponsesBackend,
    BaseToolRegistry,
    Checkpoint,
    ContentBlock,
    ContextBudget,
    Conversation,
    ErrorKind,
    EvalContext,
    FileCheckpointStore,
    FileRunStore,
    FileToolResultStore,
    LoopConfig,
    Message,
    MessageRole,
    OpenAIResponsesBackend,
    ProvenanceSink,
    ReloadHook,
    RunEnvelope,
    StateServiceBase,
    StateServiceClient,
    StateServiceHandle,
    ToolCall,
    ToolResult,
    ToolSpec,
    WorkspaceWatcher,
    load_eval_run,
    load_model_attempts,
    load_trace_events,
    load_trajectory,
    replay_loop,
    run_cartridge_evals,
    run_cartridge_evals_async,
    run_result,
    run_result_async,
    run_subprocess_tool,
    scaffold_cartridge,
    tool,
    tools_from,
)
from looplet.__main__ import main as cli_main
from looplet._storage import _HOT_MAX_ITEMS, DiskBackedDict
from looplet.backends import (
    AnthropicStreamingBackend,
    AsyncAnthropicBackend,
    AsyncOpenAIStreamingBackend,
    OpenAIStreamingBackend,
)
from looplet.budget import classify_tier
from looplet.events import LifecycleEvent
from looplet.provenance import (
    LARGE_RESULT_BLOB_BYTES,
    TRACE_EVENT_SCHEMA,
    AsyncRecordingLLMBackend,
    RecordingLLMBackend,
)
from looplet.runtime import AgentRuntime
from looplet.session import SessionLog
from looplet.state_service import StateServiceError
from looplet.testing import AsyncMockLLMBackend, MockLLMBackend

# ── Pillar 1: Consolidated trace.jsonl + CAS blobs + crash recovery ──


def test_consolidated_trace_jsonl_cas_blobs_and_manifest_refs(tmp_path: Path) -> None:
    """20-step native run streams trace.jsonl, deduplicates captures & large tool results."""
    big_payload = {"records": [{"idx": i, "blob": "X" * 512} for i in range(30)]}
    assert len(json.dumps(big_payload)) > LARGE_RESULT_BLOB_BYTES

    @tool
    def fetch_page(page: int) -> dict[str, Any]:
        """Fetch structured page payload."""
        if page == 1:
            return big_payload
        return {"page": page, "ok": True}

    tools = tools_from([fetch_page], include_done=True)

    class TwentyStepNativeBackend:
        conversation_mode = "messages"

        def __init__(self) -> None:
            self.turn = 0
            self.last_request: dict[str, Any] | None = None
            self.last_stop_reason = "tool_use"
            self.last_usage = {"input_tokens": 50, "output_tokens": 10}
            self._snap: dict[str, Any] | None = None
            self._cb: Any = None

        def set_conversation_input(self, snap: dict[str, Any] | None) -> None:
            self._snap = snap

        def set_request_attempt_callback(self, cb: Any) -> None:
            self._cb = cb

        @property
        def request_attempt_callback(self) -> Any:
            return self._cb

        def generate(self, prompt: str, **kw: Any) -> str:
            return "ok"

        def generate_with_tools(
            self, prompt: str, *, tools: list[dict[str, Any]], **kw: Any
        ) -> list[dict[str, Any]]:
            msgs = self._snap["conversation"]["messages"] if self._snap else []
            self.last_request = {"messages": msgs, "tools": tools}
            if self._cb is not None:
                self._cb()
            self.turn += 1
            if self.turn <= 18:
                return [
                    {
                        "type": "tool_use",
                        "id": f"call_{self.turn}",
                        "name": "fetch_page",
                        "input": {"page": self.turn},
                    }
                ]
            self.last_stop_reason = "end_turn"
            return [
                {
                    "type": "tool_use",
                    "id": "call_done",
                    "name": "done",
                    "input": {"summary": "fetched all 18 pages"},
                }
            ]

    trace_dir = tmp_path / "trace_compact"
    sink = ProvenanceSink(dir=trace_dir, compact_blobs=True)
    res = run_result(
        llm=sink.wrap_llm(TwentyStepNativeBackend()),
        tools=tools,
        hooks=[sink.trajectory_hook()],
        config=LoopConfig(max_steps=25, use_native_tools=True, disabled_defaults=["persistence"]),
        task={"goal": "fetch 18 pages"},
    )
    assert res.completed
    sink.flush()

    assert (trace_dir / "trace.jsonl").is_file()
    assert (trace_dir / "blobs").is_dir()

    # Raw trace.jsonl uses compact blob refs for large step data and >12 message manifests
    raw_events = load_trace_events(trace_dir, rehydrate=False)
    assert [ev["seq"] for ev in raw_events] == list(range(len(raw_events)))
    assert all(ev["schema"] == TRACE_EVENT_SCHEMA for ev in raw_events)
    kinds = [ev["kind"] for ev in raw_events]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    assert "llm_intent" in kinds and "llm_dispatch" in kinds and "llm_outcome" in kinds
    step_1_ev = next(
        ev for ev in raw_events if ev["kind"] == "step" and ev["payload"]["step_num"] == 1
    )
    assert "data_blob_ref" in step_1_ev["payload"]["tool_result"]
    assert "data" not in step_1_ev["payload"]["tool_result"]

    last_intent_ev = [ev for ev in raw_events if ev["kind"] == "llm_intent"][-1]
    assert (
        "messages_manifest_blob_ref"
        in last_intent_ev["payload"]["request"]["conversation_input"]["conversation"]
    )

    # Raw trajectory.json with compact_blobs=True also stores data_blob_ref + manifest refs
    raw_traj = json.loads((trace_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert "data_blob_ref" in raw_traj["steps"][0]["tool_result"]
    assert "data" not in raw_traj["steps"][0]["tool_result"]
    assert "messages_manifest_blob_ref" in raw_traj["llm_calls"][-1]["capture"]["provider_request"]

    # Rehydrated loaders restore 100% lossless data, messages, and tool schemas!
    hydrated_events = load_trace_events(trace_dir, rehydrate=True)
    hydrated_step_1 = next(
        ev for ev in hydrated_events if ev["kind"] == "step" and ev["payload"]["step_num"] == 1
    )
    assert hydrated_step_1["payload"]["tool_result"]["data"] == big_payload

    hydrated_traj = load_trajectory(trace_dir, rehydrate=True)
    assert hydrated_traj["steps"][0]["tool_result"]["data"] == big_payload
    assert len(hydrated_traj["llm_calls"][-1]["capture"]["provider_request"]["messages"]) >= 36
    assert len(hydrated_traj["llm_calls"][-1]["capture"]["request"]["tools"]) == 2

    # EvalContext.from_trajectory_dir and replay_loop(stub_tools=True) rehydrate seamlessly!
    eval_ctx = EvalContext.from_trajectory_dir(trace_dir)
    assert eval_ctx.steps[0].tool_result.data == big_payload

    replayed_steps = list(replay_loop(trace_dir, tools=tools, stub_tools=True))
    assert len(replayed_steps) == 19
    assert replayed_steps[0].tool_result.data == big_payload


def test_trace_jsonl_crash_recovery_without_flush_and_torn_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a run crashes mid-turn before flush(), load_trajectory & CLI show recover."""

    @tool
    def compute_chunk(n: int) -> dict[str, Any]:
        """Return large payload on step 1."""
        return {"n": n, "payload": "Z" * 12_000}

    tools = tools_from([compute_chunk], include_done=True)

    class CrashOnSecondTurnBackend:
        conversation_mode = "messages"

        def __init__(self) -> None:
            self.calls = 0

        def set_conversation_input(self, snap: Any) -> None:
            self._snap = snap

        def generate(self, prompt: str, **kw: Any) -> str:
            return "ok"

        def generate_with_tools(self, prompt: str, *, tools: Any, **kw: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return [
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "compute_chunk",
                        "input": {"n": 7},
                    }
                ]
            raise KeyboardInterrupt("simulated abrupt process kill")

    trace_dir = tmp_path / "crashed_trace"
    sink = ProvenanceSink(dir=trace_dir)
    rec_llm = sink.wrap_llm(CrashOnSecondTurnBackend())
    hook = sink.trajectory_hook()

    # Execute without calling sink.flush(); KeyboardInterrupt aborts mid-run!
    with pytest.raises(KeyboardInterrupt, match="simulated abrupt process kill"):
        run_result(
            llm=rec_llm,
            tools=tools,
            hooks=[hook],
            config=LoopConfig(max_steps=5, use_native_tools=True, harness_profile="minimal"),
            task={"goal": "survive crash"},
        )
    assert not (trace_dir / "trajectory.json").exists()
    assert not (trace_dir / "manifest.jsonl").exists()
    assert (trace_dir / "trace.jsonl").is_file()

    # Simulate an uncompleted in-flight intent + a torn trailing record from SIGKILL!
    orphan_id = "f" * 32
    with (trace_dir / "trace.jsonl").open("a", encoding="utf-8") as fh:
        existing_count = sum(
            1
            for ln in (trace_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        )
        fh.write(
            json.dumps(
                {
                    "schema": TRACE_EVENT_SCHEMA,
                    "seq": existing_count,
                    "timestamp": time.time(),
                    "run_id": hook.trajectory.run_id,
                    "kind": "llm_intent",
                    "payload": {
                        "attempt_id": orphan_id,
                        "request_id": "loop/99",
                        "started_at": time.time(),
                        "step_num": 2,
                        "request": {"method": "generate", "prompt": "orphan"},
                    },
                }
            )
            + "\n"
        )
        # Torn trailing line (no newline, invalid truncated JSON)
        fh.write('{"schema":"looplet.trace-event.v1","seq":999,"kind":"step","paylo')

    recovered = load_trajectory(trace_dir)
    assert recovered["metadata"]["recovered_from_trace_jsonl"] is True
    assert recovered["task"] == {"goal": "survive crash"}
    assert recovered["step_count"] == 1
    assert recovered["steps"][0]["tool_result"]["data"] == {"n": 7, "payload": "Z" * 12_000}
    assert recovered["metadata"]["unfinished_attempts"] == [
        {
            "attempt_id": orphan_id,
            "request_id": "loop/99",
            "started_at": recovered["metadata"]["unfinished_attempts"][0]["started_at"],
            "step_num": 2,
            "scope": None,
        }
    ]

    # CLI `looplet show` also renders crashed trace.jsonl-only directories!
    rc = cli_main(["show", str(trace_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 steps" in out


def test_file_run_store_deduplicates_large_tool_results_into_blobs(tmp_path: Path) -> None:
    """FileRunStore stores >8 KiB step tool results in blobs/ and rehydrates on load()."""
    large_text = "RESULT_PAYLOAD_" * 1200

    @tool
    def emit_big() -> dict[str, Any]:
        """Return oversized tool data."""
        return {"big": large_text}

    from looplet.presets import AgentPreset
    from looplet.types import DefaultState

    store = FileRunStore(tmp_path / "runs")
    preset = AgentPreset(
        tools=tools_from([emit_big], include_done=True),
        state=DefaultState(max_steps=4),
        hooks=[],
        config=LoopConfig(max_steps=4, use_native_tools=False, harness_profile="minimal"),
    )
    rt = AgentRuntime(preset, store=store)
    llm = MockLLMBackend(
        [
            json.dumps({"tool": "emit_big", "args": {}}),
            json.dumps({"tool": "done", "args": {"summary": "ok"}}),
        ]
    )
    result = rt.run(llm, task="emit big")
    assert result.completed
    run_id = result.run_envelope.run_id  # type: ignore[union-attr]

    raw_run_json = json.loads((tmp_path / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
    raw_tr = raw_run_json["result"]["steps"][0]["tool_result"]
    assert "data_blob_ref" in raw_tr
    assert "data" not in raw_tr
    assert (tmp_path / "runs" / run_id / "blobs").is_dir()

    loaded = store.load(run_id)
    assert loaded is not None and loaded.result is not None
    assert loaded.result["steps"][0]["tool_result"]["data"] == {"big": large_text}


# ── Pillars 2 & 3: Fast Conversation/Budget Stamping & Responses Cursor ──


def test_conversation_mutation_budget_tier_and_pre_llm_hook_refresh() -> None:
    """Fast stamping preserves nested mutation detection, classify_tier, and hook edits."""
    conv = Conversation()
    conv.append(
        Message(
            role=MessageRole.TOOL,
            content="",
            tool_result=ToolResult(
                tool="lookup",
                args_summary="x=1",
                data={"outer": {"inner": ["alpha", "beta"]}},
            ),
        )
    )
    first_est = conv.serialized_char_estimate()
    budget = ContextBudget(
        context_window=10_000, warning_at=900, error_at=6_000, compact_buffer=1_000
    )
    tier_1, tok_1 = classify_tier(budget, session_log=SessionLog(), conversation=conv)
    assert tier_1 == "ok"
    assert tok_1 == max(1, first_est // 4)

    # Mutate deeply nested list inside ToolResult.data in place!
    conv.messages[0].tool_result.data["outer"]["inner"][1] = "Y" * 4000  # type: ignore[index,union-attr]
    tier_2, tok_2 = classify_tier(budget, session_log=SessionLog(), conversation=conv)
    assert tok_2 > tok_1 + 900
    assert tier_2 == "warning"

    # Verify custom PRE_LLM_CALL hook mutating active_conv still refreshes conversation_input
    observed_messages: list[int] = []

    class MutatingPreLLMHook:
        def on_event(self, payload: Any) -> None:
            if payload.event == LifecycleEvent.PRE_LLM_CALL and payload.step_num == 1:
                conv_ref.append(
                    Message(
                        role=MessageRole.USER,
                        content="injected during PRE_LLM_CALL",
                    )
                )

    conv_ref = Conversation()

    class InspectingBackend:
        conversation_mode = "messages"

        def set_conversation_input(self, snap: dict[str, Any] | None) -> None:
            msgs = snap["conversation"]["messages"] if snap else []
            observed_messages.append(len(msgs))

        def generate(self, prompt: str, **kw: Any) -> str:
            return "ok"

        def generate_with_tools(self, prompt: str, *, tools: Any, **kw: Any) -> Any:
            return [{"type": "tool_use", "id": "d1", "name": "done", "input": {"summary": "ok"}}]

    res = run_result(
        llm=InspectingBackend(),
        tools=tools_from([], include_done=True),
        hooks=[MutatingPreLLMHook()],
        conversation=conv_ref,
        config=LoopConfig(max_steps=2, use_native_tools=True, disabled_defaults=["persistence"]),
    )
    assert res.completed
    assert observed_messages[-1] == 1


@pytest.mark.asyncio
async def test_openai_responses_incremental_cursor_sync_and_async() -> None:
    """OpenAIResponsesBackend & AsyncOpenAIResponsesBackend translate O(delta) per turn."""
    call_num = 0

    def make_fake_response(fn_name: str, args: dict[str, Any], rid: str) -> Any:
        return SimpleNamespace(
            id=rid,
            output=[
                {
                    "type": "function_call",
                    "call_id": f"fc_{rid}",
                    "name": fn_name,
                    "arguments": json.dumps(args),
                }
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8, total_tokens=28),
        )

    class SyncFakeClient:
        class responses:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                nonlocal call_num
                call_num += 1
                if call_num == 3 and "previous_response_id" in kwargs:
                    raise RuntimeError("404 previous_response_id not found")
                if call_num > 1 and "previous_response_id" in kwargs:
                    assert any(
                        item.get("type") == "function_call_output" for item in kwargs["input"]
                    ), f"Missing function_call_output at call {call_num}: {kwargs['input']}"
                return make_fake_response("ping", {"step": call_num}, f"resp_{call_num}")

    backend = OpenAIResponsesBackend(client=SyncFakeClient(), model="gpt-6-luna")
    conv = Conversation()
    translated_counts: list[int] = []
    chat_translations = 0
    orig_slice = backends_mod._translate_responses_slice
    orig_chat = backends_mod._conversation_messages

    def spy_slice(msgs: Any, items: Any, pending: Any) -> Any:
        translated_counts.append(len(msgs))
        return orig_slice(msgs, items, pending)

    def spy_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal chat_translations
        chat_translations += 1
        return orig_chat(*args, **kwargs)

    with (
        patch.object(backends_mod, "_translate_responses_slice", side_effect=spy_slice),
        patch.object(backends_mod, "_conversation_messages", side_effect=spy_chat),
    ):
        for turn in range(20):
            backend._set_detached_conversation_input(
                {
                    "prompt": "Goal",
                    "conversation": conv.serialize(),
                    **({"turn_update": f"Step {turn + 1}/25."} if turn > 0 else {}),
                }
            )
            blocks = backend.generate_with_tools(
                "Goal",
                tools=[
                    {
                        "name": "ping",
                        "description": "p",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            )
            cid = blocks[0]["id"]
            conv.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        ContentBlock(
                            kind="tool_use",
                            data={"id": cid, "name": "ping", "input": {"step": turn}},
                        )
                    ],
                    metadata={"origin": "model"},
                )
            )
            conv.append(
                Message(
                    role=MessageRole.TOOL,
                    content="",
                    tool_result=ToolResult(
                        tool="ping", args_summary="", data={"pong": turn}, call_id=cid
                    ),
                    metadata={"tool_call_id": cid},
                )
            )

    # Zero wasted Chat Completions translations!
    assert chat_translations == 0
    # After turn 0 (0 msgs), every turn translates ONLY 2 delta messages (not O(T))!
    assert sum(translated_counts) == 2 * 19

    # Verify compaction resets cursor cleanly on AsyncOpenAIResponsesBackend too
    async_calls = 0

    class AsyncFakeClient:
        class responses:
            @staticmethod
            async def create(**kwargs: Any) -> Any:
                nonlocal async_calls
                async_calls += 1
                return SimpleNamespace(
                    id=f"aresp_{async_calls}",
                    output_text=f"reply-{async_calls}",
                    output=[],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
                )

    ab = AsyncOpenAIResponsesBackend(client=AsyncFakeClient(), model="gpt-6-luna")
    ab.set_conversation_input({"prompt": "Hello", "conversation": conv.serialize()})
    out1 = await ab.generate("Hello")
    assert out1 == "reply-1"
    assert ab.previous_response_id == "aresp_1"
    assert ab.apply_context_compaction(
        {"prompt": "Hello", "conversation": {"version": 1, "messages": []}}
    )
    assert ab.previous_response_id is None


# ── Pillars 4 & 5: Streaming Subprocess, SSP Reconnect, & SSE Failure Fidelity ──


def test_subprocess_streaming_bounds_ram_and_reports_stderr_truncation(tmp_path: Path) -> None:
    """run_subprocess_tool bounds RAM on multi-MB pipes, splits UTF-8 safely, and spills stderr."""
    spill = tmp_path / "spill"
    # Emit 6 MiB stdout with multi-byte UTF-8 ("é" = 2 bytes) across 16 KiB boundaries + 24 KB stderr
    script = (
        "import sys; "
        "sys.stdout.buffer.write(('A' * 16383 + 'é').encode('utf-8') * 384); "
        "sys.stdout.buffer.flush(); "
        "sys.stderr.write('ERR_LINE\\n' * 3000); "
        "sys.stderr.flush()"
    )
    tracemalloc.start()
    try:
        res = run_subprocess_tool(
            [sys.executable, "-c", script],
            max_output_chars=3000,
            spill_dir=spill,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert res["returncode"] == 0
    assert peak_bytes < 1_500_000  # < 1.5 MiB peak despite >6 MiB child output!
    assert res["stdout_truncated"] is True
    assert res["stderr_truncated"] is True
    assert res["persisted_output_chars"] == 16384 * 384
    assert res["stderr_total_chars"] == 9 * 3000
    assert res["persisted_stderr_chars"] == 9 * 3000
    out_file = Path(res["persisted_output_path"])
    err_file = Path(res["persisted_stderr_path"])
    assert out_file.is_file() and len(out_file.read_text(encoding="utf-8")) == 16384 * 384
    assert "\ufffd" not in out_file.read_text(encoding="utf-8")
    assert err_file.is_file() and len(err_file.read_text(encoding="utf-8")) == 9 * 3000


class _SlowLedgerService(StateServiceBase):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[int] = []

    def slow_append(self, value: int, delay_s: float = 0.15) -> int:
        time.sleep(delay_s)
        self.items.append(value)
        return len(self.items)

    def snapshot(self) -> list[int]:
        return list(self.items)


def test_state_service_timeout_retires_socket_reconnects_and_flags_effect_unknown(
    tmp_path: Path,
) -> None:
    """Timed-out SSP RPC retires poisoned socket, marks effect_unknown, and reconnects cleanly."""
    import threading

    sock_path = str(tmp_path / "ssp.sock")
    svc = _SlowLedgerService()
    srv_thread = threading.Thread(target=svc.serve, args=(sock_path,), daemon=True)
    srv_thread.start()
    try:
        client = StateServiceClient(sock_path, connect_timeout=3.0, rpc_timeout=0.04)
        try:
            with pytest.raises(StateServiceError, match="timed out") as exc_info:
                client.slow_append(42, delay_s=0.18)
            assert exc_info.value.effect_unknown is True

            # Wait for server-side mutation to finish; next call MUST reconnect cleanly
            # without "response id mismatch: expected 3, got 2"!
            time.sleep(0.22)
            client._rpc_timeout = 2.0
            assert client.snapshot() == [42]

            # Also verify BaseToolRegistry propagates effect_unknown into ToolResult.metadata!
            reg = BaseToolRegistry()
            reg.register(
                ToolSpec(
                    name="mutate_remote",
                    description="Mutate state over SSP.",
                    parameters={"val": "int"},
                    execute=lambda val: client._rpc(
                        "state/call",
                        {"name": "slow_append", "args": [int(val), 0.18], "kwargs": {}},
                        timeout_override=0.03,
                    ),
                )
            )
            tr = reg.dispatch(ToolCall(tool="mutate_remote", args={"val": 99}))
            assert tr.error_kind == ErrorKind.TIMEOUT
            assert tr.error_detail is not None
            assert tr.error_detail.context == {"effect_unknown": True}
            assert tr.metadata.get("execution_status") == "effect_unknown"
        finally:
            client.close()
    finally:
        svc._stop.set()
        srv_thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_mid_stream_sse_failure_preserves_partial_text_and_incomplete_tool_use(
    tmp_path: Path,
) -> None:
    """Sync & async OpenAI and Anthropic streaming backends preserve partial output on SSE drop."""

    def make_oai_chunk(
        *,
        text: str | None = None,
        tc_index: int | None = None,
        tc_id: str | None = None,
        fn_name: str | None = None,
        fn_args: str | None = None,
        finish_reason: str | None = None,
    ) -> Any:
        tc_list = None
        if tc_index is not None:
            tc_list = [
                SimpleNamespace(
                    index=tc_index,
                    id=tc_id,
                    function=SimpleNamespace(name=fn_name, arguments=fn_args),
                )
            ]
        return SimpleNamespace(
            id="chatcmpl-partial",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=text, tool_calls=tc_list),
                    finish_reason=finish_reason,
                )
            ],
            usage=None,
        )

    def broken_sync_stream() -> Any:
        yield make_oai_chunk(text="Thinking step 1... ")
        yield make_oai_chunk(
            tc_index=0, tc_id="call_ok", fn_name="read_file", fn_args='{"path":"a.py"}'
        )
        yield make_oai_chunk(finish_reason="tool_calls")
        yield make_oai_chunk(
            tc_index=1, tc_id="call_cut", fn_name="write_file", fn_args='{"path":"b.p'
        )
        raise ConnectionResetError("SSE socket dropped mid-arguments")

    class BrokenSyncOpenAIClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw: Any) -> Any:
                    return broken_sync_stream()

    trace_sync = tmp_path / "sse_sync"
    sink_sync = ProvenanceSink(dir=trace_sync)
    rec_sync = sink_sync.wrap_llm(
        OpenAIStreamingBackend(client=BrokenSyncOpenAIClient(), model="gpt-6-luna")
    )
    chunks_seen: list[str] = []
    tools_seen: list[dict[str, Any]] = []
    with pytest.raises(ConnectionResetError, match="SSE socket dropped"):
        rec_sync.generate_with_tools(
            "Edit files",
            tools=[
                {
                    "name": "read_file",
                    "description": "r",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            on_chunk=lambda ch, _: chunks_seen.append(ch),
            on_tool_call=lambda tc: tools_seen.append(tc),
        )

    assert chunks_seen == ["Thinking step 1... "]
    assert len(tools_seen) == 1 and tools_seen[0]["name"] == "read_file"
    assert len(rec_sync.calls) == 1
    captured_blocks = rec_sync.calls[0].response
    assert captured_blocks == [
        {"type": "text", "text": "Thinking step 1... "},
        {"type": "tool_use", "id": "call_ok", "name": "read_file", "input": {"path": "a.py"}},
        {
            "type": "incomplete_tool_use",
            "id": "call_cut",
            "name": "write_file",
            "partial_arguments": '{"path":"b.p',
        },
    ]
    assert rec_sync.calls[0].capture["stop_reason"] == "stream_error"
    attempts_sync = load_model_attempts(trace_sync)
    assert attempts_sync[0]["outcome"]["response_complete"] is False
    assert attempts_sync[0]["outcome"]["capture"]["response"] == captured_blocks

    # Async Anthropic mid-stream drop
    class BrokenAsyncAnthropicStream:
        async def __aenter__(self) -> "BrokenAsyncAnthropicStream":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        def __aiter__(self) -> Any:
            async def _gen() -> Any:
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Partial Anthropic "),
                )
                yield SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="bash"),
                )
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="input_json_delta", partial_json='{"cmd":"ls '),
                )
                raise ConnectionResetError("Anthropic SSE reset")

            return _gen()

    class BrokenAsyncAnthropicClient:
        class messages:
            @staticmethod
            def stream(**kw: Any) -> Any:
                return BrokenAsyncAnthropicStream()

    anth_backend = AsyncAnthropicBackend(client=BrokenAsyncAnthropicClient())
    anth_backend.stream_in_loop = True
    rec_anth = AsyncRecordingLLMBackend(
        anth_backend, attempt_directory=tmp_path / "sse_anth" / "model-attempts"
    )
    with pytest.raises(ConnectionResetError, match="Anthropic SSE reset"):
        await rec_anth.generate_with_tools(
            "Run ls",
            tools=[
                {
                    "name": "bash",
                    "description": "b",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            on_chunk=lambda ch, _: None,
        )
    assert rec_anth.calls[0].response == [
        {"type": "text", "text": "Partial Anthropic "},
        {
            "type": "incomplete_tool_use",
            "id": "toolu_1",
            "name": "bash",
            "partial_arguments": '{"cmd":"ls ',
        },
    ]


# ── Pillars 6 & 7: Fork/Async Evals, Reload Cadence, O(1) Checkpoint & SQL Paging ──


@pytest.mark.asyncio
async def test_fork_backed_and_async_cartridge_evals_plus_reload_cadence(
    tmp_path: Path,
) -> None:
    """run_cartridge_evals reuses preset.fork() and run_cartridge_evals_async runs concurrently."""
    cart_dir = scaffold_cartridge(tmp_path / "calc.cartridge", name="calc", tools=["greet"])
    (cart_dir / "tools" / "greet" / "tool.yaml").write_text(
        "name: greet\ndescription: Greet user\nparameters:\n  who: str\n",
        encoding="utf-8",
    )
    (cart_dir / "tools" / "greet" / "execute.py").write_text(
        'def execute(who: str) -> dict:\n    return {"greeting": f"hello {who}"}\n',
        encoding="utf-8",
    )
    from looplet import EvalCase, save_cartridge_cases

    evals_dir = cart_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    (evals_dir / "eval_greeting.py").write_text(
        "def eval_completed(ctx):\n"
        '    return ctx.final_output.get("summary") == ctx.task.get("expected", {}).get("summary")\n',
        encoding="utf-8",
    )
    save_cartridge_cases(
        cart_dir,
        [
            EvalCase(
                id=f"case_{idx}",
                task={"goal": f"greet user {idx}"},
                expected={"summary": f"greeted_{idx}"},
            )
            for idx in range(4)
        ],
    )

    import looplet.cartridge as cart_pkg

    load_calls = 0
    orig_c2p = cart_pkg.cartridge_to_preset

    def counting_c2p(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        return orig_c2p(*args, **kwargs)

    def sync_llm_factory() -> Any:
        def _gen(prompt: str, **kw: Any) -> str:
            return "ok"

        def _gen_tools(prompt: str, *, tools: Any, **kw: Any) -> list[dict[str, Any]]:
            for idx in range(4):
                if f"greet user {idx}" in prompt:
                    return [
                        {
                            "type": "tool_use",
                            "id": f"d_{idx}",
                            "name": "done",
                            "input": {"summary": f"greeted_{idx}"},
                        }
                    ]
            return [
                {"type": "tool_use", "id": "d_u", "name": "done", "input": {"summary": "unknown"}}
            ]

        return SimpleNamespace(generate=_gen, generate_with_tools=_gen_tools)

    with patch.object(cart_pkg, "cartridge_to_preset", side_effect=counting_c2p):
        sync_records = run_cartridge_evals(cart_dir, llm=sync_llm_factory, max_concurrency=2)
    assert len(sync_records) == 4
    assert all(all(r.passed for r in rec.results) for rec in sync_records)
    # Built ONCE via template_preset + forked 4 times (instead of 4 full cartridge_to_preset loads)!
    assert load_calls == 1
    for rec in sync_records:
        rec.cleanup()

    # Native async concurrent evals via run_cartridge_evals_async!
    out_async = tmp_path / "async_eval_out"
    closed_count = 0

    def async_llm_factory() -> Any:
        class _AsyncBackend:
            async def close(self) -> None:
                nonlocal closed_count
                closed_count += 1

            async def generate(self, prompt: str, **kw: Any) -> str:
                return "ok"

            async def generate_with_tools(
                self, prompt: str, *, tools: Any, **kw: Any
            ) -> list[dict[str, Any]]:
                await asyncio.sleep(0.01)
                for idx in range(4):
                    if f"greet user {idx}" in prompt:
                        return [
                            {
                                "type": "tool_use",
                                "id": f"ad_{idx}",
                                "name": "done",
                                "input": {"summary": f"greeted_{idx}"},
                            }
                        ]
                return [
                    {
                        "type": "tool_use",
                        "id": "ad_u",
                        "name": "done",
                        "input": {"summary": "unknown"},
                    }
                ]

        return _AsyncBackend()

    async_records = await run_cartridge_evals_async(
        cart_dir,
        llm=async_llm_factory,
        output_dir=out_async,
        max_concurrency=4,
    )
    assert [rec.case.id for rec in async_records if rec.case is not None] == [
        f"case_{i}" for i in range(4)
    ]
    assert all(all(r.passed for r in rec.results) for rec in async_records)
    assert closed_count == 4
    reloaded_eval = load_eval_run(out_async / "case_2")
    assert all(r.passed for r in reloaded_eval.results)

    # ReloadHook check_every_n_steps cadence
    with WorkspaceWatcher(cart_dir) as watcher:
        preset = watcher.preset()
        rhook = ReloadHook(watcher, check_every_n_steps=3)
        fp_calls = 0
        orig_fp = watcher.fingerprint

        def spy_fp() -> dict[str, str]:
            nonlocal fp_calls
            fp_calls += 1
            return orig_fp()

        rhook.bind(
            SimpleNamespace(tools=preset.tools, config=preset.config, resources=preset.resources)
        )
        with patch.object(watcher, "fingerprint", side_effect=spy_fp):
            for step_n in range(1, 7):
                rhook.pre_prompt(preset.state, SessionLog(), None, step_n)
        # Checked on steps 1 and 4 only (2 calls out of 6 steps)!
        assert fp_calls == 2
        assert rhook.provenance_config()["check_every_n_steps"] == 3


def test_indexed_file_checkpoint_store_and_sql_paged_result_store(tmp_path: Path) -> None:
    """FileCheckpointStore O(1) run index avoids scanning sibling runs; DiskBackedDict pages via SQL."""
    ckpt_dir = tmp_path / "ckpts"
    store = FileCheckpointStore(ckpt_dir)
    for r_idx in range(80):
        rid = f"run-{r_idx:03d}"
        for step in (1, 2, 3):
            cp = Checkpoint(
                step_number=step,
                session_log_data={},
                conversation_data=None,
                config_snapshot={},
                tool_results_store={},
                metadata={"r": r_idx},
                run_envelope=RunEnvelope(run_id=rid).to_dict(),
            )
            store.save(cp, f"step_{step}")

    read_calls = 0
    orig_read = store._read

    def counting_read(path: Path, **kw: Any) -> Checkpoint:
        nonlocal read_calls
        read_calls += 1
        return orig_read(path, **kw)

    with patch.object(store, "_read", side_effect=counting_read):
        latest = store.load_latest(run_id="run-042")
        resumed = store.load_latest_for_resume(run_id="run-042")

    assert latest is not None and latest.step_number == 3 and latest.run_id == "run-042"
    assert resumed is not None and resumed.step_number == 3 and resumed.run_id == "run-042"
    # Only the 1 indexed winner per call was read (2 reads total instead of 2 * 240 = 480)!
    assert read_calls == 2

    # Corrupting run-042's file in place invalidates the stat check and fails closed!
    (ckpt_dir / "run-042__step_3.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot safely resume"):
        store.load_latest_for_resume(run_id="run-042")

    # DiskBackedDict.slice_values + FileToolResultStore.list_results over spilled SQLite catalog
    rstore = FileToolResultStore(str(tmp_path / "results"))
    total_items = _HOT_MAX_ITEMS + 40
    keys = [rstore.store_result({"idx": idx, "tag": f"v_{idx}"}) for idx in range(total_items)]
    assert rstore._catalog._mem is None  # Confirmed spilled to SQLite!

    decode_calls = 0
    orig_decode = rstore._catalog._decode

    def counting_decode(val: Any) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return orig_decode(val)

    with patch.object(rstore._catalog, "_decode", side_effect=counting_decode):
        page = rstore.list_results(offset=total_items - 15, limit=10)

    assert page["total_results"] == total_items
    assert len(page["results"]) == 10
    assert [entry["result_key"] for entry in page["results"]] == keys[
        total_items - 15 : total_items - 5
    ]
    assert page["next_offset"] == total_items - 5
    # SQL LIMIT/OFFSET decoded ONLY the 10 requested rows (not total_items)!
    assert decode_calls == 10


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_cursor_survives_turn_update_and_interleaved_subagent_gateway(
    asynchronous: bool,
) -> None:
    """OpenAIResponsesBackend keeps per-conversation cursors intact across turn_update and interleaved subagent calls."""
    calls: list[dict[str, Any]] = []

    def _make_resp(resp_id: str, *, text: str | None = None, call_id: str | None = None) -> Any:
        output: list[Any] = []
        if call_id is not None:
            output.append(
                SimpleNamespace(
                    type="function_call",
                    call_id=call_id,
                    name="delegate",
                    arguments='{"q": "sub"}',
                )
            )
        if text is not None:
            output.append(
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=text)],
                )
            )
        return SimpleNamespace(id=resp_id, status="completed", output=output, usage=None)

    def create_sync(**kwargs: Any) -> Any:
        calls.append(kwargs)
        idx = len(calls)
        if idx == 1:
            return _make_resp("resp_parent_1", call_id="call_p1")
        if idx == 2:
            # Interleaved subagent / ModelGateway call with a different prompt!
            return _make_resp("resp_child_1", text="child_result")
        assert kwargs.get("previous_response_id") == "resp_parent_1"
        assert [
            item["call_id"]
            for item in kwargs["input"]
            if item.get("type") == "function_call_output"
        ] == ["call_p1"]
        return _make_resp("resp_parent_2", text="parent_done")

    async def create_async(**kwargs: Any) -> Any:
        return create_sync(**kwargs)

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_async if asynchronous else create_sync)
    )
    tools = [{"name": "delegate", "description": "d", "input_schema": {"type": "object"}}]

    # Parent Turn 1 (with transient turn_update present!)
    parent_snap_1: dict[str, Any] = {
        "prompt": "Parent task",
        "conversation": {"messages": []},
        "turn_update": "Step 1/8.",
    }
    parent_snap_2: dict[str, Any] = {
        "prompt": "Parent task",
        "conversation": {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "kind": "tool_use",
                            "data": {"id": "call_p1", "name": "delegate", "input": {"q": "sub"}},
                        }
                    ],
                    "metadata": {"origin": "model"},
                },
                {
                    "role": "user",
                    "content": "",
                    "metadata": {"tool_call_id": "call_p1"},
                    "tool_result": {
                        "tool": "delegate",
                        "call_id": "call_p1",
                        "data": {"answer": "child_result"},
                        "error": None,
                        "error_kind": None,
                        "error_retriable": None,
                        "result_key": None,
                        "warnings": [],
                        "metadata": {},
                    },
                },
            ]
        },
        "turn_update": "Step 2/8.",
    }
    if asynchronous:
        abackend = AsyncOpenAIResponsesBackend(client, model="gpt-6-luna")
        abackend.set_conversation_input(parent_snap_1)
        b1 = await abackend.generate_with_tools("Parent task", tools=tools)
        assert b1 == [
            {"type": "tool_use", "id": "call_p1", "name": "delegate", "input": {"q": "sub"}}
        ]
        abackend.set_conversation_input(None)
        c_out = await abackend.generate("Child subagent task")
        assert c_out == "child_result"
        abackend.set_conversation_input(parent_snap_2)
        b2 = await abackend.generate_with_tools("Parent task", tools=tools)
        assert b2 == [{"type": "text", "text": "parent_done"}]
    else:
        sbackend = OpenAIResponsesBackend(client, model="gpt-6-luna")
        sbackend.set_conversation_input(parent_snap_1)
        b1 = sbackend.generate_with_tools("Parent task", tools=tools)
        assert b1 == [
            {"type": "tool_use", "id": "call_p1", "name": "delegate", "input": {"q": "sub"}}
        ]
        sbackend.set_conversation_input(None)
        c_out = sbackend.generate("Child subagent task")
        assert c_out == "child_result"
        sbackend.set_conversation_input(parent_snap_2)
        b2 = sbackend.generate_with_tools("Parent task", tools=tools)
        assert b2 == [{"type": "text", "text": "parent_done"}]
    assert len(calls) == 3
    assert calls[2]["previous_response_id"] == "resp_parent_1"
