from __future__ import annotations

import asyncio

import pytest

from looplet import (
    AgentPreset,
    AgentRuntime,
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    MemoryRunStore,
    MockLLMBackend,
    RunStatus,
    register_done_tool,
)
from looplet.testing import AsyncMockLLMBackend


def test_diagnostics_match_saved_results_without_exposing_content() -> None:
    import json

    from looplet import RunEnvelope, RunPhase, RunResult
    from looplet.run_records import RunRecord

    class MustNotInspect:
        def __repr__(self):
            pytest.fail("diagnostics must not inspect opaque content")

    result = RunResult(
        status=RunStatus.FAILED,
        phase=RunPhase.TERMINAL,
        termination_reason="private-reason",
        output=MustNotInspect(),
        steps=(),
        run_envelope=RunEnvelope(run_id="same-run"),
        metadata={
            "llm_calls": 3,
            "usage_total": {"input_tokens": 0, "output_tokens": 8},
            "error": "private-error",
            "expected": MustNotInspect(),
            "api_key": "private-key",
        },
    )
    diagnostics = result.diagnostics()
    stored = RunRecord(
        run_id="same-run",
        envelope=result.run_envelope,
        status="failed",
        result={
            "status": "failed",
            "phase": "terminal",
            "termination_reason": "private-reason",
            "metadata": result.metadata,
            "run_envelope": {"run_id": "same-run"},
            "steps": [],
        },
    )
    assert stored.diagnostics() == diagnostics
    assert diagnostics["run_id"] == "same-run"
    assert diagnostics["usage_known"] is True
    assert diagnostics["usage"]["input_tokens"] == 0
    assert diagnostics["cost_usd"] is None
    assert diagnostics["run_error"] is True
    assert "private" not in json.dumps(diagnostics, allow_nan=False)


def test_unknown_usage_and_cost_are_not_zero() -> None:
    from looplet import RunPhase, RunResult

    result = RunResult(
        status=RunStatus.RUNNING, phase=RunPhase.LLM, termination_reason=None, output=None, steps=()
    )
    diagnostics = result.diagnostics()
    assert diagnostics["phase"] == "llm"
    assert diagnostics["llm_calls"] is None
    assert diagnostics["usage"] is None
    assert diagnostics["usage_known"] is False
    assert diagnostics["cost_usd"] is None


@pytest.mark.parametrize("reason", ["budget_exhausted", "cancelled", "llm_error", "done"])
def test_diagnostics_preserve_standard_kernel_stop_reasons(reason: str) -> None:
    from looplet import RunPhase, RunResult
    from looplet.run_records import RunRecord

    result = RunResult(
        status=RunStatus.STOPPED,
        phase=RunPhase.TERMINAL,
        termination_reason=reason,
        output=None,
        steps=(),
    )
    assert result.diagnostics()["termination_reason"] == reason
    assert RunRecord.from_result(result).diagnostics()["termination_reason"] == reason


def test_diagnostics_count_typed_errors_and_reject_invalid_measurements() -> None:
    import json

    from looplet import RunPhase, RunResult
    from looplet.types import ErrorKind, Step, ToolCall, ToolError, ToolResult

    step = Step(
        number=1,
        tool_call=ToolCall(tool="slow", args={}),
        tool_result=ToolResult(
            tool="slow",
            args_summary="private-args",
            data={"secret": "private-output"},
            error="private-error",
            error_detail=ToolError(kind=ErrorKind.TIMEOUT, message="private-error"),
        ),
    )
    result = RunResult(
        status=RunStatus.FAILED,
        phase=RunPhase.TERMINAL,
        termination_reason="llm_error",
        output=None,
        steps=(step,),
        metadata={
            "usage_total": {"input_tokens": True, "output_tokens": float("nan"), "cost_usd": -1},
            "looplet_run_stats": {"llm_calls": 2, "duration_ms": float("inf")},
        },
    )
    diagnostic = result.diagnostics()
    assert diagnostic["tool_error_count"] == 1
    assert diagnostic["tool_error_kinds"] == {"timeout": 1}
    assert diagnostic["usage"] is None
    assert diagnostic["cost_usd"] is None
    assert diagnostic["duration_ms"] is None
    assert diagnostic["run_error"] is True
    assert "private" not in json.dumps(diagnostic, allow_nan=False)


def test_result_does_not_promote_historical_output_without_new_acceptance() -> None:
    from looplet import RunPhase, RunResult
    from looplet.types import Step, ToolCall, ToolResult

    state = DefaultState()
    state.steps.append(
        Step(
            number=1,
            tool_call=ToolCall(tool="done", args={"summary": "old"}),
            tool_result=ToolResult(tool="done", args_summary="", data={"summary": "old"}),
        )
    )
    state.run_status = RunStatus.CANCELLED
    state.run_phase = RunPhase.TERMINAL
    state._accepted_terminal_tool = None
    assert RunResult.from_state(state).output is None
    assert RunResult.from_state(state, tool_name="done").output == {"summary": "old"}


def test_failed_bootstrap_does_not_reuse_previous_run_measurements() -> None:
    from looplet import RunResult, composable_loop

    state = DefaultState()
    state._looplet_run_stats = {"llm_calls": 999, "duration_ms": 999}

    class BrokenBootstrap:
        def pre_loop(self, *args):
            raise RuntimeError("bootstrap failed")

    tools = BaseToolRegistry()
    register_done_tool(tools)
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        list(
            composable_loop(
                llm=MockLLMBackend([]),
                tools=tools,
                state=state,
                config=LoopConfig(use_native_tools=False),
                hooks=[BrokenBootstrap()],
            )
        )
    diagnostic = RunResult.from_state(state).diagnostics()
    assert diagnostic["llm_calls"] == 0
    assert diagnostic["duration_ms"] is None


def test_active_record_retains_unmatched_model_event_and_phase() -> None:
    from looplet import RunEnvelope
    from looplet.run_records import RunEvent, RunRecord

    envelope = RunEnvelope(run_id="active")
    event = RunEvent(
        envelope=envelope,
        sequence=0,
        timestamp=1,
        kind="pre_llm_call",
        payload={"run_phase": "llm"},
    )
    record = RunRecord(run_id="active", envelope=envelope, status="running", events=(event,))
    diagnostic = record.diagnostics()
    assert diagnostic["status"] == "running"
    assert diagnostic["phase"] == "llm"
    assert diagnostic["completed"] is False
    assert diagnostic["event_counts"] == {"pre_llm_call": 1}
    assert diagnostic["usage_known"] is False


def test_failed_runtime_preserves_observed_usage_and_steps() -> None:
    from looplet import RunEnvelope
    from looplet.runtime import _failed_runtime_result

    state = DefaultState()
    state.metadata.update(usage_total={"input_tokens": 7}, llm_calls=2)
    result = _failed_runtime_result(
        state, RunEnvelope(run_id="failed"), RuntimeError("private-error")
    )
    diagnostic = result.diagnostics()
    assert diagnostic["usage"]["input_tokens"] == 7
    assert diagnostic["llm_calls"] == 2
    assert diagnostic["run_error"] is True
    assert state.metadata == {"usage_total": {"input_tokens": 7}, "llm_calls": 2}


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize("entrypoint", ["direct", "preset", "runtime"])
@pytest.mark.parametrize("terminal_name", ["done", "escalate"])
async def test_diagnostics_share_observed_counters_across_entrypoints(
    async_run, entrypoint, terminal_name
):
    import json

    from looplet import EvalHook, RunEnvelope, RunResult, async_composable_loop, composable_loop
    from looplet.tools import ToolSpec

    preset = _preset(use_native_tools=False)
    if terminal_name == "escalate":
        preset.config.done_tools = [terminal_name]
        preset.tools.register(
            ToolSpec(
                name=terminal_name,
                description="Escalate",
                parameters={"summary": "str"},
                execute=lambda *, summary: {"summary": summary},
            )
        )
    eval_hook = EvalHook(evaluators=[])
    preset.hooks.append(eval_hook)
    preset.config.run_envelope = RunEnvelope(run_id=f"{entrypoint}-{async_run}")
    backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
        [json.dumps({"tool": terminal_name, "args": {"summary": "ok"}})]
    )
    try:
        if entrypoint == "runtime":
            with AgentRuntime(preset) as runtime:
                result = await runtime.run_async(backend) if async_run else runtime.run(backend)
        elif entrypoint == "preset":
            if async_run:
                async for _ in preset.run_async(backend):
                    pass
            else:
                for _ in preset.run(backend):
                    pass
            result = RunResult.from_state(preset.state)
        else:
            kwargs = dict(
                llm=backend,
                tools=preset.tools,
                state=preset.state,
                config=preset.config,
                hooks=preset.hooks,
            )
            if async_run:
                async for _ in async_composable_loop(**kwargs):
                    pass
            else:
                for _ in composable_loop(**kwargs):
                    pass
            result = RunResult.from_state(preset.state)
        diagnostic = result.diagnostics()
        assert diagnostic["completed"] is True
        assert diagnostic["phase"] == "terminal"
        assert diagnostic["run_id"] is not None
        assert diagnostic["llm_calls"] == 1
        assert diagnostic["duration_ms"] >= 0
        assert diagnostic["usage_known"] is False
        assert diagnostic["cost_usd"] is None
        assert result.output is not None
        assert result.output["summary"] == "ok"
        if terminal_name == "escalate":
            assert RunResult.from_state(preset.state, tool_name="done").output is None
        assert eval_hook.context is not None
        assert eval_hook.context.final_output["summary"] == "ok"
        assert eval_hook.context.diagnostics()["llm_calls"] == diagnostic["llm_calls"]
        assert eval_hook.context.diagnostics()["run_id"] == diagnostic["run_id"]
    finally:
        preset.close()


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize(
    "fault", ["create", "append_event", "save_checkpoint", "complete", "session"]
)
async def test_runtime_faults_share_restoration_and_persistence_contract(async_run, fault):
    from looplet import CancelToken, RunEnvelope

    class BrokenStore(MemoryRunStore):
        def create(self, *args, **kwargs):
            if fault == "create":
                raise OSError("injected create")
            return super().create(*args, **kwargs)

        def append_event(self, *args, **kwargs):
            if fault == "append_event":
                raise OSError("injected append_event")
            return super().append_event(*args, **kwargs)

        def save_checkpoint(self, *args, **kwargs):
            if fault == "save_checkpoint":
                raise OSError("injected save_checkpoint")
            return super().save_checkpoint(*args, **kwargs)

        def complete(self, *args, **kwargs):
            if fault == "complete":
                raise OSError("injected complete")
            return super().complete(*args, **kwargs)

    class BrokenSession:
        def attach(self, run_id):
            raise OSError("injected session")

        def close(self):
            pass

    preset = _preset(use_native_tools=False)
    original_envelope = RunEnvelope(run_id="original")
    original_token = CancelToken()
    preset.config.run_envelope = original_envelope
    preset.config.cancel_token = original_token
    backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
        ['{"tool":"done","args":{"summary":"ok"}}']
    )
    with AgentRuntime(
        preset,
        store=BrokenStore(),
        session=BrokenSession() if fault == "session" else None,
        checkpoint_every_n_steps=1,
    ) as runtime:
        result = await runtime.run_async(backend) if async_run else runtime.run(backend)
        assert not runtime._active_tokens

    assert preset.config.run_envelope is original_envelope
    assert preset.config.cancel_token is original_token
    assert result.status is (RunStatus.FAILED if fault == "session" else RunStatus.COMPLETED)
    assert result.metadata["runtime_duration_ms"] >= 0
    if fault == "session":
        assert "injected session" in result.metadata["error"]
    else:
        assert any(
            f"injected {fault}" in warning for warning in result.metadata["persistence_warnings"]
        )


@pytest.mark.parametrize("store_kind", ["none", "memory", "file"])
@pytest.mark.parametrize("checkpoints", [False, True])
@pytest.mark.parametrize("flush_fault", ["none", "oserror", "cancelled"])
async def test_external_async_cancellation_restores_and_releases_runtime(
    tmp_path, monkeypatch, caplog, store_kind, checkpoints, flush_fault
):
    from looplet import FileRunStore, RunEnvelope, RunPhase, RunResult
    from looplet.provenance import ProvenanceSink

    entered = asyncio.Event()
    blocked = asyncio.Event()

    class Backend:
        async def generate(self, prompt, **kwargs):
            entered.set()
            await blocked.wait()
            return '{"tool":"done","args":{"summary":"ok"}}'

    preset = _preset(use_native_tools=False)
    preset.config.harness_profile = "execution-v1"
    preset.config.run_dir = str(tmp_path / "evidence")
    envelope = preset.config.run_envelope
    token = preset.config.cancel_token
    store = (
        MemoryRunStore()
        if store_kind == "memory"
        else FileRunStore(tmp_path / "runs")
        if store_kind == "file"
        else None
    )
    runtime = AgentRuntime(preset, store=store, checkpoint_every_n_steps=1 if checkpoints else None)
    run_envelope = RunEnvelope(run_id="external-cancel")
    original_flush = ProvenanceSink.flush

    def fail_flush(self):
        if flush_fault == "oserror":
            raise OSError("private flush failure details")
        raise asyncio.CancelledError("secondary persistence cancellation")

    if flush_fault != "none":
        monkeypatch.setattr(ProvenanceSink, "flush", fail_flush)
    pending = asyncio.create_task(runtime.run_async(Backend(), envelope=run_envelope))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel("host request")
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        assert caught.value.args == ("host request",)
        assert pending.cancelled()
        flush_warnings = [
            record.message
            for record in caplog.records
            if "Persistence failed during async cancellation" in record.message
        ]
        if flush_fault == "none":
            assert not flush_warnings
        else:
            category = "OSError" if flush_fault == "oserror" else "CancelledError"
            assert flush_warnings == [f"Persistence failed during async cancellation ({category})"]
            assert "private flush failure details" not in caplog.text
            assert "secondary persistence cancellation" not in caplog.text
        assert not runtime._active_tokens
        assert preset.config.run_envelope is envelope
        assert preset.config.cancel_token is token
        result = RunResult.from_state(preset.state)
        assert result.status is RunStatus.CANCELLED
        assert result.phase is RunPhase.TERMINAL
        assert result.termination_reason == "cancelled"
        assert result.output is None
        if store is not None:
            record = store.load(run_envelope.run_id)
            assert record is not None
            assert record.status == "cancelled"
            assert record.result is not None
            assert record.result["status"] == "cancelled"
            assert record.result["phase"] == "terminal"
            assert record.result["termination_reason"] == "cancelled"
            assert record.result["run_envelope"] == run_envelope.to_dict()
            assert record.result["metadata"]["runtime_duration_ms"] >= 0
            assert bool(record.checkpoint_keys) is checkpoints
            if checkpoints:
                checkpoint = store.load_checkpoint(run_envelope.run_id, record.checkpoint_keys[-1])
                assert checkpoint is not None
                assert checkpoint.is_terminal
                assert checkpoint.run_status == "cancelled"
                assert checkpoint.run_phase == "terminal"
                assert checkpoint.termination_reason == "cancelled"
        monkeypatch.setattr(ProvenanceSink, "flush", original_flush)
    finally:
        blocked.set()
        runtime.close()
    fresh_preset = _preset(use_native_tools=False)
    fresh_preset.tools = preset.tools
    fresh_preset.config.run_dir = str(tmp_path / "subsequent-evidence")
    with AgentRuntime(fresh_preset, store=store) as fresh_runtime:
        subsequent = await fresh_runtime.run_async(
            AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"reused"}}']),
            envelope=RunEnvelope(run_id="after-external-cancel"),
        )
        assert subsequent.completed
        assert not fresh_runtime._active_tokens


async def test_evidence_flush_failure_without_cancellation_remains_failure(tmp_path, monkeypatch):
    from looplet import RunEnvelope
    from looplet.provenance import ProvenanceSink

    def fail_flush(self):
        raise OSError("injected evidence flush failure")

    preset = _preset(use_native_tools=False)
    preset.config.harness_profile = "execution-v1"
    preset.config.run_dir = str(tmp_path / "evidence")
    store = MemoryRunStore()
    monkeypatch.setattr(ProvenanceSink, "flush", fail_flush)
    with AgentRuntime(preset, store=store) as runtime:
        result = await runtime.run_async(
            AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"finished"}}']),
            envelope=RunEnvelope(run_id="flush-failure"),
        )
    assert result.status is RunStatus.FAILED
    assert store.load("flush-failure").status == "failed"


def _preset(*, use_native_tools: bool = True) -> AgentPreset:
    tools = BaseToolRegistry()
    register_done_tool(tools)
    return AgentPreset(
        config=LoopConfig(max_steps=1, use_native_tools=use_native_tools),
        hooks=[],
        tools=tools,
        state=DefaultState(max_steps=1),
    )


def test_runtime_runs_and_persists_events() -> None:
    store = MemoryRunStore()
    with AgentRuntime(_preset(), store=store) as runtime:
        result = runtime.run(
            MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={"goal": "finish"},
        )

    assert result.status is RunStatus.COMPLETED
    record = store.load(result.run_envelope.run_id)
    assert record is not None
    assert record.status == "completed"
    assert any(event.kind == "session_start" for event in record.events)


@pytest.mark.asyncio
async def test_runtime_runs_async_loop() -> None:
    with AgentRuntime(_preset()) as runtime:
        result = await runtime.run_async(
            AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={"goal": "finish"},
        )

    assert result.status is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_runtime_rejects_concurrent_async_runs_on_shared_preset() -> None:
    runtime = AgentRuntime(_preset())

    async def run_once():
        return await runtime.run_async(
            AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={},
        )

    first, second = await asyncio.gather(run_once(), run_once(), return_exceptions=True)
    try:
        outcomes = (first, second)
        assert sum(isinstance(outcome, RuntimeError) for outcome in outcomes) == 1
        assert (
            sum(getattr(outcome, "status", None) is RunStatus.COMPLETED for outcome in outcomes)
            == 1
        )
    finally:
        runtime.close()


def test_runtime_close_cancels_active_handle() -> None:
    class SlowBackend:
        def generate(self, prompt, **kwargs):
            import time

            time.sleep(0.2)
            return '{"tool":"done","args":{"summary":"ok"}}'

    runtime = AgentRuntime(_preset(use_native_tools=False))
    handle = runtime.start(SlowBackend())
    report = runtime.close(timeout=1)

    assert not report.errors
    assert handle.result(timeout=1).status in {RunStatus.CANCELLED, RunStatus.COMPLETED}


def test_runtime_start_returns_handle() -> None:
    runtime = AgentRuntime(_preset())
    handle = runtime.start(MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']))
    try:
        result = handle.result(timeout=5)
        assert result.status is RunStatus.COMPLETED
        assert handle.run_id == result.run_envelope.run_id
        assert handle.events()
    finally:
        runtime.close()


def test_runtime_store_can_checkpoint_each_step() -> None:
    store = MemoryRunStore()
    with AgentRuntime(_preset(), store=store, checkpoint_every_n_steps=1) as runtime:
        result = runtime.run(
            MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={"goal": "finish"},
        )

    record = store.load(result.run_envelope.run_id)
    assert record is not None
    assert record.checkpoint_keys == ("step_1",)
    checkpoint = store.load_checkpoint(result.run_envelope.run_id, "step_1")
    assert checkpoint is not None
    assert checkpoint.is_terminal
    assert checkpoint.run_status == "completed"
    assert checkpoint.session_log_data["entries"]


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize("store_kind", ["memory", "file"])
@pytest.mark.parametrize("profile", ["minimal", "execution-v1"])
async def test_runtime_checkpoint_capture_is_detached_and_io_respects_owner(
    tmp_path, monkeypatch, async_run, store_kind, profile
):
    from threading import get_ident

    from looplet import FileRunStore, tools_from

    owner = get_ident()
    ledger = {"remaining": 17}
    retained_capture = {"host_ledger": ledger}
    capture_threads = []
    write_threads = []
    backend_threads = []

    def spend():
        ledger["remaining"] -= 6
        return {"spent": 6}

    def capture(context):
        capture_threads.append(get_ident())
        return retained_capture

    class Backend(AsyncMockLLMBackend if async_run else MockLLMBackend):
        def checkpoint_state(self):
            backend_threads.append(get_ident())
            assert get_ident() == owner
            return {"calls": self.calls}

    preset = AgentPreset(
        tools=tools_from([spend], include_done=True, done_parameters={"summary": "Outcome"}),
        state=DefaultState(max_steps=2),
        hooks=[],
        config=LoopConfig(
            max_steps=2,
            use_native_tools=False,
            harness_profile=profile,
            run_dir=str(tmp_path / "evidence"),
            checkpoint_state=capture,
        ),
    )
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path / "runs")
    original_save = store.save_checkpoint

    def save(run_id, checkpoint):
        write_threads.append(get_ident())
        return original_save(run_id, checkpoint)

    monkeypatch.setattr(store, "save_checkpoint", save)
    backend = Backend(['{"tool":"spend","args":{}}', '{"tool":"done","args":{"summary":"ok"}}'])
    with AgentRuntime(preset, store=store, checkpoint_every_n_steps=9) as runtime:
        result = await runtime.run_async(backend) if async_run else runtime.run(backend)
    assert result.completed
    assert "persistence_warnings" not in result.metadata
    assert ledger == {"remaining": 11}
    assert set(retained_capture) == {"host_ledger"}
    checkpoint = store.load_checkpoint(result.run_envelope.run_id, "step_2")
    assert checkpoint is not None and checkpoint.is_terminal
    assert checkpoint.domain_state == {"host_ledger": {"remaining": 11}, "llm": {"calls": 2}}
    assert "pending_calls" not in checkpoint.metadata
    ledger["remaining"] = 0
    assert checkpoint.domain_state["host_ledger"] == {"remaining": 11}
    assert capture_threads and all(thread == owner for thread in capture_threads)
    assert backend_threads and all(thread == owner for thread in backend_threads)
    assert write_threads
    assert all((thread != owner) is async_run for thread in write_threads)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("store_kind", ["memory", "file"])
@pytest.mark.parametrize("mirror", [False, True])
async def test_runtime_checkpoint_keeps_unknown_effect_until_acknowledged(
    tmp_path, native, store_kind, mirror
):
    from looplet import FileRunStore, RunEnvelope, tools_from
    from looplet.checkpoint import FileCheckpointStore, UncertainToolEffect, resume_loop_state

    entered = asyncio.Event()
    marker = tmp_path / "effect.txt"
    ledger = {"effects": 0}

    async def effect():
        marker.write_text("effect applied\n", encoding="ascii")
        ledger["effects"] += 1
        entered.set()
        await asyncio.get_running_loop().create_future()

    envelope = RunEnvelope(run_id="interrupted-effect")
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path / "runs")
    preset = AgentPreset(
        tools=tools_from([effect], include_done=True),
        state=DefaultState(max_steps=1),
        hooks=[],
        config=LoopConfig(
            max_steps=1,
            use_native_tools=native,
            run_dir=str(tmp_path / "evidence"),
            checkpoint_dir=str(tmp_path / "checkpoints") if mirror else None,
            checkpoint_state=lambda context: {"host_ledger": dict(ledger)},
        ),
    )
    runtime = AgentRuntime(preset, store=store, checkpoint_every_n_steps=9)
    pending = asyncio.create_task(
        runtime.run_async(AsyncMockLLMBackend(['{"tool":"effect","args":{}}']), envelope=envelope)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel("host interruption")
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        assert caught.value.args == ("host interruption",)
        assert not runtime._active_tokens
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        runtime.close()
    assert marker.read_text(encoding="ascii") == "effect applied\n"
    assert ledger == {"effects": 1}
    record = store.load(envelope.run_id)
    assert record is not None and record.status == "cancelled"
    checkpoint = store.load_checkpoint(envelope.run_id, record.checkpoint_keys[-1])
    assert checkpoint is not None and checkpoint.is_terminal
    assert checkpoint.run_status == "cancelled"
    assert checkpoint.domain_state["host_ledger"] == {"effects": 1}
    assert checkpoint.metadata["pending_calls"][0]["execution_status"] == "effect_unknown"
    with pytest.raises(UncertainToolEffect):
        resume_loop_state(checkpoint)
    if mirror:
        mirrored = FileCheckpointStore(tmp_path / "checkpoints").load_latest(run_id=envelope.run_id)
        assert mirrored is not None and mirrored.is_terminal
        with pytest.raises(UncertainToolEffect):
            resume_loop_state(mirrored)


@pytest.mark.parametrize("store_kind", ["memory", "file"])
@pytest.mark.parametrize("mirror", [False, True])
async def test_runtime_checkpoint_preserves_interrupted_model_admission(
    tmp_path, store_kind, mirror
):
    from looplet import FileRunStore, RunEnvelope
    from looplet.assistant_turns import TURN_STATE_KEY
    from looplet.checkpoint import UncertainModelCall, resume_loop_state

    entered = asyncio.Event()

    class Backend:
        async def generate(self, prompt, **kwargs):
            entered.set()
            await asyncio.get_running_loop().create_future()

    envelope = RunEnvelope(run_id="interrupted-model")
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path / "runs")
    preset = _preset(use_native_tools=False)
    preset.config.max_model_calls = 3
    preset.config.run_dir = str(tmp_path / "evidence")
    preset.config.checkpoint_dir = str(tmp_path / "checkpoints") if mirror else None
    runtime = AgentRuntime(preset, store=store, checkpoint_every_n_steps=9)
    pending = asyncio.create_task(runtime.run_async(Backend(), envelope=envelope))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending_record = store.load(envelope.run_id)
        assert pending_record is not None and pending_record.checkpoint_keys
        uncertain = store.load_checkpoint(envelope.run_id, pending_record.checkpoint_keys[-1])
        assert uncertain is not None
        assert uncertain.metadata[TURN_STATE_KEY]["pending_model_call"]
        pending.cancel("host interruption")
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        runtime.close()
    record = store.load(envelope.run_id)
    assert record is not None and record.status == "cancelled"
    checkpoint = store.load_checkpoint(envelope.run_id, record.checkpoint_keys[-1])
    assert checkpoint is not None and checkpoint.is_terminal
    assert checkpoint.config_snapshot["max_model_calls"] == 3
    assert checkpoint.metadata[TURN_STATE_KEY]["model_calls"] == 1
    assert "pending_model_call" not in checkpoint.metadata[TURN_STATE_KEY]
    with pytest.raises(UncertainModelCall):
        resume_loop_state(uncertain)


@pytest.mark.parametrize("async_run", [False, True])
@pytest.mark.parametrize("store_kind", ["memory", "file"])
@pytest.mark.parametrize("abrupt", [False, True])
async def test_runtime_request_guard_distinguishes_abrupt_from_observed_failure(
    tmp_path, async_run, store_kind, abrupt
):
    from looplet import FileRunStore, RunEnvelope
    from looplet.assistant_turns import TURN_STATE_KEY
    from looplet.checkpoint import UncertainModelCall, resume_loop_state

    class Crash(BaseException):
        pass

    marker = tmp_path / "request.txt"

    class Backend(MockLLMBackend):
        def fail(self):
            self.calls += 1
            marker.write_text("request admitted\n", encoding="ascii")
            if abrupt:
                raise Crash("abrupt request interruption")
            raise RuntimeError("observed backend failure")

        if async_run:

            async def generate(self, prompt, **kwargs):
                return self.fail()

        else:

            def generate(self, prompt, **kwargs):
                return self.fail()

    envelope = RunEnvelope(run_id="request-failure")
    store = MemoryRunStore() if store_kind == "memory" else FileRunStore(tmp_path / "runs")
    preset = _preset(use_native_tools=False)
    preset.config.max_model_calls = 1
    preset.config.run_dir = str(tmp_path / "evidence")
    backend = Backend([])
    with AgentRuntime(preset, store=store, checkpoint_every_n_steps=9) as runtime:
        if abrupt:
            with pytest.raises(Crash):
                if async_run:
                    await runtime.run_async(backend, envelope=envelope)
                else:
                    runtime.run(backend, envelope=envelope)
        else:
            result = (
                await runtime.run_async(backend, envelope=envelope)
                if async_run
                else runtime.run(backend, envelope=envelope)
            )
            assert not result.completed
        assert not runtime._active_tokens
    assert backend.calls == 1
    assert marker.read_text(encoding="ascii") == "request admitted\n"
    record = store.load(envelope.run_id)
    assert record is not None and record.checkpoint_keys
    checkpoint = store.load_checkpoint(envelope.run_id, record.checkpoint_keys[-1])
    assert checkpoint is not None
    assert checkpoint.metadata[TURN_STATE_KEY]["model_calls"] == 1
    assert checkpoint.config_snapshot["max_model_calls"] == 1
    if abrupt:
        with pytest.raises(UncertainModelCall):
            resume_loop_state(checkpoint)
    else:
        assert "pending_model_call" not in checkpoint.metadata[TURN_STATE_KEY]


def test_runtime_failed_run_persists_terminal_checkpoint() -> None:
    class FailingBackend:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("backend failed")

    store = MemoryRunStore()
    with AgentRuntime(_preset(), store=store, checkpoint_every_n_steps=1) as runtime:
        result = runtime.run(FailingBackend(), task={})

    checkpoint = store.load_checkpoint(result.run_envelope.run_id, "step_1")
    assert result.status is RunStatus.FAILED
    assert checkpoint is not None
    assert checkpoint.is_terminal
    assert checkpoint.run_status == "failed"


def test_runtime_checkpoint_carries_checkpointable_llm_state() -> None:
    class CheckpointLLM(MockLLMBackend):
        def __init__(self):
            super().__init__(['{"tool":"done","args":{"summary":"ok"}}'])

        def checkpoint_state(self):
            return {"provider_response_id": "resp-1"}

    store = MemoryRunStore()
    with AgentRuntime(_preset(), store=store, checkpoint_every_n_steps=1) as runtime:
        result = runtime.run(CheckpointLLM(), task={})

    checkpoint = store.load_checkpoint(result.run_envelope.run_id, "step_1")
    assert checkpoint is not None
    assert checkpoint.domain_state["llm"] == {"provider_response_id": "resp-1"}


def test_runtime_checkpoint_failure_is_reported_without_masking_result() -> None:
    class BrokenStore(MemoryRunStore):
        def save_checkpoint(self, run_id, checkpoint):
            raise OSError("checkpoint disk full")

    store = BrokenStore()
    with AgentRuntime(_preset(), store=store, checkpoint_every_n_steps=1) as runtime:
        result = runtime.run(
            MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={},
        )

    assert result.status is RunStatus.COMPLETED
    assert any("checkpoint" in warning for warning in result.metadata["persistence_warnings"])


def test_runtime_completion_store_failure_is_reported_without_masking_result() -> None:
    class BrokenStore(MemoryRunStore):
        def complete(self, run_id, result, *, artifacts=()):
            raise OSError("store unavailable")

    with AgentRuntime(_preset(), store=BrokenStore()) as runtime:
        result = runtime.run(
            MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={},
        )

    assert result.status is RunStatus.COMPLETED
    assert any("complete" in warning for warning in result.metadata["persistence_warnings"])


def test_runtime_session_attach_failure_returns_failed_result_and_restores_config() -> None:
    class BrokenSession:
        def attach(self, run_id):
            raise OSError("session unavailable")

        def close(self):
            pass

    preset = _preset()
    old_envelope = preset.config.run_envelope
    old_token = preset.config.cancel_token
    with AgentRuntime(preset, session=BrokenSession()) as runtime:
        result = runtime.run(MockLLMBackend(), task={})

    assert result.status is RunStatus.FAILED
    assert "session unavailable" in result.metadata["error"]
    assert preset.config.run_envelope is old_envelope
    assert preset.config.cancel_token is old_token


def test_runtime_store_create_failure_returns_failed_result_and_restores_config() -> None:
    class BrokenStore(MemoryRunStore):
        def create(self, envelope, *, metadata=None):
            raise OSError("store unavailable")

    preset = _preset()
    old_envelope = preset.config.run_envelope
    old_token = preset.config.cancel_token
    with AgentRuntime(preset, store=BrokenStore()) as runtime:
        result = runtime.run(MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']), task={})

    assert result.status is RunStatus.COMPLETED
    assert any("create" in warning for warning in result.metadata["persistence_warnings"])
    assert preset.config.run_envelope is old_envelope
    assert preset.config.cancel_token is old_token


def test_runtime_provider_checkpoint_failure_is_reported_without_masking_result() -> None:
    class BrokenBackend(MockLLMBackend):
        def checkpoint_state(self):
            raise RuntimeError("provider state unavailable")

    with AgentRuntime(_preset(), store=MemoryRunStore(), checkpoint_every_n_steps=1) as runtime:
        result = runtime.run(
            BrokenBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={},
        )

    assert result.status is RunStatus.COMPLETED
    assert any(
        "provider state unavailable" in warning
        for warning in result.metadata["persistence_warnings"]
    )


@pytest.mark.asyncio
async def test_async_runtime_checkpoint_failure_is_reported_without_masking_result() -> None:
    class BrokenStore(MemoryRunStore):
        def save_checkpoint(self, run_id, checkpoint):
            raise OSError("async checkpoint unavailable")

    with AgentRuntime(_preset(), store=BrokenStore(), checkpoint_every_n_steps=1) as runtime:
        result = await runtime.run_async(
            AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={},
        )

    assert result.status is RunStatus.COMPLETED
    assert any(
        "async checkpoint unavailable" in warning
        for warning in result.metadata["persistence_warnings"]
    )


@pytest.mark.asyncio
async def test_runtime_handle_wait_shuts_down_executor() -> None:
    runtime = AgentRuntime(_preset())
    handle = runtime.start(MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']))
    try:
        result = await handle.wait()
        assert result.status is RunStatus.COMPLETED
        assert handle._executor._shutdown is True
    finally:
        runtime.close()


def test_runtime_rejects_reuse_after_first_run() -> None:
    runtime = AgentRuntime(_preset())
    try:
        runtime.run(
            MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            task={},
        )
        with pytest.raises(RuntimeError, match="single-use"):
            runtime.run(
                MockLLMBackend(['{"tool":"done","args":{"summary":"again"}}']),
                task={},
            )
    finally:
        runtime.close()
