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


async def test_external_async_cancellation_restores_and_releases_runtime():
    entered = asyncio.Event()
    blocked = asyncio.Event()

    class Backend:
        async def generate(self, prompt, **kwargs):
            entered.set()
            await blocked.wait()
            return '{"tool":"done","args":{"summary":"ok"}}'

    preset = _preset(use_native_tools=False)
    envelope = preset.config.run_envelope
    token = preset.config.cancel_token
    runtime = AgentRuntime(preset)
    pending = asyncio.create_task(runtime.run_async(Backend()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not runtime._active_tokens
        assert preset.config.run_envelope is envelope
        assert preset.config.cancel_token is token
    finally:
        blocked.set()
        runtime.close()


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
