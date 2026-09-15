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


def _preset() -> AgentPreset:
    tools = BaseToolRegistry()
    register_done_tool(tools)
    return AgentPreset(
        config=LoopConfig(max_steps=1),
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

    runtime = AgentRuntime(_preset())
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
