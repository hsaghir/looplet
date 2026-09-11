"""Host-facing sync/async and lifecycle conformance tests."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    CancelToken,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    RunEnvelope,
    RunPhase,
    RunResult,
    RunStatus,
    async_composable_loop,
    composable_loop,
    register_done_tool,
)
from looplet.checkpoint import Checkpoint, CheckpointHook, FileCheckpointStore
from looplet.testing import AsyncMockLLMBackend


def _tools() -> BaseToolRegistry:
    tools = BaseToolRegistry()
    register_done_tool(tools)
    return tools


def _envelope() -> RunEnvelope:
    return RunEnvelope(
        run_id="host-run",
        request_id="request-host-run",
        tenant_id="tenant-a",
        actor_id="actor-a",
        deployment="staging",
        model_id="model-a",
        policy_version="policy-1",
        trace_id="trace-host-run",
    )


def test_sync_and_async_hosts_share_run_result_contract():
    envelope = _envelope()
    sync_state = DefaultState(max_steps=1)
    sync_config = LoopConfig(max_steps=1, run_envelope=envelope)
    sync_steps = list(
        composable_loop(
            llm=MockLLMBackend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
            tools=_tools(),
            state=sync_state,
            config=sync_config,
            task={"goal": "finish"},
        )
    )
    sync_result = RunResult.from_state(sync_state, steps=sync_steps)

    assert sync_result.status is RunStatus.COMPLETED
    assert sync_result.phase is RunPhase.TERMINAL
    assert sync_result.termination_reason == "done"
    assert sync_result.output == {"status": "completed", "summary": "ok"}
    assert sync_result.run_envelope == envelope
    assert len(sync_result.steps) == 1


@pytest.mark.asyncio
async def test_async_host_result_matches_sync_shape():
    envelope = _envelope()
    state = DefaultState(max_steps=1)
    steps = []
    async for step in async_composable_loop(
        llm=AsyncMockLLMBackend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
        tools=_tools(),
        state=state,
        config=LoopConfig(max_steps=1, run_envelope=envelope),
        task={"goal": "finish"},
    ):
        steps.append(step)

    result = RunResult.from_state(state, steps=steps)

    assert result.status is RunStatus.COMPLETED
    assert result.phase is RunPhase.TERMINAL
    assert result.termination_reason == "done"
    assert result.output == {"status": "completed", "summary": "ok"}
    assert result.run_envelope == envelope
    assert len(result.steps) == 1


def test_cancelled_result_has_terminal_lifecycle():
    token = CancelToken()
    token.cancel()
    state = DefaultState(max_steps=2)

    list(
        composable_loop(
            llm=MockLLMBackend(responses=[]),
            tools=_tools(),
            state=state,
            config=LoopConfig(max_steps=2, cancel_token=token, run_envelope=_envelope()),
            task={"goal": "cancel"},
        )
    )
    result = RunResult.from_state(state)

    assert result.status is RunStatus.CANCELLED
    assert result.phase is RunPhase.TERMINAL
    assert result.termination_reason == "cancelled"
    assert result.cancelled
    assert result.output is None


def test_checkpoint_preserves_host_lifecycle_context(tmp_path):
    envelope = _envelope()
    state = DefaultState(max_steps=1)
    store = FileCheckpointStore(tmp_path / "checkpoints")

    def build_checkpoint(step_number: int) -> Checkpoint:
        return Checkpoint(
            step_number=step_number,
            session_log_data={"entries": [], "current_theory": ""},
            conversation_data=None,
            config_snapshot={"max_steps": state.max_steps},
            tool_results_store={},
            metadata=dict(state.metadata),
            run_status=str(getattr(state, "run_status", "running")),
            run_phase=str(getattr(state, "run_phase", "dispatching")),
            run_envelope=envelope.to_dict(),
        )

    tools = _tools()
    checkpoint_hook = CheckpointHook(store, build_checkpoint, save_every_n_steps=1)
    list(
        composable_loop(
            llm=MockLLMBackend(responses=['{"tool":"done","args":{"summary":"checkpointed"}}']),
            tools=tools,
            state=state,
            config=LoopConfig(max_steps=1, run_envelope=envelope),
            hooks=[checkpoint_hook],
            task={"goal": "checkpoint"},
        )
    )

    checkpoint = store.load("step_1")
    assert checkpoint is not None
    assert checkpoint.run_envelope == envelope.to_dict()
    assert checkpoint.step_number == 1
    assert checkpoint.run_status
    assert checkpoint.run_phase
