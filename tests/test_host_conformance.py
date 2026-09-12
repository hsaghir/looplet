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
    ToolSpec,
    async_composable_loop,
    composable_loop,
    register_done_tool,
)
from looplet.checkpoint import Checkpoint, CheckpointHook, FileCheckpointStore, resume_loop_state
from looplet.conversation import Conversation, Message, MessageRole
from looplet.testing import AsyncMockLLMBackend


class _MetadataOnlyState:
    """AgentState-compatible object that rejects dynamic lifecycle fields."""

    __slots__ = ("steps", "queries_used", "max_steps", "metadata", "step_context")

    def __init__(self, max_steps: int = 1):
        self.steps = []
        self.queries_used = 0
        self.max_steps = max_steps
        self.metadata = {}
        self.step_context = {}

    @property
    def step_count(self):
        return len(self.steps)

    @property
    def budget_remaining(self):
        return max(0, self.max_steps - len(self.steps))

    def context_summary(self):
        return ""

    def snapshot(self):
        return {}


def _tools() -> BaseToolRegistry:
    tools = BaseToolRegistry()
    register_done_tool(tools)
    return tools


class _RecallRegistry(BaseToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, object] = {}

    def _store_result(self, call, result_data):
        key = f"result-{len(self.values)}"
        self.values[key] = result_data
        return key

    def recall(self, key: str):
        return self.values[key]

    def snapshot_results(self):
        return dict(self.values)

    def restore_results(self, snapshot):
        self.values = dict(snapshot)


def _recall_tools() -> _RecallRegistry:
    tools = _RecallRegistry()
    tools.register(
        ToolSpec(name="search", description="search", parameters={}, execute=lambda: {"answer": 42})
    )
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
    resumed = resume_loop_state(checkpoint)
    assert resumed["run_envelope"] == envelope.to_dict()
    assert resumed["step_offset"] == 1


def test_sync_checkpoint_restores_custom_result_store(tmp_path):
    tools = _recall_tools()
    list(
        composable_loop(
            llm=MockLLMBackend(['{"tool":"search","args":{}}']),
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(max_steps=1, checkpoint_dir=tmp_path),
            task={},
        )
    )
    checkpoint = FileCheckpointStore(tmp_path).load("step_1")
    assert checkpoint is not None

    fresh = _recall_tools()
    list(
        composable_loop(
            llm=MockLLMBackend(['{"tool":"done","args":{"summary":"resumed"}}']),
            tools=fresh,
            state=DefaultState(max_steps=2),
            config=LoopConfig(max_steps=2, initial_checkpoint=checkpoint),
            task={},
        )
    )
    assert fresh.recall("result-0") == {"answer": 42}


@pytest.mark.asyncio
async def test_async_checkpoint_restores_custom_result_store(tmp_path):
    tools = _recall_tools()
    async for _ in async_composable_loop(
        llm=AsyncMockLLMBackend(responses=['{"tool":"search","args":{}}']),
        tools=tools,
        state=DefaultState(max_steps=1),
        config=LoopConfig(max_steps=1, checkpoint_dir=tmp_path),
        task={},
    ):
        pass
    checkpoint = FileCheckpointStore(tmp_path).load("step_1")
    assert checkpoint is not None

    fresh = _recall_tools()
    async for _ in async_composable_loop(
        llm=AsyncMockLLMBackend(responses=['{"tool":"done","args":{"summary":"resumed"}}']),
        tools=fresh,
        state=DefaultState(max_steps=2),
        config=LoopConfig(max_steps=2, initial_checkpoint=checkpoint),
        task={},
    ):
        pass
    assert fresh.recall("result-0") == {"answer": 42}


def test_sync_checkpoint_restores_conversation_context():
    conversation = Conversation([Message(MessageRole.USER, "remember this context")])
    checkpoint = Checkpoint(
        step_number=1,
        session_log_data={"entries": [], "current_theory": ""},
        conversation_data=conversation.serialize(),
        config_snapshot={"max_steps": 2},
        tool_results_store={},
        metadata={},
    )
    llm = MockLLMBackend(
        responses=['{"tool":"done","args":{"summary":"resumed"}}'],
        cycle=False,
    )
    state = DefaultState(max_steps=2)

    list(
        composable_loop(
            llm=llm,
            tools=_tools(),
            state=state,
            config=LoopConfig(max_steps=2, initial_checkpoint=checkpoint),
            task={"goal": "continue"},
        )
    )

    restored = getattr(state, "conversation", None)
    assert restored is not None
    assert any(message.text == "remember this context" for message in restored.messages)


def test_sync_host_result_supports_metadata_only_state():
    state = _MetadataOnlyState()
    list(
        composable_loop(
            llm=MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            tools=_tools(),
            state=state,
            config=LoopConfig(max_steps=1),
            task={"goal": "finish"},
        )
    )

    result = RunResult.from_state(state)

    assert result.completed
    assert result.phase is RunPhase.TERMINAL
    assert result.termination_reason == "done"


def test_run_result_to_dict_is_json_safe():
    import json

    result = RunResult(
        status=RunStatus.COMPLETED,
        phase=RunPhase.TERMINAL,
        termination_reason="done",
        output={"value": object()},
        steps=(),
        metadata={"custom": object()},
    )

    payload = result.to_dict()

    json.dumps(payload)
    assert isinstance(payload["output"]["value"], str)
    assert isinstance(payload["metadata"]["custom"], str)


@pytest.mark.asyncio
async def test_async_host_result_supports_metadata_only_state():
    state = _MetadataOnlyState()
    async for _ in async_composable_loop(
        llm=AsyncMockLLMBackend(responses=['{"tool":"done","args":{"summary":"ok"}}']),
        tools=_tools(),
        state=state,
        config=LoopConfig(max_steps=1),
        task={"goal": "finish"},
    ):
        pass

    result = RunResult.from_state(state)

    assert result.completed
    assert result.phase is RunPhase.TERMINAL
    assert result.termination_reason == "done"
