"""Contract tests for host run context and policy audit records."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    PolicyDecision,
    RunEnvelope,
    ToolSpec,
    composable_loop,
)
from looplet.checkpoint import FileCheckpointStore
from looplet.events import EventPayload, LifecycleEvent
from looplet.hook_decision import HookDecision
from looplet.permissions import PermissionDecision, PermissionEngine, PermissionHook
from looplet.types import ToolCall, ToolContext


def _tools(execute) -> BaseToolRegistry:
    registry = BaseToolRegistry()
    registry.register(ToolSpec("inspect", "Inspect context", {}, execute))
    registry.register(
        ToolSpec(
            "done",
            "Finish",
            {"summary": "summary"},
            lambda *, summary: {"summary": summary},
        )
    )
    return registry


def test_run_envelope_round_trips_as_json_safe_data() -> None:
    envelope = RunEnvelope(
        run_id="run-1",
        request_id="req-2",
        tenant_id="tenant-a",
        actor_id="user-7",
        deadline_at=123.5,
        policy_version="policy-3",
    )

    restored = RunEnvelope.from_dict(envelope.to_dict())

    assert restored == envelope
    assert restored.to_dict()["run_id"] == "run-1"
    assert "deployment" not in restored.to_dict()


def test_run_envelope_reaches_tools_events_and_checkpoint() -> None:
    envelope = RunEnvelope(run_id="run-42", tenant_id="tenant-a", trace_id="trace-9")
    seen_contexts: list[RunEnvelope | None] = []
    seen_events: list[dict] = []

    def inspect(*, ctx: ToolContext) -> dict:
        seen_contexts.append(ctx.run_envelope)
        return {"ok": True}

    class Recorder:
        def on_event(self, payload: EventPayload):
            if payload.run_envelope is not None:
                seen_events.append(payload.run_envelope)

    with tempfile.TemporaryDirectory() as tmp:
        state = DefaultState(max_steps=3)
        list(
            composable_loop(
                MockLLMBackend(
                    responses=[
                        '{"tool":"inspect","args":{}}',
                        '{"tool":"done","args":{"summary":"ok"}}',
                    ]
                ),
                tools=_tools(inspect),
                state=state,
                config=LoopConfig(
                    max_steps=3,
                    run_envelope=envelope,
                    checkpoint_dir=tmp,
                ),
                hooks=[Recorder()],
                task={},
            )
        )
        checkpoint = FileCheckpointStore(tmp).load("step_2_done")

    assert seen_contexts == [envelope]
    assert seen_events
    assert all(event == envelope.to_dict() for event in seen_events)
    assert checkpoint is not None
    assert checkpoint.run_envelope == envelope.to_dict()


def test_policy_decision_is_wire_safe_and_does_not_change_permission_field() -> None:
    audit = PolicyDecision(
        decision="deny",
        tool="delete_file",
        policy_id="filesystem-policy",
        policy_version="v2",
        reason="outside workspace",
    )
    decision = HookDecision(permission="deny", block="outside workspace", policy_decision=audit)

    restored = HookDecision.from_wire(decision.to_wire())

    assert restored is not None
    assert restored.permission == "deny"
    assert restored.block == "outside workspace"
    assert restored.policy_decision == audit


def test_permission_hook_emits_policy_audit_record() -> None:
    engine = PermissionEngine(default=PermissionDecision.DENY)
    hook = PermissionHook(engine)
    payload = EventPayload(
        event=LifecycleEvent.PRE_TOOL_USE,
        tool_call=ToolCall(tool="shell", args={}, reasoning="", call_id="c1"),
    )

    decision = hook.on_event(payload)

    assert decision.permission == "deny"
    assert decision.policy_decision is not None
    assert decision.policy_decision.decision == "deny"
    assert decision.policy_decision.tool == "shell"


def test_expired_deadline_cancels_sync_run_before_llm_call() -> None:
    envelope = RunEnvelope(run_id="expired", deadline_at=time.time() - 1)
    calls = []

    class Backend:
        def generate(self, prompt, **kwargs):
            calls.append(True)
            return '{"tool":"done","args":{"summary":"ok"}}'

    state = DefaultState(max_steps=2)
    steps = list(
        composable_loop(
            Backend(),
            tools=_tools(lambda **kwargs: {}),
            state=state,
            config=LoopConfig(max_steps=2, run_envelope=envelope),
            task={},
        )
    )

    assert steps == []
    assert calls == []
    assert state.run_status == "cancelled"
    assert state.termination_reason == "deadline_exceeded"
