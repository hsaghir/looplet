"""Tests for the DONE_ACCEPTED lifecycle event."""

from __future__ import annotations

from looplet import (
    BaseToolRegistry,
    DefaultState,
    HookDecision,
    LifecycleEvent,
    LoopConfig,
    composable_loop,
)
from looplet.events import EventPayload
from looplet.hook_decision import Block
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec


class DoneAcceptedRecorder:
    def __init__(self, *, decision: HookDecision | None = None) -> None:
        self.done_accepted_payloads: list[EventPayload] = []
        self.stop_reasons: list[str | None] = []
        self._decision = decision

    def on_event(self, payload: EventPayload) -> HookDecision | None:
        if payload.event == LifecycleEvent.DONE_ACCEPTED:
            self.done_accepted_payloads.append(payload)
            return self._decision
        if payload.event == LifecycleEvent.STOP:
            self.stop_reasons.append(payload.termination_reason)
        return None


def _tools() -> BaseToolRegistry:
    registry = BaseToolRegistry()
    registry.register(
        ToolSpec(
            name="add",
            description="Add two numbers.",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    registry.register(
        ToolSpec(
            name="done",
            description="Finish.",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return registry


def _run_loop(
    responses: list[str],
    hooks: list[object],
    *,
    max_steps: int = 5,
):
    return list(
        composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=_tools(),
            state=DefaultState(max_steps=max_steps),
            hooks=hooks,
            config=LoopConfig(max_steps=max_steps),
        )
    )


def test_done_accepted_fires_once_with_done_payload() -> None:
    recorder = DoneAcceptedRecorder()

    steps = _run_loop(
        ['{"tool":"done","args":{"answer":"hi"},"reasoning":"finished"}'],
        [recorder],
    )

    assert len(recorder.done_accepted_payloads) == 1
    payload = recorder.done_accepted_payloads[0]
    assert payload.step_num == steps[-1].number
    assert payload.tool_call.tool == "done"
    assert payload.tool_result is not None
    assert payload.tool_result.tool == "done"


def test_done_accepted_does_not_fire_when_check_done_rejects_done() -> None:
    class RejectDone:
        def check_done(self, state, session_log, context, step_num):
            return Block("not yet")

    recorder = DoneAcceptedRecorder()

    _run_loop(
        ['{"tool":"done","args":{"answer":"hi"},"reasoning":"finished"}'],
        [RejectDone(), recorder],
        max_steps=1,
    )

    assert recorder.done_accepted_payloads == []


def test_done_accepted_does_not_fire_on_max_steps_without_done() -> None:
    recorder = DoneAcceptedRecorder()

    _run_loop(
        ['{"tool":"add","args":{"a":1,"b":2},"reasoning":"add"}'],
        [recorder],
        max_steps=1,
    )

    assert recorder.done_accepted_payloads == []


def test_done_accepted_decisions_are_observer_only() -> None:
    recorder = DoneAcceptedRecorder(decision=HookDecision(block="ignored", stop="ignored"))

    _run_loop(
        ['{"tool":"done","args":{"answer":"hi"},"reasoning":"finished"}'],
        [recorder],
    )

    assert len(recorder.done_accepted_payloads) == 1
    assert recorder.stop_reasons == ["done"]
