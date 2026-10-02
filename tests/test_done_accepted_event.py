"""Tests for the DONE_ACCEPTED lifecycle event."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("is_async", [False, True])
async def test_secondary_terminal_uses_its_own_schema_before_dispatch(is_async: bool) -> None:
    from looplet.async_loop import async_composable_loop
    from looplet.testing import AsyncMockLLMBackend
    from looplet.validation import FieldSpec, OutputSchema

    dispatched = []

    def escalate(*, reason: str) -> dict[str, str]:
        dispatched.append(reason)
        return {"reason": reason}

    tools = _tools()
    tools.register(
        ToolSpec(
            name="escalate", description="Escalate", parameters={"reason": "str"}, execute=escalate
        )
    )
    config = LoopConfig(
        max_steps=2,
        use_native_tools=False,
        done_tools=["escalate"],
        output_schema=OutputSchema(fields={"answer": FieldSpec("answer", "str")}),
        done_tool_schemas={"escalate": OutputSchema(fields={"reason": FieldSpec("reason", "str")})},
    )
    responses = [
        '{"tool":"escalate","args":{},"reasoning":"not ready"}',
        '{"tool":"escalate","args":{"reason":"human review"},"reasoning":"ready"}',
    ]
    recorder = DoneAcceptedRecorder()
    if is_async:
        steps = [
            step
            async for step in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=responses),
                tools=tools,
                config=config,
                hooks=[recorder],
            )
        ]
    else:
        steps = list(
            composable_loop(
                llm=MockLLMBackend(responses=responses),
                tools=tools,
                config=config,
                hooks=[recorder],
            )
        )
    assert len(steps) == 2
    assert "schema" in steps[0].tool_result.error.lower()
    assert dispatched == ["human review"]
    assert len(recorder.done_accepted_payloads) == 1
    assert recorder.done_accepted_payloads[0].tool_call.tool == "escalate"


@pytest.mark.parametrize("accept_retry", [True, False])
@pytest.mark.parametrize(
    "rejection",
    [
        {"error": "not accepted yet", "status": "rejected"},
        {"reason": "not accepted yet", "rejected": True},
    ],
)
def test_dispatched_done_rejection_retries_within_budget(
    accept_retry: bool,
    rejection: dict[str, str | bool],
) -> None:
    attempts = 0

    def submit(*, answer: str) -> dict[str, str | bool]:
        nonlocal attempts
        attempts += 1
        if attempts == 1 or not accept_retry:
            return rejection
        return {"answer": answer}

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="done",
            description="Finish.",
            parameters={"answer": "str"},
            execute=submit,
        )
    )
    recorder = DoneAcceptedRecorder()
    state = DefaultState(max_steps=2)
    steps = list(
        composable_loop(
            llm=MockLLMBackend(
                responses=['{"tool":"done","args":{"answer":"hi"},"reasoning":"finished"}'] * 2
            ),
            tools=tools,
            state=state,
            hooks=[recorder],
            config=LoopConfig(max_steps=2),
        )
    )

    assert len(steps) == attempts == 2
    assert steps[0].tool_result.data["rejected"] is True
    assert steps[0].tool_result.data["reason"] == "not accepted yet"
    assert steps[0].tool_result.error == "not accepted yet"
    if accept_retry:
        assert [event.step_num for event in recorder.done_accepted_payloads] == [2]
        assert recorder.stop_reasons == ["done"]
        assert state.termination_reason == "done"
    else:
        assert steps[1].tool_result.data["rejected"] is True
        assert recorder.done_accepted_payloads == []
        assert recorder.stop_reasons == ["budget_exhausted"]
        assert state.termination_reason == "budget_exhausted"


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
