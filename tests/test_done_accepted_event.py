"""Tests for the DONE_ACCEPTED lifecycle event."""

from __future__ import annotations

import warnings

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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", ["missing", "default", "custom", "secondary"])
async def test_terminal_setup_warning_precedes_model_calls(is_async, native, mode) -> None:
    import json

    from looplet import RunResult, async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    tools = _tools()
    config = LoopConfig(max_steps=1, use_native_tools=native)
    if mode == "missing":
        tools._tools.pop("done")
    elif mode == "custom":
        terminal = tools._tools.pop("done")
        terminal.name = "finish"
        tools.register(terminal)
        config.done_tool = "finish"
    elif mode == "secondary":
        config.done_tool = "finish"
        config.done_tools = ["done"]
    terminal_name = "finish" if mode == "custom" else "done"
    backend_type = AsyncMockLLMBackend if is_async else MockLLMBackend
    backend = backend_type([json.dumps({"tool": terminal_name, "args": {"answer": "finished"}})])
    method = "generate_with_tools" if native else "generate"
    original = getattr(backend, method)
    warnings_before_call = []

    def warned() -> bool:
        return any("No configured terminal tool" in str(item.message) for item in captured)

    if is_async:

        async def observed(*args, **kwargs):
            warnings_before_call.append(warned())
            return await original(*args, **kwargs)

    else:

        def observed(*args, **kwargs):
            warnings_before_call.append(warned())
            return original(*args, **kwargs)

    setattr(backend, method, observed)
    state = DefaultState(max_steps=1)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if is_async:
            steps = [
                step
                async for step in async_composable_loop(
                    llm=backend, tools=tools, state=state, config=config
                )
            ]
        else:
            steps = list(composable_loop(llm=backend, tools=tools, state=state, config=config))
    missing = mode == "missing"
    assert warnings_before_call == [missing]
    messages = [
        str(item.message) for item in captured if "No configured terminal tool" in str(item.message)
    ]
    assert len(messages) == int(missing)
    if missing:
        assert "include_done=True" in messages[0]
        assert steps[0].tool_result.error
    else:
        assert steps[0].tool_result.error is None
    assert RunResult.from_state(state).completed is not missing


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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("invalid_args", [{}, {"summary": "wrong field"}])
@pytest.mark.parametrize("accept_retry", [False, True])
async def test_failed_terminal_dispatch_is_not_accepted(
    is_async, native, invalid_args, accept_retry
) -> None:
    import json

    from looplet import RunResult, async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    responses = [json.dumps({"tool": "done", "args": invalid_args})]
    if accept_retry:
        responses.append(json.dumps({"tool": "done", "args": {"answer": "accepted"}}))
    state = DefaultState(max_steps=len(responses))
    config = LoopConfig(max_steps=len(responses), use_native_tools=native)
    recorder = DoneAcceptedRecorder()
    if is_async:
        steps = [
            step
            async for step in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=responses),
                tools=_tools(),
                state=state,
                config=config,
                hooks=[recorder],
            )
        ]
    else:
        steps = list(
            composable_loop(
                llm=MockLLMBackend(responses=responses),
                tools=_tools(),
                state=state,
                config=config,
                hooks=[recorder],
            )
        )

    assert steps[0].tool_result.error
    assert steps[0].tool_result.data["rejected"] is True
    assert steps[0].tool_result.data["reason"] == steps[0].tool_result.error
    result = RunResult.from_state(state)
    assert result.completed is accept_retry
    assert len(recorder.done_accepted_payloads) == int(accept_retry)
    if accept_retry:
        assert result.output == {"answer": "accepted"}
        assert recorder.done_accepted_payloads[0].step_num == 2
    else:
        assert result.output is None
        assert result.termination_reason == "budget_exhausted"


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", ["paid", "free", "failed-free"])
@pytest.mark.parametrize("accept_completion", [False, True])
async def test_free_tool_preserves_completion_budget(
    is_async, native, mode, accept_completion, monkeypatch
) -> None:
    import json
    from unittest.mock import AsyncMock, Mock

    from looplet import RunResult, async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    def note() -> dict[str, str]:
        if mode == "failed-free":
            raise ValueError("Could not record the note")
        return {"note": "recorded"}

    tools = _tools()
    tools.register(
        ToolSpec(
            name="note",
            description="Record a note.",
            parameters={},
            execute=note,
            free=mode != "paid",
        )
    )
    responses = [json.dumps({"tool": "note", "args": {}})] * 3
    responses.append(
        json.dumps({"tool": "done", "args": {"answer": "accepted"} if accept_completion else {}})
    )
    backend = (AsyncMockLLMBackend if is_async else MockLLMBackend)(responses=responses)
    method = "generate_with_tools" if native else "generate"
    generate = (AsyncMock if is_async else Mock)(wraps=getattr(backend, method))
    monkeypatch.setattr(backend, method, generate)
    state = DefaultState(max_steps=1)
    config = LoopConfig(max_steps=1, use_native_tools=native)
    remaining = []
    if is_async:
        async for _step in async_composable_loop(
            llm=backend, tools=tools, state=state, config=config
        ):
            remaining.append(state.budget_remaining)
    else:
        for _step in composable_loop(llm=backend, tools=tools, state=state, config=config):
            remaining.append(state.budget_remaining)

    is_free = mode == "free"
    assert remaining == ([1, 1, 1, 0] if is_free else [0])
    assert state.step_count == (4 if is_free else 1)
    assert state.queries_used == int(mode == "paid")
    assert state.steps[0].metadata.get("budget_free", False) is is_free
    prompts = [
        call.args[0] if call.args else call.kwargs["prompt"] for call in generate.call_args_list
    ]
    assert len(prompts) == state.step_count
    for step_number, prompt in enumerate(prompts, start=1):
        assert f"STEP {step_number} - charged-step budget: 1/1 remaining" in prompt
    result = RunResult.from_state(state)
    assert result.completed is (is_free and accept_completion)
    if result.completed:
        assert result.output == {"answer": "accepted"}
    else:
        assert result.termination_reason == "budget_exhausted"
        assert result.output is None


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("deadline_enabled", [False, True])
async def test_deadline_bounds_repeated_free_calls(
    is_async, native, deadline_enabled, monkeypatch
) -> None:
    import json

    from looplet import RunEnvelope, RunResult, async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    clock = [100.0]
    monkeypatch.setattr("looplet.loop.time.time", lambda: clock[0])

    def note() -> dict[str, str]:
        clock[0] += 1.0
        return {"note": "recorded"}

    tools = _tools()
    tools.register(ToolSpec("note", "Record a note.", {}, note, free=True))
    responses = [json.dumps({"tool": "note", "args": {}})] * 3
    responses.append(json.dumps({"tool": "done", "args": {"answer": "accepted"}}))
    config = LoopConfig(
        max_steps=1,
        use_native_tools=native,
        run_envelope=(
            RunEnvelope(run_id="bounded-free", deadline_at=103.0) if deadline_enabled else None
        ),
    )
    state = DefaultState(max_steps=config.max_steps)
    if is_async:
        steps = [
            step
            async for step in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=responses),
                tools=tools,
                state=state,
                config=config,
            )
        ]
    else:
        steps = list(
            composable_loop(
                llm=MockLLMBackend(responses=responses), tools=tools, state=state, config=config
            )
        )

    assert all(step.metadata["budget_free"] for step in steps[:3])
    assert len(steps) == (3 if deadline_enabled else 4)
    assert state.budget_remaining == int(deadline_enabled)
    result = RunResult.from_state(state)
    assert result.completed is not deadline_enabled
    assert result.termination_reason == ("deadline_exceeded" if deadline_enabled else "done")


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
