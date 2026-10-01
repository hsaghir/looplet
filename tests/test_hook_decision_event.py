from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    Block,
    Continue,
    DefaultState,
    InjectContext,
    LifecycleEvent,
    LoopConfig,
    Stop,
    composable_loop,
)
from looplet.events import EventPayload
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["normal", "parse", "rejected"])
@pytest.mark.parametrize("slot", ["pre_event", "post_event", "should_stop"])
async def test_post_step_stop_applies_to_every_turn_kind(async_mode, kind, slot):
    from looplet.async_loop import async_composable_loop

    class Backend:
        calls = 0

        def generate_with_tools(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                if kind == "parse":
                    return []
                if kind == "normal":
                    return [
                        {
                            "type": "tool_use",
                            "id": "first",
                            "name": "add",
                            "input": {"a": 1, "b": 2},
                        }
                    ]
            return [
                {"type": "tool_use", "id": "done", "name": "done", "input": {"answer": "finish"}}
            ]

        def generate(self, prompt, **kwargs):
            raise AssertionError("native protocol expected")

    observed_steps = []

    class Hook:
        def on_event(self, payload):
            event = (
                LifecycleEvent.PRE_LLM_CALL
                if slot == "pre_event"
                else LifecycleEvent.POST_LLM_RESPONSE
            )
            if slot != "should_stop" and payload.event is event:
                return Stop("requested_stop")

        def should_stop(self, state, step_num, new_entities):
            if slot == "should_stop":
                return Stop("requested_stop")

        def check_done(self, *args, **kwargs):
            if kind == "rejected":
                return Block("not accepted")

        def post_step(self, state, session_log, step_num):
            observed_steps.append((step_num, state.steps[-1].tool_call.tool))

    backend = Backend()
    state = DefaultState(max_steps=3)
    kwargs = {
        "llm": backend,
        "tools": _tools(),
        "state": state,
        "hooks": [Hook()],
        "config": LoopConfig(max_steps=3),
    }
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert backend.calls == 1
    assert len(steps) == 1
    assert observed_steps == [(1, steps[0].tool_call.tool)]
    assert state._stop_reason == "requested_stop"
    assert steps[0].tool_call.tool == (
        "__parse_error__" if kind == "parse" else "add" if kind == "normal" else "done"
    )
    if kind == "rejected":
        assert steps[0].tool_result.error is not None


@pytest.mark.parametrize("async_mode", [False, True])
async def test_pending_stop_does_not_launch_text_parse_recovery(async_mode):
    from looplet.async_loop import async_composable_loop

    class Backend:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return (
                "not JSON"
                if self.calls == 1
                else '{"tool":"done","args":{"answer":"finish"},"reasoning":""}'
            )

    class Hook:
        def on_event(self, payload):
            if payload.event is LifecycleEvent.PRE_LLM_CALL:
                return Stop("requested_stop")

    backend = Backend()
    state = DefaultState(max_steps=3)
    kwargs = {
        "llm": backend,
        "tools": _tools(),
        "state": state,
        "hooks": [Hook()],
        "config": LoopConfig(max_steps=3, use_native_tools=False),
    }
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert backend.calls == 1
    assert len(steps) == 1 and steps[0].tool_call.tool == "__parse_error__"
    assert state._stop_reason == "requested_stop"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_recovery_abort_still_runs_post_step_stop_checks(async_mode):
    from types import SimpleNamespace

    from looplet.async_loop import async_composable_loop

    class Backend:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return "not JSON"

    class Hook:
        def should_stop(self, state, step_num, new_entities):
            return Stop("recovery_observed")

    registry = SimpleNamespace(
        attempt_recovery=lambda *args, **kwargs: SimpleNamespace(
            action_type="abort", message="no repair"
        )
    )
    state = DefaultState(max_steps=3)
    backend = Backend()
    kwargs = {
        "llm": backend,
        "tools": _tools(),
        "state": state,
        "hooks": [Hook()],
        "config": LoopConfig(max_steps=3, use_native_tools=False, recovery_registry=registry),
    }
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert backend.calls == 1 and len(steps) == 1
    assert "recovery aborted" in steps[0].tool_result.error
    assert state._stop_reason == "recovery_observed"


async def test_async_should_stop_is_awaited_after_synthetic_step():
    from looplet.async_loop import async_composable_loop

    class Backend:
        def generate(self, prompt, **kwargs):
            return "not JSON"

        def generate_with_tools(self, prompt, **kwargs):
            return []

    class Hook:
        async def should_stop(self, state, step_num, new_entities):
            assert state.steps[-1].tool_call.tool == "__parse_error__"
            return Stop("async_stop")

    state = DefaultState(max_steps=3)
    steps = [
        step
        async for step in async_composable_loop(
            llm=Backend(),
            tools=_tools(),
            state=state,
            config=LoopConfig(max_steps=3),
            hooks=[Hook()],
        )
    ]
    assert len(steps) == 1 and state._stop_reason == "async_stop"


class _HookDecisionRecorder:
    def __init__(self) -> None:
        self.payloads: list[EventPayload] = []

    def on_event(self, payload: EventPayload) -> None:
        if payload.event == LifecycleEvent.HOOK_DECISION:
            self.payloads.append(payload)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_fatal_model_step_is_observed_but_not_accepted(async_mode, monkeypatch):
    from looplet.async_loop import async_composable_loop
    from looplet.session import SessionLog

    monkeypatch.setattr("looplet.scaffolding.time.sleep", lambda delay: None)
    observed = []

    class Backend:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("provider failed")

        def generate_with_tools(self, prompt, **kwargs):
            raise RuntimeError("provider failed")

    class Hook:
        def post_step(self, state, session_log, step_num):
            observed.append(state.steps[-1].tool_call.tool)

    state = DefaultState(max_steps=3)
    session = SessionLog()
    kwargs = {
        "llm": Backend(),
        "tools": _tools(),
        "state": state,
        "session_log": session,
        "hooks": [Hook()],
        "config": LoopConfig(max_steps=3),
    }
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert len(steps) == 1 and observed == ["__llm_error__"]
    assert state._stop_reason == "llm_error"
    assert "__llm_error__" in str(session.to_list())


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="add",
            description="add",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="done",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


def _run(responses: list[str], hooks: list[object]) -> None:
    list(
        composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=_tools(),
            state=DefaultState(max_steps=5),
            hooks=hooks,
            config=LoopConfig(max_steps=5),
        )
    )


def test_hook_decision_events_include_slot_hook_name_and_decision_dict() -> None:
    class DecisionHook:
        def on_event(self, payload: EventPayload):
            if payload.event == LifecycleEvent.PRE_LLM_CALL:
                return InjectContext("test")
            return None

        def pre_prompt(self, state, session_log, context, step_num):
            return InjectContext("test")

        def pre_dispatch(self, state, session_log, tool_call, step_num):
            if tool_call.tool == "add":
                return InjectContext("test")
            return None

        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            if tool_call.tool == "add":
                return InjectContext("test")
            return None

        def check_done(self, state, session_log, context, step_num):
            if not any(step.tool_call.tool == "done" for step in state.steps):
                return Block("test")
            return None

        def should_stop(self, state, step_num, new_entities):
            if state.steps and state.steps[-1].tool_call.tool == "add":
                return Stop("test")
            return None

    recorder = _HookDecisionRecorder()

    _run(
        [
            '{"tool":"done","args":{"answer":"early"},"reasoning":""}',
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ],
        [recorder, DecisionHook()],
    )

    by_slot = {}
    for payload in recorder.payloads:
        by_slot.setdefault(payload.hook_slot, []).append(payload)

    assert LifecycleEvent.HOOK_DECISION.value == "hook_decision"
    assert set(by_slot) >= {
        "on_event",
        "pre_prompt",
        "pre_dispatch",
        "post_dispatch",
        "check_done",
        "should_stop",
    }
    assert all(payload.hook_name == "DecisionHook" for payload in recorder.payloads)

    assert by_slot["check_done"][0].extra["decision"]["block"] == "test"
    assert by_slot["should_stop"][0].extra["decision"]["stop"] == "test"
    assert by_slot["pre_prompt"][0].extra["decision"]["additional_context"] == "test"
    assert by_slot["pre_dispatch"][0].extra["decision"]["additional_context"] == "test"
    assert by_slot["post_dispatch"][0].extra["decision"]["additional_context"] == "test"

    on_event_payload = by_slot["on_event"][0]
    assert on_event_payload.extra["originating_event"] == "pre_llm_call"
    assert on_event_payload.extra["decision"]["additional_context"] == "test"


def test_no_hook_decision_events_for_none_or_noop_decisions() -> None:
    class NoopHook:
        def on_event(self, payload: EventPayload):
            if payload.event == LifecycleEvent.PRE_LLM_CALL:
                return Continue()
            return None

        def pre_prompt(self, state, session_log, context, step_num):
            return Continue()

        def pre_dispatch(self, state, session_log, tool_call, step_num):
            return Continue()

        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            return Continue()

        def check_done(self, state, session_log, context, step_num):
            return Continue()

        def should_stop(self, state, step_num, new_entities):
            return Continue()

    recorder = _HookDecisionRecorder()

    _run(
        [
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ],
        [recorder, NoopHook()],
    )

    assert recorder.payloads == []
