from __future__ import annotations

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


class _HookDecisionRecorder:
    def __init__(self) -> None:
        self.payloads: list[EventPayload] = []

    def on_event(self, payload: EventPayload) -> None:
        if payload.event == LifecycleEvent.HOOK_DECISION:
            self.payloads.append(payload)


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
