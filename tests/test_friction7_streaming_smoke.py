"""Round-7 friction fixes: streaming events fire end-to-end."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.streaming import (
    CallbackEmitter,
    Event,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LoopEndEvent,
    LoopStartEvent,
    StepEndEvent,
    StepStartEvent,
    ToolDispatchEvent,
    ToolResultEvent,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="add",
            description="Add",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="Finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


def _run_collect() -> list[Event]:
    events: list[Event] = []
    emitter = CallbackEmitter(events.append)
    responses = [
        '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
    ]
    list(
        composable_loop(
            llm=MockLLMBackend(responses=responses),
            tools=_tools(),
            state=DefaultState(max_steps=5),
            config=LoopConfig(max_steps=5),
            stream=emitter,
        )
    )
    return events


class TestStreamingEmitsAllEvents:
    def test_loop_bracketed(self):
        events = _run_collect()
        assert isinstance(events[0], LoopStartEvent)
        assert isinstance(events[-1], LoopEndEvent)

    def test_llm_call_end_fires(self):
        events = _run_collect()
        starts = [e for e in events if isinstance(e, LLMCallStartEvent)]
        ends = [e for e in events if isinstance(e, LLMCallEndEvent)]
        assert len(starts) == len(ends) == 2

    def test_tool_result_fires_per_dispatch(self):
        events = _run_collect()
        dispatched = [e for e in events if isinstance(e, ToolDispatchEvent)]
        resulted = [e for e in events if isinstance(e, ToolResultEvent)]
        # Both add and done should emit dispatch + result.
        assert len(dispatched) == 2
        assert len(resulted) == 2
        assert {e.tool_name for e in resulted} == {"add", "done"}

    def test_step_end_fires_per_step(self):
        events = _run_collect()
        step_starts = [e for e in events if isinstance(e, StepStartEvent)]
        step_ends = [e for e in events if isinstance(e, StepEndEvent)]
        # One start per LLM turn, one end per dispatched tool (add + done)
        assert len(step_starts) == 2
        assert len(step_ends) == 2


class TestEvalHookCapturesTask:
    def test_eval_hook_reads_task_from_state(self):
        from looplet.evals import EvalContext, EvalHook

        captured: dict = {}

        def eval_task_is_visible(ctx: EvalContext) -> bool:
            captured["task"] = ctx.task
            return bool(ctx.task)

        hook = EvalHook(evaluators=[eval_task_is_visible])
        responses = [
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ]
        list(
            composable_loop(
                llm=MockLLMBackend(responses=responses),
                tools=_tools(),
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={"description": "add 1+2", "id": "t1"},
                hooks=[hook],
            )
        )
        assert captured["task"].get("description") == "add 1+2"
        assert hook.results[0].score == 1.0
