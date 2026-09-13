from __future__ import annotations

from looplet import (
    BaseToolRegistry,
    ContextPlan,
    ContextProjection,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    composable_loop,
    register_done_tool,
)


def test_context_plan_reports_budget() -> None:
    plan = ContextPlan(("task", "memory"), estimated_tokens=12, budget_tokens=10)
    assert not plan.within_budget
    assert plan.to_dict()["dropped_sections"] == []


def test_projection_serializes_context_plan() -> None:
    plan = ContextPlan(("task",), estimated_tokens=2)
    projection = ContextProjection(messages=(), default_prompt="p", step_num=1, context_plan=plan)
    assert projection.to_dict()["context_plan"]["estimated_tokens"] == 2


def test_loop_passes_plan_to_message_projection() -> None:
    seen = []

    def planner(**kwargs):
        return ContextPlan(("task",), estimated_tokens=3, budget_tokens=4)

    def render(*, projection):
        seen.append(projection.context_plan)
        return projection.default_prompt

    tools = BaseToolRegistry()
    register_done_tool(tools)
    list(
        composable_loop(
            MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
            tools=tools,
            state=DefaultState(max_steps=1),
            config=LoopConfig(
                max_steps=1,
                context_planner=planner,
                render_messages_override=render,
            ),
            task={},
        )
    )
    assert seen[0].estimated_tokens == 3
