from __future__ import annotations

import pytest

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
from looplet.context_plan import ContextSourceSelector, ScopedContextSource
from looplet.memory import CallableMemorySource, StaticMemorySource


def test_context_plan_reports_budget() -> None:
    plan = ContextPlan(("task", "memory"), estimated_tokens=12, budget_tokens=10)
    assert not plan.within_budget
    assert plan.to_dict()["dropped_sections"] == []


def test_projection_serializes_context_plan() -> None:
    plan = ContextPlan(("task",), estimated_tokens=2)
    projection = ContextProjection(messages=(), default_prompt="p", step_num=1, context_plan=plan)
    assert projection.to_dict()["context_plan"]["estimated_tokens"] == 2


def test_scoped_sources_skip_out_of_scope_and_report_budget() -> None:
    loaded = []
    sources = [
        ScopedContextSource(
            "case",
            CallableMemorySource(lambda state: loaded.append("case") or "case facts"),
            scope=lambda *, task, state, step_num: task["case"] == "a",
            origin="host/cases/a",
            trust="host",
            retention="run",
        ),
        ScopedContextSource(
            "other",
            CallableMemorySource(lambda state: loaded.append("other") or "private"),
            scope=lambda *, task, state, step_num: task["case"] == "b",
        ),
        ScopedContextSource(
            "extra",
            CallableMemorySource(lambda state: loaded.append("extra") or "X" * 500),
        ),
    ]
    selector = ContextSourceSelector(sources, budget_tokens=50)

    text, plan = selector.select(task={"case": "a"}, state=None, step_num=1)
    assert "case facts" in text
    assert "private" not in text
    assert plan.sources == ("case",)
    assert [(choice.source_id, choice.reason) for choice in plan.source_decisions] == [
        ("case", "included"),
        ("other", "out_of_scope"),
        ("extra", "budget"),
    ]
    assert loaded == ["case", "extra"]
    assert plan.within_budget
    assert plan.to_dict()["source_decisions"][0]["content_hash"]
    assert "case facts" not in str(plan.to_dict())

    selector.select(task={"case": "a"}, state=None, step_num=2)
    assert loaded == ["case", "extra", "extra"]


def test_scoped_sources_dedupe_retain_and_validate_ids() -> None:
    loads = []
    retained = ScopedContextSource(
        "case",
        CallableMemorySource(lambda state: loads.append("case") or "case facts"),
        scope=lambda *, task, state, step_num: task["case"] == "a",
        retention="run",
    )
    other = ScopedContextSource("other", StaticMemorySource("extra facts"))
    selector = ContextSourceSelector([retained, retained, other], budget_tokens=20)

    text, plan = selector.select(task={"case": "a"}, state=None, step_num=1)
    assert "case facts" in text
    assert "extra facts" not in text
    assert [(choice.source_id, choice.reason) for choice in plan.source_decisions] == [
        ("case", "included"),
        ("other", "budget"),
    ]
    text, plan = selector.select(task={"case": "b"}, state=None, step_num=2)
    assert "case facts" in text
    assert plan.sources == ("case",)
    assert loads == ["case"]

    with pytest.raises(ValueError, match="Conflicting context source ID"):
        ContextSourceSelector([retained, ScopedContextSource("case", other.source)])
    with pytest.raises(ValueError, match="non-negative"):
        ContextSourceSelector([retained], budget_tokens=-1)


def test_retained_scoped_context_does_not_cross_runs() -> None:
    tools = BaseToolRegistry()
    register_done_tool(tools)
    config = LoopConfig(
        max_steps=1,
        use_native_tools=False,
        scoped_context_sources=[
            ScopedContextSource(
                "tenant",
                CallableMemorySource(lambda state: state.metadata["tenant"]),
                retention="run",
                trust="host",
                origin="host/tenant",
            )
        ],
    )

    def run(tenant):
        backend = MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}'])
        list(
            composable_loop(
                backend,
                tools=tools,
                state=DefaultState(max_steps=1, metadata={"tenant": tenant}),
                config=config,
            )
        )
        return backend.last_prompt

    first = run("TENANT_A_PRIVATE")
    second = run("TENANT_B_PRIVATE")
    assert "TENANT_A_PRIVATE" in first
    assert "TENANT_B_PRIVATE" in second
    assert "TENANT_A_PRIVATE" not in second


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


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_scoped_context_reaches_prompt_and_projection_in_both_loops(async_mode):
    from looplet import Conversation, ToolSpec, async_composable_loop

    class RecordingBackend:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, **kwargs):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return '{"tool":"advance","args":{}}'
            return '{"tool":"done","args":{"summary":"ok"}}'

    class AsyncRecordingBackend(RecordingBackend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    loaded = []
    run_source = ScopedContextSource(
        source_id="case",
        origin="host/cases/a",
        trust="host",
        retention="run",
        scope=lambda *, task, state, step_num: task["path"] == "cases/a" and step_num == 1,
        source=CallableMemorySource(lambda state: loaded.append("case") or "case guidance"),
    )
    out_of_scope = ScopedContextSource(
        source_id="elsewhere",
        origin="host/cases/b",
        scope=lambda *, task, state, step_num: task["path"] == "cases/b",
        source=CallableMemorySource(lambda state: loaded.append("elsewhere") or "wrong guidance"),
    )
    projections = []

    def render(*, projection):
        projections.append(projection)
        return projection.default_prompt

    def select_tools(**kwargs):
        return ("advance", "done") if kwargs["step_num"] == 1 else ("done",)

    tools = BaseToolRegistry()
    register_done_tool(tools)
    tools.register(ToolSpec("advance", "Continue", {}, lambda: None))
    config = LoopConfig(
        max_steps=2,
        use_native_tools=False,
        tool_view_selector=select_tools,
        render_messages_override=render,
        scoped_context_sources=[run_source, out_of_scope],
        scoped_context_budget_tokens=50,
    )
    backend = AsyncRecordingBackend() if async_mode else RecordingBackend()
    conversation = Conversation()
    kwargs = dict(
        llm=backend,
        tools=tools,
        task={"path": "cases/a"},
        state=DefaultState(max_steps=2),
        config=config,
        conversation=conversation,
    )
    if async_mode:
        steps = [step async for step in async_composable_loop(**kwargs)]
    else:
        steps = list(composable_loop(**kwargs))

    assert [step.tool_call.tool for step in steps] == ["advance", "done"]
    assert loaded == ["case"]
    assert len(projections) == 2
    prompt_records = [m for m in conversation.serialize()["messages"] if m["role"] == "user"]
    for prompt, projection, record, names in zip(
        backend.prompts, projections, prompt_records, (("advance", "done"), ("done",)), strict=True
    ):
        assert "SCOPED CONTEXT" in prompt
        assert "case guidance" in prompt
        assert "wrong guidance" not in prompt
        assert projection.scoped_context_plan.sources == ("case",)
        assert (
            projection.to_dict()["scoped_context_plan"]["source_decisions"][1]["reason"]
            == "out_of_scope"
        )
        assert record["metadata"]["scoped_context_plan"] == projection.scoped_context_plan.to_dict()
        assert record["metadata"]["scoped_context_plan"]["source_decisions"][0]["content_hash"]
        assert record["metadata"]["tool_view"] == {
            "names": list(names),
            "version": tools.tool_view(names).version,
        }
