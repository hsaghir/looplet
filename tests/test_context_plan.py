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
@pytest.mark.parametrize("native", [False, True])
async def test_recovery_uses_the_same_prompt_preparation(async_mode, native):
    from looplet import ToolSpec, async_composable_loop

    class Backend:
        def __init__(self):
            self.prompts = []
            self.schemas = []

        def reply(self, prompt, schemas=None):
            self.prompts.append(prompt)
            self.schemas.append(schemas)
            if len(self.prompts) == 1:
                raise RuntimeError("prompt is too long: 200000 tokens > 180000 limit")
            if schemas is not None:
                return [
                    {"type": "tool_use", "id": "done-1", "name": "done", "input": {"summary": "ok"}}
                ]
            return '{"tool":"done","args":{"summary":"ok"}}'

        def generate(self, prompt, **kwargs):
            return self.reply(prompt)

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return self.reply(prompt)

    backend = AsyncBackend() if async_mode else Backend()
    if native:

        def generate_with_tools(prompt, *, tools, **kwargs):
            return backend.reply(prompt, tools)

        async def async_generate_with_tools(prompt, *, tools, **kwargs):
            return backend.reply(prompt, tools)

        backend.generate_with_tools = (
            async_generate_with_tools if async_mode else generate_with_tools
        )

    loaded = []
    sources = [
        ScopedContextSource(
            "allowed",
            CallableMemorySource(lambda state: loaded.append("allowed") or "allowed guidance"),
            retention="run",
        ),
        ScopedContextSource(
            "private",
            CallableMemorySource(lambda state: loaded.append("private") or "private guidance"),
            scope=lambda **kwargs: False,
        ),
    ]
    projections = []

    class Builder:
        def build_prompt(self, **kwargs):
            return "HOOK\n" + kwargs["memory"] + "\n" + kwargs["tool_catalog"]

    def render(*, projection):
        projections.append(projection)
        return "RENDER\n" + projection.default_prompt

    tools = BaseToolRegistry()
    register_done_tool(tools)
    tools.register(ToolSpec("hidden", "Hidden tool", {}, lambda: None))
    state = DefaultState(max_steps=2)
    config = LoopConfig(
        max_steps=2,
        use_native_tools=native,
        reactive_recovery=True,
        scoped_context_sources=sources,
        scoped_context_budget_tokens=50,
        context_planner=lambda **kwargs: ContextPlan(
            ("task",), estimated_tokens=100, budget_tokens=1
        ),
        render_messages_override=render,
        tool_view_selector=lambda **kwargs: ("done",),
    )
    kwargs = dict(llm=backend, tools=tools, config=config, state=state, hooks=[Builder()])
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )

    assert [step.tool_call.tool for step in steps] == ["done"]
    assert len(backend.prompts) == len(projections) == 2
    assert loaded == ["allowed"]
    assert all(
        "RENDER\nHOOK" in prompt and "allowed guidance" in prompt and "hidden(" not in prompt
        for prompt in backend.prompts
    )
    assert all(projection.scoped_context_plan.sources == ("allowed",) for projection in projections)
    assert all(not projection.context_plan.within_budget for projection in projections)
    if native:
        assert all(
            [schema["name"] for schema in schemas] == ["done"] for schemas in backend.schemas
        )


@pytest.mark.parametrize("async_mode", [False, True])
async def test_dynamic_memory_is_rendered_for_each_turn(async_mode):
    from looplet import ToolSpec, async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    rendered = []
    memory = CallableMemorySource(
        lambda state: rendered.append(state.step_count) or f"turn-{state.step_count}"
    )
    tools = BaseToolRegistry()
    register_done_tool(tools)
    tools.register(ToolSpec("advance", "Advance", {}, lambda: None))
    backend = (AsyncMockLLMBackend if async_mode else MockLLMBackend)(
        [
            '{"tool":"advance","args":{}}',
            '{"tool":"done","args":{"summary":"ok"}}',
        ]
    )
    kwargs = dict(
        llm=backend,
        tools=tools,
        config=LoopConfig(max_steps=2, use_native_tools=False, memory_sources=[memory]),
    )
    if async_mode:
        [step async for step in async_composable_loop(**kwargs)]
    else:
        list(composable_loop(**kwargs))
    assert rendered == [0, 1]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_permanently_oversized_renderer_never_reaches_provider(async_mode):
    from looplet import async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    backend = (AsyncMockLLMBackend if async_mode else MockLLMBackend)(
        ['{"tool":"done","args":{"summary":"ok"}}']
    )
    tools = BaseToolRegistry()
    register_done_tool(tools)
    state = DefaultState(max_steps=2)
    config = LoopConfig(
        max_steps=2,
        context_window=6000,
        use_native_tools=False,
        render_messages_override=lambda **kwargs: "x" * 24_000,
    )
    kwargs = dict(llm=backend, tools=tools, config=config, state=state)
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )

    assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
    assert backend.last_prompt == ""
    assert state._stop_reason == "llm_error"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_async_callbacks_builder_fallback_and_legacy_renderer(async_mode):
    from looplet import async_composable_loop
    from looplet.testing import AsyncMockLLMBackend

    class BrokenBuilder:
        def build_prompt(self, **kwargs):
            raise ValueError("broken builder")

    class EmptyBuilder:
        def build_prompt(self, **kwargs):
            return None

    def builder(**kwargs):
        return "CONFIG " + str(kwargs["step_number"])

    async def async_builder(**kwargs):
        return builder(**kwargs)

    def planner(**kwargs):
        return ContextPlan(("task",), estimated_tokens=100, budget_tokens=1)

    async def async_planner(**kwargs):
        return planner(**kwargs)

    rendered = []

    def legacy_renderer(*, messages, default_prompt, step_num):
        rendered.append((messages, step_num))
        return default_prompt + " RENDERED"

    tools = BaseToolRegistry()
    register_done_tool(tools)
    backend = (AsyncMockLLMBackend if async_mode else MockLLMBackend)(
        ['{"tool":"done","args":{"summary":"ok"}}']
    )
    config = LoopConfig(
        max_steps=1,
        use_native_tools=False,
        build_prompt=async_builder if async_mode else builder,
        context_planner=async_planner if async_mode else planner,
        render_messages_override=legacy_renderer,
    )
    kwargs = dict(llm=backend, tools=tools, config=config, hooks=[BrokenBuilder(), EmptyBuilder()])
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert steps[0].tool_call.tool == "done"
    assert backend.last_prompt == "CONFIG 1 RENDERED"
    assert rendered == [([], 1)]


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
