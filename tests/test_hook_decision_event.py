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


@pytest.mark.parametrize("mode", ["sync", "async", "async_hooks"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("writer_kind", ["method", "event"])
@pytest.mark.parametrize("observer_kind", ["method", "event"])
@pytest.mark.parametrize("case", ["permission", "result"])
async def test_tool_hooks_compose_in_registration_order(
    mode, native, writer_kind, observer_kind, case
):
    import json
    from types import SimpleNamespace

    from looplet import Deny, HookDecision, ToolResult, run_result, run_result_async

    effects = []
    observed = []
    audit = []

    class Audit:
        def on_event(self, payload):
            if payload.event is LifecycleEvent.HOOK_DECISION:
                audit.append(payload)

    def read_resource(scope: str) -> dict:
        effects.append(scope)
        return {"scope": scope, "secret": "synthetic-private-value"}

    class Backend:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return json.dumps(
                {"tool": "read_resource", "args": {"scope": "public"}}
                if self.calls == 1
                else {"tool": "done", "args": {"summary": "finished"}}
            )

        def generate_with_tools(self, prompt, **kwargs):
            self.calls += 1
            return [
                {
                    "type": "tool_use",
                    "id": str(self.calls),
                    "name": "read_resource" if self.calls == 1 else "done",
                    "input": {"scope": "public"} if self.calls == 1 else {"summary": "finished"},
                }
            ]

    def rewrite(tool_call, tool_result=None):
        if case == "permission":
            return HookDecision(updated_args={"scope": "private"})
        return HookDecision(
            updated_result=ToolResult(
                tool=tool_result.tool,
                args_summary=tool_result.args_summary,
                data={"scope": tool_result.data["scope"]},
            )
        )

    def inspect_result(tool_call, tool_result=None):
        if case == "permission":
            observed.append(tool_call.args["scope"])
            if tool_call.args["scope"] != "public":
                return Deny("private scope is not authorized")
        else:
            observed.append(dict(tool_result.data))

    def make_hook(kind, effect):
        if kind == "event":
            slot = "on_event"

            def callback(payload):
                event = (
                    LifecycleEvent.PRE_TOOL_USE
                    if case == "permission"
                    else LifecycleEvent.POST_TOOL_USE
                )
                if payload.event is event and payload.tool_call.tool == "read_resource":
                    return effect(payload.tool_call, payload.tool_result)
        elif case == "permission":
            slot = "pre_dispatch"

            def callback(state, session_log, tool_call, step_num):
                if tool_call.tool == "read_resource":
                    return effect(tool_call)
        else:
            slot = "post_dispatch"

            def callback(state, session_log, tool_call, tool_result, step_num):
                if tool_call.tool == "read_resource":
                    return effect(tool_call, tool_result)

        if mode == "async_hooks":
            synchronous = callback

            async def callback(*args, **kwargs):
                return synchronous(*args, **kwargs)

        hook = SimpleNamespace(**{slot: callback})
        if kind == "method":

            def reject_duplicate(payload):
                event = (
                    LifecycleEvent.PRE_TOOL_USE
                    if case == "permission"
                    else LifecycleEvent.POST_TOOL_USE
                )
                if payload.event is event:
                    pytest.fail("A method-style hook must not receive its equivalent event")

            hook.on_event = reject_duplicate
        return hook

    backend = Backend()
    options = dict(
        tools=[read_resource],
        hooks=[Audit(), make_hook(writer_kind, rewrite), make_hook(observer_kind, inspect_result)],
        task="Read only an authorized public resource.",
        config=LoopConfig(harness_profile="minimal", max_steps=2, use_native_tools=native),
    )
    result = (
        run_result(backend, **options)
        if mode == "sync"
        else await run_result_async(backend, **options)
    )
    assert result.completed and backend.calls == 2
    assert len(audit) == (2 if case == "permission" else 1)
    if case == "permission":
        assert effects == []
        assert observed == ["private"]
        assert result.steps[0].tool_result.data is None
        assert "Permission denied" in result.steps[0].tool_result.error
    else:
        assert effects == ["public"]
        assert observed == [{"scope": "public"}]
        assert result.steps[0].tool_result.data == {"scope": "public"}


@pytest.mark.parametrize("async_mode", ["sync", "async", "async_hooks"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "case",
    ["rewrite", "block", "cache", "failure", "allow_rewrite", "allow_cache", "observer_failure"],
)
async def test_pre_tool_event_preserves_permission_contract(async_mode, native, case):
    import json
    from types import SimpleNamespace

    from looplet import Deny, HookDecision, ToolResult, run_result, run_result_async

    effects = []
    checked_scopes = []

    def read_resource(scope: str) -> dict:
        effects.append(scope)
        return {"scope": scope}

    class Backend:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            return json.dumps(
                {"tool": "read_resource", "args": {"scope": "public"}}
                if self.calls == 1
                else {"tool": "done", "args": {"summary": "finished"}}
            )

        def generate_with_tools(self, prompt, **kwargs):
            self.calls += 1
            return [
                {
                    "type": "tool_use",
                    "id": str(self.calls),
                    "name": "read_resource" if self.calls == 1 else "done",
                    "input": {"scope": "public"} if self.calls == 1 else {"summary": "finished"},
                }
            ]

    class Intercept:
        def on_event(self, payload):
            if case == "observer_failure" and payload.event is LifecycleEvent.POST_TOOL_USE:
                raise RuntimeError("observer unavailable")
            if (
                payload.event is not LifecycleEvent.PRE_TOOL_USE
                or payload.tool_call.tool != "read_resource"
            ):
                return None
            if case == "failure":
                raise RuntimeError("policy lookup unavailable")
            if case in {"rewrite", "block", "allow_rewrite"}:
                return HookDecision(updated_args={"scope": "private"})
            if case in {"cache", "allow_cache"}:
                return HookDecision(
                    updated_result=ToolResult(
                        tool="read_resource", args_summary="", data="private cached value"
                    )
                )
            return None

    class Permission:
        def on_event(self, payload):
            if (
                case in {"rewrite", "block", "allow_rewrite"}
                and payload.event is LifecycleEvent.PRE_TOOL_USE
                and payload.tool_call.tool == "read_resource"
            ):
                checked_scopes.append(payload.tool_call.args["scope"])
                if payload.tool_call.args["scope"] != "public" and case != "allow_rewrite":
                    decision = Block if case == "block" else Deny
                    return decision("Only public data is authorized")
            return None

        def check_permission(self, tool_call, state):
            if case in {"cache", "allow_cache"} and tool_call.tool == "read_resource":
                checked_scopes.append(tool_call.args["scope"])
                return case == "allow_cache"
            return True

    intercept = Intercept()
    permission = Permission()
    if async_mode == "async_hooks":

        async def intercept_event(payload):
            return intercept.on_event(payload)

        async def permission_event(payload):
            return permission.on_event(payload)

        async def check_permission(tool_call, state):
            return permission.check_permission(tool_call, state)

        hooks = [
            SimpleNamespace(on_event=intercept_event),
            SimpleNamespace(on_event=permission_event, check_permission=check_permission),
        ]
    else:
        hooks = [intercept, permission]
    asynchronous = async_mode != "sync"
    backend = Backend()
    options = {
        "task": "Read public data.",
        "tools": [read_resource],
        "hooks": hooks,
        "config": LoopConfig(harness_profile="minimal", max_steps=2, use_native_tools=native),
    }
    if case == "failure":
        with pytest.raises(RuntimeError, match="policy lookup unavailable"):
            if asynchronous:
                await run_result_async(backend, **options)
            else:
                run_result(backend, **options)
        assert not effects
        assert backend.calls == 1
        return

    result = (
        await run_result_async(backend, **options)
        if asynchronous
        else run_result(backend, **options)
    )
    assert result.completed
    assert backend.calls == 2
    if case in {"rewrite", "block", "cache"}:
        assert not effects
        assert checked_scopes == (["public"] if case == "cache" else ["private"])
        assert "Permission denied" in result.steps[0].tool_result.error
        assert result.steps[0].tool_result.data is None
    else:
        expected_scope = "private" if case == "allow_rewrite" else "public"
        assert effects == ([] if case == "allow_cache" else [expected_scope])
        assert checked_scopes == ([] if case == "observer_failure" else [expected_scope])
        assert result.steps[0].tool_result.error is None
        assert result.steps[0].tool_result.data == (
            "private cached value" if case == "allow_cache" else {"scope": expected_scope}
        )


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
