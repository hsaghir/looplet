"""Smoke tests for HookDecision + normalize_hook_return + event-name API.

Validates that:
* the dataclass and ergonomic constructors behave as documented
* legacy hook returns (``str``, ``bool``, ``ToolResult``, ``None``) coerce
  correctly via ``normalize_hook_return``
* the composable loop honors the new HookDecision fields end-to-end
  at ``pre_dispatch``, ``post_dispatch``, ``check_permission``,
  ``check_done``, and ``should_stop``
"""

from __future__ import annotations

import pytest

from looplet import (
    Allow,
    BaseToolRegistry,
    Block,
    Continue,
    DefaultState,
    Deny,
    HookDecision,
    InjectContext,
    LifecycleEvent,
    LoopConfig,
    Stop,
    composable_loop,
)
from looplet.hook_decision import normalize_hook_return
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import ToolResult


@pytest.mark.parametrize("is_async", [False, True])
async def test_event_dispatch_preserves_deduplication_errors_and_live_order(is_async: bool, caplog):
    from looplet.loop import emit_event, emit_event_async

    seen = []
    hooks = []

    class DualHook:
        def pre_dispatch(self, *args):
            return None

        def on_event(self, payload):
            pytest.fail("method-equivalent event must be deduplicated")

    class BrokenHook:
        def on_event(self, payload):
            raise RuntimeError("observer failure")

    class LateHook:
        def on_event(self, payload):
            seen.append("late")

    class RegisterHook:
        def on_event(self, payload):
            seen.append("register")
            hooks.append(LateHook())

    hooks.extend([DualHook(), BrokenHook(), RegisterHook()])
    if is_async:
        decisions = await emit_event_async(hooks, LifecycleEvent.PRE_TOOL_USE)
    else:
        decisions = emit_event(hooks, LifecycleEvent.PRE_TOOL_USE)
    assert decisions == []
    assert seen == ["register", "late"]
    assert "observer failure" in caplog.text


class TestHookDecisionDataclass:
    def test_defaults_are_noop(self):
        d = HookDecision()
        assert d.is_noop()
        assert not d.is_block()
        assert not d.is_stop()

    def test_block_detected(self):
        assert HookDecision(block="nope").is_block()
        assert Block("nope").is_block()

    def test_deny_detected_as_block(self):
        d = Deny("not allowed")
        assert d.is_block()
        assert d.block == "not allowed"
        assert d.permission == "deny"

    def test_stop_detected(self):
        assert Stop("budget").is_stop()
        assert Stop("budget").stop == "budget"

    def test_allow_default_shape(self):
        d = Allow()
        assert d.permission == "allow"
        assert d.updated_args is None

    def test_allow_with_updated_args(self):
        d = Allow(updated_args={"x": 1})
        assert d.updated_args == {"x": 1}

    def test_continue_with_context(self):
        d = Continue("hint")
        assert d.additional_context == "hint"
        assert not d.is_stop()

    def test_inject_context_sets_additional_context(self):
        d = InjectContext("remember this")
        assert d.additional_context == "remember this"
        assert d.is_noop() is False


class TestNormaliseHookReturn:
    @pytest.mark.parametrize(
        ("slot", "decision", "ignored"),
        [
            ("check_done", Stop("not a completion gate"), ("stop",)),
            ("pre_dispatch", Block("not a permission decision"), ("block",)),
            ("pre_prompt", HookDecision(updated_args={"x": 1}), ("updated_args",)),
            ("stop", Stop("already terminal"), ("stop",)),
        ],
    )
    def test_unsupported_effects_are_visible_without_changing_legacy_behavior(
        self, caplog, slot, decision, ignored
    ):
        assert normalize_hook_return(decision, slot=slot) is decision
        assert decision.ignored_effects(slot) == ignored
        assert "ignores effect fields" in caplog.text
        assert all(name in caplog.text for name in ignored)

    def test_applicability_is_shared_and_audit_metadata_is_not_an_effect(self, caplog):
        from looplet.hook_decision import hook_effect_fields

        assert hook_effect_fields("pre_dispatch") == hook_effect_fields("pre_tool_use")
        assert hook_effect_fields("post_dispatch") == hook_effect_fields("post_tool_use")
        assert hook_effect_fields("post_dispatch") == hook_effect_fields("post_tool_failure")
        decision = HookDecision(additional_context="hint", metadata={"host": True})
        assert normalize_hook_return(decision, slot="pre_prompt") is decision
        assert decision.ignored_effects("pre_prompt") == ()
        assert not caplog.text

    def test_none_is_none(self):
        assert normalize_hook_return(None, slot="pre_prompt") is None

    def test_passthrough_hook_decision(self):
        d = Block("stop")
        assert normalize_hook_return(d, slot="check_done") is d

    def test_str_to_inject_for_briefing_slots(self):
        out = normalize_hook_return("hi", slot="pre_prompt")
        assert out is not None
        assert out.additional_context == "hi"

    def test_str_to_block_for_check_done(self):
        out = normalize_hook_return("not yet", slot="check_done")
        assert out is not None
        assert out.block == "not yet"

    def test_bool_to_permission_for_check_permission(self):
        assert normalize_hook_return(True, slot="check_permission").permission == "allow"
        d = normalize_hook_return(False, slot="check_permission")
        assert d.permission == "deny"
        assert d.block == "permission denied"

    def test_bool_to_stop_for_should_stop(self):
        assert normalize_hook_return(False, slot="should_stop") is None
        assert normalize_hook_return(True, slot="should_stop").is_stop()

    def test_tool_result_to_updated_result(self):
        r = ToolResult(tool="x", args_summary="", data=None)
        out = normalize_hook_return(r, slot="pre_dispatch")
        assert out.updated_result is r

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            normalize_hook_return(object(), slot="pre_dispatch")


# ── End-to-end wiring tests ──────────────────────────────────


def _tools_with_add_and_done() -> BaseToolRegistry:
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


class TestHookDecisionWiringPreDispatch:
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("event_slot", ["pre_llm_call", "post_llm_response"])
    @pytest.mark.parametrize("request_stop", [False, True])
    async def test_model_event_effects_conform_between_drivers(
        self, async_mode, event_slot, request_stop
    ):
        from looplet import async_composable_loop
        from looplet.testing import AsyncMockLLMBackend

        seen = []

        class Hook:
            def on_event(self, payload):
                if payload.event.value == event_slot:
                    seen.append(payload.step_num)
                    return (
                        Stop("model_event_stop")
                        if request_stop
                        else InjectContext("model event context")
                    )
                return None

        backend = (AsyncMockLLMBackend if async_mode else MockLLMBackend)(
            [
                '{"tool":"add","args":{"a":1,"b":2}}',
                '{"tool":"done","args":{"answer":"ok"}}',
            ]
        )
        state = DefaultState(max_steps=3)
        kwargs = dict(
            llm=backend,
            tools=_tools_with_add_and_done(),
            state=state,
            hooks=[Hook()],
            config=LoopConfig(max_steps=3, use_native_tools=False),
        )
        steps = (
            [step async for step in async_composable_loop(**kwargs)]
            if async_mode
            else list(composable_loop(**kwargs))
        )
        if request_stop:
            assert [step.tool_call.tool for step in steps] == ["add"]
            assert state._stop_reason == "model_event_stop"
            assert seen == [1]
        else:
            assert [step.tool_call.tool for step in steps] == ["add", "done"]
            assert "model event context" in backend.last_prompt
            assert seen == [1, 2]

    def test_check_done_hook_failure_rejects_completion(self):
        llm = MockLLMBackend(
            responses=[
                '{"tool":"done","args":{"answer":"first"},"reasoning":""}',
                '{"tool":"done","args":{"answer":"second"},"reasoning":""}',
            ]
        )

        class BrokenGate:
            def check_done(self, state, session_log, context, step_num, tool_call=None):
                raise RuntimeError("gate unavailable")

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=2),
                hooks=[BrokenGate()],
                config=LoopConfig(max_steps=2),
            )
        )

        assert steps[0].tool_result.data["rejected"] is True
        assert "quality gate" in steps[0].tool_result.error.lower()

    def test_updated_args_rewrites_tool_input(self):
        """A pre_dispatch hook that returns Allow(updated_args=...) rewrites
        the call before dispatch."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":1},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class Rewriter:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "add":
                    return Allow(updated_args={"a": 99, "b": 1})
                return None

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[Rewriter()],
                config=LoopConfig(max_steps=5),
            )
        )
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": 100}

    def test_deny_short_circuits_into_permission_error(self):
        """Deny(reason) from pre_dispatch records PERMISSION_DENIED."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class Blocker:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "add":
                    return Deny("not in sandbox")
                return None

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[Blocker()],
                config=LoopConfig(max_steps=5),
            )
        )
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.error is not None
        assert "not in sandbox" in add_step.tool_result.error

    def test_legacy_tool_result_still_intercepts(self):
        """Returning a plain ToolResult from pre_dispatch still works."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )
        fixture = ToolResult(tool="add", args_summary="", data={"sum": 42})

        class Mock:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "add":
                    return fixture
                return None

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[Mock()],
                config=LoopConfig(max_steps=5),
            )
        )
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": 42}


class TestHookDecisionWiringPostDispatch:
    def test_updated_result_rewrites(self):
        """A post_dispatch hook can rewrite tool_result before it's recorded."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class Masker:
            def post_dispatch(self, state, session_log, tc, tr, step_num):
                if tc.tool == "add":
                    return HookDecision(
                        updated_result=ToolResult(
                            tool=tc.tool,
                            args_summary="",
                            data={"sum": "***"},
                        )
                    )
                return None

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[Masker()],
                config=LoopConfig(max_steps=5),
            )
        )
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": "***"}

    def test_stop_terminates_after_step(self):
        """HookDecision.stop from post_dispatch exits the loop cleanly
        after the current step without spending another LLM call."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                # These would be spent only if the loop kept running.
                '{"tool":"add","args":{"a":3,"b":4},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class EarlyStop:
            def post_dispatch(self, state, session_log, tc, tr, step_num):
                return Stop("budget_probe")

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[EarlyStop()],
                config=LoopConfig(max_steps=5),
            )
        )
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "add"


class TestHookDecisionWiringCheckPermission:
    def test_deny_surfaces_custom_reason(self):
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class ReasonedDeny:
            def check_permission(self, tool_call, state):
                if tool_call.tool == "add":
                    return Deny("sandbox forbids arithmetic")
                return True

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[ReasonedDeny()],
                config=LoopConfig(max_steps=5),
            )
        )
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert "sandbox forbids arithmetic" in (add_step.tool_result.error or "")


class TestHookDecisionWiringCheckDone:
    def test_block_rejects_done(self):
        llm = MockLLMBackend(
            responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok2"},"reasoning":""}',
            ]
        )

        class BlockOnce:
            called = 0

            def check_done(self, state, session_log, context, step_num):
                self.called += 1
                if self.called == 1:
                    return Block("not yet")
                return None

        b = BlockOnce()
        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[b],
                config=LoopConfig(max_steps=5),
            )
        )
        # Should have looped through an add between blocked and accepted done.
        assert any(s.tool_call.tool == "add" for s in steps)


class TestHookDecisionWiringShouldStop:
    def test_stop_with_reason(self):
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"add","args":{"a":3,"b":4},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class Cap:
            def should_stop(self, state, step_num, new_entities):
                if step_num >= 1:
                    return Stop("step_cap")
                return None

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[Cap()],
                config=LoopConfig(max_steps=5),
            )
        )
        # Stopped after first add, before the second.
        assert len(steps) == 1

    def test_legacy_bool_still_works(self):
        """should_stop returning True stops the loop (back-compat)."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            ]
        )

        class LegacyStop:
            def should_stop(self, state, step_num, new_entities):
                return step_num >= 1

        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools_with_add_and_done(),
                state=DefaultState(max_steps=5),
                hooks=[LegacyStop()],
                config=LoopConfig(max_steps=5),
            )
        )
        assert len(steps) == 1


class TestLifecycleEventEnum:
    def test_canonical_events_present(self):
        # The curated events that we ship.
        want = {
            "session_start",
            "pre_llm_call",
            "post_llm_response",
            "pre_tool_use",
            "tool_progress",
            "post_tool_use",
            "post_tool_failure",
            "pre_compact",
            "post_compact",
            "hook_decision",
            "done_accepted",
            "stop",
            "subagent_start",
            "subagent_stop",
        }
        assert set(e.value for e in LifecycleEvent) == want
