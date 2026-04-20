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

from openharness import (
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
from openharness.hook_decision import normalize_hook_return
from openharness.testing import MockLLMBackend
from openharness.tools import ToolSpec
from openharness.types import ToolResult


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
    reg.register(ToolSpec(
        name="add",
        description="add",
        parameters={"a": "int", "b": "int"},
        execute=lambda *, a, b: {"sum": a + b},
    ))
    reg.register(ToolSpec(
        name="done",
        description="done",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


class TestHookDecisionWiringPreDispatch:
    def test_updated_args_rewrites_tool_input(self):
        """A pre_dispatch hook that returns Allow(updated_args=...) rewrites
        the call before dispatch."""
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":1},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class Rewriter:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "add":
                    return Allow(updated_args={"a": 99, "b": 1})
                return None

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[Rewriter()], config=LoopConfig(max_steps=5),
        ))
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": 100}

    def test_deny_short_circuits_into_permission_error(self):
        """Deny(reason) from pre_dispatch records PERMISSION_DENIED."""
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class Blocker:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "add":
                    return Deny("not in sandbox")
                return None

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[Blocker()], config=LoopConfig(max_steps=5),
        ))
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.error is not None
        assert "not in sandbox" in add_step.tool_result.error

    def test_legacy_tool_result_still_intercepts(self):
        """Returning a plain ToolResult from pre_dispatch still works."""
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])
        fixture = ToolResult(tool="add", args_summary="", data={"sum": 42})

        class Mock:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "add":
                    return fixture
                return None

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[Mock()], config=LoopConfig(max_steps=5),
        ))
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": 42}


class TestHookDecisionWiringPostDispatch:
    def test_updated_result_rewrites(self):
        """A post_dispatch hook can rewrite tool_result before it's recorded."""
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class Masker:
            def post_dispatch(self, state, session_log, tc, tr, step_num):
                if tc.tool == "add":
                    return HookDecision(updated_result=ToolResult(
                        tool=tc.tool, args_summary="", data={"sum": "***"},
                    ))
                return None

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[Masker()], config=LoopConfig(max_steps=5),
        ))
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert add_step.tool_result.data == {"sum": "***"}

    def test_stop_terminates_after_step(self):
        """HookDecision.stop from post_dispatch exits the loop cleanly
        after the current step without spending another LLM call."""
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            # These would be spent only if the loop kept running.
            '{"tool":"add","args":{"a":3,"b":4},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class EarlyStop:
            def post_dispatch(self, state, session_log, tc, tr, step_num):
                return Stop("budget_probe")

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[EarlyStop()], config=LoopConfig(max_steps=5),
        ))
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "add"


class TestHookDecisionWiringCheckPermission:
    def test_deny_surfaces_custom_reason(self):
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class ReasonedDeny:
            def check_permission(self, tool_call, state):
                if tool_call.tool == "add":
                    return Deny("sandbox forbids arithmetic")
                return True

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[ReasonedDeny()], config=LoopConfig(max_steps=5),
        ))
        add_step = next(s for s in steps if s.tool_call.tool == "add")
        assert "sandbox forbids arithmetic" in (add_step.tool_result.error or "")


class TestHookDecisionWiringCheckDone:
    def test_block_rejects_done(self):
        llm = MockLLMBackend(responses=[
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok2"},"reasoning":""}',
        ])

        class BlockOnce:
            called = 0

            def check_done(self, state, session_log, context, step_num):
                self.called += 1
                if self.called == 1:
                    return Block("not yet")
                return None

        b = BlockOnce()
        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[b], config=LoopConfig(max_steps=5),
        ))
        # Should have looped through an add between blocked and accepted done.
        assert any(s.tool_call.tool == "add" for s in steps)


class TestHookDecisionWiringShouldStop:
    def test_stop_with_reason(self):
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"add","args":{"a":3,"b":4},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class Cap:
            def should_stop(self, state, step_num, new_entities):
                if step_num >= 1:
                    return Stop("step_cap")
                return None

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[Cap()], config=LoopConfig(max_steps=5),
        ))
        # Stopped after first add, before the second.
        assert len(steps) == 1

    def test_legacy_bool_still_works(self):
        """should_stop returning True stops the loop (back-compat)."""
        llm = MockLLMBackend(responses=[
            '{"tool":"add","args":{"a":1,"b":2},"reasoning":""}',
            '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
        ])

        class LegacyStop:
            def should_stop(self, state, step_num, new_entities):
                return step_num >= 1

        steps = list(composable_loop(
            llm=llm, tools=_tools_with_add_and_done(),
            state=DefaultState(max_steps=5),
            hooks=[LegacyStop()], config=LoopConfig(max_steps=5),
        ))
        assert len(steps) == 1


class TestLifecycleEventEnum:
    def test_canonical_ten_events_present(self):
        # The curated events that we ship.
        want = {
            "session_start", "pre_llm_call", "post_llm_response",
            "pre_tool_use", "tool_progress",
            "post_tool_use", "post_tool_failure",
            "pre_compact", "post_compact", "stop",
            "subagent_start", "subagent_stop",
        }
        assert set(e.value for e in LifecycleEvent) == want
