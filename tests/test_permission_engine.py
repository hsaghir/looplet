"""Tests for the declarative PermissionEngine."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from unittest.mock import patch

import pytest

from looplet.async_loop import async_composable_loop
from looplet.loop import LoopConfig, composable_loop
from looplet.permissions import (
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    PermissionOutcome,
    PermissionRule,
)
from looplet.testing import AsyncMockLLMBackend, MockLLMBackend
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import DefaultState, ErrorKind, LLMBackend, ToolCall


class TestRuleMatching:
    def test_exact_tool_name_matches(self):
        r = PermissionRule(tool="bash", decision=PermissionDecision.DENY)
        assert r.matches(ToolCall(tool="bash", args={}))
        assert not r.matches(ToolCall(tool="read", args={}))

    def test_wildcard_matches_any_tool(self):
        r = PermissionRule(tool="*", decision=PermissionDecision.ALLOW)
        assert r.matches(ToolCall(tool="anything", args={}))

    def test_arg_matcher_gates_rule(self):
        r = PermissionRule(
            tool="bash",
            decision=PermissionDecision.DENY,
            arg_matcher=lambda a: "rm" in str(a.get("cmd", "")),
        )
        assert r.matches(ToolCall(tool="bash", args={"cmd": "rm -rf /"}))
        assert not r.matches(ToolCall(tool="bash", args={"cmd": "ls"}))

    def test_buggy_matcher_fails_closed(self):
        """A buggy arg_matcher raises but the rule still matches (fail closed)."""
        r = PermissionRule(
            tool="bash",
            decision=PermissionDecision.DENY,
            arg_matcher=lambda a: 1 / 0,
        )
        # Fail closed: rule matches even though matcher raised - safe for DENY rules.
        assert r.matches(ToolCall(tool="bash", args={})) is True

    def test_buggy_matcher_on_allow_rule_does_not_grant(self):
        """A buggy arg_matcher on an ALLOW rule must NOT match (fail closed)."""
        r = PermissionRule(
            tool="bash",
            decision=PermissionDecision.ALLOW,
            arg_matcher=lambda a: 1 / 0,
        )
        # Fail closed for ALLOW means: skip this rule so it doesn't grant access.
        assert r.matches(ToolCall(tool="bash", args={})) is False


class TestEngineEvaluate:
    @pytest.mark.parametrize(
        "default, expected",
        [
            (PermissionDecision.ALLOW, PermissionDecision.ALLOW),
            (PermissionDecision.DENY, PermissionDecision.DENY),
            (PermissionDecision.ASK, PermissionDecision.DENY),
            (PermissionDecision.DEFAULT, PermissionDecision.DENY),
        ],
    )
    def test_matching_default_rule_returns_concrete_decision(self, default, expected):
        engine = PermissionEngine(
            default=default,
            rules=[PermissionRule(tool="danger", decision=PermissionDecision.DEFAULT)],
        )

        outcome = engine.evaluate(ToolCall(tool="danger", args={}))

        assert outcome.decision == expected
        assert outcome.allowed != outcome.denied

    @pytest.mark.parametrize("source", ["matcher", "approval", "invalid_approval"])
    def test_permission_failure_logs_do_not_expose_argument_values(self, caplog, source):
        sentinel = "permission-secret-sentinel"

        def unavailable(*args):
            raise ValueError(f"invalid credential: {sentinel}")

        engine = PermissionEngine(default=PermissionDecision.DENY)
        if source == "matcher":
            engine.deny("danger", arg_matcher=unavailable)
        else:
            engine.ask("danger")
            if source == "approval":
                engine.ask_handler = unavailable
            else:
                engine.ask_handler = lambda call, rule: {"credential": sentinel}

        outcome = engine.evaluate(ToolCall(tool="danger", args={"credential": sentinel}))

        assert outcome.denied
        assert caplog.records
        assert sentinel not in caplog.text

    def test_first_match_wins(self):
        eng = PermissionEngine()
        eng.deny("bash", arg_matcher=lambda a: "rm" in a.get("cmd", ""), reason="danger")
        eng.allow("bash")
        out = eng.evaluate(ToolCall(tool="bash", args={"cmd": "rm -rf /"}))
        assert out.decision == PermissionDecision.DENY
        assert out.reason == "danger"

    def test_default_when_no_rule_matches(self):
        eng = PermissionEngine(default=PermissionDecision.DENY)
        out = eng.evaluate(ToolCall(tool="unknown", args={}))
        assert out.decision == PermissionDecision.DENY

    def test_ask_uses_handler(self):
        eng = PermissionEngine(
            ask_handler=lambda call, rule: PermissionDecision.ALLOW,
        )
        eng.ask("bash", reason="confirm")
        out = eng.evaluate(ToolCall(tool="bash", args={}))
        assert out.decision == PermissionDecision.ALLOW

    def test_ask_without_handler_falls_back_to_default(self):
        eng = PermissionEngine(default=PermissionDecision.DENY)
        eng.ask("bash")
        out = eng.evaluate(ToolCall(tool="bash", args={}))
        assert out.decision == PermissionDecision.DENY

    def test_explicit_ask_without_handler_denies_even_when_default_allows(self):
        eng = PermissionEngine(default=PermissionDecision.ALLOW)
        eng.ask("bash", reason="human review")
        assert eng.evaluate(ToolCall(tool="bash", args={})).denied
        assert eng.evaluate(ToolCall(tool="read", args={})).allowed
        assert len(eng.denials) == 1

    def test_handler_cannot_approve_changed_action_or_policy(self):
        eng = PermissionEngine()
        eng.ask("bash", reason="review")

        def change_action(call, rule):
            call.args["cmd"] = "different"
            return PermissionDecision.ALLOW

        eng.ask_handler = change_action
        assert eng.evaluate(ToolCall(tool="bash", args={"cmd": "original"})).denied

        def change_policy(call, rule):
            eng.rules.clear()
            return PermissionDecision.ALLOW

        eng.ask_handler = change_policy
        assert eng.evaluate(ToolCall(tool="bash", args={"cmd": "original"})).denied

    @pytest.mark.parametrize(
        "original, changed",
        [(True, 1), (False, 0), (1, 1.0), ([True], [1]), ({"value": False}, {"value": 0})],
    )
    def test_approval_rejects_argument_type_changes(self, original, changed):
        engine = PermissionEngine(default=PermissionDecision.DENY)
        engine.ask("danger")

        def approve(call, rule):
            call.args["value"] = changed
            return PermissionDecision.ALLOW

        engine.ask_handler = approve

        assert engine.evaluate(ToolCall(tool="danger", args={"value": original})).denied

    @pytest.mark.parametrize("change_action", [False, True])
    def test_seeded_nested_approval_mutations(self, change_action):
        generator = random.Random(0x5EC0)
        for case_index in range(128):
            original, changed = generator.choice(
                [(True, 1), (False, 0), (1, 1.0), (0, False), (1.0, 1)]
            )
            for depth in range(generator.randrange(1, 6)):
                if generator.choice([False, True]):
                    original = [None, original, "unchanged"]
                    changed = [None, changed, "unchanged"]
                else:
                    key = f"field-{depth}"
                    original = {key: original, "unchanged": "value"}
                    changed = {key: changed, "unchanged": "value"}
            replacement = changed if change_action else deepcopy(original)

            def approve(call, rule):
                call.args["value"] = replacement
                return PermissionDecision.ALLOW

            engine = PermissionEngine(ask_handler=approve, default=PermissionDecision.DENY)
            engine.ask("danger")
            outcome = engine.evaluate(ToolCall(tool="danger", args={"value": original}))

            assert outcome.denied is change_action, f"seed=0x5EC0 case={case_index}"
            assert outcome.allowed is not change_action, f"seed=0x5EC0 case={case_index}"

    def test_ask_handler_failure_denies(self):
        def unavailable(call, rule):
            raise RuntimeError("unavailable")

        eng = PermissionEngine(ask_handler=unavailable)
        eng.ask("bash")
        assert eng.evaluate(ToolCall(tool="bash", args={})).denied

    def test_ask_matcher_failure_denies_even_with_approval_handler(self):
        approvals = []
        eng = PermissionEngine(
            ask_handler=lambda call, rule: approvals.append(call) or PermissionDecision.ALLOW
        )
        eng.ask("bash", arg_matcher=lambda args: 1 / 0)
        assert eng.evaluate(ToolCall(tool="bash", args={})).denied
        assert approvals == []

    def test_denials_are_recorded(self):
        eng = PermissionEngine()
        eng.deny("bash", reason="blocked")
        eng.evaluate(ToolCall(tool="bash", args={"cmd": "ls"}))
        eng.evaluate(ToolCall(tool="bash", args={"cmd": "pwd"}))
        assert len(eng.denials) == 2
        assert eng.denials[0]["tool"] == "bash"
        assert eng.denials[0]["reason"] == "blocked"


class _LLM(LLMBackend):
    def __init__(self, *calls) -> None:
        self.scripts = list(calls)
        self.n = 0

    def generate(self, prompt: str, **kw) -> str:
        s = self.scripts[self.n]
        self.n += 1
        return s


class TestLoopIntegration:
    def _reg(self, ran: list[str]):
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="danger",
                description="d",
                parameters={"cmd": "c"},
                execute=lambda cmd="": ran.append(cmd) or {"ok": True},
                concurrent_safe=False,
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="",
                parameters={"summary": "s"},
                execute=lambda summary="": {"done": True, "summary": summary},
            )
        )
        return reg

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("native", [False, True])
    async def test_matching_default_rule_cannot_bypass_default_deny(self, async_mode, native):
        ran: list[str] = []
        engine = PermissionEngine(default=PermissionDecision.DENY)
        engine.rules.append(PermissionRule(tool="danger", decision=PermissionDecision.DEFAULT))
        response = '{"tool":"danger","args":{"cmd":"controlled-effect"}}'
        backend = (
            AsyncMockLLMBackend(responses=[response], cycle=False)
            if async_mode
            else MockLLMBackend(responses=[response], cycle=False)
        )
        arguments = {
            "tools": self._reg(ran),
            "config": LoopConfig(max_steps=1, use_native_tools=native),
            "state": DefaultState(max_steps=1),
            "hooks": [PermissionHook(engine)],
        }
        with patch.object(
            backend, "generate_with_tools", wraps=backend.generate_with_tools
        ) as native_generate:
            if async_mode:
                steps = [step async for step in async_composable_loop(llm=backend, **arguments)]
            else:
                steps = list(composable_loop(llm=backend, **arguments))

        assert native_generate.call_count == int(native)
        assert backend.calls == 1

        assert ran == []
        assert steps[0].tool_result.error_kind == ErrorKind.PERMISSION_DENIED
        assert len(engine.denials) == 1

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize(
        "original, changed",
        [(True, 1), (False, 0), (1, 1.0), ([True], [1]), ({"value": False}, {"value": 0})],
    )
    async def test_changed_approved_argument_types_never_execute(
        self, async_mode, native, original, changed
    ):
        ran: list[str] = []
        approvals = []
        engine = PermissionEngine(default=PermissionDecision.DENY)
        engine.ask("danger")

        def approve(call, rule):
            approvals.append(call.args["cmd"])
            return PermissionDecision.ALLOW

        class ChangeArgs:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                tool_call.args["cmd"] = changed

        engine.ask_handler = approve
        response = json.dumps({"tool": "danger", "args": {"cmd": original}})
        backend = (
            AsyncMockLLMBackend(responses=[response], cycle=False)
            if async_mode
            else MockLLMBackend(responses=[response], cycle=False)
        )
        arguments = {
            "tools": self._reg(ran),
            "config": LoopConfig(max_steps=1, use_native_tools=native),
            "state": DefaultState(max_steps=1),
            "hooks": [PermissionHook(engine), ChangeArgs()],
        }

        if async_mode:
            steps = [step async for step in async_composable_loop(llm=backend, **arguments)]
        else:
            steps = list(composable_loop(llm=backend, **arguments))

        assert approvals == [original]
        assert ran == []
        assert steps[0].tool_result.error_kind == ErrorKind.PERMISSION_DENIED

    def test_engine_blocks_dispatch(self):
        ran: list[str] = []
        eng = PermissionEngine()
        eng.deny("danger", reason="forbidden")
        cfg = LoopConfig(max_steps=3, use_native_tools=False)
        llm = _LLM(
            '```json\n{"tool": "danger", "args": {"cmd": "rm -rf /"}}\n```',
            '```json\n{"tool": "done", "args": {"summary": "x"}}\n```',
        )
        steps = list(
            composable_loop(
                llm=llm,
                task={"id": "T"},
                tools=self._reg(ran),
                config=cfg,
                state=DefaultState(max_steps=3),
                hooks=[PermissionHook(eng)],
            )
        )
        assert ran == []  # tool body never ran
        # The denied step should have a ToolError with PERMISSION_DENIED.
        denied_step = steps[0]
        assert denied_step.tool_result.error is not None
        assert denied_step.tool_result.error_kind == ErrorKind.PERMISSION_DENIED
        # Engine recorded the denial.
        assert len(eng.denials) == 1

    def test_ask_prompts_once_for_exact_call(self):
        ran: list[str] = []
        approvals: list[str] = []

        def approve(call, rule):
            approvals.append(call.args["cmd"])
            return PermissionDecision.ALLOW

        eng = PermissionEngine(ask_handler=approve)
        eng.ask("danger")
        steps = list(
            composable_loop(
                llm=_LLM(
                    '{"tool":"danger","args":{"cmd":"ls"}}',
                    '{"tool":"done","args":{"summary":"ok"}}',
                ),
                tools=self._reg(ran),
                config=LoopConfig(max_steps=3, use_native_tools=False),
                state=DefaultState(max_steps=3),
                hooks=[PermissionHook(eng)],
            )
        )
        assert approvals == ["ls"]
        assert ran == ["ls"]
        assert steps[0].tool_result.call_id == steps[0].tool_call.call_id

    def test_changed_approved_args_are_denied_without_second_approval(self):
        ran: list[str] = []
        approvals: list[str] = []

        def approve(call, rule):
            approvals.append(call.args["cmd"])
            return PermissionDecision.ALLOW

        class ChangeArgs:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "danger":
                    tool_call.args["cmd"] = "rm"

        eng = PermissionEngine(ask_handler=approve)
        eng.ask("danger")
        steps = list(
            composable_loop(
                llm=_LLM(
                    '{"tool":"danger","args":{"cmd":"ls"}}',
                    '{"tool":"done","args":{"summary":"ok"}}',
                ),
                tools=self._reg(ran),
                config=LoopConfig(max_steps=3, use_native_tools=False),
                state=DefaultState(max_steps=3),
                hooks=[PermissionHook(eng), ChangeArgs()],
            )
        )
        assert approvals == ["ls"]
        assert ran == []
        assert steps[0].tool_result.error_kind == ErrorKind.PERMISSION_DENIED
        assert len(eng.denials) == 1

    def test_changed_approved_policy_is_denied_without_second_approval(self):
        ran: list[str] = []
        approvals: list[str] = []

        def approve(call, rule):
            approvals.append(call.args["cmd"])
            return PermissionDecision.ALLOW

        eng = PermissionEngine(ask_handler=approve, policy_version="v1")
        eng.ask("danger")

        class ChangePolicy:
            def pre_dispatch(self, state, session_log, tool_call, step_num):
                if tool_call.tool == "danger":
                    eng.policy_version = "v2"

        steps = list(
            composable_loop(
                llm=_LLM(
                    '{"tool":"danger","args":{"cmd":"ls"}}',
                    '{"tool":"done","args":{"summary":"ok"}}',
                ),
                tools=self._reg(ran),
                config=LoopConfig(max_steps=3, use_native_tools=False),
                state=DefaultState(max_steps=3),
                hooks=[PermissionHook(eng), ChangePolicy()],
            )
        )
        assert approvals == ["ls"]
        assert ran == []
        assert steps[0].tool_result.error_kind == ErrorKind.PERMISSION_DENIED
        assert len(eng.denials) == 1

    def test_engine_allows_tool_by_default(self):
        ran: list[str] = []
        eng = PermissionEngine(default=PermissionDecision.ALLOW)
        cfg = LoopConfig(max_steps=3, use_native_tools=False)
        llm = _LLM(
            '```json\n{"tool": "danger", "args": {"cmd": "ls"}}\n```',
            '```json\n{"tool": "done", "args": {"summary": "x"}}\n```',
        )
        list(
            composable_loop(
                llm=llm,
                task={"id": "T"},
                tools=self._reg(ran),
                config=cfg,
                state=DefaultState(max_steps=3),
                hooks=[PermissionHook(eng)],
            )
        )
        assert ran == ["ls"]
