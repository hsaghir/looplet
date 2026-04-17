"""Tests for the declarative PermissionEngine."""

from __future__ import annotations

from openharness.loop import LoopConfig, composable_loop
from openharness.permissions import (
    PermissionDecision,
    PermissionEngine,
    PermissionOutcome,
    PermissionRule,
)
from openharness.tools import BaseToolRegistry, ToolSpec
from openharness.types import DefaultState, ErrorKind, LLMBackend, ToolCall


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
            tool="bash", decision=PermissionDecision.DENY,
            arg_matcher=lambda a: "rm" in str(a.get("cmd", "")),
        )
        assert r.matches(ToolCall(tool="bash", args={"cmd": "rm -rf /"}))
        assert not r.matches(ToolCall(tool="bash", args={"cmd": "ls"}))

    def test_buggy_matcher_fails_closed(self):
        """A buggy arg_matcher raises but the rule still matches (fail closed)."""
        r = PermissionRule(
            tool="bash", decision=PermissionDecision.DENY,
            arg_matcher=lambda a: 1 / 0,
        )
        # Fail closed: rule matches even though matcher raised — safe for DENY rules.
        assert r.matches(ToolCall(tool="bash", args={})) is True

    def test_buggy_matcher_on_allow_rule_does_not_grant(self):
        """A buggy arg_matcher on an ALLOW rule must NOT match (fail closed)."""
        r = PermissionRule(
            tool="bash", decision=PermissionDecision.ALLOW,
            arg_matcher=lambda a: 1 / 0,
        )
        # Fail closed for ALLOW means: skip this rule so it doesn't grant access.
        assert r.matches(ToolCall(tool="bash", args={})) is False


class TestEngineEvaluate:
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
        reg.register(ToolSpec(
            name="danger", description="d", parameters={"cmd": "c"},
            execute=lambda cmd="": (ran.append(cmd) or {"ok": True}),
            concurrent_safe=False,
        ))
        reg.register(ToolSpec(
            name="done", description="", parameters={"summary": "s"},
            execute=lambda summary="": {"done": True, "summary": summary},
        ))
        return reg

    def test_engine_blocks_dispatch(self):
        ran: list[str] = []
        eng = PermissionEngine()
        eng.deny("danger", reason="forbidden")
        cfg = LoopConfig(max_steps=3, permissions=eng)
        llm = _LLM(
            '```json\n{"tool": "danger", "args": {"cmd": "rm -rf /"}}\n```',
            '```json\n{"tool": "done", "args": {"summary": "x"}}\n```',
        )
        steps = list(composable_loop(
            llm=llm, task={"id": "T"}, tools=self._reg(ran),
            config=cfg, state=DefaultState(max_steps=3),
        ))
        assert ran == []  # tool body never ran
        # The denied step should have a ToolError with PERMISSION_DENIED.
        denied_step = steps[0]
        assert denied_step.tool_result.error is not None
        assert denied_step.tool_result.error_kind == ErrorKind.PERMISSION_DENIED
        # Engine recorded the denial.
        assert len(eng.denials) == 1

    def test_engine_allows_tool_by_default(self):
        ran: list[str] = []
        eng = PermissionEngine(default=PermissionDecision.ALLOW)
        cfg = LoopConfig(max_steps=3, permissions=eng)
        llm = _LLM(
            '```json\n{"tool": "danger", "args": {"cmd": "ls"}}\n```',
            '```json\n{"tool": "done", "args": {"summary": "x"}}\n```',
        )
        list(composable_loop(
            llm=llm, task={"id": "T"}, tools=self._reg(ran),
            config=cfg, state=DefaultState(max_steps=3),
        ))
        assert ran == ["ls"]
