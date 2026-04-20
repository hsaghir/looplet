"""Smoke tests for :class:`PermissionHook` — the unified permission path."""
from __future__ import annotations

import pytest

from openharness import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    PermissionEngine,
    PermissionHook,
    composable_loop,
)
from openharness.permissions import PermissionDecision
from openharness.testing import MockLLMBackend
from openharness.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="dangerous",
        description="dangerous",
        parameters={"cmd": "str"},
        execute=lambda *, cmd: {"ran": cmd},
    ))
    reg.register(ToolSpec(
        name="safe",
        description="safe",
        parameters={},
        execute=lambda: {"ok": True},
    ))
    reg.register(ToolSpec(
        name="done",
        description="done",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


def _run(hooks, *, calls=None):
    calls = calls or [
        '{"tool":"dangerous","args":{"cmd":"rm -rf /"},"reasoning":"r"}',
        '{"tool":"safe","args":{},"reasoning":"r"}',
        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
    ]
    state = DefaultState(max_steps=5)
    return list(composable_loop(
        llm=MockLLMBackend(responses=calls),
        tools=_tools(),
        state=state,
        hooks=hooks,
        config=LoopConfig(max_steps=5),
    ))


class TestPermissionHook:
    def test_deny_rule_blocks_tool(self):
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny("dangerous", reason="too risky")
        steps = _run([PermissionHook(engine)])
        danger = next(s for s in steps if s.tool_call.tool == "dangerous")
        assert danger.tool_result.error
        assert "too risky" in danger.tool_result.error

    def test_allow_rule_lets_tool_run(self):
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.allow("dangerous")
        steps = _run([PermissionHook(engine)])
        danger = next(s for s in steps if s.tool_call.tool == "dangerous")
        assert danger.tool_result.error is None
        assert danger.tool_result.data == {"ran": "rm -rf /"}

    def test_default_deny_blocks_unknown_tools(self):
        engine = PermissionEngine(default=PermissionDecision.DENY)
        engine.allow("safe")
        engine.allow("done")
        steps = _run([PermissionHook(engine)])
        danger = next(s for s in steps if s.tool_call.tool == "dangerous")
        assert danger.tool_result.error

    def test_arg_matcher_blocks_specific_calls(self):
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny("dangerous", arg_matcher=lambda a: "rm -rf" in a.get("cmd", ""))
        steps = _run([PermissionHook(engine)])
        danger = next(s for s in steps if s.tool_call.tool == "dangerous")
        assert danger.tool_result.error

    def test_denial_recorded_on_engine(self):
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny("dangerous")
        _run([PermissionHook(engine)])
        assert len(engine.denials) == 1
        assert engine.denials[0]["tool"] == "dangerous"

    def test_unified_hook_and_method_both_work(self):
        """Back-compat: check_permission method still works for legacy callers."""
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny("dangerous")
        hook = PermissionHook(engine)

        from openharness.types import ToolCall
        tc = ToolCall(tool="dangerous", args={"cmd": "x"}, reasoning="r")
        assert hook.check_permission(tc, None) is False

        tc_safe = ToolCall(tool="safe", args={}, reasoning="r")
        assert hook.check_permission(tc_safe, None) is True
