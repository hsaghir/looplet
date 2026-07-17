"""Round-10 friction fixes: clone_tools_excluding typo warning + permission audit dunder strip."""

from __future__ import annotations

import logging

import pytest

from looplet.permissions import PermissionDecision, PermissionEngine
from looplet.subagent import clone_tools_excluding
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import ToolCall

pytestmark = pytest.mark.smoke


class TestCloneToolsExcludingTypo:
    def _make_parent(self) -> BaseToolRegistry:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(name="finalize", description="d", parameters={}, execute=lambda: None)
        )
        reg.register(ToolSpec(name="search", description="d", parameters={}, execute=lambda: None))
        return reg

    def test_exclude_typo_logs_warning_and_tool_leaks(self, caplog):
        parent = self._make_parent()
        with caplog.at_level(logging.WARNING, logger="looplet.subagent"):
            sub = clone_tools_excluding(parent, ["finish"])  # typo: should be "finalize"
        assert any("not registered on the parent" in rec.message for rec in caplog.records)
        # And the state-mutating tool is still present - this is the real footgun
        # the warning is trying to surface.
        assert "finalize" in sub._tools
        assert "search" in sub._tools

    def test_correct_exclude_no_warning(self, caplog):
        parent = self._make_parent()
        with caplog.at_level(logging.WARNING, logger="looplet.subagent"):
            sub = clone_tools_excluding(parent, ["finalize"])
        assert not any("not registered" in rec.message for rec in caplog.records)
        assert "finalize" not in sub._tools
        assert "search" in sub._tools


class TestPermissionAuditStripsDunderArgs:
    def test_denial_entry_excludes_dunder_keys(self):
        engine = PermissionEngine(default=PermissionDecision.DENY)
        engine.deny("bash", reason="no shell")
        call = ToolCall(
            tool="bash",
            args={"cmd": "ls", "__theory__": "poking around", "__internal": "x"},
            reasoning="r",
        )
        outcome = engine.evaluate(call)
        assert outcome.denied
        assert len(engine.denials) == 1
        audited = engine.denials[0]["args"]
        assert audited == {"cmd": "ls"}
        assert "__theory__" not in audited
        assert "__internal" not in audited

    def test_denial_preserves_clean_args(self):
        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny("bash")
        call = ToolCall(tool="bash", args={"cmd": "rm -rf /", "flags": "-y"}, reasoning="r")
        engine.evaluate(call)
        assert engine.denials[0]["args"] == {"cmd": "rm -rf /", "flags": "-y"}
