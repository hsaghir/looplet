"""Smoke tests for Skill — composable tool+context bundles."""
from __future__ import annotations

import pytest

from looplet import BaseToolRegistry, Skill, StaticMemorySource
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} tool",
                    parameters={}, execute=lambda: {})


class TestSkill:
    def test_register_adds_tools(self):
        reg = BaseToolRegistry()
        skill = Skill(name="test", tools=[_spec("a"), _spec("b")])
        n = skill.register(reg)
        assert n == 2
        assert "a" in reg.tool_names
        assert "b" in reg.tool_names

    def test_tool_names(self):
        skill = Skill(name="x", tools=[_spec("foo"), _spec("bar")])
        assert skill.tool_names() == ["foo", "bar"]

    def test_catalog_entry(self):
        skill = Skill(name="py", description="Python dev", tools=[_spec("bash")])
        entry = skill.as_catalog_entry()
        assert "[py]" in entry
        assert "bash" in entry

    def test_with_memory(self):
        mem = StaticMemorySource("PEP 8 rules")
        skill = Skill(name="py", memory=mem, instructions="Write tests.")
        assert skill.memory is mem
        assert "tests" in skill.instructions

    def test_empty_skill(self):
        skill = Skill(name="empty")
        assert skill.register(BaseToolRegistry()) == 0
        assert skill.tool_names() == []

    def test_from_looplet_import(self):
        from looplet import Skill as S
        assert S is Skill
