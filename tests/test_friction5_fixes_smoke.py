"""Round-5 friction fix (2026-04-24).

``install_skills(skills, registry, ...)`` lets users load a bundle
of skills in one call instead of manually registering tools,
concatenating system prompts, and extending memory_sources -
three operations that were easy to forget piecemeal.
"""

from __future__ import annotations

import pytest

from looplet import BaseToolRegistry, LoopConfig, Skill, ToolSpec, install_skills
from looplet.memory import StaticMemorySource

pytestmark = pytest.mark.smoke


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} tool", parameters={}, execute=lambda: {})


class TestInstallSkills:
    def test_registers_tools(self) -> None:
        reg = BaseToolRegistry()
        a = Skill(name="a", tools=[_spec("alpha"), _spec("beta")])
        install_skills([a], reg)
        assert set(reg.tool_names) == {"alpha", "beta"}

    def test_concatenates_instructions(self) -> None:
        reg = BaseToolRegistry()
        a = Skill(name="a", tools=[], instructions="A inst")
        b = Skill(name="b", tools=[], instructions="B inst")
        out = install_skills([a, b], reg)
        assert "A inst" in out["system_prompt"]
        assert "B inst" in out["system_prompt"]
        # Blank line separator by default.
        assert "A inst\n\nB inst" in out["system_prompt"]

    def test_base_system_prompt_prepended(self) -> None:
        reg = BaseToolRegistry()
        a = Skill(name="a", tools=[], instructions="A")
        out = install_skills([a], reg, base_system_prompt="BASE")
        assert out["system_prompt"].startswith("BASE")
        assert "A" in out["system_prompt"]

    def test_memory_sources_collected(self) -> None:
        reg = BaseToolRegistry()
        mem_a = StaticMemorySource("memA")
        mem_b = StaticMemorySource("memB")
        a = Skill(name="a", tools=[], memory=mem_a)
        b = Skill(name="b", tools=[], memory=mem_b)
        c = Skill(name="c", tools=[])  # no memory
        out = install_skills([a, b, c], reg)
        assert out["memory_sources"] == [mem_a, mem_b]

    def test_base_memory_sources_preserved(self) -> None:
        reg = BaseToolRegistry()
        existing = StaticMemorySource("existing")
        new = StaticMemorySource("new")
        a = Skill(name="a", tools=[], memory=new)
        out = install_skills([a], reg, base_memory_sources=[existing])
        assert out["memory_sources"] == [existing, new]

    def test_skills_with_no_instructions_produce_empty_prompt(self) -> None:
        reg = BaseToolRegistry()
        a = Skill(name="a", tools=[_spec("x")])
        out = install_skills([a], reg)
        assert out["system_prompt"] == ""

    def test_result_feeds_LoopConfig(self) -> None:
        reg = BaseToolRegistry()
        a = Skill(name="a", tools=[_spec("x")], instructions="A", memory=StaticMemorySource("M"))
        out = install_skills([a], reg)
        cfg = LoopConfig(max_steps=5, **out)
        assert cfg.system_prompt == "A"
        assert len(cfg.memory_sources) == 1

    def test_empty_skill_list(self) -> None:
        reg = BaseToolRegistry()
        out = install_skills([], reg)
        assert out == {"system_prompt": "", "memory_sources": []}
