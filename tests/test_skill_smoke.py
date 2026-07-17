"""Smoke tests for Skill - composable tool+context bundles."""

from __future__ import annotations

import textwrap

import pytest

from looplet import (
    BaseToolRegistry,
    LoopConfig,
    Skill,
    StaticMemorySource,
    composable_loop,
    register_done_tool,
)
from looplet.skills import FileSkillStore, SkillActivationHook, SkillManager, make_skill_tools
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import DefaultState, ToolCall

pytestmark = pytest.mark.smoke


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} tool", parameters={}, execute=lambda: {})


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


def _write_skill(root, dirname: str, text: str) -> None:
    skill_dir = root / dirname
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


class TestFileSkillStore:
    def test_loads_anthropic_style_skill_frontmatter(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf",
            """
            ---
            name: pdf
            description: Use this skill whenever the user wants to do anything with PDF files.
            license: Proprietary. LICENSE.txt has complete terms
            ---

            # PDF Processing Guide

            Use pypdf or pdfplumber depending on the task.
            """,
        )

        store = FileSkillStore(tmp_path)
        cards = store.list()
        skill = store.load("pdf")

        assert [card.name for card in cards] == ["pdf"]
        assert cards[0].description.startswith("Use this skill")
        assert skill.name == "pdf"
        assert skill.description.startswith("Use this skill")
        assert skill.metadata["license"].startswith("Proprietary")
        assert skill.instructions.startswith("# PDF Processing Guide")

    def test_search_scores_name_description_tags_and_body(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf",
            """
            ---
            name: pdf
            description: Work with PDF documents.
            tags: [documents]
            ---

            Extract text and merge pages.
            """,
        )
        _write_skill(
            tmp_path,
            "xlsx",
            """
            ---
            name: xlsx
            description: Create and edit spreadsheet files.
            tags: [documents, finance]
            ---

            Always use formulas and verify workbook recalculation.
            """,
        )

        store = FileSkillStore(tmp_path)

        assert store.search("spreadsheet formulas", limit=1)[0].name == "xlsx"
        assert store.search("pdf pages", limit=1)[0].name == "pdf"

    def test_can_load_root_skill_directory(self, tmp_path):
        (tmp_path / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: root-skill
                description: Loaded directly from the root directory.
                ---

                # Root Skill
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        store = FileSkillStore(tmp_path)

        assert store.list()[0].name == "root-skill"


class TestSkillActivation:
    def test_activation_hook_injects_only_active_skills(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf",
            """
            ---
            name: pdf
            description: Work with PDF files.
            ---

            # PDF Processing Guide

            Follow the PDF workflow.
            """,
        )
        manager = SkillManager(FileSkillStore(tmp_path))
        hook = SkillActivationHook(manager)

        assert hook.pre_prompt(DefaultState(), None, None, 1) is None

        manager.activate("pdf")
        injected = hook.pre_prompt(DefaultState(), None, None, 2)

        assert injected is not None
        assert "ACTIVE SKILLS" in injected
        assert "# PDF Processing Guide" in injected
        assert manager.active_names == ["pdf"]

    def test_skill_tools_search_and_activate(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf",
            """
            ---
            name: pdf
            description: Work with PDF files.
            ---

            # PDF Processing Guide
            """,
        )
        manager = SkillManager(FileSkillStore(tmp_path))
        registry = BaseToolRegistry()
        for spec in make_skill_tools(manager):
            registry.register(spec)

        search = registry.dispatch(ToolCall("search_skills", {"query": "pdf"}))
        activate = registry.dispatch(ToolCall("activate_skill", {"name": "pdf"}))

        assert search.error is None
        assert search.data["skills"][0]["name"] == "pdf"
        assert activate.error is None
        assert activate.data["activated"] == "pdf"
        assert activate.data["active_skills"] == ["pdf"]

    def test_agent_loop_searches_activates_and_injects_skill(self, tmp_path):
        _write_skill(
            tmp_path,
            "pdf",
            """
            ---
            name: pdf
            description: Work with PDF files.
            ---

            # PDF Processing Guide

            Extract text with the PDF workflow.
            """,
        )
        manager = SkillManager(FileSkillStore(tmp_path))
        registry = BaseToolRegistry()
        for spec in make_skill_tools(manager):
            registry.register(spec)
        register_done_tool(registry)

        class CapturePromptsLLM(MockLLMBackend):
            def __init__(self, responses: list[str]) -> None:
                super().__init__(responses)
                self.prompts: list[str] = []

            def generate(
                self,
                prompt: str,
                *,
                max_tokens: int = 2000,
                system_prompt: str = "",
                temperature: float = 0.2,
            ) -> str:
                self.prompts.append(prompt)
                return super().generate(
                    prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )

        llm = CapturePromptsLLM(
            [
                '{"tool": "search_skills", "args": {"query": "extract text from a pdf"}, "reasoning": "find skill"}',
                '{"tool": "activate_skill", "args": {"name": "pdf"}, "reasoning": "load skill"}',
                '{"tool": "done", "args": {"summary": "PDF skill is active."}, "reasoning": "finish"}',
            ]
        )

        steps = list(
            composable_loop(
                llm=llm,
                tools=registry,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                hooks=[SkillActivationHook(manager)],
                task={"goal": "Use the right skill for extracting text from a PDF."},
            )
        )

        assert [step.tool_call.tool for step in steps] == [
            "search_skills",
            "activate_skill",
            "done",
        ]
        assert manager.active_names == ["pdf"]
        assert "PDF Processing Guide" not in llm.prompts[0]
        assert "PDF Processing Guide" not in llm.prompts[1]
        assert "=== ACTIVE SKILLS ===" in llm.prompts[2]
        assert "PDF Processing Guide" in llm.prompts[2]
