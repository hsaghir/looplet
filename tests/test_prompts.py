"""Tests for cadence.prompts — structured prompt assembly."""

from __future__ import annotations

import inspect

import pytest

from openharness.prompts import build_prompt

pytestmark = pytest.mark.smoke


# ── Basic prompt structure ────────────────────────────────────────


class TestBasicPromptSections:
    def test_task_section_present(self) -> None:
        prompt = build_prompt(task={"id": "t1", "description": "Do X"})
        assert "TASK" in prompt

    def test_tools_section_present(self) -> None:
        prompt = build_prompt(tool_catalog="search(query)")
        assert "TOOLS" in prompt
        assert "search(query)" in prompt

    def test_step_section_present(self) -> None:
        prompt = build_prompt(step_number=3, max_steps=10)
        assert "STEP 3/10" in prompt

    def test_all_sections_in_order(self) -> None:
        prompt = build_prompt(
            task={"id": "1"},
            tool_catalog="tool_a",
            state_summary={"items": [{"description": "found x"}]},
            session_log="SESSION LOG\n- step 1",
            briefing="be careful",
            context_history="result data",
            step_number=2,
            max_steps=5,
        )
        task_pos = prompt.index("TASK")
        tools_pos = prompt.index("TOOLS")
        facts_pos = prompt.index("FACTS")
        step_pos = prompt.index("STEP")
        assert task_pos < tools_pos < facts_pos < step_pos

    def test_returns_string(self) -> None:
        assert isinstance(build_prompt(), str)

    def test_task_fields_in_prompt(self) -> None:
        prompt = build_prompt(task={"id": "abc", "description": "Test task"})
        assert "id: abc" in prompt
        assert "description: Test task" in prompt


class TestCustomSectionHeaders:
    def test_custom_task_header(self) -> None:
        prompt = build_prompt(
            task={"id": "1"},
            section_headers={"task": "OBJECTIVE"},
        )
        assert "OBJECTIVE" in prompt
        assert "TASK" not in prompt

    def test_custom_tools_header(self) -> None:
        prompt = build_prompt(
            tool_catalog="my_tool",
            section_headers={"tools": "CAPABILITIES"},
        )
        assert "CAPABILITIES" in prompt

    def test_custom_facts_header(self) -> None:
        prompt = build_prompt(
            state_summary={"items": [{"description": "x"}]},
            section_headers={"facts": "KNOWN STATE"},
        )
        assert "KNOWN STATE" in prompt

    def test_custom_session_header(self) -> None:
        prompt = build_prompt(
            session_log="log content",
            section_headers={"session": "HISTORY"},
        )
        assert "HISTORY" in prompt

    def test_custom_assessment_header(self) -> None:
        prompt = build_prompt(
            briefing="my briefing",
            section_headers={"assessment": "GUIDANCE"},
        )
        assert "GUIDANCE" in prompt

    def test_custom_results_header(self) -> None:
        prompt = build_prompt(
            context_history="some results",
            section_headers={"results": "LATEST DATA"},
        )
        assert "LATEST DATA" in prompt

    def test_custom_step_header(self) -> None:
        prompt = build_prompt(
            step_number=1,
            max_steps=5,
            section_headers={"step": "TURN"},
        )
        assert "TURN 1/5" in prompt

    def test_partial_override_keeps_defaults(self) -> None:
        prompt = build_prompt(
            task={"id": "1"},
            tool_catalog="t",
            section_headers={"task": "GOAL"},
        )
        assert "GOAL" in prompt
        assert "TOOLS" in prompt  # default still in place


class TestFactsRendering:
    def test_dict_task_renders_facts(self) -> None:
        state = {"items": [{"description": "fact A"}, {"description": "fact B"}]}
        prompt = build_prompt(state_summary=state)
        assert "fact A" in prompt
        assert "fact B" in prompt

    def test_empty_state_no_facts_section(self) -> None:
        prompt = build_prompt(state_summary={})
        assert "ESTABLISHED FACTS" not in prompt

    def test_render_facts_callable_override(self) -> None:
        def my_renderer(state: dict) -> list[str]:
            return ["custom fact line"]

        prompt = build_prompt(
            state_summary={"x": 1},
            render_facts=my_renderer,
        )
        assert "custom fact line" in prompt

    def test_render_facts_callable_empty_list_hides_section(self) -> None:
        def empty_renderer(state: dict) -> list[str]:
            return []

        prompt = build_prompt(
            state_summary={"x": 1},
            render_facts=empty_renderer,
        )
        assert "ESTABLISHED FACTS" not in prompt

    def test_scalar_count_fields_rendered(self) -> None:
        state = {"item_count": 5}
        prompt = build_prompt(state_summary=state)
        assert "item_count: 5" in prompt


class TestLowBudgetWarning:
    def test_low_budget_warning_shown(self) -> None:
        state = {"budget_remaining": 2}
        prompt = build_prompt(state_summary=state, step_number=8, max_steps=10)
        assert "LOW BUDGET" in prompt or "consolidate" in prompt.lower() or "⚠" in prompt

    def test_no_warning_at_full_budget(self) -> None:
        state = {"budget_remaining": 10}
        prompt = build_prompt(state_summary=state)
        assert "LOW BUDGET" not in prompt

    def test_custom_low_budget_warning(self) -> None:
        state = {"budget_remaining": 1}
        prompt = build_prompt(
            state_summary=state,
            low_budget_warning="WRAP IT UP",
        )
        assert "WRAP IT UP" in prompt

    def test_budget_shown_in_step_line(self) -> None:
        state = {"budget_remaining": 7}
        prompt = build_prompt(state_summary=state, step_number=3, max_steps=10)
        assert "7" in prompt


class TestEmptySessionLog:
    def test_empty_session_log_no_extra_section(self) -> None:
        prompt = build_prompt(session_log="")
        # Should not have a dangling session header
        assert prompt.count("SESSION LOG") == 0

    def test_session_log_rendered_when_provided(self) -> None:
        prompt = build_prompt(session_log="SESSION LOG\nstep 1: done")
        assert "step 1: done" in prompt


class TestTaskFields:
    def test_task_fields_filter(self) -> None:
        task = {"id": "1", "description": "my task", "internal": "skip"}
        prompt = build_prompt(task=task, task_fields=["id", "description"])
        assert "id: 1" in prompt
        assert "description: my task" in prompt
        assert "internal" not in prompt

    def test_empty_task_fields_shows_all(self) -> None:
        task = {"id": "1", "name": "test"}
        prompt = build_prompt(task=task, task_fields=None)
        assert "id: 1" in prompt
        assert "name: test" in prompt


class TestNoBackwardCompatParams:
    def test_no_alert_param_in_signature(self) -> None:
        sig = inspect.signature(build_prompt)
        assert "alert" not in sig.parameters

    def test_no_investigation_log_param_in_signature(self) -> None:
        sig = inspect.signature(build_prompt)
        assert "investigation_log" not in sig.parameters

    def test_task_param_exists(self) -> None:
        sig = inspect.signature(build_prompt)
        assert "task" in sig.parameters

    def test_session_log_param_exists(self) -> None:
        sig = inspect.signature(build_prompt)
        assert "session_log" in sig.parameters
