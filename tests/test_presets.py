"""Tests for looplet.presets — high-level agent presets."""

import pytest

pytestmark = pytest.mark.smoke


# ── Import tests ─────────────────────────────────────────────────


class TestPresetsImports:
    def test_import_module(self):
        import looplet.presets  # noqa: F401

    def test_import_coding_agent_preset(self):
        from looplet.presets import coding_agent_preset  # noqa: F401

    def test_import_research_agent_preset(self):
        from looplet.presets import research_agent_preset  # noqa: F401

    def test_import_minimal_preset(self):
        from looplet.presets import minimal_preset  # noqa: F401

    def test_import_agent_preset(self):
        from looplet.presets import AgentPreset  # noqa: F401

    def test_importable_from_top_level(self):
        from looplet import (  # noqa: F401
            AgentPreset,
            coding_agent_preset,
            minimal_preset,
            research_agent_preset,
        )


# ── AgentPreset container ────────────────────────────────────────


class TestAgentPreset:
    def test_coding_preset_returns_agent_preset(self, tmp_path):
        from looplet.presets import AgentPreset, coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path))
        assert isinstance(preset, AgentPreset)

    def test_coding_preset_has_tools(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path))
        names = preset.tools.tool_names
        assert "bash" in names
        assert "read" in names
        assert "write" in names
        assert "edit" in names
        assert "glob" in names
        assert "grep" in names
        assert "think" in names
        assert "done" in names

    def test_coding_preset_has_hooks(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path))
        assert len(preset.hooks) >= 1  # guardrail + budget hook

    def test_coding_preset_has_config(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path))
        assert preset.config.max_steps == 20
        assert preset.config.system_prompt != ""
        assert preset.config.compact_service is not None

    def test_coding_preset_has_state(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path))
        assert preset.state.max_steps == 20
        assert preset.state.step_count == 0

    def test_coding_preset_custom_max_steps(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path), max_steps=50)
        assert preset.config.max_steps == 50
        assert preset.state.max_steps == 50

    def test_coding_preset_custom_system_prompt(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(
            workspace=str(tmp_path),
            system_prompt="You are a Go developer.",
        )
        assert "Go developer" in preset.config.system_prompt

    def test_coding_preset_no_tests_requirement(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset_with = coding_agent_preset(workspace=str(tmp_path), require_tests=True)
        preset_without = coding_agent_preset(workspace=str(tmp_path), require_tests=False)
        # With tests: guardrail hook + budget hook = 2
        # Without tests: only budget hook = 1
        assert len(preset_with.hooks) > len(preset_without.hooks)

    def test_coding_preset_memory_sources(self, tmp_path):
        from looplet.presets import coding_agent_preset
        preset = coding_agent_preset(workspace=str(tmp_path))
        assert len(preset.config.memory_sources) >= 1


# ── Research preset ──────────────────────────────────────────────


class TestResearchPreset:
    def test_returns_agent_preset(self, tmp_path):
        from looplet.presets import AgentPreset, research_agent_preset
        preset = research_agent_preset(workspace=str(tmp_path))
        assert isinstance(preset, AgentPreset)

    def test_has_larger_budget(self, tmp_path):
        from looplet.presets import research_agent_preset
        preset = research_agent_preset(workspace=str(tmp_path))
        assert preset.config.max_steps == 30

    def test_has_tools(self, tmp_path):
        from looplet.presets import research_agent_preset
        preset = research_agent_preset(workspace=str(tmp_path))
        names = preset.tools.tool_names
        assert "read" in names
        assert "grep" in names
        assert "glob" in names


# ── Minimal preset ───────────────────────────────────────────────


class TestMinimalPreset:
    def test_returns_agent_preset(self):
        from looplet.presets import AgentPreset, minimal_preset
        preset = minimal_preset()
        assert isinstance(preset, AgentPreset)

    def test_has_done_tool(self):
        from looplet.presets import minimal_preset
        preset = minimal_preset()
        assert "done" in preset.tools.tool_names

    def test_custom_tools(self):
        from looplet.presets import minimal_preset
        from looplet.tools import ToolSpec
        preset = minimal_preset(tools=[
            ToolSpec(name="search", description="Search",
                     parameters={"q": "str"},
                     execute=lambda *, q: {"results": []}),
        ])
        assert "search" in preset.tools.tool_names
        assert "done" in preset.tools.tool_names  # auto-added

    def test_custom_max_steps(self):
        from looplet.presets import minimal_preset
        preset = minimal_preset(max_steps=5)
        assert preset.config.max_steps == 5
        assert preset.state.max_steps == 5

    def test_no_hooks(self):
        from looplet.presets import minimal_preset
        preset = minimal_preset()
        assert preset.hooks == []


# ── Integration: preset works with composable_loop ───────────────


class TestPresetIntegration:
    def test_coding_preset_runs_loop(self, tmp_path):
        """Verify a preset can drive composable_loop with a mock LLM."""
        from looplet import composable_loop
        from looplet.presets import coding_agent_preset
        from looplet.testing import MockLLMBackend

        llm = MockLLMBackend(responses=[
            '{"tool": "bash", "args": {"command": "echo hello"}, "reasoning": "test"}',
            '{"tool": "done", "args": {"summary": "done"}, "reasoning": "finished"}',
        ])
        preset = coding_agent_preset(workspace=str(tmp_path), require_tests=False)
        steps = list(composable_loop(
            llm=llm, tools=preset.tools, state=preset.state,
            config=preset.config, hooks=preset.hooks,
            task={"description": "echo hello"},
        ))
        assert len(steps) >= 1
        assert steps[0].tool_call.tool == "bash"

    def test_minimal_preset_runs_loop(self):
        """Verify minimal preset works with composable_loop."""
        from looplet import composable_loop
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        llm = MockLLMBackend(responses=[
            '{"tool": "done", "args": {"summary": "all good"}, "reasoning": "done"}',
        ])
        preset = minimal_preset()
        steps = list(composable_loop(
            llm=llm, tools=preset.tools, state=preset.state,
            config=preset.config, hooks=preset.hooks,
            task={"goal": "finish"},
        ))
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "done"
