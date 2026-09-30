"""Tests for looplet.presets - high-level agent presets."""

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
    @pytest.mark.parametrize("async_run", [False, True])
    @pytest.mark.parametrize(
        "component_slot",
        ["hooks", "mcp_adapters", "state_service_handles", "model_gateway", "extra_hooks"],
    )
    async def test_setup_failure_releases_claim_and_can_be_retried(self, async_run, component_slot):
        from looplet import MockLLMBackend, RunEnvelope, minimal_preset
        from looplet.testing import AsyncMockLLMBackend

        class Binding:
            fail = True

            def set_run_envelope(self, envelope):
                if self.fail:
                    raise RuntimeError("binding failed")
                self.envelope = envelope

            def set_backend(self, backend):
                self.backend = backend

            def pre_loop(self, state, session_log, context):
                pass

            def close(self):
                pass

        preset = minimal_preset(max_steps=1)
        preset.config.use_native_tools = False
        preset.config.run_envelope = RunEnvelope(run_id="binding-run")
        binding = Binding()
        extra_hooks = [binding] if component_slot == "extra_hooks" else []
        if component_slot != "extra_hooks":
            setattr(
                preset, component_slot, binding if component_slot == "model_gateway" else [binding]
            )
        backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
            responses=['{"tool": "done", "args": {"summary": "ok"}}']
        )
        try:
            with pytest.raises(RuntimeError, match="binding failed"):
                if async_run:
                    iterator = preset.run_async(backend, extra_hooks=extra_hooks)
                    await iterator.__anext__()
                else:
                    next(preset.run(backend, extra_hooks=extra_hooks))
            assert not preset.run_claimed

            binding.fail = False
            if async_run:
                steps = [step async for step in preset.run_async(backend, extra_hooks=extra_hooks)]
            else:
                steps = list(preset.run(backend, extra_hooks=extra_hooks))
            assert steps[0].tool_call.tool == "done"
            assert binding.envelope is preset.config.run_envelope
            assert preset.run_claimed
        finally:
            preset.close()

    def test_sync_iterator_preserves_custom_loop_return_trace(self):
        from looplet import MockLLMBackend, minimal_preset

        preset = minimal_preset(max_steps=1)
        preset.config.use_native_tools = False
        trace = {"custom": "trace"}
        preset.config.build_trace = lambda **kwargs: trace
        iterator = preset.run(MockLLMBackend(['{"tool": "done", "args": {"summary": "ok"}}']))
        try:
            assert next(iterator).tool_call.tool == "done"
            with pytest.raises(StopIteration) as stopped:
                next(iterator)
            assert stopped.value.value is trace
        finally:
            preset.close()

    @pytest.mark.parametrize("async_run", [False, True])
    async def test_gateway_binding_failure_is_fail_closed_and_releases_claim(self, async_run):
        from looplet import MockLLMBackend, minimal_preset

        class Gateway:
            def set_backend(self, backend):
                raise RuntimeError("gateway binding failed")

            def close(self):
                pass

        preset = minimal_preset(max_steps=1)
        preset.model_gateway = Gateway()
        try:
            with pytest.raises(RuntimeError, match="gateway binding failed"):
                if async_run:
                    preset.run_async(MockLLMBackend())
                else:
                    preset.run(MockLLMBackend())
            assert not preset.run_claimed
            assert preset.state.step_count == 0
        finally:
            preset.close()

    def test_run_rejects_missing_configured_terminal_tool(self):
        from looplet import BaseToolRegistry, DefaultState, LoopConfig, MockLLMBackend
        from looplet.presets import AgentPreset

        preset = AgentPreset(
            config=LoopConfig(max_steps=1, done_tool="finish"),
            hooks=[],
            tools=BaseToolRegistry(),
            state=DefaultState(max_steps=1),
        )

        with pytest.raises(ValueError, match="terminal"):
            list(preset.run(MockLLMBackend(responses=[]), task={}))

    def test_close_returns_idempotent_shutdown_report(self):
        from looplet.presets import AgentPreset

        class Resource:
            def __init__(self):
                self.calls = 0

            def close(self):
                self.calls += 1

        resource = Resource()
        preset = AgentPreset(
            config=__import__("looplet").LoopConfig(max_steps=1),
            hooks=[],
            tools=__import__("looplet").BaseToolRegistry(),
            state=__import__("looplet").DefaultState(max_steps=1),
            resources={"resource": resource},
            owned_resources=[resource, resource],
        )

        first = preset.close()
        second = preset.close()

        assert resource.calls == 1
        assert first.closed_components == ("resource",)
        assert second.closed_components == ()

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

    def test_coding_preset_tool_schema_preserves_defaults(self, tmp_path):
        from looplet.presets import coding_agent_preset

        preset = coding_agent_preset(workspace=str(tmp_path))
        info = {tool["name"]: tool for tool in preset.tools.introspect()["tools"]}

        assert info["bash"]["parameters"]["required"] == ["command"]
        assert info["read"]["parameters"]["required"] == ["file_path"]
        assert info["write"]["parameters"]["required"] == ["file_path", "content"]
        assert info["grep"]["parameters"]["properties"]["path"]["default"] == "."
        assert info["think"]["free"] is True
        assert "summary" in info["done"]["parameters"]["properties"]

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
        assert "bash" in names
        assert "read" in names
        assert "grep" in names
        assert "glob" in names
        assert "write" not in names
        assert "edit" not in names


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

        preset = minimal_preset(
            tools=[
                ToolSpec(
                    name="search",
                    description="Search",
                    parameters={"q": "str"},
                    execute=lambda *, q: {"results": []},
                ),
            ]
        )
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
    def test_preset_run_is_single_use(self):
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        preset = minimal_preset()
        list(preset.run(MockLLMBackend(['{"tool":"done","args":{"summary":"one"}}']), task={}))

        with pytest.raises(RuntimeError, match="single-use"):
            preset.run(MockLLMBackend(['{"tool":"done","args":{"summary":"two"}}']), task={})

    def test_preset_rejects_concurrent_claims_before_execution(self):
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        preset = minimal_preset()
        first = preset.run(MockLLMBackend(['{"tool":"done","args":{"summary":"one"}}']), task={})

        with pytest.raises(RuntimeError, match="single-use"):
            preset.run(MockLLMBackend(['{"tool":"done","args":{"summary":"two"}}']), task={})

        steps = list(first)
        assert len(steps) == 1

    def test_closing_unstarted_preset_run_releases_claim(self):
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        preset = minimal_preset()
        abandoned = preset.run(MockLLMBackend(['{"tool":"done","args":{}}']), task={})
        abandoned.close()
        assert not preset.run_claimed
        assert list(preset.run(MockLLMBackend(['{"tool":"done","args":{}}']), task={}))

    def test_first_iteration_failure_releases_claim(self):
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        class FailingHook:
            def pre_loop(self, state, session_log, context):
                raise RuntimeError("first iteration failed")

        preset = minimal_preset()
        preset.hooks.append(FailingHook())
        run = preset.run(MockLLMBackend(), task={})
        with pytest.raises(RuntimeError, match="first iteration failed"):
            next(run)
        assert not preset.run_claimed

    @pytest.mark.asyncio
    async def test_first_async_iteration_failure_releases_claim(self):
        from looplet.presets import minimal_preset
        from looplet.testing import AsyncMockLLMBackend

        class FailingHook:
            async def pre_loop(self, state, session_log, context):
                raise RuntimeError("first async iteration failed")

        preset = minimal_preset()
        preset.hooks.append(FailingHook())
        run = preset.run_async(AsyncMockLLMBackend(), task={})
        with pytest.raises(RuntimeError, match="first async iteration failed"):
            await run.__anext__()
        assert not preset.run_claimed

    @pytest.mark.asyncio
    async def test_async_preset_run_is_single_use(self):
        from looplet.async_loop import async_composable_loop
        from looplet.presets import minimal_preset
        from looplet.testing import AsyncMockLLMBackend

        preset = minimal_preset()
        first = preset.run_async(
            AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"one"}}']), task={}
        )

        with pytest.raises(RuntimeError, match="single-use"):
            preset.run_async(
                AsyncMockLLMBackend(['{"tool":"done","args":{"summary":"two"}}']), task={}
            )

        steps = []
        async for step in first:
            steps.append(step)
        assert len(steps) == 1

    def test_coding_preset_runs_loop(self, tmp_path):
        """Verify a preset can drive composable_loop with a mock LLM."""
        from looplet import composable_loop
        from looplet.presets import coding_agent_preset
        from looplet.testing import MockLLMBackend

        llm = MockLLMBackend(
            responses=[
                '{"tool": "bash", "args": {"command": "echo hello"}, "reasoning": "test"}',
                '{"tool": "done", "args": {"summary": "done"}, "reasoning": "finished"}',
            ]
        )
        preset = coding_agent_preset(workspace=str(tmp_path), require_tests=False)
        steps = list(
            composable_loop(
                llm=llm,
                tools=preset.tools,
                state=preset.state,
                config=preset.config,
                hooks=preset.hooks,
                task={"description": "echo hello"},
            )
        )
        assert len(steps) >= 1
        assert steps[0].tool_call.tool == "bash"

    def test_minimal_preset_runs_loop(self):
        """Verify minimal preset works with composable_loop."""
        from looplet import composable_loop
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        llm = MockLLMBackend(
            responses=[
                '{"tool": "done", "args": {"summary": "all good"}, "reasoning": "done"}',
            ]
        )
        preset = minimal_preset()
        steps = list(
            composable_loop(
                llm=llm,
                tools=preset.tools,
                state=preset.state,
                config=preset.config,
                hooks=preset.hooks,
                task={"goal": "finish"},
            )
        )
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "done"

    def test_preset_run_drives_loop(self):
        """``AgentPreset.run`` collapses the 6-arg composable_loop call."""
        from looplet.presets import minimal_preset
        from looplet.testing import MockLLMBackend

        llm = MockLLMBackend(
            responses=[
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "done"}',
            ]
        )
        preset = minimal_preset()
        steps = list(preset.run(llm, task={"goal": "finish"}))
        assert len(steps) == 1
        assert steps[0].tool_call.tool == "done"
