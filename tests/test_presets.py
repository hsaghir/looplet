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


class TestLoopControl:
    @pytest.mark.parametrize("async_run", [False, True])
    async def test_raw_loop_yields_caller_state_and_stops_at_caller_boundary(self, async_run):
        from looplet import (
            AsyncMockLLMBackend,
            DefaultState,
            LoopConfig,
            MockLLMBackend,
            RunResult,
            async_composable_loop,
            composable_loop,
            tools_from,
        )

        greeted = []
        ended = []

        def greet(name: str) -> dict:
            greeted.append(name)
            return {"greeting": f"Hello, {name}!"}

        class Observer:
            def on_loop_end(self, *args):
                ended.append("ended")

        state = DefaultState(max_steps=3)
        loop = async_composable_loop if async_run else composable_loop
        backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
            [
                '{"tool":"greet","args":{"name":"Ada"}}',
                '{"tool":"greet","args":{"name":"Bob"}}',
                '{"tool":"done","args":{"summary":"finished"}}',
            ]
        )
        iterator = loop(
            llm=backend,
            tools=tools_from([greet], include_done=True),
            state=state,
            config=LoopConfig(max_steps=3, use_native_tools=False),
            hooks=[Observer()],
            task={"description": "Greet Ada and Bob"},
        )
        try:
            step = await iterator.__anext__() if async_run else next(iterator)
            assert state.steps == [step]
            assert greeted == ["Ada"]
            assert ended == []
        finally:
            if async_run:
                await iterator.aclose()
            else:
                iterator.close()
        assert greeted == ["Ada"]
        assert ended == ["ended"]
        result = RunResult.from_state(state)
        assert result.status == "stopped"
        assert result.termination_reason == "stream_closed"
        assert not result.completed


class TestAgentPreset:
    @pytest.mark.parametrize("async_run", [False, True])
    async def test_host_resources_bind_to_python_registry(self, async_run):
        from looplet import (
            AgentPreset,
            AsyncMockLLMBackend,
            DefaultState,
            LoopConfig,
            MockLLMBackend,
            ToolContext,
            tool,
            tools_from,
        )

        @tool
        def lookup(ctx: ToolContext):
            return ctx.resources["settings"]

        lookup.requires = ["settings"]
        settings = {"value": 7}
        with AgentPreset(
            tools=tools_from([lookup], include_done=True),
            hooks=[],
            state=DefaultState(max_steps=2),
            resources={"settings": settings},
            config=LoopConfig(max_steps=2, use_native_tools=False),
        ) as preset:
            assert "settings" not in preset.tools.resources
            backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
                [
                    '{"tool":"lookup","args":{}}',
                    '{"tool":"done","args":{"summary":"checked"}}',
                ],
                cycle=False,
            )
            result = (
                await preset.run_result_async(backend) if async_run else preset.run_result(backend)
            )
            assert result.completed, result.diagnostics()
            assert result.steps[0].tool_result.data == settings
            assert preset.tools.resources["settings"] is settings

    @pytest.mark.parametrize("async_run", [False, True])
    async def test_invalid_result_store_is_rejected_before_inference(self, async_run):
        from looplet import AgentPreset, DefaultState, LoopConfig, tools_from
        from looplet.builtin_tools.tool_results import SPECS

        class Backend:
            def generate(self, prompt, **kwargs):
                raise AssertionError("Invalid resources must fail before inference")

        with AgentPreset(
            tools=tools_from(list(SPECS), include_done=True),
            hooks=[],
            state=DefaultState(max_steps=1),
            resources={"tool_result_store": object()},
            config=LoopConfig(max_steps=1),
        ) as preset:
            with pytest.raises(ValueError, match="tool_result_store must provide callable"):
                if async_run:
                    preset.run_async(Backend())
                else:
                    preset.run(Backend())

    @pytest.mark.parametrize("async_run", [False, True])
    async def test_early_stream_close_finalizes_loop_immediately(self, async_run):
        from looplet import AsyncMockLLMBackend, MockLLMBackend, minimal_preset

        ended = []

        class Observer:
            def on_loop_end(self, *args, **kwargs):
                ended.append("ended")

        with minimal_preset(max_steps=2) as preset:
            preset.config.use_native_tools = False
            preset.hooks.append(Observer())
            backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
                ['{"tool":"missing","args":{}}']
            )
            if async_run:
                iterator = preset.run_async(backend)
                await iterator.__anext__()
                await iterator.aclose()
            else:
                iterator = preset.run(backend)
                next(iterator)
                iterator.close()
            assert ended == ["ended"]
            assert preset.state.run_status == "stopped"
            assert preset.state.termination_reason == "stream_closed"

    @pytest.mark.parametrize("async_run", [False, True])
    async def test_failed_cleanup_preserves_primary_error_and_closes_other_hooks(self, async_run):
        from looplet import AsyncMockLLMBackend, MockLLMBackend, minimal_preset

        ended = []

        class BrokenHook:
            def pre_loop(self, *args):
                raise ValueError("primary error")

            def on_loop_end(self, *args):
                ended.append("broken")
                raise RuntimeError("cleanup error")

        class OtherHook:
            def on_loop_end(self, *args):
                ended.append("other")

        with minimal_preset(max_steps=1) as preset:
            preset.hooks.extend([BrokenHook(), OtherHook()])
            with pytest.raises(ValueError, match="primary error"):
                if async_run:
                    await preset.run_result_async(AsyncMockLLMBackend())
                else:
                    preset.run_result(MockLLMBackend())
            assert ended == ["broken", "other"]

    @pytest.mark.parametrize("async_run", [False, True])
    @pytest.mark.parametrize("terminal", [False, True])
    async def test_result_helpers_preserve_execution_and_caller_ownership(
        self, async_run, terminal
    ):
        from looplet import (
            AgentPreset,
            DefaultState,
            LoopConfig,
            MockLLMBackend,
            RunResult,
            tools_from,
        )
        from looplet.testing import AsyncMockLLMBackend

        closed = []

        class Resource:
            def close(self):
                closed.append(self)

        def submit(answer: str) -> dict:
            return {"answer": answer}

        resource = Resource()
        preset = AgentPreset(
            config=LoopConfig(max_steps=1, use_native_tools=False, done_tools=["submit"]),
            tools=tools_from([submit], include_done=True),
            state=DefaultState(max_steps=1),
            hooks=[],
            owned_resources=[resource],
        )
        if not terminal:
            preset.config.done_tools = []
        backend = (AsyncMockLLMBackend if async_run else MockLLMBackend)(
            ['{"tool":"submit","args":{"answer":"accepted"}}']
        )
        with preset:
            if async_run:
                result = await preset.run_result_async(backend, task={"description": "Finish"})
            else:
                result = preset.run_result(backend, task={"description": "Finish"})
            assert result == RunResult.from_state(preset.state)
            assert result.completed is terminal
            assert result.output == ({"answer": "accepted"} if terminal else None)
            assert len(result.steps) == 1
            assert closed == []
            with pytest.raises(RuntimeError, match="single-use"):
                preset.run_result(backend)
        assert closed == [resource]

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

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_coding_preset_rejects_done_after_a_breaking_write(self, tmp_path, asynchronous):
        import json
        import shlex
        import subprocess
        import sys

        from looplet import AsyncMockLLMBackend, MockLLMBackend, RunStatus
        from looplet.presets import coding_agent_preset

        source = tmp_path / "calculator.py"
        source.write_text("def add(left, right):\n    return left + right\n")
        (tmp_path / "test_calculator.py").write_text(
            "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
        )
        broken = "def add(left, right):\n    return left - right + 0\n"
        responses = [
            json.dumps(
                {
                    "tool": "bash",
                    "args": {
                        "command": f"{shlex.quote(sys.executable)} -m pytest -q test_calculator.py"
                    },
                }
            ),
            json.dumps(
                {"tool": "write", "args": {"file_path": "calculator.py", "content": broken}}
            ),
            '{"tool":"done","args":{"summary":"finished"}}',
        ]
        backend = (AsyncMockLLMBackend if asynchronous else MockLLMBackend)(responses, cycle=False)
        with coding_agent_preset(workspace=str(tmp_path), max_steps=len(responses)) as preset:
            preset.config.use_native_tools = False
            result = (
                await preset.run_result_async(backend, task="Update the calculator.")
                if asynchronous
                else preset.run_result(backend, task="Update the calculator.")
            )
        assert backend.calls == len(responses)
        assert result.steps[0].tool_result.data["exit_code"] == 0
        assert result.status is RunStatus.STOPPED
        assert result.output is None
        assert source.read_text() == broken
        grade = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_calculator.py"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert grade.returncode == 1, grade.stdout + grade.stderr

    @pytest.mark.parametrize("mutation", ["write", "edit", "bash"])
    @pytest.mark.parametrize("failed", [False, True])
    def test_coding_guardrail_invalidates_test_evidence(self, mutation, failed):
        from looplet.presets import _CodingGuardrailHook
        from looplet.types import ErrorKind, ToolCall, ToolError, ToolResult

        hook = _CodingGuardrailHook()
        hook.post_dispatch(
            None,
            None,
            ToolCall(tool="bash", args={"command": "python -m pytest -q"}),
            ToolResult(tool="bash", args_summary="", data={"exit_code": 0}),
            1,
        )
        assert hook.check_done(None, None, None, 1) is None
        hook.post_dispatch(
            None,
            None,
            ToolCall(
                tool=mutation,
                args={"command": "python update_source.py"}
                if mutation == "bash"
                else {"file_path": "module.py"},
            ),
            ToolResult(
                tool=mutation,
                args_summary="",
                data={"exit_code": 1 if failed else 0},
                error=ToolError(kind=ErrorKind.EXECUTION, message="mutation failed")
                if failed
                else None,
            ),
            2,
        )
        blocked = hook.check_done(None, None, None, 2)
        assert blocked is not None
        assert blocked.block
        hook.post_dispatch(
            None,
            None,
            ToolCall(tool="bash", args={"command": "python -m pytest -q"}),
            ToolResult(tool="bash", args_summary="", data={"exit_code": 0}),
            3,
        )
        assert hook.check_done(None, None, None, 3) is None

    @pytest.mark.parametrize("tool_name", ["read", "glob", "grep", "think"])
    def test_coding_guardrail_keeps_evidence_for_read_only_tools(self, tool_name):
        from looplet.presets import _CodingGuardrailHook
        from looplet.types import ToolCall, ToolResult

        hook = _CodingGuardrailHook()
        hook.post_dispatch(
            None,
            None,
            ToolCall(tool="bash", args={"command": "python -m pytest -q"}),
            ToolResult(tool="bash", args_summary="", data={"exit_code": 0}),
            1,
        )
        hook.post_dispatch(
            None,
            None,
            ToolCall(tool=tool_name, args={}),
            ToolResult(tool=tool_name, args_summary="", data={}),
            2,
        )
        assert hook.check_done(None, None, None, 2) is None

    @pytest.mark.parametrize("failure", ["nonzero", "error", "timeout"])
    def test_coding_guardrail_rejects_unsuccessful_test_results(self, failure):
        from looplet.presets import _CodingGuardrailHook
        from looplet.types import ErrorKind, ToolCall, ToolError, ToolResult

        hook = _CodingGuardrailHook()
        error = (
            ToolError(
                kind=ErrorKind.TIMEOUT if failure == "timeout" else ErrorKind.EXECUTION,
                message="test execution failed",
            )
            if failure != "nonzero"
            else None
        )
        hook.post_dispatch(
            None,
            None,
            ToolCall(tool="bash", args={"command": "python -m pytest -q"}),
            ToolResult(
                tool="bash",
                args_summary="",
                data={"exit_code": 1 if error is None else 0},
                error=error,
            ),
            1,
        )
        blocked = hook.check_done(None, None, None, 1)
        assert blocked is not None
        assert blocked.block

    def test_coding_guardrail_starts_each_run_without_test_evidence(self):
        from looplet.presets import _CodingGuardrailHook
        from looplet.types import ToolCall, ToolResult

        hook = _CodingGuardrailHook()
        hook.post_dispatch(
            None,
            None,
            ToolCall(tool="bash", args={"command": "python -m pytest -q"}),
            ToolResult(tool="bash", args_summary="", data={"exit_code": 0}),
            1,
        )
        assert hook.check_done(None, None, None, 1) is None
        hook.bind(None)
        blocked = hook.check_done(None, None, None, 1)
        assert blocked is not None
        assert blocked.block

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
