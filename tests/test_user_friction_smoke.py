"""Smoke tests for user-facing convenience helpers.

Covers the small UX wins:
  - ``composable_loop(..., max_steps=N, system_prompt=...)``
  - ``OpenAIBackend.from_env`` / ``AnthropicBackend.from_env``
  - ``OpenAIBackend(api_key=...)`` no longer requires ``base_url``
  - ``BaseToolRegistry.tool`` decorator
"""

from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any
from unittest import mock

import pytest

from looplet import (
    AnthropicBackend,
    BaseToolRegistry,
    OpenAIBackend,
    composable_loop,
)
from looplet.testing import MockLLMBackend


class _FakeOpenAIClient:
    """Minimal stand-in for openai.OpenAI - captures init kwargs."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs


class _FakeAnthropicClient:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs


class TestComposableLoopShortcuts:
    @pytest.mark.parametrize("state_limit", [None, 1, 9])
    @pytest.mark.parametrize("override", [None, 2])
    def test_config_is_the_single_default_state_limit(self, state_limit, override):
        from looplet import DefaultState, LoopConfig, ToolSpec

        registry = BaseToolRegistry()
        registry.register(ToolSpec("ping", "Ping", {}, lambda: {}))
        state = DefaultState(max_steps=state_limit) if state_limit is not None else None
        config = LoopConfig(max_steps=3, use_native_tools=False)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            steps = list(
                composable_loop(
                    llm=MockLLMBackend(['{"tool": "ping", "args": {}}']),
                    tools=registry,
                    config=config,
                    state=state,
                    max_steps=override,
                )
            )

        expected = override if override is not None else 3
        assert len(steps) == expected
        assert config.max_steps == expected
        if state is not None:
            assert state.max_steps == expected
            assert state.budget_remaining == 0
        assert bool([warning for warning in recorded if "max_steps" in str(warning.message)]) is (
            state_limit is not None
        )

    def test_max_steps_kwarg_seeds_default_config(self) -> None:
        registry = BaseToolRegistry()
        gen = composable_loop(
            llm=MockLLMBackend(["done"]),
            tools=registry,
            max_steps=7,
        )
        # Generator has not started; just confirm we got one back without error.
        assert gen is not None
        gen.close()

    def test_max_steps_overrides_config(self) -> None:
        from looplet import LoopConfig
        from looplet.tools import register_done_tool

        registry = BaseToolRegistry()
        register_done_tool(registry)
        cfg = LoopConfig(max_steps=3)
        gen = composable_loop(
            llm=MockLLMBackend(['{"tool": "done", "args": {}}']),
            tools=registry,
            config=cfg,
            max_steps=11,
        )
        # Drive one step so the generator body runs and the override fires.
        try:
            next(gen)
        except StopIteration:
            pass
        gen.close()
        assert cfg.max_steps == 11

    def test_system_prompt_kwarg_overrides_config(self) -> None:
        from looplet import LoopConfig
        from looplet.tools import register_done_tool

        registry = BaseToolRegistry()
        register_done_tool(registry)
        cfg = LoopConfig(system_prompt="old")
        gen = composable_loop(
            llm=MockLLMBackend(['{"tool": "done", "args": {}}']),
            tools=registry,
            config=cfg,
            system_prompt="new",
        )
        try:
            next(gen)
        except StopIteration:
            pass
        gen.close()
        assert cfg.system_prompt == "new"


class TestPlainPythonInputs:
    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("container", [list, tuple])
    @pytest.mark.parametrize("native", [False, True])
    def test_functions_validate_arguments_and_supply_completion(
        self, asynchronous, container, native
    ):
        from looplet import LoopConfig, async_composable_loop
        from looplet.testing import AsyncMockLLMBackend

        calls = []

        def lookup_owner(service: str) -> str:
            """Find the team that owns a service."""
            calls.append(service)
            return "fintech-platform"

        responses = [
            '{"tool":"lookup_owner","args":{"service":7}}',
            '{"tool":"lookup_owner","args":{"service":"payments"}}',
            '{"tool":"done","args":{"summary":"fintech-platform"}}',
        ]
        config = LoopConfig(harness_profile="minimal", use_native_tools=native)
        if asynchronous:

            async def collect():
                return [
                    step
                    async for step in async_composable_loop(
                        AsyncMockLLMBackend(responses, cycle=False),
                        task="Find the payments owner",
                        tools=container([lookup_owner]),
                        config=config,
                    )
                ]

            steps = asyncio.run(collect())
        else:
            steps = list(
                composable_loop(
                    MockLLMBackend(responses, cycle=False),
                    task="Find the payments owner",
                    tools=container([lookup_owner]),
                    config=config,
                )
            )
        assert calls == ["payments"]
        assert steps[0].tool_result.error is not None
        assert steps[1].tool_result.data == "fintech-platform"
        assert steps[2].tool_result.tool == "done"
        assert steps[2].tool_result.error is None

    def test_explicit_registry_and_terminal_tools_remain_authoritative(self):
        from looplet import LoopConfig
        from looplet.loop import _normalize_loop_inputs
        from looplet.tools import tools_from

        def finish(answer: str) -> str:
            return answer

        config = LoopConfig(done_tools=["finish"])
        task = {"goal": "Answer"}
        registry = tools_from([finish])
        normalized_task, normalized_registry = _normalize_loop_inputs(task, registry, config)
        assert normalized_task is task
        assert normalized_registry is registry
        assert registry.tool_names == ["finish"]
        _, inferred_registry = _normalize_loop_inputs("Answer", [finish], config)
        assert inferred_registry.tool_names == ["finish"]

    def test_sequences_get_fresh_registries(self):
        from looplet import LoopConfig
        from looplet.loop import _normalize_loop_inputs

        functions = []
        task, first = _normalize_loop_inputs("Answer", functions, LoopConfig())
        _, second = _normalize_loop_inputs("Answer", functions, LoopConfig())
        assert task == {"description": "Answer"}
        assert first is not second
        assert first.tool_names == second.tool_names == ["done"]
        assert functions == []

    def test_function_sequences_allow_overlapping_async_runs(self, tmp_path, monkeypatch):
        from looplet import run_result_async
        from looplet.testing import AsyncMockLLMBackend

        monkeypatch.chdir(tmp_path)

        async def collect():
            ready = asyncio.Event()
            entered = 0

            class OverlappingBackend(AsyncMockLLMBackend):
                async def generate_with_tools(self, *args, **kwargs):
                    nonlocal entered
                    entered += 1
                    if entered == 2:
                        ready.set()
                    await ready.wait()
                    return await super().generate_with_tools(*args, **kwargs)

            functions = []
            return await asyncio.gather(
                run_result_async(
                    OverlappingBackend(['{"tool":"done","args":{"summary":"first"}}'], cycle=False),
                    task="First task",
                    tools=functions,
                ),
                run_result_async(
                    OverlappingBackend(
                        ['{"tool":"done","args":{"summary":"second"}}'], cycle=False
                    ),
                    task="Second task",
                    tools=functions,
                ),
            )

        first, second = asyncio.run(collect())
        assert first.completed and second.completed
        assert first.output["summary"] == "first"
        assert second.output["summary"] == "second"
        assert first.metadata["run_dir"] != second.metadata["run_dir"]

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("invalid_tools", ["lookup_owner", [None], [7]])
    def test_invalid_sequences_fail_before_persistence(self, asynchronous, invalid_tools, tmp_path):
        from looplet import LoopConfig, async_composable_loop
        from looplet.testing import AsyncMockLLMBackend

        config = LoopConfig(run_dir=str(tmp_path / "must-not-exist"))
        if asynchronous:
            backend = AsyncMockLLMBackend()

            async def collect():
                return [
                    step
                    async for step in async_composable_loop(
                        backend, tools=invalid_tools, config=config
                    )
                ]

            with pytest.raises(TypeError, match="tools must"):
                asyncio.run(collect())
        else:
            backend = MockLLMBackend()
            with pytest.raises(TypeError, match="tools must"):
                list(composable_loop(backend, tools=invalid_tools, config=config))
        assert backend.calls == 0
        assert not list(tmp_path.iterdir())


class TestRunResultShortcuts:
    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("supply_state", [False, True])
    def test_collectors_use_the_same_state_and_terminal_contract(self, asynchronous, supply_state):
        from looplet import DefaultState, LoopConfig, RunResult, run_result, run_result_async
        from looplet.testing import AsyncMockLLMBackend

        def finish(answer: str) -> dict:
            return {"answer": answer}

        class Observer:
            def on_event(self, payload):
                events.append(payload)

        events = []
        config = LoopConfig(harness_profile="minimal", use_native_tools=False, done_tool="finish")
        state = DefaultState(max_steps=2) if supply_state else None
        responses = ['{"tool":"finish","args":{"answer":"fintech-platform"}}']
        kwargs = {
            "task": "Find the payments owner",
            "tools": [finish],
            "config": config,
            "state": state,
            "max_steps": 2,
            "system_prompt": "Use the supplied functions.",
            "hooks": [Observer()],
        }
        if asynchronous:
            result = asyncio.run(
                run_result_async(AsyncMockLLMBackend(responses, cycle=False), **kwargs)
            )
        else:
            result = run_result(MockLLMBackend(responses, cycle=False), **kwargs)
        assert result.completed
        assert result.output == {"answer": "fintech-platform"}
        assert len(result.steps) == 1
        assert events
        assert config.max_steps == 2
        assert config.system_prompt == "Use the supplied functions."
        if state is not None:
            assert result == RunResult.from_state(state)

    @pytest.mark.parametrize("asynchronous", [False, True])
    def test_default_native_runs_keep_evidence_and_lazy_history(self, asynchronous, tmp_path):
        import json
        from pathlib import Path

        from looplet import LoopConfig, run_result, run_result_async
        from looplet.testing import AsyncMockLLMBackend

        def lookup_owner(service: str) -> str:
            return "fintech-platform"

        responses = [
            '{"tool":"lookup_owner","args":{"service":"payments"}}',
            '{"tool":"done","args":{"summary":"fintech-platform"}}',
        ]
        config = LoopConfig(run_dir=str(tmp_path / "run"))
        kwargs = {"task": "Find the payments owner", "tools": [lookup_owner], "config": config}
        if asynchronous:
            result = asyncio.run(
                run_result_async(AsyncMockLLMBackend(responses, cycle=False), **kwargs)
            )
        else:
            result = run_result(MockLLMBackend(responses, cycle=False), **kwargs)
        assert result.completed
        assert result.output == {"status": "completed", "summary": "fintech-platform"}
        assert len(result.steps) == 2
        assert result.steps[0].tool_result.data == "fintech-platform"
        trajectory = json.loads(
            (Path(result.metadata["trace_dir"]) / "trajectory.json").read_text()
        )
        assert "harness_snapshot" in trajectory["metadata"]
        assert len(trajectory["llm_calls"]) == 2
        assert (tmp_path / "run" / "session.jsonl").is_file()

    @pytest.mark.parametrize("asynchronous", [False, True])
    def test_noncompletion_remains_visible(self, asynchronous):
        from looplet import LoopConfig, run_result, run_result_async
        from looplet.testing import AsyncMockLLMBackend

        def ping() -> str:
            return "pong"

        responses = ['{"tool":"ping","args":{}}']
        config = LoopConfig(harness_profile="minimal")
        kwargs = {"task": "Keep checking", "tools": [ping], "config": config, "max_steps": 1}
        if asynchronous:
            result = asyncio.run(
                run_result_async(AsyncMockLLMBackend(responses, cycle=False), **kwargs)
            )
        else:
            result = run_result(MockLLMBackend(responses, cycle=False), **kwargs)
        assert not result.completed
        assert result.output is None
        assert result.termination_reason == "budget_exhausted"
        assert len(result.steps) == 1

    @pytest.mark.parametrize("asynchronous", [False, True])
    def test_precancelled_runs_do_not_call_the_backend(self, asynchronous):
        from looplet import CancelToken, LoopConfig, run_result, run_result_async
        from looplet.testing import AsyncMockLLMBackend

        token = CancelToken()
        token.cancel()
        config = LoopConfig(harness_profile="minimal", cancel_token=token)
        if asynchronous:
            backend = AsyncMockLLMBackend()
            result = asyncio.run(run_result_async(backend, tools=[], config=config))
        else:
            backend = MockLLMBackend()
            result = run_result(backend, tools=[], config=config)
        assert result.cancelled
        assert backend.calls == 0
        assert config.cancel_token is token

    @pytest.mark.parametrize("asynchronous", [False, True])
    def test_early_close_releases_an_explicit_registry(self, asynchronous):
        from contextlib import aclosing, closing

        from looplet import DefaultState, LoopConfig, RunResult, async_composable_loop
        from looplet.testing import AsyncMockLLMBackend
        from looplet.tools import tools_from

        def ping() -> str:
            return "pong"

        registry = tools_from([ping], include_done=True, validate=True)
        config = LoopConfig(harness_profile="minimal", max_steps=2)
        state = DefaultState(max_steps=2)
        response = '{"tool":"ping","args":{}}'
        terminal = '{"tool":"done","args":{"summary":"finished"}}'
        if asynchronous:

            async def collect():
                backend = AsyncMockLLMBackend([response, terminal], cycle=False)
                async with aclosing(
                    async_composable_loop(backend, tools=registry, state=state, config=config)
                ) as steps:
                    async for _step in steps:
                        break
                assert backend.calls == 1
                assert RunResult.from_state(state).termination_reason == "stream_closed"
                return [
                    step
                    async for step in async_composable_loop(
                        AsyncMockLLMBackend([terminal], cycle=False), tools=registry, config=config
                    )
                ]

            later = asyncio.run(collect())
        else:
            backend = MockLLMBackend([response, terminal], cycle=False)
            with closing(
                composable_loop(backend, tools=registry, state=state, config=config)
            ) as steps:
                for _step in steps:
                    break
            assert backend.calls == 1
            assert RunResult.from_state(state).termination_reason == "stream_closed"
            later = list(
                composable_loop(
                    MockLLMBackend([terminal], cycle=False), tools=registry, config=config
                )
            )
        assert len(later) == 1
        assert later[0].tool_result.error is None


class TestOpenAIBackendFromEnv:
    def test_api_key_only_no_longer_requires_base_url(self) -> None:
        # Previously raised TypeError; should now construct successfully
        # by handing api_key to the OpenAI client.
        pytest.importorskip("openai")  # optional extra; CI installs it
        with mock.patch("openai.OpenAI", _FakeOpenAIClient):
            llm = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
            assert _FakeOpenAIClient.last_kwargs == {"api_key": "sk-test"}
            assert llm._model == "gpt-4o-mini"

    def test_no_args_delegates_env_to_openai_client(self) -> None:
        # When neither base_url nor api_key are passed, OpenAIBackend should
        # construct the client with no kwargs and let it read OPENAI_API_KEY itself.
        pytest.importorskip("openai")  # optional extra; CI installs it
        with mock.patch("openai.OpenAI", _FakeOpenAIClient):
            llm = OpenAIBackend(model="gpt-4o")
            assert _FakeOpenAIClient.last_kwargs == {}
            assert llm._model == "gpt-4o"

    def test_from_env_reads_all_three_vars(self) -> None:
        pytest.importorskip("openai")  # optional extra; CI installs it
        env = {
            "OPENAI_API_KEY": "sk-env",
            "OPENAI_BASE_URL": "http://proxy/v1",
            "OPENAI_MODEL": "llama3",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch("openai.OpenAI", _FakeOpenAIClient),
        ):
            llm = OpenAIBackend.from_env()
        assert _FakeOpenAIClient.last_kwargs == {
            "api_key": "sk-env",
            "base_url": "http://proxy/v1",
        }
        assert llm._model == "llama3"

    def test_from_env_explicit_model_wins(self) -> None:
        pytest.importorskip("openai")  # optional extra; CI installs it
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "from-env",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch("openai.OpenAI", _FakeOpenAIClient),
        ):
            llm = OpenAIBackend.from_env(model="explicit-wins")
        assert llm._model == "explicit-wins"


class TestAnthropicBackendFromEnv:
    def test_from_env_reads_key_and_model(self) -> None:
        pytest.importorskip("anthropic")  # optional extra; CI installs it
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-env",
            "ANTHROPIC_MODEL": "claude-test-2",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch("anthropic.Anthropic", _FakeAnthropicClient),
        ):
            llm = AnthropicBackend.from_env()
        assert _FakeAnthropicClient.last_kwargs == {"api_key": "sk-ant-env"}
        assert llm._model == "claude-test-2"

    def test_from_env_missing_key_raises(self) -> None:
        # ``from_env`` raises a clean RuntimeError upfront when
        # ``ANTHROPIC_API_KEY`` is missing, instead of letting the SDK
        # surface a less actionable error later.
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                AnthropicBackend.from_env()


class TestRegistryToolDecorator:
    def test_bare_decorator_registers_function(self) -> None:
        registry = BaseToolRegistry()

        @registry.tool
        def echo(msg: str) -> dict:
            """Return the message."""
            return {"echo": msg}

        assert "echo" in registry.tool_names
        assert registry._tools["echo"].description == "Return the message."

    def test_decorator_with_kwargs_passes_through(self) -> None:
        registry = BaseToolRegistry()

        @registry.tool(name="renamed", concurrent_safe=True)
        def original_name(x: int) -> dict:
            return {"x": x}

        assert "renamed" in registry.tool_names
        assert "original_name" not in registry.tool_names
        assert registry._tools["renamed"].concurrent_safe is True

    def test_decorator_returns_tool_spec(self) -> None:
        registry = BaseToolRegistry()

        @registry.tool
        def my_tool() -> dict:
            return {}

        # The decorated symbol is a ToolSpec, same as the module-level @tool.
        from looplet.tools import ToolSpec

        assert isinstance(my_tool, ToolSpec)
        assert my_tool.name == "my_tool"
