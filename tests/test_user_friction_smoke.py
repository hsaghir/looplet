"""Smoke tests for user-facing convenience helpers.

Covers the small UX wins:
  - ``composable_loop(..., max_steps=N, system_prompt=...)``
  - ``OpenAIBackend.from_env`` / ``AnthropicBackend.from_env``
  - ``OpenAIBackend(api_key=...)`` no longer requires ``base_url``
  - ``BaseToolRegistry.tool`` decorator
"""

from __future__ import annotations

import os
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
    """Minimal stand-in for openai.OpenAI — captures init kwargs."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs


class _FakeAnthropicClient:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs


class TestComposableLoopShortcuts:
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


class TestOpenAIBackendFromEnv:
    def test_api_key_only_no_longer_requires_base_url(self) -> None:
        # Previously raised TypeError; should now construct successfully
        # by handing api_key to the OpenAI client.
        with mock.patch("openai.OpenAI", _FakeOpenAIClient):
            llm = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
            assert _FakeOpenAIClient.last_kwargs == {"api_key": "sk-test"}
            assert llm._model == "gpt-4o-mini"

    def test_no_args_delegates_env_to_openai_client(self) -> None:
        # When neither base_url nor api_key are passed, OpenAIBackend should
        # construct the client with no kwargs and let it read OPENAI_API_KEY itself.
        with mock.patch("openai.OpenAI", _FakeOpenAIClient):
            llm = OpenAIBackend(model="gpt-4o")
            assert _FakeOpenAIClient.last_kwargs == {}
            assert llm._model == "gpt-4o"

    def test_from_env_reads_all_three_vars(self) -> None:
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
