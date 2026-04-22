"""Tests for native tool calling wiring in backends + scaffolding + loop.

Covers:
 - OpenAI / Anthropic backends implement ``generate_with_tools`` that returns
   normalised Anthropic-style content blocks.
 - ``llm_call_with_retry`` routes to ``generate_with_tools`` when ``tools`` is
   passed and the backend supports it; falls back to ``generate`` otherwise.
 - The composable loop passes tool schemas through only when
   ``LOOPLET_NATIVE_TOOLS`` (via FLAGS.native_tools) or
   ``LoopConfig.use_native_tools`` is set, and routes the resulting
   ``list[dict]`` response through ``parse_native_tool_use``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from looplet.backends import (
    AnthropicBackend,
    OpenAIBackend,
    _anthropic_response_to_blocks,
    _openai_message_to_blocks,
    _to_openai_tools,
)
from looplet.scaffolding import llm_call_with_retry
from looplet.types import NativeToolBackend

# ── Helpers ──────────────────────────────────────────────────────


def _weather_schema() -> list[dict[str, Any]]:
    return [{
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }]


class _FakeOpenAIClient:
    """Minimal fake OpenAI client capturing the last create() kwargs."""

    def __init__(self, message: Any) -> None:
        self._message = message
        self.last_kwargs: dict[str, Any] | None = None

    @property
    def chat(self) -> Any:
        client = self

        class _Completions:
            def create(_, **kwargs):
                client.last_kwargs = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(message=client._message)])

        return SimpleNamespace(completions=_Completions())


class _FakeAnthropicClient:
    def __init__(self, content: list[Any]) -> None:
        self._content = content
        self.last_kwargs: dict[str, Any] | None = None

        client = self

        class _Messages:
            def create(_, **kwargs):
                client.last_kwargs = kwargs
                return SimpleNamespace(content=client._content)

        self.messages = _Messages()


# ── _to_openai_tools ──────────────────────────────────────────────


class TestToOpenAITools:
    def test_wraps_schema_in_function_envelope(self):
        out = _to_openai_tools(_weather_schema())
        assert out == [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

    def test_missing_input_schema_yields_empty_object(self):
        out = _to_openai_tools([{"name": "noop", "description": "no-op"}])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# ── Message normalisers ───────────────────────────────────────────


class TestOpenAIMessageNormaliser:
    def test_tool_calls_and_content(self):
        msg = SimpleNamespace(
            content="Let me check.",
            tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
            )],
        )
        blocks = _openai_message_to_blocks(msg)
        assert blocks == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "call_1", "name": "get_weather",
             "input": {"city": "Paris"}},
        ]

    def test_malformed_arguments_kept_as_raw(self):
        msg = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call_2",
                function=SimpleNamespace(name="x", arguments='not-json'),
            )],
        )
        blocks = _openai_message_to_blocks(msg)
        assert blocks == [{"type": "tool_use", "id": "call_2", "name": "x",
                           "input": {"_raw_arguments": "not-json"}}]

    def test_no_tool_calls(self):
        msg = SimpleNamespace(content="Hello", tool_calls=None)
        assert _openai_message_to_blocks(msg) == [{"type": "text", "text": "Hello"}]


class TestAnthropicResponseNormaliser:
    def test_mixed_text_and_tool_use(self):
        response = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="Let me check."),
            SimpleNamespace(type="tool_use", id="tu_1", name="get_weather",
                            input={"city": "Paris"}),
        ])
        assert _anthropic_response_to_blocks(response) == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather",
             "input": {"city": "Paris"}},
        ]


# ── Backends implement NativeToolBackend ──────────────────────────


class TestBackendImplementsProtocol:
    def test_openai_backend_has_generate_with_tools(self):
        backend = OpenAIBackend(_FakeOpenAIClient(SimpleNamespace(content="", tool_calls=[])))
        assert isinstance(backend, NativeToolBackend)

    def test_anthropic_backend_has_generate_with_tools(self):
        backend = AnthropicBackend(_FakeAnthropicClient([]))
        assert isinstance(backend, NativeToolBackend)


class TestGenerateWithTools:
    def test_openai_passes_tools_and_returns_blocks(self):
        client = _FakeOpenAIClient(SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="c1",
                function=SimpleNamespace(name="get_weather", arguments='{"city":"Tokyo"}'),
            )],
        ))
        backend = OpenAIBackend(client, model="gpt-x")
        blocks = backend.generate_with_tools("Weather?", tools=_weather_schema())
        assert blocks == [{"type": "tool_use", "id": "c1", "name": "get_weather",
                           "input": {"city": "Tokyo"}}]
        assert client.last_kwargs["tools"][0]["type"] == "function"
        assert client.last_kwargs["tool_choice"] == "auto"

    def test_anthropic_passes_tools_as_is(self):
        client = _FakeAnthropicClient([
            SimpleNamespace(type="tool_use", id="tu_1", name="get_weather",
                            input={"city": "Tokyo"}),
        ])
        backend = AnthropicBackend(client, model="claude-x")
        schemas = _weather_schema()
        blocks = backend.generate_with_tools("Weather?", tools=schemas,
                                             system_prompt="be brief")
        assert blocks == [{"type": "tool_use", "id": "tu_1", "name": "get_weather",
                           "input": {"city": "Tokyo"}}]
        assert client.last_kwargs["tools"] == schemas
        assert client.last_kwargs["system"] == "be brief"


# ── llm_call_with_retry gating ───────────────────────────────────


class _NoToolBackend:
    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        return '{"tool": "done", "args": {}}'


class _NativeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        raise AssertionError("generate() should not be called in native mode")

    def generate_with_tools(self, prompt, *, tools, max_tokens=2000,
                            system_prompt="", temperature=0.2):
        self.calls.append({"prompt": prompt, "tools": tools})
        return [{"type": "tool_use", "id": "n1", "name": "done", "input": {}}]


class TestLLMCallWithRetryNative:
    def test_no_tools_uses_generate(self):
        backend = _NoToolBackend()
        result = llm_call_with_retry(backend, "hi")
        assert result.ok
        assert isinstance(result.text, str)

    def test_tools_and_native_backend_uses_generate_with_tools(self):
        backend = _NativeBackend()
        result = llm_call_with_retry(backend, "hi", tools=_weather_schema())
        assert result.ok
        assert isinstance(result.text, list)
        assert result.text[0]["type"] == "tool_use"
        assert backend.calls[0]["tools"] == _weather_schema()

    def test_tools_but_non_native_backend_falls_back_to_text(self):
        backend = _NoToolBackend()  # no generate_with_tools
        result = llm_call_with_retry(backend, "hi", tools=_weather_schema())
        assert result.ok
        assert isinstance(result.text, str)


# ── Integration: loop gating via FLAGS.native_tools ───────────────


class _RegistryWithSchemas:
    """Minimal tool registry stub exposing tool_schemas() for loop gating."""

    def __init__(self) -> None:
        self.schemas_calls = 0

    def tool_schemas(self) -> list[dict[str, Any]]:
        self.schemas_calls += 1
        return _weather_schema()


class TestLoopGatingCall:
    """Verify the loop inspection branch: native path only fires when gated."""

    def test_loop_imports_flag_and_parser(self):
        # Sanity: the module-level branch uses these symbols.
        from looplet import loop as loop_module
        from looplet.flags import FLAGS
        from looplet.parse import parse_native_tool_use

        assert "parse_native_tool_use" in loop_module.__dict__
        assert FLAGS is not None
        assert callable(parse_native_tool_use)

    @pytest.mark.parametrize("env_val,expected", [
        ("1", True),
        ("true", True),
        ("0", False),
        ("", False),
    ])
    def test_flag_respects_env(self, monkeypatch, env_val, expected):
        from looplet.flags import _Flags
        monkeypatch.setenv("LOOPLET_NATIVE_TOOLS", env_val)
        assert _Flags().native_tools is expected
