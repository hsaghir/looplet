"""Tests for native tool calling wiring in backends + scaffolding + loop.

Covers:
 - OpenAI / Anthropic backends implement ``generate_with_tools`` that returns
   normalised Anthropic-style content blocks.
 - ``llm_call_with_retry`` routes to ``generate_with_tools`` when ``tools`` is
   passed and the backend supports it; falls back to ``generate`` otherwise.
 - The composable loop passes tool schemas through when
     ``LoopConfig.use_native_tools`` is enabled (the default), transparently
     falls back to regular generation when native calls fail, and routes a
     resulting ``list[dict]`` response through ``parse_native_tool_use``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from looplet import (
    DefaultState,
    LoopConfig,
    async_composable_loop,
    composable_loop,
    tool,
    tools_from,
)
from looplet.backends import (
    AnthropicBackend,
    OpenAIBackend,
    _anthropic_response_to_blocks,
    _openai_message_to_blocks,
    _to_openai_tools,
)
from looplet.native_tools import NativeToolPolicy
from looplet.scaffolding import llm_call_with_retry
from looplet.types import NativeToolBackend

# ── Helpers ──────────────────────────────────────────────────────


def _weather_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


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
        assert out == [
            {
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
            }
        ]

    def test_missing_input_schema_yields_empty_object(self):
        out = _to_openai_tools([{"name": "noop", "description": "no-op"}])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# ── Message normalisers ───────────────────────────────────────────


class TestOpenAIMessageNormaliser:
    def test_tool_calls_and_content(self):
        msg = SimpleNamespace(
            content="Let me check.",
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
                )
            ],
        )
        blocks = _openai_message_to_blocks(msg)
        assert blocks == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
        ]

    def test_malformed_arguments_kept_as_raw(self):
        msg = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_2",
                    function=SimpleNamespace(name="x", arguments="not-json"),
                )
            ],
        )
        blocks = _openai_message_to_blocks(msg)
        assert blocks == [
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "x",
                "input": {"_raw_arguments": "not-json"},
            }
        ]

    def test_no_tool_calls(self):
        msg = SimpleNamespace(content="Hello", tool_calls=None)
        assert _openai_message_to_blocks(msg) == [{"type": "text", "text": "Hello"}]


class TestAnthropicResponseNormaliser:
    def test_mixed_text_and_tool_use(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me check."),
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="get_weather", input={"city": "Paris"}
                ),
            ]
        )
        assert _anthropic_response_to_blocks(response) == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
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
        client = _FakeOpenAIClient(
            SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="c1",
                        function=SimpleNamespace(name="get_weather", arguments='{"city":"Tokyo"}'),
                    )
                ],
            )
        )
        backend = OpenAIBackend(client, model="gpt-x")
        blocks = backend.generate_with_tools("Weather?", tools=_weather_schema())
        assert blocks == [
            {"type": "tool_use", "id": "c1", "name": "get_weather", "input": {"city": "Tokyo"}}
        ]
        assert client.last_kwargs["tools"][0]["type"] == "function"
        assert client.last_kwargs["tool_choice"] == "auto"

    def test_anthropic_passes_tools_as_is(self):
        client = _FakeAnthropicClient(
            [
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="get_weather", input={"city": "Tokyo"}
                ),
            ]
        )
        backend = AnthropicBackend(client, model="claude-x")
        schemas = _weather_schema()
        blocks = backend.generate_with_tools("Weather?", tools=schemas, system_prompt="be brief")
        assert blocks == [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Tokyo"}}
        ]
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

    def generate_with_tools(
        self, prompt, *, tools, max_tokens=2000, system_prompt="", temperature=0.2
    ):
        self.calls.append({"prompt": prompt, "tools": tools})
        return [{"type": "tool_use", "id": "n1", "name": "done", "input": {}}]


class _NativeFailureBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_with_tools(
        self, prompt, *, tools, max_tokens=2000, system_prompt="", temperature=0.2
    ):
        self.calls.append("native")
        raise RuntimeError("tools endpoint unsupported")

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.calls.append("regular")
        return '{"tool": "done", "args": {}}'


class _StickyNativeFailureBackend:
    def __init__(self) -> None:
        self.native_calls = 0
        self.regular_calls = 0

    def generate_with_tools(
        self, prompt, *, tools, max_tokens=2000, system_prompt="", temperature=0.2
    ):
        del prompt, tools, max_tokens, system_prompt, temperature
        self.native_calls += 1
        raise RuntimeError("tools endpoint unsupported")

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        del prompt, max_tokens, system_prompt, temperature
        self.regular_calls += 1
        if self.regular_calls == 1:
            return '{"tool": "echo", "args": {"value": "x"}}'
        return '{"tool": "done", "args": {"answer": "done"}}'


@tool(description="Echo a value.")
def _echo(*, value: str) -> dict[str, str]:
    return {"value": value}


class TestLLMCallWithRetryNative:
    def test_native_stats_are_inactive_before_any_result(self):
        from looplet.scaffolding import NativeToolStats

        assert NativeToolStats().has_activity is False

    def test_native_stats_record_success_and_fallback(self):
        from looplet.scaffolding import NativeToolStats

        stats = NativeToolStats()
        success = llm_call_with_retry(_NativeBackend(), "hi", tools=_weather_schema())
        stats.record(success)
        fallback = llm_call_with_retry(
            _NativeFailureBackend(), "hi", tools=_weather_schema(), max_retries=0
        )
        stats.record(fallback)

        assert stats.to_dict() == {
            "requested": 2,
            "attempted": 2,
            "succeeded": 1,
            "fallbacks": 1,
            "last_fallback_reason": "RuntimeError: tools endpoint unsupported",
        }

    def test_native_policy_defaults_on_and_parses_after_demotion(self):
        policy = NativeToolPolicy()
        backend = _NativeBackend()
        assert policy.enabled is True
        assert policy.should_use(backend, _weather_schema())
        assert (
            policy.parse_response([{"type": "tool_use", "id": "n1", "name": "done", "input": {}}])[
                0
            ].tool
            == "done"
        )

        policy.demote()
        assert not policy.should_use(backend, _weather_schema())
        assert policy.parse_response('{"tool": "done", "args": {}}')[0].tool == "done"

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

    def test_native_failure_falls_back_to_regular_generation(self):
        backend = _NativeFailureBackend()
        result = llm_call_with_retry(
            backend,
            "hi",
            tools=_weather_schema(),
            max_retries=0,
        )
        assert result.ok
        assert isinstance(result.text, str)
        assert backend.calls == ["native", "regular"]
        assert result.native_fallback is True

    def test_native_failure_is_demoted_for_the_rest_of_the_loop(self):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})

        for _ in composable_loop(
            llm=backend,
            task={"goal": "echo then finish"},
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
        ):
            pass

        assert backend.native_calls == 1
        assert backend.regular_calls == 2

    @pytest.mark.asyncio
    async def test_async_native_failure_is_demoted_for_the_rest_of_the_loop(self):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})

        async for _ in async_composable_loop(
            llm=backend,
            task={"goal": "echo then finish"},
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
        ):
            pass

        assert backend.native_calls == 1
        assert backend.regular_calls == 2
