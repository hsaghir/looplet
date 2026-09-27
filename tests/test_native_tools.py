"""Tests for native tool calling wiring in backends + scaffolding + loop.

Covers:
 - OpenAI / Anthropic backends implement ``generate_with_tools`` that returns
   normalised Anthropic-style content blocks.
 - ``llm_call_with_retry`` requires native tool support when enabled, and
     uses ``generate`` only for explicitly selected text mode.
 - The composable loop passes tool schemas through when
         ``LoopConfig.use_native_tools`` is enabled (the default), rejects
         text-only native responses, and parses structured ``tool_use`` blocks.
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
from looplet.native_tools import NativeToolPolicy, NativeToolUnsupportedError
from looplet.scaffolding import llm_call_with_retry
from looplet.testing import AsyncMockLLMBackend, MockLLMBackend
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

    def test_preserves_all_native_tool_calls_in_order(self):
        msg = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="first", arguments='{"value":1}'),
                ),
                SimpleNamespace(
                    id="call_2",
                    function=SimpleNamespace(name="second", arguments='{"value":2}'),
                ),
            ],
        )

        assert _openai_message_to_blocks(msg) == [
            {"type": "tool_use", "id": "call_1", "name": "first", "input": {"value": 1}},
            {"type": "tool_use", "id": "call_2", "name": "second", "input": {"value": 2}},
        ]


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

    def test_preserves_all_native_tool_calls_in_order(self):
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="tool_use", id="tu_1", name="first", input={"value": 1}),
                SimpleNamespace(type="tool_use", id="tu_2", name="second", input={"value": 2}),
            ]
        )

        assert _anthropic_response_to_blocks(response) == [
            {"type": "tool_use", "id": "tu_1", "name": "first", "input": {"value": 1}},
            {"type": "tool_use", "id": "tu_2", "name": "second", "input": {"value": 2}},
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
        from looplet.native_tools import NativeToolUnsupportedError

        raise NativeToolUnsupportedError("tools endpoint unsupported")

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
        from looplet.native_tools import NativeToolUnsupportedError

        raise NativeToolUnsupportedError("tools endpoint unsupported")

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
    def test_scripted_mock_returns_structured_native_calls(self):
        backend = MockLLMBackend(responses=['{"tool":"done","args":{},"call_id":"mock-1"}'])
        result = llm_call_with_retry(backend, "finish", tools=_weather_schema())

        assert result.ok
        assert result.text == [{"type": "tool_use", "name": "done", "input": {}, "id": "mock-1"}]
        assert NativeToolPolicy().parse_response(result.text)[0].tool == "done"
        assert backend.calls == 1

    @pytest.mark.asyncio
    async def test_async_scripted_mock_returns_structured_native_calls(self):
        from looplet.async_loop import async_llm_call

        backend = AsyncMockLLMBackend(responses=['{"tool":"done","args":{}}'])
        result = await async_llm_call(backend, "finish", tools=_weather_schema())

        assert result.ok
        assert len(result.text) == 1
        assert result.text[0]["type"] == "tool_use"
        assert result.text[0]["name"] == "done"
        assert result.text[0]["input"] == {}
        assert isinstance(result.text[0]["id"], str)
        assert NativeToolPolicy().parse_response(result.text)[0].tool == "done"
        assert backend.calls == 1

    def test_native_stats_are_inactive_before_any_result(self):
        from looplet.scaffolding import NativeToolStats

        assert NativeToolStats().has_activity is False

    def test_native_stats_record_success_and_unsupported(self):
        from looplet.scaffolding import NativeToolStats

        stats = NativeToolStats()
        success = llm_call_with_retry(_NativeBackend(), "hi", tools=_weather_schema())
        stats.record(success)
        unsupported = llm_call_with_retry(
            _NativeFailureBackend(), "hi", tools=_weather_schema(), max_retries=0
        )
        stats.record(unsupported)

        assert stats.to_dict() == {
            "requested": 2,
            "attempted": 2,
            "succeeded": 1,
            "fallbacks": 0,
            "last_fallback_reason": None,
        }
        assert isinstance(unsupported.error, NativeToolUnsupportedError)

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

    def test_native_policy_rejects_json_text_without_explicit_text_mode(self):
        policy = NativeToolPolicy()
        json_text = '{"tool": "done", "args": {}}'

        assert policy.parse_response([{"type": "text", "text": json_text}]) == []
        assert policy.parse_response(json_text) == []

        policy.enabled = False
        assert policy.parse_response(json_text)[0].tool == "done"

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

    def test_tools_but_non_native_backend_requires_explicit_text_mode(self):
        backend = _NoToolBackend()  # no generate_with_tools
        result = llm_call_with_retry(backend, "hi", tools=_weather_schema())
        assert not result.ok
        assert isinstance(result.error, NativeToolUnsupportedError)
        assert "use_native_tools=False" in str(result.error)

        result = llm_call_with_retry(
            backend, "hi", tools=_weather_schema(), native_policy=NativeToolPolicy(enabled=False)
        )
        assert result.ok
        assert isinstance(result.text, str)

    def test_loop_missing_native_backend_reports_unsupported(self):
        steps = list(
            composable_loop(
                llm=_NoToolBackend(),
                task={"goal": "finish"},
                tools=tools_from([_echo], include_done=True),
                state=DefaultState(max_steps=2),
                config=LoopConfig(max_steps=2),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
        assert "use_native_tools=False" in steps[0].tool_result.error

    def test_native_failure_stays_error_without_regular_generation(self):
        backend = _NativeFailureBackend()
        result = llm_call_with_retry(
            backend,
            "hi",
            tools=_weather_schema(),
            max_retries=0,
        )
        assert not result.ok
        assert isinstance(result.error, NativeToolUnsupportedError)
        assert backend.calls == ["native"]
        assert result.native_fallback is False

    def test_provider_failure_does_not_fallback_to_text(self):
        class ProviderFailureBackend:
            def generate_with_tools(self, prompt, *, tools, **kwargs):
                del prompt, tools, kwargs
                raise RuntimeError("authentication failed")

            def generate(self, prompt, **kwargs):
                del prompt, kwargs
                raise AssertionError("ordinary provider failures must not demote")

        result = llm_call_with_retry(
            ProviderFailureBackend(),
            "hi",
            tools=_weather_schema(),
            max_retries=0,
        )

        assert not result.ok
        assert isinstance(result.error, RuntimeError)
        assert "authentication failed" in str(result.error)
        assert result.native_fallback is False

    def test_text_only_native_response_retries_without_json_recovery(self):
        class TextThenNativeBackend:
            def __init__(self):
                self.native_calls = 0

            def generate(self, prompt, **kwargs):
                raise AssertionError("native mode must not call JSON-text recovery")

            def generate_with_tools(self, prompt, *, tools, **kwargs):
                self.native_calls += 1
                if self.native_calls == 1:
                    return [{"type": "text", "text": '{"tool":"done","args":{}}'}]
                return [{"type": "tool_use", "id": "n2", "name": "done", "input": {}}]

        backend = TextThenNativeBackend()
        tools = tools_from([_echo], include_done=True)
        steps = list(
            composable_loop(
                llm=backend,
                tools=tools,
                task={"goal": "finish"},
                state=DefaultState(max_steps=2),
                config=LoopConfig(max_steps=2),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["__parse_error__", "done"]
        assert "expected structured tool_use" in steps[0].tool_result.error
        assert backend.native_calls == 2

    @pytest.mark.asyncio
    async def test_async_text_only_native_response_retries_without_json_recovery(self):
        class TextThenNativeBackend:
            def __init__(self):
                self.native_calls = 0

            async def generate(self, prompt, **kwargs):
                raise AssertionError("native mode must not call JSON-text recovery")

            async def generate_with_tools(self, prompt, *, tools, **kwargs):
                self.native_calls += 1
                if self.native_calls == 1:
                    return [{"type": "text", "text": '{"tool":"done","args":{}}'}]
                return [{"type": "tool_use", "id": "n2", "name": "done", "input": {}}]

        backend = TextThenNativeBackend()
        tools = tools_from([_echo], include_done=True)
        steps = []
        async for step in async_composable_loop(
            llm=backend,
            tools=tools,
            task={"goal": "finish"},
            state=DefaultState(max_steps=2),
            config=LoopConfig(max_steps=2),
        ):
            steps.append(step)

        assert [step.tool_call.tool for step in steps] == ["__parse_error__", "done"]
        assert "expected structured tool_use" in steps[0].tool_result.error
        assert backend.native_calls == 2

    def test_native_failure_stops_the_loop_without_demoting(self):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})

        steps = list(
            composable_loop(
                llm=backend,
                task={"goal": "echo then finish"},
                tools=tools,
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
            )
        )

        assert backend.native_calls == 1
        assert backend.regular_calls == 0
        assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
        assert "tools endpoint unsupported" in steps[0].tool_result.error

    @pytest.mark.asyncio
    async def test_async_native_failure_stops_the_loop_without_demoting(self):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})

        steps = []
        async for step in async_composable_loop(
            llm=backend,
            task={"goal": "echo then finish"},
            tools=tools,
            state=DefaultState(max_steps=3),
            config=LoopConfig(max_steps=3),
        ):
            steps.append(step)

        assert backend.native_calls == 1
        assert backend.regular_calls == 0
        assert [step.tool_call.tool for step in steps] == ["__llm_error__"]
        assert "tools endpoint unsupported" in steps[0].tool_result.error

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.asyncio
    async def test_explicit_text_mode_completes_without_native_calls(self, async_mode):
        backend = _StickyNativeFailureBackend()
        tools = tools_from([_echo], include_done=True, done_parameters={"answer": "x"})
        config = LoopConfig(max_steps=3, use_native_tools=False)
        if async_mode:
            steps = []
            async for step in async_composable_loop(
                llm=backend,
                task={"goal": "echo then finish"},
                tools=tools,
                state=DefaultState(max_steps=3),
                config=config,
            ):
                steps.append(step)
        else:
            steps = list(
                composable_loop(
                    llm=backend,
                    task={"goal": "echo then finish"},
                    tools=tools,
                    state=DefaultState(max_steps=3),
                    config=config,
                )
            )

        assert [step.tool_call.tool for step in steps] == ["echo", "done"]
        assert backend.native_calls == 0
        assert backend.regular_calls == 2
