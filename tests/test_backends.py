"""Tests for LLM backend adapters (using mocks - no API keys needed)."""

from __future__ import annotations

import pytest

from looplet.backends import (
    AnthropicBackend,
    AnthropicStreamingBackend,
    AsyncAnthropicBackend,
    AsyncOpenAIBackend,
    OpenAIBackend,
    OpenAIStreamingBackend,
)
from looplet.cache import CacheControl, CachePolicy, compute_breakpoints


@pytest.mark.parametrize(
    "backend_type",
    [
        OpenAIBackend,
        OpenAIStreamingBackend,
        AsyncOpenAIBackend,
        AnthropicBackend,
        AnthropicStreamingBackend,
        AsyncAnthropicBackend,
    ],
)
def test_public_backend_sampling_defaults_are_unset(backend_type):
    import inspect

    for method_name in ("generate", "generate_with_tools", "stream"):
        method = getattr(backend_type, method_name, None)
        if method is not None:
            assert inspect.signature(method).parameters["temperature"].default is None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("temperature", [None, 0.3])
async def test_execution_history_does_not_require_a_steering_policy(
    provider,
    asynchronous,
    temperature,
):
    import json

    from looplet import (
        DefaultState,
        LoopConfig,
        RunResult,
        async_composable_loop,
        composable_loop,
        tools_from,
    )

    httpx = pytest.importorskip("httpx")
    sdk = pytest.importorskip(provider)
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        name = "inspect_source" if len(requests) == 1 else "done"
        arguments = {} if len(requests) == 1 else {"summary": "ok"}
        identifier = f"call-{len(requests)}"
        if provider == "openai":
            payload = {
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": identifier,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        else:
            payload = {
                "id": "msg-offline",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [
                    {"type": "tool_use", "id": identifier, "name": name, "input": arguments}
                ],
            }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handle)
    http_client = (
        httpx.AsyncClient(transport=transport)
        if asynchronous
        else httpx.Client(transport=transport)
    )
    client_type = (
        (sdk.AsyncOpenAI if asynchronous else sdk.OpenAI)
        if provider == "openai"
        else (sdk.AsyncAnthropic if asynchronous else sdk.Anthropic)
    )
    client = client_type(api_key="offline-fixture", max_retries=0, http_client=http_client)
    backend_type = (
        (AsyncOpenAIBackend if asynchronous else OpenAIBackend)
        if provider == "openai"
        else (AsyncAnthropicBackend if asynchronous else AnthropicBackend)
    )

    def inspect_source() -> dict:
        return {"evidence": "actual source"}

    state = DefaultState(max_steps=3)
    config = LoopConfig(max_steps=3, temperature=temperature)
    arguments = dict(
        llm=backend_type(client=client),
        tools=tools_from([inspect_source], include_done=True),
        state=state,
        config=config,
    )
    try:
        if asynchronous:
            async for _ in async_composable_loop(**arguments):
                pass
        else:
            list(composable_loop(**arguments))
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()

    assert config.assistant_turn_policy is None
    assert config.max_model_calls is None
    assert RunResult.from_state(state).completed
    assert len(requests) == 2
    for request in requests:
        if temperature is None:
            assert "temperature" not in request
        else:
            assert request["temperature"] == temperature
    messages = requests[1]["messages"]
    if provider == "openai":
        result = next(message for message in messages if message["role"] == "tool")
        assert result["tool_call_id"] == "call-1"
        assert "actual source" in result["content"]
    else:
        results = [
            block
            for message in messages
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert results[0]["tool_use_id"] == "call-1"
        assert "actual source" in json.dumps(results[0]["content"])


def test_public_backend_wrapper_preserves_optional_capabilities():
    from looplet.backend_contract import BackendWrapper, ConversationInputBackend

    class StatefulBackend:
        conversation_mode = "stateful"
        last_usage = {"input_tokens": 1}
        private_secret = "not a supported capability"

        def generate(self, prompt):
            return prompt

        def set_conversation_input(self, snapshot):
            self.selected = snapshot

    backend = StatefulBackend()
    wrapped = BackendWrapper(BackendWrapper(backend))
    assert isinstance(backend, ConversationInputBackend)
    assert wrapped.conversation_mode == "stateful"
    assert callable(getattr(wrapped, "set_conversation_input", None))
    wrapped.set_conversation_input({"prompt": "selected"})
    assert backend.selected == {"prompt": "selected"}
    assert wrapped.generate("hello") == "hello"
    wrapped.last_usage = {"input_tokens": 2}
    assert backend.last_usage == {"input_tokens": 2}
    assert not hasattr(wrapped, "private_secret")
    assert not hasattr(wrapped, "generate_with_tools")
    assert not hasattr(wrapped, "checkpoint_state")


async def test_public_backend_wrapper_preserves_async_generation():
    import inspect

    from looplet.backend_contract import BackendWrapper, invoke_backend

    class AsyncBackend:
        async def generate(self, prompt):
            return prompt

    wrapped = BackendWrapper(AsyncBackend())
    assert inspect.iscoroutinefunction(wrapped.generate)
    assert await invoke_backend(wrapped.generate, "legacy", {"temperature": 0.2}) == "legacy"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_explicit_provider_options_reach_requests(provider, asynchronous, native):
    from types import SimpleNamespace

    requests = []
    response = _MockResponse("response") if provider == "openai" else _MockAnthropicResponse()

    def create(**kwargs):
        requests.append(kwargs)
        return response

    async def create_async(**kwargs):
        return create(**kwargs)

    operation = create_async if asynchronous else create
    client = SimpleNamespace(
        messages=SimpleNamespace(create=operation),
        chat=SimpleNamespace(completions=SimpleNamespace(create=operation)),
    )
    backend_type = (
        (AsyncOpenAIBackend if asynchronous else OpenAIBackend)
        if provider == "openai"
        else (AsyncAnthropicBackend if asynchronous else AnthropicBackend)
    )
    options = (
        {"reasoning_effort": "high", "top_p": 0.5}
        if provider == "openai"
        else {"thinking": {"type": "enabled", "budget_tokens": 1024}, "top_p": 0.5}
    )
    backend = backend_type(client=client, request_options=options)
    assert backend.request_configuration["applied"] is None
    expected = backend.request_configuration["requested"]
    options["top_p"] = 0.9
    if provider == "anthropic":
        options["thinking"]["budget_tokens"] = 2048
    kwargs = {"max_tokens": 4096, "temperature": 1.0}
    if native:
        kwargs["tools"] = []
    method = backend.generate_with_tools if native else backend.generate
    if asynchronous:
        await method("request", **kwargs)
    else:
        method("request", **kwargs)
    assert {key: requests[0][key] for key in expected} == expected
    configuration = backend.request_configuration
    assert configuration["requested"] == configuration["applied"] == expected
    assert configuration["reported"] is None
    configuration["requested"]["top_p"] = 0.1
    assert backend.request_configuration["requested"]["top_p"] == 0.5


@pytest.mark.parametrize(
    "backend_type, options",
    [
        (OpenAIBackend, {"thinking": {"type": "enabled", "budget_tokens": 1024}}),
        (AnthropicBackend, {"reasoning_effort": "high"}),
        (OpenAIBackend, {"max_tokens": 10}),
        (OpenAIBackend, {"reasoning_effort": []}),
        (OpenAIBackend, {"top_p": float("nan")}),
        (OpenAIBackend, {"top_p": True}),
        (OpenAIBackend, {"seed": True}),
        (AnthropicBackend, {"thinking": {"type": "enabled", "budget_tokens": 1}}),
        (AnthropicBackend, {"thinking": {"type": []}}),
        (OpenAIBackend, {1: "invalid field"}),
    ],
)
def test_explicit_provider_options_reject_unsupported_values(backend_type, options):
    with pytest.raises(ValueError):
        backend_type(client=object(), request_options=options)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_explicit_provider_options_reach_streams(provider):
    client = _MockOpenAIClient() if provider == "openai" else _MockAnthropicClient()
    backend_type = OpenAIStreamingBackend if provider == "openai" else AnthropicStreamingBackend
    backend = backend_type(client, request_options={"top_p": 0.3})
    assert "".join(backend.stream("request"))
    assert client._last_kwargs["top_p"] == 0.3
    assert backend.request_configuration["applied"] == {"top_p": 0.3}


def test_public_retry_policy_replaces_private_backend_flags():
    from types import SimpleNamespace

    from looplet.backend_contract import BackendRequestPolicy, BackendWrapper, attempt_limit
    from looplet.subagent import SharedModelBudget

    backend = SimpleNamespace(
        generate=lambda prompt: prompt,
        request_policy=BackendRequestPolicy(manages_retries=True),
    )
    wrapped = BackendWrapper(backend)
    assert attempt_limit(wrapped, 3) == 1
    budget = SharedModelBudget(2)
    with pytest.raises(RuntimeError, match="single-request"):
        budget.wrap(wrapped).generate("not admitted")
    assert budget.remaining == 2


def test_legacy_retry_overrides_remain_effective_on_sdk_subclasses():
    from looplet.backend_contract import backend_request_policy

    class LegacyAdapter(OpenAIBackend):
        _looplet_manages_retries = True
        _supports_single_model_request = False

    policy = backend_request_policy(LegacyAdapter(client=object()))
    assert policy.manages_retries
    assert not policy.supports_single_model_request


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_public_contract_with_real_sdk_transport(provider, asynchronous, native, failure):
    import json

    httpx = pytest.importorskip("httpx")
    sdk = pytest.importorskip(provider)
    from looplet.backend_contract import BackendWrapper
    from looplet.subagent import SharedModelBudget

    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if failure:
            return httpx.Response(500, json={"error": {"type": "api_error", "message": "offline"}})
        payload = (
            {
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-fixture",
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "observed"},
                    }
                ],
            }
            if provider == "openai"
            else {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "offline-fixture",
                "content": [{"type": "text", "text": "observed"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            }
        )
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handle)
    http_client = (
        httpx.AsyncClient(transport=transport)
        if asynchronous
        else httpx.Client(transport=transport)
    )
    client_type = (
        (sdk.AsyncOpenAI if asynchronous else sdk.OpenAI)
        if provider == "openai"
        else (sdk.AsyncAnthropic if asynchronous else sdk.Anthropic)
    )
    client = client_type(
        api_key="offline-fixture",
        base_url="https://looplet.invalid/v1",
        http_client=http_client,
        max_retries=2,
    )
    backend_type = (
        (AsyncOpenAIBackend if asynchronous else OpenAIBackend)
        if provider == "openai"
        else (AsyncAnthropicBackend if asynchronous else AnthropicBackend)
    )
    options = (
        {"reasoning_effort": "high"}
        if provider == "openai"
        else {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    backend = backend_type(client, model="offline-fixture", request_options=options)
    budget = SharedModelBudget(3)
    wrapped = BackendWrapper(budget.wrap(BackendWrapper(backend)))
    method = wrapped.generate_with_tools if native else wrapped.generate
    kwargs = {"max_tokens": 4096, "temperature": 1.0}
    if native:
        kwargs["tools"] = []
    try:
        if failure:
            with pytest.raises(sdk.APIStatusError):
                if asynchronous:
                    await method("request", **kwargs)
                else:
                    method("request", **kwargs)
        else:
            response = (
                await method("request", **kwargs) if asynchronous else method("request", **kwargs)
            )
            assert response
            assert wrapped.last_usage["input"] == 2
            assert wrapped.last_stop_reason == "stop"
        assert len(requests) == 1
        assert budget.remaining == 2
        assert {key: requests[0][key] for key in options} == options
        assert wrapped.request_configuration["applied"] == options
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


def test_request_attempt_callbacks_remain_live_through_nested_budget_wrappers():
    from types import SimpleNamespace

    from looplet.backend_contract import BackendWrapper
    from looplet.subagent import SharedModelBudget

    observed = []
    raw = SimpleNamespace(request_attempt_callback=None, generate=lambda prompt: prompt)
    raw.set_request_attempt_callback = lambda callback: setattr(
        raw, "request_attempt_callback", callback
    )
    wrapped = BackendWrapper(SharedModelBudget(2).wrap(BackendWrapper(raw)))
    wrapped.set_request_attempt_callback(lambda: observed.append("attempt"))
    wrapped.request_attempt_callback()
    assert observed == ["attempt"]
    wrapped.request_attempt_callback = None
    assert raw.request_attempt_callback is None


def test_public_backend_contract_keeps_legacy_error_and_request_scope_identity():
    from looplet import _backend_contract, backend_contract
    from looplet.backend_contract import (
        TerminalLLMError,
        single_model_request,
        single_model_request_active,
    )

    assert set(_backend_contract.__all__) == {
        "FORWARDED_CAPABILITIES",
        "MUTABLE_CAPABILITIES",
        "TerminalLLMError",
        "attempt_limit",
        "backend_attribute",
        "invoke_backend",
        "json_snapshot",
        "single_model_request",
        "single_model_request_active",
    }
    for name in _backend_contract.__all__:
        assert getattr(_backend_contract, name) is getattr(backend_contract, name)
    assert TerminalLLMError is _backend_contract.TerminalLLMError
    assert not single_model_request_active()
    with single_model_request():
        assert _backend_contract.single_model_request_active()
        with _backend_contract.single_model_request():
            assert single_model_request_active()
        assert single_model_request_active()
    assert not single_model_request_active()


# ── Mock API objects ────────────────────────────────────────────


class _MockChoice:
    def __init__(self, text: str) -> None:
        self.message = type("M", (), {"content": text})()


class _MockResponse:
    def __init__(self, text: str) -> None:
        self.choices = [_MockChoice(text)]


class _MockDelta:
    def __init__(self, text: str) -> None:
        self.content = text


class _MockStreamChunk:
    def __init__(self, text: str) -> None:
        self.choices = [type("C", (), {"delta": _MockDelta(text)})()]


class _MockOpenAIClient:
    """Mock openai.OpenAI client."""

    def __init__(self) -> None:
        self.chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "Completions",
                    (),
                    {
                        "create": self._create,
                    },
                )(),
            },
        )()
        self._last_kwargs: dict = {}

    def _create(self, **kwargs):
        self._last_kwargs = kwargs
        if kwargs.get("stream"):
            return iter([_MockStreamChunk("Hello"), _MockStreamChunk(" world")])
        return _MockResponse("Hello world")


class _MockContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockAnthropicResponse:
    def __init__(self) -> None:
        self.content = [_MockContentBlock("Anthropic response")]


class _MockAnthropicStreamCtx:
    def __init__(self) -> None:
        self.text_stream = iter(["chunk1", "chunk2"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _MockAnthropicClient:
    """Mock anthropic.Anthropic client."""

    def __init__(self) -> None:
        self.messages = type(
            "Messages",
            (),
            {
                "create": self._create,
                "stream": self._stream,
            },
        )()
        self._last_kwargs: dict = {}

    def _create(self, **kwargs):
        self._last_kwargs = kwargs
        return _MockAnthropicResponse()

    def _stream(self, **kwargs):
        self._last_kwargs = kwargs
        return _MockAnthropicStreamCtx()


# ── Tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("capped", [False, True])
async def test_capped_sdk_requests_disable_hidden_retries_without_mutating_client(
    asynchronous, provider, capped
):
    from types import SimpleNamespace

    from looplet import DefaultState, LoopConfig
    from looplet.assistant_turns import _AssistantTurns

    options = []
    response = _MockResponse("returned") if provider == "openai" else _MockAnthropicResponse()

    def create(**kwargs):
        return response

    async def create_async(**kwargs):
        return response

    def with_options(**kwargs):
        options.append(kwargs)
        return client

    operation = create_async if asynchronous else create
    client = SimpleNamespace(
        messages=SimpleNamespace(create=operation),
        chat=SimpleNamespace(completions=SimpleNamespace(create=operation)),
        with_options=with_options,
    )
    backend_type = (
        (AsyncOpenAIBackend if asynchronous else OpenAIBackend)
        if provider == "openai"
        else (AsyncAnthropicBackend if asynchronous else AnthropicBackend)
    )
    original = backend_type(client, model="test")
    backend = (
        _AssistantTurns(DefaultState(), LoopConfig(max_model_calls=1)).wrap(original)
        if capped
        else original
    )
    result = backend.generate("Task", max_tokens=10)
    assert await result if asynchronous else result
    assert options == ([{"max_retries": 0}] if capped else [])
    assert original._client is client


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_assistant_conversation_preserves_selected_user_context(async_mode, provider):
    import json

    from looplet.conversation import Conversation, Message, MessageRole

    client = _MockOpenAIClient() if provider == "openai" else _MockAnthropicClient()

    async def create_async(**kwargs):
        return client._create(**kwargs)

    if async_mode:
        if provider == "openai":
            client.chat.completions.create = create_async
        else:
            client.messages.create = create_async
    backend_type = (
        (AsyncOpenAIBackend if async_mode else OpenAIBackend)
        if provider == "openai"
        else (AsyncAnthropicBackend if async_mode else AnthropicBackend)
    )
    backend = backend_type(client, model="test")
    selected = Conversation(
        [
            Message(role=MessageRole.SYSTEM, content="UNTRUSTED_PRIOR_SUMMARY"),
            Message(role=MessageRole.USER, content="CALLER_SUPPLIED_USER_CONSTRAINT"),
        ]
    )
    backend.set_conversation_input({"conversation": selected.serialize(), "prompt": "Task"})
    pending = backend.generate_with_tools("Task", tools=[])
    if async_mode:
        await pending
    messages = client._last_kwargs["messages"]
    assert json.dumps(messages).count("CALLER_SUPPLIED_USER_CONSTRAINT") == 1
    assert json.dumps(messages).count("UNTRUSTED_PRIOR_SUMMARY") == 1
    assert all(message["role"] == "user" for message in messages)
    rendered = selected.continuation_context()
    assert rendered.count("CALLER_SUPPLIED_USER_CONSTRAINT") == 1
    assert "[PRIOR CONTEXT, NOT VERIFIED FACTS] UNTRUSTED_PRIOR_SUMMARY" in rendered


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_assistant_conversation_preserves_provider_roles_and_tool_pairs(async_mode, provider):
    import json
    from types import SimpleNamespace

    from looplet import DefaultState, LoopConfig, async_composable_loop, composable_loop, tools_from
    from looplet.assistant_turns import AssistantTurnDecision

    requests = []
    effects = []
    responses = [
        [{"type": "text", "text": "A retained hypothesis"}],
        [
            {"type": "text", "text": "Fetch evidence"},
            {"type": "tool_use", "id": "real_lookup", "name": "lookup", "input": {"value": 7}},
        ],
        [{"type": "tool_use", "id": "real_done", "name": "done", "input": {"answer": "verified"}}],
    ]
    if provider == "anthropic":
        responses[0].insert(
            0, {"type": "thinking", "thinking": "opaque", "signature": "exact-signature"}
        )

    def create(**kwargs):
        requests.append(kwargs)
        blocks = responses[len(requests) - 1]
        if provider == "anthropic":
            return SimpleNamespace(
                content=blocks, stop_reason="end_turn" if len(requests) == 1 else "tool_use"
            )
        calls = [
            SimpleNamespace(
                id=block["id"],
                function=SimpleNamespace(name=block["name"], arguments=json.dumps(block["input"])),
            )
            for block in blocks
            if block["type"] == "tool_use"
        ]
        text = "".join(block["text"] for block in blocks if block["type"] == "text")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop" if not calls else "tool_calls",
                    message=SimpleNamespace(content=text, tool_calls=calls),
                )
            ]
        )

    async def create_async(**kwargs):
        return create(**kwargs)

    operation = create_async if async_mode else create
    client = SimpleNamespace(
        messages=SimpleNamespace(create=operation),
        chat=SimpleNamespace(completions=SimpleNamespace(create=operation)),
    )
    backend_type = (
        (AsyncOpenAIBackend if async_mode else OpenAIBackend)
        if provider == "openai"
        else (AsyncAnthropicBackend if async_mode else AnthropicBackend)
    )
    backend = backend_type(client, model="test")

    def lookup(value: int):
        effects.append(value)
        return {"observed": "UNIQUE_TOOL_OBSERVATION"}

    def done(answer: str):
        return {"answer": answer}

    state = DefaultState(max_steps=3)
    kwargs = dict(
        llm=backend,
        state=state,
        tools=tools_from([lookup, done]),
        task={"description": "Inspect this unique task"},
        config=LoopConfig(
            max_steps=3,
            max_model_calls=4,
            assistant_turn_policy=lambda context: AssistantTurnDecision(
                "continue", "Use observed evidence"
            ),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert [step.tool_call.tool for step in steps] == ["lookup", "done"]
    assert effects == [7]
    assert len(requests) == 3
    wire = json.dumps(requests[-1]["messages"])
    assert wire.count("A retained hypothesis") == 1
    assert wire.count("Inspect this unique task") == 1
    assert wire.count("Use observed evidence") == 1
    assert wire.count("UNIQUE_TOOL_OBSERVATION") == 1
    assert "real_lookup" in wire
    assert "observed" in wire
    if provider == "anthropic":
        assert wire.count("exact-signature") == 1
        assert "tool_use_id" in wire
    else:
        assert any(
            message["role"] == "tool" and message["tool_call_id"] == "real_lookup"
            for message in requests[-1]["messages"]
        )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("native_tools", [False, True])
async def test_openai_multipart_content_keeps_visible_text(async_mode, native_tools):
    from types import SimpleNamespace

    message = SimpleNamespace(
        content=[
            {"type": "text", "text": "Hello"},
            {"type": "input_file", "input_file": {"file_data": "opaque metadata"}},
            {"type": "text", "text": " world"},
        ],
        tool_calls=[
            SimpleNamespace(
                id="finish", function=SimpleNamespace(name="done", arguments='{"summary":"ok"}')
            )
        ],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def create(**kwargs):
        return response

    async def create_async(**kwargs):
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_async if async_mode else create)
        )
    )
    backend = (AsyncOpenAIBackend if async_mode else OpenAIBackend)(client, model="test")
    operation = backend.generate_with_tools if native_tools else backend.generate
    options = {"tools": []} if native_tools else {}
    result = await operation("hello", **options) if async_mode else operation("hello", **options)
    if native_tools:
        assert result == [
            {"type": "text", "text": "Hello world"},
            {"type": "tool_use", "id": "finish", "name": "done", "input": {"summary": "ok"}},
        ]
    else:
        assert result == "Hello world"


@pytest.mark.parametrize("async_mode", [False, True])
async def test_openai_native_keeps_reasoning_and_refusal(async_mode):
    from types import SimpleNamespace

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Decision summary",
                    reasoning_content="Synthetic provider reasoning",
                    refusal="Provider refusal detail",
                    tool_calls=[],
                )
            )
        ]
    )

    def create(**kwargs):
        return response

    async def create_async(**kwargs):
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_async if async_mode else create)
        )
    )
    backend = (AsyncOpenAIBackend if async_mode else OpenAIBackend)(client, model="test")
    result = (
        await backend.generate_with_tools("hello", tools=[])
        if async_mode
        else backend.generate_with_tools("hello", tools=[])
    )
    assert result == [
        {"type": "reasoning", "text": "Synthetic provider reasoning"},
        {"type": "refusal", "refusal": "Provider refusal detail"},
        {"type": "text", "text": "Decision summary"},
    ]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_anthropic_native_keeps_provider_blocks_and_snapshots(async_mode):
    import copy
    from types import SimpleNamespace

    expected = [
        {"type": "thinking", "thinking": "Synthetic reasoning", "signature": "signed-state"},
        {"type": "redacted_thinking", "data": "opaque-state"},
        {"type": "text", "text": "Decision summary"},
        {"type": "tool_use", "id": "finish", "name": "done", "input": {"summary": "ok"}},
        {"type": "provider_extension", "payload": {"value": 0}},
    ]

    class ProviderBlock:
        def __init__(self, value):
            self.value = copy.deepcopy(value)

        def model_dump(self, *, mode):
            assert mode == "json"
            return self.value

    response = SimpleNamespace(content=[ProviderBlock(value) for value in expected])

    def create(**kwargs):
        return response

    async def create_async(**kwargs):
        return response

    client = SimpleNamespace(
        messages=SimpleNamespace(create=create_async if async_mode else create)
    )
    backend = (AsyncAnthropicBackend if async_mode else AnthropicBackend)(client, model="test")
    result = (
        await backend.generate_with_tools("hello", tools=[])
        if async_mode
        else backend.generate_with_tools("hello", tools=[])
    )
    assert result == expected
    response.content[0].value["thinking"] = "Mutated after return"
    assert result == expected


class TestOpenAIBackend:
    def test_generate(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="test-model")
        result = llm.generate("hello", system_prompt="be nice")
        assert result == "Hello world"

    def test_model_passed(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="gpt-4o-mini")
        llm.generate("hello")
        assert client._last_kwargs["model"] == "gpt-4o-mini"

    def test_system_prompt_included(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="test")
        llm.generate("hello", system_prompt="be nice")
        msgs = client._last_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "be nice"
        assert msgs[1]["role"] == "user"

    def test_no_system_prompt(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="test")
        llm.generate("hello")
        msgs = client._last_kwargs["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_satisfies_llm_backend_protocol(self):
        from looplet.types import LLMBackend

        client = _MockOpenAIClient()
        llm = OpenAIBackend(client)
        assert isinstance(llm, LLMBackend)


class TestOpenAIStreamingBackend:
    def test_stream_yields_chunks(self):
        client = _MockOpenAIClient()
        llm = OpenAIStreamingBackend(client, model="test")
        chunks = list(llm.stream("hello"))
        assert chunks == ["Hello", " world"]

    @pytest.mark.parametrize("async_mode", [False, True])
    def test_stream_filters_non_text_content(self, async_mode):
        import asyncio
        from types import SimpleNamespace

        from looplet.backends import AsyncOpenAIBackend

        contents = [
            "Hello",
            [{"type": "input_file", "mime_type": "stateful_marker", "file_data": "opaque"}],
            [
                {"type": "text", "text": " world"},
                {"type": "input_file", "mime_type": "usage", "file_data": "opaque"},
            ],
            None,
            {"type": "metadata"},
        ]
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])
            for content in contents
        ]
        chunks.append(SimpleNamespace(choices=[]))

        async def asynchronous_chunks():
            for chunk in chunks:
                yield chunk

        async def create_async(**kwargs):
            return asynchronous_chunks()

        create = create_async if async_mode else lambda **kwargs: iter(chunks)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        if async_mode:
            backend = AsyncOpenAIBackend(client, model="test")

            async def consume():
                return [chunk async for chunk in backend.stream("hello")]

            actual = asyncio.run(consume())
        else:
            actual = list(OpenAIStreamingBackend(client, model="test").stream("hello"))
        assert actual == ["Hello", " world"]

    def test_generate_still_works(self):
        client = _MockOpenAIClient()
        llm = OpenAIStreamingBackend(client, model="test")
        result = llm.generate("hello")
        assert result == "Hello world"


class TestAnthropicBackend:
    def test_generate(self):
        client = _MockAnthropicClient()
        llm = AnthropicBackend(client, model="test-model")
        result = llm.generate("hello")
        assert result == "Anthropic response"

    def test_satisfies_llm_backend_protocol(self):
        from looplet.types import LLMBackend

        client = _MockAnthropicClient()
        llm = AnthropicBackend(client)
        assert isinstance(llm, LLMBackend)


class TestAnthropicStreamingBackend:
    def test_stream_yields_chunks(self):
        client = _MockAnthropicClient()
        llm = AnthropicStreamingBackend(client, model="test")
        chunks = list(llm.stream("hello"))
        assert chunks == ["chunk1", "chunk2"]


class TestLLMChunkEvent:
    def test_chunk_event_creation(self):
        from looplet.streaming import LLMChunkEvent

        e = LLMChunkEvent(step_num=3, chunk="hello", chunk_index=0)
        assert e.event_type == "LLMChunkEvent"
        assert e.step_num == 3
        assert e.chunk == "hello"


class TestExports:
    def test_backends_exported(self):
        from looplet import AnthropicBackend, OpenAIBackend
        from looplet.backends import (
            AnthropicStreamingBackend,
            AsyncAnthropicBackend,
            AsyncOpenAIBackend,
            OpenAIStreamingBackend,
        )
        from looplet.streaming import LLMChunkEvent

        assert OpenAIBackend is not None
        assert LLMChunkEvent is not None


class TestFromEnvErrors:
    """Regression: from_env() must raise a clean RuntimeError upfront
    when the required env vars are missing, not let the SDK fail later
    with an unrelated exception type."""

    def _clear_env(self, monkeypatch, prefixes):
        import os

        for k in list(os.environ):
            if any(k.startswith(p) for p in prefixes):
                monkeypatch.delenv(k, raising=False)

    def test_openai_from_env_raises_when_no_key_or_base_url(self, monkeypatch):
        from looplet import OpenAIBackend

        self._clear_env(monkeypatch, ("OPENAI_",))
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIBackend.from_env()

    def test_async_openai_from_env_raises_when_no_key_or_base_url(self, monkeypatch):
        from looplet.backends import AsyncOpenAIBackend

        self._clear_env(monkeypatch, ("OPENAI_",))
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            AsyncOpenAIBackend.from_env()

    def test_anthropic_from_env_raises_when_no_key(self, monkeypatch):
        from looplet import AnthropicBackend

        self._clear_env(monkeypatch, ("ANTHROPIC_",))
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicBackend.from_env()

    def test_openai_from_env_local_server_with_base_url_only(self, monkeypatch):
        """OPENAI_BASE_URL set without a key (local Ollama / vLLM
        convention) should succeed by defaulting api_key to a sentinel."""
        pytest.importorskip("openai")  # optional extra; CI installs it
        from looplet import OpenAIBackend

        self._clear_env(monkeypatch, ("OPENAI_",))
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        backend = OpenAIBackend.from_env()
        assert backend is not None


class TestAnthropicCacheControl:
    """AnthropicBackend translates looplet cache breakpoints into native
    ``cache_control`` markers on system, tools, and the memory prefix. All
    behaviour is additive: absent/empty/unlocatable sections are skipped."""

    def test_system_prompt_marked(self):
        client = _MockAnthropicClient()
        llm = AnthropicBackend(client, model="test")
        bps = compute_breakpoints(
            CachePolicy(system_prompt=CacheControl()),
            system_prompt="SYS",
            tool_schemas_text="",
            memory_text="",
        )
        llm.generate("hi", system_prompt="SYS", cache_breakpoints=bps)
        system = client._last_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["text"] == "SYS"
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_last_tool_marked_with_ttl(self):
        client = _MockAnthropicClient()
        llm = AnthropicBackend(client, model="test")
        tools = [
            {"name": "a", "description": "", "input_schema": {}},
            {"name": "b", "description": "", "input_schema": {}},
        ]
        bps = compute_breakpoints(
            CachePolicy(tool_schemas=CacheControl(ttl="1h")),
            system_prompt="",
            tool_schemas_text="T",
            memory_text="",
        )
        llm.generate_with_tools("hi", tools=tools, cache_breakpoints=bps)
        marked = client._last_kwargs["tools"]
        assert "cache_control" not in marked[0]
        assert marked[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        # original tool dicts must not be mutated
        assert "cache_control" not in tools[-1]

    def test_memory_prefix_split(self):
        client = _MockAnthropicClient()
        llm = AnthropicBackend(client, model="test")
        prompt = "MEMBLOB\n\nTASK: go"
        bps = compute_breakpoints(
            CachePolicy(memory=CacheControl()),
            system_prompt="",
            tool_schemas_text="",
            memory_text="MEMBLOB",
        )
        llm.generate(prompt, cache_breakpoints=bps)
        content = client._last_kwargs["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["text"] == "MEMBLOB"
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[1]["text"] == "\n\nTASK: go"
        assert "cache_control" not in content[1]

    def test_unlocatable_memory_is_skipped(self):
        client = _MockAnthropicClient()
        llm = AnthropicBackend(client, model="test")
        bps = compute_breakpoints(
            CachePolicy(memory=CacheControl()),
            system_prompt="",
            tool_schemas_text="",
            memory_text="NOT-IN-PROMPT",
        )
        llm.generate("hello world", cache_breakpoints=bps)
        assert client._last_kwargs["messages"] == [{"role": "user", "content": "hello world"}]

    def test_no_breakpoints_leaves_request_unchanged(self):
        client = _MockAnthropicClient()
        llm = AnthropicBackend(client, model="test")
        llm.generate("hi", system_prompt="SYS")
        assert client._last_kwargs["system"] == "SYS"
        assert client._last_kwargs["messages"] == [{"role": "user", "content": "hi"}]


class TestOpenAIPromptCacheKey:
    """OpenAIBackend emits ``prompt_cache_key`` (via SDK-safe ``extra_body``)
    only when explicitly opted in, so strict local/proxy servers are never
    sent an unknown field by default. OpenAI caches prefixes automatically."""

    def _bps(self):
        return compute_breakpoints(
            CachePolicy(system_prompt=CacheControl()),
            system_prompt="SYS",
            tool_schemas_text="",
            memory_text="",
        )

    def test_off_by_default(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="test")
        llm.generate("hi", cache_breakpoints=self._bps())
        assert "extra_body" not in client._last_kwargs

    def test_emitted_when_enabled(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="test", use_prompt_cache_key=True)
        llm.generate("hi", cache_breakpoints=self._bps())
        key = client._last_kwargs["extra_body"]["prompt_cache_key"]
        assert key.startswith("looplet-")

    def test_key_tracks_stable_sections_not_prompt(self):
        client = _MockOpenAIClient()
        llm = OpenAIBackend(client, model="test", use_prompt_cache_key=True)
        llm.generate("hi", cache_breakpoints=self._bps())
        k1 = client._last_kwargs["extra_body"]["prompt_cache_key"]
        llm.generate("a totally different prompt", cache_breakpoints=self._bps())
        k2 = client._last_kwargs["extra_body"]["prompt_cache_key"]
        assert k1 == k2


class TestCacheBreakpointWiring:
    """Every shipped backend entrypoint must declare ``cache_breakpoints`` so
    the loop dispatch (scaffolding._accepts_kwarg) actually forwards them."""

    def test_all_methods_accept_cache_breakpoints(self):
        import inspect

        for cls in (OpenAIBackend, AnthropicBackend, AsyncOpenAIBackend, AsyncAnthropicBackend):
            for method in ("generate", "generate_with_tools"):
                sig = inspect.signature(getattr(cls, method))
                assert "cache_breakpoints" in sig.parameters, f"{cls.__name__}.{method}"
