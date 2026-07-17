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
        return _MockAnthropicStreamCtx()


# ── Tests ────────────────────────────────────────────────────────


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
