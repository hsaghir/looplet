"""Tests for LLM backend adapters (using mocks — no API keys needed)."""

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

    def _create(self, **kwargs):
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
