"""LLM backend adapters for popular API providers.

Ready-to-use adapters that satisfy the ``LLMBackend`` and ``AsyncLLMBackend``
protocols for OpenAI-compatible and Anthropic APIs.  Each adapter accepts the
provider's native client object so cadence stays dependency-free — import the
SDK you already use and pass the client in.

Typical usage::

    # OpenAI
    from openai import OpenAI
    from openharness.backends import OpenAIBackend

    llm = OpenAIBackend(OpenAI(), model="gpt-4o")
    result = llm.generate("What is 2+2?")

    # Anthropic
    from anthropic import Anthropic
    from openharness.backends import AnthropicBackend

    llm = AnthropicBackend(Anthropic(), model="claude-sonnet-4-20250514")
    result = llm.generate("What is 2+2?")

    # Async
    from openai import AsyncOpenAI
    from openharness.backends import AsyncOpenAIBackend

    llm = AsyncOpenAIBackend(AsyncOpenAI(), model="gpt-4o")
    result = await llm.generate("What is 2+2?")

    # Streaming (token-level)
    from openharness.backends import OpenAIStreamingBackend

    llm = OpenAIStreamingBackend(OpenAI(), model="gpt-4o")
    for chunk in llm.stream("What is 2+2?"):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
from typing import Any, Generator, AsyncGenerator

logger = logging.getLogger(__name__)


# ── OpenAI Backend ───────────────────────────────────────────────


class OpenAIBackend:
    """Sync LLM backend for OpenAI-compatible APIs.

    Args:
        client: An ``openai.OpenAI`` client instance.
        model: Model name (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
        default_max_tokens: Default max_tokens when not overridden per call.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = "gpt-4o",
        default_max_tokens: int = 2000,
    ) -> None:
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class OpenAIStreamingBackend(OpenAIBackend):
    """OpenAI backend with token-level streaming support.

    Extends ``OpenAIBackend`` with a ``stream()`` method that yields
    text chunks as they arrive from the API.  The ``generate()`` method
    still works for non-streaming use.

    Usage::

        llm = OpenAIStreamingBackend(client, model="gpt-4o")

        # Streaming
        for chunk in llm.stream("prompt", system_prompt="..."):
            print(chunk, end="")

        # Non-streaming (inherited from OpenAIBackend)
        full = llm.generate("prompt")
    """

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Yield text chunks as they arrive from the API."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


# ── Anthropic Backend ────────────────────────────────────────────


class AnthropicBackend:
    """Sync LLM backend for the Anthropic Messages API.

    Args:
        client: An ``anthropic.Anthropic`` client instance.
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
        default_max_tokens: Default max_tokens when not overridden per call.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int = 2000,
    ) -> None:
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        # Anthropic returns a list of content blocks
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)


class AnthropicStreamingBackend(AnthropicBackend):
    """Anthropic backend with token-level streaming support.

    Usage::

        llm = AnthropicStreamingBackend(client, model="claude-sonnet-4-20250514")
        for chunk in llm.stream("prompt"):
            print(chunk, end="")
    """

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Yield text chunks as they arrive from the API."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


# ── Async OpenAI Backend ─────────────────────────────────────────


class AsyncOpenAIBackend:
    """Async LLM backend for OpenAI-compatible APIs.

    Args:
        client: An ``openai.AsyncOpenAI`` client instance.
        model: Model name (e.g. ``"gpt-4o"``).
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = "gpt-4o",
        default_max_tokens: int = 2000,
    ) -> None:
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks asynchronously."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content


# ── Async Anthropic Backend ──────────────────────────────────────


class AsyncAnthropicBackend:
    """Async LLM backend for the Anthropic Messages API.

    Args:
        client: An ``anthropic.AsyncAnthropic`` client instance.
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int = 2000,
    ) -> None:
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks asynchronously."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
