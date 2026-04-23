"""LLM backend adapters for popular API providers.

Ready-to-use adapters that satisfy the ``LLMBackend`` and ``AsyncLLMBackend``
protocols for OpenAI-compatible and Anthropic APIs.  Each adapter accepts the
provider's native client object so looplet stays dependency-free — import the
SDK you already use and pass the client in.

Typical usage::

    # OpenAI
    from openai import OpenAI
    from looplet.backends import OpenAIBackend

    llm = OpenAIBackend(OpenAI(), model="gpt-4o")
    result = llm.generate("What is 2+2?")

    # Anthropic
    from anthropic import Anthropic
    from looplet.backends import AnthropicBackend

    llm = AnthropicBackend(Anthropic(), model="claude-sonnet-4-20250514")
    result = llm.generate("What is 2+2?")

    # Async
    from openai import AsyncOpenAI
    from looplet.backends import AsyncOpenAIBackend

    llm = AsyncOpenAIBackend(AsyncOpenAI(), model="gpt-4o")
    result = await llm.generate("What is 2+2?")

    # Streaming (token-level)
    from looplet.backends import OpenAIStreamingBackend

    llm = OpenAIStreamingBackend(OpenAI(), model="gpt-4o")
    for chunk in llm.stream("What is 2+2?"):
        print(chunk, end="", flush=True)

    # Native tool calling (gate with LOOPLET_NATIVE_TOOLS=1)
    schemas = [
        {"name": "get_weather",
         "description": "Get current weather",
         "input_schema": {"type": "object",
                          "properties": {"city": {"type": "string"}}}},
    ]
    blocks = llm.generate_with_tools("Weather in Paris?", tools=schemas)
    # → [{"type": "tool_use", "id": "...", "name": "get_weather",
    #     "input": {"city": "Paris"}}]
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Generator

logger = logging.getLogger(__name__)


# ── Tool-schema translation helpers ──────────────────────────────


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style schemas to OpenAI function-tool format."""
    out: list[dict[str, Any]] = []
    for s in schemas:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


def _openai_message_to_blocks(message: Any) -> list[dict[str, Any]]:
    """Normalise an OpenAI chat.completion message to Anthropic-style blocks.

    Builds a list containing an optional ``{"type":"text"}`` block followed by
    zero or more ``{"type":"tool_use"}`` blocks.  JSON-decodes the
    ``function.arguments`` string into an ``input`` dict; on malformed JSON,
    keeps the raw string under ``_raw_arguments`` for debugging.
    """
    blocks: list[dict[str, Any]] = []
    text = getattr(message, "content", None)
    if text:
        blocks.append({"type": "text", "text": text})

    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        raw_args = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            input_args = json.loads(raw_args) if raw_args else {}
            if not isinstance(input_args, dict):
                input_args = {}
        except (ValueError, TypeError):
            input_args = {"_raw_arguments": raw_args}
        blocks.append(
            {
                "type": "tool_use",
                "id": getattr(tc, "id", "") or "",
                "name": name,
                "input": input_args,
            }
        )
    return blocks


def _anthropic_response_to_blocks(response: Any) -> list[dict[str, Any]]:
    """Normalise an Anthropic messages.create response to plain-dict blocks."""
    blocks: list[dict[str, Any]] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            blocks.append({"type": "text", "text": getattr(block, "text", "")})
        elif btype == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", "") or "",
                    "name": getattr(block, "name", "") or "",
                    "input": dict(getattr(block, "input", {}) or {}),
                }
            )
    return blocks


# ── OpenAI Backend ───────────────────────────────────────────────


class OpenAIBackend:
    """Sync LLM backend for OpenAI-compatible APIs.

    Args:
        client: An ``openai.OpenAI`` client instance. Optional if
            ``base_url`` and ``api_key`` are provided instead.
        model: Model name (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
        default_max_tokens: Default max_tokens when not overridden per call.
        base_url: Convenience shorthand — when provided without ``client``,
            an ``openai.OpenAI(base_url=..., api_key=...)`` client is
            created automatically. Most users hit this friction point
            immediately: ``OpenAIBackend(base_url="...", api_key="...")``
            is the natural first attempt but used to fail.
        api_key: API key for the auto-created client.

    Example (explicit client)::

        from openai import OpenAI
        llm = OpenAIBackend(OpenAI(), model="gpt-4o")

    Example (convenience)::

        llm = OpenAIBackend(base_url="http://localhost:8080/v1",
                            api_key="x", model="gpt-4o")
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "gpt-4o",
        default_max_tokens: int = 2000,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if client is None:
            if base_url is None:
                raise TypeError(
                    "OpenAIBackend requires either a client instance as the "
                    "first argument or base_url=... (and optionally api_key=...) "
                    "to auto-create one. Example:\n"
                    '  OpenAIBackend(base_url="http://localhost:8080/v1", api_key="x")\n'
                    '  OpenAIBackend(OpenAI(), model="gpt-4o")'
                )
            from openai import OpenAI  # noqa: PLC0415

            client = OpenAI(base_url=base_url, api_key=api_key or "")
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

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Native tool calling via OpenAI ``tools`` parameter.

        Returns Anthropic-normalised content blocks so the loop can use
        ``parse_native_tool_use`` uniformly regardless of provider.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
            tools=_to_openai_tools(tools),
            tool_choice="auto",
        )
        return _openai_message_to_blocks(response.choices[0].message)


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
        client: An ``anthropic.Anthropic`` client instance. Optional if
            ``api_key`` is provided instead.
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
        default_max_tokens: Default max_tokens when not overridden per call.
        api_key: Convenience shorthand — when provided without ``client``,
            an ``anthropic.Anthropic(api_key=...)`` client is created.

    Example (explicit client)::

        from anthropic import Anthropic
        llm = AnthropicBackend(Anthropic(), model="claude-sonnet-4-20250514")

    Example (convenience)::

        llm = AnthropicBackend(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int = 2000,
        api_key: str | None = None,
    ) -> None:
        if client is None:
            if api_key is None:
                raise TypeError(
                    "AnthropicBackend requires either a client instance as the "
                    "first argument or api_key=... to auto-create one. Example:\n"
                    '  AnthropicBackend(api_key="sk-ant-...")\n'
                    '  AnthropicBackend(Anthropic(), model="claude-sonnet-4-20250514")'
                )
            from anthropic import Anthropic  # noqa: PLC0415

            client = Anthropic(api_key=api_key)
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

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Native tool calling via Anthropic ``tools`` parameter."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return _anthropic_response_to_blocks(response)


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
        client: An ``openai.AsyncOpenAI`` client instance. Optional if
            ``base_url`` and ``api_key`` are provided instead.
        model: Model name (e.g. ``"gpt-4o"``).
        base_url: Convenience shorthand — auto-creates an AsyncOpenAI client.
        api_key: API key for the auto-created client.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "gpt-4o",
        default_max_tokens: int = 2000,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if client is None:
            if base_url is None:
                raise TypeError(
                    "AsyncOpenAIBackend requires either a client instance or "
                    "base_url=... to auto-create one."
                )
            from openai import AsyncOpenAI  # noqa: PLC0415

            client = AsyncOpenAI(base_url=base_url, api_key=api_key or "")
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

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Async native tool calling via OpenAI ``tools`` parameter."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
            tools=_to_openai_tools(tools),
            tool_choice="auto",
        )
        return _openai_message_to_blocks(response.choices[0].message)

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

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Async native tool calling via Anthropic ``tools`` parameter."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        return _anthropic_response_to_blocks(response)

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
