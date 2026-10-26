"""LLM backend adapters for popular API providers.

Ready-to-use adapters that satisfy the ``LLMBackend`` and ``AsyncLLMBackend``
protocols for OpenAI-compatible and Anthropic APIs.  Each adapter accepts
convenience ``base_url`` / ``api_key`` kwargs for quick setup, or a
pre-built provider client object for full control.

Typical usage::

    # OpenAI (convenience - auto-creates client)
    from looplet.backends import OpenAIBackend

    llm = OpenAIBackend(base_url="https://api.openai.com/v1",
                        api_key="sk-...", model="gpt-4o")

    # OpenAI (explicit client - full control)
    from openai import OpenAI
    llm = OpenAIBackend(OpenAI(), model="gpt-4o")

    # Anthropic
    from looplet.backends import AnthropicBackend

    llm = AnthropicBackend(api_key="sk-ant-...", model="claude-sonnet-4-20250514")

    # Async
    from looplet.backends import AsyncOpenAIBackend

    llm = AsyncOpenAIBackend(base_url="https://api.openai.com/v1",
                             api_key="sk-...", model="gpt-4o")

    # Streaming (token-level)
    from looplet.backends import OpenAIStreamingBackend
    from openai import OpenAI

    llm = OpenAIStreamingBackend(OpenAI(), model="gpt-4o")
    for chunk in llm.stream("What is 2+2?"):
        print(chunk, end="", flush=True)

    # Native tool calling (selected automatically by LoopConfig)
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

import copy
import json
import logging
import os
from typing import Any, AsyncGenerator, Generator

from looplet.backend_contract import (
    BackendRequestPolicy,
    TerminalLLMError,
    json_snapshot,
    single_model_request_active,
)

logger = logging.getLogger(__name__)


# ── max_tokens helper ────────────────────────────────────────────


def _resolve_max_tokens(
    per_call: int | None,
    default: int | None,
) -> int | None:
    """Resolve effective max_tokens for an API call.

    Priority: per-call value > backend default > None (let API decide).
    The loop passes max_tokens from LoopConfig; if a user never set it,
    it arrives as the LoopConfig default (2000). To preserve "let API
    decide" semantics, backends that set default_max_tokens=None will
    only send max_tokens when explicitly provided.
    """
    if per_call is not None and per_call > 0:
        return per_call
    return default


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


def _openai_message_text(message: Any) -> str:
    """Extract visible text from string or multipart compatible responses."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def _openai_message_to_blocks(message: Any) -> list[dict[str, Any]]:
    """Normalise an OpenAI chat.completion message to Anthropic-style blocks.

    Builds a list containing an optional ``{"type":"text"}`` block followed by
    zero or more ``{"type":"tool_use"}`` blocks.  JSON-decodes the
    ``function.arguments`` string into an ``input`` dict; on malformed JSON,
    keeps the raw string under ``_raw_arguments`` for debugging.
    """
    blocks: list[dict[str, Any]] = []
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning:
        blocks.append({"type": "reasoning", "text": reasoning})
    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal:
        blocks.append({"type": "refusal", "refusal": refusal})
    text = _openai_message_text(message)
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
        if isinstance(block, dict):
            blocks.append(copy.deepcopy(block))
            continue
        dump = getattr(block, "model_dump", None)
        if callable(dump):
            serialized = dump(mode="json")
            if isinstance(serialized, dict):
                blocks.append(copy.deepcopy(serialized))
                continue
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
        elif btype == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    "signature": getattr(block, "signature", ""),
                }
            )
        elif btype == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": getattr(block, "data", "")})
    return blocks


def _begin_request(backend: Any, request: dict[str, Any]) -> None:
    options = json_snapshot(getattr(backend, "_request_options", {}))
    request.update(options)
    if request.get("temperature") is None:
        request.pop("temperature", None)
    backend._applied_request_options = options
    backend.last_request = json_snapshot(request)
    backend.last_response = None
    backend.last_stop_reason = None
    backend.last_usage = {}


def _capture_response(backend: Any, response: Any) -> None:
    backend.last_response = json_snapshot(response)
    choices = getattr(response, "choices", None) or []
    reason = getattr(response, "stop_reason", None)
    if reason is None and choices:
        reason = getattr(choices[0], "finish_reason", None)
    backend.last_stop_reason = (
        {
            "length": "max_tokens",
            "end_turn": "stop",
            "tool_calls": "tool_use",
            "content_filter": "refusal",
        }.get(reason, reason)
        if isinstance(reason, str)
        else None
    )
    try:
        from looplet.cost import extract_usage

        backend.last_usage = extract_usage(response)
    except Exception:
        backend.last_usage = {}


# ── Prompt-cache translation ─────────────────────────────────────


def _anthropic_cache_control(ttl: str) -> dict[str, Any]:
    """Map a looplet ``CacheControl.ttl`` onto an Anthropic ``cache_control`` block.

    Anthropic marks a cacheable prefix with ``{"type": "ephemeral"}`` (5-min) or
    ``{"type": "ephemeral", "ttl": "1h"}`` (1-hour). Unknown ttl values fall back
    to the 5-min default.
    """
    block: dict[str, Any] = {"type": "ephemeral"}
    if ttl == "1h":
        block["ttl"] = "1h"
    return block


def _apply_anthropic_cache(
    kwargs: dict[str, Any],
    *,
    prompt: str,
    tools: list[dict[str, Any]] | None,
    cache_breakpoints: list[Any] | None,
) -> None:
    """Translate looplet cache breakpoints into Anthropic ``cache_control`` markers.

    Mutates ``kwargs`` in place, marking whichever stable sections the policy
    enabled:

    * ``system_prompt`` -> the ``system`` field becomes a single text block
      carrying ``cache_control`` (caches the tools + system prefix).
    * ``tool_schemas`` -> the **last** tool gains ``cache_control`` (Anthropic
      caches the whole tools array up to that marker).
    * ``memory`` -> the persistent memory prefix of the user message is split
      into its own ``cache_control``-marked text block.

    Sections absent from the policy, empty, or not locatable in the prompt are
    skipped so the request still succeeds -- caching is strictly additive.
    """
    if not cache_breakpoints:
        return
    by_label = {getattr(bp, "label", None): bp for bp in cache_breakpoints}

    def _ttl(label: str) -> str:
        return getattr(getattr(by_label.get(label), "control", None), "ttl", "ephemeral")

    # system_prompt -> cache_control on the system block
    system = kwargs.get("system")
    if "system_prompt" in by_label and isinstance(system, str) and system:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": _anthropic_cache_control(_ttl("system_prompt")),
            }
        ]

    # tool_schemas -> cache_control on the last tool (caches the whole array)
    if "tool_schemas" in by_label and tools:
        marked = [dict(t) for t in tools]
        marked[-1] = {
            **marked[-1],
            "cache_control": _anthropic_cache_control(_ttl("tool_schemas")),
        }
        kwargs["tools"] = marked

    # memory -> split the user message so its persistent prefix is cached
    mem_bp = by_label.get("memory")
    mem_content = getattr(mem_bp, "content", "") if mem_bp is not None else ""
    messages = kwargs.get("messages", [])
    if mem_content and messages and messages[0].get("role") == "user":
        first = messages[0]["content"]
        blocks = [{"type": "text", "text": first}] if isinstance(first, str) else list(first)
        for index, block in enumerate(blocks):
            text = block.get("text", "") if block.get("type") == "text" else ""
            split_at = text.find(mem_content)
            if split_at == -1:
                continue
            end = split_at + len(mem_content)
            replacements = [
                {
                    "type": "text",
                    "text": text[:end],
                    "cache_control": _anthropic_cache_control(_ttl("memory")),
                }
            ]
            if text[end:]:
                replacements.append({"type": "text", "text": text[end:]})
            kwargs["messages"] = [
                {**messages[0], "content": blocks[:index] + replacements + blocks[index + 1 :]},
                *messages[1:],
            ]
            break


def _openai_prompt_cache_key(cache_breakpoints: list[Any]) -> str:
    """Derive a stable OpenAI ``prompt_cache_key`` from looplet cache breakpoints.

    OpenAI caches prompt prefixes automatically; ``prompt_cache_key`` only steers
    routing so repeat requests that share the same stable prefix land on the same
    cache. The per-section fingerprints are already stable across turns, so we
    concatenate them into one deterministic key.
    """
    parts = [h for h in (getattr(bp, "hash", "") for bp in cache_breakpoints) if h]
    return "looplet-" + "-".join(parts)


# ── OpenAI Backend ───────────────────────────────────────────────


def _conversation_messages(
    backend: Any, prompt: str, *, native_blocks: bool
) -> list[dict[str, Any]]:
    snapshot = getattr(backend, "_conversation_input", None)
    if snapshot is None:
        return [{"role": "user", "content": prompt}]
    messages: list[dict[str, Any]] = [{"role": "user", "content": snapshot["prompt"]}]
    pending: set[str] = set()
    for message in snapshot["conversation"].get("messages", []):
        content = message.get("content", "")
        metadata = message.get("metadata", {})
        role = message["role"]
        result = message.get("tool_result")
        if result is not None:
            call_id = result.get("call_id") or metadata.get("tool_call_id")
            rendered = json.dumps({"data": result.get("data"), "error": result.get("error")})
            if call_id in pending:
                pending.remove(call_id)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": call_id, "content": rendered}
                        ],
                    }
                    if native_blocks
                    else {"role": "tool", "tool_call_id": call_id, "content": rendered}
                )
            else:
                if pending:
                    raise TerminalLLMError(
                        "Unpaired provider tool result", termination_reason="unsupported_context"
                    )
                messages.append(
                    {"role": "user", "content": f"[TOOL RESULT {result['tool']}] {rendered}"}
                )
            continue
        if metadata.get("kind") == "compaction_boundary" or role == "system":
            messages.append(
                {"role": "user", "content": "[PRIOR CONTEXT, NOT VERIFIED FACTS] " + str(content)}
            )
            continue
        if role != "user" and metadata.get("origin") not in {"model", "host"}:
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        blocks = [{"type": block["kind"], **block["data"]} for block in content]
        calls = [block for block in blocks if block.get("type") == "tool_use"]
        for call in calls:
            identifier = call.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in pending:
                raise TerminalLLMError(
                    "Invalid provider tool identifier", termination_reason="unsupported_context"
                )
            pending.add(identifier)
        if native_blocks:
            messages.append({"role": role, "content": blocks})
        else:
            if any(block.get("type") not in {"text", "tool_use"} for block in blocks):
                raise TerminalLLMError(
                    "Provider content cannot be replayed by this adapter",
                    termination_reason="unsupported_context",
                )
            translated: dict[str, Any] = {
                "role": role,
                "content": "".join(
                    block.get("text", "") for block in blocks if block.get("type") == "text"
                )
                or None,
            }
            if calls:
                translated["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["input"])},
                    }
                    for call in calls
                ]
            messages.append(translated)
    if pending:
        raise TerminalLLMError(
            "Provider tool calls lack results", termination_reason="unsupported_context"
        )
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "Continue the task."})
    if native_blocks:
        merged: list[dict[str, Any]] = []
        for message in messages:
            if merged and message["role"] == merged[-1]["role"]:
                previous = merged[-1]["content"]
                following = message["content"]
                merged[-1]["content"] = (
                    previous if isinstance(previous, list) else [{"type": "text", "text": previous}]
                ) + (
                    following
                    if isinstance(following, list)
                    else [{"type": "text", "text": following}]
                )
            else:
                merged.append(message)
        return merged
    return messages


class _ConversationBackend:
    _client: Any
    _model: str
    _provider: str
    conversation_mode = "messages"
    _native_blocks = False
    request_policy = BackendRequestPolicy(supports_single_model_request=True)

    @property
    def supported_request_options(self) -> frozenset[str]:
        """Adapter fields, not a claim of support by every model or endpoint."""
        if self._provider == "openai":
            return frozenset(
                {
                    "reasoning_effort",
                    "top_p",
                    "seed",
                    "service_tier",
                    "verbosity",
                    "response_format",
                }
            )
        return frozenset({"thinking", "top_p", "top_k", "output_config"})

    def configure_request_options(self, options: dict[str, Any] | None) -> None:
        """Explicitly bind provider-specific fields without overriding loop controls."""
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("request_options must be an object")
        if any(not isinstance(name, str) for name in options):
            raise ValueError("request_options must use string field names")
        unsupported = set(options) - self.supported_request_options
        if unsupported:
            raise ValueError(f"Unsupported {self._provider} request options: {sorted(unsupported)}")
        try:
            detached = json.loads(json.dumps(options, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("request_options must contain finite JSON values") from error
        if detached != options:
            raise ValueError("request_options must contain lossless JSON values")
        for name in ("reasoning_effort", "service_tier", "verbosity"):
            if name in detached and (
                not isinstance(detached[name], str) or not detached[name].strip()
            ):
                raise ValueError(f"{name} must be a nonempty provider-specific string")
        if "top_p" in detached and (
            type(detached["top_p"]) not in (int, float) or not 0 <= detached["top_p"] <= 1
        ):
            raise ValueError("top_p must be a number between 0 and 1")
        for name in ("seed", "top_k"):
            if name in detached and (
                type(detached[name]) is not int or (name == "top_k" and detached[name] < 1)
            ):
                raise ValueError(
                    f"{name} must be an integer" + (" greater than 0" if name == "top_k" else "")
                )
        for name in ("response_format", "output_config", "thinking"):
            if name in detached and not isinstance(detached[name], dict):
                raise ValueError(f"{name} must be an object")
        thinking = detached.get("thinking")
        if thinking is not None:
            mode = thinking.get("type")
            if not isinstance(mode, str) or mode not in {"enabled", "disabled", "adaptive"}:
                raise ValueError("thinking.type must be enabled, disabled, or adaptive")
            if mode == "enabled" and (
                type(thinking.get("budget_tokens")) is not int or thinking["budget_tokens"] < 1024
            ):
                raise ValueError("enabled thinking requires an integer budget_tokens >= 1024")
        self._request_options = detached
        self._applied_request_options = None

    @property
    def request_configuration(self) -> dict[str, Any]:
        """Detached requested/submitted fields; provider-confirmed settings are unknown."""
        return json_snapshot(
            {
                "provider": self._provider,
                "model": self._model,
                "requested": self._request_options,
                "applied": self._applied_request_options,
                "reported": None,
            }
        )

    def _request_client(self) -> Any:
        with_options = getattr(self._client, "with_options", None)
        return (
            with_options(max_retries=0)
            if single_model_request_active() and callable(with_options)
            else self._client
        )

    def set_conversation_input(self, snapshot: dict[str, Any] | None) -> None:
        detached = json_snapshot(snapshot)
        if detached is not None:
            from types import SimpleNamespace

            _conversation_messages(
                SimpleNamespace(_conversation_input=detached),
                detached["prompt"],
                native_blocks=self._native_blocks,
            )
        self._conversation_input = detached


class OpenAIBackend(_ConversationBackend):
    """Sync LLM backend for OpenAI-compatible APIs.

    Args:
        client: An ``openai.OpenAI`` client instance. Optional if
            ``base_url`` and ``api_key`` are provided instead.
        model: Model name (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
        default_max_tokens: Default max_tokens sent to the API.
            ``None`` (default) means "don't send max_tokens, let the
            provider decide." Set to an int to cap every call.
        base_url: Convenience shorthand - when provided without ``client``,
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

    _provider = "openai"

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "gpt-4o",
        default_max_tokens: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tool_choice: str = "auto",
        use_prompt_cache_key: bool = False,
        request_options: dict[str, Any] | None = None,
    ) -> None:
        self.configure_request_options(request_options)
        if client is None:
            from openai import OpenAI  # noqa: PLC0415

            kwargs: dict[str, Any] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if api_key is not None:
                kwargs["api_key"] = api_key
            # ``OpenAI()`` itself reads ``OPENAI_API_KEY``/``OPENAI_BASE_URL``
            # from the environment when neither is supplied, so passing
            # only ``model=...`` is enough for the canonical cloud path.
            client = OpenAI(**kwargs)
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._tool_choice = tool_choice
        self._use_prompt_cache_key = use_prompt_cache_key

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        default_max_tokens: int | None = None,
        tool_choice: str = "auto",
    ) -> "OpenAIBackend":
        """Build an :class:`OpenAIBackend` from environment variables.

        Reads ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` (optional, e.g.
        for a local proxy), and ``OPENAI_MODEL`` (optional, falls back
        to ``"gpt-4o"`` or the explicit ``model=`` arg).  Equivalent to
        the manual two-step::

            from openai import OpenAI
            llm = OpenAIBackend(OpenAI(), model=os.environ["OPENAI_MODEL"])

        Raises ``RuntimeError`` upfront when neither ``OPENAI_API_KEY``
        nor ``OPENAI_BASE_URL`` is set, instead of letting the OpenAI
        SDK fail later with a less actionable error.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key and not base_url:
            raise RuntimeError(
                "OpenAIBackend.from_env(): set OPENAI_API_KEY (cloud) "
                "or OPENAI_BASE_URL (local proxy / Ollama / vLLM). "
                "Both are missing in the environment."
            )
        # Local-server convention (Ollama / vLLM / llama.cpp): when
        # BASE_URL is set without a key, the SDK still requires a
        # non-empty string so default to a sentinel.
        if not api_key and base_url:
            api_key = "x"
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            default_max_tokens=default_max_tokens,
            tool_choice=tool_choice,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(_conversation_messages(self, prompt, native_blocks=False))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        _mt = _resolve_max_tokens(max_tokens, self._default_max_tokens)
        if _mt is not None:
            kwargs["max_tokens"] = _mt
        if self._use_prompt_cache_key and cache_breakpoints:
            kwargs["extra_body"] = {"prompt_cache_key": _openai_prompt_cache_key(cache_breakpoints)}

        _begin_request(self, kwargs)
        response = self._request_client().chat.completions.create(**kwargs)
        self._record_usage(response)
        return _openai_message_text(response.choices[0].message)

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Native tool calling via OpenAI ``tools`` parameter.

        Returns Anthropic-normalised content blocks so the loop can use
        ``parse_native_tool_use`` uniformly regardless of provider.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(_conversation_messages(self, prompt, native_blocks=False))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "tools": _to_openai_tools(tools),
            "tool_choice": self._tool_choice,
        }
        _mt = _resolve_max_tokens(max_tokens, self._default_max_tokens)
        if _mt is not None:
            kwargs["max_tokens"] = _mt
        if self._use_prompt_cache_key and cache_breakpoints:
            kwargs["extra_body"] = {"prompt_cache_key": _openai_prompt_cache_key(cache_breakpoints)}

        _begin_request(self, kwargs)
        response = self._request_client().chat.completions.create(**kwargs)
        self._record_usage(response)
        return _openai_message_to_blocks(response.choices[0].message)

    def _record_usage(self, response: Any) -> None:
        """Stash usage on ``self.last_usage`` for cost tracking.

        Imports ``looplet.cost.extract_usage`` lazily to avoid a hard
        dependency from backends ↔ cost.
        """
        _capture_response(self, response)


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
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Yield text chunks as they arrive from the API."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        _mt = _resolve_max_tokens(max_tokens, self._default_max_tokens)
        if _mt is not None:
            kwargs["max_tokens"] = _mt

        _begin_request(self, kwargs)
        response = self._request_client().chat.completions.create(**kwargs)
        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            text = _openai_message_text(delta)
            if text:
                yield text


# ── Anthropic Backend ────────────────────────────────────────────


class AnthropicBackend(_ConversationBackend):
    """Sync LLM backend for the Anthropic Messages API.

    Args:
        client: An ``anthropic.Anthropic`` client instance. Optional if
            ``api_key`` is provided instead.
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
        default_max_tokens: Default max_tokens when not overridden per call.
        api_key: Convenience shorthand - when provided without ``client``,
            an ``anthropic.Anthropic(api_key=...)`` client is created.

    Example (explicit client)::

        from anthropic import Anthropic
        llm = AnthropicBackend(Anthropic(), model="claude-sonnet-4-20250514")

    Example (convenience)::

        llm = AnthropicBackend(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    """

    _native_blocks = True
    _provider = "anthropic"

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int | None = None,
        api_key: str | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> None:
        self.configure_request_options(request_options)
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
        # See ``OpenAIBackend.last_usage``; same contract.
        self.last_usage: dict[str, int] = {}

    def _record_usage(self, response: Any) -> None:
        _capture_response(self, response)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        default_max_tokens: int | None = None,
    ) -> "AnthropicBackend":
        """Build an :class:`AnthropicBackend` from environment variables.

        Reads ``ANTHROPIC_API_KEY`` (required) and ``ANTHROPIC_MODEL``
        (optional, falls back to the default model or the explicit
        ``model=`` arg).

        Raises ``RuntimeError`` upfront when ``ANTHROPIC_API_KEY`` is
        missing, instead of failing later inside the constructor with
        a ``TypeError``.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AnthropicBackend.from_env(): ANTHROPIC_API_KEY is not set in the environment."
            )
        return cls(
            api_key=api_key,
            model=model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            default_max_tokens=default_max_tokens,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _conversation_messages(self, prompt, native_blocks=True),
            "max_tokens": _resolve_max_tokens(max_tokens, self._default_max_tokens) or 4096,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        _apply_anthropic_cache(
            kwargs, prompt=prompt, tools=None, cache_breakpoints=cache_breakpoints
        )

        _begin_request(self, kwargs)
        response = self._request_client().messages.create(**kwargs)
        self._record_usage(response)
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
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Native tool calling via Anthropic ``tools`` parameter."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _conversation_messages(self, prompt, native_blocks=True),
            "max_tokens": _resolve_max_tokens(max_tokens, self._default_max_tokens) or 4096,
            "temperature": temperature,
            "tools": tools,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        _apply_anthropic_cache(
            kwargs, prompt=prompt, tools=tools, cache_breakpoints=cache_breakpoints
        )

        _begin_request(self, kwargs)
        response = self._request_client().messages.create(**kwargs)
        self._record_usage(response)
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
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Yield text chunks as they arrive from the API."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _resolve_max_tokens(max_tokens, self._default_max_tokens) or 4096,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        _begin_request(self, kwargs)
        with self._request_client().messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


# ── Async OpenAI Backend ─────────────────────────────────────────


class AsyncOpenAIBackend(_ConversationBackend):
    """Async LLM backend for OpenAI-compatible APIs.

    Args:
        client: An ``openai.AsyncOpenAI`` client instance. Optional if
            ``base_url`` and ``api_key`` are provided instead.
        model: Model name (e.g. ``"gpt-4o"``).
        base_url: Convenience shorthand - auto-creates an AsyncOpenAI client.
        api_key: API key for the auto-created client.
    """

    _provider = "openai"

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "gpt-4o",
        default_max_tokens: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tool_choice: str = "auto",
        use_prompt_cache_key: bool = False,
        request_options: dict[str, Any] | None = None,
    ) -> None:
        self.configure_request_options(request_options)
        if client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            kwargs: dict[str, Any] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if api_key is not None:
                kwargs["api_key"] = api_key
            client = AsyncOpenAI(**kwargs)
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._tool_choice = tool_choice
        self._use_prompt_cache_key = use_prompt_cache_key
        # See ``OpenAIBackend.last_usage``; same contract.
        self.last_usage: dict[str, int] = {}

    def _record_usage(self, response: Any) -> None:
        _capture_response(self, response)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        default_max_tokens: int | None = None,
        tool_choice: str = "auto",
    ) -> "AsyncOpenAIBackend":
        """Async sibling of :meth:`OpenAIBackend.from_env`.

        Raises ``RuntimeError`` upfront when neither ``OPENAI_API_KEY``
        nor ``OPENAI_BASE_URL`` is set.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key and not base_url:
            raise RuntimeError(
                "AsyncOpenAIBackend.from_env(): set OPENAI_API_KEY "
                "(cloud) or OPENAI_BASE_URL (local proxy / Ollama / "
                "vLLM). Both are missing in the environment."
            )
        if not api_key and base_url:
            api_key = "x"
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            default_max_tokens=default_max_tokens,
            tool_choice=tool_choice,
        )

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(_conversation_messages(self, prompt, native_blocks=False))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        _mt = _resolve_max_tokens(max_tokens, self._default_max_tokens)
        if _mt is not None:
            kwargs["max_tokens"] = _mt
        if self._use_prompt_cache_key and cache_breakpoints:
            kwargs["extra_body"] = {"prompt_cache_key": _openai_prompt_cache_key(cache_breakpoints)}

        _begin_request(self, kwargs)
        response = await self._request_client().chat.completions.create(**kwargs)
        self._record_usage(response)
        return _openai_message_text(response.choices[0].message)

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async native tool calling via OpenAI ``tools`` parameter."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(_conversation_messages(self, prompt, native_blocks=False))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "tools": _to_openai_tools(tools),
            "tool_choice": self._tool_choice,
        }
        _mt = _resolve_max_tokens(max_tokens, self._default_max_tokens)
        if _mt is not None:
            kwargs["max_tokens"] = _mt
        if self._use_prompt_cache_key and cache_breakpoints:
            kwargs["extra_body"] = {"prompt_cache_key": _openai_prompt_cache_key(cache_breakpoints)}

        _begin_request(self, kwargs)
        response = await self._request_client().chat.completions.create(**kwargs)
        self._record_usage(response)
        return _openai_message_to_blocks(response.choices[0].message)

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks asynchronously."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        _mt = _resolve_max_tokens(max_tokens, self._default_max_tokens)
        if _mt is not None:
            kwargs["max_tokens"] = _mt

        _begin_request(self, kwargs)
        response = await self._request_client().chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            text = _openai_message_text(delta)
            if text:
                yield text


# ── Async Anthropic Backend ──────────────────────────────────────


class AsyncAnthropicBackend(_ConversationBackend):
    """Async LLM backend for the Anthropic Messages API.

    Args:
        client: An ``anthropic.AsyncAnthropic`` client instance.
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
    """

    _native_blocks = True
    _provider = "anthropic"

    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int | None = None,
        request_options: dict[str, Any] | None = None,
    ) -> None:
        self.configure_request_options(request_options)
        self._client = client
        self._model = model
        self._default_max_tokens = default_max_tokens

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _conversation_messages(self, prompt, native_blocks=True),
            "max_tokens": _resolve_max_tokens(max_tokens, self._default_max_tokens) or 4096,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        _apply_anthropic_cache(
            kwargs, prompt=prompt, tools=None, cache_breakpoints=cache_breakpoints
        )

        _begin_request(self, kwargs)
        response = await self._request_client().messages.create(**kwargs)
        _capture_response(self, response)
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
        temperature: float | None = None,
        cache_breakpoints: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async native tool calling via Anthropic ``tools`` parameter."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _conversation_messages(self, prompt, native_blocks=True),
            "max_tokens": _resolve_max_tokens(max_tokens, self._default_max_tokens) or 4096,
            "temperature": temperature,
            "tools": tools,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        _apply_anthropic_cache(
            kwargs, prompt=prompt, tools=tools, cache_breakpoints=cache_breakpoints
        )

        _begin_request(self, kwargs)
        response = await self._request_client().messages.create(**kwargs)
        _capture_response(self, response)
        return _anthropic_response_to_blocks(response)

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks asynchronously."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": _resolve_max_tokens(max_tokens, self._default_max_tokens) or 4096,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        _begin_request(self, kwargs)
        async with self._request_client().messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text


# ── Turnkey factory for the RPC `set_backend` wire ───────────────────


class ProviderSelectionWarning(UserWarning):
    """Multiple configured providers need an explicit selection."""


def make_backend(
    *,
    provider: str | None = None,
    model: str | None = None,
    request_options: dict[str, Any] | None = None,
) -> Any:
    """Build an :class:`LLMBackend` from the environment.

    A module-level factory so a non-Python orchestrator can wire a real
    backend over the RPC ``set_backend`` command without writing glue::

        {"cmd": "set_backend", "factory": "looplet.backends:make_backend",
         "kwargs": {"model": "claude-sonnet-4-20250514"}}

    ``provider`` selects the adapter; when omitted it is read from
    ``LOOPLET_PROVIDER`` and finally auto-detected from whichever API key is
    present (``ANTHROPIC_API_KEY`` → anthropic, ``OPENAI_API_KEY`` → openai).
    ``model`` overrides the provider's env/default model.
    ``request_options`` explicitly binds adapter-specific SDK fields. Cartridge
    model metadata remains nonbinding unless the host supplies these options.

    Raises ``ValueError`` with an actionable message when no provider can be
    resolved, so the orchestrator gets a clear ``error`` frame instead of an
    opaque import failure.
    """
    import os

    def configured(backend: Any) -> Any:
        if request_options is not None:
            configure = getattr(backend, "configure_request_options", None)
            if not callable(configure):
                raise ValueError("Selected backend does not support explicit request_options")
            configure(request_options)
        return backend

    if (
        not (provider or os.environ.get("LOOPLET_PROVIDER"))
        and os.environ.get("ANTHROPIC_API_KEY")
        and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))
    ):
        import warnings

        warnings.warn(
            "Multiple providers are configured. Selecting Anthropic by default. "
            "Set LOOPLET_PROVIDER=openai or LOOPLET_PROVIDER=anthropic to choose explicitly.",
            ProviderSelectionWarning,
            stacklevel=2,
        )

    prov = (
        provider
        or os.environ.get("LOOPLET_PROVIDER")
        or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "")
        or (
            "openai"
            if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL")
            else ""
        )
    ).lower()
    if prov in ("anthropic", "claude"):
        return configured(AnthropicBackend.from_env(model=model))
    if prov in ("openai", "oai"):
        return configured(OpenAIBackend.from_env(model=model))
    # OpenAI-compatible proxy (Copilot proxy / local endpoint)
    # Any host with these set drives over RPC
    # with no glue, no provider= needed.
    base = os.environ.get("COPILOT_PROXY_URL") or os.environ.get("LOOPLET_TAX_LLM_BASE_URL")
    key = os.environ.get("COPILOT_PROXY_KEY") or os.environ.get("LOOPLET_TAX_LLM_API_KEY")
    if prov in ("copilot", "proxy") or base or key:
        return configured(
            OpenAIBackend(
                base_url=base,
                api_key=key,
                model=model or os.environ.get("LOOPLET_TAX_LLM_MODEL") or "gpt-4o",
            )
        )
    # Keyless fallback (opt-in): a deterministic mock so the RPC path is runnable
    # on ANY repo/CI without secrets. Off by default so prod fails loud.
    if os.environ.get("LOOPLET_ALLOW_MOCK", "").lower() in ("1", "true", "yes"):
        import json  # noqa: PLC0415

        from looplet.testing import MockLLMBackend  # noqa: PLC0415

        return configured(
            MockLLMBackend(
                responses=[
                    json.dumps(
                        {"tool": "done", "args": {"summary": "keyless mock"}, "reasoning": "mock"}
                    )
                ],
                cycle=True,
            )
        )
    raise ValueError(
        "make_backend(): could not resolve a provider. Pass provider='anthropic'|'openai' "
        "(or kwargs.provider over RPC), set LOOPLET_PROVIDER, export ANTHROPIC_API_KEY / "
        "OPENAI_API_KEY (or OPENAI_BASE_URL / COPILOT_PROXY_URL for a proxy), or set "
        "LOOPLET_ALLOW_MOCK=1 for a keyless deterministic mock."
    )
