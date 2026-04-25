"""JSON parsing for LLM tool-call responses.

Handles common LLM quirks: markdown code fences, extra text
before/after JSON, single-tool and multi-tool batch formats.
Domain-agnostic — works for any agent.
"""

from __future__ import annotations

import json
import re
from typing import Any

from looplet.types import ToolCall


def to_text(raw: str | list[Any] | None) -> str | None:
    """Coerce an LLM response into plain text.

    Backends returning native tool-use blocks produce a ``list`` of
    content blocks (e.g. Anthropic's ``{"type": "text"|"tool_use", ...}``).
    This helper extracts and joins the ``text`` blocks so callers that
    only handle plain text get a usable string. Returns ``None`` if
    ``raw`` is ``None`` and ``""`` if no text blocks are present.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    parts: list[str] = []
    for block in raw:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def parse_tool_call(raw: str) -> ToolCall | None:
    """Parse a JSON tool call from LLM output.

    Handles markdown code fences, extra text before/after JSON.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    parsed = _try_parse_json(text)
    if parsed is not None:
        return _dict_to_tool_call(parsed)

    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        parsed = _try_parse_json(match.group())
        if parsed is not None:
            return _dict_to_tool_call(parsed)

    return None


def parse_multi_tool_calls(raw: "str | list[Any] | None") -> list[ToolCall]:
    """Parse multiple tool calls from a single LLM response.

    Accepts either a plain string or a list of native content blocks
    (native-tool path). List inputs are flattened to text via
    :func:`to_text` before parsing.

    Supports these formats:
    1. Single tool:  {"tool": "name", "args": {...}, "reasoning": "..."}
    2. Multi-tool:   {"tools": [{"tool": "name", ...}, ...], "theory": "..."}
    3. Markdown fenced JSON (```json ... ```)
    4. Extra surrounding text before/after the JSON object
    5. Malformed JSON — falls back to regex extraction, then returns []
    """
    text_raw = to_text(raw)
    if not text_raw or not text_raw.strip():
        return []

    text = text_raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    parsed = _try_parse_json(text)
    if parsed is None:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            parsed = _try_parse_json(match.group())
    if parsed is None:
        return []

    # Multi-tool format
    if "tools" in parsed and isinstance(parsed["tools"], list):
        theory = parsed.get("theory", "")
        reasoning = parsed.get("reasoning", "")
        calls: list[ToolCall] = []
        for item in parsed["tools"]:
            if isinstance(item, dict) and "tool" in item:
                args = item.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                if theory:
                    args["__theory__"] = theory
                calls.append(
                    ToolCall(
                        tool=str(item["tool"]),
                        args=args,
                        reasoning=str(item.get("reasoning", reasoning)),
                    )
                )
        return calls

    # Single tool format
    tc = _dict_to_tool_call(parsed)
    return [tc] if tc else []


def _try_parse_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Retry with literal newlines/tabs escaped — LLMs frequently put
    # raw newlines inside JSON string values (especially in edit_file
    # old_string/new_string with multi-line code).
    try:
        fixed = _escape_literal_newlines(text)
        if fixed != text:
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _escape_literal_newlines(text: str) -> str:
    """Escape literal newlines/tabs inside JSON string values.

    Walks the text character-by-character, tracking whether we're
    inside a JSON string (between unescaped quotes). When a literal
    newline or tab appears inside a string, replaces it with the
    proper JSON escape sequence.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_string:
            # Escaped character — copy both backslash and next char
            result.append(ch)
            if i + 1 < len(text):
                i += 1
                result.append(text[i])
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if in_string:
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
            else:
                result.append(ch)
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _dict_to_tool_call(d: dict) -> ToolCall | None:
    tool = d.get("tool") or d.get("name")
    if not tool:
        return None
    # Try multiple common arg-key names the model might use.
    args = d.get("args") or d.get("input") or d.get("parameters") or {}
    if isinstance(args, str):
        # LLM sent a bare string instead of a dict — stash it under
        # "_raw_arg" so dispatch can still see it (and the validation
        # error will show what was provided).  Common with simple
        # single-param tools where the model skips the key.
        args = {"_raw_arg": args}
    elif not isinstance(args, dict):
        args = {}
    # Flat args: model put parameters as siblings of "tool" instead of
    # nesting them under "args".  Detect by checking if the dict has
    # keys beyond the known meta-keys.
    if not args:
        _meta_keys = {
            "tool",
            "name",
            "args",
            "input",
            "parameters",
            "reasoning",
            "theory",
            "thought",
            "thinking",
        }
        flat = {k: v for k, v in d.items() if k not in _meta_keys}
        if flat:
            args = flat
    theory = d.get("theory", "")
    if theory:
        args["__theory__"] = theory
    return ToolCall(
        tool=str(tool),
        args=args,
        reasoning=str(d.get("reasoning", d.get("thinking", d.get("thought", "")))),
    )


# ── Native Tool Calling ─────────────────────────────────────────


def parse_native_tool_use(blocks: list[dict]) -> list[ToolCall]:
    """Parse API tool_use content blocks into ToolCalls.

    For use with the Anthropic API's native tool_use protocol,
    where the model returns structured tool_use blocks instead of
    free-text JSON. Eliminates parse failures entirely.

    Each block: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
    """
    calls: list[ToolCall] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        input_args = block.get("input", {})
        if not name:
            continue
        calls.append(
            ToolCall(
                tool=str(name),
                args=dict(input_args) if isinstance(input_args, dict) else {},
                reasoning="",  # native tool_use doesn't include reasoning
                call_id=block.get("id") or "",
            )
        )
    return calls
