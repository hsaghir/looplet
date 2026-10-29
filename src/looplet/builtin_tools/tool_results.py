"""Bounded, read-only access to host-retained tool evidence."""

from __future__ import annotations

import json
import re
from typing import Any

from looplet.tools import ToolSpec
from looplet.types import ToolContext


def _integer(value: int, name: str, *, maximum: int | None = None) -> None:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(
            f"{name} must be a nonnegative integer"
            + (f" no greater than {maximum}" if maximum is not None else "")
        )


def _text(data: Any, pointer: str) -> str:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("json_pointer must be empty or start with /")
    if pointer:
        for encoded in pointer[1:].split("/"):
            if re.search(r"~(?:[^01]|$)", encoded):
                raise ValueError("Invalid JSON Pointer escape")
            token = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(data, dict):
                data = data[token]
            elif isinstance(data, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
                data = data[int(token)]
            else:
                raise ValueError("JSON Pointer does not address a retained value")
    return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)


def list_results(
    results: dict[str, Any], *, offset: int, limit: int, end_offset: int | None = None
) -> dict[str, Any]:
    _integer(offset, "offset")
    _integer(limit, "limit", maximum=20)
    if end_offset is not None:
        _integer(end_offset, "end_offset")
    if not limit:
        raise ValueError("limit must be positive")
    keys = tuple(results)
    boundary = len(keys) if end_offset is None else min(len(keys), end_offset)
    entries = []
    for key in keys[offset : min(offset + limit, boundary)]:
        data = results[key]
        entry: dict[str, Any] = {"result_key": key, "type": type(data).__name__}
        if isinstance(data, (str, list, dict)):
            entry["size"] = len(data)
        entries.append(entry)
    end = offset + len(entries)
    return {
        "results": entries,
        "total_results": boundary,
        "end_offset": boundary,
        "next_offset": end if end < boundary else None,
    }


def read_result(
    data: Any, *, result_key: str, json_pointer: str, offset: int, limit: int
) -> dict[str, Any]:
    _integer(offset, "offset")
    _integer(limit, "limit", maximum=8000)
    if not limit:
        raise ValueError("limit must be positive")
    text = _text(data, json_pointer)
    content = text[offset : offset + limit]
    end = offset + len(content)
    return {
        "result_key": result_key,
        "json_pointer": json_pointer,
        "offset": offset,
        "offset_unit": "characters",
        "total_chars": len(text),
        "content": content,
        "next_offset": end if end < len(text) else None,
    }


def search_result(
    data: Any, *, result_key: str, query: str, json_pointer: str, offset: int, limit: int
) -> dict[str, Any]:
    _integer(offset, "offset")
    _integer(limit, "limit", maximum=10)
    if not limit or not isinstance(query, str) or not 1 <= len(query) <= 256:
        raise ValueError("Use a positive limit and a literal query of 1 to 256 characters")
    text = _text(data, json_pointer)
    matches = []
    cursor = offset
    while len(matches) < limit:
        found = text.find(query, cursor)
        if found < 0:
            break
        matches.append(
            {
                "offset": found,
                "excerpt": text[max(0, found - 80) : found + len(query) + 80],
            }
        )
        cursor = found + len(query)
    return {
        "result_key": result_key,
        "json_pointer": json_pointer,
        "offset_unit": "characters",
        "total_chars": len(text),
        "matches": matches,
        "next_offset": cursor if text.find(query, cursor) >= 0 else None,
    }


def _store(ctx: ToolContext) -> Any:
    store = ctx.resources.get("tool_result_store")
    if store is None:
        raise PermissionError("No retained tool-result store is available")
    return store


def _list(
    *, offset: int = 0, limit: int = 10, end_offset: int | None = None, ctx: ToolContext
) -> dict[str, Any]:
    if end_offset is None:
        return _store(ctx).list_results(offset=offset, limit=limit)
    return _store(ctx).list_results(offset=offset, limit=limit, end_offset=end_offset)


def _read(
    *,
    result_key: str,
    json_pointer: str = "",
    offset: int = 0,
    limit: int = 2000,
    ctx: ToolContext,
) -> dict[str, Any]:
    return _store(ctx).read_result(
        result_key, json_pointer=json_pointer, offset=offset, limit=limit
    )


def _search(
    *,
    result_key: str,
    query: str,
    json_pointer: str = "",
    offset: int = 0,
    limit: int = 5,
    ctx: ToolContext,
) -> dict[str, Any]:
    return _store(ctx).search_result(
        result_key, query, json_pointer=json_pointer, offset=offset, limit=limit
    )


_OFFSET = {
    "type": "integer",
    "minimum": 0,
    "default": 0,
    "description": "Zero-based character offset.",
}
_KEY = {"type": "string", "description": "Exact retained result_key from a tool result or listing."}
_POINTER = {
    "type": "string",
    "default": "",
    "description": (
        "RFC 6901 JSON Pointer into the original tool return value, not its preview wrapper. "
        "Empty selects the whole original value; /stdout selects an original stdout field."
    ),
}

SPECS = (
    ToolSpec(
        name="list_tool_results",
        description=(
            "List retained source references. Continue with next_offset and the returned "
            "end_offset to keep the catalog boundary fixed while new results arrive."
        ),
        parameters={
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Zero-based result offset.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
            "end_offset": {
                "type": ["integer", "null"],
                "minimum": 0,
                "default": None,
                "description": "Keep the first page's end_offset on subsequent pages.",
            },
        },
        execute=_list,
        requires=["tool_result_store"],
        concurrent_safe=True,
        result_is_view=True,
    ),
    ToolSpec(
        name="read_tool_result",
        description="Read a character page of a retained result or JSON field. Continue with next_offset.",
        parameters={
            "result_key": _KEY,
            "json_pointer": _POINTER,
            "offset": _OFFSET,
            "limit": {"type": "integer", "minimum": 1, "maximum": 8000, "default": 2000},
        },
        execute=_read,
        requires=["tool_result_store"],
        concurrent_safe=True,
        result_is_view=True,
    ),
    ToolSpec(
        name="search_tool_result",
        description="Find literal text in a retained result and return character offsets with bounded excerpts.",
        parameters={
            "result_key": _KEY,
            "query": {"type": "string", "minLength": 1, "maxLength": 256},
            "json_pointer": _POINTER,
            "offset": _OFFSET,
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        execute=_search,
        requires=["tool_result_store"],
        concurrent_safe=True,
        result_is_view=True,
    ),
)
