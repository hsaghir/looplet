"""Utilities for detecting native tool-calling support.

OpenAI-compatible proxies sometimes expose a ``generate_with_tools``
method or accept a ``tools`` parameter while still returning plain text.
These helpers probe the actual behavior so agents can choose native
``tool_use`` blocks only when the backend proves it supports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeToolProbeResult:
    """Result of a native-tool protocol probe."""

    supported: bool
    """True when the backend emitted a matching ``tool_use`` block."""

    reason: str
    """Short human-readable explanation for logs or doctor output."""

    raw_response: Any = None
    """Raw backend response, useful for debugging proxy mismatches."""


def probe_native_tool_support(
    llm: Any,
    *,
    tool_name: str = "test_probe",
    max_tokens: int = 50,
) -> NativeToolProbeResult:
    """Probe whether ``llm`` actually returns native tool-use blocks.

    The check is behavioral, not just structural: it calls
    ``generate_with_tools`` with a tiny no-argument probe tool and
    accepts support only when the response contains a dict block shaped
    like ``{"type": "tool_use", "name": tool_name, ...}``.
    """
    if not hasattr(llm, "generate_with_tools"):
        return NativeToolProbeResult(False, "backend has no generate_with_tools method")

    tools = [
        {
            "name": tool_name,
            "description": "Probe tool for native tool-calling support.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]
    try:
        blocks = llm.generate_with_tools(
            f"Call the {tool_name} tool now.",
            tools=tools,
            max_tokens=max_tokens,
            system_prompt="",
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        return NativeToolProbeResult(
            False, f"generate_with_tools raised {type(exc).__name__}: {exc}"
        )

    if not isinstance(blocks, list):
        return NativeToolProbeResult(
            False,
            f"generate_with_tools returned {type(blocks).__name__}, not content blocks",
            blocks,
        )

    for block in blocks:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == tool_name
        ):
            return NativeToolProbeResult(True, "backend returned a matching tool_use block", blocks)

    return NativeToolProbeResult(
        False,
        "generate_with_tools returned no matching tool_use block; use JSON-text fallback",
        blocks,
    )


def supports_native_tools(llm: Any, *, tool_name: str = "test_probe") -> bool:
    """Return True when ``probe_native_tool_support`` succeeds."""
    return probe_native_tool_support(llm, tool_name=tool_name).supported
