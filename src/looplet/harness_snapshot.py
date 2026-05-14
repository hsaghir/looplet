"""Stable, JSON-friendly harness snapshots for provenance metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from looplet.loop import LoopConfig
    from looplet.tools import BaseToolRegistry

__all__ = ["serialize_harness"]

_MAX_SYSTEM_PROMPT_CHARS = 4000
_MISSING = object()


def _truncate_system_prompt(prompt: str) -> str:
    if len(prompt) <= _MAX_SYSTEM_PROMPT_CHARS:
        return prompt
    keep = _MAX_SYSTEM_PROMPT_CHARS - 40
    return prompt[:keep] + f"\n... [truncated {len(prompt) - keep} chars] ..."


def _getattr_present(obj: Any, name: str) -> Any:
    value = getattr(obj, name, _MISSING)
    if value is None:
        return _MISSING
    return value


def _tool_specs(tools: Any) -> list[Any]:
    specs = getattr(tools, "_specs", None)
    if specs is None:
        specs = getattr(tools, "_tools", None)
    if specs is None:
        return []
    values = specs.values() if hasattr(specs, "values") else specs
    return list(values)


def serialize_harness(
    *,
    config: "LoopConfig | None" = None,
    hooks: list[Any] | None = None,
    tools: "BaseToolRegistry | None" = None,
    memory_sources: list[Any] | None = None,
    llm: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable, JSON-friendly snapshot of the agent harness."""
    snapshot: dict[str, Any] = {
        "schema_version": 2,
        "extra": dict(extra) if extra is not None else {},
    }

    if config is not None:
        system_prompt = _getattr_present(config, "system_prompt")
        if system_prompt is not _MISSING:
            snapshot["system_prompt"] = _truncate_system_prompt(str(system_prompt))
        for key in (
            "max_steps",
            "max_tokens",
            "temperature",
            "use_native_tools",
            "concurrent_dispatch",
            "done_tool",
        ):
            value = _getattr_present(config, key)
            if value is not _MISSING:
                snapshot[key] = value

    if tools is not None:
        snapshot["tools"] = [
            {
                "name": getattr(spec, "name"),
                "description": getattr(spec, "description", ""),
            }
            for spec in _tool_specs(tools)
            if getattr(spec, "name", None) is not None
        ]

    if hooks is not None:
        snapshot["hooks"] = [type(h).__name__ for h in hooks]
    if memory_sources is not None:
        snapshot["memory_sources"] = [type(s).__name__ for s in memory_sources]
    if llm is not None:
        snapshot["llm_backend"] = type(llm).__name__

    return snapshot
