"""Stable, JSON-friendly harness snapshots for provenance metadata."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
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


def _component_name(value: Any) -> str:
    component_type = type(value)
    module = getattr(value, "__module__", component_type.__module__)
    name = getattr(value, "__qualname__", component_type.__qualname__)
    return f"{module}.{name}"


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
            snapshot["system_prompt_sha256"] = sha256(
                str(system_prompt).encode("utf-8")
            ).hexdigest()
        for key in (
            "max_steps",
            "max_tokens",
            "temperature",
            "recovery_temperature",
            "use_native_tools",
            "concurrent_dispatch",
            "max_parallel_calls",
            "done_tool",
            "done_tools",
            "harness_profile",
            "disabled_defaults",
            "max_model_calls",
            "max_assistant_turns",
            "max_turn_continuations",
            "allow_text_completion",
            "requires_assistant_turn_policy",
            "reactive_recovery",
            "context_window",
            "max_briefing_tokens",
            "context_window_steps",
            "context_inline_per_step_chars",
            "context_window_total_chars",
            "scoped_context_budget_tokens",
        ):
            value = _getattr_present(config, key)
            if value is not _MISSING:
                snapshot[key] = deepcopy(value)
        components = {}
        for key in (
            "compact_service",
            "assistant_turn_policy",
            "build_prompt",
            "build_briefing",
            "extract_entities",
            "extract_step_metadata",
            "context_planner",
            "tool_view_selector",
            "router",
            "output_schema",
            "checkpoint_state",
            "restore_checkpoint_state",
            "execution_policy",
            "cache_policy",
        ):
            value = _getattr_present(config, key)
            if value is not _MISSING:
                components[key] = _component_name(value)
        if components:
            snapshot["components"] = components

    if tools is not None:
        snapshot["tools"] = []
        for spec in _tool_specs(tools):
            if getattr(spec, "name", None) is None:
                continue
            description = {
                "name": getattr(spec, "name"),
                "description": getattr(spec, "description", ""),
            }
            parameters = getattr(spec, "parameters", None)
            if parameters is not None:
                description["parameters"] = deepcopy(parameters)
            execute = getattr(spec, "execute", None)
            if execute is not None:
                description["implementation"] = _component_name(execute)
            snapshot["tools"].append(description)

    if hooks is not None:
        snapshot["hooks"] = [type(h).__name__ for h in hooks]
        snapshot["hook_types"] = [_component_name(hook) for hook in hooks]
    if memory_sources is not None:
        snapshot["memory_sources"] = [type(s).__name__ for s in memory_sources]
        snapshot["memory_source_types"] = [_component_name(source) for source in memory_sources]
    if llm is not None:
        snapshot["llm_backend"] = type(llm).__name__
        snapshot["llm_backend_type"] = _component_name(llm)

    return snapshot
