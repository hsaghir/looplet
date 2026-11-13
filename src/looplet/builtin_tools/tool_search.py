"""Deferred tool-schema discovery (`search_tools` + `DeferredToolViewHook`).

When an agent has dozens of MCP or domain tools, sending every full JSON
Schema on every turn wastes context tokens and degrades tool selection.
``DeferredToolViewHook`` keeps only pinned tools (plus terminal tools and
``search_tools``) in the active ``ToolView``, lists deferred tool names in a
compact one-line directory, and promotes matched tools into the active
``ToolView`` as soon as the agent calls ``search_tools(query=...)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import ToolContext

__all__ = ["DeferredToolViewHook", "SPEC", "make_tool_search_tool", "search_registered_tools"]


def search_registered_tools(
    registry: BaseToolRegistry,
    query: str,
    *,
    limit: int = 5,
    exclude: set[str] | frozenset[str] = frozenset({"search_tools"}),
) -> list[dict[str, Any]]:
    """Search ``registry`` by tool name, description, and parameter names."""
    if type(limit) is not int or limit < 1 or limit > 50:
        raise ValueError("limit must be an integer between 1 and 50")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    q_lower = query.strip().lower()
    tokens = [tok for tok in q_lower.replace("_", " ").split() if tok]
    scored: list[tuple[int, int, ToolSpec]] = []

    for idx, name in enumerate(registry.tool_names):
        if name in exclude:
            continue
        spec = registry._tools.get(name)
        if spec is None:
            continue
        name_lower = spec.name.lower()
        desc_lower = (spec.description or "").lower()
        params_lower = " ".join(spec.parameter_names()).lower()
        haystack = f"{name_lower} {desc_lower} {params_lower}"

        score = 0
        if q_lower == name_lower:
            score += 100
        elif q_lower in name_lower:
            score += 50
        elif q_lower in desc_lower:
            score += 25
        for tok in tokens:
            if tok in name_lower:
                score += 15
            elif tok in params_lower:
                score += 8
            elif tok in haystack:
                score += 4
        if score > 0:
            scored.append((score, -idx, spec))

    scored.sort(reverse=True)
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.to_json_schema(),
        }
        for _, _, spec in scored[:limit]
    ]


class DeferredToolViewHook:
    """LoopHook that defers non-pinned tool schemas until discovered via ``search_tools``."""

    def __init__(
        self,
        tools: BaseToolRegistry | None = None,
        *,
        pinned: Sequence[str] = (),
        terminal_tools: Sequence[str] = ("done",),
        max_search_results: int = 5,
    ) -> None:
        self._tools = tools
        self._pinned = set(pinned)
        self._always_visible = set(pinned) | set(terminal_tools) | {"search_tools"}
        self._active: set[str] = set(self._always_visible)
        self._max_search_results = max_search_results
        if tools is not None:
            self._install_on_registry(tools)

    def _install_on_registry(self, tools: BaseToolRegistry) -> None:
        self._tools = tools
        resources = tools.resources
        resources["deferred_tool_view"] = self
        tools.set_resources(resources)
        if "search_tools" not in tools._tools:
            tools.register(SPEC)

    @property
    def active_tool_names(self) -> list[str]:
        if self._tools is None:
            return sorted(self._active)
        return [name for name in self._tools.tool_names if name in self._active]

    @property
    def discovered_tool_names(self) -> list[str]:
        """Return tool names that were dynamically discovered and activated."""
        return [name for name in self.active_tool_names if name not in self._always_visible]

    @property
    def deferred_tool_names(self) -> list[str]:
        if self._tools is None:
            return []
        return [name for name in self._tools.tool_names if name not in self._active]

    def activate(self, *names: str) -> list[str]:
        """Promote tool names into the active ``ToolView``."""
        activated: list[str] = []
        known = set(self._tools.tool_names) if self._tools is not None else None
        for name in names:
            if known is None or name in known:
                self._active.add(name)
                activated.append(name)
        return activated

    def select_tools(
        self,
        *,
        step_num: int = 1,
        state: Any = None,
        tools: BaseToolRegistry | None = None,
        task: Any = None,
    ) -> list[str]:
        reg = tools or self._tools
        if reg is None:
            return sorted(self._active)
        return [
            name for name in reg.tool_names if name in self._active or name in self._always_visible
        ]

    def search(self, query: str, *, limit: int | None = None) -> dict[str, Any]:
        if self._tools is None:
            raise RuntimeError("DeferredToolViewHook is not bound to a tool registry")
        effective_limit = limit if limit is not None else self._max_search_results
        matches = search_registered_tools(self._tools, query, limit=effective_limit)
        activated = self.activate(*(m["name"] for m in matches))
        return {
            "query": query,
            "activated": activated,
            "matches": matches,
            "deferred_remaining": len(self.deferred_tool_names),
        }

    def bind_loop_context(self, loop_ctx: Any) -> None:
        if getattr(loop_ctx, "tools", None) is not None:
            self._install_on_registry(loop_ctx.tools)
        cfg = getattr(loop_ctx, "config", None)
        if cfg is not None:
            self._always_visible.update(cfg.resolve_terminal_outcomes().keys())
            self._active.update(self._always_visible)
            if cfg.tool_view_selector is None:
                cfg.tool_view_selector = self.select_tools

    def pre_loop(self, state: Any, session_log: Any, context: Any, *, tools: Any = None) -> None:
        if isinstance(tools, BaseToolRegistry):
            self._install_on_registry(tools)

    def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> str | None:
        deferred = self.deferred_tool_names
        if not deferred:
            return None
        return "Deferred tools available via search_tools(query=...): " + ", ".join(deferred)


def _execute_search_tools(
    *,
    query: str,
    limit: int = 5,
    ctx: ToolContext,
) -> dict[str, Any]:
    hook = ctx.resources.get("deferred_tool_view")
    if isinstance(hook, DeferredToolViewHook):
        return hook.search(query, limit=limit)
    raise PermissionError("No deferred_tool_view resource is bound to this tool registry")


SPEC = ToolSpec(
    name="search_tools",
    description=(
        "Search deferred tools by keyword and activate matching tool schemas for subsequent turns."
    ),
    parameters={
        "query": {
            "type": "string",
            "minLength": 1,
            "description": "Keyword or capability to search for across deferred tools.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 5,
            "description": "Maximum number of matching tools to activate and return.",
        },
    },
    execute=_execute_search_tools,
    requires=["deferred_tool_view"],
    concurrent_safe=True,
    read_only=True,
    free=True,
    origin="builtin",
)


def make_tool_search_tool(hook: DeferredToolViewHook) -> ToolSpec:
    """Create a ``search_tools`` ToolSpec bound directly to ``hook``."""

    def _run(*, query: str, limit: int = 5) -> dict[str, Any]:
        return hook.search(query, limit=limit)

    return ToolSpec(
        name="search_tools",
        description=SPEC.description,
        parameters=dict(SPEC.parameters),
        execute=_run,
        concurrent_safe=True,
        read_only=True,
        free=True,
        origin="builtin",
    )
