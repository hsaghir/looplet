"""Tool registry — domain-agnostic tool specification and dispatch.

Provides ToolSpec (tool definition) and BaseToolRegistry (registration,
dispatch, catalog rendering). Domain-specific agents subclass
BaseToolRegistry and register their own tools.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from openharness.types import ToolCall, ToolResult


@dataclass
class ToolSpec:
    """Specification of a tool available to the agent.

    Encapsulates everything the registry needs to invoke a tool:
    its name, human-readable description, parameter schema,
    the callable to execute, and scheduling hints.
    """

    name: str
    """Unique identifier used to reference this tool in ToolCall."""

    description: str
    """Human-readable description shown to the LLM in the tool catalog."""

    parameters: dict[str, str]
    """Mapping of parameter name → description for schema generation."""

    execute: Callable[..., Any] = field(repr=False)
    """Callable invoked when the tool is dispatched. Receives kwargs matching parameters."""

    concurrent_safe: bool = False
    """True if the tool is read-only and can run concurrently with other safe tools."""

    free: bool = False
    """True if the tool does not consume agent budget (e.g. think, reflect)."""

    def spec_text(self) -> str:
        """Format for LLM prompt inclusion."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"  {self.name}({params})\n    {self.description}"

    def to_api_schema(self) -> dict[str, Any]:
        """Generate API-compatible tool schema for native tool calling."""
        properties: dict[str, Any] = {}
        for param_name, param_desc in self.parameters.items():
            properties[param_name] = {
                "type": "string",
                "description": param_desc,
            }
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
            },
        }


class BaseToolRegistry:
    """Domain-agnostic tool registry with dispatch.

    Subclass this and call _register() in __init__ to add tools.
    dispatch() handles execution, timing, and error wrapping.
    dispatch_batch() partitions concurrent-safe vs serial calls for
    efficient execution.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._step_counter = 0

    def register(self, spec: ToolSpec) -> None:
        """Register a ToolSpec by name.

        Args:
            spec: The tool specification to register.
        """
        self._tools[spec.name] = spec

    # Backward-compat alias
    _register = register

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())

    def tool_catalog_text(self) -> str:
        """Format all registered tools for LLM prompt inclusion."""
        lines = ["Available tools:"]
        for spec in self._tools.values():
            lines.append(spec.spec_text())
        return "\n".join(lines)

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result with provenance.

        Strips dunder args (``__*``), wraps exceptions into error fields,
        and records wall-clock timing in duration_ms.
        """
        clean_args = {k: v for k, v in call.args.items() if not k.startswith("__")}

        if call.tool not in self._tools:
            return ToolResult(
                tool=call.tool,
                args_summary=str(clean_args)[:100],
                data=None,
                error=f"Unknown tool: {call.tool}. Available: {self.tool_names}",
                call_id=call.call_id,
            )

        spec = self._tools[call.tool]
        self._step_counter += 1
        t0 = time.time()
        try:
            result_data = spec.execute(**clean_args)
        except Exception as e:
            return ToolResult(
                tool=call.tool,
                args_summary=self._summarize_args(call),
                data=None,
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.time() - t0) * 1000,
                call_id=call.call_id,
            )

        elapsed = (time.time() - t0) * 1000
        result_key = self._store_result(call, result_data)

        return ToolResult(
            tool=call.tool,
            args_summary=self._summarize_args(call),
            data=result_data,
            duration_ms=elapsed,
            result_key=result_key,
            call_id=call.call_id,
        )

    def _store_result(self, call: ToolCall, result_data: Any) -> str | None:
        """Override in subclasses to enable result storage/recall."""
        return None

    def dispatch_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Dispatch multiple tool calls, preserving original order.

        Partitions consecutive concurrent-safe calls into parallel batches;
        serial (non-concurrent-safe) tools run one at a time.
        """
        if not calls:
            return []

        results: list[ToolResult] = []
        for batch in self._partition_calls(calls):
            if batch["concurrent"] and len(batch["calls"]) > 1:
                results.extend(self._dispatch_concurrent_batch(batch["calls"]))
            else:
                results.extend(self.dispatch(c) for c in batch["calls"])
        return results

    def _dispatch_concurrent_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Dispatch a batch of concurrent-safe tools in parallel via ThreadPoolExecutor."""
        if len(calls) <= 1:
            return [self.dispatch(c) for c in calls]
        with ThreadPoolExecutor(max_workers=min(10, len(calls))) as pool:
            futures = [pool.submit(self.dispatch, c) for c in calls]
            return [f.result() for f in futures]

    def _partition_calls(self, calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Partition tool calls into consecutive concurrent/serial batches.

        Consecutive concurrent-safe tools are merged into one batch.
        Non-concurrent tools each get their own single-item batch.
        """
        batches: list[dict[str, Any]] = []
        for call in calls:
            spec = self._tools.get(call.tool)
            is_safe = spec.concurrent_safe if spec else False
            if batches and batches[-1]["concurrent"] == is_safe and is_safe:
                batches[-1]["calls"].append(call)
            else:
                batches.append({"concurrent": is_safe, "calls": [call]})
        return batches

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Export all tool schemas for native API tool calling."""
        return [spec.to_api_schema() for spec in self._tools.values()]

    def _summarize_args(self, call: ToolCall) -> str:
        """Compact arg summary for logging and context."""
        parts: list[str] = []
        for k, v in call.args.items():
            s = str(v)
            if len(s) > 50:
                s = s[:50] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts)


def register_think_tool(registry: BaseToolRegistry) -> None:
    """Register the think() reasoning tool on a tool registry.

    think() lets the agent reason explicitly without taking an action
    or spending budget. The analysis is preserved in the tool result
    (and thus in the step log) but has no side effects.

    Use cases:
      - Analyze competing hypotheses before choosing the next action
      - Weigh pros and cons of different approaches
      - Plan the next 2-3 steps before executing them
      - Reflect on what prior steps have established so far
    """
    registry.register(ToolSpec(
        name="think",
        description=(
            "Pause to reason without taking an action. Use this to analyze "
            "competing hypotheses, weigh pros and cons, plan your next steps, "
            "or reflect on what you've found so far. Does NOT count against "
            "your budget. The analysis is preserved in your step log.\n"
            "Example: think(analysis='I have two plausible paths. "
            "To decide, I should first gather more data on option A, "
            "then compare against option B before committing.')"
        ),
        parameters={
            "analysis": "Your reasoning, analysis, or plan (free text)",
        },
        execute=lambda analysis="": {"acknowledged": True, "analysis": analysis},
        concurrent_safe=True,
        free=True,
    ))
