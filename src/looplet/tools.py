"""Tool registry — domain-agnostic tool specification and dispatch.

Provides ToolSpec (tool definition) and BaseToolRegistry (registration,
dispatch, catalog rendering). Domain-specific agents subclass
BaseToolRegistry and register their own tools.
"""

from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from looplet.types import ErrorKind, ToolCall, ToolContext, ToolError, ToolResult

__all__ = [
    "ToolSpec",
    "BaseToolRegistry",
    "register_think_tool",
    "register_done_tool",
]


def _classify_exception(e: BaseException) -> ToolError:
    """Map a Python exception to a :class:`ToolError`.

    Covers the common stdlib cases. Provider-specific exceptions
    (rate limits, context-overflow errors, API cancellations) are
    matched by class name as a best-effort since importing every
    provider SDK here would create a dependency mess — producers
    should attach a richer :class:`ToolError` directly when a more
    specific classification is needed.
    """
    msg = f"{type(e).__name__}: {e}"
    # Cooperative cancellation — asyncio.CancelledError inherits from
    # BaseException, so check it before the stdlib Exception branches.
    import asyncio as _asyncio  # noqa: PLC0415

    if isinstance(e, _asyncio.CancelledError):
        return ToolError(kind=ErrorKind.CANCELLED, message=msg, retriable=False)
    if isinstance(e, TimeoutError):
        return ToolError(kind=ErrorKind.TIMEOUT, message=msg, retriable=True)
    if isinstance(e, (ValueError, TypeError, KeyError)):
        return ToolError(kind=ErrorKind.VALIDATION, message=msg, retriable=False)
    if isinstance(e, PermissionError):
        return ToolError(kind=ErrorKind.PERMISSION_DENIED, message=msg, retriable=False)
    if isinstance(e, ConnectionError):
        return ToolError(kind=ErrorKind.NETWORK, message=msg, retriable=True)
    # Best-effort match on provider-specific exception class names.
    cls_name = type(e).__name__.lower()
    text = str(e).lower()
    if "ratelimit" in cls_name or "rate_limit" in cls_name or "429" in text:
        return ToolError(kind=ErrorKind.RATE_LIMIT, message=msg, retriable=True)
    if (
        "contextlengthexceeded" in cls_name
        or "context_length" in cls_name
        or "context window" in text
        or "too many tokens" in text
        or "input is too long" in text
    ):
        return ToolError(kind=ErrorKind.CONTEXT_OVERFLOW, message=msg, retriable=False)
    if "parseerror" in cls_name or "jsondecode" in cls_name:
        return ToolError(kind=ErrorKind.PARSE, message=msg, retriable=False)
    return ToolError(kind=ErrorKind.EXECUTION, message=msg, retriable=False)


def _format_param_hint(spec: "ToolSpec") -> str:
    """Render a ToolSpec's parameter schema as a short LLM-readable hint.

    For simple dicts this is ``{name: str, path: file path}``.
    For JSON Schema it is ``{name: string (required), age: integer}``.
    """
    params = spec.parameters
    if spec.is_json_schema:
        props = params.get("properties", {})
        required = set(params.get("required", []))
        parts = []
        for name, schema in props.items():
            typ = schema.get("type", "any") if isinstance(schema, dict) else "any"
            tag = f"{name}: {typ}" + (" (required)" if name in required else "")
            parts.append(tag)
        return "{" + ", ".join(parts) + "}"
    # Simple format — all params are required by convention.
    parts = [f"{name}: {desc}" for name, desc in params.items()]
    return "{" + ", ".join(parts) + "}"


def _accepts_ctx(fn: Callable[..., Any]) -> bool:
    """True if ``fn`` declares a ``ctx`` parameter (by name).

    Used to decide whether to thread ToolContext into a tool's execute
    callable. Cached per-ToolSpec to avoid repeated signature inspection.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "ctx" in sig.parameters


@dataclass
class ToolSpec:
    """Specification of a tool available to the agent.

    Encapsulates everything the registry needs to invoke a tool:
    its name, human-readable description, parameter schema,
    the callable to execute, and scheduling hints.

    ``parameters`` accepts two formats:

    1. **Simple** (backward-compatible): ``{"name": "str", "path": "file path"}``
       — keys are parameter names, values are type/description strings.

    2. **JSON Schema**: ``{"type": "object", "properties": {...}, "required": [...]}``
       — a full JSON Schema object.  Detected automatically when the dict
       contains **both** ``"type": "object"`` and a ``"properties"`` key.

    **Disambiguation:** detection requires both keys, so a simple-format
    dict that happens to contain ``"type"`` (e.g. ``{"type": "str"}``)
    is *not* misdetected — only a literal ``{"type": "object",
    "properties": {...}}`` shape triggers JSON-Schema mode.

    Example (simple)::

        ToolSpec(name="read", ..., parameters={"file_path": "str"})

    Example (JSON Schema)::

        ToolSpec(name="read", ..., parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            },
            "required": ["file_path"],
        })
    """

    name: str
    """Unique identifier used to reference this tool in ToolCall."""

    description: str
    """Human-readable description shown to the LLM in the tool catalog."""

    parameters: dict[str, Any]
    """Parameter schema — simple ``{name: desc}`` dict or full JSON Schema object."""

    execute: Callable[..., Any] = field(repr=False)
    """Callable invoked when the tool is dispatched. Receives kwargs matching parameters."""

    concurrent_safe: bool = False
    """True if the tool is read-only and can run concurrently with other safe tools."""

    free: bool = False
    """True if the tool does not consume agent budget (e.g. think, reflect)."""

    _accepts_ctx: bool | None = field(default=None, repr=False, compare=False)
    """Cached result of ``inspect.signature(execute)`` for ``ctx`` detection."""

    @property
    def is_json_schema(self) -> bool:
        """True if parameters are in JSON Schema format (has ``type: object``)."""
        return (
            isinstance(self.parameters.get("type"), str)
            and self.parameters["type"] == "object"
            and "properties" in self.parameters
        )

    def parameter_names(self) -> list[str]:
        """Return the list of parameter names regardless of schema format."""
        if self.is_json_schema:
            return list(self.parameters.get("properties", {}).keys())
        return list(self.parameters.keys())

    def required_parameters(self) -> list[str]:
        """Return required parameter names.

        For JSON Schema, reads the ``required`` field.
        For simple format, all parameters are assumed required.
        """
        if self.is_json_schema:
            return list(self.parameters.get("required", []))
        return list(self.parameters.keys())

    def spec_text(self) -> str:
        """Format for LLM prompt inclusion."""
        if self.is_json_schema:
            props = self.parameters.get("properties", {})
            required = set(self.parameters.get("required", []))
            parts = []
            for k, v in props.items():
                ptype = v.get("type", "any") if isinstance(v, dict) else "any"
                opt = "" if k in required else "?"
                parts.append(f"{k}{opt}: {ptype}")
            params = ", ".join(parts)
        else:
            params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"  {self.name}({params})\n    {self.description}"

    def to_api_schema(self) -> dict[str, Any]:
        """Generate API-compatible tool schema for native tool calling.

        When ``parameters`` is already JSON Schema, it is used directly
        as the ``input_schema``.  Otherwise, the simple format is
        auto-converted (all params typed as ``string``).
        """
        if self.is_json_schema:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.parameters,
            }
        properties: dict[str, Any] = {}
        for param_name, param_desc in self.parameters.items():
            properties[param_name] = {
                "type": "string",
                "description": str(param_desc),
            }
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(self.parameters.keys()),
            },
        }

    def to_json_schema(self) -> dict[str, Any]:
        """Return the parameter schema as a JSON Schema object.

        If ``parameters`` is already JSON Schema, returns it directly.
        Otherwise, auto-converts the simple format.
        """
        if self.is_json_schema:
            return dict(self.parameters)
        properties: dict[str, Any] = {}
        for param_name, param_desc in self.parameters.items():
            properties[param_name] = {
                "type": "string",
                "description": str(param_desc),
            }
        return {
            "type": "object",
            "properties": properties,
            "required": list(self.parameters.keys()),
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

    def register(self, spec: ToolSpec) -> None:
        """Register a ToolSpec by name.

        Args:
            spec: The tool specification to register.

        Warns (via ``logging.getLogger(__name__).warning``) when a
        tool with the same name is already registered — silent
        overwrites are a common source of bugs when composing
        multiple ``Skill`` bundles that happen to share a tool name.
        """
        if spec.name in self._tools:
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                "Tool %r is already registered — overwriting. "
                "This usually indicates a name collision between skills "
                "or tool bundles; give one of them a unique name.",
                spec.name,
            )
        # Eagerly compute ctx-acceptance so dispatch is thread-safe.
        if spec._accepts_ctx is None:
            spec._accepts_ctx = _accepts_ctx(spec.execute)
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

    def dispatch(self, call: ToolCall, *, ctx: ToolContext | None = None) -> ToolResult:
        """Execute a tool call and return the result with provenance.

        Strips dunder args (``__*``), wraps exceptions into error fields,
        and records wall-clock timing in duration_ms.

        When ``ctx`` is supplied and the tool's ``execute`` callable declares
        a ``ctx`` parameter, it is threaded through. If ``ctx.cancel_token``
        has been cancelled, the tool is skipped and a cancellation error is
        returned without invoking ``execute``.
        """
        clean_args = {k: v for k, v in call.args.items() if not k.startswith("__")}

        if call.tool not in self._tools:
            _te = ToolError(
                kind=ErrorKind.VALIDATION,
                message=f"Unknown tool: {call.tool}. Available: {self.tool_names}",
                retriable=False,
            )
            return ToolResult(
                tool=call.tool,
                args_summary=_summarize_args_dict(clean_args),
                data=None,
                error=_te.message,
                error_detail=_te,
                call_id=call.call_id,
            )

        spec = self._tools[call.tool]

        # Honor cancellation before invoking execute at all.
        if ctx is not None and ctx.cancel_token is not None and ctx.cancel_token.is_cancelled:
            _te = ToolError(
                kind=ErrorKind.CANCELLED,
                message="Tool execution cancelled before dispatch",
                retriable=False,
            )
            return ToolResult(
                tool=call.tool,
                args_summary=self._summarize_args(call),
                data=None,
                error=_te.message,
                error_detail=_te,
                call_id=call.call_id,
            )

        # _accepts_ctx is normally computed eagerly in register(), but guard
        # against ToolSpec instances constructed directly and inserted into
        # _tools without going through register().
        if spec._accepts_ctx is None:
            spec._accepts_ctx = _accepts_ctx(spec.execute)
        exec_kwargs: dict[str, Any] = dict(clean_args)
        if spec._accepts_ctx:
            exec_kwargs["ctx"] = ctx

        # Pre-validate required args so missing-param failures surface
        # as structured VALIDATION errors with the tool's parameter
        # schema — not as opaque ``TypeError: <lambda>() missing 1
        # required keyword-only argument: 'x'`` tracebacks that the LLM
        # cannot easily recover from.
        missing = [p for p in spec.required_parameters() if p not in clean_args]
        if missing:
            schema_hint = _format_param_hint(spec)
            _te = ToolError(
                kind=ErrorKind.VALIDATION,
                message=(
                    f"Tool '{spec.name}' missing required argument"
                    f"{'s' if len(missing) > 1 else ''}: {missing}. "
                    f"Expected: {schema_hint}"
                ),
                retriable=False,
            )
            return ToolResult(
                tool=call.tool,
                args_summary=self._summarize_args(call),
                data=None,
                error=_te.message,
                error_detail=_te,
                call_id=call.call_id,
            )

        t0 = time.time()
        try:
            result_data = spec.execute(**exec_kwargs)
        except Exception as e:
            _te = _classify_exception(e)
            return ToolResult(
                tool=call.tool,
                args_summary=self._summarize_args(call),
                data=None,
                error=_te.message,
                error_detail=_te,
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

    def dispatch_batch(
        self, calls: list[ToolCall], *, ctx: ToolContext | None = None
    ) -> list[ToolResult]:
        """Dispatch multiple tool calls, preserving original order.

        Partitions consecutive concurrent-safe calls into parallel batches;
        serial (non-concurrent-safe) tools run one at a time. ``ctx`` is
        forwarded to each underlying dispatch call.
        """
        if not calls:
            return []

        results: list[ToolResult] = []
        for batch in self._partition_calls(calls):
            if batch["concurrent"] and len(batch["calls"]) > 1:
                results.extend(self._dispatch_concurrent_batch(batch["calls"], ctx=ctx))
            else:
                results.extend(self.dispatch(c, ctx=ctx) for c in batch["calls"])
        return results

    def _dispatch_concurrent_batch(
        self, calls: list[ToolCall], *, ctx: ToolContext | None = None
    ) -> list[ToolResult]:
        """Dispatch a batch of concurrent-safe tools in parallel via ThreadPoolExecutor."""
        if len(calls) <= 1:
            return [self.dispatch(c, ctx=ctx) for c in calls]
        with ThreadPoolExecutor(max_workers=min(10, len(calls))) as pool:
            futures = [pool.submit(self.dispatch, c, ctx=ctx) for c in calls]
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

    def introspect(self) -> dict[str, Any]:
        """Return a machine-readable description of all registered tools.

        Useful for coding agents to discover available tools, their
        parameters, and capabilities at runtime.

        Returns a dict with:
          - ``tool_count``: number of registered tools
          - ``tools``: list of tool metadata dicts with name, description,
            parameters (JSON Schema), concurrent_safe, free
        """
        tools_info = []
        for spec in self._tools.values():
            tools_info.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.to_json_schema(),
                    "concurrent_safe": spec.concurrent_safe,
                    "free": spec.free,
                }
            )
        return {
            "tool_count": len(self._tools),
            "tools": tools_info,
        }

    def _summarize_args(self, call: ToolCall) -> str:
        """Compact arg summary for logging and context."""
        return _summarize_args_dict(call.args)


def _summarize_args_dict(args: dict[str, Any]) -> str:
    """Format tool args as ``k=v, k=v`` for consistent step rendering.

    Module-level helper so loop.py, validation.py, and permission
    denial paths can all render args the same way as successful
    dispatch. Without this, the same call looks like
    ``bash(cmd=ls)`` when allowed and ``bash({'cmd': 'ls'})`` when
    denied or intercepted — visually jarring in logs.
    """
    parts: list[str] = []
    for k, v in args.items():
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
    registry.register(
        ToolSpec(
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
        )
    )


def register_done_tool(
    registry: BaseToolRegistry,
    *,
    name: str = "done",
    parameters: dict[str, Any] | None = None,
) -> None:
    """Register the done() completion-signal tool on a tool registry.

    The composable loop expects a tool matching ``LoopConfig.done_tool``
    (default ``"done"``) to be registered. When absent, the LLM's
    ``done()`` call lands on "Unknown tool" — a common first-use
    footgun since there is no error at setup time.

    Call this alongside ``register_think_tool`` when building a
    registry manually (presets call it automatically)::

        tools = BaseToolRegistry()
        register_done_tool(tools)
        register_think_tool(tools)
        tools.register(ToolSpec(name="search", ...))

    Args:
        registry: The tool registry to add the done tool to.
        name: Tool name — must match ``LoopConfig.done_tool``.
        parameters: Custom parameter schema. Defaults to
            ``{"summary": "Brief summary of what was accomplished"}``.
    """
    params = parameters or {"summary": "Brief summary of what was accomplished"}
    registry.register(
        ToolSpec(
            name=name,
            description=(
                "Signal that the task is complete. Call this when you have "
                "finished the task and have no more actions to take. Provide "
                "a brief summary of what was accomplished."
            ),
            parameters=params,
            execute=lambda **kwargs: {"status": "completed", **kwargs},
        )
    )
