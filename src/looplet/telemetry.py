"""Telemetry — structured observability for monitoring agent behavior in production.

Provides:
  - Span: a named, timed unit of work with parent-child nesting
  - Tracer: manages the span stack and builds the trace tree
  - TracingHook: LoopHook that creates spans at each hook point
  - MetricsCollector: accumulates step-level metrics for reporting
  - MetricsHook: LoopHook that updates a MetricsCollector at each step
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ── Span ────────────────────────────────────────────────────────────


@dataclass
class Span:
    """A named, timed unit of work within a trace tree.

    Spans nest parent→child: a parent span holds references to its
    children. ``duration_ms`` is None until the span is ended.
    """

    name: str
    """Human-readable name for this operation."""

    span_id: str = field(default_factory=lambda: uuid4().hex[:12])
    """Unique 12-char hex ID for this span."""

    parent_id: str | None = None
    """span_id of the parent, or None for root spans."""

    start_time: float = field(default_factory=time.time)
    """Unix timestamp when the span started."""

    end_time: float | None = None
    """Unix timestamp when the span ended; None if still in-flight."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value metadata attached to this span."""

    children: list["Span"] = field(default_factory=list)
    """Child spans nested under this one."""

    status: str = "ok"
    """Span outcome: 'ok', 'error', or 'cancelled'."""

    @property
    def duration_ms(self) -> float | None:
        """Duration in milliseconds; None if the span has not ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0


# ── Tracer ──────────────────────────────────────────────────────────


class Tracer:
    """Manages a span stack and builds a complete trace tree.

    Usage::

        tracer = Tracer()
        root = tracer.start_span("loop.run")
        child = tracer.start_span("tool.search", attributes={"query": "..."})
        tracer.end_span(child)
        tracer.end_span(root)
        # tracer.root_spans == [root]
    """

    def __init__(self) -> None:
        self._stack: list[Span] = []
        self._root_spans: list[Span] = []

    @property
    def current_span(self) -> Span | None:
        """The innermost open span, or None if no span is active."""
        return self._stack[-1] if self._stack else None

    @property
    def root_spans(self) -> list[Span]:
        """All top-level spans recorded so far."""
        return self._root_spans

    def start_span(self, name: str, *, attributes: dict[str, Any] | None = None) -> Span:
        """Create a new span, nesting it under the current span if one is open.

        Returns the new span and pushes it onto the stack.
        """
        parent = self.current_span
        span = Span(
            name=name,
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes or {}),
        )
        if parent is None:
            self._root_spans.append(span)
        else:
            parent.children.append(span)
        self._stack.append(span)
        return span

    def end_span(
        self,
        span: Span,
        *,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End a span: set end_time, status, merge extra attributes, pop from stack."""
        if span.end_time is None:
            span.end_time = time.time()
        span.status = status
        if attributes:
            span.attributes.update(attributes)
        if self._stack and self._stack[-1] is span:
            self._stack.pop()
        else:
            # Handle out-of-order ending — scan and remove
            try:
                self._stack.remove(span)
            except ValueError:
                pass


# ── TracingHook ─────────────────────────────────────────────────────


class TracingHook:
    """LoopHook that builds a complete trace tree per loop execution.

    - ``pre_prompt``: starts a root loop span
    - ``post_dispatch``: creates a child tool span for each step and ends it
    - ``on_loop_end``: ends the root loop span
    - All other methods are no-ops that preserve normal loop flow
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._loop_span: Span | None = None

    # ── LoopHook interface ─────────────────────────────────────────

    def pre_prompt(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Start the root loop span on the first step."""
        if self._loop_span is None:
            self._loop_span = self._tracer.start_span(
                "loop.run",
                attributes={"step_num": step_num},
            )
        return None

    def pre_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        step_num: int,
    ) -> Any | None:
        return None

    def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        tool_result: Any,
        step_num: int,
    ) -> str | None:
        """Create a child tool span for the completed step."""
        tool = tool_call.tool if hasattr(tool_call, "tool") else "unknown"
        has_error = bool(tool_result.error if hasattr(tool_result, "error") else False)
        duration_ms = getattr(tool_result, "duration_ms", 0.0) or 0.0
        status = "error" if has_error else "ok"
        span = self._tracer.start_span(
            f"tool.{tool}",
            attributes={
                "tool": tool,
                "step": step_num,
                "has_error": has_error,
                "duration_ms": duration_ms,
            },
        )
        # Back-date end_time so span reflects actual tool duration
        if duration_ms > 0:
            span.end_time = span.start_time + duration_ms / 1000.0
        self._tracer.end_span(span, status=status)
        return None

    def check_done(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        return None

    def should_stop(
        self,
        state: Any,
        step_num: int,
        new_entities: int,
    ) -> bool:
        return False

    def on_loop_end(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        llm: Any,
    ) -> int:
        """End the root loop span."""
        if self._loop_span is not None:
            self._tracer.end_span(self._loop_span)
            self._loop_span = None
        return 0


# ── MetricsCollector ────────────────────────────────────────────────


@dataclass
class MetricsCollector:
    """Accumulates step-level metrics across an agent loop run.

    Call ``record_step()`` after each step; call ``report()`` at the end
    to get a human-readable summary.
    """

    total_steps: int = 0
    """Number of steps executed."""

    total_llm_calls: int = 0
    """Number of LLM calls made (updated externally if desired)."""

    total_tool_calls: int = 0
    """Number of tool invocations (one per step)."""

    total_errors: int = 0
    """Number of steps that produced an error."""

    total_input_tokens_est: int = 0
    """Estimated input tokens consumed."""

    total_output_tokens_est: int = 0
    """Estimated output tokens produced."""

    tool_call_histogram: dict[str, int] = field(default_factory=dict)
    """Count of calls per tool name."""

    step_classifications: dict[str, int] = field(default_factory=dict)
    """Count per classification label (productive/empty/redundant/error)."""

    total_duration_ms: float = 0.0
    """Total wall-clock time spent across all steps in milliseconds."""

    def record_step(
        self,
        tool_name: str,
        classification: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        has_error: bool,
    ) -> None:
        """Record metrics for one completed step.

        Args:
            tool_name: Name of the tool invoked.
            classification: Step quality label (e.g. 'productive', 'empty', 'error').
            input_tokens: Estimated input tokens for this step.
            output_tokens: Estimated output tokens for this step.
            duration_ms: Wall-clock time for this step in milliseconds.
            has_error: True if the tool result contained an error.
        """
        self.total_steps += 1
        self.total_tool_calls += 1
        self.total_input_tokens_est += input_tokens
        self.total_output_tokens_est += output_tokens
        self.total_duration_ms += duration_ms
        self.tool_call_histogram[tool_name] = self.tool_call_histogram.get(tool_name, 0) + 1
        self.step_classifications[classification] = (
            self.step_classifications.get(classification, 0) + 1
        )
        if has_error:
            self.total_errors += 1

    def report(self) -> str:
        """Return a formatted human-readable metrics summary."""
        lines: list[str] = ["═══ AGENT METRICS ═══"]
        lines.append(f"Steps: {self.total_steps}  |  Errors: {self.total_errors}")
        lines.append(
            f"Tokens (est): {self.total_input_tokens_est} in / {self.total_output_tokens_est} out"
        )
        lines.append(f"Duration: {self.total_duration_ms:.1f}ms")

        if self.tool_call_histogram:
            top = sorted(self.tool_call_histogram.items(), key=lambda x: -x[1])
            tool_str = ", ".join(f"{t}×{c}" for t, c in top[:10])
            lines.append(f"Tools: {tool_str}")

        if self.step_classifications:
            cls_str = ", ".join(f"{k}={v}" for k, v in sorted(self.step_classifications.items()))
            lines.append(f"Classifications: {cls_str}")

        if self.total_llm_calls > 0:
            lines.append(f"LLM calls: {self.total_llm_calls}")

        return "\n".join(lines)


# ── MetricsHook ─────────────────────────────────────────────────────


class MetricsHook:
    """LoopHook that updates a MetricsCollector after each step.

    ``post_dispatch`` records per-tool-call metrics; ``on_event``
    increments ``total_llm_calls`` on every ``POST_LLM_RESPONSE``
    event so the collector's LLM-call counter is populated by
    default. The classification defaults to 'productive' (callers
    can subclass or wrap to apply richer classification logic).
    """

    def __init__(
        self,
        collector: MetricsCollector,
        *,
        classify: Any | None = None,
    ) -> None:
        self._collector = collector
        self._classify = classify  # optional Callable[[Step, Any], str]

    def to_config(self) -> dict:
        """Workspace round-trip: emit ``collector`` as an ``@ref`` so the
        v2 workspace writer auto-generates ``resources/collector.py`` and
        the loader rebuilds a fresh ``MetricsCollector`` per load.
        """
        return {"collector": "@collector"}

    @property
    def collector(self) -> MetricsCollector:
        """Public accessor used by the workspace writer to pull the
        live instance for resource auto-emit."""
        return self._collector

    # ── LoopHook interface ─────────────────────────────────────────

    def pre_prompt(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        return None

    def pre_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        step_num: int,
    ) -> Any | None:
        return None

    def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        tool_result: Any,
        step_num: int,
    ) -> str | None:
        """Record step metrics into the collector."""
        tool = tool_call.tool if hasattr(tool_call, "tool") else "unknown"
        has_error = bool(tool_result.error if hasattr(tool_result, "error") else False)
        if self._classify is not None:
            classification = self._classify(tool_call, tool_result, state)
        elif has_error:
            classification = "error"
        else:
            classification = "productive"

        self._collector.record_step(
            tool_name=tool,
            classification=classification,
            input_tokens=0,
            output_tokens=0,
            duration_ms=getattr(tool_result, "duration_ms", 0.0) or 0.0,
            has_error=has_error,
        )
        return None

    def check_done(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        return None

    def should_stop(
        self,
        state: Any,
        step_num: int,
        new_entities: int,
    ) -> bool:
        return False

    def on_loop_end(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        llm: Any,
    ) -> int:
        return 0

    def on_event(self, payload: Any) -> None:
        """Increment ``total_llm_calls`` on every ``POST_LLM_RESPONSE``.

        Lifecycle events route here regardless of whether the per-method
        hooks are implemented, so this is the right place to count LLM
        calls without depending on a backend wrapper.
        """
        from looplet.events import LifecycleEvent  # noqa: PLC0415

        event = getattr(payload, "event", None)
        if event is LifecycleEvent.POST_LLM_RESPONSE:
            self._collector.total_llm_calls += 1
        return None
