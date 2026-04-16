"""Core data types and protocols for tool-using LLM agents.

Domain-agnostic: Step, ToolCall, ToolResult can represent any
tool invocation in any agent pipeline.

Protocols define the contracts that agent states and LLM backends
must satisfy to work with the cadence loop engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4


# ── Protocols ────────────────────────────────────────────────────


@runtime_checkable
class AgentState(Protocol):
    """Protocol defining the state interface the loop engine requires.

    Any agent state class must provide these attributes and methods
    to work with the cadence pipeline loop. Implementations are
    responsible for tracking steps taken, resource usage, and
    producing summaries for LLM context windows.
    """

    steps: list
    queries_used: int

    @property
    def step_count(self) -> int:
        """Total number of steps executed so far."""
        ...

    @property
    def budget_remaining(self) -> int:
        """Remaining budget (queries/steps) before the agent must stop."""
        ...

    def context_summary(self) -> str:
        """Return a brief string summarising the current agent state for the LLM."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the current state."""
        ...


@dataclass
class DefaultState:
    """Ready-to-use AgentState implementation.

    Satisfies the ``AgentState`` protocol with sensible defaults so you
    don't need to write your own state class for simple agents.

    Usage::

        state = DefaultState(max_steps=15)
        for step in composable_loop(llm, tools=reg, state=state, ...):
            ...

    For domain-specific state (findings, hypotheses, custom fields),
    subclass or write your own class satisfying the ``AgentState`` protocol.
    """

    steps: list = field(default_factory=list)
    queries_used: int = 0
    max_steps: int = 15
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - len(self.steps))

    def context_summary(self) -> str:
        if not self.steps:
            return ""
        lines = []
        for step in self.steps[-5:]:
            lines.append(step.summary() if hasattr(step, "summary") else str(step))
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "queries_used": self.queries_used,
            "budget_remaining": self.budget_remaining,
            **self.metadata,
        }


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol defining the LLM interface the loop engine requires.

    Any LLM backend must implement generate() with this exact signature
    so the pipeline can swap backends (OpenAI, Anthropic, local, mock)
    without changing loop logic.
    """

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user/context prompt to complete.
            max_tokens: Upper bound on tokens in the response.
            system_prompt: Optional system instruction prepended to the conversation.
            temperature: Sampling temperature; lower = more deterministic.

        Returns:
            The generated text as a plain string.
        """
        ...


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM.

    Carries the tool name, arguments parsed from the LLM output,
    the model's reasoning for making the call, and a unique call ID
    used to correlate this request with its ToolResult.
    """

    tool: str
    """Name of the tool to invoke."""

    args: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to pass to the tool."""

    reasoning: str = ""
    """The model's reasoning for choosing this tool (for logging/debugging)."""

    call_id: str = field(default_factory=lambda: uuid4().hex[:12])
    """Unique identifier linking this call to its result. Auto-generated if not provided."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging or context assembly."""
        return {
            "tool": self.tool,
            "args": self.args,
            "reasoning": self.reasoning,
            "call_id": self.call_id,
        }


@dataclass
class ToolResult:
    """Result of executing a tool call.

    Captures everything the loop engine needs to decide next steps:
    the raw output data, any error message, timing, an optional
    cache/recall key, and the originating call_id.
    """

    tool: str
    """Name of the tool that produced this result."""

    args_summary: str
    """Human-readable summary of the arguments used (for compact context)."""

    data: Any
    """Raw output returned by the tool — list, dict, str, or None."""

    error: str | None = None
    """Error message if the tool raised an exception; None on success."""

    duration_ms: float = 0.0
    """Wall-clock time the tool took to execute, in milliseconds."""

    result_key: str | None = None
    """Optional key for storing this result in a recall/memory store."""

    call_id: str | None = None
    """Links back to the ToolCall that produced this result."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a compact dict for inclusion in LLM context."""
        d: dict[str, Any] = {
            "tool": self.tool,
            "args": self.args_summary,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.error:
            d["error"] = self.error
        elif self.result_key:
            d["result_key"] = self.result_key
        if isinstance(self.data, list):
            d["total_items"] = len(self.data)
            d["data"] = self.data[:20]
        elif isinstance(self.data, dict):
            d["data"] = self.data
        else:
            d["data"] = str(self.data)[:2000]
        return d


@dataclass
class Step:
    """One complete step in the agent loop: a tool call paired with its result.

    Steps are accumulated in AgentState.steps and used to build
    context summaries for subsequent LLM prompts.
    """

    number: int
    """1-based step index within the current agent run."""

    tool_call: ToolCall
    """The tool invocation requested by the LLM."""

    tool_result: ToolResult
    """The result returned after executing the tool call."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging or state snapshots."""
        return {
            "step": self.number,
            "call": self.tool_call.to_dict(),
            "result": self.tool_result.to_dict(),
        }

    def summary(self) -> str:
        """One-line human-readable summary for compact context assembly."""
        r = self.tool_result
        if r.error:
            return f"S{self.number} ✗ {r.tool}({r.args_summary}) → ERROR: {r.error[:60]}"
        if isinstance(r.data, list):
            return f"S{self.number} ✓ {r.tool}({r.args_summary}) → {len(r.data)} items"
        if isinstance(r.data, dict):
            total = r.data.get("total", r.data.get("total_items", "?"))
            return f"S{self.number} ✓ {r.tool}({r.args_summary}) → {total}"
        return f"S{self.number} ✓ {r.tool}({r.args_summary})"
