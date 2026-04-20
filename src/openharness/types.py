"""Core data types and protocols for tool-using LLM agents.

Domain-agnostic: Step, ToolCall, ToolResult can represent any
tool invocation in any agent pipeline.

Protocols define the contracts that agent states and LLM backends
must satisfy to work with the openharness loop engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable
from uuid import uuid4

# ── Error taxonomy ───────────────────────────────────────────────


class ErrorKind(str, Enum):
    """Discriminator for tool and LLM errors.

    Inherits ``str`` so values compare equal to their string form and
    serialise cleanly in ``to_dict()`` / JSON.
    """

    PERMISSION_DENIED = "permission_denied"
    """Tool call blocked by a permission check (engine or hook)."""

    TIMEOUT = "timeout"
    """Tool/LLM call exceeded its deadline."""

    VALIDATION = "validation"
    """Args failed schema validation or tool name unknown."""

    EXECUTION = "execution"
    """Generic runtime failure inside the tool body."""

    PARSE = "parse"
    """LLM response couldn't be parsed into a tool call."""

    CONTEXT_OVERFLOW = "context_overflow"
    """Prompt exceeded the backend's context window."""

    RATE_LIMIT = "rate_limit"
    """Provider returned a throttling / quota error."""

    NETWORK = "network"
    """Transport-level failure reaching the provider."""

    CANCELLED = "cancelled"
    """Cancelled via ``CancelToken`` before completion."""


@dataclass
class ToolError:
    """Structured error produced by a tool or LLM call.

    Prefer this over a bare string when you need the loop, hooks, or
    permission-aware recovery to distinguish failure modes.
    """

    kind: ErrorKind
    """Discriminator — which class of failure occurred."""

    message: str
    """Human-readable description, safe to include in LLM context."""

    retriable: bool = False
    """Advisory hint for recovery hooks and external orchestrators.

    Set ``True`` for transient failures (rate limits, timeouts, network
    errors) where a retry has a real chance of success.  The built-in
    loop does **not** automatically retry based on this flag; it is
    consumed by:

    * ``RecoveryRegistry`` recipes (if registered via ``LoopConfig.recovery_registry``)
    * Custom ``LoopHook.post_dispatch`` implementations
    * External orchestrators wrapping ``composable_loop``

    Producers set this via ``_classify_exception()`` in ``tools.py``."""

    context: dict[str, Any] = field(default_factory=dict)
    """Optional structured metadata — e.g. ``{"attempts": 3,
    "next_retry_in": 2.0}`` — attached by the producer for observability."""

    def __bool__(self) -> bool:  # truthy like a string error
        return True

    def __str__(self) -> str:
        return self.message


# ── Cancellation ─────────────────────────────────────────────────


@dataclass
class CancelToken:
    """Cooperative cancellation signal for long-running tools.

    A token starts non-cancelled. Any observer can call ``cancel()`` to
    mark it. Tool implementations poll ``is_cancelled`` or call
    ``raise_if_cancelled()`` at safe checkpoints to stop early.

    The token is intentionally simple: no threading primitives, no async
    events. Tools that need those wrap the token in their own primitive.
    """

    _cancelled: bool = False

    def cancel(self) -> None:
        """Mark this token as cancelled. Idempotent."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """True once ``cancel()`` has been called."""
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Raise :class:`RuntimeError` if this token has been cancelled.

        Useful as a one-liner at tool checkpoints:
            ctx.cancel_token and ctx.cancel_token.raise_if_cancelled()
        """
        if self._cancelled:
            raise RuntimeError("Tool execution cancelled")


# ── ToolContext ──────────────────────────────────────────────────


@dataclass
class ToolContext:
    """Runtime context handed to tools that opt-in via a ``ctx`` parameter.

    Gives tools structured
    access to workspace information, cancellation, progress reporting, and
    arbitrary per-session metadata. Strictly opt-in — tools without a
    ``ctx`` kwarg in their signature never receive one.

    Fields:
        cwd: Current working directory for file operations.
        workspace_root: Root of the active workspace (session-bound).
        cancel_token: Cooperative cancellation signal. Tools should check
            periodically during long operations.
        on_progress: Optional callback ``(stage: str, data: dict) -> None``
            that tools may invoke to report incremental progress.
        session_id: Identifier for the owning loop/session.
        metadata: Arbitrary key/value bag for domain-specific context
            (e.g. ``{"task_id": "...", "permission_mode": "read-only"}``).
    """

    cwd: str | None = None
    workspace_root: str | None = None
    cancel_token: "CancelToken | None" = None
    on_progress: Callable[[str, dict], None] | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    request_approval: Callable[[str, list[str] | None], str | None] | None = None
    """Optional handler that lets a tool request approval from the
    caller (user, upstream agent, webhook) mid-execution. Signature is
    ``(prompt: str, options: list[str] | None) -> str | None``.

    Returns the caller's reply, or ``None`` when:
      * No handler is installed (headless/autonomous run).
      * The handler defers (async approval — the loop should
        checkpoint and stop; see :class:`ApprovalHook`).

    Tools should treat ``None`` as "proceed without approval" to
    remain usable in headless runs."""

    def report_progress(self, stage: str, data: dict | None = None) -> None:
        """Invoke the progress callback if one is installed. Silent if not."""
        if self.on_progress is not None:
            self.on_progress(stage, data or {})

    def approve(self, prompt: str, options: list[str] | None = None) -> str | None:
        """Request approval from the configured handler.

        Returns the approval response, or ``None`` if no handler is
        installed or the handler defers (async). Tools should treat
        ``None`` as "approved by default" or "not yet — proceed
        cautiously" depending on their risk model."""
        if self.request_approval is None:
            return None
        return self.request_approval(prompt, options)


# ── Protocols ────────────────────────────────────────────────────


@runtime_checkable
class AgentState(Protocol):
    """Protocol defining the state interface the loop engine requires.

    Any agent state class must provide these attributes and methods
    to work with the openharness pipeline loop. Implementations are
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


@runtime_checkable
class NativeToolBackend(Protocol):
    """Optional protocol for backends that support native tool calling.

    Backends satisfying this protocol can accept tool schemas and return
    structured tool-use blocks (list of dicts) instead of free-text JSON.
    The loop detects the capability via hasattr(backend, "generate_with_tools")
    and only invokes it when ``LoopConfig.use_native_tools``
    is True.

    The returned list is normalised to Anthropic-style content blocks:
        [{"type": "text", "text": "..."},
         {"type": "tool_use", "id": "...", "name": "...", "input": {...}}, ...]

    Each backend is responsible for translating its provider's native tool-call
    shape (e.g. OpenAI ``message.tool_calls``) into this unified format.
    """

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Generate a response with native tool calling.

        Args:
            prompt: The user/context prompt.
            tools: Tool schemas in Anthropic format
                ``[{"name": ..., "description": ..., "input_schema": {...}}, ...]``.
                Each backend converts to its native tool-schema format internally.
            max_tokens: Upper bound on tokens in the response.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.

        Returns:
            List of normalised content blocks (text and/or tool_use).
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
    """Error message if the tool raised an exception; None on success.
    Always a plain string for safe serialization and downstream use."""

    error_detail: ToolError | None = None
    """Structured error metadata (kind, retriable, context). Populated
    by the tool dispatch layer alongside ``error``. Read via the
    ``error_kind`` / ``error_retriable`` accessors."""

    duration_ms: float = 0.0
    """Wall-clock time the tool took to execute, in milliseconds."""

    result_key: str | None = None
    """Optional key for storing this result in a recall/memory store."""

    call_id: str | None = None
    """Links back to the ToolCall that produced this result."""

    @property
    def error_message(self) -> str | None:
        """Human-readable message — same as ``error``."""
        return self.error

    @property
    def error_kind(self) -> ErrorKind | None:
        """Discriminator from the structured error, if available.
        Defaults to ``EXECUTION`` when only a plain string error is set."""
        if self.error is None:
            return None
        if self.error_detail is not None:
            return self.error_detail.kind
        return ErrorKind.EXECUTION

    @property
    def error_retriable(self) -> bool:
        """Retriable hint from the structured error, if available."""
        if self.error_detail is not None:
            return self.error_detail.retriable
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a compact dict for inclusion in LLM context."""
        d: dict[str, Any] = {
            "tool": self.tool,
            "args": self.args_summary,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.error:
            d["error"] = self.error
            if self.error_detail is not None:
                d["error_kind"] = self.error_detail.kind.value
                d["error_retriable"] = self.error_detail.retriable
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

    The most common usage is printing each step as you iterate::

        for step in composable_loop(...):
            print(step.pretty())   # "#1 ✓ search(query='x') → 12 items [182ms]"
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

    def pretty(self) -> str:
        """One-line human-readable summary for CLI / log display.

        Unlike :meth:`summary` (which is tuned for LLM context assembly),
        ``pretty`` prioritises human readability: it prefixes with the
        step number, flags success/failure with ``✓``/``✗``, and includes
        per-step duration when the tool registry recorded it. Safe to
        ``print()`` directly while iterating the loop::

            for step in composable_loop(...):
                print(step.pretty())
        """
        r = self.tool_result
        status = "✗" if r.error else "✓"
        header = f"#{self.number} {status} {r.tool}({r.args_summary})"
        if r.error:
            tail = f"ERROR: {r.error[:80]}"
        elif isinstance(r.data, list):
            tail = f"{len(r.data)} items"
        elif isinstance(r.data, dict):
            tail = f"{len(r.data)} keys"
        elif r.data is None:
            tail = ""
        else:
            snippet = str(r.data)
            tail = snippet if len(snippet) <= 60 else snippet[:57] + "..."
        ms = f" [{r.duration_ms:.0f}ms]" if r.duration_ms > 0 else ""
        return f"{header} → {tail}{ms}" if tail else f"{header}{ms}"
