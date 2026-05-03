"""Core data types and protocols for tool-using LLM agents.

Domain-agnostic: Step, ToolCall, ToolResult can represent any
tool invocation in any agent pipeline.

Protocols define the contracts that agent states and LLM backends
must satisfy to work with the looplet loop engine.
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

    recovery_hint: dict[str, Any] | str | None = None
    """Optional structured suggestion for how the caller could recover.

    Carries information that the LLM can act on directly. Two common
    shapes:

    * **Schema/shape hint** (``dict``): name the expected argument
      shape so the model can correct a malformed call without
      re-discovering the tool spec. The dispatcher's built-in
      ``"got unexpected argument"`` and ``"missing required argument"``
      errors set this to ``{"expected": <param_schema>}``.
    * **Suggestion** (``str``): a "did you mean?" hint surfaced from
      the tool. The dispatcher's ``"unknown tool"`` error sets this
      to ``"Did you mean '<closest>'?"`` when one is found.

    Tool authors should populate this on any error the model could
    fix by changing its next call — leaving it ``None`` means "no
    actionable recovery information" (a hard failure, e.g. a
    permission deny). The recovery hint is included in the rendered
    error text the loop hands back to the model on the next turn.
    """

    def __bool__(self) -> bool:  # truthy like a string error
        return True

    def __str__(self) -> str:
        return self.message


# ── Tool-author exceptions ───────────────────────────────────────


class ToolValidationError(Exception):
    """Raised by tool implementations to signal a caller/input mistake.

    Tool authors should raise this instead of returning pseudo-error
    sentinels inside their normal output (e.g. ``{"error": "..."}``
    mixed into a result list). Agents — and human callers — then see a
    uniform :class:`ToolResult` shape: successful results carry
    ``data``; bad inputs carry ``error`` / ``error_kind = VALIDATION``
    / ``error_retriable = False``.

    This is the recommended pattern for:

    * Unknown or mistyped column / field / path names, with a
      "did you mean '<x>'?" suggestion baked into the message.
    * Nested-path lookups that would return all-NULL (silent-empty
      footgun) — raise with a diagnostic instead of returning ``[]``.
    * Any precondition the LLM itself could fix on the next turn by
      adjusting its arguments.

    Example::

        def rank(column: str, choices: list[str]) -> list[dict]:
            if column not in choices:
                hint = suggest_similar(column, choices)
                raise ToolValidationError(
                    f"column {column!r} not found."
                    + (f" Did you mean {hint!r}?" if hint else "")
                )
            ...

    The dispatcher classifies this as
    :attr:`ErrorKind.VALIDATION`, non-retriable. Use plain exceptions
    (or :class:`TimeoutError`, :class:`ConnectionError`, etc.) for
    infrastructure failures instead, so retries / backoff hooks can
    still kick in.
    """


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
    llm: Any = None
    """LLM backend available for tool-internal use (summarize, classify,
    extract). ``None`` in headless/test contexts.

    The loop populates this from the active backend (or from
    ``router.select("tool_internal")`` when a router is configured).
    Tool-internal calls made through ``ctx.llm`` are:

    * Tracked by :class:`RecordingLLMBackend` (same manifest, tagged
      with ``scope="tool:<name>"``)
    * Accounted for by :class:`CostTracker`
    * Visible in ``trajectory.json`` alongside the loop's own calls

    Use for single-call operations inside a tool (summarize a large
    result, classify text, extract fields). For multi-step sub-tasks,
    use :func:`run_sub_loop` instead.

    Example::

        def search(*, query: str, ctx: ToolContext) -> dict:
            raw = external_api(query)
            if len(raw) > 10_000 and ctx.llm is not None:
                summary = ctx.llm.generate(
                    f"Summarize in 3 bullets:\\n{raw[:8000]}"
                )
                return {"summary": summary, "raw_chars": len(raw)}
            return {"results": raw}
    """
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

    warnings: list[str] = field(default_factory=list)
    """Soft advisories emitted by the tool during execution.

    Populated via :meth:`warn`. After ``execute`` returns, the dispatcher
    copies this list into :attr:`ToolResult.warnings` and clears it, so
    the next call starts with a clean slate.

    Use warnings for information the caller *should* see but which is
    not a failure — e.g. "result used a low-confidence heuristic",
    "truncated to first 20 items of 3345", "column X may not contain
    the expected schema". This avoids the historical anti-pattern of
    either (a) staying silent and producing a confidently-wrong result
    or (b) failing hard with :class:`ToolValidationError` when the
    caller could still act on the partial data."""

    resources: dict[str, Any] = field(default_factory=dict)
    """Shared resource registry keyed by ``@<name>`` ref name.

    Populated by the dispatcher when the tool's ``ToolSpec.requires``
    list declares dependencies (workspace ``tool.yaml`` ``requires:``
    field). The dispatcher resolves each requested ref against the
    workspace's resource registry and hands the live instance to the
    tool through this dict.

    Example::

        # tools/read_file/tool.yaml
        # name: read_file
        # parameters: {file_path: {type: string, description: ...}}
        # requires: [workspace_config, file_cache]

        def execute(*, file_path, ctx):
            ws = ctx.resources['workspace_config'].path
            cache = ctx.resources['file_cache']
            ...

    Tools without a ``requires:`` list receive an empty dict here.
    """

    def report_progress(self, stage: str, data: dict | None = None) -> None:
        """Invoke the progress callback if one is installed. Silent if not."""
        if self.on_progress is not None:
            self.on_progress(stage, data or {})

    def warn(self, message: str) -> None:
        """Record a soft advisory about the in-flight tool result.

        Complements :class:`ToolValidationError` — which aborts the
        call — and plain errors — which report pure failure. A warning
        lets the tool say "here is your data, *and* you should know
        something about how I got it". The dispatcher attaches every
        warning to :attr:`ToolResult.warnings` for the agent to see.

        Example::

            def detect_timestamp(ctx, rows):
                col = pick_timestamp_column(rows)
                if col.confidence < 0.7:
                    ctx.warn(
                        f"time column {col.name!r} was a low-confidence "
                        f"guess — results may be inaccurate"
                    )
                return col.name
        """
        if message:
            self.warnings.append(str(message))

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
    to work with the looplet pipeline loop. Implementations are
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
    step_context: dict[str, Any] = field(default_factory=dict)
    """Ephemeral per-step shared state for hook-to-hook communication.

    The loop clears this dict at the start of every step.  Hooks write
    to it during the step (e.g. ``state.step_context["entities"] = [...]``);
    other hooks read from it within the same step.  Unlike ``metadata``
    (which persists across the entire run), ``step_context`` is scoped
    to a single step and automatically cleaned up.

    Use this instead of ``metadata`` for data that is:
    - Produced by one hook and consumed by another in the same step
    - Not meaningful after the step completes
    - Not part of the agent's persistent state
    """

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - len(self.steps))

    def context_summary(self) -> str:
        """Render recent step results into the LLM's context.

        Mirrors Claude Code's tool-result-block pattern: the model
        sees the **actual data** that previous tools returned, not a
        digest. This eliminates the fabrication failure mode where
        the model invents plausible content for a chained tool's
        argument because the previous step's data was elided.

        Three nested budgets (centralised in
        :mod:`looplet.context_budget`):

        * ``CONTEXT_WINDOW_STEPS`` — sliding window of recent steps
          (older are out of scope for this layer; the compact layer
          handles them).
        * ``CONTEXT_INLINE_PER_STEP_CHARS`` — per-step soft cap;
          longer steps get a "[truncated; full result N chars]" tail.
        * ``CONTEXT_WINDOW_TOTAL_CHARS`` — aggregate cap across the
          whole window. When exceeded, the largest step contributions
          are progressively truncated until the total fits.
        """
        if not self.steps:
            return ""

        # Lazy import keeps the module-import order stable and lets
        # tests monkeypatch budgets without re-importing types.
        import json as _json  # noqa: PLC0415

        from looplet.context_budget import (  # noqa: PLC0415
            CONTEXT_INLINE_PER_STEP_CHARS,
            CONTEXT_WINDOW_STEPS,
            CONTEXT_WINDOW_TOTAL_CHARS,
        )

        # Pass 1: render each step in the window with per-step cap.
        # The first tuple element is just an ordering tag (we don't use
        # it for anything but stable iteration) so ``int`` covers both
        # the real step number and the ``id(step)`` fallback.
        rendered: list[tuple[int, str]] = []
        for step in self.steps[-CONTEXT_WINDOW_STEPS:]:
            tr = getattr(step, "tool_result", None)
            if tr is None:
                rendered.append((id(step), str(step)))
                continue
            tool_name = getattr(tr, "tool", "?")
            args_summary = getattr(tr, "args_summary", "") or ""
            err = getattr(tr, "error", None)
            data = getattr(tr, "data", None)
            raw_number = getattr(step, "number", None)
            number_int = int(raw_number) if isinstance(raw_number, int) else len(rendered) + 1
            number_label = str(raw_number) if raw_number is not None else str(number_int)
            if err:
                block = f"S{number_label} ✗ {tool_name}({args_summary}) → ERROR: {err[:200]}"
            else:
                # Show the actual data so the model can reference it
                # verbatim on the next turn (e.g. pipe ``commits`` from
                # ``fetch_commits`` into ``group_by_type``). Pre-truncated
                # at dispatch time by ``truncate_tool_result``; this layer
                # caps the *serialized* form for prompt assembly.
                payload = _json.dumps(data, default=str, ensure_ascii=False)
                if len(payload) > CONTEXT_INLINE_PER_STEP_CHARS:
                    keep = CONTEXT_INLINE_PER_STEP_CHARS
                    payload = (
                        payload[:keep] + f"\n... [truncated; full result {len(payload)} chars]"
                    )
                block = f"S{number_label} ✓ {tool_name}({args_summary}) → {payload}"
            rendered.append((number_int, block))

        # Pass 2: enforce aggregate cap. When the total exceeds budget,
        # progressively shrink the LARGEST entries (not the most recent)
        # so the most recent step retains as much detail as possible.
        # We use a two-pointer / repeated-shrink loop because greedy
        # one-shot truncation can over-cut a single block when many
        # blocks are large.
        def _total(blocks: list[tuple[int, str]]) -> int:
            return sum(len(b) for _, b in blocks) + (len(blocks) - 1) * 2  # "\n\n" joins

        while _total(rendered) > CONTEXT_WINDOW_TOTAL_CHARS:
            # Find the largest block index.
            largest_idx = max(range(len(rendered)), key=lambda i: len(rendered[i][1]))
            num, block = rendered[largest_idx]
            # If we can still meaningfully shrink it, halve it; otherwise drop oldest.
            if len(block) > 200:
                shrunk_to = max(200, len(block) // 2)
                rendered[largest_idx] = (
                    num,
                    block[:shrunk_to] + f"\n... [aggregate-cap truncated; was {len(block)} chars]",
                )
            else:
                # Every block is already minimal; drop the oldest until under cap.
                rendered.pop(0)
                if not rendered:
                    break
        return "\n\n".join(b for _, b in rendered)

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

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form annotations for external hooks.

    Hooks can attach arbitrary tags here (e.g. ``ledger_node_id``,
    ``provenance_source``, ``policy_version``) before or after
    dispatch; ``TrajectoryRecorder`` copies the dict into the
    matching ``StepRecord.metadata['tool_call']`` so the annotation
    survives in saved trajectories."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging or context assembly."""
        return {
            "tool": self.tool,
            "args": self.args,
            "reasoning": self.reasoning,
            "call_id": self.call_id,
            "metadata": dict(self.metadata),
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

    warnings: list[str] = field(default_factory=list)
    """Soft advisories emitted by the tool alongside a successful result.

    Populated from :attr:`ToolContext.warnings` by the dispatcher when a
    tool calls :meth:`ToolContext.warn`. Unlike :attr:`error`, a
    warning does not indicate failure — the ``data`` field still carries
    the tool's output. Rendered into :meth:`to_dict` so agents see them
    when building their next-step prompt.

    Example: a timestamp-detection tool returns the column name in
    ``data`` but adds a warning when the pick was a low-confidence
    substring match."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form annotations for external hooks.

    Hooks can attach arbitrary tags to a tool result (e.g.
    ``ledger_node_id``, ``credit_score``, ``confidence``) inside
    ``post_dispatch``. ``TrajectoryRecorder`` copies the dict into the
    matching ``StepRecord.metadata['tool_result']`` so the annotation
    survives in saved trajectories."""

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
        if self.warnings:
            d["warnings"] = list(self.warnings)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
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
            "tool_call": self.tool_call.to_dict(),
            "tool_result": self.tool_result.to_dict(),
        }

    def summary(self) -> str:
        """One-line human-readable summary for compact context assembly."""
        r = self.tool_result
        if r.error:
            return f"S{self.number} ✗ {r.tool}({r.args_summary}) → ERROR: {r.error[:60]}"
        if isinstance(r.data, list):
            return f"S{self.number} ✓ {r.tool}({r.args_summary}) → {len(r.data)} items"
        if isinstance(r.data, dict):
            total = r.data.get("total", r.data.get("total_items"))
            if total is not None:
                return f"S{self.number} ✓ {r.tool}({r.args_summary}) → {total}"
            # Show a compact preview of the dict
            preview = ", ".join(
                f"{k}: {v!r}" if not isinstance(v, (list, dict)) else f"{k}: ({len(v)})"
                for k, v in list(r.data.items())[:3]
            )
            if len(r.data) > 3:
                preview += f", … ({len(r.data)} keys)"
            return f"S{self.number} ✓ {r.tool}({r.args_summary}) → {preview}"
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
        elif isinstance(r.data, dict) and r.data.get("needs_approval"):
            # Surface approval-gated tool calls clearly — otherwise
            # ApprovalHook silently stops the loop and the user sees
            # a ✓ step with no indication that anything is pending.
            desc = str(r.data.get("approval_description", "")).strip()
            tail = f"⏸ awaiting approval: {desc[:80]}" if desc else "⏸ awaiting approval"
        elif isinstance(r.data, list):
            tail = f"{len(r.data)} items"
        elif isinstance(r.data, dict):
            # Prefer showing list-valued fields — a {"files": [...]} or
            # {"results": [...]} shape is common and "N keys" throws
            # away the most useful part for both humans and skimmable
            # eval output.
            list_keys = [(k, v) for k, v in r.data.items() if isinstance(v, list)]
            if len(list_keys) == 1:
                k, v = list_keys[0]
                tail = f"{len(v)} {k}"
            elif len(r.data) == 1:
                # Single-scalar-value dict (e.g. {"answer": "..."} from
                # a done tool) — show the value, not "1 keys".
                k, v = next(iter(r.data.items()))
                snippet = str(v)
                snippet = snippet if len(snippet) <= 60 else snippet[:57] + "..."
                tail = f"{k}: {snippet}"
            else:
                tail = f"{len(r.data)} keys"
        elif r.data is None:
            tail = ""
        else:
            snippet = str(r.data)
            tail = snippet if len(snippet) <= 60 else snippet[:57] + "..."
        ms = f" [{r.duration_ms:.0f}ms]" if r.duration_ms > 0 else ""
        return f"{header} → {tail}{ms}" if tail else f"{header}{ms}"
