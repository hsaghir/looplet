"""Composable agent loop — domain-agnostic hook-based architecture.

The loop handles orchestration: LLM call → parse → dispatch → continue/stop.
Domain-specific behavior is injected via hooks and LoopConfig callables.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generator, Protocol, runtime_checkable

from openharness.checkpoint import (
    Checkpoint as _Checkpoint,
)
from openharness.checkpoint import (
    FileCheckpointStore as _FileCheckpointStore,
)
from openharness.checkpoint import (
    resume_loop_state as _resume_loop_state,
)
from openharness.flags import FLAGS
from openharness.history import HistoryRecorder
from openharness.parse import coerce_text, parse_multi_tool_calls, parse_native_tool_use
from openharness.recovery import FailureScenario as _FailureScenario
from openharness.recovery_strategies import (
    rebuild_prompt as _rebuild_prompt,
)
from openharness.recovery_strategies import (
    recovery_aggressive_budget as _recovery_aggressive_budget,
)
from openharness.recovery_strategies import (
    recovery_clear_old_results as _recovery_clear_old_results,
)
from openharness.recovery_strategies import (
    recovery_reactive_compact as _recovery_reactive_compact,
)
from openharness.scaffolding import (
    PARSE_RECOVERY_MAX,
    LLMResult,
    build_parse_recovery_prompt,
    estimate_prompt_tokens,
    llm_call_with_retry,
    truncate_tool_result,
)
from openharness.session import SessionLog
from openharness.tools import BaseToolRegistry
from openharness.types import Step, ToolCall, ToolContext, ToolResult
from openharness.validation import validate_args as _validate_args

if TYPE_CHECKING:
    from openharness.permissions import PermissionEngine
    from openharness.types import CancelToken

# streaming imports are lazy (inside composable_loop) to avoid circular import:
# streaming.py imports openharness.loop.LoopHook

logger = logging.getLogger(__name__)


# ── Hook Protocol ────────────────────────────────────────────────


@runtime_checkable
class LoopHook(Protocol):
    """Protocol for composable loop hooks.

    Hooks inject domain-specific behavior into the generic agent loop.
    All methods are optional — implement only what you need.

    Hook methods (called in this order per step):
        pre_loop:      once at loop start — setup, state initialization
        pre_prompt:    before each LLM prompt — inject briefing text
        pre_dispatch:  before each tool call — intercept/cache results
        post_dispatch: after each tool call — inject follow-up text
        check_done:    when done() called — reject premature completion
        should_stop:   after each step — force early termination
        on_loop_end:   once after loop exits — cleanup, summary generation

    Args for hook methods:
        state: The agent's state object (satisfies AgentState protocol).
        session_log: The session log recording what the agent has done.
        context: Domain-specific backend (e.g. query engine, file system, API client).
            Hooks typically capture this at __init__ time; this parameter
            provides it for hooks that don't.
        step_num: Current step number in the loop.
    """

    def pre_loop(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
    ) -> None:
        """Called once at the start of the loop, before any steps.

        Use for initialization, state setup, or emitting start events.
        """
        ...

    def pre_prompt(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Called before each LLM prompt is built.

        Returns optional text to inject into the briefing section of the
        prompt.  Multiple hooks may contribute; all non-None returns are
        concatenated (subject to max_briefing_tokens).
        """
        ...

    def pre_dispatch(
        self,
        state: Any,
        session_log: SessionLog,
        tool_call: ToolCall,
        step_num: int,
    ) -> ToolResult | None:
        """Called before each tool is dispatched.

        Returns an intercepted ToolResult to skip execution, or None to
        allow normal dispatch.  The first hook to return non-None wins.
        """
        ...

    def check_permission(
        self,
        tool_call: ToolCall,
        state: Any,
    ) -> bool:
        """Called before tool dispatch to gate execution on permission.

        Returns True to allow execution, False to deny.  When denied, the
        tool call is skipped and a ToolResult with error='permission denied'
        is recorded. All hooks must return True for the call to proceed
        (AND semantics — any single deny blocks).

        Typical uses: approval gates, sandboxing, rate limits, read-only
        mode enforcement.
        """
        ...

    def post_dispatch(
        self,
        state: Any,
        session_log: SessionLog,
        tool_call: ToolCall,
        tool_result: ToolResult,
        step_num: int,
    ) -> str | None:
        """Called after each tool execution.

        All non-None returns are accumulated and injected into the next
        prompt's briefing section.
        """
        ...

    def check_done(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Called when the agent calls done().

        Returns a rejection message (string) to block premature stopping,
        or None to allow termination.
        """
        ...

    def should_stop(
        self,
        state: Any,
        step_num: int,
        new_entities: int,
    ) -> bool:
        """Called at the end of each step.

        Returns True to stop the loop early (e.g. diminishing returns
        or external signal).
        """
        ...

    def on_loop_end(
        self,
        state: Any,
        session_log: SessionLog,
        context: Any,
        llm: Any,
    ) -> int:
        """Called once after the loop exits.

        Returns an integer count of extra LLM calls made during cleanup
        (e.g. summary generation), or 0.
        """
        ...


# ── Loop Configuration ──────────────────────────────────────────


@dataclass
class LoopConfig:
    """Configuration for the composable agent loop."""

    max_steps: int = 15
    max_tokens: int = 2000
    system_prompt: str = ""
    temperature: float = 0.2
    recovery_temperature: float = 0.1
    # Name of the tool that signals task completion.
    done_tool: str = "done"

    # Domain-specific callables — injected by the agent

    build_briefing: Callable[..., str] | None = None
    """Callable[[state, session_log, context], str] — builds the briefing
    section injected at the top of each prompt."""

    extract_entities: Callable[..., list[str]] | None = None
    """Callable[[data], list[str]] — extracts entity strings from a tool
    result's data for entity tracking and session log recording."""

    build_trace: Callable[..., Any] | None = None
    """Callable[[Any, SessionLog, Any], Any] — builds the final output
    artifact from (state, session_log, context).

    Receives keyword args: task, state, session_log, done, llm, llm_calls,
    elapsed_ms.  Returns any serialisable object; stored as the generator
    return value.
    """

    build_prompt: Callable[..., str] | None = None
    """Callable[..., str] — builds the full LLM prompt from loop state.

    Receives keyword args: task, tool_catalog, state_summary,
    context_history, step_number, max_steps, session_log, briefing.
    """

    extract_step_metadata: Callable[..., tuple[list[str], list[str]]] | None = None
    """Callable[[Any, int], tuple[list[str], list[str]]] — returns
    (findings, highlights) from (state, step_num).

    Called after each non-done tool dispatch to gather per-step metadata
    for the session log.
    """

    use_native_tools: bool = field(default_factory=lambda: FLAGS.native_tools)
    """If True, pass tool schemas to the LLM and parse tool_use blocks
    instead of JSON text.  Requires LLM backend support.

    Defaults to the ``OPENHARNESS_NATIVE_TOOLS`` environment variable
    (via ``FLAGS.native_tools``), which is ``False`` when unset."""

    acceptance_criteria: list[str] | None = None
    """Optional acceptance criteria checked by quality gate hooks.
    Domain-specific: e.g. ['check at least 3 data sources'].
    """

    max_briefing_tokens: int | None = None
    """Max estimated tokens for the briefing section (all hook pre_prompt
    outputs combined).  When exceeded, later hook outputs are dropped with
    a truncation note.  None = no limit.
    """

    # ── Optional wired capabilities ──────────────────────────────

    router: Any | None = None
    """ModelRouter — when set, router.select(purpose='reasoning') is called at
    each step instead of the llm argument passed directly.
    """

    checkpoint_dir: str | None = None
    """Directory path for checkpoint files.  When set, a FileCheckpointStore
    saves a checkpoint after every step so the loop can be resumed.
    """

    tracer: Any | None = None
    """Tracer instance — when set, wraps each LLM call and tool dispatch in
    a span so call timings are recorded in the trace tree.
    """

    recovery_registry: Any | None = None
    """RecoveryRegistry — when set, consulted on PARSE_ERROR instead of the
    built-in hardcoded 3-strategy recovery chain.
    """

    output_schema: Any | None = None
    """OutputSchema — when set, validate_args(schema, done_payload) is called
    in the done() quality gate; invalid payloads are rejected with a message.
    """

    initial_checkpoint: Any | None = None
    """Checkpoint — when set, resume_loop_state(checkpoint) is called at loop
    start to restore session_log and step offset (crash-resume support).
    """

    memory_sources: list[Any] = field(default_factory=list)
    """Optional list of ``PersistentMemorySource`` objects rendered into
    the default prompt's top ``MEMORY`` section on every turn. Each
    source must expose ``load(state) -> str | None``. When ``build_prompt``
    is user-supplied, the loop still renders memory but passes it to
    the custom function as a ``memory=`` kwarg.
    """

    cancel_token: CancelToken | None = None
    """Optional :class:`openharness.types.CancelToken` that signals the
    loop should terminate. The token is threaded through every LLM call
    (forwarded to backends that accept ``cancel_token=``) and every
    ``ToolContext`` so tools share the same cancellation channel. When
    observed cancelled between turns the loop exits cleanly without
    further LLM calls.
    """

    permissions: PermissionEngine | None = None
    """Optional :class:`openharness.permissions.PermissionEngine`
    consulted before every tool dispatch. Denied calls are short-
    circuited into a synthetic ``ToolResult`` carrying a
    :class:`ToolError` with ``kind=ErrorKind.PERMISSION_DENIED``; the
    tool body is never invoked. Complements per-hook ``check_permission``
    (both run — AND semantics)."""

    elicit_handler: Callable[[str, list[str] | None], str | None] | None = None
    """Optional callable surfaced to tools via ``ToolContext.elicit``.
    Lets a tool pause and ask the caller (user, upstream agent, test
    harness) for clarification mid-execution. Signature is
    ``(prompt, options) -> str | None``; returning ``None`` means "no
    answer — proceed". Leave unset for fully-autonomous runs."""

    context_window: int = 128_000
    """Maximum context window (in tokens) for the backend.  Used by the
    pre-flight prompt-size check to decide whether to trigger reactive
    recovery *before* sending the LLM call.  Override this to match your
    actual backend's window (e.g. 200_000 for Claude, 128_000 for GPT-4o)."""


def _default_extract_entities(data: Any) -> list[str]:
    """Fallback: no entity extraction."""
    return []


def _default_build_briefing(state: Any, session_log: SessionLog, context: Any) -> str:
    """Fallback: empty briefing."""
    return ""


def _default_extract_step_metadata(state: Any, step_num: int) -> tuple[list[str], list[str]]:
    """Fallback: no step metadata extraction."""
    return [], []


def _build_tool_ctx(config: "LoopConfig") -> ToolContext | None:
    """Build a ToolContext for tool dispatch if the config has anything
    worth threading (cancel_token, elicit_handler). Returns None when
    there's nothing to carry so tools that don't opt-in are unaffected.
    """
    if config.cancel_token is None and config.elicit_handler is None:
        return None
    return ToolContext(
        cancel_token=config.cancel_token,
        elicit=config.elicit_handler,
    )


# ── Composable Agent Loop ───────────────────────────────────────


def composable_loop(
    llm: Any,
    task: dict[str, Any] | None = None,
    tools: BaseToolRegistry | None = None,
    context: Any = None,
    hooks: list[Any] | None = None,
    config: LoopConfig | None = None,
    state: Any = None,
    session_log: SessionLog | None = None,
    stream: Any | None = None,
    conversation: Any | None = None,
) -> Generator[Step, None, Any]:
    """Domain-agnostic agent loop with composable hooks.

    Yields Steps, returns a trace object built by config.build_trace.

    Args:
        llm: LLM backend (must implement generate()).
        task: The task description dict (e.g. an alert, a coding task, etc.).
        tools: Tool registry with available tools.
        context: Domain-specific backend passed to hooks and build_briefing.
            Hooks can also capture their backends at __init__ time.
        hooks: Composable hook instances for domain behavior.
        config: Loop configuration (steps, tokens, callables).
        state: Agent state (must satisfy AgentState protocol).
        session_log: Session log for recording agent memory.
        stream: Optional EventEmitter — when set, emits structured events for
            each loop lifecycle moment (start, step, LLM call, dispatch, end).
        conversation: Optional Conversation — when set, the loop auto-records
            each LLM prompt/response and tool call/result as Messages in the
            conversation thread. Works alongside session_log (both are populated).
    """
    if task is None:
        task = {}
    if tools is None:
        raise ValueError("tools is required")
    if config is None:
        config = LoopConfig()
    if hooks is None:
        hooks = []
    t0 = time.time()

    if session_log is None:
        session_log = SessionLog()

    # ── Conversation thread — always active (single source of truth) ──
    from openharness.conversation import Conversation as _Conversation  # noqa: PLC0415
    _conv = conversation if conversation is not None else _Conversation()

    # ── Unified history recorder — single write path for step/turn events ──
    _history = HistoryRecorder(
        state=state,
        session_log=session_log,
        conversation=_conv,
    )

    # ── Lazy streaming imports (avoid circular: streaming imports loop.LoopHook)
    _LoopStartEvent = _StepStartEvent = _LLMCallStartEvent = None
    _ToolDispatchEvent = _LoopEndEvent = None
    if stream is not None:
        try:
            from openharness.streaming import (
                LLMCallStartEvent as _LLMCallStartEvent,
            )
            from openharness.streaming import (
                LoopEndEvent as _LoopEndEvent,
            )
            from openharness.streaming import (  # noqa: PLC0415
                LoopStartEvent as _LoopStartEvent,
            )
            from openharness.streaming import (
                StepStartEvent as _StepStartEvent,
            )
            from openharness.streaming import (
                ToolDispatchEvent as _ToolDispatchEvent,
            )
        except ImportError:
            pass

    # ── Resolve effective LLM (router overrides direct llm) ────
    def _get_llm() -> Any:
        if config.router is not None:
            return config.router.select(purpose="reasoning")
        return llm

    # ── Checkpoint store setup ──────────────────────────────────
    _ckpt_store = None
    if config.checkpoint_dir is not None:
        _ckpt_store = _FileCheckpointStore(config.checkpoint_dir)

    # ── Crash-resume from initial checkpoint ───────────────────
    _step_offset = 0
    if config.initial_checkpoint is not None:
        resumed = _resume_loop_state(config.initial_checkpoint)
        _step_offset = resumed.get("step_offset", 0)
        # Restore session log entries into session_log
        restored_log = resumed.get("session_log")
        if restored_log is not None:
            session_log.entries = restored_log.entries[:]
            session_log.current_theory = restored_log.current_theory
        # Restore state counters (queries_used, budget_remaining) so
        # budget enforcement continues where the checkpoint left off.
        # Some state classes expose budget_remaining as a read-only property
        # derived from steps/max_steps; skip fields that can't be assigned.
        for _k, _v in (resumed.get("state_counters") or {}).items():
            try:
                setattr(state, _k, _v)
            except AttributeError:
                pass

    build_briefing = config.build_briefing or _default_build_briefing
    extract_entities = config.extract_entities or _default_extract_entities
    build_prompt_fn = config.build_prompt

    # ── Loop state ──────────────────────────────────────────
    consecutive_parse_failures = 0
    quality_gate_message = ""
    post_dispatch_parts: list[str] = []
    llm_calls = 0
    done = False
    stop_reason = "budget_exhausted"  # tracks why the loop exited
    # Recovery state — each strategy fires at most once
    recovery_state = {
        "budget_enforcement": False,
        "reactive_compact": False,
        "result_clearing": False,
    }

    extract_step_metadata = config.extract_step_metadata or _default_extract_step_metadata

    # ── Pre-loop hooks ──────────────────────────────────────────
    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            hook.pre_loop(state, session_log, context)

    # ── Emit LoopStartEvent ─────────────────────────────────────
    # Skip if a StreamingHook is in hooks — it already emits LoopStartEvent
    # from its pre_loop method and we don't want duplicates.
    from openharness.streaming import StreamingHook as _StreamingHook  # noqa: PLC0415
    _has_streaming_hook = any(isinstance(h, _StreamingHook) for h in hooks)
    if stream is not None and _LoopStartEvent is not None and not _has_streaming_hook:
        stream.emit(_LoopStartEvent(task_summary=str(task.get("id", "")), max_steps=config.max_steps))

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset

        # Cancellation check between turns — stop cleanly, no more LLM calls.
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        # ── Pre-prompt hooks ────────────────────────────────
        briefing_parts = [build_briefing(state, session_log, context)]
        _briefing_budget = config.max_briefing_tokens
        _briefing_used = len(briefing_parts[0]) // 4 if _briefing_budget else 0

        for hook in hooks:
            if hasattr(hook, "pre_prompt"):
                text = hook.pre_prompt(state, session_log, context, step_num)
                if text:
                    if _briefing_budget:
                        text_tokens = len(text) // 4
                        if _briefing_used + text_tokens > _briefing_budget:
                            briefing_parts.append(
                                "(briefing truncated — token budget exceeded)"
                            )
                            break
                        _briefing_used += text_tokens
                    briefing_parts.append(text)

        if post_dispatch_parts:
            briefing_parts.append("\n".join(post_dispatch_parts))
            post_dispatch_parts = []

        # ── Build prompt ────────────────────────────────────
        context_history = state.context_summary()
        if quality_gate_message:
            context_history += "\n" + quality_gate_message
            quality_gate_message = ""

        # Persistent memory (CLAUDE.md generalization): rendered once
        # per turn, placed above TASK by the default prompt builder.
        _memory_sources = getattr(config, "memory_sources", None)
        if _memory_sources:
            from openharness.memory import render_memory as _render_memory  # noqa: PLC0415
            _rendered_memory = _render_memory(_memory_sources, state)
        else:
            _rendered_memory = ""

        if build_prompt_fn is not None:
            prompt = build_prompt_fn(
                task=task,
                tool_catalog=tools.tool_catalog_text(),
                state_summary=state.snapshot(),
                context_history=context_history,
                step_number=step_num,
                max_steps=config.max_steps,
                session_log=session_log.render(),
                briefing="\n".join(briefing_parts),
                memory=_rendered_memory,
            )
        else:
            # Domain-agnostic default: 7-section structured prompt.
            from openharness.prompts import build_prompt as _default_build_prompt  # noqa: PLC0415
            prompt = _default_build_prompt(
                task=task,
                tool_catalog=tools.tool_catalog_text(),
                state_summary=state.snapshot(),
                context_history=context_history,
                step_number=step_num,
                max_steps=config.max_steps,
                session_log=session_log.render(),
                briefing="\n".join(briefing_parts),
                memory=_rendered_memory,
            )

        # ── Pre-flight context check ──────────────────────────
        estimated_tokens = estimate_prompt_tokens(prompt)
        preflight_too_long = estimated_tokens > config.context_window - 3_000

        # Emit StepStartEvent
        if stream is not None and _StepStartEvent is not None:
            stream.emit(_StepStartEvent(step_num=step_num))

        # Resolve effective LLM once per step — used by both the main call and
        # parse-recovery below, regardless of whether pre-flight fires.
        effective_llm = _get_llm()

        if preflight_too_long and FLAGS.reactive_recovery:
            logger.warning(
                "Pre-flight block: prompt ~%d tokens exceeds safe limit — "
                "running recovery before LLM call", estimated_tokens,
            )
            llm_result = LLMResult(None, Exception("pre-flight: prompt is too long"))
        else:
            # ── LLM call with retry + reactive recovery ───────────
            # Emit LLMCallStartEvent
            if stream is not None and _LLMCallStartEvent is not None:
                stream.emit(_LLMCallStartEvent(step_num=step_num))
            # Tracer: start span for LLM call
            _llm_span = None
            if config.tracer is not None:
                _llm_span = config.tracer.start_span(
                    f"llm.call.step_{step_num}",
                    attributes={"step": step_num},
                )
            _native_on = (config.use_native_tools or FLAGS.native_tools) and hasattr(
                effective_llm, "generate_with_tools"
            )
            _tool_schemas = tools.tool_schemas() if _native_on else None
            llm_result = llm_call_with_retry(
                effective_llm, prompt,
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                tools=_tool_schemas,
                cancel_token=config.cancel_token,
            )
            if _llm_span is not None and config.tracer is not None:
                config.tracer.end_span(_llm_span)
            llm_calls += 1

        # Reactive recovery: if prompt-too-long, try chained strategies
        if not llm_result.ok and llm_result.is_prompt_too_long and FLAGS.reactive_recovery:
            raw_response = _reactive_recovery_chain(
                llm_result, recovery_state, state, session_log, effective_llm,
                tools, context, build_briefing, build_prompt_fn,
                task, config, step_num,
            )
            llm_calls += recovery_state.get("_last_recovery_llm_calls", 0)
            llm_result = LLMResult(raw_response)

        raw_response = llm_result.text

        # ── Record LLM turn in conversation thread via unified recorder ──
        _history.record_llm_turn(prompt=prompt, response=raw_response)

        if raw_response is None:
            # If cancellation caused the failure, exit cleanly — no error step.
            if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
                stop_reason = "cancelled"
                break
            logger.error("LLM call failed after retries at step %d", step_num)
            error_call = ToolCall(tool="__llm_error__", reasoning="LLM call failed after retries")
            error_result = ToolResult(
                tool="__llm_error__", args_summary="", data=None,
                error="LLM call failed after all retry attempts",
            )
            step = Step(number=step_num, tool_call=error_call, tool_result=error_result)
            state.steps.append(step)
            yield step
            _history.record_step(step, theory="", entities=[], findings=[], highlights=[], recall_key="")
            break

        # ── Parse response (native tool_use or JSON text) ────
        if (config.use_native_tools or FLAGS.native_tools) and isinstance(raw_response, list):
            tool_calls = parse_native_tool_use(raw_response)
        else:
            tool_calls = parse_multi_tool_calls(raw_response)
        if not tool_calls:
            consecutive_parse_failures += 1
            # Consult recovery_registry if set — use returned action
            _recovery_action = None
            if config.recovery_registry is not None:
                _recovery_action = config.recovery_registry.attempt_recovery(
                    _FailureScenario.PARSE_ERROR,
                    {"step": step_num, "raw_response": raw_response},
                )
                if _recovery_action is not None and _recovery_action.action_type == "abort":
                    logger.warning("Recovery registry aborted parse recovery at step %d", step_num)
                    tool_call = ToolCall(tool="__parse_error__", reasoning=(coerce_text(raw_response) or "")[:200])
                    tool_result = ToolResult(
                        tool="__parse_error__", args_summary="", data=None,
                        error=f"Parse error — recovery aborted: {_recovery_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                    state.steps.append(step)
                    yield step
                    _history.record_step(step, theory="", entities=[], findings=[], highlights=[], recall_key="")
                    continue
                if _recovery_action is not None and _recovery_action.message:
                    post_dispatch_parts.append(_recovery_action.message)
            if consecutive_parse_failures <= PARSE_RECOVERY_MAX:
                logger.warning(
                    "Parse failure %d/%d at step %d — attempting recovery",
                    consecutive_parse_failures, PARSE_RECOVERY_MAX, step_num,
                )
                recovery_prompt = build_parse_recovery_prompt(prompt, coerce_text(raw_response) or "")
                recovery_result = llm_call_with_retry(
                    effective_llm, recovery_prompt,
                    max_tokens=config.max_tokens,
                    system_prompt=config.system_prompt,
                    temperature=config.recovery_temperature,
                    cancel_token=config.cancel_token,
                )
                llm_calls += 1
                if recovery_result.ok:
                    tool_calls = parse_multi_tool_calls(recovery_result.text)

            if not tool_calls:
                logger.warning("Unparseable LLM response at step %d after recovery", step_num)
                tool_call = ToolCall(tool="__parse_error__", reasoning=(coerce_text(raw_response) or "")[:200])
                tool_result = ToolResult(
                    tool="__parse_error__", args_summary="", data=None,
                    error=f"Could not parse JSON: {(coerce_text(raw_response) or '')[:200]}",
                )
                step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                _history.record_step(step, theory="", entities=[], findings=[], highlights=[], recall_key="")
                continue
        else:
            consecutive_parse_failures = 0

        # ── Dispatch tool calls ──────────────────────────────
        all_step_entities: list[str] = []

        done_tool_name = config.done_tool
        done_idx = None
        for i, tc in enumerate(tool_calls):
            if tc.tool == done_tool_name:
                done_idx = i
                break

        # Dispatch non-done tools
        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls
        if regular_calls:
            # Pre-dispatch hooks: allow hooks to intercept/block calls
            intercepted_results: dict[int, ToolResult] = {}
            for tc_idx, tc in enumerate(regular_calls):
                for hook in hooks:
                    if hasattr(hook, "pre_dispatch"):
                        cached = hook.pre_dispatch(state, session_log, tc, step_num + tc_idx)
                        if cached is not None:
                            intercepted_results[tc_idx] = cached
                            break

            calls_to_dispatch = [
                tc for i, tc in enumerate(regular_calls) if i not in intercepted_results
            ]

            # Permission check: engine + hooks can deny tool calls (AND semantics)
            permitted_calls = []
            for tc in calls_to_dispatch:
                denied = False
                # Declarative PermissionEngine first — if present.
                if config.permissions is not None:
                    outcome = config.permissions.evaluate(tc)
                    if outcome.denied:
                        from openharness.types import ErrorKind, ToolError  # noqa: PLC0415
                        idx = next(i for i, c in enumerate(regular_calls) if c is tc)
                        _te = ToolError(
                            kind=ErrorKind.PERMISSION_DENIED,
                            message=outcome.reason
                                or f"Permission denied for tool '{tc.tool}'",
                            retriable=False,
                        )
                        intercepted_results[idx] = ToolResult(
                            tool=tc.tool, args_summary=str(tc.args)[:100],
                            data=None,
                            error=_te.message,
                            error_detail=_te,
                        )
                        denied = True
                if denied:
                    continue
                for hook in hooks:
                    if hasattr(hook, "check_permission"):
                        if not hook.check_permission(tc, state):
                            from openharness.types import ErrorKind, ToolError  # noqa: PLC0415
                            idx = next(i for i, c in enumerate(regular_calls) if c is tc)
                            _te = ToolError(
                                kind=ErrorKind.PERMISSION_DENIED,
                                message=f"Permission denied for tool '{tc.tool}'",
                                retriable=False,
                            )
                            intercepted_results[idx] = ToolResult(
                                tool=tc.tool, args_summary=str(tc.args)[:100],
                                data=None, error=_te.message, error_detail=_te,
                            )
                            denied = True
                            break
                if not denied:
                    permitted_calls.append(tc)

            if permitted_calls:
                _tool_ctx = _build_tool_ctx(config)
                _dispatch_kw = {"ctx": _tool_ctx} if _tool_ctx is not None else {}
                if FLAGS.concurrent_dispatch:
                    dispatch_results = tools.dispatch_batch(permitted_calls, **_dispatch_kw)
                else:
                    dispatch_results = [tools.dispatch(c, **_dispatch_kw) for c in permitted_calls]
            else:
                dispatch_results = []

            dispatch_iter = iter(dispatch_results)
            batch_results = []
            for i in range(len(regular_calls)):
                if i in intercepted_results:
                    batch_results.append(intercepted_results[i])
                else:
                    batch_results.append(next(dispatch_iter))

            for tc_idx, (tool_call, tool_result) in enumerate(zip(regular_calls, batch_results)):
                cur_step = step_num + tc_idx
                was_intercepted = tc_idx in intercepted_results
                tool_spec = tools._tools.get(tool_call.tool)
                if not (tool_spec and tool_spec.free) and not was_intercepted:
                    state.queries_used += 1

                tool_result.data = truncate_tool_result(tool_result.data)

                # Emit ToolDispatchEvent
                if stream is not None and _ToolDispatchEvent is not None:
                    stream.emit(_ToolDispatchEvent(
                        step_num=cur_step,
                        tool_name=tool_call.tool,
                        args_summary=str(tool_call.args)[:200],
                    ))

                for hook in hooks:
                    if hasattr(hook, "post_dispatch"):
                        text = hook.post_dispatch(
                            state, session_log, tool_call, tool_result, cur_step,
                        )
                        if text:
                            post_dispatch_parts.append(text)

                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                cur_step_count = state.step_count
                state.steps.append(step)
                yield step

                # Save checkpoint after each step
                if _ckpt_store is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={"entries": session_log.to_list(), "current_theory": session_log.current_theory},
                            conversation_data=_conv.serialize(),
                            config_snapshot={
                                "max_steps": config.max_steps,
                                "queries_used": getattr(state, "queries_used", 0),
                                "budget_remaining": getattr(state, "budget_remaining", 0),
                            },
                            tool_results_store={},
                            metadata={"task": str(task)},
                        ),
                        key=f"step_{cur_step}",
                    )

                step_findings, step_highlights = extract_step_metadata(state, cur_step_count)
                step_entities = extract_entities(tool_result.data)
                all_step_entities.extend(step_entities)
                recall_key = tool_result.result_key or ""
                theory = tool_call.args.get("__theory__", "")

                # Unified write: state.steps was already appended above, so the
                # recorder dedups there and fills in the session log + conversation.
                _history.record_step(
                    step,
                    theory=theory,
                    entities=step_entities,
                    findings=step_findings,
                    highlights=step_highlights,
                    recall_key=recall_key,
                )

        # Handle done() if present
        if done_idx is not None:
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    w = hook.check_done(state, session_log, context, step_num)
                    if w is not None:
                        gate_warning = w
                        break

            # Output schema validation — reject done() if payload is invalid
            if gate_warning is None and config.output_schema is not None:
                validation = _validate_args(config.output_schema, tool_call.args)
                if not validation.valid:
                    gate_warning = f"Output schema validation failed: {'; '.join(validation.errors)}"

            # Emit ToolDispatchEvent for done
            if stream is not None and _ToolDispatchEvent is not None:
                stream.emit(_ToolDispatchEvent(
                    step_num=cur_step,
                    tool_name=tool_call.tool,
                    args_summary=str(tool_call.args)[:200],
                ))

            if gate_warning is not None:
                logger.info("Quality gate rejected done() at step %d", step_num)
                quality_gate_message = gate_warning
                tool_result = ToolResult(
                    tool=done_tool_name, args_summary="rejected",
                    data={"rejected": True, "reason": gate_warning},
                )
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                _history.record_step(
                    step,
                    theory="",
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key="",
                )
            else:
                # done() dispatch intentionally bypasses PermissionEngine — it's
                # a loop signal, not a side-effecting tool. Permission-gating a
                # termination signal would prevent the agent from ever stopping.
                _ctx = _build_tool_ctx(config)
                tool_result = tools.dispatch(tool_call, ctx=_ctx) if _ctx is not None else tools.dispatch(tool_call)
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                # Record accepted done() to session_log + conversation
                _history.record_step(
                    step,
                    theory=tool_call.args.get("__theory__", ""),
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key="",
                )
                # Save checkpoint after done step (after yield, matching non-done pattern)
                if _ckpt_store is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={"entries": session_log.to_list(), "current_theory": session_log.current_theory},
                            conversation_data=_conv.serialize(),
                            config_snapshot={
                                "max_steps": config.max_steps,
                                "queries_used": getattr(state, "queries_used", 0),
                                "budget_remaining": getattr(state, "budget_remaining", 0),
                            },
                            tool_results_store={},
                            metadata={"task": str(task), "status": "done"},
                        ),
                        key=f"step_{cur_step}_done",
                    )
                done = True
                stop_reason = "done"

        if done:
            continue

        for hook in hooks:
            if hasattr(hook, "should_stop"):
                if hook.should_stop(state, step_num, len(all_step_entities)):
                    logger.info("Hook %s requested stop at step %d", type(hook).__name__, step_num)
                    done = True
                    stop_reason = "hook_stop"
                    break
        if done:
            break

    # ── Post-loop hooks ─────────────────────────────────────
    # Stash stop_reason on state so hooks (e.g. StreamingHook) can read it
    if state is not None:
        state._stop_reason = stop_reason
    for hook in hooks:
        if hasattr(hook, "on_loop_end"):
            extra = hook.on_loop_end(state, session_log, context, llm)
            if isinstance(extra, int):
                llm_calls += extra

    # Emit LoopEndEvent — skip if StreamingHook already emits it
    if stream is not None and _LoopEndEvent is not None and not _has_streaming_hook:
        stream.emit(_LoopEndEvent(
            total_steps=state.step_count,
            total_llm_calls=llm_calls,
            reason=stop_reason,
        ))

    # Build trace via injected callable
    elapsed = (time.time() - t0) * 1000
    if config.build_trace is not None:
        trace = config.build_trace(
            task=task, state=state, session_log=session_log,
            done=stop_reason == "done", llm=llm, llm_calls=llm_calls, elapsed_ms=elapsed,
        )
    else:
        trace = {
            "task": task, "steps": [s.to_dict() for s in state.steps],
            "llm_calls": llm_calls, "total_time_ms": elapsed,
            "conversation": _conv,
        }
    return trace


# ── Reactive Recovery Chain ──────────────────────────────────────


def _reactive_recovery_chain(
    llm_result: Any,
    recovery_state: dict,
    state: Any,
    session_log: Any,
    llm: Any,
    tools: Any,
    context: Any,
    build_briefing: Any,
    build_prompt_fn: Any,
    task: dict,
    config: Any,
    step_num: int,
) -> Any:
    """Multi-strategy recovery chain for prompt-too-long errors.

    Tries strategies in order, each at most once:
      1. Aggressive budget enforcement (shrink all results to 2KB)
      2. Emergency session log compression (reactive_compact)
      3. Clear all old result data entirely

    After each strategy, rebuilds the prompt and retries the LLM call.
    Returns the response text if recovery succeeds, None if all fail.
    """
    extra_llm_calls = 0

    strategies = [
        ("budget_enforcement", _recovery_aggressive_budget),
        ("reactive_compact", _recovery_reactive_compact),
        ("result_clearing", _recovery_clear_old_results),
    ]

    for name, strategy_fn in strategies:
        if recovery_state.get(name):
            continue

        recovery_state[name] = True
        logger.warning("Recovery strategy '%s' at step %d", name, step_num)

        strategy_llm_calls = strategy_fn(state, session_log, llm, step_num)
        extra_llm_calls += strategy_llm_calls

        prompt = _rebuild_prompt(
            state, session_log, context, build_briefing, build_prompt_fn,
            task, tools, config, step_num,
        )
        retry_result = llm_call_with_retry(
            llm, prompt,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            cancel_token=config.cancel_token,
        )
        extra_llm_calls += 1

        if retry_result.ok:
            logger.info("Recovery strategy '%s' succeeded at step %d", name, step_num)
            recovery_state["_last_recovery_llm_calls"] = extra_llm_calls
            return retry_result.text

        if not retry_result.is_prompt_too_long:
            break

    logger.error("All recovery strategies exhausted at step %d", step_num)
    recovery_state["_last_recovery_llm_calls"] = extra_llm_calls
    return None
