"""Composable agent loop — domain-agnostic hook-based architecture.

The loop handles orchestration: LLM call → parse → dispatch → continue/stop.
Domain-specific behavior is injected via hooks and LoopConfig callables.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, Any, Callable, Generator, Protocol, runtime_checkable

from looplet.checkpoint import (
    Checkpoint as _Checkpoint,
)
from looplet.checkpoint import (
    FileCheckpointStore as _FileCheckpointStore,
)
from looplet.checkpoint import (
    resume_loop_state as _resume_loop_state,
)
from looplet.history import HistoryRecorder
from looplet.hook_decision import normalize_hook_return
from looplet.parse import parse_multi_tool_calls, parse_native_tool_use, to_text
from looplet.recovery import FailureScenario as _FailureScenario
from looplet.recovery_strategies import (
    rebuild_prompt as _rebuild_prompt,
)
from looplet.recovery_strategies import (
    recovery_aggressive_budget as _recovery_aggressive_budget,
)
from looplet.recovery_strategies import (
    recovery_clear_old_results as _recovery_clear_old_results,
)
from looplet.scaffolding import (
    PARSE_RECOVERY_MAX,
    LLMResult,
    build_parse_recovery_prompt,
    estimate_prompt_tokens,
    llm_call_with_retry,
    truncate_tool_result,
)
from looplet.session import SessionLog
from looplet.tools import BaseToolRegistry, _summarize_args_dict
from looplet.types import AgentState, Step, ToolCall, ToolContext, ToolResult
from looplet.validation import validate_args as _validate_args

if TYPE_CHECKING:
    from looplet.cache import CachePolicy
    from looplet.checkpoint import Checkpoint
    from looplet.compact import CompactService
    from looplet.conversation import Conversation
    from looplet.recovery import RecoveryRegistry
    from looplet.router import ModelRouter
    from looplet.telemetry import Tracer
    from looplet.types import CancelToken, LLMBackend
    from looplet.validation import OutputSchema

# streaming imports are lazy (inside composable_loop) to avoid circular import:
# streaming.py imports looplet.loop.LoopHook

logger = logging.getLogger(__name__)


# ── Hook Protocol ────────────────────────────────────────────────


@runtime_checkable
class LoopHook(Protocol):
    """Protocol for composable loop hooks.

    Hooks inject domain-specific behavior into the generic agent loop.
    All methods are optional — implement only what you need.

    Minimal example::

        class MyHook:
            def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
                if tool_call.tool == "write":
                    return InjectContext("Remember to write tests too.")
                return None

            def check_done(self, state, session_log, context, step_num):
                if not self._tests_passed:
                    return HookDecision(block="Run tests first.")
                return None

            # Only implement what you need. All other methods are optional.

    Hook methods (called in this order per step):
        pre_loop:        once at loop start
        pre_prompt:      before each LLM call — inject briefing text (additive, all hooks contribute)
        build_briefing:  override the briefing section (first hook returning non-None wins)
        build_prompt:    override the entire prompt (first hook returning non-None wins)
        pre_dispatch:    before each tool call — intercept, cache, or deny
        post_dispatch:   after each tool call — inject follow-up context
        check_done:      when done() is called — reject premature completion
        should_stop:     after each step — force early termination
        should_compact:  at step start — trigger proactive compaction
        on_loop_end:     once after loop exits — cleanup

    Return types:
        All hook methods accept ``HookDecision`` as return type.
        Legacy returns (``str``, ``bool``, ``None``) still work
        and are auto-converted via ``normalize_hook_return()``.

    Precedence for prompt injection:
        1. ``build_prompt`` hook (first non-None wins) — full prompt override
        2. ``config.build_prompt`` callable — full prompt override
        3. ``build_briefing`` hook (first non-None wins) — briefing section only
        4. ``config.build_briefing`` callable — briefing section only
        5. ``pre_prompt`` hooks — additive text appended to briefing
        6. Default 7-section template from ``looplet.prompts``
    """

    def pre_loop(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
    ) -> None:
        """Called once at the start of the loop, before any steps.

        Use for initialization, state setup, or emitting start events.
        """
        ...

    def pre_prompt(
        self,
        state: AgentState,
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
        state: AgentState,
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
        state: AgentState,
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
        state: AgentState,
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
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        step_num: int,
        tool_call: "ToolCall | None" = None,
    ) -> str | None:
        """Called when the agent calls done().

        Returns a rejection message (string) to block premature stopping,
        or None to allow termination.

        ``tool_call`` carries the candidate ``done()`` invocation (with
        its proposed final-answer arguments) so quality gates can
        inspect the agent's pending answer. The loop dispatches with or
        without this kwarg based on the hook's signature, so existing
        ``check_done(self, state, session_log, context, step_num)``
        implementations continue to work unchanged.
        """
        ...

    def should_stop(
        self,
        state: AgentState,
        step_num: int,
        new_entities: int,
    ) -> bool:
        """Called at the end of each step.

        Returns True to stop the loop early (e.g. diminishing returns
        or external signal).
        """
        ...

    def should_compact(
        self,
        state: AgentState,
        session_log: SessionLog,
        conversation: Conversation | None,
        step_num: int,
    ) -> bool:
        """Called at the top of each step, before prompt build.

        Returns True to proactively trigger the configured
        :class:`looplet.compact.CompactService` before the next
        LLM call. Complements the reactive path (which fires only on
        a ``prompt_too_long`` error) — use this when you want to
        preempt context pressure based on message count, token
        estimates, or wall-clock heuristics. If any hook returns
        True the compaction runs; otherwise the step proceeds as
        usual.
        """
        ...

    def build_briefing(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
    ) -> str | None:
        """Optional per-hook briefing builder.

        When any hook returns a non-``None`` string, the loop uses
        that text as the briefing for the current step and skips
        :attr:`LoopConfig.build_briefing` / the default. First hook
        wins — subsequent hooks don't run for this slot.

        Intended for domain adapters that want to bundle briefing
        logic alongside the rest of their hook surface instead of
        wiring it through ``config.build_briefing`` separately.
        Return ``None`` to pass through to the config/default.
        """
        ...

    def build_prompt(
        self,
        *,
        task: Any,
        tool_catalog: str,
        state_summary: dict,
        context_history: str,
        step_number: int,
        max_steps: int,
        session_log: str,
        briefing: str,
        memory: str,
    ) -> str | None:
        """Optional per-hook prompt builder.

        When any hook returns a non-``None`` string, the loop uses
        that as the full prompt for the current step and skips
        :attr:`LoopConfig.build_prompt` / the default 7-section
        template. First hook wins. Return ``None`` to pass through.

        The kwargs mirror :func:`looplet.prompts.build_prompt`
        so implementations can extend the default template by
        calling the reference implementation themselves and
        splicing extra sections.
        """
        ...

    def on_loop_end(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        llm: LLMBackend,
    ) -> int:
        """Called once after the loop exits.

        Returns an integer count of extra LLM calls made during cleanup
        (e.g. summary generation), or 0.
        """
        ...

    def on_event(self, payload: Any) -> Any:
        """Single-slot subscriber for named lifecycle events.

        Complements the per-method hook API. Hooks that implement this
        method receive an :class:`looplet.events.EventPayload`
        with ``event: LifecycleEvent`` and slot-specific fields
        populated. Returning a :class:`HookDecision` for
        :attr:`LifecycleEvent.PRE_TOOL_USE`,
        :attr:`LifecycleEvent.POST_TOOL_USE`, or
        :attr:`LifecycleEvent.STOP` has the same effect as returning
        it from the matching per-method hook.

        Implementing ``on_event`` is strictly additive — hooks can
        still implement per-method slots and mix both styles.
        """
        ...


# ── Loop Configuration ──────────────────────────────────────────


@dataclass
class DomainAdapter:
    """Bundle the five domain-specific callables a loop needs.

    Grouping keeps :class:`LoopConfig` flat and readable, and gives
    agent packages a single handle to pass around instead of threading
    five separate callables.

    When a :class:`LoopConfig` has a :attr:`LoopConfig.domain` set,
    each adapter field seeds the corresponding flat ``LoopConfig``
    field only if the flat field is ``None``. Per-field overrides
    on ``LoopConfig`` therefore win over the bundled adapter, which
    wins over the loop's built-in defaults.

    All fields are optional — provide only what differs from the
    defaults. See :class:`LoopConfig` for callable signatures.
    """

    build_briefing: Callable[..., str] | None = None
    extract_entities: Callable[..., list[str]] | None = None
    build_trace: Callable[..., Any] | None = None
    build_prompt: Callable[..., str] | None = None
    extract_step_metadata: Callable[..., tuple[list[str], list[str]]] | None = None


@dataclass
class LoopConfig:
    """Configuration for the composable agent loop.

    **Start here** — most fields have sensible defaults. For your
    first agent, you only need::

        config = LoopConfig(max_steps=10)
        state = DefaultState(max_steps=10)  # must match

    Essential fields (set these first):
      - ``max_steps`` — how many tool calls before the loop stops.
        **Important:** also pass the same value to ``DefaultState(max_steps=N)``
        — the state tracks budget_remaining, config tracks the loop limit.
        If they differ, the lower one wins.
      - ``system_prompt`` — who the agent is
      - ``compact_service`` — how to manage growing context
      - ``checkpoint_dir`` — crash-safe auto-resume (one directory path)

    Everything else is optional and can be added later as your agent
    matures. See the tutorial in README.md for a progressive walkthrough.
    """

    max_steps: int = 15
    max_tokens: int = 2000
    system_prompt: str = ""
    temperature: float = 0.2
    recovery_temperature: float = 0.1
    # Name of the tool that signals task completion.
    done_tool: str = "done"

    max_turn_continuations: int = 0
    """When > 0, ``llm_call_with_retry`` will issue up to this many
    follow-up calls for a single step if the backend reports
    ``stop_reason == "max_tokens"`` mid-response, concatenating outputs
    so long thoughts aren't truncated. Requires the backend to expose
    ``last_stop_reason`` after each call. Default ``0`` (off)."""

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

    domain: "DomainAdapter | None" = None
    """Optional :class:`DomainAdapter` bundling the five domain callables
    above. When set, each adapter field seeds the matching flat field
    above only if that flat field is ``None``. Direct field assignments
    on :class:`LoopConfig` therefore win over the adapter. Prefer this
    when composing a reusable agent package — pass one adapter instead
    of threading five callables through config kwargs."""

    use_native_tools: bool = False
    """If True, pass tool schemas to the LLM and parse tool_use blocks
    instead of JSON text. Requires LLM backend support."""

    concurrent_dispatch: bool = False
    """If True, dispatch non-dependent tool calls in parallel via
    ThreadPoolExecutor. Default False — some backends and tools
    are not thread-safe."""

    reactive_recovery: bool = True
    """If True, attempt multi-strategy recovery when a prompt exceeds
    the context window (prompt-too-long error). Default True — essential
    for reliability in long sessions."""

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

    router: ModelRouter | None = None
    """When set, ``router.select(purpose='reasoning')`` is called at each
    step instead of the ``llm`` argument passed directly.  Import from
    ``looplet.router`` — e.g. ``SimpleRouter``, ``FallbackRouter``.
    """

    checkpoint_dir: str | None = None
    """Directory path for checkpoint files.  When set, a FileCheckpointStore
    saves a checkpoint after every step so the loop can be resumed.
    """

    tracer: Tracer | None = None
    """When set, wraps each LLM call and tool dispatch in a span so call
    timings are recorded in the trace tree.  Import from
    ``looplet.telemetry``.
    """

    recovery_registry: RecoveryRegistry | None = None
    """When set, consulted on PARSE_ERROR instead of the built-in
    hardcoded 3-strategy recovery chain.  Import from
    ``looplet.recovery``.
    """

    compact_service: CompactService | None = None
    """When set, replaces the default reactive-compact strategy in the
    prompt-too-long recovery chain.  The service is invoked via
    :func:`run_compact`, which fires ``PRE_COMPACT`` / ``POST_COMPACT``
    events.  ``None`` keeps the default :class:`TruncateCompact`.
    """

    output_schema: OutputSchema | None = None
    """When set, ``validate_args(schema, done_payload)`` is called in the
    done() quality gate; invalid payloads are rejected with a message.
    Import from ``looplet.validation``.
    """

    initial_checkpoint: Checkpoint | None = None
    """When set, ``resume_loop_state(checkpoint)`` is called at loop start
    to restore session_log and step offset (crash-resume support).
    """

    memory_sources: list[Any] = field(default_factory=list)
    """Optional list of ``PersistentMemorySource`` objects rendered into
    the default prompt's top ``MEMORY`` section on every turn. Each
    source must expose ``load(state) -> str | None``. When ``build_prompt``
    is user-supplied, the loop still renders memory but passes it to
    the custom function as a ``memory=`` kwarg.
    """

    cache_policy: "CachePolicy | None" = None
    """Optional :class:`looplet.cache.CachePolicy` declaring which
    stable prompt sections (system prompt, tool schemas, memory) should
    carry ``cache_control`` markers. When set and the backend exposes
    ``generate_with_cache(..., cache_breakpoints=[...])``, the loop
    computes per-turn :class:`CacheBreakpoint` lists and passes them
    through. Backends without the method keep working unchanged;
    caching is strictly opt-in and additive.
    """

    cancel_token: CancelToken | None = None
    """Optional :class:`looplet.types.CancelToken` that signals the
    loop should terminate. The token is threaded through every LLM call
    (forwarded to backends that accept ``cancel_token=``) and every
    ``ToolContext`` so tools share the same cancellation channel. When
    observed cancelled between turns the loop exits cleanly without
    further LLM calls.
    """

    approval_handler: Callable[[str, list[str] | None], str | None] | None = None
    """Optional callable surfaced to tools via ``ToolContext.request_approval``.
    Lets a tool pause and request approval from the caller (user,
    upstream agent, webhook). Signature is
    ``(prompt, options) -> str | None``.

    * Returns a string → tool receives the approval and continues.
    * Returns ``None`` → tool receives ``None`` (async: the tool can
      set ``needs_approval=True`` in its result and
      :class:`ApprovalHook` will stop the loop for external approval).

    Leave unset for fully-autonomous runs."""

    context_window: int = 128_000
    """Maximum context window (in tokens) for the backend.  Used by:

    * The **pre-flight** prompt-size check — if the estimated prompt exceeds
      ``context_window - 3000`` tokens, reactive recovery fires *before*
      the LLM call (avoiding a wasted API call).
    * ``ThresholdCompactHook`` — when you also set ``compact_service``,
      proactive compaction triggers based on this value.

    Override to match your actual backend's window:
    200 000 for Claude, 128 000 for GPT-4o, 32 000 for GPT-4, etc.
    Default: 128 000."""

    render_messages_override: Callable[..., str] | None = None
    """Byte-exact prompt escape hatch.

    Receives keyword arguments ``messages: list[Message]``,
    ``default_prompt: str`` (the string the loop would have sent), and
    ``step_num: int``. Must return a string; whatever it returns is
    what the backend sees. When ``None`` (default), the loop uses the
    default or user-supplied :attr:`build_prompt` path.

    Prefer this over :attr:`build_prompt` when you want to inspect or
    mutate the full conversation thread (compaction, redaction, role
    reordering, role swaps) before send. Prefer :attr:`build_prompt`
    when you only want to change the section layout."""

    tool_metadata: dict[str, Any] = field(default_factory=dict)
    """Static key-value pairs merged into every ``ToolContext.metadata``.

    Use this to pass application-level configuration to tools without
    threading it through ``state.metadata`` (which is agent-level) or
    closures (which are invisible to provenance).

    ``tool_metadata`` is merged *under* ``state.metadata`` — state
    values win on key conflicts. This makes ``tool_metadata`` the
    right place for defaults (``db_path``, ``workspace``, ``api_base``)
    while ``state.metadata`` carries per-run overrides.

    Example::

        config = LoopConfig(
            tool_metadata={
                "db_path": "/data/prod.db",
                "read_only": True,
            },
        )
        # Every tool with ctx= sees ctx.metadata["db_path"]
    """

    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments passed through to every LLM call.

    Forwarded to ``llm.generate()`` and ``llm.generate_with_tools()``
    only when the backend's method signature accepts the key (checked
    via ``inspect.signature``). Unknown keys are silently skipped, so
    provider-specific kwargs don't break other backends.

    Use this for provider-specific parameters that looplet's
    ``generate(prompt, max_tokens, system_prompt, temperature)``
    protocol doesn't cover:

    * ``chat_template_kwargs`` for llama-server (e.g. ``{"enable_thinking": False}``)
    * ``response_format`` for OpenAI structured output
    * ``top_p``, ``top_k``, ``presence_penalty`` for fine-tuning
    * ``thinking`` for Anthropic extended thinking

    Example::

        config = LoopConfig(
            generate_kwargs={
                "chat_template_kwargs": {"enable_thinking": False},
                "top_p": 0.9,
            },
        )
    """


def _default_extract_entities(data: Any) -> list[str]:
    """Fallback: no entity extraction."""
    return []


def _default_build_briefing(state: Any, session_log: SessionLog, context: Any) -> str:
    """Fallback: empty briefing."""
    return ""


def _default_extract_step_metadata(state: Any, step_num: int) -> tuple[list[str], list[str]]:
    """Fallback: no step metadata extraction."""
    return [], []


def _build_tool_ctx(
    config: "LoopConfig",
    *,
    hooks: list[Any] | None = None,
    tool_call: ToolCall | None = None,
    step_num: int = 0,
    state: Any = None,
    session_log: Any = None,
    llm: Any = None,
) -> ToolContext:
    """Build a ToolContext for tool dispatch.

    Always returns a ToolContext — tools that declare ``ctx`` should
    never receive ``None``.  The context carries the cancel token,
    approval handler, LLM, progress callback, and metadata from the
    agent state.
    """
    _hooks = hooks or []
    _has_progress_subscribers = any(hasattr(h, "on_event") for h in _hooks)

    _progress_fn: Callable[[str, dict], None] | None = None
    if _has_progress_subscribers:
        from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

        def _on_progress(stage: str, data: dict) -> None:
            emit_event(
                _hooks,
                _LE.TOOL_PROGRESS,
                step_num=step_num,
                state=state,
                session_log=session_log,
                tool_call=tool_call,
                extra={"stage": stage, "data": data},
            )

        _progress_fn = _on_progress

    # Build a scoped LLM for tool-internal use.  If the effective LLM
    # is a recording backend, wrap it so tool-internal calls are tagged
    # with scope="tool:<name>" for nested provenance.
    _tool_llm = _scope_llm_for_tool(llm, tool_call) if llm is not None else None

    # Populate metadata: config.tool_metadata (defaults) + state.metadata
    # (per-run overrides). State wins on key conflicts.
    _metadata: dict[str, Any] = dict(config.tool_metadata) if config.tool_metadata else {}
    if state is not None:
        _state_meta = getattr(state, "metadata", None)
        if isinstance(_state_meta, dict):
            _metadata.update(_state_meta)

    return ToolContext(
        cancel_token=config.cancel_token,
        request_approval=config.approval_handler,
        on_progress=_progress_fn,
        llm=_tool_llm,
        metadata=_metadata,
    )


def _scope_llm_for_tool(llm: Any, tool_call: ToolCall | None) -> Any:
    """Wrap a recording LLM so tool-internal calls get a scope tag.

    If ``llm`` is (or wraps) a ``_RecordingBase``, returns a thin proxy
    that sets ``scope`` on every captured :class:`LLMCall`.  Otherwise
    returns ``llm`` unchanged — no overhead for non-recording backends.
    """
    tool_name = getattr(tool_call, "tool", "unknown") if tool_call else "unknown"

    # Unwrap ResilientBackend / CostTracker to find the recording layer
    inner = llm
    recording = None
    for _ in range(5):  # bounded unwrap
        if hasattr(inner, "calls") and hasattr(inner, "_record"):
            recording = inner
            break
        inner = getattr(inner, "_inner", None) or getattr(inner, "_backend", None)
        if inner is None:
            break

    if recording is None:
        return llm  # no recording layer — pass through unchanged

    return _ScopedLLMProxy(llm, recording, f"tool:{tool_name}")


class _ScopedLLMProxy:
    """Thin proxy that tags recorded LLM calls with a scope string.

    Delegates ``generate`` (and ``generate_with_tools`` when present)
    to the wrapped backend. After each call, stamps
    ``scope=<scope_str>`` on the most-recently-recorded
    :class:`LLMCall` so provenance consumers can distinguish
    loop-level calls from tool-internal calls.

    Zero overhead when the backend isn't recording — the proxy is only
    created by :func:`_scope_llm_for_tool` when a recording layer is
    detected.
    """

    def __init__(self, backend: Any, recording: Any, scope: str) -> None:
        self._backend = backend
        self._recording = recording
        self._scope = scope
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        n_before = len(self._recording.calls)
        result = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        self._tag_new_calls(n_before)
        return result

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Any:
        n_before = len(self._recording.calls)
        result = self._backend.generate_with_tools(
            prompt,
            tools=tools,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        self._tag_new_calls(n_before)
        return result

    def _tag_new_calls(self, n_before: int) -> None:
        for call in self._recording.calls[n_before:]:
            call.scope = self._scope


def emit_event(
    hooks: list[Any],
    event: Any,
    **payload_kwargs: Any,
) -> list[Any]:
    """Dispatch a :class:`LifecycleEvent` to every hook that opts in.

    Public API — safe to call from subagents, custom hooks, or external
    orchestrators that need to fire lifecycle events on a hook list.

    Hooks without ``on_event`` are silently skipped — this is the
    additive surface, nobody has to implement it. Returned
    :class:`HookDecision` objects are collected so the caller can act
    on ``block`` / ``stop`` / ``updated_*`` fields; ``None`` returns
    are filtered out.

    Exceptions from a hook are swallowed and logged — event dispatch
    must never break the loop.
    """
    from looplet.events import EventPayload  # noqa: PLC0415
    from looplet.hook_decision import HookDecision  # noqa: PLC0415

    decisions: list[Any] = []
    payload = EventPayload(event=event, **payload_kwargs)
    for hook in hooks:
        fn = getattr(hook, "on_event", None)
        if fn is None:
            continue
        # Deduplicate: when the event has a per-method equivalent
        # (PRE_TOOL_USE → pre_dispatch, POST_TOOL_USE/FAILURE → post_dispatch),
        # skip hooks that implement the per-method slot — they already fired.
        _equiv = _EVENT_METHOD_EQUIV.get(event)
        if _equiv is not None and hasattr(hook, _equiv):
            continue
        try:
            result = fn(payload)
        except Exception:  # noqa: BLE001
            logger.exception("on_event hook raised; continuing")
            continue
        if isinstance(result, HookDecision):
            from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

            if event != _LE.HOOK_DECISION and not result.is_noop():
                _emit_hook_decision_event(
                    hooks,
                    decision=result,
                    hook_slot="on_event",
                    hook_name=type(hook).__name__,
                    step_num=payload.step_num,
                    state=payload.state,
                    session_log=payload.session_log,
                    context=payload.context,
                    extra={"originating_event": getattr(event, "value", str(event))},
                )
            decisions.append(result)
    return decisions


def _emit_hook_decision_event(
    hooks: list[Any],
    *,
    decision: Any,
    hook_slot: str,
    hook_name: str,
    step_num: int = 0,
    state: Any = None,
    session_log: Any = None,
    context: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a structured event for a non-noop HookDecision."""
    if decision is None or decision.is_noop():
        return

    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

    event_extra = {"decision": asdict(decision)}
    if extra:
        event_extra.update(extra)
    emit_event(
        hooks,
        _LE.HOOK_DECISION,
        step_num=step_num,
        state=state,
        session_log=session_log,
        context=context,
        hook_slot=hook_slot,
        hook_name=hook_name,
        extra=event_extra,
    )


# ── Event → per-method deduplication map ────────────────────────
# When an on_event LifecycleEvent has a per-method hook equivalent,
# hooks that implement the per-method version are skipped in on_event
# to avoid double-firing.  Hooks that ONLY use on_event still get it.
_EVENT_METHOD_EQUIV: dict[Any, str] = {}  # populated after LifecycleEvent import


def _init_event_method_equiv() -> None:
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

    _EVENT_METHOD_EQUIV[_LE.PRE_TOOL_USE] = "pre_dispatch"
    _EVENT_METHOD_EQUIV[_LE.POST_TOOL_USE] = "post_dispatch"
    _EVENT_METHOD_EQUIV[_LE.POST_TOOL_FAILURE] = "post_dispatch"


_init_event_method_equiv()


# ── check_done dispatch (backward-compatible tool_call kwarg) ───

_CHECK_DONE_ACCEPTS_TOOL_CALL: dict[int, bool] = {}
"""Per-method cache mapping ``id(check_done)`` to whether the bound
method accepts a ``tool_call`` keyword argument. Populated lazily.
Cached by id so we never re-inspect the same method object twice."""


def _accepts_tool_call_kwarg(method: Any) -> bool:
    key = id(method)
    cached = _CHECK_DONE_ACCEPTS_TOOL_CALL.get(key)
    if cached is not None:
        return cached
    import inspect  # noqa: PLC0415

    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        accepts = False
    else:
        params = sig.parameters
        accepts = "tool_call" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    _CHECK_DONE_ACCEPTS_TOOL_CALL[key] = accepts
    return accepts


def _call_check_done(
    hook: Any,
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
    tool_call: Any,
) -> Any:
    """Invoke a hook's ``check_done`` with or without the ``tool_call``
    kwarg depending on its signature. Lets new gates inspect the agent's
    pending ``done()`` answer without breaking legacy hooks."""
    method = hook.check_done
    if _accepts_tool_call_kwarg(method):
        return method(state, session_log, context, step_num, tool_call=tool_call)
    return method(state, session_log, context, step_num)


# ── Hook method names (for typo detection) ──────────────────────

_KNOWN_HOOK_METHODS = frozenset(
    {
        "pre_loop",
        "pre_prompt",
        "pre_dispatch",
        "post_dispatch",
        "check_done",
        "check_permission",
        "should_stop",
        "should_compact",
        "build_briefing",
        "build_prompt",
        "on_loop_end",
        "on_event",
    }
)


def _validate_hooks(hooks: list[Any]) -> None:
    """Warn when a hook object has no recognized hook methods.

    Catches typos like ``post_dispach`` by checking that at least one
    method name matches the known set.  Silently accepts hooks with
    at least one valid method — partial implementations are the norm.
    """
    import warnings  # noqa: PLC0415

    for hook in hooks:
        has_any = any(hasattr(hook, m) for m in _KNOWN_HOOK_METHODS)
        if not has_any:
            warnings.warn(
                f"Hook {type(hook).__name__} has no recognized hook methods "
                f"({', '.join(sorted(_KNOWN_HOOK_METHODS)[:5])}, ...). "
                f"Did you misspell a method name?",
                UserWarning,
                stacklevel=3,
            )


# ── Composable Agent Loop ───────────────────────────────────────


# ── Extracted dispatch helpers ────────────────────────────────────
# These reduce composable_loop's nesting depth and make the heaviest
# phases independently readable + testable.


@dataclass
class _InterceptResult:
    """Return value of _intercept_tool_calls — collected pre-dispatch outcomes."""

    intercepted: dict[int, ToolResult] = field(default_factory=dict)
    """Map of call-index → ToolResult for intercepted/denied calls."""

    extra_context: list[str] = field(default_factory=list)
    """Additional context strings to inject into the next prompt."""


def _intercept_tool_calls(
    calls: list[ToolCall],
    hooks: list[Any],
    state: AgentState,
    session_log: SessionLog,
    context: Any,
    step_num: int,
) -> _InterceptResult:
    """Run pre-dispatch hooks and permission checks on tool calls.

    Processes both the event-style ``on_event(PRE_TOOL_USE)`` path and
    the per-method ``pre_dispatch`` + ``check_permission`` paths.
    Returns the set of intercepted results so the caller can skip
    dispatch for those calls.
    """
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415
    from looplet.types import ErrorKind, ToolError  # noqa: PLC0415

    result = _InterceptResult()

    for tc_idx, tc in enumerate(calls):
        cur_step = step_num + tc_idx

        # ── Event-style hooks (on_event) ────────────────────
        _pre_tool_decisions = emit_event(
            hooks,
            _LE.PRE_TOOL_USE,
            step_num=cur_step,
            state=state,
            session_log=session_log,
            context=context,
            tool_call=tc,
        )
        _handled = False
        for _d in _pre_tool_decisions:
            if _d.updated_args is not None:
                tc.args = _d.updated_args
            if _d.permission == "deny":
                _te = ToolError(
                    kind=ErrorKind.PERMISSION_DENIED,
                    message=_d.block or f"Permission denied for tool '{tc.tool}'",
                    retriable=False,
                )
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=_te.message,
                    error_detail=_te,
                )
                _handled = True
                break
            if _d.updated_result is not None:
                result.intercepted[tc_idx] = _d.updated_result
                _handled = True
                break
            if _d.additional_context:
                result.extra_context.append(_d.additional_context)
        if _handled:
            continue

        # ── Per-method pre_dispatch hooks ───────────────────
        for hook in hooks:
            if not hasattr(hook, "pre_dispatch"):
                continue
            cached = hook.pre_dispatch(state, session_log, tc, cur_step)
            _decision = normalize_hook_return(cached, slot="pre_dispatch")
            if _decision is None:
                if cached is not None and isinstance(cached, ToolResult):
                    result.intercepted[tc_idx] = cached
                    break
                continue
            _emit_hook_decision_event(
                hooks,
                decision=_decision,
                hook_slot="pre_dispatch",
                hook_name=type(hook).__name__,
                step_num=cur_step,
                state=state,
                session_log=session_log,
                context=context,
            )
            if _decision.updated_args is not None:
                tc.args = _decision.updated_args
            if _decision.permission == "deny":
                _te = ToolError(
                    kind=ErrorKind.PERMISSION_DENIED,
                    message=_decision.block or f"Permission denied for tool '{tc.tool}'",
                    retriable=False,
                )
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=_te.message,
                    error_detail=_te,
                )
                break
            if _decision.updated_result is not None:
                result.intercepted[tc_idx] = _decision.updated_result
                break
            if _decision.additional_context:
                result.extra_context.append(_decision.additional_context)

        if tc_idx in result.intercepted:
            continue

        # ── Per-method check_permission hooks ──────────────
        for hook in hooks:
            if not hasattr(hook, "check_permission"):
                continue
            _raw = hook.check_permission(tc, state)
            _decision = normalize_hook_return(_raw, slot="check_permission")
            if _decision is not None:
                _emit_hook_decision_event(
                    hooks,
                    decision=_decision,
                    hook_slot="check_permission",
                    hook_name=type(hook).__name__,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
            allowed = _decision is None or _decision.permission != "deny"
            if not allowed:
                _msg = (
                    _decision.block
                    if _decision and _decision.block
                    else f"Permission denied for tool '{tc.tool}'"
                )
                _te = ToolError(
                    kind=ErrorKind.PERMISSION_DENIED,
                    message=_msg,
                    retriable=False,
                )
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=_te.message,
                    error_detail=_te,
                )
                break

    return result


@dataclass
class _PostDispatchOutcome:
    """Return value of _run_post_dispatch_hooks."""

    tool_result: ToolResult
    """Possibly rewritten tool result."""

    extra_context: list[str] = field(default_factory=list)
    """Additional context strings for next prompt."""

    stop_reason: str | None = None
    """Non-None if any hook requested stop."""


def _run_post_dispatch_hooks(
    tool_call: ToolCall,
    tool_result: ToolResult,
    hooks: list[Any],
    state: AgentState,
    session_log: SessionLog,
    context: Any,
    step_num: int,
    *,
    emit_lifecycle: bool = True,
) -> _PostDispatchOutcome:
    """Run post_dispatch hooks + POST_TOOL_USE/FAILURE events.

    Returns a potentially rewritten ToolResult, additional context
    parts, and an optional stop reason.

    ``emit_lifecycle`` is set to ``False`` for the terminal ``done()``
    dispatch: done is a loop signal rather than a side-effecting tool,
    so PRE/POST_TOOL_USE events deliberately skip it. Per-method
    post_dispatch hooks still run in both modes so metrics/tracing/
    audit hooks see the final step.
    """
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

    outcome = _PostDispatchOutcome(tool_result=tool_result)

    # ── Per-method post_dispatch ───────────────────────────
    for hook in hooks:
        if not hasattr(hook, "post_dispatch"):
            continue
        text = hook.post_dispatch(state, session_log, tool_call, tool_result, step_num)
        _decision = normalize_hook_return(text, slot="post_dispatch")
        if _decision is not None:
            _emit_hook_decision_event(
                hooks,
                decision=_decision,
                hook_slot="post_dispatch",
                hook_name=type(hook).__name__,
                step_num=step_num,
                state=state,
                session_log=session_log,
                context=context,
            )
            if _decision.updated_result is not None:
                outcome.tool_result = _decision.updated_result
                tool_result = outcome.tool_result
            if _decision.additional_context:
                outcome.extra_context.append(_decision.additional_context)
            if _decision.stop is not None:
                outcome.stop_reason = _decision.stop

    if not emit_lifecycle:
        return outcome

    # ── Event-style POST_TOOL_USE / POST_TOOL_FAILURE ──────
    _post_tool_event = _LE.POST_TOOL_FAILURE if tool_result.error else _LE.POST_TOOL_USE
    _post_tool_decisions = emit_event(
        hooks,
        _post_tool_event,
        step_num=step_num,
        state=state,
        session_log=session_log,
        context=context,
        tool_call=tool_call,
        tool_result=tool_result,
    )
    for _d in _post_tool_decisions:
        if _d.updated_result is not None:
            outcome.tool_result = _d.updated_result
        if _d.additional_context:
            outcome.extra_context.append(_d.additional_context)
        if _d.stop is not None:
            outcome.stop_reason = _d.stop

    return outcome


# ── Composable Agent Loop ───────────────────────────────────────


def composable_loop(
    llm: Any,
    task: Any = None,
    tools: BaseToolRegistry | None = None,
    context: Any = None,
    hooks: list[Any] | None = None,
    config: LoopConfig | None = None,
    state: AgentState | None = None,
    session_log: SessionLog | None = None,
    stream: Any | None = None,
    conversation: Any | None = None,
    *,
    max_steps: int | None = None,
    system_prompt: str | None = None,
) -> Generator[Step, None, Any]:
    """Domain-agnostic agent loop with composable hooks.

    Yields Steps, returns a trace object built by config.build_trace.

    Args:
        llm: LLM backend — must implement ``generate()`` (see :class:`LLMBackend`).
        task: The task description — dict, string, or any domain object.
            When a dict, ``task.get("id")`` is used for event labels.
        tools: Tool registry with available tools.
        context: Opaque domain handle passed verbatim to hook methods
            (``pre_prompt``, ``check_done``, ``build_briefing``,
            ``on_loop_end``).  Hooks that need domain state at call time
            should receive it here; hooks that capture state at ``__init__``
            can ignore it.  Defaults to ``None`` for simple agents.
        hooks: Composable hook instances for domain behavior.
        config: Loop configuration (steps, tokens, callables).
        state: Agent state (must satisfy AgentState protocol).
        session_log: Session log for recording agent memory.
        stream: Optional EventEmitter — when set, emits structured events for
            each loop lifecycle moment (start, step, LLM call, dispatch, end).
        conversation: Optional Conversation — when set, the loop auto-records
            each LLM prompt/response and tool call/result as Messages in the
            conversation thread. Works alongside session_log (both are populated).
        max_steps: Convenience shorthand. When set, configures both
            ``LoopConfig.max_steps`` and ``DefaultState.max_steps`` so
            simple agents don't need to construct either explicitly. If
            ``config`` is also passed, this overrides ``config.max_steps``.
        system_prompt: Convenience shorthand for ``LoopConfig.system_prompt``;
            same override semantics as ``max_steps``.
    """
    if task is None:
        task = {}
    if tools is None:
        raise ValueError("tools is required")
    if config is None:
        config = LoopConfig()
    if max_steps is not None:
        config.max_steps = max_steps
    if system_prompt is not None:
        config.system_prompt = system_prompt
    if hooks is None:
        hooks = []

    # ── Input guards ────────────────────────────────────────────
    if not callable(getattr(llm, "generate", None)):
        raise TypeError(
            f"llm must implement generate() (got {type(llm).__name__}). "
            "Use OpenAIBackend(client, model=...) or AnthropicBackend(client, model=...)."
        )

    # Default state when none provided.
    from looplet.types import DefaultState as _DefaultState  # noqa: PLC0415

    if state is None:
        state = _DefaultState(max_steps=config.max_steps)

    # Sync max_steps: config is the source of truth.  Warn once when the
    # two disagree — a common footgun for agents assembling a loop for
    # the first time.
    if isinstance(state, _DefaultState) and state.max_steps != config.max_steps:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"DefaultState(max_steps={state.max_steps}) differs from "
            f"LoopConfig(max_steps={config.max_steps}); using the config "
            f"value. Pass the same max_steps to both to silence this.",
            UserWarning,
            stacklevel=2,
        )
        state.max_steps = config.max_steps

    # Warn when a hook object has no recognized hook methods — likely a typo.
    _validate_hooks(hooks)

    t0 = time.time()

    if session_log is None:
        session_log = SessionLog()

    # ── Conversation thread — always active (single source of truth) ──
    from looplet.conversation import Conversation as _Conversation  # noqa: PLC0415

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
    _LLMCallEndEvent = None
    _ToolResultEvent = None
    _StepEndEvent = None
    if stream is not None:
        try:
            from looplet.streaming import (
                LLMCallEndEvent as _LLMCallEndEvent,
            )
            from looplet.streaming import (
                LLMCallStartEvent as _LLMCallStartEvent,
            )
            from looplet.streaming import (
                LoopEndEvent as _LoopEndEvent,
            )
            from looplet.streaming import (  # noqa: PLC0415
                LoopStartEvent as _LoopStartEvent,
            )
            from looplet.streaming import (
                StepEndEvent as _StepEndEvent,
            )
            from looplet.streaming import (
                StepStartEvent as _StepStartEvent,
            )
            from looplet.streaming import (
                ToolDispatchEvent as _ToolDispatchEvent,
            )
            from looplet.streaming import (
                ToolResultEvent as _ToolResultEvent,
            )
        except ImportError:
            pass

    # ── Resolve effective LLM (router overrides direct llm) ────
    def _get_llm() -> Any:
        if config is not None and config.router is not None:  # pyright: ignore[reportOptionalMemberAccess]
            return config.router.select(purpose="reasoning")  # pyright: ignore[reportOptionalMemberAccess]
        return llm

    # ── Checkpoint store setup ──────────────────────────────────
    _ckpt_store = None
    if config.checkpoint_dir is not None:
        _ckpt_store = _FileCheckpointStore(config.checkpoint_dir)

        # Auto-resume: if checkpoint_dir has checkpoints and no
        # explicit initial_checkpoint was given, load the latest.
        # This makes crash-resume a one-liner:
        #   LoopConfig(checkpoint_dir="./ckpt")
        # — saves after every step, resumes on restart.
        if config.initial_checkpoint is None:
            _latest = _ckpt_store.load_latest()
            if _latest is not None:
                config = _dc_replace(config, initial_checkpoint=_latest)
                logger.info(
                    "Auto-resuming from checkpoint at step %d",
                    _latest.step_number,
                )

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

    # Domain adapter seeding: when ``config.domain`` is set, fall back
    # to each adapter field only if the flat field is still ``None``.
    # Flat fields therefore override the adapter; adapter overrides
    # built-in defaults.
    _dom = config.domain
    build_briefing = (
        config.build_briefing or (_dom.build_briefing if _dom else None) or _default_build_briefing
    )
    extract_entities = (
        config.extract_entities
        or (_dom.extract_entities if _dom else None)
        or _default_extract_entities
    )
    build_prompt_fn = config.build_prompt or (_dom.build_prompt if _dom else None)

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
        "emergency_truncate": False,
        "result_clearing": False,
    }

    extract_step_metadata = (
        config.extract_step_metadata
        or (_dom.extract_step_metadata if _dom else None)
        or _default_extract_step_metadata
    )

    # ── Pre-loop hooks ──────────────────────────────────────────
    # Stash task + conversation on state so hooks (EvalHook, budget
    # telemetry, trajectory recorders) can read them without needing
    # extra parameters. Use setattr so this works on any AgentState.
    try:
        setattr(state, "task", task)  # noqa: B010
    except AttributeError:
        pass
    try:
        setattr(state, "conversation", _conv)  # noqa: B010
    except AttributeError:
        pass
    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            hook.pre_loop(state, session_log, context)

    # Fire SESSION_START — single-slot subscribers to lifecycle
    # events get it in one place alongside the per-method pre_loop.
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

    emit_event(
        hooks,
        _LE.SESSION_START,
        state=state,
        session_log=session_log,
        context=context,
    )

    # ── Emit LoopStartEvent ─────────────────────────────────────
    # Skip if a StreamingHook is in hooks — it already emits LoopStartEvent
    # from its pre_loop method and we don't want duplicates.
    from looplet.streaming import StreamingHook as _StreamingHook  # noqa: PLC0415

    _has_streaming_hook = any(isinstance(h, _StreamingHook) for h in hooks)
    if stream is not None and _LoopStartEvent is not None and not _has_streaming_hook:
        _task_id = task.get("id", "") if isinstance(task, dict) else str(task)[:80]
        stream.emit(_LoopStartEvent(task_summary=str(_task_id), max_steps=config.max_steps))

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset
        _hook_requested_stop = False  # reset per step; honored after dispatch

        # Clear per-step hook context so hooks start each step with
        # a clean slate.  Hooks write to state.step_context during a
        # step; other hooks read from it within the same step.  The
        # loop owns the lifecycle — cleared here, populated by hooks.
        try:
            setattr(state, "step_context", {})  # noqa: B010
        except AttributeError:
            pass

        # Cancellation check between turns — stop cleanly, no more LLM calls.
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        # ── Proactive compaction ────────────────────────────
        # If any hook votes yes, run the configured compact service
        # before building the next prompt. Complements the reactive
        # path that fires only on prompt_too_long errors.
        _want_compact = False
        for hook in hooks:
            if hasattr(hook, "should_compact"):
                try:
                    if hook.should_compact(state, session_log, _conv, step_num):
                        _want_compact = True
                        break
                except Exception:  # noqa: BLE001
                    logger.exception("should_compact hook raised; skipping")
        if _want_compact and config.compact_service is not None:
            from looplet.compact import run_compact as _run_compact  # noqa: PLC0415

            _run_compact(
                config.compact_service,
                hooks=hooks,
                state=state,
                session_log=session_log,
                llm=_get_llm(),
                conversation=_conv,
                step_num=step_num,
                reason="proactive",
            )

        # ── Pre-prompt hooks ────────────────────────────────
        # First-hook-wins ``build_briefing`` slot on LoopHook takes
        # precedence over ``config.build_briefing`` / the default.
        _briefing_text: str | None = None
        for hook in hooks:
            if hasattr(hook, "build_briefing"):
                try:
                    _bt = hook.build_briefing(state, session_log, context)
                except Exception:  # noqa: BLE001
                    logger.exception("build_briefing hook raised; falling back")
                    _bt = None
                if _bt is not None:
                    _briefing_text = _bt
                    break
        if _briefing_text is None:
            _briefing_text = build_briefing(state, session_log, context)
        briefing_parts = [_briefing_text]
        _briefing_budget = config.max_briefing_tokens
        _briefing_used = len(briefing_parts[0]) // 4 if _briefing_budget else 0

        for hook in hooks:
            if hasattr(hook, "pre_prompt"):
                text = hook.pre_prompt(state, session_log, context, step_num)
                # HookDecision-aware: accept both legacy str and decisions.
                _decision = normalize_hook_return(text, slot="pre_prompt")
                if _decision is not None:
                    _emit_hook_decision_event(
                        hooks,
                        decision=_decision,
                        hook_slot="pre_prompt",
                        hook_name=type(hook).__name__,
                        step_num=step_num,
                        state=state,
                        session_log=session_log,
                        context=context,
                    )
                text = _decision.additional_context if _decision else None
                if text:
                    if _briefing_budget:
                        text_tokens = len(text) // 4
                        if _briefing_used + text_tokens > _briefing_budget:
                            briefing_parts.append("(briefing truncated — token budget exceeded)")
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

        # Persistent memory: rendered once
        # per turn, placed above TASK by the default prompt builder.
        _memory_sources = getattr(config, "memory_sources", None)
        if _memory_sources:
            from looplet.memory import render_memory as _render_memory  # noqa: PLC0415

            _rendered_memory = _render_memory(_memory_sources, state)
        else:
            _rendered_memory = ""

        _prompt_kwargs = dict(
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

        # First-hook-wins ``build_prompt`` slot on LoopHook takes
        # precedence over ``config.build_prompt`` / the default.
        prompt: str | None = None
        for hook in hooks:
            if hasattr(hook, "build_prompt"):
                try:
                    _hp = hook.build_prompt(**_prompt_kwargs)
                except Exception:  # noqa: BLE001
                    logger.exception("build_prompt hook raised; falling back")
                    _hp = None
                if _hp is not None:
                    prompt = _hp
                    break

        if prompt is None:
            if build_prompt_fn is not None:
                prompt = build_prompt_fn(**_prompt_kwargs)
            else:
                # Domain-agnostic default: 7-section structured prompt.
                from looplet.prompts import (
                    build_prompt as _default_build_prompt,  # noqa: PLC0415
                )

                prompt = _default_build_prompt(**_prompt_kwargs)  # pyright: ignore[reportArgumentType]

        # ── Byte-exact escape hatch (render_messages_override) ──
        # If configured, the user takes full control of the prompt
        # bytes after seeing the live conversation thread. We pass
        # the would-be default prompt so the user can fall back to
        # it for sections they don't want to change.
        if config.render_messages_override is not None:
            prompt = config.render_messages_override(
                messages=list(_conv.messages),
                default_prompt=prompt,
                step_num=step_num,
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

        # Fire PRE_LLM_CALL — observers only; return decisions are
        # collected but only ``additional_context`` is honored (the
        # prompt string is already built at this point; mutating it
        # would invalidate the briefing budget accounting).
        _pre_llm_decisions = emit_event(
            hooks,
            _LE.PRE_LLM_CALL,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
        )
        for _d in _pre_llm_decisions:
            if _d.additional_context:
                post_dispatch_parts.append(_d.additional_context)
            if _d.stop is not None:
                stop_reason = _d.stop
                _hook_requested_stop = True

        if preflight_too_long and config.reactive_recovery:
            logger.warning(
                "Pre-flight block: prompt ~%d tokens exceeds safe limit — "
                "running recovery before LLM call",
                estimated_tokens,
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
            _native_on = (config.use_native_tools) and hasattr(effective_llm, "generate_with_tools")
            _tool_schemas = tools.tool_schemas() if _native_on else None
            # ── Prompt cache breakpoints (opt-in) ─────────────────
            # When a ``cache_policy`` is configured, compute hashes for
            # the stable sections and hand them to the backend. If a
            # ``CacheBreakDetector`` is present among hooks we prefer
            # its ``record`` path so cache-break telemetry is captured.
            _cache_bps: list[Any] | None = None
            if config.cache_policy is not None:
                from looplet.cache import (  # noqa: PLC0415
                    CacheBreakDetector as _CBD,
                )
                from looplet.cache import (
                    compute_breakpoints as _compute_bps,
                )

                _schemas_text = tools.tool_catalog_text()
                _detector = next(
                    (h for h in hooks if isinstance(h, _CBD)),
                    None,
                )
                if _detector is not None:
                    _cache_bps = _detector.record(
                        step_num,
                        system_prompt=config.system_prompt,
                        tool_schemas_text=_schemas_text,
                        memory_text=_rendered_memory,
                    )
                else:
                    _cache_bps = _compute_bps(
                        config.cache_policy,
                        system_prompt=config.system_prompt,
                        tool_schemas_text=_schemas_text,
                        memory_text=_rendered_memory,
                    )
            _llm_t0 = time.perf_counter()
            llm_result = llm_call_with_retry(
                effective_llm,
                prompt,
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                tools=_tool_schemas,
                cancel_token=config.cancel_token,
                max_continuations=config.max_turn_continuations,
                cache_breakpoints=_cache_bps,
                generate_kwargs=config.generate_kwargs or None,
            )
            _llm_dur_ms = (time.perf_counter() - _llm_t0) * 1000.0
            if _llm_span is not None and config.tracer is not None:
                config.tracer.end_span(_llm_span)
            llm_calls += 1
            # Emit LLMCallEndEvent
            if stream is not None and _LLMCallEndEvent is not None:
                stream.emit(
                    _LLMCallEndEvent(
                        step_num=step_num,
                        response_length=len(llm_result.text or ""),
                        duration_ms=_llm_dur_ms,
                    )
                )

        # Reactive recovery: if prompt-too-long, try chained strategies
        if not llm_result.ok and llm_result.is_prompt_too_long and config.reactive_recovery:
            raw_response = _recovery_chain(
                llm_result,
                recovery_state,
                state,
                session_log,
                effective_llm,
                tools,
                context,
                build_briefing,
                build_prompt_fn,
                task,
                config,
                step_num,
                hooks=hooks,
                conversation=_conv,
            )
            llm_calls += recovery_state.get("_last_recovery_llm_calls", 0)
            llm_result = LLMResult(raw_response)

        raw_response = llm_result.text

        # ── Record LLM turn in conversation thread via unified recorder ──
        _history.record_llm_turn(prompt=prompt, response=raw_response)

        # Fire POST_LLM_RESPONSE — hooks can observe raw text before
        # it hits the parser. Stop requests are honored at end-of-step.
        _post_llm_decisions = emit_event(
            hooks,
            _LE.POST_LLM_RESPONSE,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
            raw_response=raw_response,
        )
        for _d in _post_llm_decisions:
            if _d.stop is not None:
                stop_reason = _d.stop
                _hook_requested_stop = True
            if _d.additional_context:
                post_dispatch_parts.append(_d.additional_context)

        if raw_response is None:
            # If cancellation caused the failure, exit cleanly — no error step.
            if config.cancel_token is not None and getattr(
                config.cancel_token, "is_cancelled", False
            ):
                stop_reason = "cancelled"
                break
            logger.error("LLM call failed after retries at step %d", step_num)
            error_call = ToolCall(tool="__llm_error__", reasoning="LLM call failed after retries")
            error_result = ToolResult(
                tool="__llm_error__",
                args_summary="",
                data=None,
                error="LLM call failed after all retry attempts",
            )
            step = Step(number=step_num, tool_call=error_call, tool_result=error_result)
            state.steps.append(step)
            yield step
            _history.record_step(
                step, theory="", entities=[], findings=[], highlights=[], recall_key=""
            )
            break

        # ── Parse response (native tool_use or JSON text) ────
        if (config.use_native_tools) and isinstance(raw_response, list):
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
                    tool_call = ToolCall(
                        tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200]
                    )
                    tool_result = ToolResult(
                        tool="__parse_error__",
                        args_summary="",
                        data=None,
                        error=f"Parse error — recovery aborted: {_recovery_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                    state.steps.append(step)
                    yield step
                    _history.record_step(
                        step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                    )
                    continue
                if _recovery_action is not None and _recovery_action.message:
                    post_dispatch_parts.append(_recovery_action.message)
            if consecutive_parse_failures <= PARSE_RECOVERY_MAX:
                logger.warning(
                    "Parse failure %d/%d at step %d — attempting recovery",
                    consecutive_parse_failures,
                    PARSE_RECOVERY_MAX,
                    step_num,
                )
                recovery_prompt = build_parse_recovery_prompt(prompt, to_text(raw_response) or "")
                recovery_result = llm_call_with_retry(
                    effective_llm,
                    recovery_prompt,
                    max_tokens=config.max_tokens,
                    system_prompt=config.system_prompt,
                    temperature=config.recovery_temperature,
                    cancel_token=config.cancel_token,
                    generate_kwargs=config.generate_kwargs or None,
                )
                llm_calls += 1
                if recovery_result.ok:
                    tool_calls = parse_multi_tool_calls(recovery_result.text)

            if not tool_calls:
                logger.warning("Unparseable LLM response at step %d after recovery", step_num)
                tool_call = ToolCall(
                    tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200]
                )
                tool_result = ToolResult(
                    tool="__parse_error__",
                    args_summary="",
                    data=None,
                    error=f"Could not parse JSON: {(to_text(raw_response) or '')[:200]}",
                )
                step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                _history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
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
            # ── Pre-dispatch interception (hooks + permissions) ──
            _intercept = _intercept_tool_calls(
                regular_calls,
                hooks,
                state,
                session_log,
                context,
                step_num,
            )
            intercepted_results = _intercept.intercepted
            post_dispatch_parts.extend(_intercept.extra_context)

            # ── Dispatch permitted calls ────────────────────────
            dispatch_items = [
                (i, tc) for i, tc in enumerate(regular_calls) if i not in intercepted_results
            ]
            calls_to_dispatch = [tc for _, tc in dispatch_items]

            if calls_to_dispatch:

                def _ctx_for(_c: ToolCall, _cur_step: int) -> ToolContext | None:
                    return _build_tool_ctx(
                        config,
                        hooks=hooks,
                        tool_call=_c,
                        step_num=_cur_step,
                        state=state,
                        session_log=session_log,
                        llm=effective_llm,
                    )

                if config.concurrent_dispatch:
                    _tool_ctxs = [_ctx_for(_c, step_num + _idx) for _idx, _c in dispatch_items]
                    dispatch_results = tools.dispatch_batch(calls_to_dispatch, ctx=_tool_ctxs)
                else:
                    dispatch_results = []
                    for _idx, _c in dispatch_items:
                        _tool_ctx = _ctx_for(_c, step_num + _idx)
                        dispatch_results.append(tools.dispatch(_c, ctx=_tool_ctx))
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
                    stream.emit(
                        _ToolDispatchEvent(
                            step_num=cur_step,
                            tool_name=tool_call.tool,
                            args_summary=_summarize_args_dict(tool_call.args),
                        )
                    )

                # ── Post-dispatch hooks + events ────────────────
                _pd = _run_post_dispatch_hooks(
                    tool_call,
                    tool_result,
                    hooks,
                    state,
                    session_log,
                    context,
                    cur_step,
                )
                tool_result = _pd.tool_result
                post_dispatch_parts.extend(_pd.extra_context)
                if _pd.stop_reason is not None:
                    stop_reason = _pd.stop_reason
                    _hook_requested_stop = True

                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                cur_step_count = state.step_count
                state.steps.append(step)
                yield step

                # Emit ToolResultEvent + StepEndEvent
                if stream is not None and _ToolResultEvent is not None:
                    stream.emit(
                        _ToolResultEvent(
                            step_num=cur_step,
                            tool_name=tool_result.tool,
                            duration_ms=tool_result.duration_ms,
                            has_error=tool_result.error is not None,
                        )
                    )
                if stream is not None and _StepEndEvent is not None:
                    stream.emit(
                        _StepEndEvent(
                            step_num=cur_step,
                            classification="continue",
                            new_entities_count=0,
                        )
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

                # Save checkpoint after the session log and conversation include this step.
                if _ckpt_store is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={
                                "entries": session_log.to_list(),
                                "current_theory": session_log.current_theory,
                            },
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

        # Handle done() if present
        if done_idx is not None:
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    w = _call_check_done(hook, state, session_log, context, step_num, tool_call)
                    _decision = normalize_hook_return(w, slot="check_done")
                    if _decision is not None:
                        _emit_hook_decision_event(
                            hooks,
                            decision=_decision,
                            hook_slot="check_done",
                            hook_name=type(hook).__name__,
                            step_num=cur_step,
                            state=state,
                            session_log=session_log,
                            context=context,
                        )
                    if _decision is not None and _decision.is_block():
                        gate_warning = _decision.block or "blocked by hook"
                        break

            # Output schema validation — reject done() if payload is invalid
            if gate_warning is None and config.output_schema is not None:
                validation = _validate_args(config.output_schema, tool_call.args)
                if not validation.valid:
                    gate_warning = (
                        f"Output schema validation failed: {'; '.join(validation.errors)}"
                    )

            # Emit ToolDispatchEvent for done
            if stream is not None and _ToolDispatchEvent is not None:
                stream.emit(
                    _ToolDispatchEvent(
                        step_num=cur_step,
                        tool_name=tool_call.tool,
                        args_summary=_summarize_args_dict(tool_call.args),
                    )
                )

            if gate_warning is not None:
                logger.info("Quality gate rejected done() at step %d", step_num)
                quality_gate_message = gate_warning
                tool_result = ToolResult(
                    tool=done_tool_name,
                    args_summary="rejected",
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
                # done() dispatch intentionally bypasses permission checks — it's
                # a loop signal, not a side-effecting tool. Permission-gating a
                # termination signal would prevent the agent from ever stopping.
                _ctx = _build_tool_ctx(
                    config,
                    hooks=hooks,
                    tool_call=tool_call,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    llm=effective_llm,
                )
                tool_result = tools.dispatch(tool_call, ctx=_ctx)
                # Run post_dispatch hooks for done() too — otherwise
                # MetricsHook / TracingHook / AuditHook silently miss
                # the final step of every run. Lifecycle events
                # (PRE/POST_TOOL_USE) deliberately skip done since it
                # is a loop signal, not a side-effecting tool call.
                _pd_done = _run_post_dispatch_hooks(
                    tool_call,
                    tool_result,
                    hooks,
                    state,
                    session_log,
                    context,
                    cur_step,
                    emit_lifecycle=False,
                )
                tool_result = _pd_done.tool_result
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                # Emit ToolResultEvent + StepEndEvent for the done step
                if stream is not None and _ToolResultEvent is not None:
                    stream.emit(
                        _ToolResultEvent(
                            step_num=cur_step,
                            tool_name=tool_result.tool,
                            duration_ms=tool_result.duration_ms,
                            has_error=tool_result.error is not None,
                        )
                    )
                if stream is not None and _StepEndEvent is not None:
                    stream.emit(
                        _StepEndEvent(
                            step_num=cur_step,
                            classification="done",
                            new_entities_count=0,
                        )
                    )
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
                            session_log_data={
                                "entries": session_log.to_list(),
                                "current_theory": session_log.current_theory,
                            },
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
                # DONE_ACCEPTED is observer-only; the loop is already terminating.
                emit_event(
                    hooks,
                    _LE.DONE_ACCEPTED,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                    tool_call=tool_call,
                    tool_result=tool_result,
                )

        if done:
            continue

        # Honor HookDecision.stop signalled during post_dispatch.
        if _hook_requested_stop:
            done = True
            break

        for hook in hooks:
            if hasattr(hook, "should_stop"):
                _raw = hook.should_stop(state, step_num, len(all_step_entities))
                _decision = normalize_hook_return(_raw, slot="should_stop")
                if _decision is not None:
                    _emit_hook_decision_event(
                        hooks,
                        decision=_decision,
                        hook_slot="should_stop",
                        hook_name=type(hook).__name__,
                        step_num=step_num,
                        state=state,
                        session_log=session_log,
                        context=context,
                    )
                _stopped = _decision.is_stop() if _decision else False
                if _stopped:
                    logger.info("Hook %s requested stop at step %d", type(hook).__name__, step_num)
                    done = True
                    stop_reason = _decision.stop if _decision and _decision.stop else "hook_stop"
                    break
        if done:
            break

    # ── Post-loop hooks ─────────────────────────────────────
    # Stash stop_reason on state so hooks (e.g. StreamingHook) can read it
    if state is not None:
        state._stop_reason = stop_reason  # pyright: ignore[reportAttributeAccessIssue]

    # Fire STOP — event-style hooks see termination reason before
    # on_loop_end cleanup runs. Return values are ignored (the loop
    # is already exiting); hooks should use on_loop_end for llm-call
    # side effects.
    emit_event(
        hooks,
        _LE.STOP,
        state=state,
        session_log=session_log,
        context=context,
        termination_reason=stop_reason,
    )

    for hook in hooks:
        if hasattr(hook, "on_loop_end"):
            extra = hook.on_loop_end(state, session_log, context, llm)
            if isinstance(extra, int):
                llm_calls += extra

    # Emit LoopEndEvent — skip if StreamingHook already emits it
    if stream is not None and _LoopEndEvent is not None and not _has_streaming_hook:
        stream.emit(
            _LoopEndEvent(
                total_steps=state.step_count,
                total_llm_calls=llm_calls,
                reason=stop_reason,
            )
        )

    # Build trace via injected callable
    elapsed = (time.time() - t0) * 1000
    _build_trace_fn = config.build_trace or (config.domain.build_trace if config.domain else None)
    if _build_trace_fn is not None:
        trace = _build_trace_fn(
            task=task,
            state=state,
            session_log=session_log,
            done=stop_reason == "done",
            llm=llm,
            llm_calls=llm_calls,
            elapsed_ms=elapsed,
        )
    else:
        trace = {
            "task": task,
            "steps": [s.to_dict() for s in state.steps],
            "llm_calls": llm_calls,
            "total_time_ms": elapsed,
            "conversation": _conv,
        }
    return trace


# ── Reactive Recovery Chain ──────────────────────────────────────


def _recovery_chain(
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
    *,
    hooks: list[Any] | None = None,
    conversation: Any | None = None,
) -> Any:
    """Multi-strategy recovery chain for prompt-too-long errors.

    Tries strategies in order, each at most once:
      1. Aggressive budget enforcement (shrink all results to 2KB)
      2. Emergency session log compression (emergency_truncate — routed
         through :func:`looplet.compact.run_compact` so
         ``PRE_COMPACT`` / ``POST_COMPACT`` events fire and users can
         swap in a custom :class:`CompactService`)
      3. Clear all old result data entirely

    After each strategy, rebuilds the prompt and retries the LLM call.
    Returns the response text if recovery succeeds, None if all fail.
    """
    extra_llm_calls = 0
    hooks = hooks or []

    # Pick the compact service: user-supplied or the built-in default.
    from looplet.compact import TruncateCompact, run_compact  # noqa: PLC0415

    _compact_service = config.compact_service or TruncateCompact()

    def _run_emergency_truncate(_state: Any, _sl: Any, _llm: Any, _sn: int) -> int:
        outcome = run_compact(
            _compact_service,
            hooks=hooks,
            state=_state,
            session_log=_sl,
            llm=_llm,
            conversation=conversation,
            step_num=_sn,
            reason="prompt_too_long",
        )
        return outcome.llm_calls_spent

    strategies = [
        ("budget_enforcement", _recovery_aggressive_budget),
        ("emergency_truncate", _run_emergency_truncate),
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
            state,
            session_log,
            context,
            build_briefing,
            build_prompt_fn,
            task,
            tools,
            config,
            step_num,
        )
        retry_result = llm_call_with_retry(
            llm,
            prompt,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            cancel_token=config.cancel_token,
            generate_kwargs=config.generate_kwargs or None,
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
