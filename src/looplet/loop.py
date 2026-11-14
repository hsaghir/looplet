"""Composable agent loop - domain-agnostic hook-based architecture.

The loop handles orchestration: LLM call → parse → dispatch → continue/stop.
Domain-specific behavior is injected via hooks and LoopConfig callables.
"""

from __future__ import annotations

import inspect
import logging
import time
import warnings
from collections.abc import Sequence
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)
from weakref import WeakKeyDictionary

from looplet.capabilities import ExecutionPolicy
from looplet.context_plan import ContextPlan, ScopedContextSource
from looplet.context_projection import ContextProjection
from looplet.hook_decision import (
    _hook_accepts_keyword,
    _invoke_hook,
    _invoke_hook_async,
    normalize_hook_return,
)
from looplet.prompt_preparation import render_projection
from looplet.scaffolding import (
    ContextBudgetSnapshot,
    LLMResult,
    NativeToolStats,
)
from looplet.scaffolding import (
    estimate_prompt_tokens as estimate_prompt_tokens,
)
from looplet.session import SessionLog
from looplet.tools import (
    BaseToolRegistry,
    ToolSpec,
    register_done_tool,
    tools_from,
)
from looplet.types import (
    AgentState,
    RunEnvelope,
    RunPhase,
    RunResult,
    RunStatus,
    Step,
    ToolCall,
    ToolContext,
    ToolResult,
    validate_run_transition,
)
from looplet.validation import validate_args as _validate_args


def _get_persist_threshold() -> int:
    """Read ``TOOL_RESULT_PERSIST_THRESHOLD_CHARS`` lazily.

    Lazy lookup so tests / runtime tuning that overwrites the
    constant on the :mod:`looplet.context_budget` module after import
    sees the new value (the dispatcher reads it on every tool call).
    """
    from looplet.context_budget import TOOL_RESULT_PERSIST_THRESHOLD_CHARS  # noqa: PLC0415

    return TOOL_RESULT_PERSIST_THRESHOLD_CHARS


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
_CallbackResult = TypeVar("_CallbackResult")


# ── LoopContext ──────────────────────────────────────────────────


@dataclass
class LoopContext:
    """Mutable handle to live loop state, passed to hooks via :meth:`LoopHook.bind`.

    Hooks that need long-lived access to ``state``, ``tools``,
    ``conversation``, or the current ``step_num`` outside their
    declared method signatures used to capture them in closures -
    which then refused to round-trip through ``to_config()`` because
    closures can't be serialised. ``LoopContext`` makes this
    first-class:

        class MyHook:
            def bind(self, ctx):
                self.ctx = ctx

            def pre_prompt(self, state, session_log, context, step_num):
                # Read live values from ctx anytime, not just from the
                # method args. Tools registry, conversation history,
                # current step_num, and shared resources are all here.
                if self.ctx.step_num > 5:
                    return self.ctx.tools.list_names()

    Mirrors :class:`looplet.types.ToolContext` (the runtime handle for
    tools) for symmetry. The loop owns the instance and mutates
    ``step_num`` and ``conversation`` as it progresses, so hooks read
    the latest values without having to re-bind.
    """

    state: Any
    session_log: Any
    conversation: Any
    tools: Any
    config: Any
    resources: dict[str, Any] = field(default_factory=dict)
    step_num: int = 0
    status: RunStatus = RunStatus.CREATED
    phase: RunPhase = RunPhase.STARTING
    termination_reason: str | None = None
    native_tool_stats: NativeToolStats = field(default_factory=NativeToolStats)
    context_budget: ContextBudgetSnapshot = field(default_factory=ContextBudgetSnapshot)
    policy_decisions: list[dict[str, Any]] = field(default_factory=list)
    run_envelope: RunEnvelope | None = None
    llm: Any = field(default=None, kw_only=True, repr=False)


# ── Hook Protocol ────────────────────────────────────────────────


@runtime_checkable
class LoopHook(Protocol):
    """Complete interface for composable loop hooks.

    Hooks inject domain-specific behavior into the generic agent loop.
    The loop accepts partial, duck-typed implementations of this interface.
    Implement only the methods you need; inheritance is not required.
    ``isinstance(hook, LoopHook)`` checks the complete interface.
    Use :func:`is_loop_hook` to recognize partial hooks.

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
        pre_prompt:      before each LLM call - inject briefing text (additive, all hooks contribute)
        build_briefing:  override the briefing section (first hook returning non-None wins)
        build_prompt:    override the entire prompt (first hook returning non-None wins)
        pre_dispatch:    before each tool call - intercept, cache, or deny
        post_dispatch:   after each tool call - inject follow-up context
        check_done:      when done() is called - reject premature completion
        should_stop:     after each step - force early termination
        should_compact:  at step start - trigger proactive compaction
        on_loop_end:     once after loop exits - cleanup

    Return types:
        All hook methods accept ``HookDecision`` as return type.
        Legacy returns (``str``, ``bool``, ``None``) still work
        and are auto-converted via ``normalize_hook_return()``.

    Precedence for prompt injection:
        1. ``build_prompt`` hook (first non-None wins) - full prompt override
        2. ``config.build_prompt`` callable - full prompt override
        3. ``build_briefing`` hook (first non-None wins) - briefing section only
        4. ``config.build_briefing`` callable - briefing section only
        5. ``pre_prompt`` hooks - additive text appended to briefing
        6. Default 7-section template from ``looplet.prompts``

    Optional ``bind(self, ctx: LoopContext) -> None`` method:
        Hooks may also define ``bind`` to receive a long-lived handle
        to live loop state (``state``, ``tools``, ``conversation``,
        ``resources``, current ``step_num``). Called once after preset
        load and before :meth:`pre_loop`. The loop mutates
        ``ctx.step_num`` and ``ctx.conversation`` as it progresses,
        so a hook that stores ``self.ctx = ctx`` reads live values
        forever after - replacing the closures-over-loop-state
        anti-pattern that broke ``to_config()`` workspace round-trip.

        ``bind`` is not part of the Protocol body, so complete adapters
        need not implement it for ``isinstance(h, LoopHook)`` checks.
    """

    def pre_loop(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
    ) -> None:
        """Called once at the start of the loop, before any steps.

        Use for initialization, state setup, or emitting start events.

        Hooks that need access to the tool registry (e.g. to register
        derived tools at load time) can declare an optional ``tools=``
        kwarg::

            def pre_loop(self, state, session_log, context, tools=None):
                tools.register(my_derived_spec)

        The loop introspects the signature once and only passes
        ``tools`` to hooks that accept it; legacy 3-arg hooks keep
        working unchanged.
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
        (AND semantics - any single deny blocks).

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
        a ``prompt_too_long`` error) - use this when you want to
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
        wins - subsequent hooks don't run for this slot.

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
        :attr:`LifecycleEvent.POST_TOOL_FAILURE` follows the corresponding
        tool-slot contract. Model events support context and stop effects;
        terminal :attr:`LifecycleEvent.STOP` is observation-only. See
        ``hook_effect_fields`` for explicit applicability.

        Implementing ``on_event`` is strictly additive - hooks can
        still implement per-method slots and mix both styles.
        """
        ...


# ── Loop Configuration ──────────────────────────────────────────


@dataclass
class DomainAdapter:
    """Bundle optional domain-specific callables used by a loop.

    Grouping keeps :class:`LoopConfig` flat and readable, and gives
    agent packages a single handle to pass around instead of threading
    separate callables through config kwargs.

    When a :class:`LoopConfig` has a :attr:`LoopConfig.domain` set,
    each adapter field seeds the corresponding flat ``LoopConfig``
    field only if the flat field is ``None``. Per-field overrides
    on ``LoopConfig`` therefore win over the bundled adapter, which
    wins over the loop's built-in defaults.

    All fields are optional - provide only what differs from the
    defaults. See :class:`LoopConfig` for callable signatures.
    """

    build_briefing: Callable[..., str] | None = None
    extract_entities: Callable[..., list[str]] | None = None
    build_trace: Callable[..., Any] | None = None
    build_prompt: Callable[..., str] | None = None
    extract_step_metadata: Callable[..., tuple[list[str], list[str]]] | None = None
    checkpoint_state: Callable[["LoopContext"], dict[str, Any]] | None = None
    restore_checkpoint_state: Callable[["LoopContext", dict[str, Any]], None] | None = None


@dataclass
class LoopConfig:
    """Configuration for the composable agent loop.

    **Start here** - most fields have sensible defaults. For your
    first agent, you only need::

        config = LoopConfig(max_steps=10)

    Omit ``state`` when calling the loop to create its default state from
    this config. Supply a custom or restored state when the host needs it.

    Essential fields (set these first):
            - ``max_steps`` - charged-step budget for ``DefaultState``.
                Successful free calls do not consume this budget.
                It does not bound total tool or model calls.
        This is authoritative for ``DefaultState``; a supplied state with a
        different limit is synchronized with a warning. The loop's explicit
        ``max_steps=`` convenience argument overrides the config value.
      - ``system_prompt`` - who the agent is
    - ``compact_service`` - optionally replace standard context management
      - ``checkpoint_dir`` - crash-safe auto-resume (one directory path)

    Everything else is optional and can be added later as your agent
    matures. See the tutorial in README.md for a progressive walkthrough.

    Config objects are live host configuration, not immutable snapshots.
    Loop argument overrides update a supplied config in place.
    Use separate configs when runs need independent settings.
    """

    max_steps: int = 15
    max_tokens: int | None = None
    system_prompt: str = ""
    temperature: float | None = None
    recovery_temperature: float | None = None
    # Name of the tool that signals task completion.
    done_tool: str = "done"

    harness_profile: str = field(default="execution-v1", kw_only=True)
    """Versioned shared defaults; ``minimal`` disables automatic composition."""

    disabled_defaults: list[str] = field(default_factory=list, kw_only=True)
    """Default capabilities to omit by name.

    Names are compaction, stagnation, tool_budget, history, result_retention,
    provenance, checkpointing, branching, and persistence (all persistence defaults).

    Explicitly supplied services and hooks are never removed by this setting.
    """

    run_dir: str | None = field(default=None, kw_only=True)
    """Private run directory for default evidence, results, checkpoints, and branches.

    Standard profiles allocate a unique directory under ``.looplet/runs``.
    Explicit stores, recorders, and checkpoint paths take precedence.
    ``minimal`` or ``disabled_defaults=["persistence"]`` disables this composition.
    """

    done_tools: list[str] = field(default_factory=list)
    """Additional terminal sentinel tools for host-side configuration.

    By default the loop ends only when the agent invokes the single
    ``done_tool``. Host code may add tools such as ``report`` and
    ``escalate`` through ``LoopConfig(done_tools=[...])``. Schema-v2
    cartridges declare one ``done_tool`` and express distinct outcomes
    in its payload and output schema; ``done_tools`` is not a cartridge field.

    The legacy ``done_tool`` is always treated as terminal too;
    ``done_tools`` is *additive*. Empty list (the default) preserves
    single-sentinel behaviour exactly.
    """

    done_tool_schemas: dict[str, OutputSchema] = field(default_factory=dict)
    """Per-sentinel ``OutputSchema`` map for host-configured terminal tools.
    Keyed by sentinel tool name.

    When a secondary sentinel from :attr:`done_tools` is invoked and an
    entry exists here, the loop validates the call's args against the
    matching schema with the same quality-gate behaviour as
    :attr:`output_schema` on the primary :attr:`done_tool`. Sentinels
    without an entry are unvalidated.

    Set directly on ``LoopConfig`` for hand-built loops. Schema-v2
    cartridges express outcomes through their primary terminal tool's
    ``tool.yaml: output_schema:`` block.
    """

    @property
    def uses_typed_history(self) -> bool:
        """Preserve native history independently of assistant completion policy."""
        return self.assistant_turn_policy is not None or (
            self.harness_profile == "execution-v1" and self.use_native_tools
        )

    def validate_assistant_turns(self, *, require_binding: bool = True) -> None:
        """Validate assistant-turn limits and required host policy binding."""
        from looplet.assistant_turns import validate_assistant_config

        validate_assistant_config(self, require_binding=require_binding)

    def resolve_terminal_outcomes(self) -> dict[str, OutputSchema | None]:
        """Return the effective terminal name-to-schema table.

        The primary output schema wins over its plural-map entry. Schema
        entries alone never make ordinary tools terminal. Returned tables
        are independent snapshots; schema objects retain their identity.
        """
        names = dict.fromkeys((self.done_tool, *(self.done_tools or ())))
        schemas = self.done_tool_schemas or {}
        outcomes = {name: schemas.get(name) for name in names}
        if self.output_schema is not None:
            outcomes[self.done_tool] = self.output_schema
        return outcomes

    tool_result_persist_dir: str | None = None
    """Optional directory for Layer-1 persist-and-preview of large
    tool results. When set, tool results whose serialized size exceeds
    :data:`looplet.context_budget.TOOL_RESULT_PERSIST_THRESHOLD_CHARS`
    are written to ``<persist_dir>/tool-output-<hash>.txt`` and the
    LLM-facing result is replaced with a preview + the file path. The
    model can then read more on demand via its own file-reading tools.

    Uses a per-session ``tool-results/`` directory.
    When ``None`` (default) only inline truncation applies - the full
    output simply gets capped at ``TOOL_RESULT_MAX_CHARS``.
    """

    max_turn_continuations: int = 0
    """When > 0, ``llm_call_with_retry`` will issue up to this many
    follow-up calls for a single step if the backend reports
    ``stop_reason == "max_tokens"`` mid-response, concatenating outputs
    so long thoughts aren't truncated. Requires the backend to expose
    ``last_stop_reason`` after each call. Default ``0`` (off)."""

    allow_text_completion: bool = False
    """Accept a tool-free assistant message as the primary terminal payload.

    The payload is ``{"answer": text}`` and still passes ``check_done``
    hooks and the primary output schema. The recorded terminal step is
    marked as an assistant message; no tool body or provider tool-result
    callback is invoked. Empty, refused, truncated, and malformed tool
    responses remain invalid. Default ``False`` requires a terminal tool.
    """

    assistant_turn_policy: Callable[..., Any] | None = field(default=None, kw_only=True)
    requires_assistant_turn_policy: bool = field(default=False, kw_only=True)
    """Host policy for valid assistant text. Requires an explicit model-call limit."""

    max_model_calls: int | None = field(default=None, kw_only=True)
    """Maximum backend invocations, including recovery. None preserves legacy limits."""

    max_assistant_turns: int | None = field(default=None, kw_only=True)
    """Optional limit on consecutive assistant-only turns. No implicit default applies."""

    # Domain-specific callables - injected by the agent

    build_briefing: Callable[..., str] | None = None
    """Callable[[state, session_log, context], str] - builds the briefing
    section injected at the top of each prompt."""

    extract_entities: Callable[..., list[str]] | None = None
    """Callable[[data], list[str]] - extracts entity strings from a tool
    result's data for entity tracking and session log recording."""

    build_trace: Callable[..., Any] | None = None
    """Callable[[Any, SessionLog, Any], Any] - builds the final output
    artifact from (state, session_log, context).

    Receives keyword args: task, state, session_log, done, llm, llm_calls,
    elapsed_ms.  Returns any serialisable object; stored as the generator
    return value.
    """

    build_prompt: Callable[..., str] | None = None
    """Callable[..., str] - builds the full LLM prompt from loop state.

    Receives keyword args: task, tool_catalog, state_summary,
    context_history, step_number, max_steps, session_log, briefing.
    """

    context_planner: Callable[..., ContextPlan | None] | None = None
    """Optional host planner that describes prompt selection and budgets.

    The returned :class:`ContextPlan` is passed to
    :class:`ContextProjection`; prompt rendering remains unchanged.
    """

    extract_step_metadata: Callable[..., tuple[list[str], list[str]]] | None = None
    """Callable[[Any, int], tuple[list[str], list[str]]] - returns
    (findings, highlights) from (state, step_num).

    Called after each non-done tool dispatch to gather per-step metadata
    for the session log.
    """

    domain: "DomainAdapter | None" = None
    """Optional :class:`DomainAdapter` bundling the five domain callables
    above. When set, each adapter field seeds the matching flat field
    above only if that flat field is ``None``. Direct field assignments
    on :class:`LoopConfig` therefore win over the adapter. Prefer this
    when composing a reusable agent package - pass one adapter instead
    of threading five callables through config kwargs."""

    use_native_tools: bool = True
    """If True, pass tool schemas to the LLM and parse tool_use blocks
    instead of JSON text. Default True - a missing or unsupported
    ``generate_with_tools`` raises a visible error rather than switching
    protocols. Set to False to explicitly use the JSON-text path."""

    tool_view_selector: Callable[..., Sequence[str] | None] | None = None
    """Optional host callback selecting model-visible tool names per turn.

    Called with ``step_num``, ``state``, ``tools``, and ``task``. ``None``
    exposes the full registry. Selection does not change dispatch authority.
    """

    concurrent_dispatch: bool = False
    """If True, dispatch non-dependent tool calls in parallel via
    ThreadPoolExecutor. Default False - some backends and tools
    are not thread-safe."""

    max_parallel_calls: int = 10
    """Maximum number of concurrent-safe calls executed at once."""

    streaming_tool_execution: bool = False
    """If True, start executing concurrent-safe native tool calls via
    StreamingToolExecutor as their blocks finish streaming from the LLM.
    Provider recovery stops once early tool work has been submitted.
    Hooks and checkpointing disable early execution."""

    reactive_recovery: bool = True
    """If True, attempt multi-strategy recovery when a prompt exceeds
    the context window (prompt-too-long error). Default True - essential
    for reliability in long sessions."""

    max_briefing_tokens: int | None = None
    """Max estimated tokens for the briefing section (all hook pre_prompt
    outputs combined).  When exceeded, later hook outputs are dropped with
    a truncation note.  None = no limit.
    """

    # ── Recent-results window (per-cartridge override of env defaults) ─

    context_window_steps: int | None = None
    """How many recent steps to inline in the LLM's RECENT RESULTS section.

    The default (``None``) reads :data:`looplet.context_budget.CONTEXT_WINDOW_STEPS`
    (env var ``CONTEXT_WINDOW_STEPS``, default 5). For chained tool-use
    cartridges where step N references data from step N-K (K > 5), set
    this higher so the source data stays in the LLM's view. A SOC
    triage cartridge that pivots from ``get_alert`` (step 1) to
    ``lookup_user`` (step 8) needs at least 8 here, or it will
    hallucinate a username that wasn't in its context.
    """

    context_inline_per_step_chars: int | None = None
    """Per-step soft cap on inlined recent-result size (characters).
    ``None`` reads :data:`looplet.context_budget.CONTEXT_INLINE_PER_STEP_CHARS`
    (env default 3000). Increase for tools that return rich JSON the
    agent must reason over downstream.
    """

    context_window_total_chars: int | None = None
    """Aggregate cap across the entire recent-results window. ``None``
    reads :data:`looplet.context_budget.CONTEXT_WINDOW_TOTAL_CHARS`
    (env default 20 000). The largest steps are progressively
    truncated when the total exceeds this.
    """

    # ── Optional wired capabilities ──────────────────────────────

    router: ModelRouter | None = None
    """When set, ``router.select(purpose='reasoning')`` is called at each
    step instead of the ``llm`` argument passed directly.  Import from
    ``looplet.router`` - e.g. ``SimpleRouter``, ``FallbackRouter``.
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
    """Explicit context strategy for proactive and reactive compaction.
    The standard harness fills ``None`` with deterministic
    :class:`DefaultCompactService` for the invocation; the minimal profile
    retains reactive :class:`TruncateCompact`. :func:`run_compact` emits
    ``PRE_COMPACT`` / ``POST_COMPACT`` events.
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

    checkpoint_state: Callable[["LoopContext"], dict[str, Any]] | None = None
    """Optional callback returning JSON-safe cartridge state for checkpoints."""

    restore_checkpoint_state: Callable[["LoopContext", dict[str, Any]], None] | None = None
    """Optional callback restoring cartridge state from a checkpoint."""

    memory_sources: list[Any] = field(default_factory=list)
    """Optional list of ``PersistentMemorySource`` objects rendered into
    the default prompt's top ``MEMORY`` section on every turn. Each
    source must expose ``load(state) -> str | None``. When ``build_prompt``
    is user-supplied, the loop still renders memory but passes it to
    the custom function as a ``memory=`` kwarg.
    """

    scoped_context_sources: list[ScopedContextSource] = field(default_factory=list)
    """Host-selected, scoped sources evaluated each turn. Disabled by default."""

    scoped_context_budget_tokens: int | None = None
    """Token budget for selected scoped sources, separate from the full prompt budget."""

    cache_policy: "CachePolicy | None" = None
    """Optional :class:`looplet.cache.CachePolicy` declaring which
    stable prompt sections (system prompt, tool schemas, memory) should
    carry ``cache_control`` markers. When set and the backend's
    ``generate`` / ``generate_with_tools`` accepts a ``cache_breakpoints``
    keyword, the loop computes per-turn :class:`CacheBreakpoint` lists
    and passes them through. The shipped OpenAI/Anthropic backends
    honour it; backends without the keyword keep working unchanged.
    Caching is strictly opt-in and additive.
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

    execution_policy: ExecutionPolicy | None = field(default=None, kw_only=True)
    """Host-owned capabilities available to declared tools in this run."""

    context_window: int = 128_000
    """Maximum context window (in tokens) for the backend.  Used by:

    * The **pre-flight** prompt-size check - if the estimated prompt exceeds
      ``context_window - 3000`` tokens, reactive recovery fires *before*
      the LLM call (avoiding a wasted API call).
        * ``ThresholdCompactHook`` - the standard harness supplies a proactive
            trigger and compact service based on this value.

    Override to match your actual backend's window:
    200 000 for Claude, 128 000 for GPT-4o, 32 000 for GPT-4, etc.
    Default: 128 000."""

    render_messages_override: Callable[..., str] | None = None
    """Render the prompt from a :class:`ContextProjection`.

    New callbacks receive one keyword argument, ``projection``. Legacy
    callbacks accepting ``messages``, ``default_prompt``, and ``step_num``
    remain supported. The returned string is what the backend sees.

    Prefer this over :attr:`build_prompt` when you want to inspect or
    mutate the full conversation thread (compaction, redaction, role
    reordering, role swaps) before send. Prefer :attr:`build_prompt`
    when you only want to change the section layout."""

    tool_metadata: dict[str, Any] = field(default_factory=dict)
    """Static key-value pairs merged into every ``ToolContext.metadata``.

    Use this to pass application-level configuration to tools without
    threading it through ``state.metadata`` (which is agent-level) or
    closures (which are invisible to provenance).

    ``tool_metadata`` is merged *under* ``state.metadata`` - state
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

    run_envelope: RunEnvelope | None = None
    """Optional host identity, tenancy, deadline, and policy context."""

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

    @overload
    def resolve_callable(
        self, name: str, default: Callable[..., _CallbackResult]
    ) -> Callable[..., _CallbackResult]: ...

    @overload
    def resolve_callable(self, name: str, default: None = None) -> Callable[..., Any] | None: ...

    def resolve_callable(
        self, name: str, default: Callable[..., Any] | None = None
    ) -> Callable[..., Any] | None:
        """Resolve one domain callback; explicit config wins over the adapter."""
        if name not in (
            "build_briefing",
            "build_prompt",
            "build_trace",
            "extract_entities",
            "extract_step_metadata",
            "checkpoint_state",
            "restore_checkpoint_state",
        ):
            raise ValueError(f"not a domain callback: {name}")
        configured = getattr(self, name)
        if configured:
            return configured
        adapted = getattr(self.domain, name, None) if self.domain else None
        return adapted or default

    def record_sources(self, sources: dict[str, str]) -> None:
        """Record known declaration origins without adding serialized config fields."""
        origins = dict(getattr(self, "_configuration_sources", {}))
        snapshots = dict(getattr(self, "_configuration_source_values", {}))
        for name, source in sources.items():
            if name not in self.__dataclass_fields__:
                continue
            origins[name] = source
            snapshots[name] = _configuration_value(getattr(self, name))
        self._configuration_sources = origins
        self._configuration_source_values = snapshots

    def explain(self, *, include_sensitive: bool = False) -> dict[str, dict[str, Any]]:
        """Describe effective values and known origins without running components.

        Python constructors do not distinguish an explicitly supplied default
        from an omitted one. Such values are honestly labelled ``config``.
        Prompts, tool metadata, provider kwargs and host envelopes are hidden
        unless the host explicitly requests them. Opaque objects expose only
        their type, never ``repr`` or executable state.
        """
        origins = getattr(self, "_configuration_sources", {})
        snapshots = getattr(self, "_configuration_source_values", {})
        sensitive = {"system_prompt", "tool_metadata", "generate_kwargs", "run_envelope"}
        explained: dict[str, dict[str, Any]] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            source = origins.get(name, "config")
            described = _configuration_value(value)
            if name in snapshots and described != snapshots[name]:
                source = "host mutation"
            if (
                name
                in (
                    "build_briefing",
                    "build_prompt",
                    "build_trace",
                    "extract_entities",
                    "extract_step_metadata",
                    "checkpoint_state",
                    "restore_checkpoint_state",
                )
                and not value
                and self.domain is not None
            ):
                adapted = getattr(self.domain, name, None)
                if adapted:
                    described = _configuration_value(adapted)
                    source = "domain"
            hidden = name in sensitive and not include_sensitive
            explained[name] = {
                "value": None if hidden else described,
                "source": source,
                "redacted": hidden,
            }
        return explained


def _configuration_value(value: Any, *, depth: int = 0) -> Any:
    """Bounded structural description that never invokes user code."""
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if depth < 5 and type(value) in (list, tuple):
        return [_configuration_value(item, depth=depth + 1) for item in value[:100]]
    if depth < 5 and type(value) is dict:
        return {
            name: _configuration_value(item, depth=depth + 1)
            for name, item in list(value.items())[:100]
            if type(name) is str
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _resolve_loop_config(
    config: LoopConfig | None,
    *,
    max_steps: int | None = None,
    system_prompt: str | None = None,
) -> LoopConfig:
    """Resolve entry-point overrides without replacing live host controls."""
    effective = config if config is not None else LoopConfig()
    sources = {name: "default" for name in effective.__dataclass_fields__} if config is None else {}
    if max_steps is not None:
        effective.max_steps = max_steps
        sources["max_steps"] = "loop argument"
    if system_prompt is not None:
        effective.system_prompt = system_prompt
        sources["system_prompt"] = "loop argument"
    effective.record_sources(sources)
    return effective


def _resolve_loop_state(state: AgentState | None, config: LoopConfig) -> AgentState:
    """Apply the configured budget to default state without replacing custom rules."""
    from looplet.types import DefaultState

    if state is None:
        return DefaultState(max_steps=config.max_steps)
    if isinstance(state, DefaultState) and state.max_steps != config.max_steps:
        warnings.warn(
            f"DefaultState(max_steps={state.max_steps}) differs from "
            f"LoopConfig(max_steps={config.max_steps}); using the config "
            f"value. Pass the same max_steps to both to silence this.",
            UserWarning,
            stacklevel=3,
        )
        state.max_steps = config.max_steps
    return state


def _default_extract_entities(data: Any, state: Any = None) -> list[str]:
    """Fallback: no entity extraction.

    Accepts a ``state`` kwarg (defaulting to ``None``) so domain
    extractors that need the live state can have a stateless signature
    in ``${py:...}`` workspace refs and still receive it from the loop.
    Stateless extractors keep the 1-arg form working - the loop only
    passes ``state`` when the callable's signature accepts it.
    """
    return []


def _default_build_briefing(state: Any, session_log: SessionLog, context: Any) -> str:
    """Fallback: empty briefing."""
    return ""


def _default_extract_step_metadata(state: Any, step_num: int) -> tuple[list[str], list[str]]:
    """Fallback: no step metadata extraction."""
    return [], []


def _bind_loop_context(loop_ctx: LoopContext, hooks: list[Any]) -> None:
    """Bind the live loop context to hooks that opt into the protocol."""

    for hook in hooks:
        bind_fn = getattr(hook, "bind", None)
        if bind_fn is None:
            continue
        try:
            import inspect as _inspect  # noqa: PLC0415

            sig = _inspect.signature(bind_fn)
            positional = [
                parameter
                for parameter in sig.parameters.values()
                if parameter.kind
                in (
                    _inspect.Parameter.POSITIONAL_ONLY,
                    _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    _inspect.Parameter.VAR_POSITIONAL,
                )
            ]
            if positional:
                bind_fn(loop_ctx)
        except (TypeError, ValueError):
            try:
                bind_fn(loop_ctx)
            except TypeError:
                pass


def _set_run_lifecycle(
    loop_ctx: LoopContext,
    *,
    status: RunStatus | None = None,
    phase: RunPhase | None = None,
    termination_reason: str | None = None,
    terminal_tool: str | None = None,
) -> None:
    """Update the live lifecycle view and compatible state metadata."""

    next_status = loop_ctx.status if status is None else status
    next_phase = loop_ctx.phase if phase is None else phase
    validate_run_transition(
        loop_ctx.status,
        loop_ctx.phase,
        next_status=next_status,
        next_phase=next_phase,
    )

    if status is not None:
        loop_ctx.status = status
    if phase is not None:
        loop_ctx.phase = phase
    if termination_reason is not None:
        loop_ctx.termination_reason = termination_reason

    state = loop_ctx.state
    if loop_ctx.phase is RunPhase.STARTING:
        try:
            setattr(state, "_looplet_run_stats", {"llm_calls": 0, "usage_calls": 0})
        except AttributeError:
            pass
    if loop_ctx.phase is RunPhase.STARTING or terminal_tool is not None:
        try:
            setattr(state, "_accepted_terminal_tool", terminal_tool)
        except AttributeError:
            pass
    for name, value in (
        ("run_status", loop_ctx.status.value),
        ("run_phase", loop_ctx.phase.value),
        ("termination_reason", loop_ctx.termination_reason),
    ):
        try:
            setattr(state, name, value)
        except AttributeError:
            pass
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        metadata.update(
            {
                "run_status": loop_ctx.status.value,
                "run_phase": loop_ctx.phase.value,
            }
        )
        if loop_ctx.termination_reason is not None:
            metadata["termination_reason"] = loop_ctx.termination_reason


def _count_usage_calls(result: LLMResult) -> int:
    from looplet.run_records import _has_diagnostic_usage

    return sum(1 for record in result.model_calls if _has_diagnostic_usage(record.get("usage")))


def _accumulate_step_usage(state: Any, step_usage: Any) -> None:
    if not isinstance(step_usage, dict) or not step_usage:
        return
    meta = getattr(state, "metadata", None)
    if not isinstance(meta, dict):
        return
    meta["last_usage"] = dict(step_usage)
    totals = meta.get("usage_total")
    if not isinstance(totals, dict):
        totals = {}
    for key, value in step_usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] = totals.get(key, 0) + value
    meta["usage_total"] = totals


def _record_run_stats(
    state: Any,
    *,
    llm_calls: int,
    usage_calls: int | None = None,
    duration_ms: float | None = None,
    config: LoopConfig | None = None,
) -> None:
    """Retain host-only measurements without changing model/tool metadata."""
    from looplet.assistant_turns import TURN_STATE_KEY
    from looplet.run_records import _has_diagnostic_usage

    metadata = getattr(state, "metadata", {})
    turns = metadata.get(TURN_STATE_KEY) if isinstance(metadata, dict) else None
    if isinstance(turns, dict) and type(turns.get("model_calls")) is int:
        llm_calls = (
            turns["model_calls"]
            if turns.get("max_model_calls") is not None
            else max(llm_calls, turns["model_calls"])
        )
    if isinstance(turns, dict) and isinstance(turns.get("requests"), list):
        turns_usage_calls = sum(
            1
            for request in turns["requests"]
            if isinstance(request, dict) and _has_diagnostic_usage(request.get("usage"))
        )
        usage_calls = (
            turns_usage_calls
            if turns.get("max_model_calls") is not None or usage_calls is None
            else max(usage_calls, turns_usage_calls)
        )
    stats = getattr(state, "_looplet_run_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        try:
            setattr(state, "_looplet_run_stats", stats)
        except AttributeError:
            return
    stats["llm_calls"] = llm_calls
    if usage_calls is not None:
        stats["usage_calls"] = usage_calls
    if isinstance(turns, dict) and type(turns.get("model_calls")) is int:
        stats["model_requests"] = turns["model_calls"]
    if config is not None:
        stats["model_turn_config"] = {
            name: getattr(config, name)
            for name in (
                "allow_text_completion",
                "max_turn_continuations",
                "max_tokens",
                "done_tool",
            )
        }
        if config.done_tools:
            stats["model_turn_config"]["done_tools"] = list(config.done_tools)
        for name in ("max_model_calls", "max_assistant_turns"):
            if getattr(config, name) is not None:
                stats["model_turn_config"][name] = getattr(config, name)
        if config.assistant_turn_policy is not None or config.requires_assistant_turn_policy:
            stats["model_turn_config"]["requires_assistant_turn_policy"] = True
    if duration_ms is not None:
        stats["duration_ms"] = duration_ms


def _record_policy_decision(state: Any, decision: Any) -> None:
    """Persist a JSON-safe policy audit record on the live run."""
    record = getattr(decision, "policy_decision", None)
    if record is None:
        return
    data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    envelope = getattr(state, "run_envelope", None)
    if envelope is not None:
        data.setdefault("run_id", envelope.run_id)
        if envelope.tenant_id is not None:
            data.setdefault("tenant_id", envelope.tenant_id)
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        decisions = metadata.setdefault("policy_decisions", [])
        if isinstance(decisions, list):
            decisions.append(data)


def _policy_checkpoint_metadata(state: Any, checkpoint_store: Any = None) -> dict[str, Any]:
    from looplet.assistant_turns import assistant_checkpoint_metadata

    metadata = getattr(state, "metadata", None)
    decisions = metadata.get("policy_decisions") if isinstance(metadata, dict) else None
    return {
        **({"policy_decisions": list(decisions)} if isinstance(decisions, list) else {}),
        **assistant_checkpoint_metadata(state, checkpoint_store=checkpoint_store),
    }


def _deadline_expired(loop_ctx: LoopContext) -> bool:
    """Cancel the run when its host deadline has passed."""
    envelope = loop_ctx.run_envelope
    if envelope is None or envelope.deadline_at is None:
        return False
    if time.time() < envelope.deadline_at:
        return False
    cancel_token = getattr(loop_ctx.config, "cancel_token", None)
    if cancel_token is not None:
        cancel_token.cancel()
    _set_run_lifecycle(
        loop_ctx,
        status=RunStatus.RUNNING,
        phase=RunPhase.STOPPING,
        termination_reason="deadline_exceeded",
    )
    return True


def _build_tool_ctx(
    config: "LoopConfig",
    *,
    tools: Any = None,
    hooks: list[Any] | None = None,
    tool_call: ToolCall | None = None,
    step_num: int = 0,
    state: Any = None,
    session_log: Any = None,
    llm: Any = None,
    async_llm: Any = None,
) -> ToolContext:
    """Build a ToolContext for tool dispatch.

    Always returns a ToolContext - tools that declare ``ctx`` should
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
    _tool_llm = _scope_llm_for_tool(llm, tool_call, step_num) if llm is not None else None

    # Populate metadata: config.tool_metadata (defaults) + state.metadata
    # (per-run overrides). State wins on key conflicts.
    _metadata: dict[str, Any] = dict(config.tool_metadata) if config.tool_metadata else {}
    if tool_call is not None:
        spec = getattr(tools, "_tools", {}).get(tool_call.tool)
        if spec is not None:
            idempotency = getattr(spec, "idempotency", "unknown")
            retryable = bool(getattr(spec, "retryable", False))
            if idempotency != "unknown" or retryable:
                _metadata.setdefault("tool_idempotency", idempotency)
                _metadata.setdefault("tool_retryable", retryable)
    if state is not None:
        _state_meta = getattr(state, "metadata", None)
        if isinstance(_state_meta, dict):
            _metadata.update(
                {
                    key: value
                    for key, value in _state_meta.items()
                    if key
                    not in {"run_status", "run_phase", "termination_reason", "run_dir", "trace_dir"}
                }
            )

    return ToolContext(
        cancel_token=config.cancel_token,
        cwd=(config.execution_policy.workspace_root if config.execution_policy else None),
        workspace_root=(
            config.execution_policy.workspace_root if config.execution_policy else None
        ),
        request_approval=config.approval_handler,
        on_progress=_progress_fn,
        llm=_tool_llm,
        async_llm=(
            _scope_llm_for_tool(async_llm, tool_call, step_num)
            if async_llm is not None
            else (_SyncToAsyncLLM(_tool_llm) if _tool_llm is not None else None)
        ),
        parent_budget_remaining=getattr(state, "budget_remaining", None),
        parent_hooks=tuple(_hooks),
        run_envelope=config.run_envelope,
        execution_policy=config.execution_policy,
        metadata=_metadata,
    )


class _SyncToAsyncLLM:
    """Expose an awaitable ``ctx.async_llm`` interface over a synchronous backend."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str = "",
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        import asyncio
        import contextvars

        call_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "temperature": temperature,
            **kwargs,
        }
        if inspect.iscoroutinefunction(self._backend.generate):
            return await self._backend.generate(prompt, **call_kwargs)
        ctx = contextvars.copy_context()
        result = await asyncio.to_thread(
            ctx.run, lambda: self._backend.generate(prompt, **call_kwargs)
        )
        if inspect.isawaitable(result):
            return await result
        return result

    async def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        system_prompt: str = "",
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        import asyncio
        import contextvars

        call_kwargs: dict[str, Any] = {
            "tools": tools,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "temperature": temperature,
            **kwargs,
        }
        if inspect.iscoroutinefunction(self._backend.generate_with_tools):
            return await self._backend.generate_with_tools(prompt, **call_kwargs)
        ctx = contextvars.copy_context()
        result = await asyncio.to_thread(
            ctx.run, lambda: self._backend.generate_with_tools(prompt, **call_kwargs)
        )
        if inspect.isawaitable(result):
            return await result
        return result


def _scope_llm_for_tool(llm: Any, tool_call: ToolCall | None, step_num: int | None = None) -> Any:
    """Preserve tool-call scope in recording and replay backends.

    Backends without call scoping pass through unchanged.
    """
    tool_name = getattr(tool_call, "tool", "unknown") if tool_call else "unknown"

    # Unwrap ResilientBackend / CostTracker to find the recording layer
    inner = llm
    recording = None
    for _ in range(8):  # bounded unwrap
        if callable(getattr(inner, "_scope_calls", None)):
            recording = inner
            break
        attributes = getattr(inner, "__dict__", {})
        inner = next(
            (
                attributes[name]
                for name in ("backend", "_backend", "_inner")
                if attributes.get(name) is not None
            ),
            None,
        )
        if inner is None:
            break

    if recording is None:
        return llm  # no recording layer - pass through unchanged

    return _ScopedLLMProxy(llm, recording, f"tool:{tool_name}", step_num)


_ACTIVE_SCOPED_PROXIES: ContextVar[frozenset[object]] = ContextVar(
    "looplet_active_scoped_proxies", default=frozenset()
)


class _ScopedLLMProxy:
    """Tag helper requests with their task-local tool scope and step."""

    def __init__(
        self,
        backend: Any,
        recording: Any,
        scope: str,
        step_num: int | None = None,
        *,
        _scope_id: object | None = None,
    ) -> None:
        self._backend = backend
        self._recording = recording
        self._scope = scope
        self._step_num = step_num
        self._scope_id = _scope_id if _scope_id is not None else object()
        if inspect.iscoroutinefunction(backend.generate):
            self.generate = self._async_method(backend.generate)
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = (
                self._async_method(backend.generate_with_tools)
                if inspect.iscoroutinefunction(backend.generate_with_tools)
                else self._generate_with_tools_impl
            )

    def __getattr__(self, name: str) -> Any:
        from looplet._backend_contract import backend_attribute

        return backend_attribute(self._backend, name)

    @contextmanager
    def _enter_own_scope(self) -> Generator[None, None, None]:
        active = _ACTIVE_SCOPED_PROXIES.get()
        if self._scope_id in active:
            yield
            return
        token = _ACTIVE_SCOPED_PROXIES.set(active | {self._scope_id})
        try:
            with self._recording._scope_calls(self._scope, step_num=self._step_num):
                yield
        finally:
            _ACTIVE_SCOPED_PROXIES.reset(token)

    @contextmanager
    def _scope_calls(
        self, scope: str, *, step_num: int | None = None
    ) -> Generator[None, None, None]:
        with self._enter_own_scope():
            with self._recording._scope_calls(scope, step_num=step_num):
                yield

    def for_backend(self, selected: Any) -> "_ScopedLLMProxy":
        inner = (
            self._recording.for_backend(selected)
            if isinstance(self._recording, _ScopedLLMProxy)
            else selected
        )
        return _ScopedLLMProxy(
            inner,
            inner,
            self._scope,
            self._step_num,
            _scope_id=self._scope_id,
        )

    def _async_method(self, method: Any) -> Any:
        from functools import wraps

        @wraps(method)
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            with self._enter_own_scope():
                return await method(*args, **kwargs)

        return invoke

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> str:
        with self._enter_own_scope():
            return self._backend.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                **options,
            )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        with self._enter_own_scope():
            return self._backend.generate_with_tools(
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                **options,
            )


def _event_hook_callbacks(hooks: list[Any], event: Any) -> Generator[tuple[Any, Any], None, None]:
    """Select ordered event subscribers without invoking or flattening effects."""
    for hook in hooks:
        callback = getattr(hook, "on_event", None)
        if callback is None:
            continue
        subscribed = getattr(hook, "_subscribed_events", None)
        if subscribed is not None and event not in subscribed:
            continue
        equivalent = _EVENT_METHOD_EQUIV.get(event)
        if equivalent is not None and hasattr(hook, equivalent):
            continue
        yield hook, callback


def emit_event(
    hooks: list[Any],
    event: Any,
    **payload_kwargs: Any,
) -> list[Any]:
    """Dispatch a :class:`LifecycleEvent` to every hook that opts in."""
    from looplet.turn_engine import _emit_event_gen, run_sync_gen

    return run_sync_gen(_emit_event_gen(hooks, event, **payload_kwargs))


async def emit_event_async(
    hooks: list[Any],
    event: Any,
    **payload_kwargs: Any,
) -> list[Any]:
    """Async counterpart to :func:`emit_event`."""
    from looplet.turn_engine import _emit_event_gen, run_async_gen

    return await run_async_gen(_emit_event_gen(hooks, event, **payload_kwargs))


async def _emit_hook_decision_event_async(
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
    """Async version of the structured HookDecision audit event."""
    from looplet.turn_engine import _emit_hook_decision_gen, run_async_gen

    await run_async_gen(
        _emit_hook_decision_gen(
            hooks,
            decision=decision,
            hook_slot=hook_slot,
            hook_name=hook_name,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            extra=extra,
        )
    )


async def _bind_loop_context_async(loop_ctx: LoopContext, hooks: list[Any]) -> None:
    """Bind hooks while awaiting async ``bind`` implementations."""
    for hook in hooks:
        bind_fn = getattr(hook, "bind", None)
        if bind_fn is None:
            continue
        try:
            result = bind_fn(loop_ctx)
            if inspect.isawaitable(result):
                await result
        except (TypeError, ValueError):
            continue


async def _intercept_tool_calls_async(
    calls: list[ToolCall],
    hooks: list[Any],
    state: AgentState,
    session_log: SessionLog,
    context: Any,
    step_num: int,
) -> _InterceptResult:
    """Awaitable version of :func:`_intercept_tool_calls`."""
    from looplet.turn_engine import _intercept_tool_calls_gen, run_async_gen

    return await run_async_gen(
        _intercept_tool_calls_gen(calls, hooks, state, session_log, context, step_num)
    )


async def _run_post_dispatch_hooks_async(
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
    """Awaitable version of :func:`_run_post_dispatch_hooks`."""
    from looplet.turn_engine import _run_post_dispatch_hooks_gen, run_async_gen

    return await run_async_gen(
        _run_post_dispatch_hooks_gen(
            tool_call,
            tool_result,
            hooks,
            state,
            session_log,
            context,
            step_num,
            emit_lifecycle=emit_lifecycle,
        )
    )


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
    from looplet.turn_engine import _emit_hook_decision_gen, run_sync_gen

    run_sync_gen(
        _emit_hook_decision_gen(
            hooks,
            decision=decision,
            hook_slot=hook_slot,
            hook_name=hook_name,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            extra=extra,
        )
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

_CHECK_DONE_ACCEPTS_TOOL_CALL: WeakKeyDictionary[Any, bool] = WeakKeyDictionary()
"""Weak cache mapping the unbound ``check_done`` function to whether it
accepts a ``tool_call`` keyword argument.

Bound methods are ephemeral in CPython - ``obj.check_done`` creates a
fresh bound-method object on each access, so caching by
``id(bound_method)`` is unsound: bound methods get garbage-collected
and their ids get reused for unrelated methods on other classes,
poisoning the cache. Integer ids of unbound functions have the same
problem when local hook classes are collected. Weak keys preserve the
callable's identity without retaining short-lived hook classes.
"""


def _cache_key(method: Any) -> Any:
    """Return the underlying callable for a possibly bound method."""
    return getattr(method, "__func__", method)


def _accepts_tool_call_kwarg(method: Any) -> bool:
    key = _cache_key(method)
    cacheable = inspect.isfunction(key)
    cached = _CHECK_DONE_ACCEPTS_TOOL_CALL.get(key) if cacheable else None
    if cached is not None:
        return cached
    accepts = _hook_accepts_keyword(method, "tool_call")
    if cacheable:
        _CHECK_DONE_ACCEPTS_TOOL_CALL[key] = accepts
    return accepts


def _notify_post_step(hooks: list[Any], state: Any, session_log: Any, step_num: int) -> None:
    """Notify step observers for every recorded step kind."""
    for hook in hooks:
        method = getattr(hook, "post_step", None)
        if method is not None:
            method(state, session_log, step_num)


async def _notify_post_step_async(
    hooks: list[Any], state: Any, session_log: Any, step_num: int
) -> None:
    """Await the same step observer contract in the async driver."""
    for hook in hooks:
        method = getattr(hook, "post_step", None)
        if method is not None:
            result = method(state, session_log, step_num)
            if inspect.isawaitable(result):
                await result


def _post_step_stop_reason(
    hooks: list[Any],
    *,
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
    new_entities: int,
    requested_reason: str | None,
) -> str | None:
    """Apply post-step stopping to ordinary, rejected, and synthetic turns."""
    if requested_reason is not None:
        return requested_reason
    for hook in hooks:
        if hasattr(hook, "should_stop"):
            decision = normalize_hook_return(
                _invoke_hook(hook, "should_stop", state, step_num, new_entities),
                slot="should_stop",
            )
            if decision is not None:
                _emit_hook_decision_event(
                    hooks,
                    decision=decision,
                    hook_slot="should_stop",
                    hook_name=type(hook).__name__,
                    step_num=step_num,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
                if decision.is_stop():
                    return decision.stop or "hook_stop"
    return None


async def _post_step_stop_reason_async(
    hooks: list[Any],
    *,
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
    new_entities: int,
    requested_reason: str | None,
) -> str | None:
    """Await hook invocations with the same post-step stopping policy."""
    if requested_reason is not None:
        return requested_reason
    for hook in hooks:
        if hasattr(hook, "should_stop"):
            raw = await _invoke_hook_async(hook, "should_stop", state, step_num, new_entities)
            decision = normalize_hook_return(raw, slot="should_stop")
            if decision is not None:
                await _emit_hook_decision_event_async(
                    hooks,
                    decision=decision,
                    hook_slot="should_stop",
                    hook_name=type(hook).__name__,
                    step_num=step_num,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
                if decision.is_stop():
                    return decision.stop or "hook_stop"
    return None


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
    return _invoke_hook(
        hook,
        "check_done",
        state,
        session_log,
        context,
        step_num,
        _optional_kwargs={"tool_call": tool_call},
    )


async def _call_check_done_async(
    hook: Any,
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
    tool_call: Any,
) -> Any:
    return await _invoke_hook_async(
        hook,
        "check_done",
        state,
        session_log,
        context,
        step_num,
        _optional_kwargs={"tool_call": tool_call},
    )


# ── Hook method names (for typo detection) ──────────────────────

_KNOWN_HOOK_METHODS = frozenset(
    {
        "pre_loop",
        "pre_prompt",
        "pre_dispatch",
        "post_dispatch",
        "post_step",
        "check_done",
        "check_permission",
        "should_stop",
        "should_compact",
        "build_briefing",
        "build_prompt",
        "on_loop_end",
        "on_event",
        "bind",
    }
)


def is_loop_hook(value: Any) -> bool:
    """Return whether an object provides any supported, callable hook slot."""
    return any(callable(getattr(value, name, None)) for name in _KNOWN_HOOK_METHODS)


def _validate_hooks(hooks: list[Any]) -> None:
    """Warn about empty hooks and likely misspellings of supported methods."""
    import inspect  # noqa: PLC0415
    import warnings  # noqa: PLC0415
    from difflib import get_close_matches  # noqa: PLC0415

    for hook in hooks:
        if not is_loop_hook(hook):
            warnings.warn(
                f"Hook {type(hook).__name__} has no recognized hook methods "
                f"({', '.join(sorted(_KNOWN_HOOK_METHODS)[:5])}, ...). "
                f"Did you misspell a method name?",
                UserWarning,
                stacklevel=3,
            )
            continue
        for name in dir(hook):
            if name.startswith("_") or name in _KNOWN_HOOK_METHODS:
                continue
            matches = get_close_matches(name, _KNOWN_HOOK_METHODS, n=1, cutoff=0.9)
            if not matches:
                continue
            method = inspect.getattr_static(hook, name, None)
            if isinstance(method, (classmethod, staticmethod)):
                method = method.__func__
            if callable(method):
                warnings.warn(
                    f"Hook {type(hook).__name__}.{name} is not a supported hook method. "
                    f"Did you mean {matches[0]}?",
                    UserWarning,
                    stacklevel=3,
                )


def _normalize_loop_inputs(
    task: Any,
    tools: BaseToolRegistry | Sequence[ToolSpec | Callable[..., Any]] | None,
    config: LoopConfig,
) -> tuple[Any, BaseToolRegistry]:
    if isinstance(task, str):
        task = {"description": task}
    if tools is None:
        raise ValueError("tools is required")
    if isinstance(tools, Sequence):
        if isinstance(tools, (str, bytes)):
            raise TypeError("tools must be a registry or a sequence of Python functions")
        if any(not isinstance(item, ToolSpec) and not callable(item) for item in tools):
            raise TypeError("tools must contain Python functions or ToolSpec objects")
        registry = tools_from(tuple(tools), validate=True)
        if not config.allow_text_completion and not set(
            config.resolve_terminal_outcomes()
        ).intersection(registry.tool_names):
            register_done_tool(registry, name=config.done_tool)
        tools = registry
    return task, tools


def _validate_loop_inputs(task: Any, tools: BaseToolRegistry, config: LoopConfig) -> None:
    """Reject impossible loop wiring before prompt assembly begins."""
    from looplet.assistant_turns import validate_assistant_config

    validate_assistant_config(config)
    if isinstance(config.max_steps, bool) or not isinstance(config.max_steps, int):
        raise ValueError("config.max_steps must be a positive integer")
    if config.max_steps <= 0:
        raise ValueError("config.max_steps must be a positive integer")
    if config.max_parallel_calls <= 0:
        raise ValueError("config.max_parallel_calls must be positive")
    terminal_names = config.resolve_terminal_outcomes()
    if not set(terminal_names).intersection(tools.tool_names):
        warnings.warn(
            "No configured terminal tool is registered "
            f"({', '.join(terminal_names)}). "
            "Register a matching tool before running the loop. "
            "For default completion, use tools_from(..., include_done=True). "
            "Ignore this warning only when hooks or host code intentionally stop the run.",
            UserWarning,
            stacklevel=3,
        )


def _set_context_overrides(config: LoopConfig) -> list[tuple[Any, Any]]:
    """Set per-run context budget overrides and return reset tokens."""
    from looplet.context_budget import (  # noqa: PLC0415
        _CONTEXT_INLINE_PER_STEP_CHARS_OVERRIDE,
        _CONTEXT_WINDOW_STEPS_OVERRIDE,
        _CONTEXT_WINDOW_TOTAL_CHARS_OVERRIDE,
    )

    overrides = [
        (variable, int(value))
        for variable, value in (
            (_CONTEXT_WINDOW_STEPS_OVERRIDE, config.context_window_steps),
            (_CONTEXT_INLINE_PER_STEP_CHARS_OVERRIDE, config.context_inline_per_step_chars),
            (_CONTEXT_WINDOW_TOTAL_CHARS_OVERRIDE, config.context_window_total_chars),
        )
        if value is not None
    ]
    return [(variable, variable.set(value)) for variable, value in overrides]


def _reset_context_overrides(tokens: list[tuple[Any, Any]]) -> None:
    """Restore per-run context budget overrides."""
    for var, token in tokens:
        try:
            var.reset(token)
        except (LookupError, ValueError):
            pass


def _mark_failed_state(state: Any) -> None:
    """Keep caller-owned state truthful when an unhandled run error escapes."""
    _mark_terminal_state(state, status="failed", reason="error")


def _mark_terminal_state(state: Any, *, status: str, reason: str) -> None:
    """Project exceptional or explicit-close termination onto public state."""
    if state is None:
        return
    for name, value in (
        ("run_status", status),
        ("run_phase", "terminal"),
        ("termination_reason", reason),
    ):
        try:
            setattr(state, name, value)
        except AttributeError:
            pass
    try:
        state._stop_reason = reason
    except AttributeError:
        pass
    metadata = getattr(state, "metadata", None)
    if isinstance(metadata, dict):
        metadata.update(
            {
                "run_status": status,
                "run_phase": "terminal",
                "termination_reason": reason,
            }
        )


def _render_projection(
    renderer: Callable[..., str],
    projection: ContextProjection,
) -> str:
    """Invoke a new projection renderer or a legacy keyword renderer."""
    return render_projection(renderer, projection)


# ── Composable Agent Loop ───────────────────────────────────────


# ── Extracted dispatch helpers ────────────────────────────────────
# These reduce composable_loop's nesting depth and make the heaviest
# phases independently readable + testable.

_TURN_METADATA_KEY = "looplet_turn"


def _call_reference(tool_call: ToolCall) -> dict[str, str]:
    return {"tool": tool_call.tool, "call_id": tool_call.call_id}


def _annotate_tool_turn(
    calls: list[ToolCall],
    *,
    turn_id: str,
    batch_size: int,
    budget_skipped: list[ToolCall],
) -> None:
    """Attach one compact, host-owned turn record to each parsed call."""

    for call_index, tool_call in enumerate(calls):
        turn = {
            "turn_id": turn_id,
            "call_index": call_index,
            "batch_size": batch_size,
        }
        if call_index == 0 and budget_skipped:
            turn["budget_skipped"] = [_call_reference(call) for call in budget_skipped]
        tool_call.metadata[_TURN_METADATA_KEY] = turn


def _step_metadata(
    tool_call: ToolCall,
    *,
    dispatch_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turn = tool_call.metadata.get(_TURN_METADATA_KEY)
    if not isinstance(turn, dict):
        return {}
    annotated = dict(turn)
    annotated["dispatch_mode"] = dispatch_mode
    if extra:
        annotated.update(extra)
    return {"turn": annotated}


@dataclass
class _InterceptResult:
    """Return value of _intercept_tool_calls - collected pre-dispatch outcomes."""

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
    """Run pre-dispatch hooks and permission checks on tool calls."""
    from looplet.turn_engine import _intercept_tool_calls_gen, run_sync_gen

    return run_sync_gen(
        _intercept_tool_calls_gen(calls, hooks, state, session_log, context, step_num)
    )


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
    """Run post_dispatch hooks + POST_TOOL_USE/FAILURE events."""
    from looplet.turn_engine import _run_post_dispatch_hooks_gen, run_sync_gen

    return run_sync_gen(
        _run_post_dispatch_hooks_gen(
            tool_call,
            tool_result,
            hooks,
            state,
            session_log,
            context,
            step_num,
            emit_lifecycle=emit_lifecycle,
        )
    )


# ── Composable Agent Loop ───────────────────────────────────────


class _LoopFinalizer:
    """Invoke initialized loop-end hooks once on every exit path."""

    def __init__(self) -> None:
        self.state: Any = None
        self._hooks: list[Any] = []
        self._all_hooks: list[Any] = []
        self._args: tuple[Any, ...] = ()
        self._stop_emitted: bool = False

    def bind(self, hooks: list[Any], state: Any, session_log: Any, context: Any, llm: Any) -> None:
        self.state = state
        self._all_hooks = list(hooks)
        self._hooks = list(hooks)
        self._args = (state, session_log, context, llm)

    def mark_stop_emitted(self) -> None:
        self._stop_emitted = True

    def _take_hooks(self) -> list[Any]:
        hooks, self._hooks = self._hooks, []
        return [hook for hook in hooks if hasattr(hook, "on_loop_end")]

    def _emit_missing_stop_sync(self) -> None:
        if self._stop_emitted or not self._all_hooks or not self._args:
            return
        self._stop_emitted = True
        from looplet.events import LifecycleEvent as _LE

        state, session_log, context, _llm = self._args
        reason = (
            getattr(state, "termination_reason", None)
            or getattr(state, "_stop_reason", None)
            or "error"
        )
        try:
            emit_event(
                self._all_hooks,
                _LE.STOP,
                state=state,
                session_log=session_log,
                context=context,
                termination_reason=reason,
            )
        except Exception:  # noqa: BLE001
            logger.warning("STOP event emission failed during cleanup", exc_info=True)

    async def _emit_missing_stop_async(self) -> None:
        if self._stop_emitted or not self._all_hooks or not self._args:
            return
        self._stop_emitted = True
        from looplet.events import LifecycleEvent as _LE

        state, session_log, context, _llm = self._args
        reason = (
            getattr(state, "termination_reason", None)
            or getattr(state, "_stop_reason", None)
            or "error"
        )
        try:
            await emit_event_async(
                self._all_hooks,
                _LE.STOP,
                state=state,
                session_log=session_log,
                context=context,
                termination_reason=reason,
            )
        except Exception:  # noqa: BLE001
            logger.warning("STOP event emission failed during async cleanup", exc_info=True)

    def finish(self, *, suppress_errors: bool = False) -> int:
        from looplet.assistant_turns import _AdmissionDenied

        self._emit_missing_stop_sync()
        calls = 0
        failure: BaseException | None = None
        for hook in self._take_hooks():
            try:
                extra = hook.on_loop_end(*self._args)
                if isinstance(extra, int):
                    calls += extra
            except _AdmissionDenied as exc:
                self.state.metadata.setdefault("post_run_model_denials", []).append(
                    exc.termination_reason
                )
            except BaseException as exc:
                failure = failure or exc
                if suppress_errors:
                    logger.warning("on_loop_end cleanup failed", exc_info=True)
        if failure is not None and not suppress_errors:
            raise failure
        return calls

    async def finish_async(self, *, suppress_errors: bool = False) -> int:

        from looplet.assistant_turns import _AdmissionDenied

        await self._emit_missing_stop_async()
        calls = 0
        failure: BaseException | None = None
        for hook in self._take_hooks():
            try:
                extra = await _invoke_hook_async(hook, "on_loop_end", *self._args)
                if isinstance(extra, int):
                    calls += extra
            except _AdmissionDenied as exc:
                self.state.metadata.setdefault("post_run_model_denials", []).append(
                    exc.termination_reason
                )
            except BaseException as exc:
                failure = failure or exc
                if suppress_errors:
                    logger.warning("on_loop_end cleanup failed", exc_info=True)
        if failure is not None and not suppress_errors:
            raise failure
        return calls


def _trace_steps(state: Any) -> Any:
    from looplet._storage import DiskBackedList

    return state.steps[-5:] if isinstance(state.steps, DiskBackedList) else state.steps


def composable_loop(
    llm: Any,
    task: Any = None,
    tools: BaseToolRegistry | Sequence[ToolSpec | Callable[..., Any]] | None = None,
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
    """Run the synchronous tool-calling loop and yield recorded steps.

    Args:
        llm: Backend implementing synchronous ``generate()``.
        task: A task string or a dict of task fields. Strings become
            ``{'description': task}``. Defaults to an empty dict.
        tools: Python functions, tool specs, or an explicit registry.
            Sequences get argument validation and a standard completion tool
            unless completion is already configured. Explicit registries stay unchanged.
        context: Optional application object passed unchanged to hook methods.
        hooks: Explicit policies composed with the configured harness defaults.
        config: ``LoopConfig`` with budgets, prompt builders, and runtime options.
            Defaults to a new ``LoopConfig``.
        state: Mutable ``AgentState`` for this run. If omitted, creates ``DefaultState``
            with the effective ``config.max_steps`` limit.
        session_log: Memory log for this run. Defaults to a new ``SessionLog``.
        stream: Optional event emitter for structured loop lifecycle events.
        conversation: Optional conversation that records prompts, responses, and tool messages.
        max_steps: Optional override for ``config.max_steps``.
            An omitted state also receives this limit. A supplied config is updated in place.
        system_prompt: Optional override for ``config.system_prompt``.
            A supplied config is updated in place; live host controls retain their identity.

    Yields:
        A ``Step`` for each recorded loop step, including failed tool calls.

    Returns:
        The trace built by ``config.build_trace``, available through ``StopIteration.value``.

    Consume the iterator to completion. If stopping early, call its ``close()`` method
    or use ``contextlib.closing``. Cleanup runs on exhaustion, closure, and exceptions.
    Early closure marks an unfinished run as stopped, not completed.
    """
    from looplet.harness import prepare_harness_run

    effective_config = _resolve_loop_config(
        config, max_steps=max_steps, system_prompt=system_prompt
    )
    task, tools = _normalize_loop_inputs(task, tools, effective_config)
    prepared = prepare_harness_run(
        effective_config, hooks or [], llm, tools, state, session_log, conversation
    )
    tokens: list[tuple[Any, Any]] = []
    finalizer = _LoopFinalizer()
    try:
        tokens = _set_context_overrides(effective_config)
        trace = yield from _composable_loop_impl(
            llm=prepared.llm,
            task=task,
            tools=tools,
            context=context,
            hooks=prepared.hooks,
            config=prepared.config,
            state=prepared.state,
            session_log=prepared.session_log,
            stream=stream,
            conversation=conversation,
            _finalizer=finalizer,
            _checkpoint_store=prepared.checkpoint_store,
            _initial_checkpoint=prepared.initial_checkpoint,
            _compact_service=prepared.compact_service,
        )
        return trace
    except GeneratorExit:
        if getattr(finalizer.state, "run_phase", None) != "terminal":
            _mark_terminal_state(finalizer.state, status="stopped", reason="stream_closed")
        raise
    except BaseException:
        _mark_failed_state(finalizer.state if finalizer.state is not None else prepared.state)
        raise
    finally:
        try:
            finalizer.finish(suppress_errors=True)
        finally:
            try:
                prepared.finish()
            finally:
                _reset_context_overrides(tokens)


def run_result(
    llm: Any,
    task: Any = None,
    tools: BaseToolRegistry | Sequence[ToolSpec | Callable[..., Any]] | None = None,
    *,
    config: LoopConfig | None = None,
    state: AgentState | None = None,
    max_steps: int | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> RunResult:
    """Consume and close the ordinary loop, returning its accepted output and status.

    Functions and task strings use the same defaults as :func:`composable_loop`.
    Omitted state is created for this invocation. Supplied state uses ordinary budget rules.
    Additional keyword arguments pass unchanged to :func:`composable_loop`.
    Host-supplied backends, tools, hooks, and resources are not closed.
    """
    config = _resolve_loop_config(config, max_steps=max_steps, system_prompt=system_prompt)
    if state is None:
        state = _resolve_loop_state(None, config)
    with closing(
        composable_loop(llm, task=task, tools=tools, config=config, state=state, **kwargs)
    ) as steps:
        for _step in steps:
            pass
    return RunResult.from_state(state)


@dataclass(frozen=True)
class _ToolTurnPlan:
    calls: list[ToolCall]
    terminal_index: int | None


def _prepare_tool_turn(
    tools: BaseToolRegistry,
    calls: list[ToolCall],
    config: LoopConfig,
    state: AgentState,
    *,
    turn_id: str,
) -> _ToolTurnPlan | None:
    remaining = max(0, int(getattr(state, "budget_remaining", 0)))
    if remaining <= 0:
        return None
    admitted = calls[:remaining]
    for call in admitted:
        tools.prepare_call(call)
    if len(calls) > 1:
        _annotate_tool_turn(
            admitted,
            turn_id=turn_id,
            batch_size=len(calls),
            budget_skipped=calls[remaining:],
        )
    outcomes = config.resolve_terminal_outcomes()
    terminal_index = next(
        (index for index, call in enumerate(admitted) if call.tool in outcomes), None
    )
    return _ToolTurnPlan(admitted, terminal_index)


def _completion_gate_warning(config: LoopConfig, call: ToolCall, warning: str | None) -> str | None:
    if warning is not None:
        return warning
    schema = config.resolve_terminal_outcomes().get(call.tool)
    if schema is not None:
        validation = _validate_args(schema, call.args)
        if not validation.valid:
            return f"Output schema validation failed: {'; '.join(validation.errors)}"
    return None


def _terminal_step_metadata(call: ToolCall, ignored_calls: list[ToolCall]) -> dict[str, Any]:
    return _step_metadata(
        call,
        dispatch_mode="serial",
        extra=(
            {"ignored_after_terminal": [_call_reference(ignored) for ignored in ignored_calls]}
            if ignored_calls
            else None
        ),
    )


def _rejected_terminal_step(
    call: ToolCall, step_number: int, warning: str, metadata: dict[str, Any]
) -> Step:
    metadata["terminal_accepted"] = False
    return Step(
        number=step_number,
        tool_call=call,
        tool_result=ToolResult(
            tool=call.tool,
            args_summary="rejected",
            data={"rejected": True, "reason": warning},
            error=warning,
        ),
        metadata=metadata,
    )


def _normalize_done_rejection(tool_result: ToolResult) -> str | None:
    """Normalize terminal tool execution or post-dispatch errors into rejection payloads."""
    if tool_result.error is None:
        return None
    data = tool_result.data if isinstance(tool_result.data, dict) else None
    reason = str(tool_result.error) if tool_result.error else "Terminal tool rejected completion"
    merged_data = dict(data) if isinstance(data, dict) else {}
    merged_data["rejected"] = True
    merged_data["reason"] = reason
    tool_result.data = merged_data
    tool_result.error = reason
    return reason


def _failed_model_step(step_num: int, result: LLMResult) -> Step:
    return Step(
        number=step_num,
        tool_call=ToolCall(tool="__llm_error__", reasoning="LLM call failed"),
        tool_result=ToolResult(
            tool="__llm_error__",
            args_summary="",
            data=None,
            error=(
                str(result.error)
                if result.error is not None
                else "LLM call failed after all retry attempts"
            ),
        ),
    )


def _composable_loop_impl(
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
    _finalizer: _LoopFinalizer | None = None,
    _checkpoint_store: Any = None,
    _initial_checkpoint: Any = None,
    _compact_service: Any = None,
) -> Generator[Step, None, Any]:
    """Domain-agnostic agent loop with composable hooks.

    Yields Steps, returns a trace object built by config.build_trace.

    Args:
        llm: LLM backend - must implement ``generate()`` (see :class:`LLMBackend`).
        task: The task description - dict, string, or any domain object.
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
        stream: Optional EventEmitter - when set, emits structured events for
            each loop lifecycle moment (start, step, LLM call, dispatch, end).
        conversation: Optional Conversation - when set, the loop auto-records
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
    if config is None or max_steps is not None or system_prompt is not None:
        config = _resolve_loop_config(config, max_steps=max_steps, system_prompt=system_prompt)
    if hooks is None:
        hooks = []
    _validate_loop_inputs(task, tools, config)

    # ── Input guards ────────────────────────────────────────────
    if not callable(getattr(llm, "generate", None)):
        raise TypeError(
            f"llm must implement generate() (got {type(llm).__name__}). "
            "Use OpenAIBackend(client, model=...) or AnthropicBackend(client, model=...)."
        )

    state = _resolve_loop_state(state, config)

    # Warn when a hook object has no recognized hook methods - likely a typo.
    _validate_hooks(hooks)

    if session_log is None:
        session_log = SessionLog()
    finalizer = _finalizer or _LoopFinalizer()
    finalizer.bind(hooks, state, session_log, context, llm)

    from looplet.streaming import StreamingToolExecutor
    from looplet.turn_engine import drive_turn_machine_sync, run_turn_machine

    return (
        yield from drive_turn_machine_sync(
            run_turn_machine(
                llm=llm,
                task=task,
                tools=tools,
                context=context,
                hooks=hooks,
                config=config,
                state=state,
                session_log=session_log,
                stream=stream,
                explicit_conversation=conversation,
                finalizer=finalizer,
                checkpoint_store=_checkpoint_store,
                initial_checkpoint=_initial_checkpoint,
                compact_service=_compact_service,
                make_tool_ctx=lambda tc, sn, eff_llm: _build_tool_ctx(
                    config,
                    tools=tools,
                    hooks=hooks,
                    tool_call=tc,
                    step_num=sn,
                    state=state,
                    session_log=session_log,
                    llm=eff_llm,
                ),
                make_stream_executor=lambda: StreamingToolExecutor(
                    tools, max_workers=max(1, config.max_parallel_calls)
                ),
                build_final_trace=True,
            )
        )
    )
