"""Composable agent loop - domain-agnostic hook-based architecture.

The loop handles orchestration: LLM call → parse → dispatch → continue/stop.
Domain-specific behavior is injected via hooks and LoopConfig callables.
"""

from __future__ import annotations

import inspect
import logging
import time
import warnings
from collections.abc import AsyncIterator, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field
from dataclasses import replace as _dc_replace
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
from looplet.checkpoint import _capture_checkpoint
from looplet.checkpoint import (
    resume_loop_state as _resume_loop_state,
)
from looplet.checkpoint import save_pending_dispatch as _save_pending_dispatch
from looplet.checkpoint import validate_checkpoint_identity as _validate_checkpoint_identity
from looplet.context_plan import ContextPlan, ContextSourceSelector, ScopedContextSource
from looplet.context_projection import ContextProjection
from looplet.history import HistoryRecorder
from looplet.hook_decision import (
    HookDecision,
    _hook_accepts_keyword,
    _invoke_hook,
    normalize_hook_return,
)
from looplet.native_tools import NativeToolPolicy
from looplet.parse import parse_multi_tool_calls, to_text
from looplet.prompt_preparation import PreparedPrompt, prepare_prompt, render_projection
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
from looplet.result_projection import project_tool_result
from looplet.scaffolding import (
    PARSE_RECOVERY_MAX,
    ContextBudgetSnapshot,
    ContextOverflowError,
    LLMResult,
    NativeToolStats,
    build_parse_recovery_prompt,
    llm_call_with_retry,
)
from looplet.scaffolding import (
    estimate_prompt_tokens as estimate_prompt_tokens,
)
from looplet.session import SessionLog
from looplet.tools import (
    BaseToolRegistry,
    ToolSpec,
    _summarize_args_dict,
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
    ``done_tool``. Cartridges that need multiple distinct outcomes
    (e.g. ``report`` for the normal path AND ``escalate`` for the
    alternative path) declare ``done_tools: [report, escalate]`` in
    ``config.yaml`` to let the loop terminate on either.

    The legacy ``done_tool`` is always treated as terminal too;
    ``done_tools`` is *additive*. Empty list (the default) preserves
    single-sentinel behaviour exactly.
    """

    done_tool_schemas: dict[str, OutputSchema] = field(default_factory=dict)
    """Per-sentinel ``OutputSchema`` map for plural-sentinel cartridges
    (cartridge-spec v2). Keyed by sentinel tool name.

    When a secondary sentinel from :attr:`done_tools` is invoked and an
    entry exists here, the loop validates the call's args against the
    matching schema with the same quality-gate behaviour as
    :attr:`output_schema` on the primary :attr:`done_tool`. Sentinels
    without an entry are unvalidated.

    Populated by the cartridge loader from each sentinel's
    ``tool.yaml: output_schema:`` block; can also be set directly on
    ``LoopConfig`` for hand-built loops.
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

    Mirrors Claude Code's per-session ``tool-results/`` directory.
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
            setattr(state, "_looplet_run_stats", {"llm_calls": 0})
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


def _record_run_stats(
    state: Any,
    *,
    llm_calls: int,
    duration_ms: float | None = None,
    config: LoopConfig | None = None,
) -> None:
    """Retain host-only measurements without changing model/tool metadata."""
    from looplet.assistant_turns import TURN_STATE_KEY

    metadata = getattr(state, "metadata", {})
    turns = metadata.get(TURN_STATE_KEY) if isinstance(metadata, dict) else None
    if isinstance(turns, dict) and type(turns.get("model_calls")) is int:
        llm_calls = (
            turns["model_calls"]
            if turns.get("max_model_calls") is not None
            else max(llm_calls, turns["model_calls"])
        )
    stats = getattr(state, "_looplet_run_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        try:
            setattr(state, "_looplet_run_stats", stats)
        except AttributeError:
            return
    stats["llm_calls"] = llm_calls
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
    _tool_llm = _scope_llm_for_tool(llm, tool_call) if llm is not None else None

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
        async_llm=_scope_llm_for_tool(async_llm, tool_call) if async_llm is not None else None,
        parent_budget_remaining=getattr(state, "budget_remaining", None),
        parent_hooks=tuple(_hooks),
        run_envelope=config.run_envelope,
        execution_policy=config.execution_policy,
        metadata=_metadata,
    )


def _scope_llm_for_tool(llm: Any, tool_call: ToolCall | None) -> Any:
    """Wrap a recording LLM so tool-internal calls get a scope tag.

    If ``llm`` is (or wraps) a ``_RecordingBase``, returns a thin proxy
    that sets ``scope`` on every captured :class:`LLMCall`.  Otherwise
    returns ``llm`` unchanged - no overhead for non-recording backends.
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
        return llm  # no recording layer - pass through unchanged

    return _ScopedLLMProxy(llm, recording, f"tool:{tool_name}")


class _ScopedLLMProxy:
    """Thin proxy that tags recorded LLM calls with a scope string.

    Delegates ``generate`` (and ``generate_with_tools`` when present)
    to the wrapped backend. Task-local recording scope separates
    concurrent tool calls from loop-level calls.

    Zero overhead when the backend isn't recording - the proxy is only
    created by :func:`_scope_llm_for_tool` when a recording layer is
    detected.
    """

    def __init__(self, backend: Any, recording: Any, scope: str) -> None:
        self._backend = backend
        self._recording = recording
        self._scope = scope
        if inspect.iscoroutinefunction(backend.generate):
            self.generate = self._async_method(backend.generate)
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = (
                self._async_method(backend.generate_with_tools)
                if inspect.iscoroutinefunction(backend.generate_with_tools)
                else self._generate_with_tools_impl
            )

    def _async_method(self, method: Any) -> Any:
        from functools import wraps

        @wraps(method)
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            with self._recording._scope_calls(self._scope):
                return await method(*args, **kwargs)

        return invoke

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> str:
        with self._recording._scope_calls(self._scope):
            return self._backend.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> Any:
        with self._recording._scope_calls(self._scope):
            return self._backend.generate_with_tools(
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )


def _event_hook_callbacks(hooks: list[Any], event: Any) -> Generator[tuple[Any, Any], None, None]:
    """Select ordered event subscribers without invoking or flattening effects."""
    for hook in hooks:
        callback = getattr(hook, "on_event", None)
        if callback is None:
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
    """Dispatch a :class:`LifecycleEvent` to every hook that opts in.

    Public API - safe to call from subagents, custom hooks, or external
    orchestrators that need to fire lifecycle events on a hook list.

    Hooks without ``on_event`` are silently skipped - this is the
    additive surface, nobody has to implement it. Returned
    :class:`HookDecision` objects are collected so the caller can act
    on ``block`` / ``stop`` / ``updated_*`` fields; ``None`` returns
    are filtered out.

    Observer errors are logged. The loop separately applies fail-closed
    handling when these hooks act as tool gates.
    """
    return list(_iter_event_decisions(hooks, event, **payload_kwargs))


def _iter_event_decisions(
    hooks: list[Any],
    event: Any,
    *,
    fail_closed: bool = False,
    **payload_kwargs: Any,
) -> Generator[Any, None, None]:
    """Yield each decision before invoking the next event subscriber."""
    from looplet.events import EventPayload  # noqa: PLC0415
    from looplet.events import LifecycleEvent as _LE
    from looplet.hook_decision import HookDecision  # noqa: PLC0415

    state = payload_kwargs.get("state")
    if state is not None:
        from looplet.context_projection import model_turn_context

        for name, value in model_turn_context(state).items():
            payload_kwargs.setdefault(name, value)
        payload_kwargs.setdefault("run_status", getattr(state, "run_status", None))
        payload_kwargs.setdefault("run_phase", getattr(state, "run_phase", None))
        envelope = getattr(state, "run_envelope", None)
        if envelope is not None:
            payload_kwargs.setdefault(
                "run_envelope",
                envelope.to_dict() if hasattr(envelope, "to_dict") else envelope,
            )
    payload = EventPayload(event=event, **payload_kwargs)
    for hook, fn in _event_hook_callbacks(hooks, event):
        try:
            result = fn(payload)
        except Exception:  # noqa: BLE001
            if fail_closed:
                raise
            logger.exception("on_event hook raised; continuing")
            continue
        if isinstance(result, HookDecision):
            decision = normalize_hook_return(result, slot=getattr(event, "value", str(event)))
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
            yield decision


async def emit_event_async(
    hooks: list[Any],
    event: Any,
    **payload_kwargs: Any,
) -> list[Any]:
    """Async counterpart to :func:`emit_event`.

    Hooks are awaited in registration order so decisions retain the same
    precedence and mutation semantics as the synchronous loop.
    """
    return [
        decision async for decision in _iter_event_decisions_async(hooks, event, **payload_kwargs)
    ]


async def _iter_event_decisions_async(
    hooks: list[Any],
    event: Any,
    *,
    fail_closed: bool = False,
    **payload_kwargs: Any,
) -> AsyncIterator[Any]:
    """Await and yield each decision before invoking the next subscriber."""
    import inspect as _inspect  # noqa: PLC0415

    from looplet.events import EventPayload  # noqa: PLC0415
    from looplet.events import LifecycleEvent as _LE
    from looplet.hook_decision import HookDecision  # noqa: PLC0415

    state = payload_kwargs.get("state")
    if state is not None:
        from looplet.context_projection import model_turn_context

        for name, value in model_turn_context(state).items():
            payload_kwargs.setdefault(name, value)
        payload_kwargs.setdefault("run_status", getattr(state, "run_status", None))
        payload_kwargs.setdefault("run_phase", getattr(state, "run_phase", None))
        envelope = getattr(state, "run_envelope", None)
        if envelope is not None:
            payload_kwargs.setdefault(
                "run_envelope",
                envelope.to_dict() if hasattr(envelope, "to_dict") else envelope,
            )
    payload = EventPayload(event=event, **payload_kwargs)
    for hook, fn in _event_hook_callbacks(hooks, event):
        try:
            result = fn(payload)
            if _inspect.isawaitable(result):
                result = await result
        except Exception:  # noqa: BLE001
            if fail_closed:
                raise
            logger.exception("async on_event hook raised; continuing")
            continue
        if isinstance(result, HookDecision):
            decision = normalize_hook_return(result, slot=getattr(event, "value", str(event)))
            if event != _LE.HOOK_DECISION and not result.is_noop():
                await _emit_hook_decision_event_async(
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
            yield decision


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
    if decision is None or decision.is_noop():
        return
    _record_policy_decision(state, decision)
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415

    event_extra = {"decision": asdict(decision)}
    if extra:
        event_extra.update(extra)
    await emit_event_async(
        hooks,
        _LE.HOOK_DECISION,
        step_num=step_num,
        state=state,
        session_log=session_log,
        context=context,
        hook_slot=hook_slot,
        hook_name=hook_name,
        policy_decision=getattr(decision, "policy_decision", None),
        extra=event_extra,
    )


async def _bind_loop_context_async(loop_ctx: LoopContext, hooks: list[Any]) -> None:
    """Bind hooks while awaiting async ``bind`` implementations."""
    import inspect as _inspect  # noqa: PLC0415

    for hook in hooks:
        bind_fn = getattr(hook, "bind", None)
        if bind_fn is None:
            continue
        try:
            result = bind_fn(loop_ctx)
            if _inspect.isawaitable(result):
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
    import inspect as _inspect  # noqa: PLC0415

    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415
    from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415
    from looplet.types import ErrorKind, ToolError  # noqa: PLC0415

    result = _InterceptResult()
    for tc_idx, tc in enumerate(calls):
        cur_step = step_num + tc_idx
        decisions = _iter_event_decisions_async(
            hooks,
            _LE.PRE_TOOL_USE,
            fail_closed=True,
            step_num=cur_step,
            state=state,
            session_log=session_log,
            context=context,
            tool_call=tc,
        )
        denied = False
        async for decision in decisions:
            if decision.updated_args is not None:
                tc.args = decision.updated_args
            if decision.permission == "deny":
                reason = decision.block or ""
                message = f"Permission denied for tool '{tc.tool}'"
                if reason:
                    message += f": {reason}"
                error = ToolError(kind=ErrorKind.PERMISSION_DENIED, message=message)
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=message,
                    error_detail=error,
                )
                denied = True
                break
            if decision.updated_result is not None:
                result.intercepted[tc_idx] = decision.updated_result
                break
            if decision.additional_context:
                result.extra_context.append(decision.additional_context)
        if denied:
            continue

        pre_dispatch_denied = False
        for hook in hooks:
            if tc_idx in result.intercepted:
                break
            method = getattr(hook, "pre_dispatch", None)
            if method is None:
                continue
            raw = _invoke_hook(hook, "pre_dispatch", state, session_log, tc, cur_step)
            if _inspect.isawaitable(raw):
                raw = await raw
            decision = normalize_hook_return(raw, slot="pre_dispatch")
            if decision is None:
                if raw is not None and isinstance(raw, ToolResult):
                    result.intercepted[tc_idx] = raw
                    break
                continue
            await _emit_hook_decision_event_async(
                hooks,
                decision=decision,
                hook_slot="pre_dispatch",
                hook_name=type(hook).__name__,
                step_num=cur_step,
                state=state,
                session_log=session_log,
                context=context,
            )
            if decision.updated_args is not None:
                tc.args = decision.updated_args
            if decision.permission == "deny":
                pre_dispatch_denied = True
                reason = decision.block or ""
                message = f"Permission denied for tool '{tc.tool}'"
                if reason:
                    message += f": {reason}"
                error = ToolError(kind=ErrorKind.PERMISSION_DENIED, message=message)
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=message,
                    error_detail=error,
                )
                break
            if decision.updated_result is not None:
                result.intercepted[tc_idx] = decision.updated_result
                break
            if decision.additional_context:
                result.extra_context.append(decision.additional_context)

        if pre_dispatch_denied:
            continue
        for hook in hooks:
            method = getattr(hook, "check_permission", None)
            if method is None:
                continue
            raw = _invoke_hook(hook, "check_permission", tc, state)
            if _inspect.isawaitable(raw):
                raw = await raw
            decision = normalize_hook_return(raw, slot="check_permission")
            if decision is not None:
                await _emit_hook_decision_event_async(
                    hooks,
                    decision=decision,
                    hook_slot="check_permission",
                    hook_name=type(hook).__name__,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
            if decision is not None and decision.permission == "deny":
                reason = decision.block or ""
                message = f"Permission denied for tool '{tc.tool}'"
                if reason:
                    message += f": {reason}"
                error = ToolError(kind=ErrorKind.PERMISSION_DENIED, message=message)
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=message,
                    error_detail=error,
                )
                break
    return result


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
    import inspect as _inspect  # noqa: PLC0415

    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415
    from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415

    outcome = _PostDispatchOutcome(tool_result=tool_result)
    for hook in hooks:
        method = getattr(hook, "post_dispatch", None)
        if method is None:
            continue
        raw = _invoke_hook(
            hook, "post_dispatch", state, session_log, tool_call, tool_result, step_num
        )
        if _inspect.isawaitable(raw):
            raw = await raw
        decision = normalize_hook_return(raw, slot="post_dispatch")
        if decision is None:
            continue
        await _emit_hook_decision_event_async(
            hooks,
            decision=decision,
            hook_slot="post_dispatch",
            hook_name=type(hook).__name__,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
        )
        if decision.updated_result is not None:
            outcome.tool_result = decision.updated_result
            tool_result = outcome.tool_result
        if decision.additional_context:
            outcome.extra_context.append(decision.additional_context)
        if decision.stop is not None:
            outcome.stop_reason = decision.stop
    if not emit_lifecycle:
        return outcome
    event = _LE.POST_TOOL_FAILURE if tool_result.error else _LE.POST_TOOL_USE
    decisions = await emit_event_async(
        hooks,
        event,
        step_num=step_num,
        state=state,
        session_log=session_log,
        context=context,
        tool_call=tool_call,
        tool_result=tool_result,
    )
    for decision in decisions:
        if decision.updated_result is not None:
            outcome.tool_result = decision.updated_result
        if decision.additional_context:
            outcome.extra_context.append(decision.additional_context)
        if decision.stop is not None:
            outcome.stop_reason = decision.stop
    return outcome


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

    _record_policy_decision(state, decision)

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
        policy_decision=getattr(decision, "policy_decision", None),
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
            raw = _invoke_hook(hook, "should_stop", state, step_num, new_entities)
            if inspect.isawaitable(raw):
                raw = await raw
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
        _pre_tool_decisions = _iter_event_decisions(
            hooks,
            _LE.PRE_TOOL_USE,
            fail_closed=True,
            step_num=cur_step,
            state=state,
            session_log=session_log,
            context=context,
            tool_call=tc,
        )
        _denied = False
        for _d in _pre_tool_decisions:
            if _d.updated_args is not None:
                tc.args = _d.updated_args
            if _d.permission == "deny":
                _prefix = f"Permission denied for tool '{tc.tool}'"
                _reason = _d.block or ""
                _msg = f"{_prefix}: {_reason}" if _reason else _prefix
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
                _denied = True
                break
            if _d.updated_result is not None:
                result.intercepted[tc_idx] = _d.updated_result
                break
            if _d.additional_context:
                result.extra_context.append(_d.additional_context)
        if _denied:
            continue

        # ── Per-method pre_dispatch hooks ───────────────────
        pre_dispatch_denied = False
        for hook in hooks:
            if tc_idx in result.intercepted:
                break
            if not hasattr(hook, "pre_dispatch"):
                continue
            cached = _invoke_hook(hook, "pre_dispatch", state, session_log, tc, cur_step)
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
                pre_dispatch_denied = True
                _prefix = f"Permission denied for tool '{tc.tool}'"
                _reason = _decision.block or ""
                _msg = f"{_prefix}: {_reason}" if _reason else _prefix
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
            if _decision.updated_result is not None:
                result.intercepted[tc_idx] = _decision.updated_result
                break
            if _decision.additional_context:
                result.extra_context.append(_decision.additional_context)

        if pre_dispatch_denied:
            continue

        # ── Per-method check_permission hooks ──────────────
        for hook in hooks:
            if not hasattr(hook, "check_permission"):
                continue
            _raw = _invoke_hook(hook, "check_permission", tc, state)
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
                # Always prefix the canonical phrase 'Permission denied for
                # tool X' so builders grepping for ``permission`` or ``denied``
                # find it. The hook's specific reason (rule.reason in the
                # PermissionEngine, or _decision.block elsewhere) is appended
                # as context, preserving the actionable detail.
                _prefix = f"Permission denied for tool '{tc.tool}'"
                _reason = _decision.block if _decision and _decision.block else ""
                _msg = f"{_prefix}: {_reason}" if _reason else _prefix
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
        text = _invoke_hook(
            hook, "post_dispatch", state, session_log, tool_call, tool_result, step_num
        )
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


class _LoopFinalizer:
    """Invoke initialized loop-end hooks once on every exit path."""

    def __init__(self) -> None:
        self.state: Any = None
        self._hooks: list[Any] = []
        self._args: tuple[Any, ...] = ()

    def bind(self, hooks: list[Any], state: Any, session_log: Any, context: Any, llm: Any) -> None:
        self.state = state
        self._hooks = list(hooks)
        self._args = (state, session_log, context, llm)

    def _take_hooks(self) -> list[Any]:
        hooks, self._hooks = self._hooks, []
        return [hook for hook in hooks if hasattr(hook, "on_loop_end")]

    def finish(self, *, suppress_errors: bool = False) -> int:
        from looplet.assistant_turns import _AdmissionDenied

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
        import inspect

        from looplet.assistant_turns import _AdmissionDenied

        calls = 0
        failure: BaseException | None = None
        for hook in self._take_hooks():
            try:
                extra = hook.on_loop_end(*self._args)
                if inspect.isawaitable(extra):
                    extra = await extra
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

    t0 = time.time()

    if session_log is None:
        session_log = SessionLog()
    finalizer = _finalizer or _LoopFinalizer()
    finalizer.bind(hooks, state, session_log, context, llm)

    # ── Conversation thread - always active (single source of truth) ──
    from looplet.conversation import Conversation as _Conversation  # noqa: PLC0415

    _conv = conversation if conversation is not None else _Conversation()

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
            selected = config.router.select(purpose="reasoning")  # pyright: ignore[reportOptionalMemberAccess]
            from looplet.provenance import _RecordingBase

            if isinstance(llm, _RecordingBase):
                selected = llm.for_backend(selected)
        else:
            selected = llm
        loop_ctx.llm = _turns.wrap(selected) if _turns is not None else selected
        return loop_ctx.llm

    _context_sources = (
        ContextSourceSelector(
            config.scoped_context_sources, budget_tokens=config.scoped_context_budget_tokens
        )
        if config.scoped_context_sources
        else None
    )

    _ckpt_store = _checkpoint_store
    saved_checkpoint = _initial_checkpoint
    _step_offset = 0
    if saved_checkpoint is not None:
        _validate_checkpoint_identity(
            saved_checkpoint,
            config.run_envelope.run_id if config.run_envelope is not None else None,
        )
        resumed = _resume_loop_state(saved_checkpoint, max_steps=config.max_steps)
        restore_backend = (
            config.router.select(purpose="reasoning") if config.router is not None else llm
        )
        restore_backend_state = getattr(restore_backend, "restore_checkpoint_state", None)
        if callable(restore_backend_state):
            from copy import deepcopy

            restore_backend_state(deepcopy((resumed.get("domain_state") or {}).get("llm", {})))
        if config.run_envelope is None and isinstance(resumed.get("run_envelope"), dict):
            config = _dc_replace(
                config,
                run_envelope=RunEnvelope.from_dict(resumed["run_envelope"]),
            )
        _step_offset = resumed.get("step_offset", 0)
        # Restore session log entries into session_log
        restored_log = resumed.get("session_log")
        if restored_log is not None:
            session_log.entries = restored_log.entries[:]
            session_log.current_theory = restored_log.current_theory
        restored_conv = resumed.get("conversation")
        if restored_conv is not None and conversation is None:
            _conv = restored_conv
        # Restore state counters (queries_used, budget_remaining) so
        # budget enforcement continues where the checkpoint left off.
        # Some state classes expose budget_remaining as a read-only property
        # derived from steps/max_steps; skip fields that can't be assigned.
        for _k, _v in (resumed.get("state_counters") or {}).items():
            try:
                setattr(state, _k, _v)
            except AttributeError:
                pass

        from looplet.assistant_turns import TURN_STATE_KEY, restore_assistant_checkpoint

        config = restore_assistant_checkpoint(state, config, saved_checkpoint)
        if TURN_STATE_KEY in saved_checkpoint.metadata:
            _step_offset = saved_checkpoint.step_number - state.step_count

    # ── Unified history recorder - single write path for step/turn events ──
    _history = HistoryRecorder(
        state=state,
        session_log=session_log,
        conversation=_conv,
        typed_turns=config.uses_typed_history,
        observe_tool_result=getattr(llm, "record_tool_receipt", None),
        project_tool_result=lambda result: project_tool_result(
            result,
            persist_dir=config.tool_result_persist_dir,
            persist_threshold=_get_persist_threshold(),
        ),
    )

    # Domain adapter seeding: when ``config.domain`` is set, fall back
    # to each adapter field only if the flat field is still ``None``.
    # Flat fields therefore override the adapter; adapter overrides
    # built-in defaults.
    build_briefing = config.resolve_callable("build_briefing", _default_build_briefing)
    _raw_extract_entities = config.resolve_callable("extract_entities", _default_extract_entities)
    # Adapt the extractor's call signature: callables that accept a
    # ``state`` kwarg (or take 2+ positional args) get the live state
    # passed; legacy 1-arg ``extract_entities(data)`` callables keep
    # working unchanged. This lets domain extractors that need to
    # update ``state.*`` fields ship as static functions referenced
    # from a workspace via ``${py:...}`` instead of method-local
    # closures.
    import inspect as _inspect  # noqa: PLC0415

    try:
        _ee_sig = _inspect.signature(_raw_extract_entities)
        _ee_params = _ee_sig.parameters
        _ee_takes_state = "state" in _ee_params or any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in _ee_params.values()
        )
    except (TypeError, ValueError):  # builtins / C-impls without signatures
        _ee_takes_state = False

    if _ee_takes_state:

        def extract_entities(data: Any) -> list[str]:
            return _raw_extract_entities(data, state=state)
    else:
        extract_entities = _raw_extract_entities
    build_prompt_fn = config.resolve_callable("build_prompt")
    checkpoint_state = config.resolve_callable("checkpoint_state")
    restore_checkpoint_state = config.resolve_callable("restore_checkpoint_state")

    # ── Loop state ──────────────────────────────────────────
    consecutive_parse_failures = 0
    native_policy = NativeToolPolicy(enabled=config.use_native_tools)
    quality_gate_message = ""
    post_dispatch_parts: list[str] = []
    llm_calls = 0
    done = False
    stop_reason = "budget_exhausted"  # tracks why the loop exited
    llm_failed = False
    # Recovery state - each strategy fires at most once
    recovery_state = {
        "budget_enforcement": False,
        "emergency_truncate": False,
        "result_clearing": False,
    }

    extract_step_metadata = config.resolve_callable(
        "extract_step_metadata", _default_extract_step_metadata
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
    try:
        setattr(state, "run_envelope", config.run_envelope)  # noqa: B010
    except AttributeError:
        pass

    # ── Bind LoopContext ──────────────────────────────────────
    # Hooks that need long-lived access to state/tools/conversation
    # outside their declared method signatures used to capture them
    # in closures, which broke ``to_config()`` workspace round-trip
    # (closures can't be serialised). ``LoopContext`` is the
    # principled replacement: a mutable handle the loop owns and
    # mutates as it progresses (step_num, conversation), so a hook
    # that binds once reads live values forever after.
    loop_ctx = LoopContext(
        state=state,
        session_log=session_log,
        conversation=_conv,
        tools=tools,
        config=config,
        resources=tools.resources,
        step_num=0,
        run_envelope=config.run_envelope,
    )
    _bind_loop_context(loop_ctx, hooks)
    _set_run_lifecycle(loop_ctx, status=RunStatus.RUNNING, phase=RunPhase.STARTING)
    loop_ctx.step_num = _step_offset
    if saved_checkpoint is not None and restore_checkpoint_state is not None:
        restore_checkpoint_state(loop_ctx, resumed.get("domain_state", {}))
    if saved_checkpoint is not None:
        tools.restore_results(resumed.get("tool_results_store") or {})

    from looplet.assistant_turns import _AssistantTurns

    if saved_checkpoint is not None and saved_checkpoint.is_terminal:
        from looplet.assistant_turns import TURN_STATE_KEY

        if TURN_STATE_KEY in saved_checkpoint.metadata:
            saved = saved_checkpoint
            if saved.run_status != "completed":
                raise ValueError("stopped assistant runs require an explicit new run identity")
            _set_run_lifecycle(
                loop_ctx,
                status=RunStatus.COMPLETED,
                phase=RunPhase.TERMINAL,
                termination_reason=saved.termination_reason,
                terminal_tool=saved.metadata.get("accepted_terminal_tool"),
            )
            finalizer.bind([], state, session_log, context, llm)
            return
    _turns = (
        _AssistantTurns(state, config)
        if config.max_model_calls is not None or config.uses_typed_history
        else None
    )
    if _turns is not None:
        checkpoint_state = _turns.wrap_checkpoint(checkpoint_state)
        _turns.checkpoint = lambda: _turns.save(_ckpt_store, loop_ctx, checkpoint_state)
        finalizer.bind(hooks, state, session_log, context, _turns.wrap(llm))
        loop_ctx.llm = _turns.wrap(llm)

    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            _invoke_hook(
                hook, "pre_loop", state, session_log, context, _optional_kwargs={"tools": tools}
            )

    # Fire SESSION_START - single-slot subscribers to lifecycle
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
    # Skip if a StreamingHook is in hooks - it already emits LoopStartEvent
    # from its pre_loop method and we don't want duplicates.
    from looplet.streaming import StreamingHook as _StreamingHook  # noqa: PLC0415

    _has_streaming_hook = any(isinstance(h, _StreamingHook) for h in hooks)
    if stream is not None and _LoopStartEvent is not None and not _has_streaming_hook:
        _task_id = task.get("id", "") if isinstance(task, dict) else str(task)[:80]
        stream.emit(_LoopStartEvent(task_summary=str(_task_id), max_steps=config.max_steps))

    def _save_checkpoint(step_number: int, *, status: str | None = None) -> None:
        if _ckpt_store is None:
            return
        metadata = {"task": str(task)}
        if status is not None:
            metadata["status"] = status
        loop_ctx.step_num = step_number
        checkpoint = _capture_checkpoint(
            loop_ctx,
            step_number=step_number,
            checkpoint_state=checkpoint_state,
            metadata={**metadata, **_policy_checkpoint_metadata(state, _ckpt_store)},
        )
        _ckpt_store.save(
            checkpoint,
            key=f"step_{step_number}" if status is None else f"step_{step_number}_{status}",
        )

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset
        # Mirror current step + conversation into the bound LoopContext
        # so any hook that captured ``self.ctx = ctx`` in ``bind()``
        # reads the live values (rather than a snapshot from start-of-loop).
        loop_ctx.step_num = step_num
        loop_ctx.conversation = _conv
        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break
        _set_run_lifecycle(loop_ctx, phase=RunPhase.PROMPTING)
        _hook_requested_stop = False  # reset per step; honored after dispatch

        # Clear per-step hook context so hooks start each step with
        # a clean slate.  Hooks write to state.step_context during a
        # step; other hooks read from it within the same step.  The
        # loop owns the lifecycle - cleared here, populated by hooks.
        try:
            setattr(state, "step_context", {})  # noqa: B010
        except AttributeError:
            pass

        # Cancellation check between turns - stop cleanly, no more LLM calls.
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        if _turns is not None:
            if _turns.stop_reason is not None:
                stop_reason = _turns.stop_reason
                break
            _turns.begin_turn()

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
        compact_service = (
            config.compact_service if config.compact_service is not None else _compact_service
        )
        if _want_compact and compact_service is not None:
            from looplet.compact import run_compact as _run_compact  # noqa: PLC0415

            _run_compact(
                compact_service,
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
                text = _invoke_hook(hook, "pre_prompt", state, session_log, context, step_num)
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
                            briefing_parts.append("(briefing truncated - token budget exceeded)")
                            break
                        _briefing_used += text_tokens
                    briefing_parts.append(text)

        if post_dispatch_parts:
            briefing_parts.append("\n".join(post_dispatch_parts))
            post_dispatch_parts = []

        # ── Build prompt ────────────────────────────────────
        context_history = state.context_summary()
        if quality_gate_message:
            if _turns is None or not _turns.enabled:
                context_history += "\n" + quality_gate_message
            quality_gate_message = ""

        _prepared = prepare_prompt(
            config=config,
            state=state,
            tools=tools,
            session_log=session_log,
            task=task,
            step_num=step_num,
            briefing="\n".join(briefing_parts),
            context_sources=_context_sources,
            context_history=context_history,
            hooks=hooks,
            builder=build_prompt_fn,
            conversation=_conv,
        )
        prompt = _prepared.prompt
        _tool_view = _prepared.tool_view
        _rendered_memory = _prepared.rendered_memory
        _scoped_plan = _prepared.scoped_context_plan
        loop_ctx.context_budget = _prepared.budget
        estimated_tokens = _prepared.budget.estimated_tokens
        preflight_too_long = _prepared.budget.pressure
        _state_metadata = getattr(state, "metadata", None)
        if loop_ctx.context_budget.pressure and isinstance(_state_metadata, dict):
            _state_metadata["context_budget"] = loop_ctx.context_budget.to_dict()

        # Emit StepStartEvent
        if stream is not None and _StepStartEvent is not None and _turns is None:
            stream.emit(_StepStartEvent(step_num=step_num))

        # Resolve effective LLM once per step - used by both the main call and
        # parse-recovery below, regardless of whether pre-flight fires.
        effective_llm = _get_llm()
        if (
            _turns is not None
            and config.uses_typed_history
            and config.render_messages_override is None
        ):
            effective_llm.prepare_conversation(
                _conv, _prepared.conversation_prompt, request_prompt=prompt
            )

        # Fire PRE_LLM_CALL - observers only; return decisions are
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
            context_budget=loop_ctx.context_budget.to_dict(),
        )
        for _d in _pre_llm_decisions:
            if _d.additional_context:
                post_dispatch_parts.append(_d.additional_context)
            if _d.stop is not None:
                stop_reason = _d.stop
                _hook_requested_stop = True

        if _turns is not None and _turns.persistent_admission and _hook_requested_stop:
            break

        if preflight_too_long:
            logger.warning(
                "Pre-flight block: prompt ~%d tokens exceeds safe limit - %s",
                estimated_tokens,
                "running recovery before LLM call"
                if config.reactive_recovery
                else "stopping before LLM call",
            )
            llm_result = LLMResult(
                None,
                ContextOverflowError(
                    f"prompt estimate {estimated_tokens} tokens exceeds safe context "
                    f"limit {config.context_window - 3_000}"
                ),
            )
        else:
            # ── LLM call with retry + reactive recovery ───────────
            # Emit LLMCallStartEvent
            if (
                stream is not None
                and _LLMCallStartEvent is not None
                and (_turns is None or not _has_streaming_hook)
            ):
                stream.emit(
                    _LLMCallStartEvent(
                        step_num=step_num,
                        model_turn_id=_turns.turn_id if _turns is not None else None,
                    )
                )
            # Tracer: start span for LLM call
            _llm_span = None
            if config.tracer is not None:
                _llm_span = config.tracer.start_span(
                    f"llm.call.step_{step_num}",
                    attributes={"step": step_num},
                )
            _tool_schemas = native_policy.tool_schemas(effective_llm, tools)
            if _tool_schemas is not None:
                _tool_schemas = list(_tool_view.schemas)
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

                _schemas_text = _tool_view.catalog_text
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
            _set_run_lifecycle(loop_ctx, phase=RunPhase.LLM)
            llm_result = llm_call_with_retry(
                effective_llm,
                prompt,
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt,
                temperature=config.temperature,
                tools=_tool_schemas,
                native_policy=native_policy,
                cancel_token=config.cancel_token,
                max_continuations=config.max_turn_continuations,
                cache_breakpoints=_cache_bps,
                generate_kwargs=config.generate_kwargs or None,
                should_continue=lambda: not _deadline_expired(loop_ctx),
            )
            loop_ctx.native_tool_stats.record(llm_result)
            llm_calls += llm_result.calls + (
                0 if llm_result.native_attempted else llm_result.continuations
            )
            _record_run_stats(state, llm_calls=llm_calls, config=config)
            if _deadline_expired(loop_ctx) and _turns is None:
                stop_reason = "deadline_exceeded"
                done = True
                break
            _state_metadata = getattr(state, "metadata", None)
            if isinstance(_state_metadata, dict) and loop_ctx.native_tool_stats.has_activity:
                _state_metadata["native_tool_stats"] = loop_ctx.native_tool_stats.to_dict()
            _llm_dur_ms = (time.perf_counter() - _llm_t0) * 1000.0
            if _llm_span is not None and config.tracer is not None:
                config.tracer.end_span(_llm_span)
            # Emit LLMCallEndEvent
            if (
                stream is not None
                and _LLMCallEndEvent is not None
                and (_turns is None or not _has_streaming_hook)
            ):
                stream.emit(
                    _LLMCallEndEvent(
                        step_num=step_num,
                        model_turn_id=_turns.turn_id if _turns is not None else None,
                        response_length=len(llm_result.text or ""),
                        duration_ms=_llm_dur_ms,
                        usage=llm_result.usage
                        or (
                            (getattr(effective_llm, "last_usage", None) or {})
                            if llm_result.ok
                            else {}
                        ),
                    )
                )

        # Reactive recovery: if prompt-too-long, try chained strategies
        if not llm_result.ok and llm_result.is_prompt_too_long and config.reactive_recovery:

            def _prepare_recovery_prompt() -> PreparedPrompt:
                nonlocal prompt, _tool_view, _rendered_memory, _scoped_plan
                prepared = prepare_prompt(
                    config=config,
                    state=state,
                    tools=tools,
                    session_log=session_log,
                    task=task,
                    step_num=step_num,
                    briefing="\n".join(briefing_parts),
                    context_sources=_context_sources,
                    hooks=hooks,
                    builder=build_prompt_fn,
                    conversation=_conv,
                )
                prompt = prepared.prompt
                _tool_view = prepared.tool_view
                _rendered_memory = prepared.rendered_memory
                _scoped_plan = prepared.scoped_context_plan
                loop_ctx.context_budget = prepared.budget
                if (
                    _turns is not None
                    and config.uses_typed_history
                    and config.render_messages_override is None
                ):
                    effective_llm.prepare_conversation(
                        _conv, prepared.conversation_prompt, request_prompt=prompt
                    )
                return prepared

            llm_result = _recovery_chain(
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
                prepare_prompt_fn=_prepare_recovery_prompt,
                native_policy=native_policy,
                should_continue=lambda: not _deadline_expired(loop_ctx),
                compact_service=_compact_service,
            )
            llm_calls += recovery_state.get("_last_recovery_llm_calls", 0)
            _record_run_stats(state, llm_calls=llm_calls)
            loop_ctx.native_tool_stats.record(llm_result)

        raw_response = llm_result.text
        if _turns is not None:
            llm_calls = _turns.data["model_calls"]
            llm_result.model_calls = effective_llm.request_records
            measured_usage: dict[str, Any] = {}
            for record in llm_result.model_calls:
                for name, value in record["usage"].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        measured_usage[name] = measured_usage.get(name, 0) + value
            llm_result.usage = measured_usage or None
            _record_run_stats(state, llm_calls=llm_calls)

        # ── Record LLM turn in conversation thread via unified recorder ──
        _history.record_llm_turn(
            prompt=prompt,
            response=raw_response,
            metadata={
                "tool_view": {"names": list(_tool_view.names), "version": _tool_view.version},
                **({"model_turn_id": _turns.turn_id} if _turns is not None else {}),
                **(
                    {"scoped_context_plan": _scoped_plan.to_dict()}
                    if _scoped_plan is not None
                    else {}
                ),
            },
        )

        # Surface provider token usage / cost onto state so budget hooks
        # read it from the declared view instead of a backend side-channel
        # (keeps cost/budget hooks portable across runtimes - §9 T9).
        _step_usage = llm_result.usage or (
            getattr(effective_llm, "last_usage", None) if llm_result.ok else None
        )
        if _step_usage and _turns is None:
            try:
                _meta = getattr(state, "metadata", None)
                if isinstance(_meta, dict):
                    _meta["last_usage"] = dict(_step_usage)
                    _totals = _meta.get("usage_total")
                    if not isinstance(_totals, dict):
                        _totals = {}
                    for _k, _v in _step_usage.items():
                        if isinstance(_v, (int, float)):
                            _totals[_k] = _totals.get(_k, 0) + _v
                    _meta["usage_total"] = _totals
            except Exception:  # pragma: no cover - defensive
                logger.debug("usage surfacing failed", exc_info=True)

        # Fire POST_LLM_RESPONSE - hooks can observe raw text before
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
            usage=_step_usage,
            extra={
                "native_tool_stats": loop_ctx.native_tool_stats.to_dict(),
                "model_calls": llm_result.model_calls,
            },
        )
        for _d in _post_llm_decisions:
            if _d.stop is not None:
                stop_reason = _d.stop
                _hook_requested_stop = True
            if _d.additional_context:
                post_dispatch_parts.append(_d.additional_context)

        if _turns is not None and (
            (_hook_requested_stop and _turns.persistent_admission)
            or _deadline_expired(loop_ctx)
            or (config.cancel_token is not None and config.cancel_token.is_cancelled)
        ):
            if _deadline_expired(loop_ctx):
                stop_reason = "deadline_exceeded"
            elif config.cancel_token is not None and config.cancel_token.is_cancelled:
                stop_reason = "cancelled"
            break
        if _turns is not None and llm_result.failure_reason in {
            "model_call_budget",
            "context_limit",
            "cancelled",
            "deadline_exceeded",
            "unsupported_context",
        }:
            stop_reason = llm_result.failure_reason
            break

        if raw_response is None:
            # If cancellation caused the failure, exit cleanly - no error step.
            if config.cancel_token is not None and getattr(
                config.cancel_token, "is_cancelled", False
            ):
                stop_reason = "cancelled"
                break
            llm_failed = True
            logger.error("LLM call failed at step %d: %s", step_num, llm_result.error)
            error_call = ToolCall(tool="__llm_error__", reasoning="LLM call failed")
            error_result = ToolResult(
                tool="__llm_error__",
                args_summary="",
                data=None,
                error=(
                    str(llm_result.error)
                    if llm_result.error is not None
                    else "LLM call failed after all retry attempts"
                ),
            )
            step = Step(number=step_num, tool_call=error_call, tool_result=error_result)
            state.steps.append(step)
            yield step
            stop_reason = llm_result.failure_reason
            _history.record_step(
                step, theory="", entities=[], findings=[], highlights=[], recall_key=""
            )
            _notify_post_step(hooks, state, session_log, step_num)
            break

        # ── Parse response (native tool_use or JSON text) ────
        tool_calls = native_policy.parse_response(
            raw_response,
            allow_text_completion=config.allow_text_completion
            and config.assistant_turn_policy is None,
            done_tool=config.done_tool,
            stop_reason=llm_result.stop_reason,
        )
        assistant_text = native_policy.assistant_text(
            raw_response, stop_reason=llm_result.stop_reason
        )
        if not tool_calls and assistant_text is not None and _turns is not None and _turns.enabled:
            try:
                decision = _turns.propose(assistant_text, step_num)
                if _inspect.isawaitable(decision):
                    if _inspect.iscoroutine(decision):
                        decision.close()
                    raise TypeError("synchronous assistant policy cannot return an awaitable")
                decision = _turns.record_decision(decision)
            except Exception as error:
                _turns.metadata["assistant_policy_error"] = type(error).__name__
                stop_reason = "assistant_policy_error"
                llm_failed = True
                break
            _assistant_decisions = emit_event(
                hooks,
                _LE.ASSISTANT_TURN_DECISION,
                step_num=step_num,
                state=state,
                session_log=session_log,
                context=context,
                extra={
                    "model_turn_id": _turns.turn_id,
                    "action": decision.action,
                    "model_calls": _turns.data["model_calls"],
                },
            )
            if stream is not None and not _has_streaming_hook:
                from looplet.streaming import AssistantTurnDecisionEvent

                stream.emit(
                    AssistantTurnDecisionEvent(
                        step_num=step_num,
                        model_turn_id=_turns.turn_id,
                        action=decision.action,
                        model_calls=_turns.data["model_calls"],
                    )
                )
            _history.record_host_feedback(decision.feedback, model_turn_id=_turns.turn_id)
            if _turns.checkpoint is not None:
                _turns.checkpoint()
            consecutive_parse_failures = 0
            interruption = _turns.interruption_reason(_assistant_decisions)
            if interruption is not None:
                stop_reason = interruption
                break
            if decision.action == "stop":
                stop_reason = "assistant_stop"
                break
            if decision.action == "continue":
                reason = _post_step_stop_reason(
                    hooks,
                    state=state,
                    session_log=session_log,
                    context=context,
                    step_num=step_num,
                    new_entities=0,
                    requested_reason=None,
                )
                if reason is not None:
                    stop_reason = reason
                    break
                continue
            tool_calls = native_policy.parse_response(
                raw_response,
                allow_text_completion=True,
                done_tool=config.done_tool,
                stop_reason=llm_result.stop_reason,
            )
        if _turns is not None and any(
            tool_call.metadata.get("completion_source") != "assistant_message"
            for tool_call in tool_calls
        ):
            _turns.data["consecutive_assistant_turns"] = 0
        if not tool_calls:
            consecutive_parse_failures += 1
            # Consult recovery_registry if set - use returned action
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
                        error=f"Parse error - recovery aborted: {_recovery_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                    state.steps.append(step)
                    yield step
                    _history.record_step(
                        step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                    )
                    _notify_post_step(hooks, state, session_log, step_num)
                    reason = _post_step_stop_reason(
                        hooks,
                        state=state,
                        session_log=session_log,
                        context=context,
                        step_num=step_num,
                        new_entities=0,
                        requested_reason=stop_reason if _hook_requested_stop else None,
                    )
                    if reason is not None:
                        stop_reason = reason
                        done = True
                        break
                    continue
                if _recovery_action is not None and _recovery_action.message:
                    post_dispatch_parts.append(_recovery_action.message)
            if (
                not _hook_requested_stop
                and consecutive_parse_failures <= PARSE_RECOVERY_MAX
                and (not native_policy.enabled or native_policy.demoted)
            ):
                logger.warning(
                    "Parse failure %d/%d at step %d - attempting recovery",
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
                _record_run_stats(state, llm_calls=llm_calls)
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
                    error=(
                        "Native tool protocol error: expected structured tool_use block; "
                        f"received {(to_text(raw_response) or '')[:200]}"
                        if native_policy.enabled and not native_policy.demoted
                        else f"Could not parse JSON: {(to_text(raw_response) or '')[:200]}"
                    ),
                )
                step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                _history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
                _notify_post_step(hooks, state, session_log, step_num)
                reason = _post_step_stop_reason(
                    hooks,
                    state=state,
                    session_log=session_log,
                    context=context,
                    step_num=step_num,
                    new_entities=0,
                    requested_reason=stop_reason if _hook_requested_stop else None,
                )
                if reason is not None:
                    stop_reason = reason
                    done = True
                    break
                continue
        else:
            consecutive_parse_failures = 0

        # ── Dispatch tool calls ──────────────────────────────
        all_step_entities: list[str] = []
        if (
            _turns is not None
            and stream is not None
            and _StepStartEvent is not None
            and not _has_streaming_hook
        ):
            stream.emit(_StepStartEvent(step_num=step_num))

        # A provider response is one model turn even when it contains many
        # calls. Bound the number of calls consumed by the remaining step
        # budget before terminal handling or dispatch can overshoot it.
        available_steps = max(0, int(getattr(state, "budget_remaining", 0)))
        if available_steps <= 0:
            stop_reason = "budget_exhausted"
            done = True
            break
        original_tool_calls = tool_calls
        budget_skipped = original_tool_calls[available_steps:]
        tool_calls = original_tool_calls[:available_steps]
        for call in tool_calls:
            tools.prepare_call(call)
        if len(original_tool_calls) > 1:
            _annotate_tool_turn(
                tool_calls,
                turn_id=f"turn-{llm_calls}",
                batch_size=len(original_tool_calls),
                budget_skipped=budget_skipped,
            )

        # Effective set of terminal sentinels: legacy ``done_tool``
        # plus the host-side ``done_tools`` list. The first one the agent
        # invokes ends the loop; the legacy field stays the canonical
        # name used in error messages and pretty-printers.
        done_tool_name = config.done_tool
        terminal_set = config.resolve_terminal_outcomes()
        done_idx = None
        for i, tc in enumerate(tool_calls):
            if tc.tool in terminal_set:
                done_idx = i
                # Pin the canonical name to the actual tool that fired so
                # downstream code (eval ctx, trace builders) reports the
                # specific terminal sentinel, not just the legacy default.
                done_tool_name = tc.tool
                break

        # Dispatch non-done tools
        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls
        if regular_calls:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.DISPATCHING)

            def _dispatch_regular_calls():
                groups = (
                    [regular_calls]
                    if config.concurrent_dispatch
                    else [[tool_call] for tool_call in regular_calls]
                )
                offset = 0
                for group in groups:
                    group_step = step_num + offset
                    intercepted = _intercept_tool_calls(
                        group, hooks, state, session_log, context, group_step
                    )
                    post_dispatch_parts.extend(intercepted.extra_context)
                    dispatch_items = [
                        (index, call)
                        for index, call in enumerate(group)
                        if index not in intercepted.intercepted
                    ]
                    calls_to_dispatch = [call for _, call in dispatch_items]
                    if calls_to_dispatch:
                        if _ckpt_store is not None:
                            _save_pending_dispatch(
                                _ckpt_store,
                                calls=group,
                                dispatching=calls_to_dispatch,
                                step_number=group_step + len(group) - 1,
                                session_log=session_log,
                                conversation_data=_conv.serialize(),
                                config_snapshot={
                                    "max_steps": config.max_steps,
                                    "queries_used": getattr(state, "queries_used", 0),
                                    "budget_remaining": getattr(state, "budget_remaining", 0),
                                },
                                tool_results_store=tools.checkpoint_results(),
                                domain_state=(
                                    checkpoint_state(loop_ctx)
                                    if checkpoint_state is not None
                                    else {}
                                ),
                                run_envelope=(
                                    loop_ctx.run_envelope.to_dict()
                                    if loop_ctx.run_envelope is not None
                                    else None
                                ),
                                metadata={
                                    "task": str(task),
                                    **_policy_checkpoint_metadata(state, _ckpt_store),
                                },
                                run_status=loop_ctx.status.value,
                                run_phase=loop_ctx.phase.value,
                            )
                        tool_contexts = [
                            _build_tool_ctx(
                                config,
                                tools=tools,
                                hooks=hooks,
                                tool_call=call,
                                step_num=group_step + index,
                                state=state,
                                session_log=session_log,
                                llm=effective_llm,
                            )
                            for index, call in dispatch_items
                        ]
                        if config.concurrent_dispatch:
                            if type(tools).dispatch_batch is BaseToolRegistry.dispatch_batch:
                                results = tools.dispatch_batch(
                                    calls_to_dispatch,
                                    ctx=tool_contexts,
                                    max_workers=config.max_parallel_calls,
                                )
                            else:
                                results = tools.dispatch_batch(calls_to_dispatch, ctx=tool_contexts)
                        else:
                            results = [tools.dispatch(calls_to_dispatch[0], ctx=tool_contexts[0])]
                    else:
                        results = []
                    dispatch_results = iter(results)
                    for index, call in enumerate(group):
                        was_intercepted = index in intercepted.intercepted
                        result = (
                            intercepted.intercepted[index]
                            if was_intercepted
                            else next(dispatch_results)
                        )
                        yield offset + index, call, result, was_intercepted
                    offset += len(group)

            for tc_idx, tool_call, tool_result, was_intercepted in _dispatch_regular_calls():
                cur_step = step_num + tc_idx
                tool_spec = tools._tools.get(tool_call.tool)
                if not (tool_spec and tool_spec.free) and not was_intercepted:
                    state.queries_used += 1

                dispatch_mode = (
                    "parallel"
                    if config.concurrent_dispatch
                    and tool_spec is not None
                    and tool_spec.concurrent_safe
                    else "serial"
                )
                event_metadata = _step_metadata(
                    tool_call,
                    dispatch_mode=dispatch_mode,
                ).get("turn", {})

                # Emit ToolDispatchEvent
                if stream is not None and _ToolDispatchEvent is not None:
                    stream.emit(
                        _ToolDispatchEvent(
                            step_num=cur_step,
                            tool_name=tool_call.tool,
                            args_summary=_summarize_args_dict(tool_call.args),
                            metadata=event_metadata,
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

                step = Step(
                    number=cur_step,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    metadata=_step_metadata(
                        tool_call,
                        dispatch_mode=dispatch_mode,
                    ),
                )
                if (
                    tool_spec
                    and tool_spec.free
                    and not was_intercepted
                    and tool_result.error is None
                ):
                    step.metadata["budget_free"] = True
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
                            metadata=event_metadata,
                        )
                    )
                if stream is not None and _StepEndEvent is not None:
                    stream.emit(
                        _StepEndEvent(
                            step_num=cur_step,
                            classification="continue",
                            new_entities_count=0,
                            metadata=event_metadata,
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
                _notify_post_step(hooks, state, session_log, cur_step)

                # Save checkpoint after the session log and conversation include this step.
                _save_checkpoint(cur_step)

                if not config.concurrent_dispatch:
                    reason = None
                    if _deadline_expired(loop_ctx):
                        reason = "deadline_exceeded"
                    elif config.cancel_token is not None and config.cancel_token.is_cancelled:
                        reason = "cancelled"
                    elif tc_idx + 1 < len(regular_calls) or done_idx is not None:
                        reason = _post_step_stop_reason(
                            hooks,
                            state=state,
                            session_log=session_log,
                            context=context,
                            step_num=cur_step,
                            new_entities=len(step_entities),
                            requested_reason=stop_reason if _hook_requested_stop else None,
                        )
                    if reason is not None:
                        stop_reason = reason
                        done = True
                        break

        if not config.concurrent_dispatch and done:
            break
        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            done = True
            break

        # Handle done() if present
        if done_idx is not None:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.FINALIZING)
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    try:
                        w = _call_check_done(hook, state, session_log, context, cur_step, tool_call)
                        _decision = normalize_hook_return(w, slot="check_done")
                    except Exception:  # noqa: BLE001
                        # A broken quality gate must fail closed. The agent
                        # gets one actionable rejection and can retry after
                        # the host records the hook failure.
                        logger.exception(
                            "check_done hook %s raised; rejecting done()",
                            type(hook).__name__,
                        )
                        _decision = HookDecision(
                            block=(
                                f"Quality gate {type(hook).__name__} failed; "
                                "the final answer was not accepted."
                            ),
                            metadata={"hook_error": True},
                        )
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

            _schema_for_call = (
                config.resolve_terminal_outcomes().get(tool_call.tool)
                if gate_warning is None
                else None
            )
            if _schema_for_call is not None:
                validation = _validate_args(_schema_for_call, tool_call.args)
                if not validation.valid:
                    gate_warning = (
                        f"Output schema validation failed: {'; '.join(validation.errors)}"
                    )

            ignored_after_terminal = tool_calls[done_idx + 1 :]
            done_metadata = _step_metadata(
                tool_call,
                dispatch_mode="serial",
                extra=(
                    {
                        "ignored_after_terminal": [
                            _call_reference(call) for call in ignored_after_terminal
                        ]
                    }
                    if ignored_after_terminal
                    else None
                ),
            )

            # Emit ToolDispatchEvent for done
            if stream is not None and _ToolDispatchEvent is not None:
                stream.emit(
                    _ToolDispatchEvent(
                        step_num=cur_step,
                        tool_name=tool_call.tool,
                        args_summary=_summarize_args_dict(tool_call.args),
                        metadata=done_metadata.get("turn", {}),
                    )
                )

            if gate_warning is not None:
                logger.info("Quality gate rejected done() at step %d", step_num)
                quality_gate_message = gate_warning
                # Mirror the gate reason into ``error`` as well as
                # ``data['rejected']``: builders grepping for errors
                # otherwise miss schema-level rejections entirely (a
                # frequent friction surfaced during dogfooding).
                tool_result = ToolResult(
                    tool=done_tool_name,
                    args_summary="rejected",
                    data={"rejected": True, "reason": gate_warning},
                    error=gate_warning,
                )
                step = Step(
                    number=cur_step,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    metadata=done_metadata,
                )
                state.steps.append(step)
                yield step
                _history.record_step(
                    step,
                    theory="",
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key=tool_result.result_key or "",
                )
                _notify_post_step(hooks, state, session_log, cur_step)
                _save_checkpoint(cur_step)
            else:
                # done() dispatch intentionally bypasses permission checks - it's
                # a loop signal, not a side-effecting tool. Permission-gating a
                # termination signal would prevent the agent from ever stopping.
                _ctx = _build_tool_ctx(
                    config,
                    tools=tools,
                    hooks=hooks,
                    tool_call=tool_call,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    llm=effective_llm,
                )
                if (
                    _ckpt_store is not None
                    and tool_call.metadata.get("completion_source") != "assistant_message"
                ):
                    _save_pending_dispatch(
                        _ckpt_store,
                        calls=[tool_call],
                        dispatching=[tool_call],
                        step_number=cur_step,
                        session_log=session_log,
                        conversation_data=_conv.serialize(),
                        config_snapshot={
                            "max_steps": config.max_steps,
                            "queries_used": getattr(state, "queries_used", 0),
                            "budget_remaining": getattr(state, "budget_remaining", 0),
                        },
                        tool_results_store=tools.checkpoint_results(),
                        domain_state=(
                            checkpoint_state(loop_ctx) if checkpoint_state is not None else {}
                        ),
                        run_envelope=(
                            loop_ctx.run_envelope.to_dict()
                            if loop_ctx.run_envelope is not None
                            else None
                        ),
                        metadata={
                            "task": str(task),
                            **_policy_checkpoint_metadata(state, _ckpt_store),
                        },
                        run_status=loop_ctx.status.value,
                        run_phase=loop_ctx.phase.value,
                    )
                if tool_call.metadata.get("completion_source") == "assistant_message":
                    tool_result = ToolResult(
                        tool=tool_call.tool,
                        args_summary="assistant message",
                        data=dict(tool_call.args),
                    )
                else:
                    tool_result = tools.dispatch(tool_call, ctx=_ctx)
                # Run post_dispatch hooks for done() too - otherwise
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
                done_data = tool_result.data if isinstance(tool_result.data, dict) else {}
                rejected_done = tool_result.error is not None or (
                    done_data.get("rejected") is True or done_data.get("status") == "rejected"
                )
                if rejected_done:
                    reason = str(
                        done_data.get("reason")
                        or done_data.get("error")
                        or tool_result.error
                        or "Final answer was not accepted"
                    )
                    tool_result.data = {**done_data, "rejected": True, "reason": reason}
                    tool_result.error = reason
                    quality_gate_message = reason
                step = Step(
                    number=cur_step,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    metadata=done_metadata,
                )
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
                            metadata=done_metadata.get("turn", {}),
                        )
                    )
                if stream is not None and _StepEndEvent is not None:
                    stream.emit(
                        _StepEndEvent(
                            step_num=cur_step,
                            classification="rejected" if rejected_done else "done",
                            new_entities_count=0,
                            metadata=done_metadata.get("turn", {}),
                        )
                    )
                # Record done() to session_log + conversation
                _history.record_step(
                    step,
                    theory=tool_call.args.get("__theory__", ""),
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key="",
                )
                if rejected_done:
                    _notify_post_step(hooks, state, session_log, cur_step)
                    _save_checkpoint(cur_step)
                    continue
                _set_run_lifecycle(
                    loop_ctx,
                    status=RunStatus.COMPLETED,
                    phase=RunPhase.TERMINAL,
                    termination_reason="done",
                    terminal_tool=tool_call.tool,
                )
                _notify_post_step(hooks, state, session_log, cur_step)
                # Save checkpoint after done step (after yield, matching non-done pattern)
                _save_checkpoint(cur_step, status="done")
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

        reason = _post_step_stop_reason(
            hooks,
            state=state,
            session_log=session_log,
            context=context,
            step_num=step_num,
            new_entities=len(all_step_entities),
            requested_reason=stop_reason if _hook_requested_stop else None,
        )
        if reason is not None:
            stop_reason = reason
            done = True
            break

    # ── Post-loop hooks ─────────────────────────────────────
    # Stash stop_reason on state so hooks (e.g. StreamingHook) can read it
    final_status = (
        RunStatus.COMPLETED
        if stop_reason == "done"
        else RunStatus.CANCELLED
        if stop_reason in {"cancelled", "deadline_exceeded"}
        else RunStatus.FAILED
        if llm_failed or stop_reason in {"error", "llm_error"}
        else RunStatus.STOPPED
    )
    _set_run_lifecycle(
        loop_ctx,
        status=final_status,
        phase=RunPhase.TERMINAL,
        termination_reason=stop_reason,
    )
    if state is not None:
        try:
            state._stop_reason = stop_reason  # pyright: ignore[reportAttributeAccessIssue]
        except AttributeError:
            pass

    if _turns is not None:
        llm_calls = _turns.data["model_calls"]
        if _turns.checkpoint is not None:
            _turns.checkpoint()
    _record_run_stats(state, llm_calls=llm_calls)

    # Fire STOP - event-style hooks see termination reason before
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

    llm_calls += finalizer.finish()
    if _turns is not None:
        llm_calls = _turns.call_total(llm_calls)
        if _turns.checkpoint is not None:
            _turns.checkpoint()
    _record_run_stats(state, llm_calls=llm_calls)

    # Emit LoopEndEvent - skip if StreamingHook already emits it
    if (
        _turns is None
        and stream is not None
        and _LoopEndEvent is not None
        and not _has_streaming_hook
    ):
        stream.emit(
            _LoopEndEvent(
                total_steps=state.step_count,
                total_llm_calls=llm_calls,
                reason=stop_reason,
            )
        )

    # Build trace via injected callable
    elapsed = (time.time() - t0) * 1000
    _record_run_stats(state, llm_calls=llm_calls, duration_ms=elapsed)
    _build_trace_fn = config.resolve_callable("build_trace")
    if _build_trace_fn is not None:
        # Pass ``context`` through when the callable's signature
        # accepts it. Domain trace-builders that need the live loop
        # context (e.g. an exploration object) can declare
        # ``def build_trace(*, task, state, ..., context=None)`` and
        # ship as a static ``${py:...}`` ref instead of a closure.
        import inspect as _inspect  # noqa: PLC0415

        try:
            _bt_params = _inspect.signature(_build_trace_fn).parameters
            _bt_takes_context = "context" in _bt_params or any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in _bt_params.values()
            )
        except (TypeError, ValueError):
            _bt_takes_context = False
        _bt_kwargs: dict[str, Any] = dict(
            task=task,
            state=state,
            session_log=session_log,
            done=stop_reason == "done",
            llm=_turns.wrap(llm) if _turns is not None else llm,
            llm_calls=llm_calls,
            elapsed_ms=elapsed,
        )
        if _bt_takes_context:
            _bt_kwargs["context"] = context
        from looplet.assistant_turns import _AdmissionDenied

        try:
            trace = _build_trace_fn(**_bt_kwargs)
        except _AdmissionDenied as exc:
            getattr(state, "metadata", {}).setdefault("post_run_model_denials", []).append(
                exc.termination_reason
            )
            trace = {
                "task": task,
                "steps": [step.to_dict() for step in _trace_steps(state)],
                "steps_total": len(state.steps),
                "trace_dir": getattr(state, "metadata", {}).get("trace_dir"),
                "llm_calls": _turns.data["model_calls"] if _turns is not None else llm_calls,
                "total_time_ms": elapsed,
                "conversation": _conv,
            }
    else:
        trace = {
            "task": task,
            "steps": [s.to_dict() for s in _trace_steps(state)],
            "steps_total": len(state.steps),
            "trace_dir": getattr(state, "metadata", {}).get("trace_dir"),
            "llm_calls": llm_calls,
            "total_time_ms": elapsed,
            "conversation": _conv,
        }
    if _turns is not None:
        llm_calls = _turns.call_total(llm_calls)
        _record_run_stats(state, llm_calls=llm_calls, duration_ms=(time.time() - t0) * 1000)
        if _turns.checkpoint is not None:
            _turns.checkpoint()
        if stream is not None and _LoopEndEvent is not None and not _has_streaming_hook:
            stream.emit(
                _LoopEndEvent(
                    total_steps=state.step_count, total_llm_calls=llm_calls, reason=stop_reason
                )
            )
    return trace


# ── Reactive Recovery Chain ──────────────────────────────────────


def _recovery_chain(
    llm_result: LLMResult,
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
    prepare_prompt_fn: Callable[[], PreparedPrompt] | None = None,
    native_policy: NativeToolPolicy | None = None,
    should_continue: Callable[[], bool] | None = None,
    compact_service: Any = None,
) -> LLMResult:
    """Multi-strategy recovery chain for prompt-too-long errors.

    Tries strategies in order, each at most once:
      1. Aggressive budget enforcement (shrink all results to 2KB)
      2. Emergency session log compression (emergency_truncate - routed
         through :func:`looplet.compact.run_compact` so
         ``PRE_COMPACT`` / ``POST_COMPACT`` events fire and users can
         swap in a custom :class:`CompactService`)
      3. Clear all old result data entirely

    After each strategy, rebuilds the prompt and retries the LLM call.
    Returns the final result with observed request accounting preserved.
    """
    extra_llm_calls = 0
    hooks = hooks or []

    # Pick the compact service: user-supplied or the built-in default.
    from looplet.compact import TruncateCompact, run_compact  # noqa: PLC0415

    _compact_service = (
        config.compact_service if config.compact_service is not None else compact_service
    )
    if _compact_service is None:
        _compact_service = TruncateCompact()

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

        prepared = prepare_prompt_fn() if prepare_prompt_fn is not None else None
        if prepared is not None and prepared.budget.pressure:
            continue
        prompt = (
            prepared.prompt
            if prepared is not None
            else _rebuild_prompt(
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
        )
        tool_schemas = native_policy.tool_schemas(llm, tools) if native_policy is not None else None
        if tool_schemas is not None and prepared is not None:
            tool_schemas = list(prepared.tool_view.schemas)
        retry_result = llm_call_with_retry(
            llm,
            prompt,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            tools=tool_schemas,
            native_policy=native_policy,
            cancel_token=config.cancel_token,
            max_continuations=config.max_turn_continuations,
            generate_kwargs=config.generate_kwargs or None,
            should_continue=should_continue,
        )
        extra_llm_calls += retry_result.calls + (
            0 if retry_result.native_attempted else retry_result.continuations
        )
        retry_result._include_previous(llm_result)
        llm_result = retry_result

        if retry_result.ok:
            logger.info("Recovery strategy '%s' succeeded at step %d", name, step_num)
            recovery_state["_last_recovery_llm_calls"] = extra_llm_calls
            return retry_result

        if not retry_result.is_prompt_too_long:
            break

    logger.error("All recovery strategies exhausted at step %d", step_num)
    recovery_state["_last_recovery_llm_calls"] = extra_llm_calls
    return llm_result
