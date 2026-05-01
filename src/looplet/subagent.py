"""Sub-agent spawning — run focused sub-tasks with isolated context.

Provides run_sub_loop() which creates an isolated composable_loop() call
with its own state and session log. The parent agent gets back
a concise summary without the sub-agent's raw data polluting context.

Usage:
    from looplet.subagent import run_sub_loop

    result = run_sub_loop(
        llm=llm, task=task, tools=tools,
        max_steps=5, system_prompt="Focus on this...",
    )
    summary = result["summary"]  # concise finding for parent context
"""

from __future__ import annotations

import logging
from typing import Any, Callable

__all__ = [
    "run_sub_loop",
    "clone_tools_excluding",
]


logger = logging.getLogger(__name__)


def run_sub_loop(
    llm: Any,
    task: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    max_steps: int = 5,
    system_prompt: str = "",
    hooks: list[Any] | None = None,
    parent_hooks: list[Any] | None = None,
    context: Any = None,
    state: Any = None,
    sub_tools: Any = None,
    build_summary: Callable[[Any, Any, list[dict]], dict[str, Any]] | None = None,
    state_mutating_tools: list[str] | None = None,
    conversation: Any | None = None,
    subagent_id: str | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run a sub-agent loop with isolated state.

    Args:
        llm: LLM backend satisfying the LLMBackend protocol.
        task: Task dict describing what the sub-agent should do.
        tools: Parent tool registry. Cloned (minus state-mutating tools) for sub-agent.
        max_steps: Maximum number of steps for the sub-agent.
        system_prompt: System prompt for the sub-agent LLM calls.
        hooks: Optional list of LoopHook instances **for the sub-loop**.
        parent_hooks: Optional list of LoopHook instances from the
            parent loop. When supplied, the sub-loop also fires its
            lifecycle events (PRE_TOOL_USE, POST_TOOL_USE, etc.) on
            the parent's hooks so observability stacks on the parent
            (MetricsHook, StreamingHook, TrajectoryRecorder, …) see
            the sub-loop's per-step activity. The parent hooks are
            **not** invoked through their full ``LoopHook`` interface
            (no ``pre_loop`` / ``check_done`` from the sub-loop —
            those would conflate parent + sub state); only their
            event-driven ``on_event`` method is forwarded. Opt-in:
            callers building tool-as-subagent patterns pass
            ``parent_hooks=ctx.hooks`` (or pull from a parent context).
            Defaults to ``None`` — no forwarding, fully isolated.
        context: Domain-specific backend passed through to the loop.
        state: Optional custom state. If None, uses _MinimalState.
        sub_tools: Optional custom tool registry. If None, clones parent
            tools with state-mutating tools removed.
        build_summary: Optional callable(state, session_log, steps_dicts) -> dict.
            If None, builds a generic summary from session log entities.
        state_mutating_tools: Tool names to exclude when cloning parent tools.
            Defaults to ["done"]. Only used when sub_tools is None.
        config: Optional full LoopConfig. When supplied, its
            ``max_steps`` and ``system_prompt`` override the matching
            kwargs so that callers who already have a LoopConfig can
            pass it through uniformly with ``composable_loop``.

    Returns a dict with:
      - summary: one-line summary of what was found
      - entities: entities discovered
      - findings: list of findings from session log entries
      - highlights: list of notable items from session log entries
      - llm_calls: number of LLM calls used
      - steps: list of step dicts (step-by-step trace)
      (build_summary may add additional keys)
    """
    from looplet.loop import LoopConfig, composable_loop
    from looplet.session import SessionLog

    if task is None:
        task = {}

    # Create minimal isolated state if not provided
    if state is None:
        state = _MinimalState(task=task, max_steps=max_steps)
    session_log = SessionLog()

    # Create isolated tool registry if not provided
    if sub_tools is None:
        exclude = state_mutating_tools or ["done"]
        sub_tools = clone_tools_excluding(tools, exclude)

    # Fork conversation for sub-agent isolation (if provided)
    _sub_conv = None
    if conversation is not None and hasattr(conversation, "fork"):
        _sub_conv = conversation.fork()

    # Generate a stable id for lifecycle events so the caller can
    # correlate SUBAGENT_START / SUBAGENT_STOP payloads.
    if subagent_id is None:
        import uuid  # noqa: PLC0415

        subagent_id = uuid.uuid4().hex[:12]

    # Wrap each parent hook so its ``on_event`` receives every event
    # the sub-loop emits, tagged with this subagent_id so consumers
    # can route / nest. We do NOT forward the full LoopHook interface
    # (pre_loop, check_done, etc.) — those would conflate parent + sub
    # state. Only event-stream observers see the activity.
    sub_hooks: list[Any] = list(hooks or [])
    if parent_hooks:
        sub_hooks.append(_ParentHookForwarder(parent_hooks, subagent_id))

    # Fire SUBAGENT_START on the parent's hooks so observers see the
    # spawn. Import lazily to avoid a circular import with loop.py.
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415
    from looplet.loop import emit_event  # noqa: PLC0415

    emit_event(
        list(parent_hooks or []) + (hooks or []),
        _LE.SUBAGENT_START,
        state=state,
        context=context,
        subagent_id=subagent_id,
    )

    # Allow callers to pass a full LoopConfig for parity with
    # composable_loop. If provided, its values override the shorthand
    # kwargs (max_steps, system_prompt).
    if config is not None:
        sub_config = config
    else:
        sub_config = LoopConfig(
            max_steps=max_steps,
            system_prompt=system_prompt,
        )

    gen = composable_loop(
        llm=llm,
        task=task,
        tools=sub_tools,
        context=context,
        hooks=sub_hooks,
        config=sub_config,
        state=state,
        session_log=session_log,
        conversation=_sub_conv,
    )

    # Exhaust generator — collect step dicts
    steps: list[dict[str, Any]] = []
    trace: Any = None
    try:
        while True:
            step = next(gen)
            steps.append(step.to_dict())
    except StopIteration as e:
        trace = e.value

    # Aggregate findings and highlights from session log entries
    all_findings: list[str] = []
    all_highlights: list[str] = []
    if hasattr(session_log, "entries"):
        for entry in session_log.entries:
            if hasattr(entry, "findings"):
                all_findings.extend(entry.findings or [])
            if hasattr(entry, "highlights"):
                all_highlights.extend(entry.highlights or [])

    # Build summary via injected callable or generic default
    result: dict[str, Any]
    if build_summary is not None:
        result = build_summary(state, session_log, steps)
    else:
        entities = sorted(session_log.all_entities())
        summary = f"Entities: {', '.join(entities[:10])}" if entities else "No findings"
        result = {
            "summary": summary,
            "entities": entities,
        }

    result["steps"] = steps
    result["llm_calls"] = trace.get("llm_calls", 0) if isinstance(trace, dict) else 0
    result.setdefault("findings", all_findings)
    result.setdefault("highlights", all_highlights)

    # Fire SUBAGENT_STOP — observers see completion, final state, and
    # the llm-call cost via EventPayload.extra. Swallowing exceptions
    # is already handled by emit_event.
    emit_event(
        list(parent_hooks or []) + (hooks or []),
        _LE.SUBAGENT_STOP,
        state=state,
        context=context,
        subagent_id=subagent_id,
        extra={
            "llm_calls": result["llm_calls"],
            "step_count": len(steps),
            "entities": result.get("entities", []),
        },
    )
    result["subagent_id"] = subagent_id
    return result


class _ParentHookForwarder:
    """Forwards a sub-loop's lifecycle events to the parent's hooks.

    Implements only ``on_event`` (not the full ``LoopHook`` interface):
    the sub-loop's per-step events are surfaced to parent observability
    (MetricsHook, StreamingHook, TrajectoryRecorder, …) without letting
    the parent's flow-control methods (``check_done``, ``pre_prompt``,
    etc.) re-enter on sub state.

    The forwarded :class:`EventPayload` is augmented with
    ``subagent_id`` in its ``extra`` dict so consumers can distinguish
    nested activity from the parent's own.
    """

    __slots__ = ("_parent_hooks", "_subagent_id")

    def __init__(self, parent_hooks: list[Any], subagent_id: str) -> None:
        self._parent_hooks = list(parent_hooks)
        self._subagent_id = subagent_id

    def on_event(self, payload: Any) -> None:
        # Tag the payload's extra dict so parent observers can filter /
        # nest sub-activity. Mutating the live payload is acceptable
        # here — it's about to be discarded by every other observer at
        # the end of this dispatch.
        try:
            extra = getattr(payload, "extra", None)
            if isinstance(extra, dict):
                extra.setdefault("subagent_id", self._subagent_id)
        except Exception:  # noqa: BLE001
            pass
        for hook in self._parent_hooks:
            handler = getattr(hook, "on_event", None)
            if handler is None:
                continue
            try:
                handler(payload)
            except Exception:  # noqa: BLE001
                # Parent hook misbehaviour must never break the sub-loop.
                logger.warning(
                    "parent hook %r raised during sub-loop event forward "
                    "(subagent_id=%s); swallowing",
                    type(hook).__name__,
                    self._subagent_id,
                )


class _MinimalState:
    """Minimal agent state for sub-loops.

    Provides the interface the composable_loop expects from state:
    budget_remaining, step_count, steps, queries_used,
    context_summary(), snapshot().
    """

    def __init__(
        self, task: dict[str, Any] | None = None, max_steps: int = 5, **kwargs: Any
    ) -> None:
        self.task = task or {}
        self.max_steps = max_steps
        self.steps: list = []
        self.queries_used: int = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    def context_summary(self) -> str:
        if not self.steps:
            return "(no steps taken yet)"
        parts: list[str] = []
        for s in self.steps[-3:]:
            parts.append(s.summary())
        return "\n".join(parts)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "budget_remaining": self.budget_remaining,
        }


def clone_tools_excluding(parent_tools: Any, exclude: list[str]) -> Any:
    """Clone a tool registry, excluding specified tool names.

    Warns (via the module logger) when ``exclude`` contains names that
    are not present in ``parent_tools``. Typos here are silent
    correctness bugs — e.g. a caller passing
    ``exclude=["finish"]`` when the parent's done tool is named
    ``"finalize"`` would otherwise ship the state-mutating tool into
    the sub-agent without any signal.
    """
    from looplet.tools import BaseToolRegistry, ToolSpec

    parent_names = set(parent_tools._tools.keys())
    missing = [name for name in exclude if name not in parent_names]
    if missing:
        logger.warning(
            "clone_tools_excluding: names %s are not registered on the parent "
            "(available: %s) — nothing to exclude for those entries. Typo?",
            missing,
            sorted(parent_names),
        )

    sub = BaseToolRegistry()
    for name, spec in parent_tools._tools.items():
        if name in exclude:
            continue
        sub.register(
            ToolSpec(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                execute=spec.execute,
                concurrent_safe=spec.concurrent_safe,
                free=spec.free,
            )
        )
    return sub
