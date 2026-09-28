"""Sub-agent spawning - run focused sub-tasks with isolated context.

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
import threading
import time
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from looplet.types import CancelToken

__all__ = [
    "run_sub_loop",
    "clone_tools_excluding",
    "ChildRunPolicy",
    "SharedModelBudget",
    "ChildRunLimitExceeded",
]


logger = logging.getLogger(__name__)


class ChildRunLimitExceeded(RuntimeError):
    """A child-run deadline, cancellation, or model allowance was exhausted."""


@dataclass
class SharedModelBudget:
    """Atomic model-call allowance shared by host-owned parent and child backends."""

    remaining: int
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def charge(
        self, *, deadline: float | None = None, cancel_token: CancelToken | None = None
    ) -> None:
        if cancel_token is not None and cancel_token.is_cancelled:
            raise ChildRunLimitExceeded("child run cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise ChildRunLimitExceeded("child run deadline exceeded")
        with self._lock:
            if self.remaining <= 0:
                raise ChildRunLimitExceeded("model call budget exhausted")
            self.remaining -= 1

    def wrap(
        self,
        backend: Any,
        *,
        deadline: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Any:
        if isinstance(backend, _BudgetedBackend) and backend.budget is self:
            if backend.deadline is not None:
                deadline = (
                    min(backend.deadline, deadline) if deadline is not None else backend.deadline
                )
            if backend.cancel_token is not None and cancel_token is not None:
                if backend.cancel_token is not cancel_token:
                    raise ValueError("shared model budget cannot replace a cancellation token")
            cancel_token = cancel_token or backend.cancel_token
            backend = backend.backend
        return _BudgetedBackend(backend, self, deadline, cancel_token)


@dataclass(frozen=True)
class ChildRunPolicy:
    """Optional parent-owned limits for an in-process child loop."""

    parent_id: str
    allowed_tools: frozenset[str] | None = None
    max_steps: int | None = None
    model_budget: SharedModelBudget | None = None
    deadline: float | None = None
    cancel_token: CancelToken | None = None


class _BudgetedBackend:
    def __init__(
        self,
        backend: Any,
        budget: SharedModelBudget,
        deadline: float | None,
        cancel_token: CancelToken | None,
    ) -> None:
        self.backend = backend
        self.budget = budget
        self.deadline = deadline
        self.cancel_token = cancel_token

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.backend, name)
        if name not in ("generate", "generate_with_tools"):
            return attribute

        @wraps(attribute)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            self.budget.charge(deadline=self.deadline, cancel_token=self.cancel_token)
            return attribute(*args, **kwargs)

        return guarded


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
    policy: ChildRunPolicy | None = None,
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
            (no ``pre_loop`` / ``check_done`` from the sub-loop -
            those would conflate parent + sub state); only their
            event-driven ``on_event`` method is forwarded. Opt-in:
            callers building tool-as-subagent patterns pass
            ``parent_hooks=ctx.hooks`` (or pull from a parent context).
            Defaults to ``None`` - no forwarding, fully isolated.
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
        policy: Optional parent-owned limits. Without it, child behavior is unchanged.

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
    if policy is not None:
        if not policy.parent_id or (policy.max_steps is not None and policy.max_steps <= 0):
            raise ValueError("child policy requires a parent_id and positive max_steps")
        if policy.deadline is not None and time.monotonic() >= policy.deadline:
            raise ChildRunLimitExceeded("child run deadline exceeded")
        if policy.model_budget is not None and policy.model_budget.remaining <= 0:
            raise ChildRunLimitExceeded("model call budget exhausted")

    # Create minimal isolated state if not provided
    if state is None:
        state = _MinimalState(task=task, max_steps=max_steps)
    session_log = SessionLog()

    # Create isolated tool registry if not provided
    if sub_tools is None:
        exclude = state_mutating_tools or ["done"]
        sub_tools = clone_tools_excluding(tools, exclude)
    if policy is not None and policy.allowed_tools is not None:
        sub_tools = _restrict_tools(sub_tools, policy.allowed_tools)

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
    # (pre_loop, check_done, etc.) - those would conflate parent + sub
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
        sub_config = replace(config) if policy is not None else config
    else:
        sub_config = LoopConfig(
            max_steps=max_steps,
            system_prompt=system_prompt,
        )
    if policy is not None:
        if policy.max_steps is not None:
            sub_config.max_steps = min(sub_config.max_steps, policy.max_steps)
        if isinstance(state, _MinimalState):
            state.max_steps = min(state.max_steps, sub_config.max_steps)
        if policy.cancel_token is not None:
            sub_config.cancel_token = policy.cancel_token
        sub_config.initial_checkpoint = None
        if sub_config.checkpoint_dir is not None:
            sub_config.checkpoint_dir = str(
                Path(sub_config.checkpoint_dir) / f"child_{subagent_id}"
            )
        if policy.model_budget is not None:
            llm = policy.model_budget.wrap(
                llm, deadline=policy.deadline, cancel_token=sub_config.cancel_token
            )
        sub_hooks.append(_ChildBudgetStopHook(policy))

    steps: list[dict[str, Any]] = []
    trace: Any = None
    result: dict[str, Any] | None = None
    gen = None
    try:
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
        while True:
            try:
                step = next(gen)
            except StopIteration as finished:
                trace = finished.value
                break
            steps.append(step.to_dict())

        all_findings: list[str] = []
        all_highlights: list[str] = []
        if hasattr(session_log, "entries"):
            for entry in session_log.entries:
                if hasattr(entry, "findings"):
                    all_findings.extend(entry.findings or [])
                if hasattr(entry, "highlights"):
                    all_highlights.extend(entry.highlights or [])

        if build_summary is not None:
            result = build_summary(state, session_log, steps)
        else:
            entities = sorted(session_log.all_entities())
            summary = f"Entities: {', '.join(entities[:10])}" if entities else "No findings"
            result = {"summary": summary, "entities": entities}

        result["steps"] = steps
        result["llm_calls"] = trace.get("llm_calls", 0) if isinstance(trace, dict) else 0
        result["stop_reason"] = getattr(state, "_stop_reason", None) or "budget_exhausted"
        if policy is not None:
            result["parent_id"] = policy.parent_id
        result.setdefault("findings", all_findings)
        result.setdefault("highlights", all_highlights)
        result["subagent_id"] = subagent_id
        return result
    finally:
        if gen is not None:
            gen.close()
        emit_event(
            list(parent_hooks or []) + (hooks or []),
            _LE.SUBAGENT_STOP,
            state=state,
            context=context,
            subagent_id=subagent_id,
            termination_reason=result["stop_reason"] if result is not None else "error",
            extra={
                "llm_calls": result["llm_calls"] if result is not None else 0,
                "step_count": len(steps),
                "entities": result.get("entities", []) if result is not None else [],
                "parent_id": policy.parent_id if policy is not None else None,
            },
        )


class _ChildBudgetStopHook:
    def __init__(self, policy: ChildRunPolicy) -> None:
        self.policy = policy

    def should_stop(self, state: Any, step_num: int, new_entities: int) -> Any:
        from looplet.hook_decision import HookDecision  # noqa: PLC0415

        if self.policy.cancel_token is not None and self.policy.cancel_token.is_cancelled:
            return HookDecision(stop="cancelled")
        if self.policy.deadline is not None and time.monotonic() >= self.policy.deadline:
            return HookDecision(stop="deadline_exceeded")
        if self.policy.model_budget is not None and self.policy.model_budget.remaining <= 0:
            return HookDecision(stop="model_budget_exhausted")
        return None


def _restrict_tools(registry: Any, allowed: frozenset[str]) -> Any:
    from looplet.tools import BaseToolRegistry  # noqa: PLC0415

    if not isinstance(registry, BaseToolRegistry):
        raise TypeError("child tool attenuation requires a BaseToolRegistry")
    restricted = BaseToolRegistry()
    restricted.set_resources(registry._resources)
    for name, spec in registry._tools.items():
        if name in allowed:
            restricted.register(replace(spec))
    return restricted


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
        # here - it's about to be discarded by every other observer at
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
    correctness bugs - e.g. a caller passing
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
            "(available: %s) - nothing to exclude for those entries. Typo?",
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
                timeout_s=getattr(spec, "timeout_s", None),
                infer_optional_from_signature=getattr(spec, "infer_optional_from_signature", False),
                idempotency=getattr(spec, "idempotency", "unknown"),
                retryable=getattr(spec, "retryable", False),
                requires=list(getattr(spec, "requires", ())),
                tags=list(getattr(spec, "tags", ())),
                render=dict(getattr(spec, "render", {})),
                capabilities=list(getattr(spec, "capabilities", ())),
            )
        )
    return sub
