"""Sub-agent spawning — run focused sub-tasks with isolated context.

Provides run_sub_loop() which creates an isolated composable_loop() call
with its own state and session log. The parent agent gets back
a concise summary without the sub-agent's raw data polluting context.

Usage:
    from openharness.subagent import run_sub_loop

    result = run_sub_loop(
        llm=llm, task=task, tools=tools,
        max_steps=5, system_prompt="Focus on this...",
    )
    summary = result["summary"]  # concise finding for parent context
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_sub_loop(
    llm: Any,
    task: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    max_steps: int = 5,
    system_prompt: str = "",
    hooks: list[Any] | None = None,
    context: Any = None,
    state: Any = None,
    sub_tools: Any = None,
    build_summary: Callable[[Any, Any, list[dict]], dict[str, Any]] | None = None,
    state_mutating_tools: list[str] | None = None,
    conversation: Any | None = None,
    subagent_id: str | None = None,
) -> dict[str, Any]:
    """Run a sub-agent loop with isolated state.

    Args:
        llm: LLM backend satisfying the LLMBackend protocol.
        task: Task dict describing what the sub-agent should do.
        tools: Parent tool registry. Cloned (minus state-mutating tools) for sub-agent.
        max_steps: Maximum number of steps for the sub-agent.
        system_prompt: System prompt for the sub-agent LLM calls.
        hooks: Optional list of LoopHook instances.
        context: Domain-specific backend passed through to the loop.
        state: Optional custom state. If None, uses _MinimalState.
        sub_tools: Optional custom tool registry. If None, clones parent
            tools with state-mutating tools removed.
        build_summary: Optional callable(state, session_log, steps_dicts) -> dict.
            If None, builds a generic summary from session log entities.
        state_mutating_tools: Tool names to exclude when cloning parent tools.
            Defaults to ["done"]. Only used when sub_tools is None.

    Returns a dict with:
      - summary: one-line summary of what was found
      - entities: entities discovered
      - findings: list of findings from session log entries
      - highlights: list of notable items from session log entries
      - llm_calls: number of LLM calls used
      - steps: list of step dicts (step-by-step trace)
      (build_summary may add additional keys)
    """
    from openharness.loop import LoopConfig, composable_loop
    from openharness.session import SessionLog

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

    # Fire SUBAGENT_START on the parent's hooks so observers see the
    # spawn. Import lazily to avoid a circular import with loop.py.
    from openharness.events import LifecycleEvent as _LE  # noqa: PLC0415
    from openharness.loop import _emit_event  # noqa: PLC0415
    _emit_event(
        hooks or [], _LE.SUBAGENT_START,
        state=state, context=context, subagent_id=subagent_id,
    )

    config = LoopConfig(
        max_steps=max_steps,
        system_prompt=system_prompt,
    )

    gen = composable_loop(
        llm=llm,
        task=task,
        tools=sub_tools,
        context=context,
        hooks=hooks or [],
        config=config,
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
    # is already handled by _emit_event.
    _emit_event(
        hooks or [], _LE.SUBAGENT_STOP,
        state=state, context=context, subagent_id=subagent_id,
        extra={
            "llm_calls": result["llm_calls"],
            "step_count": len(steps),
            "entities": result.get("entities", []),
        },
    )
    result["subagent_id"] = subagent_id
    return result


class _MinimalState:
    """Minimal agent state for sub-loops.

    Provides the interface the composable_loop expects from state:
    budget_remaining, step_count, steps, queries_used,
    context_summary(), snapshot().
    """

    def __init__(self, task: dict[str, Any] | None = None, max_steps: int = 5, **kwargs: Any) -> None:
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
    """Clone a tool registry, excluding specified tool names."""
    from openharness.tools import BaseToolRegistry, ToolSpec

    sub = BaseToolRegistry()
    for name, spec in parent_tools._tools.items():
        if name in exclude:
            continue
        sub.register(ToolSpec(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            execute=spec.execute,
            concurrent_safe=spec.concurrent_safe,
            free=spec.free,
        ))
    return sub
