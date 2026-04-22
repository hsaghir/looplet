"""Shared recovery strategies for prompt-too-long errors.

Used by ``composable_loop`` when
reactive recovery is triggered. Each strategy mutates agent state to
reduce prompt size, then the caller rebuilds the prompt and retries.

Strategies (tried in order, each fires at most once):
  1. **Aggressive budget** — shrink all tool results to 2 KB each
  2. **Reactive compact** — deterministic session log compression
  3. **Clear old results** — drop all result data except last 2 steps
"""

from __future__ import annotations

import logging
from typing import Any

from looplet.scaffolding import emergency_truncate, trim_results

logger = logging.getLogger(__name__)


def recovery_aggressive_budget(state: Any, session_log: Any, llm: Any, step_num: int) -> int:
    """Strategy 1: Enforce aggressive per-result budget (2 KB each)."""
    if hasattr(state, "steps") and state.steps:
        trim_results(state.steps, per_result_chars=2000, aggregate_chars=20_000)
    return 0


def recovery_emergency_truncate(state: Any, session_log: Any, llm: Any, step_num: int) -> int:
    """Strategy 2: Emergency session log compression (deterministic)."""
    emergency_truncate(state, session_log, keep_recent=2)
    return 0


def recovery_clear_old_results(state: Any, session_log: Any, llm: Any, step_num: int) -> int:
    """Strategy 3: Clear all result data except last 2 steps."""
    if hasattr(state, "steps"):
        for step in state.steps[:-2]:
            step.tool_result.data = None
    return 0


def rebuild_prompt(
    state: Any, session_log: Any, context: Any,
    build_briefing: Any, build_prompt_fn: Any,
    task: dict, tools: Any, config: Any, step_num: int,
) -> str:
    """Rebuild prompt after a recovery strategy modified state."""
    context_history = state.context_summary()
    briefing = build_briefing(state, session_log, context) if build_briefing else ""

    # Render persistent memory — same as the main loop body.
    _memory_sources = getattr(config, "memory_sources", None)
    if _memory_sources:
        from looplet.memory import render_memory as _render_memory  # noqa: PLC0415
        _rendered_memory = _render_memory(_memory_sources, state)
    else:
        _rendered_memory = ""

    if build_prompt_fn is not None:
        return build_prompt_fn(
            task=task,
            tool_catalog=tools.tool_catalog_text(),
            state_summary=state.snapshot(),
            context_history=context_history,
            step_number=step_num,
            max_steps=config.max_steps,
            session_log=session_log.render(),
            briefing=briefing,
            memory=_rendered_memory,
        )
    # Fallback: use the same structured default prompt as the main loop.
    from looplet.prompts import build_prompt as _default_build_prompt  # noqa: PLC0415
    return _default_build_prompt(
        task=task,
        tool_catalog=tools.tool_catalog_text(),
        state_summary=state.snapshot(),
        context_history=context_history,
        step_number=step_num,
        max_steps=config.max_steps,
        session_log=session_log.render(),
        briefing=briefing,
        memory=_rendered_memory,
    )

