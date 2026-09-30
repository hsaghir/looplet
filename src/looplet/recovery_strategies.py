"""Shared recovery strategies for prompt-too-long errors.

Used by ``composable_loop`` when
reactive recovery is triggered. Each strategy mutates agent state to
reduce prompt size, then the caller rebuilds the prompt and retries.

Strategies (tried in order, each fires at most once):
  1. **Aggressive budget** - shrink all tool results to 2 KB each
  2. **Reactive compact** - deterministic session log compression
  3. **Clear old results** - drop all result data except last 2 steps
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
    state: Any,
    session_log: Any,
    context: Any,
    build_briefing: Any,
    build_prompt_fn: Any,
    task: dict,
    tools: Any,
    config: Any,
    step_num: int,
) -> str:
    """Compatibility helper using the same preparation stages as the loop."""
    from looplet.context_plan import ContextSourceSelector
    from looplet.conversation import Conversation
    from looplet.prompt_preparation import prepare_prompt

    sources = config.scoped_context_sources
    selector = (
        ContextSourceSelector(sources, budget_tokens=config.scoped_context_budget_tokens)
        if sources
        else None
    )
    return prepare_prompt(
        config=config,
        state=state,
        tools=tools,
        session_log=session_log,
        task=task,
        step_num=step_num,
        briefing=build_briefing(state, session_log, context) if build_briefing else "",
        context_sources=selector,
        hooks=[],
        builder=build_prompt_fn,
        conversation=Conversation(),
    ).prompt
