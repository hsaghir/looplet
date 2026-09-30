"""Shared gather, select, render, and budget stages for every prompt attempt."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable

from looplet.context_plan import ContextPlan, ContextSourceSelector
from looplet.context_projection import ContextProjection
from looplet.memory import render_memory
from looplet.prompts import build_prompt
from looplet.scaffolding import ContextBudgetSnapshot, estimate_prompt_tokens
from looplet.tools import ToolView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedPrompt:
    """One rendered attempt and the selections used to produce it."""

    prompt: str
    tool_view: ToolView
    rendered_memory: str
    scoped_context_plan: ContextPlan | None
    budget: ContextBudgetSnapshot


@dataclass
class _PromptInputs:
    kwargs: dict[str, Any]
    tool_view: ToolView
    rendered_memory: str
    scoped_context: str
    scoped_plan: ContextPlan | None

    def planner_kwargs(self) -> dict[str, Any]:
        return {key: value for key, value in self.kwargs.items() if key != "max_steps"}

    def default_prompt(self, **kwargs: Any) -> str:
        return build_prompt(
            **{**kwargs, "memory": self.rendered_memory},
            scoped_context=self.scoped_context,
        )


def _gather(
    *,
    config: Any,
    state: Any,
    tools: Any,
    session_log: Any,
    task: Any,
    step_num: int,
    briefing: str,
    context_sources: ContextSourceSelector | None,
    context_history: str | None = None,
) -> _PromptInputs:
    memory = render_memory(config.memory_sources, state) if config.memory_sources else ""
    view = tools.tool_view(
        config.tool_view_selector(step_num=step_num, state=state, tools=tools, task=task)
        if config.tool_view_selector is not None
        else None
    )
    scoped_context, scoped_plan = (
        context_sources.select(task=task, state=state, step_num=step_num)
        if context_sources is not None
        else ("", None)
    )
    snapshot = state.snapshot() if hasattr(state, "snapshot") else {}
    if context_history is None:
        context_history = state.context_summary() if hasattr(state, "context_summary") else ""
    return _PromptInputs(
        kwargs={
            "task": task,
            "tool_catalog": view.catalog_text,
            "state_summary": snapshot if isinstance(snapshot, dict) else {},
            "context_history": context_history,
            "step_number": step_num,
            "max_steps": config.max_steps,
            "session_log": str(session_log.render()) if hasattr(session_log, "render") else "",
            "briefing": briefing,
            "memory": "\n\n".join(part for part in (memory, scoped_context) if part),
        },
        tool_view=view,
        rendered_memory=memory,
        scoped_context=scoped_context,
        scoped_plan=scoped_plan,
    )


def _builders(inputs: _PromptInputs, hooks: list[Any], builder: Any) -> list[tuple[Any, bool]]:
    candidates = [(hook.build_prompt, True) for hook in hooks if hasattr(hook, "build_prompt")]
    candidates.append((builder if builder is not None else inputs.default_prompt, False))
    return candidates


def render_projection(renderer: Callable[..., str], projection: ContextProjection) -> str:
    """Support both detached-projection and legacy keyword renderers."""
    try:
        parameters = inspect.signature(renderer).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "projection" in parameters:
        return renderer(projection=projection)
    return renderer(
        messages=list(projection.messages),
        default_prompt=projection.default_prompt,
        step_num=projection.step_num,
    )


def _finish(
    inputs: _PromptInputs, prompt: str, config: Any, conversation: Any, plan: Any
) -> PreparedPrompt:
    kwargs = inputs.kwargs
    if config.render_messages_override is not None:
        projection = ContextProjection(
            messages=tuple(conversation.messages),
            default_prompt=prompt,
            step_num=kwargs["step_number"],
            task=kwargs["task"],
            tool_catalog=kwargs["tool_catalog"],
            state_summary=kwargs["state_summary"],
            context_history=kwargs["context_history"],
            session_log=kwargs["session_log"],
            briefing=kwargs["briefing"],
            memory=kwargs["memory"],
            context_plan=plan,
            scoped_context=inputs.scoped_context,
            scoped_context_plan=inputs.scoped_plan,
        )
        prompt = render_projection(config.render_messages_override, projection)
    tokens = estimate_prompt_tokens(prompt)
    return PreparedPrompt(
        prompt=prompt,
        tool_view=inputs.tool_view,
        rendered_memory=inputs.rendered_memory,
        scoped_context_plan=inputs.scoped_plan,
        budget=ContextBudgetSnapshot(
            prompt_chars=len(prompt),
            estimated_tokens=tokens,
            context_window_tokens=config.context_window,
            briefing_chars=len(kwargs["briefing"]),
            context_history_chars=len(kwargs["context_history"]),
            pressure=tokens > config.context_window - 3_000,
        ),
    )


def prepare_prompt(
    *, hooks: list[Any], builder: Any, conversation: Any, **kwargs: Any
) -> PreparedPrompt:
    """Prepare one synchronous attempt with first-hook-wins precedence."""
    inputs = _gather(**kwargs)
    config = kwargs["config"]
    plan = config.context_planner(**inputs.planner_kwargs()) if config.context_planner else None
    prompt: str | None = None
    for candidate, fallback in _builders(inputs, hooks, builder):
        try:
            result = candidate(**inputs.kwargs)
        except Exception:
            if not fallback:
                raise
            logger.exception("build_prompt hook raised; falling back")
            continue
        if result is not None:
            prompt = result
            break
    if prompt is None:
        raise TypeError("prompt builder must return a string")
    return _finish(inputs, prompt, config, conversation, plan)


async def prepare_prompt_async(
    *, hooks: list[Any], builder: Any, conversation: Any, **kwargs: Any
) -> PreparedPrompt:
    """Await async planners/builders without duplicating context policy."""
    inputs = _gather(**kwargs)
    config = kwargs["config"]
    plan = config.context_planner(**inputs.planner_kwargs()) if config.context_planner else None
    if inspect.isawaitable(plan):
        plan = await plan
    prompt: str | None = None
    for candidate, fallback in _builders(inputs, hooks, builder):
        try:
            result = candidate(**inputs.kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            if not fallback:
                raise
            logger.exception("build_prompt hook raised; falling back")
            continue
        if result is not None:
            prompt = str(result) if fallback else result
            break
    if prompt is None:
        raise TypeError("prompt builder must return a string")
    return _finish(inputs, prompt, config, conversation, plan)
