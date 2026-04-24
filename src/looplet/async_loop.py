"""Async composable loop — ``async for step in async_composable_loop(...)``.

Async mirror of :func:`looplet.loop.composable_loop` for use with
async LLM backends (:class:`AsyncOpenAIBackend`, etc.). All hooks,
tools, parsing, and state management remain synchronous — only
LLM calls are awaited.

Usage::

    from looplet.async_loop import async_composable_loop
    from looplet.backends import AsyncOpenAIBackend

    llm = AsyncOpenAIBackend(base_url="...", api_key="...", model="gpt-4o")

    async for step in async_composable_loop(
        llm=llm, tools=tools, state=state, config=config, task=task,
    ):
        print(step.pretty())

The function accepts the same arguments as ``composable_loop`` and
yields the same :class:`Step` objects. Hooks are still synchronous
Protocol methods — they run inline between awaited LLM calls.

When to use this instead of ``composable_loop``:
- Your LLM backend has ``async def generate()``
- You're inside an async context (FastAPI, aiohttp, Discord bot)
- You want to run multiple agent loops concurrently via asyncio.gather

When NOT to use this:
- Your LLM backend is synchronous — use ``composable_loop`` instead
- You need async hooks — not yet supported
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, AsyncGenerator

from looplet.loop import (
    LoopConfig,
    _build_tool_ctx,
    _intercept_tool_calls,
    _run_post_dispatch_hooks,
    emit_event,
)
from looplet.parse import parse_multi_tool_calls, parse_native_tool_use, to_text
from looplet.scaffolding import (
    LLMResult,
    _is_prompt_too_long,
    truncate_tool_result,
)
from looplet.session import SessionLog
from looplet.tools import BaseToolRegistry
from looplet.types import AgentState, DefaultState, Step, ToolCall, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "async_composable_loop",
    "async_llm_call",
]

# ── Retry constants (mirror scaffolding.py) ──────────────────────
MAX_LLM_RETRIES = 2
RETRY_BACKOFF_BASE = 1.0


# ── Async LLM call with retry ───────────────────────────────────


async def async_llm_call(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int = 2000,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_retries: int = MAX_LLM_RETRIES,
    tools: list[dict[str, Any]] | None = None,
    cancel_token: Any | None = None,
) -> LLMResult:
    """Async version of :func:`looplet.scaffolding.llm_call_with_retry`.

    Awaits ``llm.generate()`` or ``llm.generate_with_tools()`` when
    they are coroutines; calls them synchronously otherwise (supporting
    sync backends used from async context).
    """
    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
        return LLMResult(None, RuntimeError("cancelled before LLM call"))

    use_native = tools is not None and hasattr(llm, "generate_with_tools")
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
            return LLMResult(None, RuntimeError("cancelled during retry"))
        try:
            if use_native:
                result = llm.generate_with_tools(
                    prompt,
                    tools=tools,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                if inspect.isawaitable(result):
                    result = await result
                return LLMResult(result, stop_reason=getattr(llm, "last_stop_reason", None))

            result = llm.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            if inspect.isawaitable(result):
                result = await result
            return LLMResult(result, stop_reason=getattr(llm, "last_stop_reason", None))

        except Exception as e:
            last_error = e
            if _is_prompt_too_long(e):
                return LLMResult(None, e)
            if attempt < max_retries:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "Async LLM call attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)

    return LLMResult(None, last_error)


# ── Async composable loop ───────────────────────────────────────


async def async_composable_loop(
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
) -> AsyncGenerator[Step, None]:
    """Async version of :func:`looplet.loop.composable_loop`.

    Yields the same :class:`Step` objects. LLM calls are awaited;
    hooks, tools, and parsing remain synchronous.

    Usage::

        async for step in async_composable_loop(llm=llm, tools=tools, ...):
            print(step.pretty())
    """
    # ── Defaults ────────────────────────────────────────────────
    if task is None:
        task = {}
    if tools is None:
        raise ValueError("tools is required")
    if config is None:
        config = LoopConfig()
    if hooks is None:
        hooks = []

    if not callable(getattr(llm, "generate", None)):
        raise TypeError(
            f"llm must implement generate() (got {type(llm).__name__}). "
            "Use AsyncOpenAIBackend(base_url=...) or similar."
        )

    if state is None:
        state = DefaultState(max_steps=config.max_steps)
    if session_log is None:
        session_log = SessionLog()

    # Import lazily to avoid circular deps
    from looplet.conversation import Conversation  # noqa: PLC0415
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415
    from looplet.history import HistoryRecorder  # noqa: PLC0415
    from looplet.prompts import build_prompt as _build_prompt  # noqa: PLC0415

    _conv = conversation if conversation is not None else Conversation()

    # Stash task + conversation on state (same as sync loop)
    try:
        setattr(state, "task", task)  # noqa: B010
    except AttributeError:
        pass
    try:
        setattr(state, "conversation", _conv)  # noqa: B010
    except AttributeError:
        pass

    _history = HistoryRecorder(
        state=state,
        session_log=session_log,
        conversation=_conv,
    )

    # Domain callables
    _default_ee = lambda data: []  # noqa: E731
    extract_entities = (
        config.extract_entities
        or (config.domain.extract_entities if config.domain else None)
        or _default_ee
    )
    build_prompt_fn = config.build_prompt or (config.domain.build_prompt if config.domain else None)

    # ── Pre-loop hooks ──────────────────────────────────────────
    try:
        setattr(state, "step_context", {})  # noqa: B010
    except AttributeError:
        pass

    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            hook.pre_loop(state, session_log, context)

    emit_event(hooks, _LE.SESSION_START, state=state, session_log=session_log, context=context)

    # ── Render memory once (stable across steps) ────────────────
    _rendered_memory = ""
    if config.memory_sources:
        parts = []
        for src in config.memory_sources:
            if hasattr(src, "load"):
                text = src.load(state)
                if text:
                    parts.append(text)
        _rendered_memory = "\n".join(parts)

    # ── Main loop ───────────────────────────────────────────────
    done = False
    stop_reason = "budget_exhausted"
    llm_calls = 0
    consecutive_parse_failures = 0
    post_dispatch_parts: list[str] = []
    _step_offset = 0

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset

        # Clear step_context
        try:
            setattr(state, "step_context", {})  # noqa: B010
        except AttributeError:
            pass

        # Cancellation check
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        # ── Build prompt ────────────────────────────────────────
        briefing_parts: list[str] = list(post_dispatch_parts)
        post_dispatch_parts.clear()
        for hook in hooks:
            if hasattr(hook, "pre_prompt"):
                text = hook.pre_prompt(state, session_log, context, step_num)
                if isinstance(text, str) and text:
                    briefing_parts.append(text)

        _briefing = "\n".join(briefing_parts)
        _catalog = tools.tool_catalog_text()
        _state_summary = state.snapshot() if hasattr(state, "snapshot") else {}
        _log_text = session_log.render() if hasattr(session_log, "render") else ""
        _context_history = state.context_summary() if hasattr(state, "context_summary") else ""

        if build_prompt_fn is not None:
            prompt = build_prompt_fn(
                task=task,
                tool_catalog=_catalog,
                state_summary=_state_summary,
                context_history=_context_history,
                step_number=step_num,
                max_steps=config.max_steps,
                session_log=_log_text,
                briefing=_briefing,
                memory=_rendered_memory,
            )
        else:
            prompt = _build_prompt(
                task=task,
                tool_catalog=_catalog,
                state_summary=_state_summary,
                context_history=_context_history,
                step_number=step_num,
                max_steps=config.max_steps,
                session_log=_log_text,
                briefing=_briefing,
                memory=_rendered_memory,
            )

        # ── Native tool schemas ─────────────────────────────────
        _native_on = config.use_native_tools and hasattr(llm, "generate_with_tools")
        _tool_schemas = tools.tool_schemas() if _native_on else None

        # ── AWAIT: LLM call ─────────────────────────────────────
        _llm_t0 = time.perf_counter()
        llm_result = await async_llm_call(
            llm,
            prompt,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            tools=_tool_schemas,
            cancel_token=config.cancel_token,
        )
        _llm_dur_ms = (time.perf_counter() - _llm_t0) * 1000.0
        llm_calls += 1

        raw_response = llm_result.text
        _history.record_llm_turn(prompt=prompt, response=raw_response)

        if raw_response is None:
            if config.cancel_token is not None and getattr(
                config.cancel_token, "is_cancelled", False
            ):
                stop_reason = "cancelled"
                break
            error_call = ToolCall(tool="__llm_error__", reasoning="LLM call failed")
            error_result = ToolResult(
                tool="__llm_error__",
                args_summary="",
                data=None,
                error="LLM call failed after all retry attempts",
            )
            step = Step(number=step_num, tool_call=error_call, tool_result=error_result)
            state.steps.append(step)
            yield step
            break

        # ── Parse response ──────────────────────────────────────
        if config.use_native_tools and isinstance(raw_response, list):
            tool_calls = parse_native_tool_use(raw_response)
        else:
            tool_calls = parse_multi_tool_calls(raw_response)

        if not tool_calls:
            consecutive_parse_failures += 1
            if consecutive_parse_failures >= 3:
                break
            # Try recovery with a simpler prompt
            recovery_result = await async_llm_call(
                llm,
                f"Your previous response could not be parsed. "
                f"Respond with ONLY a JSON tool call.\n\n{prompt}",
                max_tokens=config.max_tokens,
                system_prompt=config.system_prompt,
                temperature=max(0.0, config.temperature - 0.1),
                tools=_tool_schemas,
                cancel_token=config.cancel_token,
            )
            llm_calls += 1
            if recovery_result.ok:
                if config.use_native_tools and isinstance(recovery_result.text, list):
                    tool_calls = parse_native_tool_use(recovery_result.text)
                else:
                    tool_calls = parse_multi_tool_calls(recovery_result.text)
            if not tool_calls:
                tc = ToolCall(tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200])
                tr = ToolResult(
                    tool="__parse_error__",
                    args_summary="",
                    data=None,
                    error=f"Could not parse: {(to_text(raw_response) or '')[:200]}",
                )
                step = Step(number=step_num, tool_call=tc, tool_result=tr)
                state.steps.append(step)
                yield step
                continue
        else:
            consecutive_parse_failures = 0

        # ── Dispatch tool calls ─────────────────────────────────
        done_tool_name = config.done_tool
        done_idx = None
        for i, tc in enumerate(tool_calls):
            if tc.tool == done_tool_name:
                done_idx = i
                break

        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls
        if regular_calls:
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

            calls_to_dispatch = [
                tc for i, tc in enumerate(regular_calls) if i not in intercepted_results
            ]

            if calls_to_dispatch:
                dispatch_results = []
                for _c in calls_to_dispatch:
                    _tool_ctx = _build_tool_ctx(
                        config,
                        hooks=hooks,
                        tool_call=_c,
                        step_num=step_num,
                        state=state,
                        session_log=session_log,
                        llm=llm,
                    )
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
                tool_spec = tools._tools.get(tool_call.tool)
                was_intercepted = tc_idx in intercepted_results
                if not (tool_spec and tool_spec.free) and not was_intercepted:
                    state.queries_used += 1

                tool_result.data = truncate_tool_result(tool_result.data)

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
                    done = True

                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step

                step_entities = extract_entities(tool_result.data)
                theory = tool_call.args.get("__theory__", "")
                _history.record_step(
                    step,
                    theory=theory,
                    entities=step_entities,
                    findings=[],
                    highlights=[],
                    recall_key="",
                )

        # Handle done()
        if done_idx is not None:
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415

                    w = hook.check_done(state, session_log, context, step_num)
                    _decision = normalize_hook_return(w, slot="check_done")
                    if _decision is not None and _decision.is_block():
                        gate_warning = _decision.block or "blocked by hook"
                        break

            if gate_warning is not None:
                tool_result = ToolResult(
                    tool=done_tool_name,
                    args_summary="rejected",
                    data={"rejected": True, "reason": gate_warning},
                )
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                _history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
            else:
                _ctx = _build_tool_ctx(
                    config,
                    hooks=hooks,
                    tool_call=tool_call,
                    step_num=step_num,
                    state=state,
                    session_log=session_log,
                    llm=llm,
                )
                tool_result = tools.dispatch(tool_call, ctx=_ctx)

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
                _history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
                done = True
                stop_reason = "done"

        if done:
            continue

        # Should-stop check
        for hook in hooks:
            if hasattr(hook, "should_stop"):
                from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415

                _raw = hook.should_stop(state, step_num, 0)
                _decision = normalize_hook_return(_raw, slot="should_stop")
                if _decision is not None and _decision.is_stop():
                    stop_reason = _decision.stop or "hook_requested_stop"
                    done = True
                    break

    # ── on_loop_end ─────────────────────────────────────────────
    for hook in hooks:
        if hasattr(hook, "on_loop_end"):
            hook.on_loop_end(state, session_log, context, llm)

    emit_event(
        hooks,
        _LE.STOP,
        state=state,
        session_log=session_log,
        context=context,
        termination_reason=stop_reason,
    )
