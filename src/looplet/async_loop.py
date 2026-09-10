"""Async composable loop - ``async for step in async_composable_loop(...)``.

Async mirror of :func:`looplet.loop.composable_loop` for use with
async LLM backends (:class:`AsyncOpenAIBackend`, etc.). All hooks,
tools, parsing, and state management remain synchronous - only
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
Protocol methods - they run inline between awaited LLM calls.

When to use this instead of ``composable_loop``:
- Your LLM backend has ``async def generate()``
- You're inside an async context (FastAPI, aiohttp, Discord bot)
- You want to run multiple agent loops concurrently via asyncio.gather

When NOT to use this:
- Your LLM backend is synchronous - use ``composable_loop`` instead
- You need async hooks - not yet supported
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import replace as _dc_replace
from typing import Any, AsyncGenerator

from looplet.checkpoint import Checkpoint as _Checkpoint
from looplet.checkpoint import FileCheckpointStore as _FileCheckpointStore
from looplet.checkpoint import resume_loop_state as _resume_loop_state
from looplet.context_projection import ContextProjection
from looplet.loop import (
    ContextBudgetSnapshot,
    LoopConfig,
    LoopContext,
    RunEnvelope,
    RunPhase,
    RunStatus,
    _bind_loop_context_async,
    _build_tool_ctx,
    _call_check_done,
    _deadline_expired,
    _emit_hook_decision_event_async,
    _intercept_tool_calls_async,
    _mark_failed_state,
    _policy_checkpoint_metadata,
    _render_projection,
    _reset_context_overrides,
    _run_post_dispatch_hooks_async,
    _set_context_overrides,
    _set_run_lifecycle,
    _validate_loop_inputs,
    emit_event_async,
)
from looplet.native_tools import NativeToolPolicy
from looplet.parse import parse_multi_tool_calls, to_text
from looplet.scaffolding import (
    PARSE_RECOVERY_MAX,
    LLMResult,
    NativeToolStats,
    _is_prompt_too_long,
    build_parse_recovery_prompt,
    estimate_prompt_tokens,
    truncate_tool_result,
)
from looplet.session import SessionLog
from looplet.tools import BaseToolRegistry, _summarize_args_dict
from looplet.types import AgentState, DefaultState, Step, ToolCall, ToolResult
from looplet.validation import validate_args as _validate_args

logger = logging.getLogger(__name__)

__all__ = [
    "async_composable_loop",
    "async_llm_call",
]

# ── Retry constants (mirror scaffolding.py) ──────────────────────
MAX_LLM_RETRIES = 2
RETRY_BACKOFF_BASE = 1.0


async def _maybe_await(value: Any) -> Any:
    """Resolve sync or async hook results in registration order."""
    if inspect.isawaitable(value):
        return await value
    return value


class _SyncBridgeLLM:
    """Wraps an async LLM backend so sync tools can call generate().

    When tools receive ``ctx.llm`` in the async loop, the underlying
    backend has ``async def generate()``. Tools are sync, so they
    can't ``await``. This bridge runs the coroutine on the current
    event loop via ``asyncio.get_event_loop().run_until_complete()``
    when called from a non-async context, or falls through if the
    backend is already sync.

    This is intentionally simple - it covers the common case of
    single-call tool-internal LLM use (summarize, classify). For
    complex async tool workflows, users should write async tools
    and await directly (future feature).
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        result = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        if inspect.isawaitable(result):
            # We're inside an event loop (async context). Use a thread
            # to run the coroutine without blocking the loop.
            import concurrent.futures  # noqa: PLC0415

            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, result)  # pyright: ignore[reportArgumentType]
                    return future.result(timeout=120)
            else:
                return loop.run_until_complete(result)
        return result


# ── Async LLM call with retry ───────────────────────────────────


async def async_llm_call(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int = 2000,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_retries: int = MAX_LLM_RETRIES,
    max_continuations: int = 0,
    tools: list[dict[str, Any]] | None = None,
    native_policy: NativeToolPolicy | None = None,
    cancel_token: Any | None = None,
    cache_breakpoints: list[Any] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
) -> LLMResult:
    """Async version of :func:`looplet.scaffolding.llm_call_with_retry`.

    Awaits ``llm.generate()`` or ``llm.generate_with_tools()`` when
    they are coroutines; calls them synchronously otherwise (supporting
    sync backends used from async context). A failed native call
    transparently falls back to ``generate()``.
    """
    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
        return LLMResult(None, RuntimeError("cancelled before LLM call"))

    policy = native_policy or NativeToolPolicy()
    use_native = policy.should_use(llm, tools)
    _gk = generate_kwargs or {}

    def _method_accepts(method_name: str, name: str) -> bool:
        fn = getattr(llm, method_name, None)
        if fn is None:
            return False
        try:
            return name in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    # Filter generate_kwargs to only keys the backend accepts
    def _filtered_kwargs(method_name: str) -> dict[str, Any]:
        fn = getattr(llm, method_name, None)
        if fn is None or not _gk:
            return {}
        out: dict[str, Any] = {}
        for k, v in _gk.items():
            try:
                sig = inspect.signature(fn)
                if k in sig.parameters:
                    out[k] = v
            except (TypeError, ValueError):
                pass
        return out

    last_error: Exception | None = None
    native_fallback = False
    native_attempted = False
    native_fallback_reason: str | None = None
    attempt_limit = max_retries + 1 + int(use_native)

    for attempt in range(attempt_limit):
        if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
            return LLMResult(None, RuntimeError("cancelled during retry"))
        try:
            if use_native:
                native_attempted = True
                call_kwargs: dict[str, Any] = {
                    "tools": tools,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    **_filtered_kwargs("generate_with_tools"),
                }
                if cache_breakpoints and _method_accepts(
                    "generate_with_tools", "cache_breakpoints"
                ):
                    call_kwargs["cache_breakpoints"] = cache_breakpoints
                result = llm.generate_with_tools(prompt, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    raise RuntimeError("native tool call returned no response")
                return LLMResult(
                    result,
                    stop_reason=getattr(llm, "last_stop_reason", None),
                    native_requested=True,
                    native_attempted=True,
                    native_succeeded=True,
                )

            call_kwargs = {
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
                **_filtered_kwargs("generate"),
            }
            if cache_breakpoints and _method_accepts("generate", "cache_breakpoints"):
                call_kwargs["cache_breakpoints"] = cache_breakpoints
            result = llm.generate(prompt, **call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            stop = getattr(llm, "last_stop_reason", None)
            continuation_count = 0
            accumulated = result if isinstance(result, str) else ""
            while (
                max_continuations > 0
                and continuation_count < max_continuations
                and stop == "max_tokens"
                and isinstance(accumulated, str)
            ):
                if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
                    break
                continuation_prompt = (
                    prompt
                    + "\n\n[assistant partial output so far]\n"
                    + accumulated
                    + "\n\n[continue from exactly where you left off; "
                    "do not repeat any prior text]"
                )
                more = llm.generate(continuation_prompt, **call_kwargs)
                if inspect.isawaitable(more):
                    more = await more
                if not isinstance(more, str):
                    break
                accumulated += more
                stop = getattr(llm, "last_stop_reason", None)
                continuation_count += 1
            result = accumulated if continuation_count else result
            return LLMResult(
                result,
                stop_reason=stop,
                continuations=continuation_count,
                native_fallback=native_fallback,
                native_requested=bool(tools is not None and policy.enabled),
                native_attempted=native_attempted,
                native_succeeded=False,
                native_fallback_reason=native_fallback_reason,
            )

        except Exception as e:
            last_error = e
            if _is_prompt_too_long(e):
                return LLMResult(None, e)
            if use_native:
                logger.warning(
                    "Async native tool call failed; falling back to regular generation: %s",
                    e,
                )
                policy.demote()
                use_native = False
                native_fallback = True
                native_fallback_reason = f"{type(e).__name__}: {e}"
                continue
            if attempt < attempt_limit - 1:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "Async LLM call attempt %d/%d failed: %s - retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                return LLMResult(None, last_error)

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
    *,
    max_steps: int | None = None,
    system_prompt: str | None = None,
) -> AsyncGenerator[Step, None]:
    """Run the async loop with exception-safe per-run cleanup."""
    effective_config = config or LoopConfig()
    if max_steps is not None:
        effective_config.max_steps = max_steps
    if system_prompt is not None:
        effective_config.system_prompt = system_prompt
    tokens = _set_context_overrides(effective_config)
    try:
        async for step in _async_composable_loop_impl(
            llm=llm,
            task=task,
            tools=tools,
            context=context,
            hooks=hooks,
            config=effective_config,
            state=state,
            session_log=session_log,
            stream=stream,
            conversation=conversation,
        ):
            yield step
    except BaseException:
        _mark_failed_state(state)
        raise
    finally:
        _reset_context_overrides(tokens)


async def _async_composable_loop_impl(
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
) -> AsyncGenerator[Step, None]:
    """Async version of :func:`looplet.loop.composable_loop`.

    Yields the same :class:`Step` objects. LLM calls are awaited;
    hooks, tools, and parsing remain synchronous.

    Usage::

        async for step in async_composable_loop(llm=llm, tools=tools, ...):
            print(step.pretty())

    The ``max_steps`` and ``system_prompt`` keyword shorthands mirror
    :func:`looplet.loop.composable_loop` - when set they override the
    matching fields on ``config`` (a fresh ``LoopConfig`` is created
    if none is passed) so simple async agents don't need to construct
    a config explicitly.
    """
    # ── Defaults ────────────────────────────────────────────────
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
    _validate_loop_inputs(task, tools, config)

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

    _LoopStartEvent = _StepStartEvent = _LLMCallStartEvent = None
    _ToolDispatchEvent = _LoopEndEvent = None
    _LLMCallEndEvent = _ToolResultEvent = _StepEndEvent = None
    if stream is not None:
        try:
            from looplet.streaming import LLMCallEndEvent as _LLMCallEndEvent
            from looplet.streaming import LLMCallStartEvent as _LLMCallStartEvent
            from looplet.streaming import LoopEndEvent as _LoopEndEvent
            from looplet.streaming import LoopStartEvent as _LoopStartEvent
            from looplet.streaming import StepEndEvent as _StepEndEvent
            from looplet.streaming import StepStartEvent as _StepStartEvent
            from looplet.streaming import ToolDispatchEvent as _ToolDispatchEvent
            from looplet.streaming import ToolResultEvent as _ToolResultEvent
        except ImportError:
            pass

    _conv = conversation if conversation is not None else Conversation()

    _ckpt_store = None
    if config.checkpoint_dir is not None:
        _ckpt_store = _FileCheckpointStore(config.checkpoint_dir)
        if config.initial_checkpoint is None:
            _latest = _ckpt_store.load_latest()
            if _latest is not None:
                config = _dc_replace(config, initial_checkpoint=_latest)
                logger.info(
                    "Auto-resuming async loop from checkpoint at step %d", _latest.step_number
                )

    _step_offset = 0
    if config.initial_checkpoint is not None:
        resumed = _resume_loop_state(config.initial_checkpoint)
        if config.run_envelope is None and isinstance(resumed.get("run_envelope"), dict):
            config = _dc_replace(
                config,
                run_envelope=RunEnvelope.from_dict(resumed["run_envelope"]),
            )
        _step_offset = resumed.get("step_offset", 0)
        restored_log = resumed.get("session_log")
        if restored_log is not None:
            session_log.entries = restored_log.entries[:]
            session_log.current_theory = restored_log.current_theory
        restored_conv = resumed.get("conversation")
        if restored_conv is not None and conversation is None:
            _conv = restored_conv
        for key, value in (resumed.get("state_counters") or {}).items():
            try:
                setattr(state, key, value)
            except AttributeError:
                pass

    # Stash task + conversation on state (same as sync loop)
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

    _history = HistoryRecorder(
        state=state,
        session_log=session_log,
        conversation=_conv,
    )

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
    await _bind_loop_context_async(loop_ctx, hooks)
    _set_run_lifecycle(loop_ctx, status=RunStatus.RUNNING, phase=RunPhase.STARTING)
    loop_ctx.native_tool_stats = NativeToolStats()
    loop_ctx.step_num = _step_offset

    # Domain callables
    _default_ee = lambda data: []  # noqa: E731
    extract_entities = (
        config.extract_entities
        or (config.domain.extract_entities if config.domain else None)
        or _default_ee
    )
    extract_step_metadata = (
        config.extract_step_metadata
        or (config.domain.extract_step_metadata if config.domain else None)
        or (lambda state, step_num: ([], []))
    )
    build_prompt_fn = config.build_prompt or (config.domain.build_prompt if config.domain else None)
    checkpoint_state = config.checkpoint_state or (
        config.domain.checkpoint_state if config.domain else None
    )
    restore_checkpoint_state = config.restore_checkpoint_state or (
        config.domain.restore_checkpoint_state if config.domain else None
    )
    if config.initial_checkpoint is not None and restore_checkpoint_state is not None:
        restore_checkpoint_state(loop_ctx, resumed.get("domain_state", {}))

    # ── Pre-loop hooks ──────────────────────────────────────────
    try:
        setattr(state, "step_context", {})  # noqa: B010
    except AttributeError:
        pass

    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            await _maybe_await(hook.pre_loop(state, session_log, context))

    await emit_event_async(
        hooks,
        _LE.SESSION_START,
        state=state,
        session_log=session_log,
        context=context,
    )

    _has_streaming_hook = False
    if stream is not None and _LoopStartEvent is not None:
        task_id = task.get("id", "") if isinstance(task, dict) else str(task)[:80]
        stream.emit(_LoopStartEvent(task_summary=str(task_id), max_steps=config.max_steps))

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
    native_policy = NativeToolPolicy(enabled=config.use_native_tools)
    post_dispatch_parts: list[str] = []
    all_step_entities: list[str] = []
    # Wrap async LLM in a sync bridge so tools can use ctx.llm.generate()
    # without needing to await. The bridge handles the async→sync
    # translation via a thread pool when running inside an event loop.
    _sync_llm = _SyncBridgeLLM(llm)

    def _get_llm() -> Any:
        if config.router is not None:
            return config.router.select(purpose="reasoning")
        return llm

    def _save_checkpoint(step_number: int, *, status: str | None = None) -> None:
        if _ckpt_store is None:
            return
        metadata = {"task": str(task)}
        if status is not None:
            metadata["status"] = status
        loop_ctx.step_num = step_number
        _ckpt_store.save(
            _Checkpoint(
                step_number=step_number,
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
                domain_state=(checkpoint_state(loop_ctx) if checkpoint_state is not None else {}),
                run_envelope=(
                    loop_ctx.run_envelope.to_dict() if loop_ctx.run_envelope is not None else None
                ),
                metadata={**metadata, **_policy_checkpoint_metadata(state)},
                run_status=loop_ctx.status.value,
                run_phase=loop_ctx.phase.value,
                termination_reason=loop_ctx.termination_reason,
            ),
            key=f"step_{step_number}" if status is None else f"step_{step_number}_{status}",
        )

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset
        loop_ctx.step_num = step_num
        loop_ctx.conversation = _conv
        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break
        _set_run_lifecycle(loop_ctx, phase=RunPhase.PROMPTING)

        if stream is not None and _StepStartEvent is not None:
            stream.emit(_StepStartEvent(step_num=step_num))

        # Clear step_context
        try:
            setattr(state, "step_context", {})  # noqa: B010
        except AttributeError:
            pass

        # Cancellation check
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        _want_compact = False
        for hook in hooks:
            method = getattr(hook, "should_compact", None)
            if method is not None and await _maybe_await(
                method(state, session_log, _conv, step_num)
            ):
                _want_compact = True
                break
        if _want_compact and config.compact_service is not None:
            from looplet.compact import run_compact_async as _run_compact  # noqa: PLC0415

            await _run_compact(
                config.compact_service,
                hooks=hooks,
                state=state,
                session_log=session_log,
                llm=_get_llm(),
                conversation=_conv,
                step_num=step_num,
                reason="proactive",
            )

        # ── Build prompt ────────────────────────────────────────
        briefing_parts: list[str] = list(post_dispatch_parts)
        post_dispatch_parts.clear()
        _briefing_override: str | None = None
        for hook in hooks:
            method = getattr(hook, "build_briefing", None)
            if method is None:
                continue
            candidate = await _maybe_await(method(state, session_log, context))
            if candidate is not None:
                _briefing_override = str(candidate)
                break
        if _briefing_override is not None:
            briefing_parts.append(_briefing_override)
        for hook in hooks:
            method = getattr(hook, "pre_prompt", None)
            if method is None:
                continue
            raw = await _maybe_await(method(state, session_log, context, step_num))
            from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415

            decision = normalize_hook_return(raw, slot="pre_prompt")
            if decision is not None:
                await _emit_hook_decision_event_async(
                    hooks,
                    decision=decision,
                    hook_slot="pre_prompt",
                    hook_name=type(hook).__name__,
                    step_num=step_num,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
            text = (
                decision.additional_context if decision else raw if isinstance(raw, str) else None
            )
            if text:
                briefing_parts.append(str(text))

        _briefing = "\n".join(briefing_parts)
        _catalog = tools.tool_catalog_text()
        _state_summary = state.snapshot() if hasattr(state, "snapshot") else {}
        _log_text = session_log.render() if hasattr(session_log, "render") else ""
        _context_history = state.context_summary() if hasattr(state, "context_summary") else ""

        _hook_prompt: str | None = None
        for hook in hooks:
            method = getattr(hook, "build_prompt", None)
            if method is None:
                continue
            candidate = await _maybe_await(
                method(
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
            )
            if candidate is not None:
                _hook_prompt = str(candidate)
                break

        if _hook_prompt is not None:
            prompt = _hook_prompt
        elif build_prompt_fn is not None:
            prompt = await _maybe_await(
                build_prompt_fn(
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

        if config.render_messages_override is not None:
            projection = ContextProjection(
                messages=tuple(_conv.messages),
                default_prompt=prompt,
                step_num=step_num,
                task=task,
                tool_catalog=_catalog,
                state_summary=_state_summary,
                context_history=_context_history,
                session_log=_log_text,
                briefing=_briefing,
                memory=_rendered_memory,
            )
            prompt = _render_projection(config.render_messages_override, projection)

        _estimated_tokens = estimate_prompt_tokens(prompt)
        loop_ctx.context_budget = ContextBudgetSnapshot(
            prompt_chars=len(prompt),
            estimated_tokens=_estimated_tokens,
            context_window_tokens=config.context_window,
            briefing_chars=len(_briefing),
            context_history_chars=len(_context_history),
            pressure=_estimated_tokens > config.context_window - 3_000,
        )
        _state_metadata = getattr(state, "metadata", None)
        if loop_ctx.context_budget.pressure and isinstance(_state_metadata, dict):
            _state_metadata["context_budget"] = loop_ctx.context_budget.to_dict()

        await emit_event_async(
            hooks,
            _LE.PRE_LLM_CALL,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
            context_budget=loop_ctx.context_budget.to_dict(),
        )

        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break

        effective_llm = _get_llm()
        _sync_llm = _SyncBridgeLLM(effective_llm)

        # ── Native tool schemas ─────────────────────────────────
        _tool_schemas = native_policy.tool_schemas(effective_llm, tools)

        _cache_bps: list[Any] | None = None
        if config.cache_policy is not None:
            from looplet.cache import CacheBreakDetector as _CBD  # noqa: PLC0415
            from looplet.cache import compute_breakpoints as _compute_bps  # noqa: PLC0415

            _schemas_text = tools.tool_catalog_text()
            _detector = next((h for h in hooks if isinstance(h, _CBD)), None)
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

        # ── AWAIT: LLM call ─────────────────────────────────────
        _llm_t0 = time.perf_counter()
        _set_run_lifecycle(loop_ctx, phase=RunPhase.LLM)
        if stream is not None and _LLMCallStartEvent is not None:
            stream.emit(_LLMCallStartEvent(step_num=step_num))
        llm_result = await async_llm_call(
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
        )
        loop_ctx.native_tool_stats.record(llm_result)
        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break
        _state_metadata = getattr(state, "metadata", None)
        if isinstance(_state_metadata, dict) and loop_ctx.native_tool_stats.has_activity:
            _state_metadata["native_tool_stats"] = loop_ctx.native_tool_stats.to_dict()
        _llm_dur_ms = (time.perf_counter() - _llm_t0) * 1000.0
        llm_calls += 1
        if stream is not None and _LLMCallEndEvent is not None:
            stream.emit(
                _LLMCallEndEvent(
                    step_num=step_num,
                    response_length=len(llm_result.text or ""),
                    duration_ms=_llm_dur_ms,
                )
            )

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
            stop_reason = "llm_error"
            break

        # ── Parse response ──────────────────────────────────────
        tool_calls = native_policy.parse_response(raw_response)

        if not tool_calls:
            consecutive_parse_failures += 1
            if config.recovery_registry is not None:
                from looplet.recovery import FailureScenario as _FailureScenario  # noqa: PLC0415

                recovery_action = config.recovery_registry.attempt_recovery(
                    _FailureScenario.PARSE_ERROR,
                    {"step": step_num, "raw_response": raw_response},
                )
                if recovery_action is not None and recovery_action.action_type == "abort":
                    tc = ToolCall(
                        tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200]
                    )
                    tr = ToolResult(
                        tool="__parse_error__",
                        args_summary="",
                        data=None,
                        error=f"Parse error - recovery aborted: {recovery_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tc, tool_result=tr)
                    state.steps.append(step)
                    yield step
                    _history.record_step(
                        step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                    )
                    _save_checkpoint(step_num)
                    continue
                if recovery_action is not None and recovery_action.message:
                    post_dispatch_parts.append(recovery_action.message)
            if consecutive_parse_failures <= PARSE_RECOVERY_MAX:
                recovery_prompt = build_parse_recovery_prompt(prompt, to_text(raw_response) or "")
                recovery_result = await async_llm_call(
                    llm,
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
                _history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
                _save_checkpoint(step_num)
                continue
        else:
            consecutive_parse_failures = 0

        # ── Dispatch tool calls ─────────────────────────────────
        done_tool_name = config.done_tool
        terminal_set = {done_tool_name, *config.done_tools}
        done_idx = None
        for i, tc in enumerate(tool_calls):
            if tc.tool in terminal_set:
                done_idx = i
                done_tool_name = tc.tool  # pin canonical to the firing sentinel
                break

        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls
        if regular_calls:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.DISPATCHING)
            _intercept = await _intercept_tool_calls_async(
                regular_calls,
                hooks,
                state,
                session_log,
                context,
                step_num,
            )
            intercepted_results = _intercept.intercepted
            post_dispatch_parts.extend(_intercept.extra_context)

            dispatch_items = [
                (i, tc) for i, tc in enumerate(regular_calls) if i not in intercepted_results
            ]
            calls_to_dispatch = [tc for _, tc in dispatch_items]

            if calls_to_dispatch:

                def _ctx_for(_c: ToolCall, _cur_step: int):
                    return _build_tool_ctx(
                        config,
                        tools=tools,
                        hooks=hooks,
                        tool_call=_c,
                        step_num=_cur_step,
                        state=state,
                        session_log=session_log,
                        llm=_sync_llm,
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
                tool_spec = tools._tools.get(tool_call.tool)
                was_intercepted = tc_idx in intercepted_results
                if not (tool_spec and tool_spec.free) and not was_intercepted:
                    state.queries_used += 1

                from looplet.loop import _get_persist_threshold  # noqa: PLC0415

                tool_result.data = truncate_tool_result(
                    tool_result.data,
                    persist_dir=config.tool_result_persist_dir,
                    persist_threshold=_get_persist_threshold(),
                )

                if stream is not None and _ToolDispatchEvent is not None:
                    stream.emit(
                        _ToolDispatchEvent(
                            step_num=cur_step,
                            tool_name=tool_call.tool,
                            args_summary=_summarize_args_dict(tool_call.args),
                        )
                    )

                _pd = await _run_post_dispatch_hooks_async(
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
                all_step_entities.extend(step_entities)
                step_findings, step_highlights = extract_step_metadata(state, cur_step)
                theory = tool_call.args.get("__theory__", "")
                _history.record_step(
                    step,
                    theory=theory,
                    entities=step_entities,
                    findings=step_findings,
                    highlights=step_highlights,
                    recall_key="",
                )
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
                            new_entities_count=len(step_entities),
                        )
                    )
                _save_checkpoint(cur_step)

        # Handle done()
        if done_idx is not None:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.FINALIZING)
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415

                    try:
                        w = await _maybe_await(
                            _call_check_done(hook, state, session_log, context, step_num, tool_call)
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "check_done hook %s raised; continuing",
                            type(hook).__name__,
                        )
                        w = None
                    _decision = normalize_hook_return(w, slot="check_done")
                    if _decision is not None:
                        await _emit_hook_decision_event_async(
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

            schema_for_call = None
            if gate_warning is None:
                if tool_call.tool == config.done_tool and config.output_schema is not None:
                    schema_for_call = config.output_schema
                elif tool_call.tool in config.done_tool_schemas:
                    schema_for_call = config.done_tool_schemas[tool_call.tool]
            if schema_for_call is not None:
                validation = _validate_args(schema_for_call, tool_call.args)
                if not validation.valid:
                    gate_warning = (
                        f"Output schema validation failed: {'; '.join(validation.errors)}"
                    )

            if gate_warning is not None:
                tool_result = ToolResult(
                    tool=done_tool_name,
                    args_summary="rejected",
                    data={"rejected": True, "reason": gate_warning},
                )
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
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
                _history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
                _save_checkpoint(cur_step)
            else:
                _ctx = _build_tool_ctx(
                    config,
                    tools=tools,
                    hooks=hooks,
                    tool_call=tool_call,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    llm=_sync_llm,
                )
                tool_result = tools.dispatch(tool_call, ctx=_ctx)

                _pd_done = await _run_post_dispatch_hooks_async(
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
                _set_run_lifecycle(
                    loop_ctx,
                    status=RunStatus.COMPLETED,
                    phase=RunPhase.TERMINAL,
                    termination_reason="done",
                )
                _save_checkpoint(cur_step, status="done")
                done = True
                stop_reason = "done"
                await emit_event_async(
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

        # Should-stop check
        for hook in hooks:
            if hasattr(hook, "should_stop"):
                from looplet.hook_decision import normalize_hook_return  # noqa: PLC0415

                _raw = await _maybe_await(hook.should_stop(state, step_num, len(all_step_entities)))
                _decision = normalize_hook_return(_raw, slot="should_stop")
                if _decision is not None and _decision.is_stop():
                    stop_reason = _decision.stop or "hook_requested_stop"
                    done = True
                    break

    final_status = (
        RunStatus.COMPLETED
        if stop_reason == "done"
        else RunStatus.CANCELLED
        if stop_reason in {"cancelled", "deadline_exceeded"}
        else RunStatus.FAILED
        if stop_reason in {"error", "llm_error"}
        else RunStatus.STOPPED
    )
    _set_run_lifecycle(
        loop_ctx,
        status=final_status,
        phase=RunPhase.TERMINAL,
        termination_reason=stop_reason,
    )
    if state is not None:
        state._stop_reason = stop_reason  # pyright: ignore[reportAttributeAccessIssue]

    await emit_event_async(
        hooks,
        _LE.STOP,
        state=state,
        session_log=session_log,
        context=context,
        termination_reason=stop_reason,
    )

    if stream is not None and _LoopEndEvent is not None:
        stream.emit(
            _LoopEndEvent(
                total_steps=state.step_count,
                total_llm_calls=llm_calls,
                reason=stop_reason,
            )
        )

    # ── on_loop_end ─────────────────────────────────────────────
    for hook in hooks:
        if hasattr(hook, "on_loop_end"):
            result = hook.on_loop_end(state, session_log, context, llm)
            if inspect.isawaitable(result):
                await result
