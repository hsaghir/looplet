"""Async composable loop - ``async for step in async_composable_loop(...)``.

Async mirror of :func:`looplet.loop.composable_loop` for use with
async LLM backends (:class:`AsyncOpenAIBackend`, etc.). Tool execution,
parsing, and state management remain synchronous; LLM calls and
awaitable hook results are awaited.

Usage::

    from looplet.async_loop import async_composable_loop
    from looplet.backends import AsyncOpenAIBackend

    llm = AsyncOpenAIBackend(base_url="...", api_key="...", model="gpt-4o")

    async for step in async_composable_loop(
        llm=llm, tools=tools, state=state, config=config, task=task,
    ):
        print(step.pretty())

The function accepts the same arguments as ``composable_loop`` and
yields the same :class:`Step` objects. Hooks may be synchronous or
``async def`` and run sequentially between awaited LLM calls.

When to use this instead of ``composable_loop``:
- Your LLM backend has ``async def generate()``
- You're inside an async context (FastAPI, aiohttp, Discord bot)
- You want to run multiple agent loops concurrently via asyncio.gather

When NOT to use this:
- Your LLM backend is synchronous - use ``composable_loop`` instead
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from contextlib import aclosing
from dataclasses import replace as _dc_replace
from typing import Any, AsyncGenerator, Callable, Sequence

from looplet._backend_contract import TerminalLLMError
from looplet._backend_contract import attempt_limit as _attempt_limit
from looplet.checkpoint import _capture_checkpoint, _run_persistence
from looplet.checkpoint import _save_pending_dispatch_async as _save_pending_dispatch
from looplet.checkpoint import resume_loop_state as _resume_loop_state
from looplet.checkpoint import validate_checkpoint_identity as _validate_checkpoint_identity
from looplet.context_plan import ContextSourceSelector
from looplet.context_projection import ContextProjection as ContextProjection
from looplet.hook_decision import _invoke_hook
from looplet.loop import (
    ContextBudgetSnapshot as ContextBudgetSnapshot,
)
from looplet.loop import (
    LoopConfig,
    LoopContext,
    RunEnvelope,
    RunPhase,
    RunStatus,
    _bind_loop_context_async,
    _build_tool_ctx,
    _call_check_done,
    _completion_gate_warning,
    _deadline_expired,
    _default_build_briefing,
    _default_extract_entities,
    _emit_hook_decision_event_async,
    _intercept_tool_calls_async,
    _mark_failed_state,
    _normalize_loop_inputs,
    _notify_post_step_async,
    _policy_checkpoint_metadata,
    _post_step_stop_reason_async,
    _prepare_tool_turn,
    _record_run_stats,
    _rejected_terminal_step,
    _reset_context_overrides,
    _resolve_loop_config,
    _resolve_loop_state,
    _run_post_dispatch_hooks_async,
    _set_context_overrides,
    _set_run_lifecycle,
    _step_metadata,
    _terminal_step_metadata,
    _validate_hooks,
    _validate_loop_inputs,
    emit_event_async,
)
from looplet.loop import (
    _render_projection as _render_projection,
)
from looplet.native_tools import NativeToolPolicy, NativeToolUnsupportedError
from looplet.parse import parse_multi_tool_calls, to_text
from looplet.prompt_preparation import prepare_prompt_async
from looplet.recovery_strategies import recovery_aggressive_budget, recovery_clear_old_results
from looplet.result_projection import project_tool_result
from looplet.scaffolding import (
    PARSE_RECOVERY_MAX,
    ContextOverflowError,
    LLMResult,
    NativeToolStats,
    _is_prompt_too_long,
    _is_terminal_llm_error,
    _native_llm_result,
    _NativeTurnRecovery,
    build_parse_recovery_prompt,
)
from looplet.scaffolding import _accepts_kwarg as _sync_accepts_kwarg
from looplet.scaffolding import (
    estimate_prompt_tokens as estimate_prompt_tokens,
)
from looplet.session import SessionLog
from looplet.tools import BaseToolRegistry, ToolSpec, _summarize_args_dict
from looplet.types import AgentState, RunResult, Step, ToolCall, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "run_result_async",
    "async_composable_loop",
    "async_llm_call",
]


async def _recover_prompt(
    llm: Any,
    result: LLMResult,
    recovery_state: dict[str, Any],
    *,
    state: Any,
    session_log: Any,
    tools: Any,
    config: Any,
    step_num: int,
    hooks: list[Any],
    conversation: Any,
    native_policy: NativeToolPolicy,
    prepare: Any,
    loop_ctx: Any,
    compact_service: Any = None,
) -> tuple[LLMResult, int]:
    from looplet.compact import TruncateCompact, run_compact_async

    calls = 0
    for name in ("budget_enforcement", "emergency_truncate", "result_clearing"):
        if recovery_state.get(name):
            continue
        recovery_state[name] = True
        if name == "emergency_truncate":
            service = (
                config.compact_service if config.compact_service is not None else compact_service
            )
            outcome = await run_compact_async(
                service if service is not None else TruncateCompact(),
                hooks=hooks,
                state=state,
                session_log=session_log,
                llm=llm,
                conversation=conversation,
                step_num=step_num,
                reason="prompt_too_long",
            )
            calls += outcome.llm_calls_spent
        else:
            strategy = (
                recovery_aggressive_budget
                if name == "budget_enforcement"
                else recovery_clear_old_results
            )
            calls += strategy(state, session_log, llm, step_num)
        prepared = await prepare()
        if prepared.budget.pressure:
            continue
        schemas = native_policy.tool_schemas(llm, tools)
        previous = result
        result = await async_llm_call(
            llm,
            prepared.prompt,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            tools=list(prepared.tool_view.schemas) if schemas is not None else None,
            native_policy=native_policy,
            cancel_token=config.cancel_token,
            max_continuations=config.max_turn_continuations,
            generate_kwargs=config.generate_kwargs or None,
            should_continue=lambda: not _deadline_expired(loop_ctx),
        )
        loop_ctx.native_tool_stats.record(result)
        calls += result.calls + (0 if result.native_attempted else result.continuations)
        result._include_previous(previous)
        if result.ok or not result.is_prompt_too_long:
            break
    if result.ok:
        recovery_state.clear()
    return result, calls


# ── Retry constants (mirror scaffolding.py) ──────────────────────
MAX_LLM_RETRIES = 2
RETRY_BACKOFF_BASE = 1.0


async def _maybe_await(value: Any) -> Any:
    """Resolve sync or async hook results in registration order."""
    if inspect.isawaitable(value):
        return await value
    return value


class _SyncBridgeLLM:
    """Provide synchronous model access to tools running in worker threads.

    The async loop supplies its running loop so async backend calls stay
    on that loop. Async tools use ``ctx.async_llm`` directly.
    """

    def __init__(self, backend: Any, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._backend = backend
        self._loop = loop

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> str:
        result = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        if inspect.isawaitable(result):
            if self._loop is not None:
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None
                if running_loop is self._loop:
                    if inspect.iscoroutine(result):
                        result.close()
                    raise RuntimeError(
                        "Synchronous model calls cannot block their owner event loop. "
                        "Use await ctx.async_llm.generate() instead."
                    )

                async def resolve() -> str:
                    return await result

                return asyncio.run_coroutine_threadsafe(resolve(), self._loop).result()
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
    max_tokens: int | None = None,
    system_prompt: str = "",
    temperature: float | None = None,
    max_retries: int = MAX_LLM_RETRIES,
    max_continuations: int = 0,
    tools: list[dict[str, Any]] | None = None,
    native_policy: NativeToolPolicy | None = None,
    cancel_token: Any | None = None,
    cache_breakpoints: list[Any] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> LLMResult:
    """Async version of :func:`looplet.scaffolding.llm_call_with_retry`.

    Awaits ``llm.generate()`` or ``llm.generate_with_tools()`` when
    they are coroutines; calls them synchronously otherwise (supporting
    sync backends used from async context). Native tool support is required
    unless ``NativeToolPolicy(enabled=False)`` selects JSON-text mode.
    """
    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
        return LLMResult(None, RuntimeError("cancelled before LLM call"))

    policy = native_policy or NativeToolPolicy()
    if (
        tools is not None
        and policy.enabled
        and not policy.demoted
        and not policy.should_use(llm, tools)
    ):
        return LLMResult(
            None,
            NativeToolUnsupportedError(
                "Backend does not support native tool calls; set use_native_tools=False "
                "to explicitly use JSON-text tool calls"
            ),
            native_requested=True,
        )
    use_native = policy.should_use(llm, tools)
    _gk = generate_kwargs or {}

    def _method_accepts(method_name: str, name: str) -> bool:
        fn = getattr(llm, method_name, None)
        if fn is None:
            return False
        try:
            return _sync_accepts_kwarg(fn, name)
        except (TypeError, ValueError):
            return False

    def _method_accepts_kwarg(method_name: str, name: str) -> bool:
        fn = getattr(llm, method_name, None)
        return _sync_accepts_kwarg(fn, name) if fn is not None else False

    # Filter generate_kwargs to only keys the backend accepts
    def _filtered_kwargs(method_name: str) -> dict[str, Any]:
        fn = getattr(llm, method_name, None)
        if fn is None or not _gk:
            return {}
        out: dict[str, Any] = {}
        for k, v in _gk.items():
            try:
                sig = inspect.signature(fn)
                if k in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    out[k] = v
            except (TypeError, ValueError):
                pass
        return out

    last_error: Exception | None = None
    native_attempted = False
    attempt_limit = _attempt_limit(llm, max_retries)
    recovery: _NativeTurnRecovery | None = None

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
                if cancel_token is not None and _method_accepts_kwarg(
                    "generate_with_tools", "cancel_token"
                ):
                    call_kwargs["cancel_token"] = cancel_token
                if recovery is None:
                    recovery = _NativeTurnRecovery(llm, max_continuations)
                recovery.begin_call(prompt)
                result = llm.generate_with_tools(prompt, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result
                recovery.observe_usage(llm, result)
                stop = getattr(llm, "last_stop_reason", None)
                while stop == "max_tokens":
                    if should_continue is not None and not should_continue():
                        break
                    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
                        raise TerminalLLMError(
                            "Native recovery cancelled", termination_reason="cancelled"
                        )
                    next_options = recovery.restart_options(call_kwargs)
                    if next_options is None:
                        break
                    call_kwargs = next_options
                    recovery.begin_call(prompt)
                    result = llm.generate_with_tools(prompt, **call_kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                    recovery.observe_usage(llm, result)
                    stop = getattr(llm, "last_stop_reason", None)
                return recovery.finish(_native_llm_result(result, stop, recovery.continuations))

            call_kwargs = {
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "temperature": temperature,
                **_filtered_kwargs("generate"),
            }
            if cache_breakpoints and _method_accepts("generate", "cache_breakpoints"):
                call_kwargs["cache_breakpoints"] = cache_breakpoints
            if cancel_token is not None and _method_accepts_kwarg("generate", "cancel_token"):
                call_kwargs["cancel_token"] = cancel_token
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
                native_requested=bool(tools is not None and policy.enabled),
                native_attempted=native_attempted,
                native_succeeded=False,
            )

        except Exception as e:
            last_error = e
            if getattr(e, "request_started", None) is False:
                if recovery is not None:
                    recovery.calls -= 1
                    recovery.model_calls.pop()
                    return recovery.finish(LLMResult(None, e))
                return LLMResult(None, e, calls=0)
            if recovery is not None and (_is_terminal_llm_error(e) or attempt >= attempt_limit - 1):
                return recovery.finish(
                    LLMResult(None, e, native_requested=True, native_attempted=True)
                )
            if isinstance(e, TerminalLLMError):
                logger.warning("Backend stopped (not retrying): %s", e)
                return LLMResult(
                    None,
                    e,
                    native_requested=bool(tools is not None and policy.enabled),
                    native_attempted=native_attempted,
                )
            if _is_prompt_too_long(e):
                return LLMResult(None, e)
            if use_native and isinstance(e, NativeToolUnsupportedError):
                logger.error("Native tool call unsupported; set use_native_tools=False: %s", e)
                return LLMResult(None, e, native_requested=True, native_attempted=True)
            if type(e).__name__ == "LLMResponsesExhausted":
                logger.warning("Mock LLM exhausted (not retrying): %s", e)
                return LLMResult(None, e)
            if use_native:
                if attempt < max_retries:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Async native LLM call attempt %d/%d failed: %s - retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                return LLMResult(
                    None,
                    e,
                    native_requested=bool(tools is not None and policy.enabled),
                    native_attempted=native_attempted,
                )
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
) -> AsyncGenerator[Step, None]:
    """Run the asynchronous tool-calling loop and yield recorded steps.

    Args:
        llm: Backend whose ``generate()`` result can be awaited.
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

    Consume the iterator with ``async for``. If stopping early, await its ``aclose()`` method
    or use ``contextlib.aclosing``. Cleanup runs on exhaustion, closure, and exceptions.
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
    from looplet.loop import _LoopFinalizer, _mark_terminal_state

    finalizer = _LoopFinalizer()
    iterator = None
    original_cancellation: asyncio.CancelledError | None = None
    try:
        prepared.enable_async_persistence()
        tokens = _set_context_overrides(effective_config)
        iterator = _async_composable_loop_impl(
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
        async for step in iterator:
            yield step
    except GeneratorExit:
        _mark_terminal_state(finalizer.state, status="stopped", reason="stream_closed")
        raise
    except asyncio.CancelledError as exc:
        original_cancellation = exc
        _mark_terminal_state(
            finalizer.state if finalizer.state is not None else prepared.state,
            status="cancelled",
            reason="cancelled",
        )
        raise
    except BaseException:
        _mark_failed_state(finalizer.state if finalizer.state is not None else prepared.state)
        raise
    finally:
        try:
            try:
                if iterator is not None:
                    await iterator.aclose()
            finally:
                await finalizer.finish_async(suppress_errors=True)
        finally:
            try:
                try:
                    await _run_persistence(prepared.persist)
                except (Exception, asyncio.CancelledError) as exc:
                    if original_cancellation is None:
                        raise
                    logger.warning(
                        "Persistence failed during async cancellation (%s)", type(exc).__name__
                    )
            finally:
                try:
                    prepared.release()
                finally:
                    _reset_context_overrides(tokens)


async def run_result_async(
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
    """Await and close the ordinary async loop, returning its accepted output and status.

    Functions and task strings use the same defaults as :func:`async_composable_loop`.
    Omitted state is created for this invocation. Supplied state uses ordinary budget rules.
    Additional keyword arguments pass unchanged to :func:`async_composable_loop`.
    Host-supplied backends, tools, hooks, and resources are not closed.
    """
    config = _resolve_loop_config(config, max_steps=max_steps, system_prompt=system_prompt)
    if state is None:
        state = _resolve_loop_state(None, config)
    async with aclosing(
        async_composable_loop(llm, task=task, tools=tools, config=config, state=state, **kwargs)
    ) as steps:
        async for _step in steps:
            pass
    return RunResult.from_state(state)


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
    _finalizer: Any = None,
    _checkpoint_store: Any = None,
    _initial_checkpoint: Any = None,
    _compact_service: Any = None,
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
    if config is None or max_steps is not None or system_prompt is not None:
        config = _resolve_loop_config(config, max_steps=max_steps, system_prompt=system_prompt)
    if hooks is None:
        hooks = []
    _validate_loop_inputs(task, tools, config)

    if not callable(getattr(llm, "generate", None)):
        raise TypeError(
            f"llm must implement generate() (got {type(llm).__name__}). "
            "Use AsyncOpenAIBackend(base_url=...) or similar."
        )

    state = _resolve_loop_state(state, config)
    _validate_hooks(hooks)
    if session_log is None:
        session_log = SessionLog()
    from looplet.loop import _LoopFinalizer

    finalizer = _finalizer or _LoopFinalizer()
    finalizer.bind(hooks, state, session_log, context, llm)

    # Import lazily to avoid circular deps
    from looplet.conversation import Conversation  # noqa: PLC0415
    from looplet.events import LifecycleEvent as _LE  # noqa: PLC0415
    from looplet.history import HistoryRecorder  # noqa: PLC0415

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

        from looplet.assistant_turns import TURN_STATE_KEY, restore_assistant_checkpoint

        config = restore_assistant_checkpoint(state, config, saved_checkpoint)
        if TURN_STATE_KEY in saved_checkpoint.metadata:
            _step_offset = saved_checkpoint.step_number - state.step_count

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

    from looplet.loop import _get_persist_threshold

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
    _raw_extract_entities = config.resolve_callable("extract_entities", _default_extract_entities)
    try:
        _ee_params = inspect.signature(_raw_extract_entities).parameters
        _ee_takes_state = "state" in _ee_params or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in _ee_params.values()
        )
    except (TypeError, ValueError):
        _ee_takes_state = False
    if _ee_takes_state:

        def extract_entities(data: Any) -> list[str]:
            return _raw_extract_entities(data, state=state)

    else:
        extract_entities = _raw_extract_entities
    extract_step_metadata = config.resolve_callable(
        "extract_step_metadata", lambda state, step_num: ([], [])
    )
    build_prompt_fn = config.resolve_callable("build_prompt")
    build_briefing_fn = config.resolve_callable("build_briefing", _default_build_briefing)
    checkpoint_state = config.resolve_callable("checkpoint_state")
    restore_checkpoint_state = config.resolve_callable("restore_checkpoint_state")
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
        checkpoint_state = _turns.wrap_checkpoint(
            checkpoint_state,
            capture_backend_state=getattr(_ckpt_store, "checkpoint_owner", False) is not True,
        )
        _turns.checkpoint = lambda: _turns.save(_ckpt_store, loop_ctx, checkpoint_state)
        _turns.async_checkpoint = lambda: _turns.save_async(_ckpt_store, loop_ctx, checkpoint_state)
        finalizer.bind(hooks, state, session_log, context, _turns.wrap(llm))
        loop_ctx.llm = _turns.wrap(llm)

    # ── Pre-loop hooks ──────────────────────────────────────────
    try:
        setattr(state, "step_context", {})  # noqa: B010
    except AttributeError:
        pass

    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            result = _invoke_hook(
                hook, "pre_loop", state, session_log, context, _optional_kwargs={"tools": tools}
            )
            await _maybe_await(result)

    await emit_event_async(
        hooks,
        _LE.SESSION_START,
        state=state,
        session_log=session_log,
        context=context,
    )

    from looplet.streaming import StreamingHook as _StreamingHook  # noqa: PLC0415

    _has_streaming_hook = any(isinstance(h, _StreamingHook) for h in hooks)
    if stream is not None and _LoopStartEvent is not None and not _has_streaming_hook:
        task_id = task.get("id", "") if isinstance(task, dict) else str(task)[:80]
        stream.emit(_LoopStartEvent(task_summary=str(task_id), max_steps=config.max_steps))

    _context_sources = (
        ContextSourceSelector(
            config.scoped_context_sources, budget_tokens=config.scoped_context_budget_tokens
        )
        if config.scoped_context_sources
        else None
    )

    # ── Main loop ───────────────────────────────────────────────
    done = False
    stop_reason = "budget_exhausted"
    llm_failed = False
    llm_calls = 0
    _loop_t0 = time.perf_counter()
    consecutive_parse_failures = 0
    native_policy = NativeToolPolicy(enabled=config.use_native_tools)
    post_dispatch_parts: list[str] = []
    recovery_state: dict[str, Any] = {}
    all_step_entities: list[str] = []
    _sync_llm = _SyncBridgeLLM(llm, loop=asyncio.get_running_loop())

    def _get_llm() -> Any:
        if config.router is not None:
            selected = config.router.select(purpose="reasoning")
            from looplet.provenance import _RecordingBase

            if isinstance(llm, _RecordingBase):
                selected = llm.for_backend(selected)
        else:
            selected = llm
        loop_ctx.llm = _turns.wrap(selected) if _turns is not None else selected
        return loop_ctx.llm

    async def _save_checkpoint(step_number: int, *, status: str | None = None) -> None:
        if _ckpt_store is None:
            return
        metadata = {"task": str(task)}
        if status is not None:
            metadata["status"] = status
        loop_ctx.step_num = step_number
        _checkpoint = _capture_checkpoint(
            loop_ctx,
            step_number=step_number,
            checkpoint_state=checkpoint_state,
            metadata={**metadata, **_policy_checkpoint_metadata(state, _ckpt_store)},
        )
        await _run_persistence(
            _ckpt_store.save,
            _checkpoint,
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
        _hook_requested_stop = False

        if stream is not None and _StepStartEvent is not None and _turns is None:
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

        if _turns is not None:
            if _turns.stop_reason is not None:
                stop_reason = _turns.stop_reason
                break
            _turns.begin_turn()

        _want_compact = False
        for hook in hooks:
            method = getattr(hook, "should_compact", None)
            if method is None:
                continue
            try:
                should_compact = await _maybe_await(method(state, session_log, _conv, step_num))
            except Exception:  # noqa: BLE001
                logger.exception("should_compact hook raised; skipping")
                should_compact = False
            if should_compact:
                _want_compact = True
                break
        compact_service = (
            config.compact_service if config.compact_service is not None else _compact_service
        )
        if _want_compact and compact_service is not None:
            from looplet.compact import run_compact_async as _run_compact  # noqa: PLC0415

            await _run_compact(
                compact_service,
                hooks=hooks,
                state=state,
                session_log=session_log,
                llm=_get_llm(),
                conversation=_conv,
                step_num=step_num,
                reason="proactive",
            )

        # ── Build prompt ────────────────────────────────────────
        _briefing_override: str | None = None
        for hook in hooks:
            method = getattr(hook, "build_briefing", None)
            if method is None:
                continue
            try:
                candidate = await _maybe_await(method(state, session_log, context))
            except Exception:  # noqa: BLE001
                logger.exception("build_briefing hook raised; falling back")
                candidate = None
            if candidate is not None:
                _briefing_override = str(candidate)
                break
        if _briefing_override is None:
            _briefing_override = await _maybe_await(build_briefing_fn(state, session_log, context))
        briefing_parts: list[str] = [str(_briefing_override)]
        _briefing_budget = config.max_briefing_tokens
        _briefing_used = len(briefing_parts[0]) // 4 if _briefing_budget else 0
        for hook in hooks:
            method = getattr(hook, "pre_prompt", None)
            if method is None:
                continue
            raw = await _maybe_await(
                _invoke_hook(hook, "pre_prompt", state, session_log, context, step_num)
            )
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
                if _briefing_budget:
                    text_tokens = len(text) // 4
                    if _briefing_used + text_tokens > _briefing_budget:
                        briefing_parts.append("(briefing truncated - token budget exceeded)")
                        break
                    _briefing_used += text_tokens
                briefing_parts.append(str(text))

        if post_dispatch_parts:
            briefing_parts.append("\n".join(post_dispatch_parts))
            post_dispatch_parts.clear()

        _briefing = "\n".join(briefing_parts)
        _prepared = await prepare_prompt_async(
            config=config,
            state=state,
            tools=tools,
            session_log=session_log,
            task=task,
            step_num=step_num,
            briefing=_briefing,
            context_sources=_context_sources,
            hooks=hooks,
            builder=build_prompt_fn,
            conversation=_conv,
        )
        prompt = _prepared.prompt
        _tool_view = _prepared.tool_view
        _rendered_memory = _prepared.rendered_memory
        _scoped_plan = _prepared.scoped_context_plan
        loop_ctx.context_budget = _prepared.budget
        _estimated_tokens = _prepared.budget.estimated_tokens
        _state_metadata = getattr(state, "metadata", None)
        if loop_ctx.context_budget.pressure and isinstance(_state_metadata, dict):
            _state_metadata["context_budget"] = loop_ctx.context_budget.to_dict()

        _pre_llm_decisions = await emit_event_async(
            hooks,
            _LE.PRE_LLM_CALL,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
            context_budget=loop_ctx.context_budget.to_dict(),
        )
        for decision in _pre_llm_decisions:
            if decision.additional_context:
                post_dispatch_parts.append(decision.additional_context)
            if decision.stop is not None:
                stop_reason = decision.stop
                _hook_requested_stop = True

        if _turns is not None and _turns.persistent_admission and _hook_requested_stop:
            break

        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break

        effective_llm = _get_llm()
        if (
            _turns is not None
            and config.uses_typed_history
            and config.render_messages_override is None
        ):
            effective_llm.prepare_conversation(
                _conv, _prepared.conversation_prompt, request_prompt=prompt
            )
        _sync_llm = _SyncBridgeLLM(effective_llm, loop=asyncio.get_running_loop())

        # ── Native tool schemas ─────────────────────────────────
        _tool_schemas = native_policy.tool_schemas(effective_llm, tools)
        if _tool_schemas is not None:
            _tool_schemas = list(_tool_view.schemas)

        _cache_bps: list[Any] | None = None
        if config.cache_policy is not None:
            from looplet.cache import CacheBreakDetector as _CBD  # noqa: PLC0415
            from looplet.cache import compute_breakpoints as _compute_bps  # noqa: PLC0415

            _schemas_text = _tool_view.catalog_text
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
        if (
            stream is not None
            and _LLMCallStartEvent is not None
            and (_turns is None or not _has_streaming_hook)
        ):
            stream.emit(
                _LLMCallStartEvent(
                    step_num=step_num, model_turn_id=_turns.turn_id if _turns is not None else None
                )
            )
        if loop_ctx.context_budget.pressure:
            logger.warning(
                "Async pre-flight block: prompt ~%d tokens exceeds safe limit - %s",
                _estimated_tokens,
                "running recovery before LLM call"
                if config.reactive_recovery
                else "stopping before LLM call",
            )
            llm_result = LLMResult(
                None,
                ContextOverflowError(
                    f"prompt estimate {_estimated_tokens} tokens exceeds safe context "
                    f"limit {config.context_window - 3_000}"
                ),
            )
        else:
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
                        (getattr(effective_llm, "last_usage", None) or {}) if llm_result.ok else {}
                    ),
                )
            )

        raw_response = llm_result.text
        if not llm_result.ok and llm_result.is_prompt_too_long and config.reactive_recovery:

            async def _prepare_recovery_prompt():
                nonlocal prompt, _tool_view, _rendered_memory, _scoped_plan
                prepared = await prepare_prompt_async(
                    config=config,
                    state=state,
                    tools=tools,
                    session_log=session_log,
                    task=task,
                    step_num=step_num,
                    briefing=_briefing,
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

            llm_result, recovery_calls = await _recover_prompt(
                effective_llm,
                llm_result,
                recovery_state,
                state=state,
                session_log=session_log,
                tools=tools,
                config=config,
                step_num=step_num,
                hooks=hooks,
                conversation=_conv,
                native_policy=native_policy,
                prepare=_prepare_recovery_prompt,
                loop_ctx=loop_ctx,
                compact_service=_compact_service,
            )
            llm_calls += recovery_calls
            _record_run_stats(state, llm_calls=llm_calls)
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

        _post_llm_decisions = await emit_event_async(
            hooks,
            _LE.POST_LLM_RESPONSE,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
            raw_response=raw_response,
            usage=llm_result.usage
            or (getattr(effective_llm, "last_usage", None) if llm_result.ok else None),
            extra={
                "native_tool_stats": loop_ctx.native_tool_stats.to_dict(),
                "model_calls": llm_result.model_calls,
            },
        )
        for decision in _post_llm_decisions:
            if decision.additional_context:
                post_dispatch_parts.append(decision.additional_context)
            if decision.stop is not None:
                stop_reason = decision.stop
                _hook_requested_stop = True

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
            if config.cancel_token is not None and getattr(
                config.cancel_token, "is_cancelled", False
            ):
                stop_reason = "cancelled"
                break
            llm_failed = True
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
            await _notify_post_step_async(hooks, state, session_log, step_num)
            break

        # ── Parse response ──────────────────────────────────────
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
                decision = await _maybe_await(_turns.propose(assistant_text, step_num))
                decision = _turns.record_decision(decision)
            except Exception as error:
                _turns.metadata["assistant_policy_error"] = type(error).__name__
                stop_reason = "assistant_policy_error"
                llm_failed = True
                break
            _assistant_decisions = await emit_event_async(
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
            if _turns.async_checkpoint is not None:
                await _turns.async_checkpoint()
            consecutive_parse_failures = 0
            interruption = _turns.interruption_reason(_assistant_decisions)
            if interruption is not None:
                stop_reason = interruption
                break
            if decision.action == "stop":
                stop_reason = "assistant_stop"
                break
            if decision.action == "continue":
                reason = await _post_step_stop_reason_async(
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
                    await _notify_post_step_async(hooks, state, session_log, step_num)
                    await _save_checkpoint(step_num)
                    reason = await _post_step_stop_reason_async(
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
                if recovery_action is not None and recovery_action.message:
                    post_dispatch_parts.append(recovery_action.message)
            if (
                not _hook_requested_stop
                and consecutive_parse_failures <= PARSE_RECOVERY_MAX
                and (not native_policy.enabled or native_policy.demoted)
            ):
                recovery_prompt = build_parse_recovery_prompt(prompt, to_text(raw_response) or "")
                recovery_result = await async_llm_call(
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
                tc = ToolCall(tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200])
                tr = ToolResult(
                    tool="__parse_error__",
                    args_summary="",
                    data=None,
                    error=(
                        "Native tool protocol error: expected structured tool_use block; "
                        f"received {(to_text(raw_response) or '')[:200]}"
                        if native_policy.enabled and not native_policy.demoted
                        else f"Could not parse: {(to_text(raw_response) or '')[:200]}"
                    ),
                )
                step = Step(number=step_num, tool_call=tc, tool_result=tr)
                state.steps.append(step)
                yield step
                _history.record_step(
                    step,
                    theory="",
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key="",
                )
                await _notify_post_step_async(hooks, state, session_log, step_num)
                await _save_checkpoint(step_num)
                reason = await _post_step_stop_reason_async(
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

        # ── Dispatch tool calls ─────────────────────────────────
        if (
            _turns is not None
            and stream is not None
            and _StepStartEvent is not None
            and not _has_streaming_hook
        ):
            stream.emit(_StepStartEvent(step_num=step_num))
        turn_plan = _prepare_tool_turn(
            tools, tool_calls, config, state, turn_id=f"turn-{llm_calls}"
        )
        if turn_plan is None:
            stop_reason = "budget_exhausted"
            done = True
            break
        tool_calls = turn_plan.calls
        done_idx = turn_plan.terminal_index

        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls
        if regular_calls:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.DISPATCHING)

            async def _dispatch_regular_calls():
                groups = (
                    [regular_calls]
                    if config.concurrent_dispatch
                    else [[tool_call] for tool_call in regular_calls]
                )
                offset = 0
                for group in groups:
                    group_step = step_num + offset
                    intercepted = await _intercept_tool_calls_async(
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
                            await _save_pending_dispatch(
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
                                llm=_sync_llm,
                                async_llm=_get_llm(),
                            )
                            for index, call in dispatch_items
                        ]
                        if config.concurrent_dispatch:
                            if (
                                type(tools).dispatch_batch is not BaseToolRegistry.dispatch_batch
                                and type(tools).async_dispatch_batch
                                is BaseToolRegistry.async_dispatch_batch
                            ):
                                results = await asyncio.to_thread(
                                    tools.dispatch_batch, calls_to_dispatch, ctx=tool_contexts
                                )
                            elif (
                                type(tools).async_dispatch_batch
                                is BaseToolRegistry.async_dispatch_batch
                            ):
                                results = await tools.async_dispatch_batch(
                                    calls_to_dispatch,
                                    ctx=tool_contexts,
                                    max_concurrent=config.max_parallel_calls,
                                )
                            else:
                                results = await tools.async_dispatch_batch(
                                    calls_to_dispatch, ctx=tool_contexts
                                )
                        else:
                            results = [
                                await tools.async_dispatch(
                                    calls_to_dispatch[0], ctx=tool_contexts[0]
                                )
                            ]
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

            async for tc_idx, tool_call, tool_result, was_intercepted in _dispatch_regular_calls():
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

                if stream is not None and _ToolDispatchEvent is not None:
                    stream.emit(
                        _ToolDispatchEvent(
                            step_num=cur_step,
                            tool_name=tool_call.tool,
                            args_summary=_summarize_args_dict(tool_call.args),
                            metadata=event_metadata,
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
                    recall_key=tool_result.result_key or "",
                )
                await _notify_post_step_async(hooks, state, session_log, cur_step)
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
                            new_entities_count=len(step_entities),
                            metadata=event_metadata,
                        )
                    )
                await _save_checkpoint(cur_step)

                if not config.concurrent_dispatch:
                    reason = None
                    if _deadline_expired(loop_ctx):
                        reason = "deadline_exceeded"
                    elif config.cancel_token is not None and config.cancel_token.is_cancelled:
                        reason = "cancelled"
                    elif tc_idx + 1 < len(regular_calls) or done_idx is not None:
                        reason = await _post_step_stop_reason_async(
                            hooks,
                            state=state,
                            session_log=session_log,
                            context=context,
                            step_num=cur_step,
                            new_entities=len(step_entities),
                            requested_reason=stop_reason if done else None,
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
                            _call_check_done(hook, state, session_log, context, cur_step, tool_call)
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "check_done hook %s raised; rejecting done()",
                            type(hook).__name__,
                        )
                        from looplet.hook_decision import HookDecision  # noqa: PLC0415

                        w = HookDecision(
                            block=(
                                f"Quality gate {type(hook).__name__} failed; "
                                "the final answer was not accepted."
                            ),
                            metadata={"hook_error": True},
                        )
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

            gate_warning = _completion_gate_warning(config, tool_call, gate_warning)
            done_metadata = _terminal_step_metadata(tool_call, tool_calls[done_idx + 1 :])

            if gate_warning is not None:
                step = _rejected_terminal_step(tool_call, cur_step, gate_warning, done_metadata)
                tool_result = step.tool_result
                state.steps.append(step)
                yield step
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
                            classification="rejected",
                            new_entities_count=0,
                            metadata=done_metadata.get("turn", {}),
                        )
                    )
                _history.record_step(
                    step,
                    theory="",
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key="",
                )
                await _notify_post_step_async(hooks, state, session_log, cur_step)
                await _save_checkpoint(cur_step)
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
                    async_llm=_get_llm(),
                )
                if (
                    _ckpt_store is not None
                    and tool_call.metadata.get("completion_source") != "assistant_message"
                ):
                    await _save_pending_dispatch(
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
                    tool_result = await tools.async_dispatch(tool_call, ctx=_ctx)

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
                rejected_done = tool_result.error is not None
                done_metadata["terminal_accepted"] = not rejected_done
                if rejected_done:
                    done_data = tool_result.data if isinstance(tool_result.data, dict) else {}
                    reason = str(tool_result.error)
                    tool_result.data = {**done_data, "rejected": True, "reason": reason}
                    tool_result.error = reason
                    post_dispatch_parts.append(reason)

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
                if rejected_done:
                    await _notify_post_step_async(hooks, state, session_log, cur_step)
                    await _save_checkpoint(cur_step)
                    continue
                _set_run_lifecycle(
                    loop_ctx,
                    status=RunStatus.COMPLETED,
                    phase=RunPhase.TERMINAL,
                    termination_reason="done",
                    terminal_tool=tool_call.tool,
                )
                await _notify_post_step_async(hooks, state, session_log, cur_step)
                await _save_checkpoint(cur_step, status="done")
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

        reason = await _post_step_stop_reason_async(
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
        if _turns.async_checkpoint is not None:
            await _turns.async_checkpoint()
    _record_run_stats(state, llm_calls=llm_calls)

    await emit_event_async(
        hooks,
        _LE.STOP,
        state=state,
        session_log=session_log,
        context=context,
        termination_reason=stop_reason,
    )

    # ── on_loop_end ─────────────────────────────────────────────
    llm_calls += await finalizer.finish_async()
    if _turns is not None:
        llm_calls = _turns.call_total(llm_calls)
        if _turns.async_checkpoint is not None:
            await _turns.async_checkpoint()
    _record_run_stats(state, llm_calls=llm_calls)

    _record_run_stats(
        state, llm_calls=llm_calls, duration_ms=(time.perf_counter() - _loop_t0) * 1000
    )
    if stream is not None and _LoopEndEvent is not None and not _has_streaming_hook:
        stream.emit(
            _LoopEndEvent(
                total_steps=state.step_count,
                total_llm_calls=llm_calls,
                reason=stop_reason,
            )
        )
