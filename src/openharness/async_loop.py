"""Async composable agent loop — non-blocking version.

Provides async versions of the core loop primitives:
  - AsyncLLMBackend: Protocol for async LLM backends
  - AsyncLoopHook: Protocol for async hook implementations
  - async_composable_loop: Async generator loop (yields Steps)
  - async_llm_call_with_retry: Async LLM call with exponential backoff
  - SyncToAsyncAdapter: Wraps a sync LLMBackend for async use
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

from openharness.checkpoint import (
    Checkpoint as _Checkpoint,
)
from openharness.checkpoint import (
    FileCheckpointStore as _FileCheckpointStore,
)
from openharness.checkpoint import (
    resume_loop_state as _resume_loop_state,
)
from openharness.recovery import FailureScenario as _FailureScenario
from openharness.scaffolding import (
    PARSE_RECOVERY_MAX,
    LLMResult,
    build_parse_recovery_prompt,
    estimate_prompt_tokens,
    truncate_tool_result,
)
from openharness.types import Step, ToolCall, ToolContext, ToolResult
from openharness.validation import validate_args as _validate_args

logger = logging.getLogger(__name__)


# ── Protocols ─────────────────────────────────────────────────────


@runtime_checkable
class AsyncLLMBackend(Protocol):
    """Protocol for async LLM backends.

    Any async LLM backend must implement async generate() with this signature.
    """

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str: ...


@runtime_checkable
class AsyncLoopHook(Protocol):
    """Protocol for async loop hooks.

    Async version of LoopHook — all methods are coroutines.
    Hooks inject domain-specific behavior into the async agent loop.
    """

    async def pre_loop(
        self,
        state: Any,
        session_log: Any,
        context: Any,
    ) -> None:
        """Called once at the start of the loop, before any steps."""
        ...

    async def pre_prompt(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Called before each LLM prompt is built. Returns optional briefing text."""
        ...

    async def pre_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: ToolCall,
        step_num: int,
    ) -> ToolResult | None:
        """Called before each tool is dispatched. Returns intercepted result or None."""
        ...

    async def check_permission(
        self,
        tool_call: ToolCall,
        state: Any,
    ) -> bool:
        """Called before tool dispatch. Return True to allow, False to deny."""
        ...

    async def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: ToolCall,
        tool_result: ToolResult,
        step_num: int,
    ) -> str | None:
        """Called after each tool execution. Returns optional briefing text."""
        ...

    async def check_done(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Called when done() is requested. Returns rejection message or None."""
        ...

    async def should_stop(
        self,
        state: Any,
        step_num: int,
        new_entities: int,
    ) -> bool:
        """Returns True to stop the loop early."""
        ...

    async def on_loop_end(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        llm: Any,
    ) -> int:
        """Called once after the loop exits. Returns extra LLM call count."""
        ...


# ── SyncToAsyncAdapter ────────────────────────────────────────────


class SyncToAsyncAdapter:
    """Wraps a synchronous LLMBackend to satisfy the AsyncLLMBackend protocol.

    Runs the sync generate() in an asyncio executor to avoid blocking
    the event loop. ``generate_with_tools`` is exposed only when the
    wrapped backend implements it — so ``hasattr(adapter,
    "generate_with_tools")`` correctly reflects the underlying
    capability.
    """

    def __init__(self, sync_llm: Any) -> None:
        self._sync_llm = sync_llm
        # Expose generate_with_tools only if the wrapped backend supports it,
        # so the native-tools hasattr() gate in the loops reflects the real
        # underlying capability.
        if hasattr(sync_llm, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl  # type: ignore[method-assign]

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_llm.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            ),
        )

    async def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_llm.generate_with_tools(
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            ),
        )


# ── Async retry ────────────────────────────────────────────────────


def _build_async_tool_ctx(config: Any) -> ToolContext | None:
    """Return a ToolContext when config has anything worth threading."""
    if (getattr(config, "cancel_token", None) is None
            and getattr(config, "elicit_handler", None) is None):
        return None
    return ToolContext(
        cancel_token=getattr(config, "cancel_token", None),
        elicit=getattr(config, "elicit_handler", None),
    )


async def _maybe_await_with_ctx(fn: Any, arg: Any, ctx: ToolContext | None):
    """Call ``fn(arg)`` or ``fn(arg, ctx=ctx)`` depending on its signature,
    awaiting the result. Used so that async dispatch helpers that don't
    accept ``ctx`` still work unchanged."""
    try:
        import inspect as _inspect  # noqa: PLC0415
        sig = _inspect.signature(fn)
        if "ctx" in sig.parameters and ctx is not None:
            return await fn(arg, ctx=ctx)
    except (TypeError, ValueError):
        pass
    return await fn(arg)


async def async_llm_call_with_retry(
    llm: AsyncLLMBackend,
    prompt: str,
    *,
    max_tokens: int = 2000,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_retries: int = 2,
    tools: list[dict[str, Any]] | None = None,
    cancel_token: Any | None = None,
) -> LLMResult:
    """Async version of llm_call_with_retry with asyncio.sleep backoff.

    When ``tools`` is provided and the backend exposes
    ``generate_with_tools``, native tool calling is used and the result
    is a list of Anthropic-style content blocks stored in ``result.text``.

    Cancellation semantics match :func:`openharness.scaffolding.llm_call_with_retry`:
    if ``cancel_token`` is already cancelled we skip the call, and if the
    backend's ``generate`` accepts a ``cancel_token`` kwarg we forward it.
    """
    from openharness.scaffolding import _backend_accepts_cancel_token  # noqa: PLC0415

    if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
        return LLMResult(None, RuntimeError("cancelled before async LLM call"))

    use_native = tools is not None and hasattr(llm, "generate_with_tools")
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if cancel_token is not None and getattr(cancel_token, "is_cancelled", False):
            return LLMResult(None, RuntimeError("cancelled during async retry backoff"))
        try:
            if use_native:
                call = getattr(llm, "generate_with_tools")
                extra: dict[str, Any] = {}
                if cancel_token is not None and _backend_accepts_cancel_token(llm, "generate_with_tools"):
                    extra["cancel_token"] = cancel_token
                blocks = await call(
                    prompt,
                    tools=tools,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **extra,
                )
                return LLMResult(blocks)
            call = llm.generate
            extra = {}
            if cancel_token is not None and _backend_accepts_cancel_token(llm, "generate"):
                extra["cancel_token"] = cancel_token
            text = await call(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                **extra,
            )
            return LLMResult(text)
        except Exception as e:
            last_error = e
            result = LLMResult(None, e)
            # Never retry prompt-too-long errors
            if result.is_prompt_too_long:
                return result
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %ds",
                               attempt + 1, max_retries + 1, e, wait)
                await asyncio.sleep(wait)

    return LLMResult(None, last_error)


# ── Async reactive recovery (parity with sync loop) ───────────────


async def _async_reactive_recovery_chain(
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
    max_tokens: int,
    system_prompt: str,
    temperature: float,
) -> tuple[Any, int]:
    """Async analog of ``loop._reactive_recovery_chain``.

    Tries 3 strategies in sequence (aggressive budget, deterministic
    compact, result clearing). Each fires at most once per ``recovery_state``.
    Returns ``(raw_response, extra_llm_calls)``. ``raw_response`` is None if
    all strategies were exhausted without recovering.
    """
    from openharness.recovery_strategies import (
        rebuild_prompt,
        recovery_aggressive_budget,
        recovery_clear_old_results,
        recovery_reactive_compact,
    )
    extra_llm_calls = 0
    strategies = [
        ("budget_enforcement", recovery_aggressive_budget),
        ("reactive_compact", recovery_reactive_compact),
        ("result_clearing", recovery_clear_old_results),
    ]
    for name, strategy_fn in strategies:
        if recovery_state.get(name):
            continue
        recovery_state[name] = True
        logger.warning("Async recovery strategy '%s' at step %d", name, step_num)
        extra_llm_calls += strategy_fn(state, session_log, llm, step_num)
        # Rebuild prompt with shrunk state
        prompt = rebuild_prompt(
            state, session_log, context, build_briefing, build_prompt_fn,
            task, tools, config, step_num,
        )
        retry_result = await async_llm_call_with_retry(
            llm, prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
            cancel_token=config.cancel_token,
        )
        extra_llm_calls += 1
        if retry_result.ok:
            logger.info("Async recovery '%s' succeeded at step %d", name, step_num)
            return retry_result.text, extra_llm_calls
        if not retry_result.is_prompt_too_long:
            break
    logger.error("All async recovery strategies exhausted at step %d", step_num)
    return None, extra_llm_calls


# ── Async composable loop ──────────────────────────────────────────


async def async_composable_loop(
    llm: AsyncLLMBackend,
    task: dict[str, Any] | None = None,
    tools: Any = None,
    context: Any = None,
    hooks: list[Any] | None = None,
    config: Any = None,
    state: Any = None,
    session_log: Any = None,
    stream: Any | None = None,
    conversation: Any | None = None,
) -> AsyncGenerator[Step, None]:
    """Async agent loop — async generator that yields Steps.

    Same logic as sync composable_loop but:
    - LLM calls use await
    - Hook methods are awaited
    - Concurrent-safe tools dispatched with asyncio.gather
    - Uses asyncio.sleep for backoff
    - Supports tracer, streaming, router, checkpoint, recovery registry

    Args match sync composable_loop. config accepts the same LoopConfig.
    """
    from openharness.parse import coerce_text, parse_multi_tool_calls, parse_native_tool_use

    if task is None:
        task = {}
    if tools is None:
        raise ValueError("tools is required")
    if hooks is None:
        hooks = []
    if session_log is None:
        from openharness.session import SessionLog
        session_log = SessionLog()

    # ── Conversation thread — always active (single source of truth) ──
    from openharness.conversation import Conversation as _Conversation  # noqa: PLC0415
    _conv = conversation if conversation is not None else _Conversation()

    # ── Unified history recorder — single write path for step/turn events ──
    from openharness.history import HistoryRecorder as _HistoryRecorder  # noqa: PLC0415
    _history = _HistoryRecorder(state=state, session_log=session_log, conversation=_conv)

    # Resolve config defaults
    if config is None:
        from openharness.loop import LoopConfig
        config = LoopConfig()

    # ── Feature flags ──
    from openharness.flags import FLAGS  # noqa: PLC0415

    done_tool_name = getattr(config, "done_tool", "done")
    max_steps = getattr(config, "max_steps", 15)
    max_tokens = getattr(config, "max_tokens", 2000)
    system_prompt = getattr(config, "system_prompt", "")
    temperature = getattr(config, "temperature", 0.2)
    recovery_temperature = getattr(config, "recovery_temperature", 0.1)
    build_prompt_fn = getattr(config, "build_prompt", None)
    build_briefing_fn = getattr(config, "build_briefing", None)
    extract_entities_fn = getattr(config, "extract_entities", None)
    extract_step_metadata_fn = getattr(config, "extract_step_metadata", None)
    use_native_tools = getattr(config, "use_native_tools", False)
    max_briefing_tokens = getattr(config, "max_briefing_tokens", None)

    # ── Lazy streaming imports (avoid circular: streaming imports loop.LoopHook)
    _LoopStartEvent = _StepStartEvent = _LLMCallStartEvent = None
    _ToolDispatchEvent = _LoopEndEvent = None
    if stream is not None:
        try:
            from openharness.streaming import (
                LLMCallStartEvent as _LLMCallStartEvent,
            )
            from openharness.streaming import (
                LoopEndEvent as _LoopEndEvent,
            )
            from openharness.streaming import (  # noqa: PLC0415
                LoopStartEvent as _LoopStartEvent,
            )
            from openharness.streaming import (
                StepStartEvent as _StepStartEvent,
            )
            from openharness.streaming import (
                ToolDispatchEvent as _ToolDispatchEvent,
            )
        except ImportError:
            pass

    # ── Resolve effective LLM (router overrides direct llm) ────
    _router = getattr(config, "router", None)
    # Keyed by backend object when hashable (so we never get id() recycling
    # collisions if a backend is freed and a new one allocates at the same
    # address), falling back to id() for unhashable backends.
    _adapter_cache: dict[Any, SyncToAsyncAdapter] = {}

    def _get_llm() -> Any:
        if _router is not None:
            backend = _router.select(purpose="reasoning")
            if not inspect.iscoroutinefunction(getattr(backend, "generate", None)):
                try:
                    key: Any = backend
                    if key not in _adapter_cache:
                        _adapter_cache[key] = SyncToAsyncAdapter(backend)
                    return _adapter_cache[key]
                except TypeError:
                    key = id(backend)
                    if key not in _adapter_cache:
                        _adapter_cache[key] = SyncToAsyncAdapter(backend)
                    return _adapter_cache[key]
            return backend
        return llm

    # ── Checkpoint store setup ──────────────────────────────────
    _ckpt_store = None
    _checkpoint_dir = getattr(config, "checkpoint_dir", None)
    if _checkpoint_dir is not None:
        _ckpt_store = _FileCheckpointStore(_checkpoint_dir)

    # ── Crash-resume from initial checkpoint ───────────────────
    _step_offset = 0
    _initial_checkpoint = getattr(config, "initial_checkpoint", None)
    if _initial_checkpoint is not None:
        resumed = _resume_loop_state(_initial_checkpoint)
        _step_offset = resumed.get("step_offset", 0)
        restored_log = resumed.get("session_log")
        if restored_log is not None:
            session_log.entries = restored_log.entries[:]
            session_log.current_theory = restored_log.current_theory
        # Restore state counters (queries_used, budget_remaining) so
        # budget enforcement continues where the checkpoint left off.
        # Some state classes expose budget_remaining as a read-only property
        # derived from steps/max_steps; skip fields that can't be assigned.
        if state is not None:
            for _k, _v in (resumed.get("state_counters") or {}).items():
                try:
                    setattr(state, _k, _v)
                except AttributeError:
                    pass

    # ── Tracer and recovery registry ──────────────────────────
    _tracer = getattr(config, "tracer", None)
    _recovery_registry = getattr(config, "recovery_registry", None)
    _output_schema = getattr(config, "output_schema", None)

    # ── Loop state ─────────────────────────────────────────────
    consecutive_parse_failures = 0
    quality_gate_message = ""
    post_dispatch_parts: list[str] = []
    llm_calls = 0
    done = False
    stop_reason = "budget_exhausted"
    # Recovery state — each strategy fires at most once per loop run
    recovery_state = {
        "budget_enforcement": False,
        "reactive_compact": False,
        "result_clearing": False,
    }
    t0 = time.time()

    # ── Pre-loop hooks ─────────────────────────────────────────
    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            if inspect.iscoroutinefunction(hook.pre_loop):
                await hook.pre_loop(state, session_log, context)
            else:
                hook.pre_loop(state, session_log, context)

    # ── Emit LoopStartEvent ─────────────────────────────────────
    # Skip if a StreamingHook is in hooks — it already emits LoopStartEvent
    from openharness.streaming import StreamingHook as _StreamingHook  # noqa: PLC0415
    _has_streaming_hook = any(isinstance(h, _StreamingHook) for h in hooks)
    if stream is not None and _LoopStartEvent is not None and not _has_streaming_hook:
        stream.emit(_LoopStartEvent(task_summary=str(task.get("id", "")), max_steps=max_steps))

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset

        # Cancellation check between turns — stop cleanly, no more LLM calls.
        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        # ── Pre-prompt hooks ───────────────────────────────────
        briefing_base = ""
        if build_briefing_fn is not None:
            briefing_base = build_briefing_fn(state, session_log, context)
        briefing_parts = [briefing_base]

        _briefing_budget = max_briefing_tokens
        _briefing_used = len(briefing_base) // 4 if _briefing_budget else 0

        for hook in hooks:
            if hasattr(hook, "pre_prompt"):
                if inspect.iscoroutinefunction(hook.pre_prompt):
                    text = await hook.pre_prompt(state, session_log, context, step_num)
                else:
                    text = hook.pre_prompt(state, session_log, context, step_num)
                if text:
                    if _briefing_budget:
                        text_tokens = len(text) // 4
                        if _briefing_used + text_tokens > _briefing_budget:
                            briefing_parts.append("(briefing truncated — token budget exceeded)")
                            break
                        _briefing_used += text_tokens
                    briefing_parts.append(text)

        if post_dispatch_parts:
            briefing_parts.append("\n".join(post_dispatch_parts))
            post_dispatch_parts = []

        # ── Build prompt ───────────────────────────────────────
        context_history = state.context_summary()
        if quality_gate_message:
            context_history += "\n" + quality_gate_message
            quality_gate_message = ""

        # Persistent memory — same semantics as the sync loop
        _memory_sources = getattr(config, "memory_sources", None)
        if _memory_sources:
            from openharness.memory import render_memory as _render_memory  # noqa: PLC0415
            _rendered_memory = _render_memory(_memory_sources, state)
        else:
            _rendered_memory = ""

        if build_prompt_fn is not None:
            prompt = build_prompt_fn(
                task=task,
                tool_catalog=tools.tool_catalog_text(),
                state_summary=state.snapshot(),
                context_history=context_history,
                step_number=step_num,
                max_steps=max_steps,
                session_log=session_log.render(),
                briefing="\n".join(briefing_parts),
                memory=_rendered_memory,
            )
        else:
            # Domain-agnostic default: 7-section structured prompt.
            from openharness.prompts import build_prompt as _default_build_prompt  # noqa: PLC0415
            prompt = _default_build_prompt(
                task=task,
                tool_catalog=tools.tool_catalog_text(),
                state_summary=state.snapshot(),
                context_history=context_history,
                step_number=step_num,
                max_steps=max_steps,
                session_log=session_log.render(),
                briefing="\n".join(briefing_parts),
                memory=_rendered_memory,
            )

        # ── Emit StepStartEvent ─────────────────────────────────
        if stream is not None and _StepStartEvent is not None:
            stream.emit(_StepStartEvent(step_num=step_num))

        # Resolve effective LLM once per step (router may change per step)
        effective_llm = _get_llm()

        # ── Pre-flight context check (sync-loop parity) ────────
        # If the prompt is already too long, skip the LLM call and go
        # straight to reactive recovery. Saves a guaranteed-to-fail
        # round-trip when FLAGS.reactive_recovery is on.
        estimated_tokens = estimate_prompt_tokens(prompt)
        preflight_too_long = estimated_tokens > getattr(config, "context_window", 128_000) - 3_000

        # ── LLM call with async retry + tracer span ────────────
        _native_on = (use_native_tools or FLAGS.native_tools) and hasattr(
            effective_llm, "generate_with_tools"
        )
        _tool_schemas = tools.tool_schemas() if _native_on else None
        if preflight_too_long and FLAGS.reactive_recovery:
            logger.warning(
                "Pre-flight block (async): prompt ~%d tokens exceeds safe limit — "
                "running recovery before LLM call", estimated_tokens,
            )
            llm_result = LLMResult(None, Exception("pre-flight: prompt is too long"))
        else:
            # Emit LLMCallStartEvent only when we actually call the LLM
            if stream is not None and _LLMCallStartEvent is not None:
                stream.emit(_LLMCallStartEvent(step_num=step_num))
            _llm_span = None
            if _tracer is not None:
                _llm_span = _tracer.start_span(
                    f"llm.call.step_{step_num}",
                    attributes={"step": step_num},
                )
            llm_result = await async_llm_call_with_retry(
                effective_llm, prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
                tools=_tool_schemas,
                cancel_token=config.cancel_token,
            )
            if _llm_span is not None and _tracer is not None:
                _tracer.end_span(_llm_span)
            llm_calls += 1

        # Reactive recovery: if prompt-too-long, try chained strategies
        if not llm_result.ok and llm_result.is_prompt_too_long and FLAGS.reactive_recovery:
            raw_response_recovered, extra = await _async_reactive_recovery_chain(
                llm_result, recovery_state, state, session_log, effective_llm,
                tools, context, build_briefing_fn, build_prompt_fn,
                task, config, step_num,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            llm_calls += extra
            llm_result = LLMResult(raw_response_recovered)

        raw_response = llm_result.text

        # ── Record LLM turn in conversation thread via unified recorder ──
        _history.record_llm_turn(prompt=prompt, response=raw_response)

        if raw_response is None:
            # If cancellation caused the failure, exit cleanly — no error step.
            if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
                stop_reason = "cancelled"
                break
            logger.error("Async LLM call failed after retries at step %d", step_num)
            error_call = ToolCall(tool="__llm_error__", reasoning="LLM call failed after retries")
            error_result = ToolResult(
                tool="__llm_error__", args_summary="", data=None,
                error="LLM call failed after all retry attempts",
            )
            step = Step(number=step_num, tool_call=error_call, tool_result=error_result)
            state.steps.append(step)
            yield step
            _history.record_step(step, theory="", entities=[], findings=[], highlights=[], recall_key="")
            break

        # ── Parse response ────────────────────────────────────
        if (use_native_tools or FLAGS.native_tools) and isinstance(raw_response, list):
            tool_calls = parse_native_tool_use(raw_response)
        else:
            tool_calls = parse_multi_tool_calls(raw_response)

        if not tool_calls:
            consecutive_parse_failures += 1
            # Consult recovery_registry if set
            _recovery_action = None
            if _recovery_registry is not None:
                _recovery_action = _recovery_registry.attempt_recovery(
                    _FailureScenario.PARSE_ERROR,
                    {"step": step_num, "raw_response": raw_response},
                )
                if _recovery_action is not None and _recovery_action.action_type == "abort":
                    logger.warning("Recovery registry aborted parse recovery at step %d", step_num)
                    tool_call = ToolCall(tool="__parse_error__", reasoning=(coerce_text(raw_response) or "")[:200])
                    tool_result = ToolResult(
                        tool="__parse_error__", args_summary="", data=None,
                        error=f"Parse error — recovery aborted: {_recovery_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                    state.steps.append(step)
                    yield step
                    _history.record_step(step, theory="", entities=[], findings=[], highlights=[], recall_key="")
                    continue
                if _recovery_action is not None and _recovery_action.message:
                    post_dispatch_parts.append(_recovery_action.message)
            if consecutive_parse_failures <= PARSE_RECOVERY_MAX:
                logger.warning("Async parse failure %d at step %d — recovery attempt",
                               consecutive_parse_failures, step_num)
                recovery_prompt = build_parse_recovery_prompt(prompt, coerce_text(raw_response) or "")
                recovery_result = await async_llm_call_with_retry(
                    effective_llm, recovery_prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    temperature=recovery_temperature,
                    cancel_token=config.cancel_token,
                )
                llm_calls += 1
                if recovery_result.ok:
                    tool_calls = parse_multi_tool_calls(recovery_result.text)

            if not tool_calls:
                tool_call = ToolCall(tool="__parse_error__", reasoning=(coerce_text(raw_response) or "")[:200])
                tool_result = ToolResult(
                    tool="__parse_error__", args_summary="", data=None,
                    error=f"Could not parse JSON: {(coerce_text(raw_response) or '')[:200]}",
                )
                step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                _history.record_step(step, theory="", entities=[], findings=[], highlights=[], recall_key="")
                continue
        else:
            consecutive_parse_failures = 0

        # ── Dispatch tool calls ────────────────────────────────
        all_step_entities: list[str] = []
        done_idx = None
        for i, tc in enumerate(tool_calls):
            if tc.tool == done_tool_name:
                done_idx = i
                break

        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls

        if regular_calls:
            # Pre-dispatch hooks
            intercepted_results: dict[int, ToolResult] = {}
            for tc_idx, tc in enumerate(regular_calls):
                for hook in hooks:
                    if hasattr(hook, "pre_dispatch"):
                        if inspect.iscoroutinefunction(hook.pre_dispatch):
                            cached = await hook.pre_dispatch(state, session_log, tc, step_num + tc_idx)
                        else:
                            cached = hook.pre_dispatch(state, session_log, tc, step_num + tc_idx)
                        if cached is not None:
                            intercepted_results[tc_idx] = cached
                            break

            calls_to_dispatch = [
                (i, tc) for i, tc in enumerate(regular_calls) if i not in intercepted_results
            ]

            # Permission check: engine + hooks can deny tool calls (AND semantics)
            permitted_calls = []
            for i, tc in calls_to_dispatch:
                denied = False
                if config.permissions is not None:
                    outcome = config.permissions.evaluate(tc)
                    if outcome.denied:
                        from openharness.types import ErrorKind, ToolError  # noqa: PLC0415
                        _te = ToolError(
                            kind=ErrorKind.PERMISSION_DENIED,
                            message=outcome.reason
                                or f"Permission denied for tool '{tc.tool}'",
                            retriable=False,
                        )
                        intercepted_results[i] = ToolResult(
                            tool=tc.tool, args_summary=str(tc.args)[:100],
                            data=None,
                            error=_te.message,
                            error_detail=_te,
                        )
                        denied = True
                if denied:
                    continue
                for hook in hooks:
                    if hasattr(hook, "check_permission"):
                        if inspect.iscoroutinefunction(hook.check_permission):
                            allowed = await hook.check_permission(tc, state)
                        else:
                            allowed = hook.check_permission(tc, state)
                        if not allowed:
                            from openharness.types import ErrorKind, ToolError  # noqa: PLC0415
                            _te = ToolError(
                                kind=ErrorKind.PERMISSION_DENIED,
                                message=f"Permission denied for tool '{tc.tool}'",
                                retriable=False,
                            )
                            intercepted_results[i] = ToolResult(
                                tool=tc.tool, args_summary=str(tc.args)[:100],
                                data=None, error=_te.message, error_detail=_te,
                            )
                            denied = True
                            break
                if not denied:
                    permitted_calls.append((i, tc))

            # Dispatch: concurrent-safe tools in parallel, others sequential
            # (only when FLAGS.concurrent_dispatch is enabled — matching sync loop)
            dispatched: dict[int, ToolResult] = {}
            concurrent_indices = []
            serial_indices = []
            _tool_ctx = _build_async_tool_ctx(config)

            if FLAGS.concurrent_dispatch:
                for i, tc in permitted_calls:
                    spec = tools._tools.get(tc.tool) if hasattr(tools, "_tools") else None
                    if spec and getattr(spec, "concurrent_safe", False):
                        concurrent_indices.append((i, tc))
                    else:
                        serial_indices.append((i, tc))
            else:
                # All tools dispatched sequentially when flag is off
                serial_indices = list(permitted_calls)

            if concurrent_indices:
                if hasattr(tools, "dispatch_batch_async"):
                    conc_calls = [tc for _, tc in concurrent_indices]
                    conc_results = await _maybe_await_with_ctx(
                        tools.dispatch_batch_async, conc_calls, _tool_ctx
                    )
                    for (idx, _), result in zip(concurrent_indices, conc_results):
                        dispatched[idx] = result
                elif hasattr(tools, "dispatch_async"):
                    results = await asyncio.gather(*[
                        _maybe_await_with_ctx(tools.dispatch_async, tc, _tool_ctx)
                        for _, tc in concurrent_indices
                    ])
                    for (idx, _), result in zip(concurrent_indices, results):
                        dispatched[idx] = result
                else:
                    # Fallback: sync dispatch
                    for idx, tc in concurrent_indices:
                        dispatched[idx] = tools.dispatch(tc, ctx=_tool_ctx) if _tool_ctx is not None else tools.dispatch(tc)

            for idx, tc in serial_indices:
                if hasattr(tools, "dispatch_async"):
                    dispatched[idx] = await _maybe_await_with_ctx(
                        tools.dispatch_async, tc, _tool_ctx
                    )
                else:
                    dispatched[idx] = tools.dispatch(tc, ctx=_tool_ctx) if _tool_ctx is not None else tools.dispatch(tc)

            # Combine intercepted + dispatched
            batch_results: list[ToolResult] = []
            for i in range(len(regular_calls)):
                if i in intercepted_results:
                    batch_results.append(intercepted_results[i])
                else:
                    batch_results.append(dispatched[i])

            for tc_idx, (tool_call, tool_result) in enumerate(zip(regular_calls, batch_results)):
                cur_step = step_num + tc_idx
                was_intercepted = tc_idx in intercepted_results
                tool_spec = tools._tools.get(tool_call.tool) if hasattr(tools, "_tools") else None
                if not (tool_spec and getattr(tool_spec, "free", False)) and not was_intercepted:
                    state.queries_used += 1

                tool_result.data = truncate_tool_result(tool_result.data)

                # Emit ToolDispatchEvent
                if stream is not None and _ToolDispatchEvent is not None:
                    stream.emit(_ToolDispatchEvent(
                        step_num=cur_step,
                        tool_name=tool_call.tool,
                        args_summary=str(tool_call.args)[:200],
                    ))

                for hook in hooks:
                    if hasattr(hook, "post_dispatch"):
                        if inspect.iscoroutinefunction(hook.post_dispatch):
                            text = await hook.post_dispatch(
                                state, session_log, tool_call, tool_result, cur_step,
                            )
                        else:
                            text = hook.post_dispatch(
                                state, session_log, tool_call, tool_result, cur_step,
                            )
                        if text:
                            post_dispatch_parts.append(text)

                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                cur_step_count = state.step_count
                state.steps.append(step)
                yield step

                # Save checkpoint after each step
                if _ckpt_store is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={
                                "entries": session_log.to_list(),
                                "current_theory": getattr(session_log, "current_theory", ""),
                            },
                            conversation_data=_conv.serialize(),
                            config_snapshot={
                                "max_steps": max_steps,
                                "queries_used": getattr(state, "queries_used", 0),
                                "budget_remaining": getattr(state, "budget_remaining", 0),
                            },
                            tool_results_store={},
                            metadata={"task": str(task)},
                        ),
                        key=f"step_{cur_step}",
                    )

                if extract_entities_fn:
                    step_entities = extract_entities_fn(tool_result.data)
                else:
                    step_entities = []
                if extract_step_metadata_fn:
                    step_findings, step_highlights = extract_step_metadata_fn(state, cur_step_count)
                else:
                    step_findings, step_highlights = [], []
                all_step_entities.extend(step_entities)

                recall_key = tool_result.result_key or ""
                theory = tool_call.args.get("__theory__", "")
                _history.record_step(
                    step, theory=theory, entities=step_entities,
                    findings=step_findings, highlights=step_highlights,
                    recall_key=recall_key,
                )

        # Handle done() if present
        if done_idx is not None:
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    if inspect.iscoroutinefunction(hook.check_done):
                        w = await hook.check_done(state, session_log, context, step_num)
                    else:
                        w = hook.check_done(state, session_log, context, step_num)
                    if w is not None:
                        gate_warning = w
                        break

            # Output schema validation — reject done() if payload is invalid
            if gate_warning is None and _output_schema is not None:
                validation = _validate_args(_output_schema, tool_call.args)
                if not validation.valid:
                    gate_warning = f"Output schema validation failed: {'; '.join(validation.errors)}"

            # Emit ToolDispatchEvent for done
            if stream is not None and _ToolDispatchEvent is not None:
                stream.emit(_ToolDispatchEvent(
                    step_num=cur_step,
                    tool_name=tool_call.tool,
                    args_summary=str(tool_call.args)[:200],
                ))

            if gate_warning is not None:
                logger.info("Quality gate rejected done() at step %d", step_num)
                quality_gate_message = gate_warning
                tool_result = ToolResult(
                    tool=done_tool_name, args_summary="rejected",
                    data={"rejected": True, "reason": gate_warning},
                )
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
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
            else:
                _ctx = _build_async_tool_ctx(config)
                if hasattr(tools, 'dispatch_async'):
                    tool_result = await tools.dispatch_async(tool_call, ctx=_ctx) if _ctx is not None else await tools.dispatch_async(tool_call)
                else:
                    tool_result = tools.dispatch(tool_call, ctx=_ctx) if _ctx is not None else tools.dispatch(tool_call)
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                # Record accepted done() to session_log + conversation
                _history.record_step(
                    step,
                    theory=tool_call.args.get("__theory__", ""),
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key="",
                )
                # Save checkpoint after done step
                if _ckpt_store is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={
                                "entries": session_log.to_list(),
                                "current_theory": getattr(session_log, "current_theory", ""),
                            },
                            conversation_data=_conv.serialize(),
                            config_snapshot={
                                "max_steps": max_steps,
                                "queries_used": getattr(state, "queries_used", 0),
                                "budget_remaining": getattr(state, "budget_remaining", 0),
                            },
                            tool_results_store={},
                            metadata={"task": str(task), "status": "done"},
                        ),
                        key=f"step_{cur_step}_done",
                    )
                done = True
                stop_reason = "done"

        if done:
            continue

        for hook in hooks:
            if hasattr(hook, "should_stop"):
                if inspect.iscoroutinefunction(hook.should_stop):
                    stop = await hook.should_stop(state, step_num, len(all_step_entities))
                else:
                    stop = hook.should_stop(state, step_num, len(all_step_entities))
                if stop:
                    logger.info("Hook %s requested stop at step %d", type(hook).__name__, step_num)
                    done = True
                    stop_reason = "hook_stop"
                    break
        if done:
            break

    # ── Post-loop hooks ────────────────────────────────────────────
    # Stash stop_reason on state so hooks (e.g. StreamingHook) can read it
    if state is not None:
        state._stop_reason = stop_reason
    for hook in hooks:
        if hasattr(hook, "on_loop_end"):
            if inspect.iscoroutinefunction(hook.on_loop_end):
                extra = await hook.on_loop_end(state, session_log, context, llm)
            else:
                extra = hook.on_loop_end(state, session_log, context, llm)
            if isinstance(extra, int):
                llm_calls += extra

    # ── Emit LoopEndEvent — skip if StreamingHook already emits it ──
    if stream is not None and _LoopEndEvent is not None and not _has_streaming_hook:
        stream.emit(_LoopEndEvent(
            total_steps=state.step_count,
            total_llm_calls=llm_calls,
            reason=stop_reason,
        ))

    # ── Build trace via injected callable ─────────────────────────
    # Async generators cannot ``return`` a value (PEP 525), so we stash the
    # trace on ``state.trace`` (when state is present) for parity with the
    # sync loop which returns it via StopIteration.value.
    elapsed = (time.time() - t0) * 1000
    if getattr(config, "build_trace", None) is not None:
        trace = config.build_trace(
            task=task, state=state, session_log=session_log,
            done=stop_reason == "done", llm=llm, llm_calls=llm_calls, elapsed_ms=elapsed,
        )
    else:
        trace = {
            "task": task,
            "steps": [s.to_dict() for s in getattr(state, "steps", [])],
            "llm_calls": llm_calls,
            "total_time_ms": elapsed,
            "conversation": _conv,
        }
    if state is not None:
        setattr(state, "trace", trace)



