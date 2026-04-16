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
import logging
import time
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

from openharness.scaffolding import LLMResult, truncate_tool_result, build_parse_recovery_prompt, PARSE_RECOVERY_MAX
from openharness.types import Step, ToolCall, ToolResult

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
    the event loop.
    """

    def __init__(self, sync_llm: Any) -> None:
        self._sync_llm = sync_llm

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


# ── Async retry ────────────────────────────────────────────────────


async def async_llm_call_with_retry(
    llm: AsyncLLMBackend,
    prompt: str,
    *,
    max_tokens: int = 2000,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_retries: int = 2,
) -> LLMResult:
    """Async version of llm_call_with_retry with asyncio.sleep backoff."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            text = await llm.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
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


# ── Optional imports (same pattern as sync loop.py) ──────────────

try:
    from openharness.router import ModelRouter as _ModelRouter  # noqa: PLC0415
except ImportError:  # pragma: no cover
    _ModelRouter = None  # type: ignore[assignment,misc]

try:
    from openharness.checkpoint import (  # noqa: PLC0415
        Checkpoint as _Checkpoint,
        FileCheckpointStore as _FileCheckpointStore,
        resume_loop_state as _resume_loop_state,
    )
except ImportError:  # pragma: no cover
    _Checkpoint = None  # type: ignore[assignment,misc]
    _FileCheckpointStore = None  # type: ignore[assignment,misc]
    _resume_loop_state = None  # type: ignore[assignment]

try:
    from openharness.telemetry import Tracer as _Tracer  # noqa: PLC0415
except ImportError:  # pragma: no cover
    _Tracer = None  # type: ignore[assignment,misc]

try:
    from openharness.recovery import FailureScenario as _FailureScenario  # noqa: PLC0415
except ImportError:  # pragma: no cover
    _FailureScenario = None  # type: ignore[assignment,misc]

try:
    from openharness.validation import validate_args as _validate_args  # noqa: PLC0415
except ImportError:  # pragma: no cover
    _validate_args = None  # type: ignore[assignment]


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
    from openharness.parse import parse_multi_tool_calls, parse_native_tool_use

    if task is None:
        task = {}
    if tools is None:
        raise ValueError("tools is required")
    if hooks is None:
        hooks = []
    if session_log is None:
        try:
            from openharness.session import SessionLog
            session_log = SessionLog()
        except ImportError:
            session_log = _NullSessionLog()

    # ── Conversation thread — always active (single source of truth) ──
    from openharness.conversation import Conversation as _Conversation, Message as _Message, MessageRole as _MessageRole  # noqa: PLC0415
    _conv = conversation if conversation is not None else _Conversation()

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
            from openharness.streaming import (  # noqa: PLC0415
                LoopStartEvent as _LoopStartEvent,
                StepStartEvent as _StepStartEvent,
                LLMCallStartEvent as _LLMCallStartEvent,
                ToolDispatchEvent as _ToolDispatchEvent,
                LoopEndEvent as _LoopEndEvent,
            )
        except ImportError:
            pass

    # ── Resolve effective LLM (router overrides direct llm) ────
    _router = getattr(config, "router", None)
    _adapter_cache: dict[int, SyncToAsyncAdapter] = {}  # keyed by backend id

    def _get_llm() -> Any:
        if _router is not None:
            backend = _router.select(purpose="reasoning")
            if not asyncio.iscoroutinefunction(getattr(backend, "generate", None)):
                bid = id(backend)
                if bid not in _adapter_cache:
                    _adapter_cache[bid] = SyncToAsyncAdapter(backend)
                return _adapter_cache[bid]
            return backend
        return llm

    # ── Checkpoint store setup ──────────────────────────────────
    _ckpt_store = None
    _checkpoint_dir = getattr(config, "checkpoint_dir", None)
    if _checkpoint_dir is not None and _FileCheckpointStore is not None:
        _ckpt_store = _FileCheckpointStore(_checkpoint_dir)

    # ── Crash-resume from initial checkpoint ───────────────────
    _step_offset = 0
    _initial_checkpoint = getattr(config, "initial_checkpoint", None)
    if _initial_checkpoint is not None and _resume_loop_state is not None:
        resumed = _resume_loop_state(_initial_checkpoint)
        _step_offset = resumed.get("step_offset", 0)
        restored_log = resumed.get("session_log")
        if restored_log is not None:
            session_log.entries = restored_log.entries[:]
            session_log.current_theory = restored_log.current_theory

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
    t0 = time.time()

    # ── Pre-loop hooks ─────────────────────────────────────────
    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            if asyncio.iscoroutinefunction(hook.pre_loop):
                await hook.pre_loop(state, session_log, context)
            else:
                hook.pre_loop(state, session_log, context)

    # ── Emit LoopStartEvent ─────────────────────────────────────
    if stream is not None and _LoopStartEvent is not None:
        stream.emit(_LoopStartEvent(task_summary=str(task.get("id", "")), max_steps=max_steps))

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + _step_offset

        # ── Pre-prompt hooks ───────────────────────────────────
        briefing_base = ""
        if build_briefing_fn is not None:
            briefing_base = build_briefing_fn(state, session_log, context)
        briefing_parts = [briefing_base]

        _briefing_budget = max_briefing_tokens
        _briefing_used = len(briefing_base) // 4 if _briefing_budget else 0

        for hook in hooks:
            if hasattr(hook, "pre_prompt"):
                if asyncio.iscoroutinefunction(hook.pre_prompt):
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
            )
        else:
            prompt = (
                f"Task: {task}\n\n"
                f"{tools.tool_catalog_text()}\n\n"
                f"Step {step_num}/{max_steps}\n\n"
                f"{context_history}\n\n"
                f"{session_log.render()}\n\n"
                f"{chr(10).join(briefing_parts)}"
            )

        # ── Emit StepStartEvent ─────────────────────────────────
        if stream is not None and _StepStartEvent is not None:
            stream.emit(_StepStartEvent(step_num=step_num))

        # Resolve effective LLM once per step (router may change per step)
        effective_llm = _get_llm()

        # ── LLM call with async retry + tracer span ────────────
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
        )
        if _llm_span is not None:
            _tracer.end_span(_llm_span)
        llm_calls += 1

        raw_response = llm_result.text

        # ── Record in conversation thread if enabled ──────────
        if True:  # Conversation always active
            _conv.append(_Message(role=_MessageRole.USER, content=prompt[:5000]))
            if raw_response is not None:
                _conv.append(_Message(
                    role=_MessageRole.ASSISTANT,
                    content=raw_response[:5000] if isinstance(raw_response, str) else str(raw_response)[:5000],
                ))

        if raw_response is None:
            logger.error("Async LLM call failed after retries at step %d", step_num)
            error_call = ToolCall(tool="__llm_error__", reasoning="LLM call failed after retries")
            error_result = ToolResult(
                tool="__llm_error__", args_summary="", data=None,
                error="LLM call failed after all retry attempts",
            )
            step = Step(number=step_num, tool_call=error_call, tool_result=error_result)
            state.steps.append(step)
            yield step
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
            if _recovery_registry is not None and _FailureScenario is not None:
                _recovery_action = _recovery_registry.attempt_recovery(
                    _FailureScenario.PARSE_ERROR,
                    {"step": step_num, "raw_response": raw_response},
                )
                if _recovery_action is not None and _recovery_action.action_type == "abort":
                    logger.warning("Recovery registry aborted parse recovery at step %d", step_num)
                    tool_call = ToolCall(tool="__parse_error__", reasoning=(raw_response or "")[:200])
                    tool_result = ToolResult(
                        tool="__parse_error__", args_summary="", data=None,
                        error=f"Parse error — recovery aborted: {_recovery_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                    state.steps.append(step)
                    yield step
                    continue
                if _recovery_action is not None and _recovery_action.message:
                    post_dispatch_parts.append(_recovery_action.message)
            if consecutive_parse_failures <= PARSE_RECOVERY_MAX:
                logger.warning("Async parse failure %d at step %d — recovery attempt",
                               consecutive_parse_failures, step_num)
                recovery_prompt = build_parse_recovery_prompt(prompt, raw_response)
                recovery_result = await async_llm_call_with_retry(
                    effective_llm, recovery_prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    temperature=recovery_temperature,
                )
                llm_calls += 1
                if recovery_result.ok:
                    tool_calls = parse_multi_tool_calls(recovery_result.text)

            if not tool_calls:
                tool_call = ToolCall(tool="__parse_error__", reasoning=(raw_response or "")[:200])
                tool_result = ToolResult(
                    tool="__parse_error__", args_summary="", data=None,
                    error=f"Could not parse JSON: {(raw_response or '')[:200]}",
                )
                step = Step(number=step_num, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
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
                        if asyncio.iscoroutinefunction(hook.pre_dispatch):
                            cached = await hook.pre_dispatch(state, session_log, tc, step_num + tc_idx)
                        else:
                            cached = hook.pre_dispatch(state, session_log, tc, step_num + tc_idx)
                        if cached is not None:
                            intercepted_results[tc_idx] = cached
                            break

            calls_to_dispatch = [
                (i, tc) for i, tc in enumerate(regular_calls) if i not in intercepted_results
            ]

            # Permission check: hooks can deny tool calls (AND semantics)
            permitted_calls = []
            for i, tc in calls_to_dispatch:
                denied = False
                for hook in hooks:
                    if hasattr(hook, "check_permission"):
                        if asyncio.iscoroutinefunction(hook.check_permission):
                            allowed = await hook.check_permission(tc, state)
                        else:
                            allowed = hook.check_permission(tc, state)
                        if not allowed:
                            intercepted_results[i] = ToolResult(
                                tool=tc.tool, args_summary=str(tc.args)[:100],
                                data=None, error=f"Permission denied for tool '{tc.tool}'",
                            )
                            denied = True
                            break
                if not denied:
                    permitted_calls.append((i, tc))

            # Dispatch: concurrent-safe tools in parallel, others sequential
            dispatched: dict[int, ToolResult] = {}
            concurrent_indices = []
            serial_indices = []

            for i, tc in permitted_calls:
                spec = tools._tools.get(tc.tool) if hasattr(tools, "_tools") else None
                if spec and getattr(spec, "concurrent_safe", False):
                    concurrent_indices.append((i, tc))
                else:
                    serial_indices.append((i, tc))

            if concurrent_indices:
                if hasattr(tools, "dispatch_batch_async"):
                    conc_calls = [tc for _, tc in concurrent_indices]
                    conc_results = await tools.dispatch_batch_async(conc_calls)
                    for (idx, _), result in zip(concurrent_indices, conc_results):
                        dispatched[idx] = result
                elif hasattr(tools, "dispatch_async"):
                    results = await asyncio.gather(*[
                        tools.dispatch_async(tc) for _, tc in concurrent_indices
                    ])
                    for (idx, _), result in zip(concurrent_indices, results):
                        dispatched[idx] = result
                else:
                    # Fallback: sync dispatch
                    for idx, tc in concurrent_indices:
                        dispatched[idx] = tools.dispatch(tc)

            for idx, tc in serial_indices:
                if hasattr(tools, "dispatch_async"):
                    dispatched[idx] = await tools.dispatch_async(tc)
                else:
                    dispatched[idx] = tools.dispatch(tc)

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
                        if asyncio.iscoroutinefunction(hook.post_dispatch):
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
                state.steps.append(step)
                yield step

                # Save checkpoint after each step
                if _ckpt_store is not None and _Checkpoint is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={
                                "entries": session_log.to_list() if hasattr(session_log, "to_list") else [],
                                "current_theory": getattr(session_log, "current_theory", ""),
                            },
                            conversation_data=_conv.serialize(),
                            config_snapshot={"max_steps": max_steps},
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
                    step_findings, step_highlights = extract_step_metadata_fn(state, tc_idx)
                else:
                    step_findings, step_highlights = [], []
                all_step_entities.extend(step_entities)

                recall_key = tool_result.result_key or ""
                theory = tool_call.args.get("__theory__", "")
                session_log.record(
                    step=cur_step, theory=theory, tool=tool_call.tool,
                    reasoning=tool_call.reasoning, entities=step_entities,
                    findings=step_findings, highlights=step_highlights, recall_key=recall_key,
                )

                # Record tool call/result in conversation thread
                if True:  # Conversation always active
                    _conv.append(_Message(
                        role=_MessageRole.ASSISTANT, content="",
                        tool_call=tool_call,
                    ))
                    _conv.append(_Message(
                        role=_MessageRole.TOOL, content="",
                        tool_result=tool_result,
                    ))

        # Handle done() if present
        if done_idx is not None:
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    if asyncio.iscoroutinefunction(hook.check_done):
                        w = await hook.check_done(state, session_log, context, step_num)
                    else:
                        w = hook.check_done(state, session_log, context, step_num)
                    if w is not None:
                        gate_warning = w
                        break

            # Output schema validation — reject done() if payload is invalid
            if gate_warning is None and _output_schema is not None and _validate_args is not None:
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
                session_log.record(
                    step=cur_step, theory="", tool=done_tool_name,
                    reasoning=f"{done_tool_name}() rejected by quality gate",
                )
            else:
                tool_result = tools.dispatch(tool_call)
                step = Step(number=cur_step, tool_call=tool_call, tool_result=tool_result)
                state.steps.append(step)
                yield step
                # Save checkpoint after done step
                if _ckpt_store is not None and _Checkpoint is not None:
                    _ckpt_store.save(
                        _Checkpoint(
                            step_number=cur_step,
                            session_log_data={
                                "entries": session_log.to_list() if hasattr(session_log, "to_list") else [],
                                "current_theory": getattr(session_log, "current_theory", ""),
                            },
                            conversation_data=_conv.serialize(),
                            config_snapshot={"max_steps": max_steps},
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
                if asyncio.iscoroutinefunction(hook.should_stop):
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
    for hook in hooks:
        if hasattr(hook, "on_loop_end"):
            if asyncio.iscoroutinefunction(hook.on_loop_end):
                extra = await hook.on_loop_end(state, session_log, context, llm)
            else:
                extra = hook.on_loop_end(state, session_log, context, llm)
            if isinstance(extra, int):
                llm_calls += extra

    # ── Emit LoopEndEvent ──────────────────────────────────────────
    if stream is not None and _LoopEndEvent is not None:
        stream.emit(_LoopEndEvent(
            total_steps=state.step_count,
            total_llm_calls=llm_calls,
            reason=stop_reason,
        ))


class _NullSessionLog:
    """Minimal session log fallback when cadence.session is not available."""

    def __init__(self) -> None:
        self._entries: list = []

    def render(self) -> str:
        return ""

    def all_entities(self) -> set:
        return set()

    def record(self, **kwargs: Any) -> None:
        pass
