"""Shared Sans-I/O turn-engine state machine for synchronous and asynchronous loops."""

from __future__ import annotations

import copy
import inspect
import logging
import time
from collections.abc import AsyncIterator, Generator
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from looplet.loop import LoopConfig

logger = logging.getLogger(__name__)


# ── Sans-I/O Effect Algebra ──────────────────────────────────────


@dataclass(slots=True)
class _CallHook:
    hook: Any
    slot: str
    args: tuple[Any, ...]
    optional_kwargs: dict[str, Any] | None = None


@dataclass(slots=True)
class _CallDirect:
    fn: Callable[..., Any]
    args: tuple[Any, ...]


@dataclass(slots=True)
class _CallPropose:
    turns: Any
    text: str
    step_num: int


@dataclass(slots=True)
class _RunCompact:
    service: Any
    hooks: list[Any]
    state: Any
    session_log: Any
    llm: Any
    conversation: Any
    step_num: int
    reason: str


@dataclass(slots=True)
class _PreparePrompt:
    kwargs: dict[str, Any]


@dataclass(slots=True)
class _CallLLM:
    llm: Any
    prompt: str
    kwargs: dict[str, Any]


@dataclass(slots=True)
class _DrainStreamExecutor:
    executor: Any


@dataclass(slots=True)
class _SaveCheckpoint:
    fn: Callable[[], Any]


@dataclass(slots=True)
class _DispatchRegularBatch:
    tools: Any
    config: Any
    remaining_calls: list[Any]
    remaining_contexts: list[Any]
    remaining_indices: list[int]
    reused_by_pos: dict[int, Any]
    dispatch_items: list[tuple[int, Any]]
    offset: int
    record_settled_cb: Callable[[int, Any, Any], None]


@dataclass(slots=True)
class _DispatchSingleTool:
    tools: Any
    call: Any
    ctx: Any


@dataclass(slots=True)
class _AppendStep:
    state: Any
    step: Any


@dataclass(slots=True)
class _YieldStep:
    step: Any


def _hooks_allow_speculative_streaming(hooks: list[Any]) -> bool:
    """Allow speculative streaming when hooks do not gate pre-tool execution."""
    if not hooks:
        return True
    from looplet.budget import ThresholdCompactHook
    from looplet.builtin_tools.tool_search import DeferredToolViewHook
    from looplet.cache import CacheBreakDetector
    from looplet.cartridge.hot_reload import ReloadHook
    from looplet.checkpoint import CheckpointHook
    from looplet.presets import _ModelGatewayBinding
    from looplet.provenance import TrajectoryRecorder
    from looplet.runtime import _RuntimeEventHook
    from looplet.session_tree import SessionTreeRecorder
    from looplet.skills import SkillActivationHook
    from looplet.stagnation import StagnationHook
    from looplet.streaming import StreamingHook

    passive_types = (
        StreamingHook,
        TrajectoryRecorder,
        SessionTreeRecorder,
        ThresholdCompactHook,
        CacheBreakDetector,
        StagnationHook,
        ReloadHook,
        SkillActivationHook,
        DeferredToolViewHook,
        _RuntimeEventHook,
        _ModelGatewayBinding,
    )
    for hook in hooks:
        raw = (
            hook.hook
            if hasattr(hook, "hook") and type(hook).__name__ == "_AsyncPersistenceHook"
            else hook
        )
        if isinstance(raw, passive_types):
            continue
        if isinstance(raw, CheckpointHook) or getattr(raw, "checkpoint_owner", False) is True:
            return False
        if (
            hasattr(raw, "pre_dispatch")
            or hasattr(raw, "check_permission")
            or hasattr(raw, "on_event")
        ):
            return False
    return True


@dataclass(frozen=True)
class DomainCallables:
    """Resolved domain hooks shared across sync and async loop drivers."""

    build_briefing: Callable[..., str]
    extract_entities: Callable[[Any], list[str]]
    build_prompt: Any
    extract_step_metadata: Any
    checkpoint_state: Any
    restore_checkpoint_state: Any


def resolve_turn_llm(
    llm: Any,
    config: Any,
    turns: Any,
    loop_ctx: Any,
    *,
    replaying: bool | None = None,
) -> Any:
    """Select the effective turn backend while preserving replay and recording scopes."""
    from looplet.loop import _ScopedLLMProxy
    from looplet.provenance import _has_replay_backend, _recording_backend

    is_replay = _has_replay_backend(llm) if replaying is None else replaying
    if config is not None and config.router is not None and not is_replay:
        selected = config.router.select(purpose="reasoning")
        recorder = _recording_backend(llm)
        if recorder is not None:
            selected = recorder.for_backend(selected)
            if isinstance(llm, _ScopedLLMProxy):
                selected = llm.for_backend(selected)
    else:
        selected = llm
    effective = turns.wrap(selected) if turns is not None else selected
    loop_ctx.llm = effective
    return effective


def resolve_domain_callables(
    config: Any,
    state: Any,
    *,
    default_extract_entities: Callable[..., Any],
    default_build_briefing: Callable[..., Any],
    default_extract_step_metadata: Callable[..., Any],
) -> DomainCallables:
    """Resolve and adapt domain callables identically for sync and async engines."""
    build_briefing = config.resolve_callable("build_briefing", default_build_briefing)
    raw_extract_entities = config.resolve_callable("extract_entities", default_extract_entities)
    try:
        params = inspect.signature(raw_extract_entities).parameters
        takes_state = "state" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        takes_state = False

    if takes_state:

        def extract_entities(data: Any) -> list[str]:
            return raw_extract_entities(data, state=state)
    else:
        extract_entities = raw_extract_entities

    return DomainCallables(
        build_briefing=build_briefing,
        extract_entities=extract_entities,
        build_prompt=config.resolve_callable("build_prompt"),
        extract_step_metadata=config.resolve_callable(
            "extract_step_metadata", default_extract_step_metadata
        ),
        checkpoint_state=config.resolve_callable("checkpoint_state"),
        restore_checkpoint_state=config.resolve_callable("restore_checkpoint_state"),
    )


def restore_checkpoint_into_loop(
    saved_checkpoint: Any,
    *,
    config: LoopConfig,
    llm: Any,
    state: Any,
    session_log: Any,
    active_conv: Any,
    explicit_conversation: Any,
    validate_checkpoint_identity: Callable[[Any, str | None], None],
    resume_loop_state: Callable[..., dict[str, Any]],
    replaying: bool | None = None,
) -> tuple[LoopConfig, int, Any, dict[str, Any] | None]:
    """Validate and apply an initial checkpoint before entering the turn loop."""
    if saved_checkpoint is None:
        return config, 0, active_conv, None

    from looplet.assistant_turns import TURN_STATE_KEY, restore_assistant_checkpoint
    from looplet.provenance import _has_replay_backend
    from looplet.types import RunEnvelope

    is_replay = _has_replay_backend(llm) if replaying is None else replaying
    validate_checkpoint_identity(
        saved_checkpoint,
        config.run_envelope.run_id if config.run_envelope is not None else None,
    )
    resumed = resume_loop_state(saved_checkpoint, max_steps=config.max_steps)
    restore_backend = (
        config.router.select(purpose="reasoning")
        if config.router is not None and not is_replay
        else llm
    )
    restore_backend_state = getattr(restore_backend, "restore_checkpoint_state", None)
    if callable(restore_backend_state):
        restore_backend_state(copy.deepcopy((resumed.get("domain_state") or {}).get("llm", {})))
    if config.run_envelope is None and isinstance(resumed.get("run_envelope"), dict):
        config = replace(
            config,
            run_envelope=RunEnvelope.from_dict(resumed["run_envelope"]),
        )
    step_offset = resumed.get("step_offset", 0)
    restored_log = resumed.get("session_log")
    if restored_log is not None:
        from looplet._storage import TrackedList

        session_log.entries = TrackedList(restored_log.entries)
        session_log.current_theory = restored_log.current_theory
    restored_conv = resumed.get("conversation")
    if restored_conv is not None and explicit_conversation is None:
        active_conv = restored_conv
    for key, value in (resumed.get("state_counters") or {}).items():
        try:
            setattr(state, key, value)
        except AttributeError:
            pass

    config = restore_assistant_checkpoint(state, config, saved_checkpoint)
    if TURN_STATE_KEY in saved_checkpoint.metadata:
        step_offset = saved_checkpoint.step_number - state.step_count
    return config, step_offset, active_conv, resumed


def bind_state_attributes(state: Any, *, task: Any, conversation: Any, run_envelope: Any) -> None:
    """Attach standard runtime handles onto state when mutable."""
    for attr_name, attr_val in (
        ("task", task),
        ("conversation", conversation),
        ("run_envelope", run_envelope),
    ):
        try:
            setattr(state, attr_name, attr_val)
        except AttributeError:
            pass


def finalize_checkpoint_restore(
    saved_checkpoint: Any,
    resumed: dict[str, Any] | None,
    *,
    loop_ctx: Any,
    tools: Any,
    restore_checkpoint_state: Callable[..., Any] | None,
    finalizer: Any,
    state: Any,
    session_log: Any,
    context: Any,
    llm: Any,
    set_run_lifecycle: Callable[..., None],
) -> bool:
    """Restore domain/tool stores and return True if a terminal checkpoint short-circuits."""
    if saved_checkpoint is None or resumed is None:
        return False
    if restore_checkpoint_state is not None:
        restore_checkpoint_state(loop_ctx, resumed.get("domain_state", {}))
    tools.restore_results(resumed.get("tool_results_store") or {})
    if saved_checkpoint.is_terminal:
        from looplet.assistant_turns import TURN_STATE_KEY
        from looplet.types import RunPhase, RunStatus

        if TURN_STATE_KEY in saved_checkpoint.metadata:
            if saved_checkpoint.run_status != "completed":
                raise ValueError("stopped assistant runs require an explicit new run identity")
            set_run_lifecycle(
                loop_ctx,
                status=RunStatus.COMPLETED,
                phase=RunPhase.TERMINAL,
                termination_reason=saved_checkpoint.termination_reason,
                terminal_tool=saved_checkpoint.metadata.get("accepted_terminal_tool"),
            )
            finalizer.bind([], state, session_log, context, llm)
            return True
    return False


def build_pending_dispatch_kwargs(
    loop_ctx: Any,
    *,
    calls: list[Any],
    dispatching: list[Any],
    step_number: int,
    task: Any,
    checkpoint_state: Callable[..., Any] | None,
    ckpt_store: Any,
    policy_checkpoint_metadata: Callable[[Any, Any], dict[str, Any]],
    include_tool_results: bool = True,
) -> dict[str, Any]:
    """Construct identical pending-dispatch checkpoint arguments for sync and async engines."""
    state = loop_ctx.state
    config = loop_ctx.config
    return {
        "calls": calls,
        "dispatching": dispatching,
        "step_number": step_number,
        "session_log": loop_ctx.session_log,
        "conversation_data": loop_ctx.conversation.serialize(),
        "config_snapshot": {
            "max_steps": config.max_steps,
            "queries_used": getattr(state, "queries_used", 0),
            "budget_remaining": getattr(state, "budget_remaining", 0),
        },
        "tool_results_store": loop_ctx.tools.checkpoint_results() if include_tool_results else {},
        "domain_state": checkpoint_state(loop_ctx) if checkpoint_state is not None else {},
        "run_envelope": (
            loop_ctx.run_envelope.to_dict() if loop_ctx.run_envelope is not None else None
        ),
        "metadata": {
            "task": str(task),
            **policy_checkpoint_metadata(state, ckpt_store),
        },
        "run_status": loop_ctx.status.value,
        "run_phase": loop_ctx.phase.value,
    }


def clear_recovery_flags(recovery_state: dict[str, Any]) -> None:
    """Reset exhausted strategy markers after a successful reactive recovery."""
    for strategy_name in ("budget_enforcement", "emergency_truncate", "result_clearing"):
        recovery_state.pop(strategy_name, None)


def commit_step_to_history(
    *,
    step: Any,
    cur_step: int,
    state: Any,
    history: Any,
    extract_entities: Callable[[Any], list[str]],
    extract_step_metadata: Callable[[Any, int], tuple[list[str], list[str]]],
    all_step_entities: list[str],
    already_appended: bool = False,
) -> list[str]:
    """Atomically commit a step to state.steps, SessionLog, and Conversation.

    Both sync and async drivers pass the canonical 1-based ``cur_step`` to
    ``extract_step_metadata``. If extraction or history recording raises,
    ``state.steps`` is rolled back so state views never diverge.
    """
    if not already_appended:
        state.steps.append(step)
    try:
        step_entities = extract_entities(step.tool_result.data)
        step_findings, step_highlights = extract_step_metadata(state, cur_step)
        history.record_step(
            step,
            theory=step.tool_call.args.get("__theory__", ""),
            entities=step_entities,
            findings=step_findings,
            highlights=step_highlights,
            recall_key=step.tool_result.result_key or "",
        )
    except BaseException:
        if state.steps and state.steps[-1] is step:
            del state.steps[-1]
        raise
    all_step_entities.extend(step_entities)
    return step_entities


# ── Sans-I/O Hook & Turn Sub-Generators ──────────────────────────


def _emit_event_gen(
    hooks: list[Any],
    event: Any,
    *,
    fail_closed: bool = False,
    _audit_hooks: list[Any] | None = None,
    **payload_kwargs: Any,
) -> Generator[Any, Any, list[Any]]:
    from looplet.events import EventPayload
    from looplet.events import LifecycleEvent as _LE
    from looplet.hook_decision import HookDecision, normalize_hook_return
    from looplet.loop import _event_hook_callbacks, _record_policy_decision

    state = payload_kwargs.get("state")
    if state is not None:
        from looplet.context_projection import model_turn_context

        for name, value in model_turn_context(state).items():
            payload_kwargs.setdefault(name, value)
        payload_kwargs.setdefault("run_status", getattr(state, "run_status", None))
        payload_kwargs.setdefault("run_phase", getattr(state, "run_phase", None))
        envelope = getattr(state, "run_envelope", None)
        if envelope is not None:
            payload_kwargs.setdefault(
                "run_envelope",
                envelope.to_dict() if hasattr(envelope, "to_dict") else envelope,
            )
    payload = EventPayload(event=event, **payload_kwargs)
    collected: list[Any] = []
    for hook, fn in _event_hook_callbacks(hooks, event):
        try:
            result = yield _CallDirect(fn, (payload,))
        except Exception:  # noqa: BLE001
            if fail_closed:
                raise
            logger.exception("on_event hook raised; continuing")
            continue
        if isinstance(result, HookDecision):
            decision = normalize_hook_return(result, slot=getattr(event, "value", str(event)))
            if event != _LE.HOOK_DECISION and not result.is_noop():
                audit_targets = hooks if _audit_hooks is None else _audit_hooks
                _record_policy_decision(payload.state, result)
                event_extra = {
                    "decision": asdict(result),
                    "originating_event": getattr(event, "value", str(event)),
                }
                yield from _emit_event_gen(
                    audit_targets,
                    _LE.HOOK_DECISION,
                    step_num=payload.step_num,
                    state=payload.state,
                    session_log=payload.session_log,
                    context=payload.context,
                    hook_slot="on_event",
                    hook_name=type(hook).__name__,
                    policy_decision=getattr(result, "policy_decision", None),
                    extra=event_extra,
                )
            collected.append(decision)
    return collected


def _emit_hook_decision_gen(
    hooks: list[Any],
    *,
    decision: Any,
    hook_slot: str,
    hook_name: str,
    step_num: int = 0,
    state: Any = None,
    session_log: Any = None,
    context: Any = None,
    extra: dict[str, Any] | None = None,
) -> Generator[Any, Any, None]:
    if decision is None or decision.is_noop():
        return
    from looplet.events import LifecycleEvent as _LE
    from looplet.loop import _record_policy_decision

    _record_policy_decision(state, decision)
    event_extra = {"decision": asdict(decision)}
    if extra:
        event_extra.update(extra)
    yield from _emit_event_gen(
        hooks,
        _LE.HOOK_DECISION,
        step_num=step_num,
        state=state,
        session_log=session_log,
        context=context,
        hook_slot=hook_slot,
        hook_name=hook_name,
        policy_decision=getattr(decision, "policy_decision", None),
        extra=event_extra,
    )


def _intercept_tool_calls_gen(
    calls: list[Any],
    hooks: list[Any],
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
) -> Generator[Any, Any, Any]:
    from looplet.events import LifecycleEvent as _LE
    from looplet.hook_decision import normalize_hook_return
    from looplet.loop import _InterceptResult
    from looplet.tools import _summarize_args_dict
    from looplet.types import ErrorKind, ToolError, ToolResult

    result = _InterceptResult()
    for tc_idx, tc in enumerate(calls):
        cur_step = step_num + tc_idx
        denied = False
        for hook in hooks:
            if hasattr(hook, "pre_dispatch"):
                raw = yield _CallHook(hook, "pre_dispatch", (state, session_log, tc, cur_step))
                decision = normalize_hook_return(raw, slot="pre_dispatch")
                if decision is None:
                    continue
                yield from _emit_hook_decision_gen(
                    hooks,
                    decision=decision,
                    hook_slot="pre_dispatch",
                    hook_name=type(hook).__name__,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
                decisions = [decision]
            else:
                decisions = yield from _emit_event_gen(
                    [hook],
                    _LE.PRE_TOOL_USE,
                    fail_closed=True,
                    _audit_hooks=hooks,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                    tool_call=tc,
                )
            for decision in decisions:
                if decision.updated_args is not None:
                    tc.args = decision.updated_args
                if decision.permission == "deny":
                    reason = decision.block or ""
                    message = f"Permission denied for tool '{tc.tool}'"
                    if reason:
                        message += f": {reason}"
                    error = ToolError(
                        kind=ErrorKind.PERMISSION_DENIED,
                        message=message,
                        retriable=False,
                    )
                    result.intercepted[tc_idx] = ToolResult(
                        tool=tc.tool,
                        args_summary=_summarize_args_dict(tc.args),
                        data=None,
                        error=message,
                        error_detail=error,
                    )
                    denied = True
                    break
                if decision.updated_result is not None:
                    result.intercepted[tc_idx] = decision.updated_result
                    break
                if decision.additional_context:
                    result.extra_context.append(decision.additional_context)
            if denied:
                break
        if denied:
            continue

        for hook in hooks:
            if not hasattr(hook, "check_permission"):
                continue
            raw = yield _CallHook(hook, "check_permission", (tc, state))
            decision = normalize_hook_return(raw, slot="check_permission")
            if decision is not None:
                yield from _emit_hook_decision_gen(
                    hooks,
                    decision=decision,
                    hook_slot="check_permission",
                    hook_name=type(hook).__name__,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
            if decision is not None and decision.permission == "deny":
                reason = decision.block or ""
                message = f"Permission denied for tool '{tc.tool}'"
                if reason:
                    message += f": {reason}"
                error = ToolError(
                    kind=ErrorKind.PERMISSION_DENIED,
                    message=message,
                    retriable=False,
                )
                result.intercepted[tc_idx] = ToolResult(
                    tool=tc.tool,
                    args_summary=_summarize_args_dict(tc.args),
                    data=None,
                    error=message,
                    error_detail=error,
                )
                break
    return result


def _run_post_dispatch_hooks_gen(
    tool_call: Any,
    tool_result: Any,
    hooks: list[Any],
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
    *,
    emit_lifecycle: bool = True,
) -> Generator[Any, Any, Any]:
    from looplet.events import LifecycleEvent as _LE
    from looplet.hook_decision import normalize_hook_return
    from looplet.loop import _PostDispatchOutcome

    outcome = _PostDispatchOutcome(tool_result=tool_result)
    for hook in hooks:
        if hasattr(hook, "post_dispatch"):
            raw = yield _CallHook(
                hook, "post_dispatch", (state, session_log, tool_call, tool_result, step_num)
            )
            decision = normalize_hook_return(raw, slot="post_dispatch")
            if decision is None:
                continue
            yield from _emit_hook_decision_gen(
                hooks,
                decision=decision,
                hook_slot="post_dispatch",
                hook_name=type(hook).__name__,
                step_num=step_num,
                state=state,
                session_log=session_log,
                context=context,
            )
            decisions = [decision]
        elif emit_lifecycle:
            event = _LE.POST_TOOL_FAILURE if tool_result.error else _LE.POST_TOOL_USE
            decisions = yield from _emit_event_gen(
                [hook],
                event,
                _audit_hooks=hooks,
                step_num=step_num,
                state=state,
                session_log=session_log,
                context=context,
                tool_call=tool_call,
                tool_result=tool_result,
            )
        else:
            continue
        for decision in decisions:
            if decision.updated_result is not None:
                outcome.tool_result = decision.updated_result
                tool_result = outcome.tool_result
            if decision.additional_context:
                outcome.extra_context.append(decision.additional_context)
            if decision.stop is not None:
                outcome.stop_reason = decision.stop
    return outcome


def _notify_post_step_gen(
    hooks: list[Any], state: Any, session_log: Any, step_num: int
) -> Generator[Any, Any, None]:
    for hook in hooks:
        method = getattr(hook, "post_step", None)
        if method is not None:
            yield _CallDirect(method, (state, session_log, step_num))


def _post_step_stop_reason_gen(
    hooks: list[Any],
    *,
    state: Any,
    session_log: Any,
    context: Any,
    step_num: int,
    new_entities: int,
    requested_reason: str | None,
) -> Generator[Any, Any, str | None]:
    from looplet.hook_decision import normalize_hook_return

    if requested_reason is not None:
        return requested_reason
    for hook in hooks:
        if hasattr(hook, "should_stop"):
            raw = yield _CallHook(hook, "should_stop", (state, step_num, new_entities))
            decision = normalize_hook_return(raw, slot="should_stop")
            if decision is not None:
                yield from _emit_hook_decision_gen(
                    hooks,
                    decision=decision,
                    hook_slot="should_stop",
                    hook_name=type(hook).__name__,
                    step_num=step_num,
                    state=state,
                    session_log=session_log,
                    context=context,
                )
                if decision.is_stop():
                    return decision.stop or "hook_stop"
    return None


def _recover_prompt_gen(
    llm_result: Any,
    recovery_state: dict[str, Any],
    *,
    llm: Any,
    state: Any,
    session_log: Any,
    tools: Any,
    config: Any,
    step_num: int,
    hooks: list[Any],
    conversation: Any,
    native_policy: Any,
    prepare_kwargs_fn: Callable[[], dict[str, Any]],
    turns: Any,
    loop_ctx: Any,
    compact_service: Any,
    should_continue: Callable[[], bool],
) -> Generator[Any, Any, tuple[Any, int]]:
    from looplet.compact import TruncateCompact
    from looplet.recovery_strategies import recovery_aggressive_budget, recovery_clear_old_results

    extra_calls = 0
    for name in ("budget_enforcement", "emergency_truncate", "result_clearing"):
        if recovery_state.get(name):
            continue
        recovery_state[name] = True
        logger.warning("Recovery strategy '%s' at step %d", name, step_num)
        if name == "emergency_truncate":
            service = (
                config.compact_service if config.compact_service is not None else compact_service
            )
            outcome = yield _RunCompact(
                service if service is not None else TruncateCompact(),
                hooks,
                state,
                session_log,
                llm,
                conversation,
                step_num,
                "prompt_too_long",
            )
            extra_calls += outcome.llm_calls_spent
        else:
            strategy = (
                recovery_aggressive_budget
                if name == "budget_enforcement"
                else recovery_clear_old_results
            )
            extra_calls += strategy(state, session_log, llm, step_num)
        prepared = yield _PreparePrompt(prepare_kwargs_fn())
        loop_ctx.context_budget = prepared.budget
        if (
            turns is not None
            and config.uses_typed_history
            and config.render_messages_override is None
        ):
            llm.prepare_conversation(
                conversation,
                prepared.conversation_prompt,
                request_prompt=prepared.prompt,
                turn_update=prepared.turn_update,
            )
        if prepared.budget.pressure:
            continue
        use_schemas = tools is not None and native_policy.enabled and not native_policy.demoted
        previous = llm_result
        retry_result = yield _CallLLM(
            llm,
            prepared.prompt,
            {
                "max_tokens": config.max_tokens,
                "system_prompt": config.system_prompt,
                "temperature": config.temperature,
                "tools": list(prepared.tool_view.schemas) if use_schemas else None,
                "native_policy": native_policy,
                "cancel_token": config.cancel_token,
                "max_continuations": config.max_turn_continuations,
                "generate_kwargs": config.generate_kwargs or None,
                "should_continue": should_continue,
            },
        )
        loop_ctx.native_tool_stats.record(retry_result)
        extra_calls += retry_result.calls + (
            0 if retry_result.native_attempted else retry_result.continuations
        )
        retry_result._include_previous(previous)
        llm_result = retry_result
        if retry_result.ok:
            clear_recovery_flags(recovery_state)
            recovery_state["_last_recovery_llm_calls"] = extra_calls
            return llm_result, extra_calls
        if not retry_result.is_prompt_too_long:
            break
    logger.error("All recovery strategies exhausted at step %d", step_num)
    recovery_state["_last_recovery_llm_calls"] = extra_calls
    return llm_result, extra_calls


def run_turn_machine(
    *,
    llm: Any,
    task: Any,
    tools: Any,
    context: Any,
    hooks: list[Any],
    config: LoopConfig,
    state: Any,
    session_log: Any,
    stream: Any,
    explicit_conversation: Any,
    finalizer: Any,
    checkpoint_store: Any,
    initial_checkpoint: Any,
    compact_service: Any,
    make_tool_ctx: Callable[[Any, int, Any], Any],
    make_stream_executor: Callable[[], Any],
    on_turn_llm_selected: Callable[[Any], None] | None = None,
    build_final_trace: bool = True,
) -> Generator[Any, Any, Any]:
    """Unified Sans-I/O turn machine shared by ``composable_loop`` and ``async_composable_loop``."""
    from looplet.assistant_turns import _AdmissionDenied, _AssistantTurns
    from looplet.checkpoint import (
        _capture_checkpoint,
        _pending_dispatch_checkpoint,
        resume_loop_state,
        validate_checkpoint_identity,
    )
    from looplet.context_plan import ContextSourceSelector
    from looplet.conversation import Conversation
    from looplet.events import LifecycleEvent as _LE
    from looplet.history import HistoryRecorder
    from looplet.hook_decision import HookDecision, normalize_hook_return
    from looplet.loop import (
        LoopContext,
        _accumulate_step_usage,
        _completion_gate_warning,
        _count_usage_calls,
        _deadline_expired,
        _default_build_briefing,
        _default_extract_entities,
        _default_extract_step_metadata,
        _failed_model_step,
        _get_persist_threshold,
        _normalize_done_rejection,
        _policy_checkpoint_metadata,
        _prepare_tool_turn,
        _record_run_stats,
        _rejected_terminal_step,
        _set_run_lifecycle,
        _step_metadata,
        _terminal_step_metadata,
        _trace_steps,
    )
    from looplet.native_tools import NativeToolPolicy
    from looplet.parse import parse_multi_tool_calls, to_text
    from looplet.recovery import FailureScenario
    from looplet.result_projection import project_tool_result
    from looplet.scaffolding import (
        PARSE_RECOVERY_MAX,
        ContextOverflowError,
        LLMResult,
        build_parse_recovery_prompt,
    )
    from looplet.streaming import StreamingHook
    from looplet.tools import _summarize_args_dict
    from looplet.types import RunPhase, RunStatus, Step, ToolCall, ToolResult

    assert tools is not None
    assert state is not None
    t0 = time.time()
    perf_t0 = time.perf_counter()
    active_conv = explicit_conversation if explicit_conversation is not None else Conversation()

    _LoopStartEvent = _StepStartEvent = _LLMCallStartEvent = None
    _ToolDispatchEvent = _LoopEndEvent = None
    _LLMCallEndEvent = _ToolResultEvent = _StepEndEvent = None
    if stream is not None:
        try:
            from looplet.streaming import (
                LLMCallEndEvent as _LLMCallEndEvent,
            )
            from looplet.streaming import (
                LLMCallStartEvent as _LLMCallStartEvent,
            )
            from looplet.streaming import (
                LoopEndEvent as _LoopEndEvent,
            )
            from looplet.streaming import (
                LoopStartEvent as _LoopStartEvent,
            )
            from looplet.streaming import (
                StepEndEvent as _StepEndEvent,
            )
            from looplet.streaming import (
                StepStartEvent as _StepStartEvent,
            )
            from looplet.streaming import (
                ToolDispatchEvent as _ToolDispatchEvent,
            )
            from looplet.streaming import (
                ToolResultEvent as _ToolResultEvent,
            )
        except ImportError:
            pass

    from looplet.provenance import _has_replay_backend

    replaying = _has_replay_backend(llm)
    config, step_offset, active_conv, resumed = restore_checkpoint_into_loop(
        initial_checkpoint,
        config=config,
        llm=llm,
        state=state,
        session_log=session_log,
        active_conv=active_conv,
        explicit_conversation=explicit_conversation,
        validate_checkpoint_identity=validate_checkpoint_identity,
        resume_loop_state=resume_loop_state,
        replaying=replaying,
    )
    bind_state_attributes(
        state, task=task, conversation=active_conv, run_envelope=config.run_envelope
    )

    history = HistoryRecorder(
        state=state,
        session_log=session_log,
        conversation=active_conv,
        typed_turns=config.uses_typed_history,
        observe_tool_result=getattr(llm, "record_tool_receipt", None),
        project_tool_result=lambda result: project_tool_result(
            result,
            persist_dir=config.tool_result_persist_dir,
            persist_threshold=_get_persist_threshold(),
        ),
    )

    domain = resolve_domain_callables(
        config,
        state,
        default_extract_entities=_default_extract_entities,
        default_build_briefing=_default_build_briefing,
        default_extract_step_metadata=_default_extract_step_metadata,
    )
    build_briefing = domain.build_briefing
    extract_entities = domain.extract_entities
    build_prompt_fn = domain.build_prompt
    checkpoint_state = domain.checkpoint_state
    restore_checkpoint_state = domain.restore_checkpoint_state
    extract_step_metadata = domain.extract_step_metadata

    loop_ctx = LoopContext(
        state=state,
        session_log=session_log,
        conversation=active_conv,
        tools=tools,
        config=config,
        resources=tools.resources,
        step_num=0,
        run_envelope=config.run_envelope,
    )
    for hook in hooks:
        bind_fn = getattr(hook, "bind", None)
        if bind_fn is not None:
            try:
                yield _CallDirect(bind_fn, (loop_ctx,))
            except (TypeError, ValueError):
                continue

    _set_run_lifecycle(loop_ctx, status=RunStatus.RUNNING, phase=RunPhase.STARTING)
    loop_ctx.step_num = step_offset
    if finalize_checkpoint_restore(
        initial_checkpoint,
        resumed,
        loop_ctx=loop_ctx,
        tools=tools,
        restore_checkpoint_state=restore_checkpoint_state,
        finalizer=finalizer,
        state=state,
        session_log=session_log,
        context=context,
        llm=llm,
        set_run_lifecycle=_set_run_lifecycle,
    ):
        return None

    turns = (
        _AssistantTurns(state, config)
        if config.max_model_calls is not None or config.uses_typed_history
        else None
    )
    if turns is not None:
        checkpoint_state = turns.wrap_checkpoint(
            checkpoint_state,
            capture_backend_state=getattr(checkpoint_store, "checkpoint_owner", False) is not True,
        )
        turns.checkpoint = lambda: turns.save(checkpoint_store, loop_ctx, checkpoint_state)
        if not build_final_trace:
            turns.async_checkpoint = lambda: turns.save_async(
                checkpoint_store, loop_ctx, checkpoint_state
            )
        finalizer.bind(hooks, state, session_log, context, turns.wrap(llm))
        loop_ctx.llm = turns.wrap(llm)

    try:
        setattr(state, "step_context", {})  # noqa: B010
    except AttributeError:
        pass

    for hook in hooks:
        if hasattr(hook, "pre_loop"):
            yield _CallHook(
                hook,
                "pre_loop",
                (state, session_log, context),
                optional_kwargs={"tools": tools},
            )

    yield from _emit_event_gen(
        hooks,
        _LE.SESSION_START,
        state=state,
        session_log=session_log,
        context=context,
    )

    has_streaming_hook = any(isinstance(h, StreamingHook) for h in hooks)
    if stream is not None and _LoopStartEvent is not None and not has_streaming_hook:
        task_id = task.get("id", "") if isinstance(task, dict) else str(task)[:80]
        stream.emit(_LoopStartEvent(task_summary=str(task_id), max_steps=config.max_steps))

    context_sources = (
        ContextSourceSelector(
            config.scoped_context_sources, budget_tokens=config.scoped_context_budget_tokens
        )
        if config.scoped_context_sources
        else None
    )

    def _get_llm() -> Any:
        return resolve_turn_llm(llm, config, turns, loop_ctx, replaying=replaying)

    def _save_step_checkpoint(
        step_number: int, *, status: str | None = None
    ) -> Generator[Any, Any, None]:
        if checkpoint_store is None:
            return
        loop_ctx.step_num = step_number
        meta: dict[str, Any] = {"task": str(task)}
        if status is not None:
            meta["status"] = status
        cp = _capture_checkpoint(
            loop_ctx,
            step_number=step_number,
            checkpoint_state=checkpoint_state,
            metadata={**meta, **_policy_checkpoint_metadata(state, checkpoint_store)},
            include_tool_results=False,
        )
        key = f"step_{step_number}" if status is None else f"step_{step_number}_{status}"

        def _persist_step_cp() -> None:
            checkpoint_store.save(
                replace(cp, tool_results_store=loop_ctx.tools.checkpoint_results()),
                key=key,
            )

        yield _SaveCheckpoint(_persist_step_cp)

    def _save_turns_checkpoint() -> Generator[Any, Any, None]:
        if turns is not None and checkpoint_store is not None and turns.persistent_admission:
            cp = turns.capture(
                loop_ctx,
                checkpoint_state,
                checkpoint_store=checkpoint_store,
                include_tool_results=False,
            )
            yield _SaveCheckpoint(
                lambda: checkpoint_store.save(
                    replace(cp, tool_results_store=loop_ctx.tools.checkpoint_results()),
                    "model_turn",
                )
            )

    consecutive_parse_failures = 0
    native_policy = NativeToolPolicy(enabled=config.use_native_tools)
    quality_gate_message = ""
    post_dispatch_parts: list[str] = []
    llm_calls = 0
    usage_calls = 0
    done = False
    stop_reason = "budget_exhausted"
    llm_failed = False
    recovery_state: dict[str, Any] = {
        "budget_enforcement": False,
        "emergency_truncate": False,
        "result_clearing": False,
    }
    allow_speculative = (
        config.streaming_tool_execution
        and checkpoint_store is None
        and _hooks_allow_speculative_streaming(hooks)
    )

    while state.budget_remaining > 0 and not done:
        step_num = state.step_count + 1 + step_offset
        loop_ctx.step_num = step_num
        loop_ctx.conversation = active_conv
        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break
        _set_run_lifecycle(loop_ctx, phase=RunPhase.PROMPTING)
        hook_requested_stop = False

        try:
            setattr(state, "step_context", {})  # noqa: B010
        except AttributeError:
            pass

        if config.cancel_token is not None and getattr(config.cancel_token, "is_cancelled", False):
            stop_reason = "cancelled"
            break

        if turns is not None:
            if turns.stop_reason is not None:
                stop_reason = turns.stop_reason
                break
            turns.begin_turn()

        want_compact = False
        for hook in hooks:
            sc_method = getattr(hook, "should_compact", None)
            if sc_method is None:
                continue
            try:
                should_c = yield _CallDirect(sc_method, (state, session_log, active_conv, step_num))
            except Exception:  # noqa: BLE001
                logger.exception("should_compact hook raised; skipping")
                should_c = False
            if should_c:
                want_compact = True
                break
        eff_compact_service = (
            config.compact_service if config.compact_service is not None else compact_service
        )
        if want_compact and eff_compact_service is not None:
            yield _RunCompact(
                eff_compact_service,
                hooks,
                state,
                session_log,
                _get_llm(),
                active_conv,
                step_num,
                "proactive",
            )

        briefing_text: str | None = None
        for hook in hooks:
            bb_method = getattr(hook, "build_briefing", None)
            if bb_method is None:
                continue
            try:
                candidate = yield _CallDirect(bb_method, (state, session_log, context))
            except Exception:  # noqa: BLE001
                logger.exception("build_briefing hook raised; falling back")
                candidate = None
            if candidate is not None:
                briefing_text = str(candidate)
                break
        if briefing_text is None:
            raw_bb = yield _CallDirect(build_briefing, (state, session_log, context))
            briefing_text = str(raw_bb)

        briefing_parts: list[str] = [briefing_text]
        briefing_budget = config.max_briefing_tokens
        briefing_used = len(briefing_parts[0]) // 4 if briefing_budget else 0

        for hook in hooks:
            if hasattr(hook, "pre_prompt"):
                raw_pp = yield _CallHook(
                    hook, "pre_prompt", (state, session_log, context, step_num)
                )
                pp_decision = normalize_hook_return(raw_pp, slot="pre_prompt")
                if pp_decision is not None:
                    yield from _emit_hook_decision_gen(
                        hooks,
                        decision=pp_decision,
                        hook_slot="pre_prompt",
                        hook_name=type(hook).__name__,
                        step_num=step_num,
                        state=state,
                        session_log=session_log,
                        context=context,
                    )
                pp_text = (
                    pp_decision.additional_context
                    if pp_decision is not None
                    else (raw_pp if isinstance(raw_pp, str) else None)
                )
                if pp_text:
                    if briefing_budget:
                        text_tokens = len(pp_text) // 4
                        if briefing_used + text_tokens > briefing_budget:
                            briefing_parts.append("(briefing truncated - token budget exceeded)")
                            break
                        briefing_used += text_tokens
                    briefing_parts.append(str(pp_text))

        if post_dispatch_parts:
            briefing_parts.append("\n".join(post_dispatch_parts))
            post_dispatch_parts.clear()

        effective_llm = _get_llm()
        if on_turn_llm_selected is not None:
            on_turn_llm_selected(effective_llm)

        typed_turns = config.uses_typed_history and config.render_messages_override is None
        omit_text_history = (
            typed_turns
            and config.use_native_tools
            and getattr(effective_llm, "conversation_mode", "text") in {"messages", "stateful"}
            and callable(getattr(effective_llm, "set_conversation_input", None))
            and callable(getattr(effective_llm, "generate_with_tools", None))
        )

        context_history = "" if typed_turns else state.context_summary()
        if quality_gate_message:
            if turns is None or not turns.enabled:
                context_history = (
                    (context_history + "\n" + quality_gate_message)
                    if context_history
                    else quality_gate_message
                )
            quality_gate_message = ""

        briefing_str = "\n".join(briefing_parts)
        prepared = yield _PreparePrompt(
            {
                "config": config,
                "state": state,
                "tools": tools,
                "session_log": session_log,
                "task": task,
                "step_num": step_num,
                "briefing": briefing_str,
                "context_sources": context_sources,
                "context_history": None
                if (typed_turns and not context_history)
                else context_history,
                "hooks": hooks,
                "builder": build_prompt_fn,
                "conversation": active_conv,
                "omit_text_history": omit_text_history,
            }
        )
        prompt = prepared.prompt
        tool_view = prepared.tool_view
        rendered_memory = prepared.rendered_memory
        scoped_plan = prepared.scoped_context_plan
        loop_ctx.context_budget = prepared.budget
        estimated_tokens = prepared.budget.estimated_tokens
        preflight_too_long = prepared.budget.pressure
        state_metadata = getattr(state, "metadata", None)
        if loop_ctx.context_budget.pressure and isinstance(state_metadata, dict):
            state_metadata["context_budget"] = loop_ctx.context_budget.to_dict()

        if stream is not None and _StepStartEvent is not None and turns is None:
            stream.emit(_StepStartEvent(step_num=step_num))

        if turns is not None and typed_turns:
            effective_llm.prepare_conversation(
                active_conv,
                prepared.conversation_prompt,
                request_prompt=prompt,
                turn_update=prepared.turn_update,
            )

        pre_llm_decisions = yield from _emit_event_gen(
            hooks,
            _LE.PRE_LLM_CALL,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
            context_budget=loop_ctx.context_budget.to_dict(),
        )
        for d in pre_llm_decisions:
            if d.additional_context:
                post_dispatch_parts.append(d.additional_context)
            if d.stop is not None:
                stop_reason = d.stop
                hook_requested_stop = True

        if turns is not None and turns.persistent_admission and hook_requested_stop:
            break
        if _deadline_expired(loop_ctx):
            stop_reason = "deadline_exceeded"
            done = True
            break

        # Refresh effective_llm in case PRE_LLM_CALL or router selection updated state
        effective_llm = _get_llm()
        if turns is not None and typed_turns:
            effective_llm.prepare_conversation(
                active_conv,
                prepared.conversation_prompt,
                request_prompt=prompt,
                turn_update=prepared.turn_update,
            )
        if on_turn_llm_selected is not None:
            on_turn_llm_selected(effective_llm)

        turn_usage_calls = 0
        discarded_stream_receipts: list[dict[str, Any]] = []
        stream_executor: Any = None
        stream_consumed = False

        def _drain_unused_stream_executor() -> Generator[Any, Any, list[dict[str, Any]]]:
            nonlocal stream_consumed
            if stream_executor is None or stream_consumed:
                return []
            stream_consumed = True
            stream_executor.discard()
            raw_res = yield _DrainStreamExecutor(stream_executor)
            receipts = [
                r.to_dict() for r in raw_res if r.metadata.get("execution_status") != "not_executed"
            ]
            s_meta = getattr(state, "metadata", None)
            if receipts and isinstance(s_meta, dict):
                s_meta.setdefault("speculative_tool_receipts", []).extend(receipts)
            return receipts

        if preflight_too_long:
            logger.warning(
                "Pre-flight block: prompt ~%d tokens exceeds safe limit - %s",
                estimated_tokens,
                "running recovery before LLM call"
                if config.reactive_recovery
                else "stopping before LLM call",
            )
            llm_result = LLMResult(
                None,
                ContextOverflowError(
                    f"prompt estimate {estimated_tokens} tokens exceeds safe context "
                    f"limit {config.context_window - 3_000}"
                ),
            )
        else:
            if (
                stream is not None
                and _LLMCallStartEvent is not None
                and (turns is None or not has_streaming_hook)
            ):
                stream.emit(
                    _LLMCallStartEvent(
                        step_num=step_num,
                        model_turn_id=turns.turn_id if turns is not None else None,
                    )
                )
            llm_span = (
                config.tracer.start_span(f"llm.call.step_{step_num}", attributes={"step": step_num})
                if config.tracer is not None
                else None
            )
            tool_schemas = (
                list(tool_view.schemas)
                if (native_policy.enabled and not native_policy.demoted)
                else None
            )
            cache_bps: list[Any] | None = None
            if config.cache_policy is not None:
                from looplet.cache import CacheBreakDetector as _CBD
                from looplet.cache import compute_breakpoints as _compute_bps

                schemas_text = tool_view.catalog_text
                detector = next((h for h in hooks if isinstance(h, _CBD)), None)
                if detector is not None:
                    cache_bps = detector.record(
                        step_num,
                        system_prompt=config.system_prompt,
                        tool_schemas_text=schemas_text,
                        memory_text=rendered_memory,
                    )
                else:
                    cache_bps = _compute_bps(
                        config.cache_policy,
                        system_prompt=config.system_prompt,
                        tool_schemas_text=schemas_text,
                        memory_text=rendered_memory,
                    )

            llm_t0 = time.perf_counter()
            _set_run_lifecycle(loop_ctx, phase=RunPhase.LLM)
            chunk_emitters = [h.emitter for h in hooks if isinstance(h, StreamingHook)]
            if stream is not None and not has_streaming_hook:
                chunk_emitters.append(stream)

            def _emit_chunk(chunk: str, chunk_index: int) -> None:
                from looplet.streaming import LLMChunkEvent

                ev = LLMChunkEvent(step_num=step_num, chunk=chunk, chunk_index=chunk_index)
                for em in chunk_emitters:
                    em.emit(ev)

            stream_submitted_count = 0
            stream_halted = False
            stream_submitted_ids: set[str] = set()
            if allow_speculative:
                stream_executor = make_stream_executor()

            def _on_stream_tool_call(block: dict[str, Any]) -> None:
                nonlocal stream_submitted_count, stream_halted
                if stream_executor is None or stream_halted:
                    return
                if stream_submitted_count >= max(0, int(getattr(state, "budget_remaining", 0))):
                    stream_halted = True
                    return
                name = block.get("name", "")
                call_id = block.get("id", "")
                raw_input = block.get("input", {})
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(call_id, str)
                    or not call_id
                    or call_id in stream_submitted_ids
                    or not isinstance(raw_input, dict)
                    or "_raw_arguments" in raw_input
                    or name in config.resolve_terminal_outcomes()
                ):
                    stream_halted = True
                    return
                spec = tools._tools.get(name)
                if spec is None or not spec.is_concurrent_safe(raw_input):
                    stream_halted = True
                    return
                tc = ToolCall(tool=name, args=dict(raw_input), call_id=call_id)
                t_ctx = make_tool_ctx(tc, step_num + stream_submitted_count, effective_llm)
                stream_submitted_ids.add(call_id)
                stream_submitted_count += 1
                stream_executor.submit(tc, ctx=t_ctx)

            def _on_stream_reset() -> bool:
                return stream_submitted_count == 0

            try:
                llm_result = yield _CallLLM(
                    effective_llm,
                    prompt,
                    {
                        "max_tokens": config.max_tokens,
                        "system_prompt": config.system_prompt,
                        "temperature": config.temperature,
                        "tools": tool_schemas,
                        "native_policy": native_policy,
                        "cancel_token": config.cancel_token,
                        "max_continuations": config.max_turn_continuations,
                        "cache_breakpoints": cache_bps,
                        "generate_kwargs": config.generate_kwargs or None,
                        "should_continue": lambda: not _deadline_expired(loop_ctx),
                        "on_chunk": _emit_chunk if chunk_emitters else None,
                        "on_tool_call": _on_stream_tool_call
                        if stream_executor is not None
                        else None,
                        "on_stream_reset": _on_stream_reset
                        if stream_executor is not None
                        else None,
                    },
                )
            except BaseException:
                yield from _drain_unused_stream_executor()
                raise
            else:
                if stream_executor is not None and not llm_result.ok:
                    discarded_stream_receipts = yield from _drain_unused_stream_executor()
            finally:
                if llm_span is not None and config.tracer is not None:
                    config.tracer.end_span(llm_span)

            loop_ctx.native_tool_stats.record(llm_result)
            llm_calls += llm_result.calls + (
                0 if llm_result.native_attempted else llm_result.continuations
            )
            turn_usage_calls = _count_usage_calls(llm_result)
            usage_calls += turn_usage_calls
            _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls, config=config)
            if _deadline_expired(loop_ctx) and turns is None:
                yield from _drain_unused_stream_executor()
                stop_reason = "deadline_exceeded"
                done = True
                break
            state_metadata = getattr(state, "metadata", None)
            if isinstance(state_metadata, dict) and loop_ctx.native_tool_stats.has_activity:
                state_metadata["native_tool_stats"] = loop_ctx.native_tool_stats.to_dict()
            llm_dur_ms = (time.perf_counter() - llm_t0) * 1000.0
            if (
                stream is not None
                and _LLMCallEndEvent is not None
                and (turns is None or not has_streaming_hook)
            ):
                stream.emit(
                    _LLMCallEndEvent(
                        step_num=step_num,
                        model_turn_id=turns.turn_id if turns is not None else None,
                        response_length=len(llm_result.text or ""),
                        duration_ms=llm_dur_ms,
                        usage=llm_result.usage
                        or (
                            (getattr(effective_llm, "last_usage", None) or {})
                            if llm_result.ok
                            else {}
                        ),
                    )
                )

        if not llm_result.ok and llm_result.is_prompt_too_long and config.reactive_recovery:
            llm_result, rec_calls = yield from _recover_prompt_gen(
                llm_result,
                recovery_state,
                llm=effective_llm,
                state=state,
                session_log=session_log,
                tools=tools,
                config=config,
                step_num=step_num,
                hooks=hooks,
                conversation=active_conv,
                native_policy=native_policy,
                prepare_kwargs_fn=lambda: {
                    "config": config,
                    "state": state,
                    "tools": tools,
                    "session_log": session_log,
                    "task": task,
                    "step_num": step_num,
                    "briefing": briefing_str,
                    "context_sources": context_sources,
                    "hooks": hooks,
                    "builder": build_prompt_fn,
                    "conversation": active_conv,
                    "omit_text_history": omit_text_history,
                },
                turns=turns,
                loop_ctx=loop_ctx,
                compact_service=compact_service,
                should_continue=lambda: not _deadline_expired(loop_ctx),
            )
            prompt = getattr(effective_llm, "request_prompt", None) or prompt
            llm_calls += rec_calls
            usage_calls += _count_usage_calls(llm_result) - turn_usage_calls
            _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls)

        raw_response = llm_result.text
        if turns is not None:
            llm_calls = turns.data["model_calls"]
            llm_result.model_calls = effective_llm.request_records
            measured_usage: dict[str, Any] = {}
            for record in llm_result.model_calls:
                for name, value in record["usage"].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        measured_usage[name] = measured_usage.get(name, 0) + value
            llm_result.usage = measured_usage or None
            _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls)

        history.record_llm_turn(
            prompt=prompt,
            response=raw_response,
            metadata={
                "tool_view": {"names": list(tool_view.names), "version": tool_view.version},
                **({"model_turn_id": turns.turn_id} if turns is not None else {}),
                **(
                    {"scoped_context_plan": scoped_plan.to_dict()}
                    if scoped_plan is not None
                    else {}
                ),
            },
        )

        step_usage = llm_result.usage or (
            getattr(effective_llm, "last_usage", None) if llm_result.ok else None
        )
        if step_usage and turns is None:
            _accumulate_step_usage(state, step_usage)

        post_llm_decisions = yield from _emit_event_gen(
            hooks,
            _LE.POST_LLM_RESPONSE,
            step_num=step_num,
            state=state,
            session_log=session_log,
            context=context,
            prompt=prompt,
            raw_response=raw_response,
            usage=step_usage,
            extra={
                "native_tool_stats": loop_ctx.native_tool_stats.to_dict(),
                "model_calls": llm_result.model_calls,
            },
        )
        for d in post_llm_decisions:
            if d.stop is not None:
                stop_reason = d.stop
                hook_requested_stop = True
            if d.additional_context:
                post_dispatch_parts.append(d.additional_context)

        if turns is not None and (
            (hook_requested_stop and turns.persistent_admission)
            or _deadline_expired(loop_ctx)
            or (config.cancel_token is not None and config.cancel_token.is_cancelled)
        ):
            if _deadline_expired(loop_ctx):
                stop_reason = "deadline_exceeded"
            elif config.cancel_token is not None and config.cancel_token.is_cancelled:
                stop_reason = "cancelled"
            break
        if turns is not None and llm_result.failure_reason in {
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
            logger.error("LLM call failed at step %d: %s", step_num, llm_result.error)
            step = _failed_model_step(step_num, llm_result)
            if discarded_stream_receipts:
                step.metadata["speculative_tool_receipts"] = discarded_stream_receipts
            yield _AppendStep(state, step)
            stop_reason = llm_result.failure_reason
            history.record_step(
                step, theory="", entities=[], findings=[], highlights=[], recall_key=""
            )
            _set_run_lifecycle(
                loop_ctx,
                status=RunStatus.FAILED,
                phase=RunPhase.TERMINAL,
                termination_reason=stop_reason,
            )
            yield from _notify_post_step_gen(hooks, state, session_log, step_num)
            yield from _save_step_checkpoint(step_num)
            yield _YieldStep(step)
            break

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
        if not tool_calls and assistant_text is not None and turns is not None and turns.enabled:
            try:
                raw_dec = yield _CallPropose(turns, assistant_text, step_num)
                decision = turns.record_decision(raw_dec)
            except Exception as error:
                turns.metadata["assistant_policy_error"] = type(error).__name__
                stop_reason = "assistant_policy_error"
                llm_failed = True
                break
            assistant_decisions = yield from _emit_event_gen(
                hooks,
                _LE.ASSISTANT_TURN_DECISION,
                step_num=step_num,
                state=state,
                session_log=session_log,
                context=context,
                extra={
                    "model_turn_id": turns.turn_id,
                    "action": decision.action,
                    "model_calls": turns.data["model_calls"],
                },
            )
            if stream is not None and not has_streaming_hook:
                from looplet.streaming import AssistantTurnDecisionEvent

                stream.emit(
                    AssistantTurnDecisionEvent(
                        step_num=step_num,
                        model_turn_id=turns.turn_id,
                        action=decision.action,
                        model_calls=turns.data["model_calls"],
                    )
                )
            history.record_host_feedback(decision.feedback, model_turn_id=turns.turn_id)
            yield from _save_turns_checkpoint()
            consecutive_parse_failures = 0
            interruption = turns.interruption_reason(assistant_decisions)
            if interruption is not None:
                stop_reason = interruption
                break
            if decision.action == "stop":
                stop_reason = "assistant_stop"
                break
            if decision.action == "continue":
                reason = yield from _post_step_stop_reason_gen(
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
        if turns is not None and any(
            tc.metadata.get("completion_source") != "assistant_message" for tc in tool_calls
        ):
            turns.data["consecutive_assistant_turns"] = 0

        if not tool_calls:
            consecutive_parse_failures += 1
            if config.recovery_registry is not None:
                rec_action = config.recovery_registry.attempt_recovery(
                    FailureScenario.PARSE_ERROR,
                    {"step": step_num, "raw_response": raw_response},
                )
                if rec_action is not None and rec_action.action_type == "abort":
                    logger.warning("Recovery registry aborted parse recovery at step %d", step_num)
                    tc = ToolCall(
                        tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200]
                    )
                    tr = ToolResult(
                        tool="__parse_error__",
                        args_summary="",
                        data=None,
                        error=f"Parse error - recovery aborted: {rec_action.message}",
                    )
                    step = Step(number=step_num, tool_call=tc, tool_result=tr)
                    yield _AppendStep(state, step)
                    history.record_step(
                        step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                    )
                    yield from _notify_post_step_gen(hooks, state, session_log, step_num)
                    yield from _save_step_checkpoint(step_num)
                    yield _YieldStep(step)
                    reason = yield from _post_step_stop_reason_gen(
                        hooks,
                        state=state,
                        session_log=session_log,
                        context=context,
                        step_num=step_num,
                        new_entities=0,
                        requested_reason=stop_reason if hook_requested_stop else None,
                    )
                    if reason is not None:
                        stop_reason = reason
                        done = True
                        break
                    continue
                if rec_action is not None and rec_action.message:
                    post_dispatch_parts.append(rec_action.message)

            if (
                not hook_requested_stop
                and consecutive_parse_failures <= PARSE_RECOVERY_MAX
                and (not native_policy.enabled or native_policy.demoted)
            ):
                logger.warning(
                    "Parse failure %d/%d at step %d - attempting recovery",
                    consecutive_parse_failures,
                    PARSE_RECOVERY_MAX,
                    step_num,
                )
                recovery_prompt = build_parse_recovery_prompt(prompt, to_text(raw_response) or "")
                recovery_result = yield _CallLLM(
                    effective_llm,
                    recovery_prompt,
                    {
                        "max_tokens": config.max_tokens,
                        "system_prompt": config.system_prompt,
                        "temperature": config.recovery_temperature,
                        "cancel_token": config.cancel_token,
                        "generate_kwargs": config.generate_kwargs or None,
                    },
                )
                llm_calls += 1
                usage_calls += _count_usage_calls(recovery_result)
                if turns is None:
                    _accumulate_step_usage(
                        state,
                        recovery_result.usage
                        or (
                            getattr(effective_llm, "last_usage", None)
                            if recovery_result.ok
                            else None
                        ),
                    )
                _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls)
                if recovery_result.ok:
                    tool_calls = parse_multi_tool_calls(recovery_result.text)

            if not tool_calls:
                yield from _drain_unused_stream_executor()
                logger.warning("Unparseable LLM response at step %d after recovery", step_num)
                tc = ToolCall(tool="__parse_error__", reasoning=(to_text(raw_response) or "")[:200])
                tr = ToolResult(
                    tool="__parse_error__",
                    args_summary="",
                    data=None,
                    error=(
                        "Native tool protocol error: expected structured tool_use block; "
                        f"received {(to_text(raw_response) or '')[:200]}"
                        if native_policy.enabled and not native_policy.demoted
                        else f"Could not parse JSON: {(to_text(raw_response) or '')[:200]}"
                    ),
                )
                step = Step(number=step_num, tool_call=tc, tool_result=tr)
                yield _AppendStep(state, step)
                history.record_step(
                    step, theory="", entities=[], findings=[], highlights=[], recall_key=""
                )
                yield from _notify_post_step_gen(hooks, state, session_log, step_num)
                yield from _save_step_checkpoint(step_num)
                yield _YieldStep(step)
                reason = yield from _post_step_stop_reason_gen(
                    hooks,
                    state=state,
                    session_log=session_log,
                    context=context,
                    step_num=step_num,
                    new_entities=0,
                    requested_reason=stop_reason if hook_requested_stop else None,
                )
                if reason is not None:
                    stop_reason = reason
                    done = True
                    break
                continue
        else:
            consecutive_parse_failures = 0

        # ── Dispatch tool calls ──────────────────────────────
        all_step_entities: list[str] = []
        if (
            turns is not None
            and stream is not None
            and _StepStartEvent is not None
            and not has_streaming_hook
        ):
            stream.emit(_StepStartEvent(step_num=step_num))

        turn_plan = _prepare_tool_turn(
            tools, tool_calls, config, state, turn_id=f"turn-{llm_calls}"
        )
        if turn_plan is None:
            yield from _drain_unused_stream_executor()
            stop_reason = "budget_exhausted"
            done = True
            break
        tool_calls = turn_plan.calls
        done_idx = turn_plan.terminal_index

        regular_calls = tool_calls[:done_idx] if done_idx is not None else tool_calls
        if not regular_calls:
            yield from _drain_unused_stream_executor()
        if regular_calls:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.DISPATCHING)

            def _record_cancelled_settled_step(
                tc_offset_idx: int, settled_call: Any, settled_res: Any
            ) -> None:
                c_step = step_num + tc_offset_idx
                t_spec = tools._tools.get(settled_call.tool)
                if not (t_spec and t_spec.free):
                    state.queries_used += 1
                d_mode = (
                    "parallel"
                    if config.concurrent_dispatch
                    and t_spec is not None
                    and t_spec.is_concurrent_safe(settled_call.args)
                    else "serial"
                )
                s_obj = Step(
                    number=c_step,
                    tool_call=settled_call,
                    tool_result=settled_res,
                    metadata=_step_metadata(settled_call, dispatch_mode=d_mode),
                )
                if t_spec and t_spec.free and settled_res.error is None:
                    s_obj.metadata["budget_free"] = True
                state.steps.append(s_obj)
                s_ents = extract_entities(settled_res.data)
                all_step_entities.extend(s_ents)
                s_findings, s_highlights = extract_step_metadata(state, c_step)
                history.record_step(
                    s_obj,
                    theory=settled_call.args.get("__theory__", ""),
                    entities=s_ents,
                    findings=s_findings,
                    highlights=s_highlights,
                    recall_key=settled_res.result_key or "",
                )

            groups = (
                [regular_calls]
                if config.concurrent_dispatch
                else [[tc_item] for tc_item in regular_calls]
            )
            pre_streamed_calls = (
                stream_executor.submitted_calls() if stream_executor is not None else {}
            )
            stream_consumed = True
            if stream_executor is not None:
                raw_streamed = yield _DrainStreamExecutor(stream_executor)
                pre_streamed = {r.call_id: r for r in raw_streamed if r.call_id is not None}
            else:
                pre_streamed = {}

            dispatched_triples: list[tuple[int, Any, Any, bool]] = []
            offset = 0
            for group in groups:
                group_step = step_num + offset
                intercepted = yield from _intercept_tool_calls_gen(
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
                    if checkpoint_store is not None:
                        pending_step_no = group_step + len(group) - 1
                        pending_cp = _pending_dispatch_checkpoint(
                            **build_pending_dispatch_kwargs(
                                loop_ctx,
                                calls=list(group),
                                dispatching=list(calls_to_dispatch),
                                step_number=pending_step_no,
                                task=task,
                                checkpoint_state=checkpoint_state,
                                ckpt_store=checkpoint_store,
                                policy_checkpoint_metadata=_policy_checkpoint_metadata,
                                include_tool_results=False,
                            )
                        )
                        yield _SaveCheckpoint(
                            lambda pcp=pending_cp, ps=pending_step_no: checkpoint_store.save(
                                replace(
                                    pcp, tool_results_store=loop_ctx.tools.checkpoint_results()
                                ),
                                key=f"step_{ps}_pending",
                            )
                        )
                    reused_by_pos: dict[int, Any] = {}
                    remaining_calls: list[Any] = []
                    remaining_contexts: list[Any] = []
                    remaining_indices: list[int] = []
                    for pos, (index, call) in enumerate(dispatch_items):
                        sub_call = (
                            pre_streamed_calls.get(call.call_id)
                            if call.call_id is not None
                            else None
                        )
                        if (
                            call.call_id is not None
                            and call.call_id in pre_streamed
                            and sub_call is not None
                            and sub_call.tool == call.tool
                            and sub_call.args == call.args
                        ):
                            reused_by_pos[pos] = pre_streamed.pop(call.call_id)
                        else:
                            remaining_calls.append(call)
                            remaining_indices.append(index)
                            remaining_contexts.append(
                                make_tool_ctx(call, group_step + index, effective_llm)
                            )
                    if not remaining_calls:
                        results = [reused_by_pos[p] for p in range(len(calls_to_dispatch))]
                    else:
                        results = yield _DispatchRegularBatch(
                            tools=tools,
                            config=config,
                            remaining_calls=remaining_calls,
                            remaining_contexts=remaining_contexts,
                            remaining_indices=remaining_indices,
                            reused_by_pos=reused_by_pos,
                            dispatch_items=dispatch_items,
                            offset=offset,
                            record_settled_cb=_record_cancelled_settled_step,
                        )
                else:
                    results = []

                dispatch_results = iter(results)
                dispatched_triples.clear()
                for index, call in enumerate(group):
                    was_intercepted = index in intercepted.intercepted
                    res_item = (
                        intercepted.intercepted[index]
                        if was_intercepted
                        else next(dispatch_results)
                    )
                    dispatched_triples.append((offset + index, call, res_item, was_intercepted))
                offset += len(group)

                broke_serial = False
                for tc_idx, tool_call, tool_result, was_intercepted in dispatched_triples:
                    cur_step = step_num + tc_idx
                    tool_spec = tools._tools.get(tool_call.tool)
                    if not (tool_spec and tool_spec.free) and not was_intercepted:
                        state.queries_used += 1

                    dispatch_mode = (
                        "parallel"
                        if config.concurrent_dispatch
                        and tool_spec is not None
                        and tool_spec.is_concurrent_safe(tool_call.args)
                        else "serial"
                    )
                    event_metadata = _step_metadata(tool_call, dispatch_mode=dispatch_mode).get(
                        "turn", {}
                    )

                    if stream is not None and _ToolDispatchEvent is not None:
                        stream.emit(
                            _ToolDispatchEvent(
                                step_num=cur_step,
                                tool_name=tool_call.tool,
                                args_summary=_summarize_args_dict(tool_call.args),
                                metadata=event_metadata,
                            )
                        )

                    pd = yield from _run_post_dispatch_hooks_gen(
                        tool_call,
                        tool_result,
                        hooks,
                        state,
                        session_log,
                        context,
                        cur_step,
                    )
                    tool_result = pd.tool_result
                    post_dispatch_parts.extend(pd.extra_context)
                    if pd.stop_reason is not None:
                        stop_reason = pd.stop_reason
                        hook_requested_stop = True

                    step = Step(
                        number=cur_step,
                        tool_call=tool_call,
                        tool_result=tool_result,
                        metadata=_step_metadata(tool_call, dispatch_mode=dispatch_mode),
                    )
                    if (
                        tool_spec
                        and tool_spec.free
                        and not was_intercepted
                        and tool_result.error is None
                    ):
                        step.metadata["budget_free"] = True

                    yield _AppendStep(state, step)
                    step_entities = commit_step_to_history(
                        step=step,
                        cur_step=cur_step,
                        state=state,
                        history=history,
                        extract_entities=extract_entities,
                        extract_step_metadata=extract_step_metadata,
                        all_step_entities=all_step_entities,
                        already_appended=True,
                    )

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
                    yield from _notify_post_step_gen(hooks, state, session_log, cur_step)
                    yield from _save_step_checkpoint(cur_step)
                    yield _YieldStep(step)

                    if not config.concurrent_dispatch:
                        reason = None
                        if _deadline_expired(loop_ctx):
                            reason = "deadline_exceeded"
                        elif config.cancel_token is not None and config.cancel_token.is_cancelled:
                            reason = "cancelled"
                        elif tc_idx + 1 < len(regular_calls) or done_idx is not None:
                            reason = yield from _post_step_stop_reason_gen(
                                hooks,
                                state=state,
                                session_log=session_log,
                                context=context,
                                step_num=cur_step,
                                new_entities=len(step_entities),
                                requested_reason=stop_reason if hook_requested_stop else None,
                            )
                        if reason is not None:
                            stop_reason = reason
                            done = True
                            broke_serial = True
                            break
                if broke_serial:
                    break

            state_meta = getattr(state, "metadata", None)
            if pre_streamed and isinstance(state_meta, dict):
                leftovers = [
                    {**r.to_dict(), "unmatched_speculative": True}
                    for r in pre_streamed.values()
                    if r.metadata.get("execution_status") != "not_executed"
                ]
                if leftovers:
                    state_meta.setdefault("speculative_tool_receipts", []).extend(leftovers)

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

        # Handle done() if present
        if done_idx is not None:
            _set_run_lifecycle(loop_ctx, phase=RunPhase.FINALIZING)
            tool_call = tool_calls[done_idx]
            cur_step = step_num + done_idx

            gate_warning: str | None = None
            for hook in hooks:
                if hasattr(hook, "check_done"):
                    try:
                        w = yield _CallHook(
                            hook,
                            "check_done",
                            (state, session_log, context, cur_step),
                            optional_kwargs={"tool_call": tool_call},
                        )
                        cd_decision = normalize_hook_return(w, slot="check_done")
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "check_done hook %s raised; rejecting done()",
                            type(hook).__name__,
                        )
                        cd_decision = HookDecision(
                            block=(
                                f"Quality gate {type(hook).__name__} failed; "
                                "the final answer was not accepted."
                            ),
                            metadata={"hook_error": True},
                        )
                    if cd_decision is not None:
                        yield from _emit_hook_decision_gen(
                            hooks,
                            decision=cd_decision,
                            hook_slot="check_done",
                            hook_name=type(hook).__name__,
                            step_num=cur_step,
                            state=state,
                            session_log=session_log,
                            context=context,
                        )
                    if cd_decision is not None and cd_decision.is_block():
                        gate_warning = cd_decision.block or "blocked by hook"
                        break

            gate_warning = _completion_gate_warning(config, tool_call, gate_warning)
            done_metadata = _terminal_step_metadata(tool_call, tool_calls[done_idx + 1 :])

            if stream is not None and _ToolDispatchEvent is not None:
                stream.emit(
                    _ToolDispatchEvent(
                        step_num=cur_step,
                        tool_name=tool_call.tool,
                        args_summary=_summarize_args_dict(tool_call.args),
                        metadata=done_metadata.get("turn", {}),
                    )
                )

            if gate_warning is not None:
                logger.info("Quality gate rejected done() at step %d", step_num)
                quality_gate_message = gate_warning
                step = _rejected_terminal_step(tool_call, cur_step, gate_warning, done_metadata)
                tool_result = step.tool_result
                yield _AppendStep(state, step)
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
                history.record_step(
                    step,
                    theory="",
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key=tool_result.result_key or "",
                )
                yield from _notify_post_step_gen(hooks, state, session_log, cur_step)
                yield from _save_step_checkpoint(cur_step)
                yield _YieldStep(step)
            else:
                done_ctx = make_tool_ctx(tool_call, cur_step, effective_llm)
                if (
                    checkpoint_store is not None
                    and tool_call.metadata.get("completion_source") != "assistant_message"
                ):
                    done_pending_cp = _pending_dispatch_checkpoint(
                        **build_pending_dispatch_kwargs(
                            loop_ctx,
                            calls=[tool_call],
                            dispatching=[tool_call],
                            step_number=cur_step,
                            task=task,
                            checkpoint_state=checkpoint_state,
                            ckpt_store=checkpoint_store,
                            policy_checkpoint_metadata=_policy_checkpoint_metadata,
                            include_tool_results=False,
                        )
                    )
                    cs_done = cur_step
                    yield _SaveCheckpoint(
                        lambda dpcp=done_pending_cp, csd=cs_done: checkpoint_store.save(
                            replace(dpcp, tool_results_store=loop_ctx.tools.checkpoint_results()),
                            key=f"step_{csd}_pending",
                        )
                    )
                if tool_call.metadata.get("completion_source") == "assistant_message":
                    tool_result = ToolResult(
                        tool=tool_call.tool,
                        args_summary="assistant message",
                        data=dict(tool_call.args),
                    )
                else:
                    tool_result = yield _DispatchSingleTool(tools, tool_call, done_ctx)
                _normalize_done_rejection(tool_result)

                pd_done = yield from _run_post_dispatch_hooks_gen(
                    tool_call,
                    tool_result,
                    hooks,
                    state,
                    session_log,
                    context,
                    cur_step,
                    emit_lifecycle=False,
                )
                tool_result = pd_done.tool_result
                rejection_reason = _normalize_done_rejection(tool_result)
                rejected_done = rejection_reason is not None
                done_metadata["terminal_accepted"] = not rejected_done
                if rejection_reason is not None:
                    quality_gate_message = rejection_reason
                    if not build_final_trace:
                        post_dispatch_parts.append(rejection_reason)

                step = Step(
                    number=cur_step,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    metadata=done_metadata,
                )
                yield _AppendStep(state, step)
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
                            classification="rejected" if rejected_done else "done",
                            new_entities_count=0,
                            metadata=done_metadata.get("turn", {}),
                        )
                    )
                history.record_step(
                    step,
                    theory=tool_call.args.get("__theory__", ""),
                    entities=[],
                    findings=[],
                    highlights=[],
                    recall_key=tool_result.result_key or "",
                )
                if rejected_done:
                    yield from _notify_post_step_gen(hooks, state, session_log, cur_step)
                    yield from _save_step_checkpoint(cur_step)
                    yield _YieldStep(step)
                    continue
                _set_run_lifecycle(
                    loop_ctx,
                    status=RunStatus.COMPLETED,
                    phase=RunPhase.TERMINAL,
                    termination_reason="done",
                    terminal_tool=tool_call.tool,
                )
                yield from _notify_post_step_gen(hooks, state, session_log, cur_step)
                yield from _save_step_checkpoint(cur_step, status="done")
                done = True
                stop_reason = "done"
                yield from _emit_event_gen(
                    hooks,
                    _LE.DONE_ACCEPTED,
                    step_num=cur_step,
                    state=state,
                    session_log=session_log,
                    context=context,
                    tool_call=tool_call,
                    tool_result=tool_result,
                )
                yield _YieldStep(step)

        if done:
            continue

        reason = yield from _post_step_stop_reason_gen(
            hooks,
            state=state,
            session_log=session_log,
            context=context,
            step_num=step_num,
            new_entities=len(all_step_entities),
            requested_reason=stop_reason if hook_requested_stop else None,
        )
        if reason is not None:
            stop_reason = reason
            done = True
            break

    # ── Post-loop finalization ───────────────────────────────
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
    try:
        state._stop_reason = stop_reason
    except AttributeError:
        pass

    if turns is not None:
        llm_calls = turns.data["model_calls"]
        yield from _save_turns_checkpoint()
    _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls)

    finalizer.mark_stop_emitted()
    yield from _emit_event_gen(
        hooks,
        _LE.STOP,
        state=state,
        session_log=session_log,
        context=context,
        termination_reason=stop_reason,
    )

    for end_hook in finalizer._take_hooks():
        try:
            extra_c = yield _CallHook(end_hook, "on_loop_end", finalizer._args)
            if isinstance(extra_c, int):
                llm_calls += extra_c
        except _AdmissionDenied as exc:
            state.metadata.setdefault("post_run_model_denials", []).append(exc.termination_reason)

    if turns is not None:
        llm_calls = turns.call_total(llm_calls)
        yield from _save_turns_checkpoint()
    _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls)

    if not build_final_trace:
        _record_run_stats(
            state,
            llm_calls=llm_calls,
            usage_calls=usage_calls,
            duration_ms=(time.perf_counter() - perf_t0) * 1000,
        )
        if stream is not None and _LoopEndEvent is not None and not has_streaming_hook:
            stream.emit(
                _LoopEndEvent(
                    total_steps=state.step_count,
                    total_llm_calls=llm_calls,
                    reason=stop_reason,
                )
            )
        return None

    if (
        turns is None
        and stream is not None
        and _LoopEndEvent is not None
        and not has_streaming_hook
    ):
        stream.emit(
            _LoopEndEvent(
                total_steps=state.step_count,
                total_llm_calls=llm_calls,
                reason=stop_reason,
            )
        )

    elapsed = (time.time() - t0) * 1000
    _record_run_stats(state, llm_calls=llm_calls, usage_calls=usage_calls, duration_ms=elapsed)
    build_trace_fn = config.resolve_callable("build_trace")
    if build_trace_fn is not None:
        try:
            bt_params = inspect.signature(build_trace_fn).parameters
            bt_takes_context = "context" in bt_params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in bt_params.values()
            )
        except (TypeError, ValueError):
            bt_takes_context = False
        bt_kwargs: dict[str, Any] = dict(
            task=task,
            state=state,
            session_log=session_log,
            done=stop_reason == "done",
            llm=turns.wrap(llm) if turns is not None else llm,
            llm_calls=llm_calls,
            elapsed_ms=elapsed,
        )
        if bt_takes_context:
            bt_kwargs["context"] = context
        try:
            trace = build_trace_fn(**bt_kwargs)
        except _AdmissionDenied as exc:
            getattr(state, "metadata", {}).setdefault("post_run_model_denials", []).append(
                exc.termination_reason
            )
            trace = {
                "task": task,
                "steps": [step.to_dict() for step in _trace_steps(state)],
                "steps_total": len(state.steps),
                "trace_dir": getattr(state, "metadata", {}).get("trace_dir"),
                "llm_calls": turns.data["model_calls"] if turns is not None else llm_calls,
                "total_time_ms": elapsed,
                "conversation": active_conv,
            }
    else:
        trace = {
            "task": task,
            "steps": [s.to_dict() for s in _trace_steps(state)],
            "steps_total": len(state.steps),
            "trace_dir": getattr(state, "metadata", {}).get("trace_dir"),
            "llm_calls": llm_calls,
            "total_time_ms": elapsed,
            "conversation": active_conv,
        }
    if turns is not None:
        llm_calls = turns.call_total(llm_calls)
        _record_run_stats(
            state,
            llm_calls=llm_calls,
            usage_calls=usage_calls,
            duration_ms=(time.time() - t0) * 1000,
        )
        yield from _save_turns_checkpoint()
        if stream is not None and _LoopEndEvent is not None and not has_streaming_hook:
            stream.emit(
                _LoopEndEvent(
                    total_steps=state.step_count, total_llm_calls=llm_calls, reason=stop_reason
                )
            )
    return trace


# ── Sync & Async Effect Interpreters ─────────────────────────────


def execute_effect_sync(eff: Any) -> Any:
    """Execute one Sans-I/O loop effect synchronously."""
    from looplet.compact import run_compact
    from looplet.hook_decision import _invoke_hook
    from looplet.prompt_preparation import prepare_prompt
    from looplet.scaffolding import llm_call_with_retry
    from looplet.tools import BaseToolRegistry

    if isinstance(eff, _CallHook):
        return _invoke_hook(eff.hook, eff.slot, *eff.args, _optional_kwargs=eff.optional_kwargs)
    if isinstance(eff, _CallDirect):
        return eff.fn(*eff.args)
    if isinstance(eff, _CallPropose):
        decision = eff.turns.propose(eff.text, eff.step_num)
        if inspect.isawaitable(decision):
            if inspect.iscoroutine(decision):
                decision.close()
            raise TypeError("synchronous assistant policy cannot return an awaitable")
        return decision
    if isinstance(eff, _RunCompact):
        return run_compact(
            eff.service,
            hooks=eff.hooks,
            state=eff.state,
            session_log=eff.session_log,
            llm=eff.llm,
            conversation=eff.conversation,
            step_num=eff.step_num,
            reason=eff.reason,
        )
    if isinstance(eff, _PreparePrompt):
        return prepare_prompt(**eff.kwargs)
    if isinstance(eff, _CallLLM):
        return llm_call_with_retry(eff.llm, eff.prompt, **eff.kwargs)
    if isinstance(eff, _DrainStreamExecutor):
        return eff.executor.results()
    if isinstance(eff, _SaveCheckpoint):
        return eff.fn()
    if isinstance(eff, _DispatchRegularBatch):
        total = len(eff.dispatch_items)
        if eff.config.concurrent_dispatch:
            if type(eff.tools).dispatch_batch is BaseToolRegistry.dispatch_batch:
                fresh = eff.tools.dispatch_batch(
                    eff.remaining_calls,
                    ctx=eff.remaining_contexts,
                    max_workers=eff.config.max_parallel_calls,
                )
            else:
                fresh = eff.tools.dispatch_batch(eff.remaining_calls, ctx=eff.remaining_contexts)
        else:
            fresh = [
                eff.tools.dispatch(c, ctx=c_ctx)
                for c, c_ctx in zip(eff.remaining_calls, eff.remaining_contexts)
            ]
        fresh_iter = iter(fresh)
        return [
            eff.reused_by_pos[p] if p in eff.reused_by_pos else next(fresh_iter)
            for p in range(total)
        ]
    if isinstance(eff, _DispatchSingleTool):
        return eff.tools.dispatch(eff.call, ctx=eff.ctx)
    if isinstance(eff, _AppendStep):
        eff.state.steps.append(eff.step)
        return None
    raise TypeError(f"Unsupported sync effect: {type(eff)!r}")


async def execute_effect_async(eff: Any) -> Any:
    """Execute one Sans-I/O loop effect asynchronously without blocking the event loop."""
    import asyncio

    from looplet._storage import DiskBackedList
    from looplet.async_loop import async_llm_call
    from looplet.checkpoint import _run_persistence
    from looplet.compact import run_compact_async
    from looplet.hook_decision import _invoke_hook_async
    from looplet.prompt_preparation import prepare_prompt_async
    from looplet.tools import BaseToolRegistry

    if isinstance(eff, _CallHook):
        return await _invoke_hook_async(
            eff.hook, eff.slot, *eff.args, _optional_kwargs=eff.optional_kwargs
        )
    if isinstance(eff, _CallDirect):
        raw = eff.fn(*eff.args)
        return (await raw) if inspect.isawaitable(raw) else raw
    if isinstance(eff, _CallPropose):
        raw = eff.turns.propose(eff.text, eff.step_num)
        return (await raw) if inspect.isawaitable(raw) else raw
    if isinstance(eff, _RunCompact):
        return await run_compact_async(
            eff.service,
            hooks=eff.hooks,
            state=eff.state,
            session_log=eff.session_log,
            llm=eff.llm,
            conversation=eff.conversation,
            step_num=eff.step_num,
            reason=eff.reason,
        )
    if isinstance(eff, _PreparePrompt):
        return await prepare_prompt_async(**eff.kwargs)
    if isinstance(eff, _CallLLM):
        return await async_llm_call(eff.llm, eff.prompt, **eff.kwargs)
    if isinstance(eff, _DrainStreamExecutor):
        return await eff.executor.results()
    if isinstance(eff, _SaveCheckpoint):
        return await _run_persistence(eff.fn)
    if isinstance(eff, _DispatchRegularBatch):
        total = len(eff.dispatch_items)
        if eff.config.concurrent_dispatch:
            settled_by_rem_idx: dict[int, Any] = {}
            try:
                if (
                    type(eff.tools).dispatch_batch is not BaseToolRegistry.dispatch_batch
                    and type(eff.tools).async_dispatch_batch
                    is BaseToolRegistry.async_dispatch_batch
                ):
                    fresh = await asyncio.to_thread(
                        eff.tools.dispatch_batch,
                        eff.remaining_calls,
                        ctx=eff.remaining_contexts,
                    )
                elif type(eff.tools).async_dispatch_batch is BaseToolRegistry.async_dispatch_batch:
                    fresh = await eff.tools.async_dispatch_batch(
                        eff.remaining_calls,
                        ctx=eff.remaining_contexts,
                        max_concurrent=eff.config.max_parallel_calls,
                        settled_out=settled_by_rem_idx,
                    )
                else:
                    fresh = await eff.tools.async_dispatch_batch(
                        eff.remaining_calls, ctx=eff.remaining_contexts
                    )
            except asyncio.CancelledError:
                for pos, (index, call) in enumerate(eff.dispatch_items):
                    if pos in eff.reused_by_pos:
                        eff.record_settled_cb(eff.offset + index, call, eff.reused_by_pos[pos])
                for rem_pos, rem_res in sorted(settled_by_rem_idx.items()):
                    eff.record_settled_cb(
                        eff.offset + eff.remaining_indices[rem_pos],
                        eff.remaining_calls[rem_pos],
                        rem_res,
                    )
                raise
        else:
            fresh = []
            try:
                for call, t_ctx in zip(eff.remaining_calls, eff.remaining_contexts):
                    fresh.append(await eff.tools.async_dispatch(call, ctx=t_ctx))
            except asyncio.CancelledError:
                for pos, (index, call) in enumerate(eff.dispatch_items):
                    if pos in eff.reused_by_pos:
                        eff.record_settled_cb(eff.offset + index, call, eff.reused_by_pos[pos])
                for rem_pos, rem_res in enumerate(fresh):
                    eff.record_settled_cb(
                        eff.offset + eff.remaining_indices[rem_pos],
                        eff.remaining_calls[rem_pos],
                        rem_res,
                    )
                raise
        fresh_iter = iter(fresh)
        return [
            eff.reused_by_pos[p] if p in eff.reused_by_pos else next(fresh_iter)
            for p in range(total)
        ]
    if isinstance(eff, _DispatchSingleTool):
        return await eff.tools.async_dispatch(eff.call, ctx=eff.ctx)
    if isinstance(eff, _AppendStep):
        steps = eff.state.steps
        if isinstance(steps, DiskBackedList) and steps._mem is None:
            await _run_persistence(steps.append, eff.step)
        else:
            steps.append(eff.step)
        return None
    raise TypeError(f"Unsupported async effect: {type(eff)!r}")


def run_sync_gen(gen: Generator[Any, Any, Any]) -> Any:
    """Run a non-yielding Sans-I/O helper generator synchronously to completion."""
    sent: Any = None
    throwing: BaseException | None = None
    while True:
        try:
            eff = gen.throw(throwing) if throwing is not None else gen.send(sent)
        except StopIteration as stop:
            return stop.value
        throwing = None
        sent = None
        try:
            sent = execute_effect_sync(eff)
        except BaseException as exc:
            throwing = exc


async def run_async_gen(gen: Generator[Any, Any, Any]) -> Any:
    """Run a non-yielding Sans-I/O helper generator asynchronously to completion."""
    sent: Any = None
    throwing: BaseException | None = None
    while True:
        try:
            eff = gen.throw(throwing) if throwing is not None else gen.send(sent)
        except StopIteration as stop:
            return stop.value
        throwing = None
        sent = None
        try:
            sent = await execute_effect_async(eff)
        except BaseException as exc:
            throwing = exc


def drive_turn_machine_sync(gen: Generator[Any, Any, Any]) -> Generator[Any, Any, Any]:
    """PEP 380-faithful synchronous generator driver for ``run_turn_machine``."""
    sent: Any = None
    throwing: BaseException | None = None
    while True:
        try:
            eff = gen.throw(throwing) if throwing is not None else gen.send(sent)
        except StopIteration as stop:
            return stop.value
        throwing = None
        sent = None
        try:
            if isinstance(eff, _YieldStep):
                sent = yield eff.step
            else:
                sent = execute_effect_sync(eff)
        except GeneratorExit:
            gen.close()
            raise
        except BaseException as exc:
            throwing = exc


async def drive_turn_machine_async(gen: Generator[Any, Any, Any]) -> AsyncIterator[Any]:
    """PEP 380-faithful asynchronous generator driver for ``run_turn_machine``."""
    sent: Any = None
    throwing: BaseException | None = None
    while True:
        try:
            eff = gen.throw(throwing) if throwing is not None else gen.send(sent)
        except StopIteration:
            return
        throwing = None
        sent = None
        try:
            if isinstance(eff, _YieldStep):
                sent = yield eff.step
            else:
                sent = await execute_effect_async(eff)
        except GeneratorExit:
            gen.close()
            raise
        except BaseException as exc:
            throwing = exc
