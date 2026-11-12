"""Shared Sans-I/O turn-engine primitives for synchronous and asynchronous loops."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from looplet.loop import LoopConfig


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
) -> Any:
    """Select the effective turn backend while preserving replay and recording scopes."""
    from looplet.loop import _ScopedLLMProxy
    from looplet.provenance import _is_replay_backend, _RecordingBase

    if config is not None and config.router is not None and not _is_replay_backend(llm):
        selected = config.router.select(purpose="reasoning")
        if isinstance(llm, _RecordingBase):
            selected = llm.for_backend(selected)
        elif isinstance(llm, _ScopedLLMProxy):
            rec = llm._recording
            routed = rec.for_backend(selected) if isinstance(rec, _RecordingBase) else selected
            selected = _ScopedLLMProxy(routed, rec, llm._scope, llm._step_num)
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
) -> tuple[LoopConfig, int, Any, dict[str, Any] | None]:
    """Validate and apply an initial checkpoint before entering the turn loop."""
    if saved_checkpoint is None:
        return config, 0, active_conv, None

    from looplet.assistant_turns import TURN_STATE_KEY, restore_assistant_checkpoint
    from looplet.types import RunEnvelope

    validate_checkpoint_identity(
        saved_checkpoint,
        config.run_envelope.run_id if config.run_envelope is not None else None,
    )
    resumed = resume_loop_state(saved_checkpoint, max_steps=config.max_steps)
    restore_backend = (
        config.router.select(purpose="reasoning") if config.router is not None else llm
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
        "tool_results_store": loop_ctx.tools.checkpoint_results(),
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
