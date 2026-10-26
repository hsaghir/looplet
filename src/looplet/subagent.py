"""Sub-agent spawning - run focused sub-tasks with isolated context.

Provides run_sub_loop() which creates an isolated composable_loop() call
with its own state and session log. The parent agent gets back
a concise summary without the sub-agent's raw data polluting context.

Usage:
    from looplet.subagent import run_sub_loop

    result = run_sub_loop(
        llm=llm, task=task, tools=tools,
        max_steps=5, system_prompt="Focus on this...",
    )
    summary = result["summary"]  # concise finding for parent context
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import threading
import time
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

from looplet.backend_contract import (
    BackendRequestPolicy,
    BackendWrapper,
    TerminalLLMError,
    backend_request_policy,
    single_model_request,
)
from looplet.capabilities import ExecutionPolicy
from looplet.permissions import PermissionEngine, PermissionHook
from looplet.types import CancelToken, RunEnvelope

__all__ = [
    "run_sub_loop",
    "run_sub_loop_async",
    "run_sub_loops_async",
    "clone_tools_excluding",
    "ChildRunPolicy",
    "SharedModelBudget",
    "ChildRunLimitExceeded",
    "ChildRunFailed",
    "ChildRecoveryError",
]


logger = logging.getLogger(__name__)


class ChildRunLimitExceeded(RuntimeError):
    """A child-run deadline, cancellation, or model allowance was exhausted."""


class ChildRunFailed(RuntimeError):
    """A parallel child failed; its result remains available for inspection."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__(f"child {result['subagent_id']} stopped: {result['stop_reason']}")


class ChildRecoveryError(ValueError):
    """Saved child work does not match the requested delegation."""


@dataclass
class SharedModelBudget:
    """Atomic request admission shared by host-owned parent and child backends.

    An admitted attempt stays charged if the provider fails. Reservations and
    finalization remain host policy. Wrapping establishes single-request scope,
    so retry-capable adapters must honor separately admitted attempts.
    """

    remaining: int
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    _limit: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.remaining) is not int or self.remaining < 0:
            raise ValueError("model call allowance must be a nonnegative integer")
        self._limit = self.remaining

    @staticmethod
    def _check_constraints(*, deadline: float | None, cancel_token: CancelToken | None) -> None:
        if cancel_token is not None and cancel_token.is_cancelled:
            raise ChildRunLimitExceeded("child run cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise ChildRunLimitExceeded("child run deadline exceeded")

    def charge(
        self, *, deadline: float | None = None, cancel_token: CancelToken | None = None
    ) -> None:
        self._check_constraints(deadline=deadline, cancel_token=cancel_token)
        with self._lock:
            self._check_constraints(deadline=deadline, cancel_token=cancel_token)
            if self.remaining <= 0:
                raise ChildRunLimitExceeded("model call budget exhausted")
            self.remaining -= 1

    @contextmanager
    def admit(
        self, *, deadline: float | None = None, cancel_token: CancelToken | None = None
    ) -> Iterator[None]:
        """Admit one provider attempt, including direct host/helper SDK calls."""
        self.charge(deadline=deadline, cancel_token=cancel_token)
        with single_model_request():
            yield

    def checkpoint_state(self) -> dict[str, Any]:
        """Snapshot accounting; hosts must separately retain the original deadline."""
        with self._lock:
            return {
                "schema": "looplet.model-budget.v1",
                "limit": self._limit,
                "remaining": self.remaining,
            }

    def validate_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Reject malformed snapshots or restoration with a larger allowance."""
        if not isinstance(state, dict) or set(state) != {"schema", "limit", "remaining"}:
            raise ValueError("invalid model budget checkpoint")
        limit, remaining = state["limit"], state["remaining"]
        if (
            state["schema"] != "looplet.model-budget.v1"
            or type(limit) is not int
            or type(remaining) is not int
            or not 0 <= remaining <= limit
        ):
            raise ValueError("invalid model budget checkpoint")
        if self._limit > limit:
            raise ValueError("restoration cannot increase the model call allowance")

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Restore without replenishing either saved or already consumed calls."""
        self.validate_checkpoint_state(state)
        used = state["limit"] - state["remaining"]
        with self._lock:
            self.remaining = min(self.remaining, max(0, self._limit - used))

    def wrap(
        self,
        backend: Any,
        *,
        deadline: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Any:
        from looplet.loop import _ScopedLLMProxy

        if isinstance(backend, _ScopedLLMProxy):
            return _ScopedLLMProxy(
                self.wrap(backend._backend, deadline=deadline, cancel_token=cancel_token),
                backend._recording,
                backend._scope,
            )
        if isinstance(backend, _BudgetedBackend) and backend.budget is self:
            if backend.deadline is not None:
                deadline = (
                    min(backend.deadline, deadline) if deadline is not None else backend.deadline
                )
            if backend.cancel_token is not None and cancel_token is not None:
                if backend.cancel_token is not cancel_token:
                    raise ValueError("shared model budget cannot replace a cancellation token")
            cancel_token = cancel_token or backend.cancel_token
            backend = backend.backend
        return _BudgetedBackend(backend, self, deadline, cancel_token)


@dataclass(frozen=True)
class ChildRunPolicy:
    """Optional parent-owned limits for an in-process child loop."""

    parent_id: str
    allowed_tools: frozenset[str] | None = None
    max_steps: int | None = None
    model_budget: SharedModelBudget | None = None
    deadline: float | None = None
    cancel_token: CancelToken | None = None
    execution_policy: ExecutionPolicy | None = None
    permission_engines: tuple[PermissionEngine, ...] = ()
    approval_handler: Callable[[str, list[str] | None], str | None] | None = None
    recovery_dir: str | None = None
    recovery_version: str | None = None


class _BudgetedBackend(BackendWrapper):
    _looplet_manages_retries = False
    _supports_single_model_request = True
    request_policy = BackendRequestPolicy(supports_single_model_request=True)

    def __init__(
        self,
        backend: Any,
        budget: SharedModelBudget,
        deadline: float | None,
        cancel_token: CancelToken | None,
    ) -> None:
        self.backend = backend
        self.budget = budget
        self.deadline = deadline
        self.cancel_token = cancel_token

    @property
    def request_budgets(self) -> tuple[SharedModelBudget, ...]:
        inherited = getattr(self.backend, "request_budgets", ())
        return (self.budget, *(budget for budget in inherited if budget is not self.budget))

    @contextmanager
    def _admit(self) -> Iterator[None]:
        self.budget._check_constraints(deadline=self.deadline, cancel_token=self.cancel_token)
        policy = backend_request_policy(self.backend)
        if policy.manages_retries and not policy.supports_single_model_request:
            raise TerminalLLMError(
                "Retry-owning backend does not support single-request admission",
                termination_reason="unsupported_context",
            )
        if any(budget is self.budget for budget in getattr(self.backend, "request_budgets", ())):
            with single_model_request():
                yield
        else:
            with self.budget.admit(deadline=self.deadline, cancel_token=self.cancel_token):
                yield

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.backend, name)
        if name not in ("generate", "generate_with_tools"):
            return attribute

        if inspect.iscoroutinefunction(attribute):

            @wraps(attribute)
            async def async_guarded(*args: Any, **kwargs: Any) -> Any:
                with self._admit():
                    return await attribute(*args, **kwargs)

            return async_guarded

        @wraps(attribute)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with self._admit():
                response = attribute(*args, **kwargs)
            if inspect.isawaitable(response):

                async def finish_response() -> Any:
                    try:
                        self.budget._check_constraints(
                            deadline=self.deadline, cancel_token=self.cancel_token
                        )
                        with single_model_request():
                            return await response
                    except BaseException:
                        if inspect.iscoroutine(response):
                            response.close()
                        raise

                return finish_response()
            return response

        return guarded


class _BudgetedRouter:
    def __init__(self, router: Any, budget: SharedModelBudget, policy: ChildRunPolicy) -> None:
        self._router = router
        self._budget = budget
        self._policy = policy

    def select(self, purpose: str, **kwargs: Any) -> Any:
        return self._budget.wrap(
            self._router.select(purpose=purpose, **kwargs),
            deadline=self._policy.deadline,
            cancel_token=self._policy.cancel_token,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


@dataclass
class _ChildRun:
    inputs: dict[str, Any]
    observers: list[Any]
    subagent_id: str
    policy: ChildRunPolicy | None
    build_summary: Callable[[Any, Any, list[dict]], dict[str, Any]] | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    recovery_store: Any = None
    recovery_binding: str | None = None
    error_reason: str | None = None


class _ChildRecoveryEvidence:
    def __init__(self, checkpoint_state: Any, restore_checkpoint_state: Any) -> None:
        self._checkpoint_state = checkpoint_state
        self._restore_checkpoint_state = restore_checkpoint_state
        self._recovered_steps: list[dict[str, Any]] = []

    def checkpoint_state(self, context: Any) -> dict[str, Any]:
        domain = dict(self._checkpoint_state(context)) if self._checkpoint_state else {}
        steps = [*self._recovered_steps, *(step.to_dict() for step in context.state.steps)]
        by_number = {step["step"]: step for step in steps}
        domain["_looplet_child_receipts"] = [by_number[number] for number in sorted(by_number)]
        return domain

    def restore_checkpoint_state(self, context: Any, domain: dict[str, Any]) -> None:
        domain = dict(domain)
        self._recovered_steps = list(domain.pop("_looplet_child_receipts", []))
        if self._restore_checkpoint_state is not None:
            self._restore_checkpoint_state(context, domain)

    def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> str | None:
        if not self._recovered_steps:
            return None
        return (
            "Committed child tool receipts from before restart. "
            "These calls already ran; do not repeat their effects.\n"
            + json.dumps(self._recovered_steps, sort_keys=True)
        )


def _prepare_child_recovery(run: _ChildRun) -> None:
    from looplet.checkpoint import Checkpoint, FileCheckpointStore, resume_loop_state

    policy = run.policy
    if policy is None or policy.recovery_dir is None:
        return
    config = run.inputs["config"]
    execution = config.execution_policy
    engines = [hook.engine for hook in run.inputs["hooks"] if isinstance(hook, PermissionHook)]
    binding = {
        "version": policy.recovery_version,
        "task": run.inputs["task"],
        "parent_id": policy.parent_id,
        "config": {
            "max_steps": config.max_steps,
            "system_prompt": config.system_prompt,
            "done_tool": config.done_tool,
            "done_tools": config.done_tools,
            "use_native_tools": config.use_native_tools,
            "generate_kwargs": config.generate_kwargs,
            "run_envelope": config.run_envelope.to_dict(),
        },
        "tools": [
            {"name": name, "parameters": spec.parameters, "capabilities": list(spec.capabilities)}
            for name, spec in sorted(run.inputs["tools"].tool_specs.items())
        ],
        "execution": {
            "workspace_root": execution.workspace_root,
            "scratch_root": execution.scratch_root,
            "capabilities": sorted(execution.capabilities),
            "environment": sorted(execution.environment),
        }
        if execution is not None
        else None,
        "permissions": [
            {
                "version": engine.policy_version,
                "default": engine.default.value,
                "rules": [
                    (rule.tool, rule.decision.value, rule.reason, rule.arg_matcher is not None)
                    for rule in engine.rules
                ],
            }
            for engine in engines
        ],
    }
    digest = hashlib.sha256(
        json.dumps(binding, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    store = FileCheckpointStore(config.checkpoint_dir)
    for path in Path(config.checkpoint_dir).glob("*.json"):
        try:
            checkpoint = Checkpoint.from_dict(json.loads(path.read_text()))
        except (ValueError, OSError) as exc:
            raise ChildRecoveryError(
                "child checkpoint is corrupt; reconcile before retrying"
            ) from exc
        if checkpoint.run_id != run.subagent_id:
            raise ChildRecoveryError("child checkpoint belongs to another run")
    record = store.load("delegation", run_id=run.subagent_id)
    if record is not None:
        if record.metadata.get("binding") != digest:
            raise ChildRecoveryError("saved child task, version, or policy does not match")
        saved = record.metadata.get("child_result")
        if isinstance(saved, dict):
            run.result = {**saved, "recovered": True, "new_llm_calls": 0}
        else:
            latest = store.load_latest(run_id=run.subagent_id)
            if latest is not None and latest.metadata.get("pending_calls"):
                resume_loop_state(latest)
            latest = store.load_latest_for_resume(run_id=run.subagent_id)
            if latest is None:
                raise ChildRecoveryError(
                    "child stopped without a saved result; reconcile before retrying"
                )
            receipts = latest.domain_state.get("_looplet_child_receipts")
            if latest.step_number > 0 and (
                not isinstance(receipts, list)
                or not receipts
                or any(
                    not isinstance(step, dict)
                    or isinstance(step.get("step"), bool)
                    or not isinstance(step.get("step"), int)
                    or not isinstance(step.get("tool_call"), dict)
                    or not isinstance(step.get("tool_result"), dict)
                    for step in receipts
                )
            ):
                raise ChildRecoveryError(
                    "child checkpoint has no valid committed receipts; reconcile before retrying"
                )
            config.initial_checkpoint = latest
    else:
        if store.load_latest(run_id=run.subagent_id) is not None:
            raise ChildRecoveryError(
                "child checkpoints have no delegation record; reconcile before retrying"
            )
        store.save(
            Checkpoint(
                step_number=0,
                session_log_data={},
                conversation_data=None,
                config_snapshot={},
                tool_results_store={},
                metadata={"binding": digest},
                run_envelope=config.run_envelope.to_dict(),
            ),
            "delegation",
        )
    evidence = _ChildRecoveryEvidence(
        config.resolve_callable("checkpoint_state"),
        config.resolve_callable("restore_checkpoint_state"),
    )
    config.checkpoint_state = evidence.checkpoint_state
    config.restore_checkpoint_state = evidence.restore_checkpoint_state
    run.inputs["hooks"].append(evidence)
    run.recovery_store = store
    run.recovery_binding = digest


def _save_child_result(run: _ChildRun, result: dict[str, Any]) -> None:
    from looplet.checkpoint import Checkpoint

    if run.recovery_store is None:
        return
    run.recovery_store.save(
        Checkpoint(
            step_number=len(result["steps"]),
            session_log_data={},
            conversation_data=None,
            config_snapshot={},
            tool_results_store={},
            metadata={"binding": run.recovery_binding, "child_result": result},
            run_status=result["status"],
            run_phase=result["phase"],
            termination_reason=result["stop_reason"],
            run_envelope=result["run_envelope"],
        ),
        "delegation",
    )


def _is_async_backend(backend: Any) -> bool:
    seen: set[int] = set()
    for _ in range(8):
        if backend is None or id(backend) in seen:
            return False
        seen.add(id(backend))
        if any(
            inspect.iscoroutinefunction(getattr(backend, name, None))
            for name in ("generate", "generate_with_tools")
        ):
            return True
        backend = (
            getattr(backend, "_backend", None)
            or getattr(backend, "_inner", None)
            or getattr(backend, "backend", None)
        )
    return False


def _intersect_execution_policy(
    parent: ExecutionPolicy | None,
    child: ExecutionPolicy | None,
) -> ExecutionPolicy | None:
    if parent is None:
        return child
    if child is None:
        return parent

    def root(parent_root: str | None, child_root: str | None) -> str | None:
        if parent_root is None:
            return child_root
        if child_root is None:
            return parent_root
        try:
            Path(child_root).resolve().relative_to(Path(parent_root).resolve())
        except ValueError as exc:
            raise ValueError("child execution path leaves the parent scope") from exc
        return child_root

    return replace(
        parent,
        workspace_root=root(parent.workspace_root, child.workspace_root),
        scratch_root=root(parent.scratch_root, child.scratch_root),
        capabilities=parent.capabilities & child.capabilities,
        environment=parent.environment & child.environment,
    )


class _ParentPermissionHook:
    """Borrow only permission decisions. The parent owns the hook lifecycle."""

    def __init__(self, parent: Any) -> None:
        self.parent = parent
        self.check_permission = parent.check_permission


def _inherit_child_security(
    config: Any,
    hooks: list[Any],
    parent_hooks: list[Any] | tuple[Any, ...],
    policy: ChildRunPolicy | None,
) -> None:
    from looplet.approval import ApprovalHook

    engines = [hook.engine for hook in parent_hooks if isinstance(hook, PermissionHook)]
    if policy is not None:
        engines.extend(policy.permission_engines)
        config.execution_policy = _intersect_execution_policy(
            policy.execution_policy, config.execution_policy
        )
        if policy.approval_handler is not None:
            config.approval_handler = policy.approval_handler
    installed = {id(hook.engine) for hook in hooks if isinstance(hook, PermissionHook)}
    for engine in engines:
        if id(engine) not in installed:
            hooks.append(PermissionHook(engine))
            installed.add(id(engine))
    sources = {id(hook.parent) for hook in hooks if isinstance(hook, _ParentPermissionHook)}
    sources.update(id(hook) for hook in hooks)
    for parent in parent_hooks:
        source = parent.parent if isinstance(parent, _ParentPermissionHook) else parent
        permission = getattr(source, "check_permission", None)
        if not callable(permission):
            continue
        if (
            isinstance(source, PermissionHook)
            and getattr(permission, "__func__", None) is PermissionHook.check_permission
        ):
            continue
        if id(source) not in sources:
            hooks.append(_ParentPermissionHook(source))
            sources.add(id(source))
    if any(isinstance(hook, ApprovalHook) for hook in parent_hooks) and not any(
        isinstance(hook, ApprovalHook) for hook in hooks
    ):
        hooks.append(ApprovalHook())


def _inherit_child_policy(
    policy: ChildRunPolicy | None, parent_hooks: list[Any] | tuple[Any, ...]
) -> ChildRunPolicy | None:
    inherited = [hook.policy for hook in parent_hooks if isinstance(hook, _ChildBudgetStopHook)]
    if not inherited:
        return policy
    policy = policy or ChildRunPolicy(parent_id=inherited[-1].parent_id)
    for parent in inherited:
        if (
            parent.model_budget is not None
            and policy.model_budget is not None
            and (parent.model_budget is not policy.model_budget)
        ):
            raise ValueError("child cannot replace the parent model allowance")
        if (
            parent.cancel_token is not None
            and policy.cancel_token is not None
            and (parent.cancel_token is not policy.cancel_token)
        ):
            raise ValueError("child cannot replace the parent cancellation token")
        allowed = policy.allowed_tools
        if parent.allowed_tools is not None:
            allowed = parent.allowed_tools if allowed is None else allowed & parent.allowed_tools
        step_limits = [limit for limit in (parent.max_steps, policy.max_steps) if limit is not None]
        deadlines = [limit for limit in (parent.deadline, policy.deadline) if limit is not None]
        policy = replace(
            policy,
            allowed_tools=allowed,
            max_steps=min(step_limits) if step_limits else None,
            deadline=min(deadlines) if deadlines else None,
            model_budget=parent.model_budget or policy.model_budget,
            cancel_token=parent.cancel_token or policy.cancel_token,
            execution_policy=_intersect_execution_policy(
                parent.execution_policy, policy.execution_policy
            ),
            permission_engines=(*parent.permission_engines, *policy.permission_engines),
            approval_handler=parent.approval_handler or policy.approval_handler,
        )
    return policy


def _policy_from_tool_context(ctx: Any, policy: ChildRunPolicy | None) -> ChildRunPolicy | None:
    policy = _inherit_child_policy(policy, ctx.parent_hooks)
    if policy is None and not (
        ctx.execution_policy
        or ctx.cancel_token
        or ctx.request_approval
        or ctx.parent_hooks
        or ctx.run_envelope
    ):
        return None
    parent_id = (
        ctx.run_envelope.run_id
        if ctx.run_envelope is not None
        else policy.parent_id
        if policy is not None
        else "parent"
    )
    policy = policy or ChildRunPolicy(parent_id=parent_id)
    if (
        policy.cancel_token is not None
        and ctx.cancel_token is not None
        and (policy.cancel_token is not ctx.cancel_token)
    ):
        raise ValueError("child policy cannot replace the parent cancellation token")
    return replace(
        policy,
        parent_id=parent_id,
        cancel_token=ctx.cancel_token or policy.cancel_token,
        execution_policy=_intersect_execution_policy(ctx.execution_policy, policy.execution_policy),
        approval_handler=ctx.request_approval or policy.approval_handler,
    )


def _prepare_sub_loop(
    llm: Any,
    task: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    max_steps: int = 5,
    system_prompt: str = "",
    hooks: list[Any] | None = None,
    parent_hooks: list[Any] | None = None,
    context: Any = None,
    state: Any = None,
    sub_tools: Any = None,
    build_summary: Callable[[Any, Any, list[dict]], dict[str, Any]] | None = None,
    state_mutating_tools: list[str] | None = None,
    conversation: Any | None = None,
    subagent_id: str | None = None,
    config: Any | None = None,
    policy: ChildRunPolicy | None = None,
    async_mode: bool = False,
) -> _ChildRun:
    """Prepare isolated child inputs and optional recovery records.

    A full config overrides the shorthand step limit and system prompt.
    Parent hooks supply permissions, inherited limits, and event observation.
    Parent flow-control methods do not run on child state.
    Caller-supplied state, tools, hooks, and resources remain caller-owned.
    """
    from looplet import DefaultState, register_done_tool
    from looplet.loop import LoopConfig
    from looplet.session import SessionLog

    policy = _inherit_child_policy(policy, parent_hooks or [])
    if task is None:
        task = {}
    if subagent_id is not None and (
        not subagent_id or Path(subagent_id).name != subagent_id or "\\" in subagent_id
    ):
        raise ValueError("subagent_id must be a nonempty path-safe name")
    if policy is not None:
        if not policy.parent_id or (policy.max_steps is not None and policy.max_steps <= 0):
            raise ValueError("child policy requires a parent_id and positive max_steps")
        if policy.recovery_dir is not None and (not policy.recovery_version or not subagent_id):
            raise ValueError("child recovery requires a stable subagent_id and recovery_version")
        if policy.deadline is not None and time.monotonic() >= policy.deadline:
            raise ChildRunLimitExceeded("child run deadline exceeded")
        if (
            policy.recovery_dir is None
            and policy.model_budget is not None
            and policy.model_budget.remaining <= 0
        ):
            raise ChildRunLimitExceeded("model call budget exhausted")

    # Create minimal isolated state if not provided
    if state is None:
        state = _MinimalState(
            task=task, max_steps=config.max_steps if config is not None else max_steps
        )
    session_log = SessionLog()

    # Create isolated tool registry if not provided
    if sub_tools is None:
        exclude = state_mutating_tools or ["done"]
        sub_tools = clone_tools_excluding(tools, exclude)
        terminal_name = config.done_tool if config is not None else "done"
        if terminal_name not in sub_tools._tools:
            register_done_tool(sub_tools, name=terminal_name)
    if policy is not None and policy.allowed_tools is not None:
        sub_tools = _restrict_tools(sub_tools, policy.allowed_tools)

    # Fork conversation for sub-agent isolation (if provided)
    _sub_conv = None
    if conversation is not None and hasattr(conversation, "fork"):
        _sub_conv = conversation.fork()

    # Generate a stable id for lifecycle events so the caller can
    # correlate SUBAGENT_START / SUBAGENT_STOP payloads.
    if subagent_id is None:
        import uuid  # noqa: PLC0415

        subagent_id = uuid.uuid4().hex[:12]

    # Wrap each parent hook so its ``on_event`` receives every event
    # the sub-loop emits, tagged with this subagent_id so consumers
    # can route / nest. We do NOT forward the full LoopHook interface
    # (pre_loop, check_done, etc.) - those would conflate parent + sub
    # state. Only event-stream observers see the activity.
    sub_hooks: list[Any] = list(hooks or [])
    if parent_hooks:
        sub_hooks.append(_ParentHookForwarder(parent_hooks, subagent_id, async_mode=async_mode))

    # Allow callers to pass a full LoopConfig for parity with
    # composable_loop. If provided, its values override the shorthand
    # kwargs (max_steps, system_prompt).
    if config is not None:
        sub_config = replace(config)
    else:
        sub_config = LoopConfig(
            max_steps=max_steps,
            system_prompt=system_prompt,
        )
    if policy is not None:
        if policy.max_steps is not None:
            sub_config.max_steps = min(sub_config.max_steps, policy.max_steps)
        if policy.cancel_token is not None:
            sub_config.cancel_token = policy.cancel_token
        if policy.recovery_dir is not None:
            sub_config.checkpoint_dir = str(
                Path(policy.recovery_dir)
                / f"child_{hashlib.sha256(subagent_id.encode()).hexdigest()[:24]}"
            )
        if policy.model_budget is not None:
            llm = policy.model_budget.wrap(
                llm, deadline=policy.deadline, cancel_token=sub_config.cancel_token
            )
            if sub_config.router is not None:
                sub_config.router = _BudgetedRouter(sub_config.router, policy.model_budget, policy)
        sub_hooks.append(_ChildBudgetStopHook(policy))

    if isinstance(state, (DefaultState, _MinimalState)):
        state.max_steps = sub_config.max_steps

    parent_envelope = sub_config.run_envelope
    parent_id = (
        policy.parent_id
        if policy is not None
        else (
            parent_envelope.parent_run_id
            if parent_envelope is not None and parent_envelope.run_id == subagent_id
            else parent_envelope.run_id
            if parent_envelope is not None
            else None
        )
    )
    sub_config.run_envelope = (
        replace(parent_envelope, run_id=subagent_id, parent_run_id=parent_id)
        if parent_envelope is not None
        else RunEnvelope(run_id=subagent_id, parent_run_id=parent_id)
    )
    if (
        sub_config.initial_checkpoint is not None
        and sub_config.initial_checkpoint.run_id != subagent_id
    ):
        sub_config.initial_checkpoint = None
    if sub_config.checkpoint_dir is not None and (policy is None or policy.recovery_dir is None):
        sub_config.checkpoint_dir = str(Path(sub_config.checkpoint_dir) / f"child_{subagent_id}")
    _inherit_child_security(sub_config, sub_hooks, parent_hooks or [], policy)
    run = _ChildRun(
        inputs={
            "llm": llm,
            "task": task,
            "tools": sub_tools,
            "context": context,
            "hooks": sub_hooks,
            "config": sub_config,
            "state": state,
            "session_log": session_log,
            "conversation": _sub_conv,
        },
        observers=list(parent_hooks or []) + (hooks or []),
        subagent_id=subagent_id,
        policy=policy,
        build_summary=build_summary,
    )
    _prepare_child_recovery(run)
    if (
        run.result is None
        and policy is not None
        and policy.model_budget is not None
        and policy.model_budget.remaining <= 0
    ):
        raise ChildRunLimitExceeded("model call budget exhausted")
    return run


def _finish_child_run(run: _ChildRun, trace: Any) -> dict[str, Any]:
    from dataclasses import asdict

    from looplet.approval import ApprovalHook
    from looplet.types import RunResult

    state = run.inputs["state"]
    session_log = run.inputs["session_log"]
    outcome = RunResult.from_state(state)
    findings: list[str] = []
    highlights: list[str] = []
    for entry in getattr(session_log, "entries", ()):
        findings.extend(getattr(entry, "findings", None) or [])
        highlights.extend(getattr(entry, "highlights", None) or [])
    if run.build_summary is not None:
        result = run.build_summary(state, session_log, run.steps)
    else:
        entities = sorted(session_log.all_entities())
        summary = f"Entities: {', '.join(entities[:10])}" if entities else "No findings"
        if outcome.completed and isinstance(outcome.output, dict) and outcome.output.get("summary"):
            summary = str(outcome.output["summary"])
        result = {"summary": summary, "entities": entities}
    result["steps"] = run.steps
    result["llm_calls"] = trace.get("llm_calls", 0) if isinstance(trace, dict) else 0
    result["stop_reason"] = getattr(state, "_stop_reason", None) or "budget_exhausted"
    if outcome.run_envelope is not None and outcome.run_envelope.parent_run_id is not None:
        result["parent_id"] = outcome.run_envelope.parent_run_id
    result.setdefault("findings", findings)
    result.setdefault("highlights", highlights)
    result["subagent_id"] = run.subagent_id
    result["status"] = outcome.status.value
    result["completed"] = outcome.completed
    result["output"] = outcome.output
    result.setdefault("result", outcome.output)
    result["phase"] = outcome.phase.value
    result["run_envelope"] = outcome.run_envelope.to_dict() if outcome.run_envelope else None
    result["approval"] = next(
        (
            asdict(hook.pending)
            for hook in run.inputs["hooks"]
            if isinstance(hook, ApprovalHook) and hook.pending is not None
        ),
        None,
    )
    result["recovered"] = False
    result["new_llm_calls"] = result["llm_calls"]
    _save_child_result(run, result)
    run.result = result
    return result


def _child_stop_payload(run: _ChildRun) -> dict[str, Any]:
    result = run.result
    state = run.inputs["state"]
    reason = (
        result["stop_reason"]
        if result is not None
        else run.error_reason or getattr(state, "_stop_reason", None) or "error"
    )
    return {
        "state": state,
        "context": run.inputs["context"],
        "subagent_id": run.subagent_id,
        "termination_reason": reason,
        "extra": {
            "llm_calls": result["llm_calls"] if result is not None else 0,
            "new_llm_calls": result["new_llm_calls"] if result is not None else 0,
            "recovered": result["recovered"] if result is not None else False,
            "step_count": len(result["steps"]) if result is not None else len(run.steps),
            "entities": result.get("entities", []) if result is not None else [],
            "parent_id": result.get("parent_id")
            if result is not None
            else run.inputs["config"].run_envelope.parent_run_id,
        },
    }


def run_sub_loop(
    llm: Any,
    task: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    max_steps: int = 5,
    system_prompt: str = "",
    hooks: list[Any] | None = None,
    parent_hooks: list[Any] | None = None,
    context: Any = None,
    state: Any = None,
    sub_tools: Any = None,
    build_summary: Callable[[Any, Any, list[dict]], dict[str, Any]] | None = None,
    state_mutating_tools: list[str] | None = None,
    conversation: Any | None = None,
    subagent_id: str | None = None,
    config: Any | None = None,
    policy: ChildRunPolicy | None = None,
) -> dict[str, Any]:
    """Run a synchronous child loop with separate state and tools."""
    from looplet.events import LifecycleEvent
    from looplet.loop import composable_loop, emit_event

    run = _prepare_sub_loop(
        llm,
        task,
        tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        hooks=hooks,
        parent_hooks=parent_hooks,
        context=context,
        state=state,
        sub_tools=sub_tools,
        build_summary=build_summary,
        state_mutating_tools=state_mutating_tools,
        conversation=conversation,
        subagent_id=subagent_id,
        config=config,
        policy=policy,
    )
    generator = None
    try:
        emit_event(
            run.observers,
            LifecycleEvent.SUBAGENT_START,
            state=run.inputs["state"],
            context=context,
            subagent_id=run.subagent_id,
            extra={
                "parent_id": run.inputs["config"].run_envelope.parent_run_id,
                "recovered": run.result is not None,
            },
        )
        if run.result is not None:
            return run.result
        generator = composable_loop(**run.inputs)
        while True:
            try:
                run.steps.append(next(generator).to_dict())
            except StopIteration as finished:
                return _finish_child_run(run, finished.value)
    except BaseException:
        run.error_reason = "error"
        raise
    finally:
        try:
            if generator is not None:
                generator.close()
        finally:
            emit_event(run.observers, LifecycleEvent.SUBAGENT_STOP, **_child_stop_payload(run))


async def run_sub_loop_async(
    llm: Any,
    task: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    max_steps: int = 5,
    system_prompt: str = "",
    hooks: list[Any] | None = None,
    parent_hooks: list[Any] | None = None,
    context: Any = None,
    state: Any = None,
    sub_tools: Any = None,
    build_summary: Callable[[Any, Any, list[dict]], dict[str, Any]] | None = None,
    state_mutating_tools: list[str] | None = None,
    conversation: Any | None = None,
    subagent_id: str | None = None,
    config: Any | None = None,
    policy: ChildRunPolicy | None = None,
) -> dict[str, Any]:
    """Run an async child loop and await its tools and hooks."""
    from looplet.async_loop import async_composable_loop
    from looplet.events import LifecycleEvent
    from looplet.loop import emit_event_async

    run = _prepare_sub_loop(
        llm,
        task,
        tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        hooks=hooks,
        parent_hooks=parent_hooks,
        context=context,
        state=state,
        sub_tools=sub_tools,
        build_summary=build_summary,
        state_mutating_tools=state_mutating_tools,
        conversation=conversation,
        subagent_id=subagent_id,
        config=config,
        policy=policy,
        async_mode=True,
    )
    generator = None
    try:
        await emit_event_async(
            run.observers,
            LifecycleEvent.SUBAGENT_START,
            state=run.inputs["state"],
            context=context,
            subagent_id=run.subagent_id,
            extra={
                "parent_id": run.inputs["config"].run_envelope.parent_run_id,
                "recovered": run.result is not None,
            },
        )
        if run.result is not None:
            return run.result
        generator = async_composable_loop(**run.inputs)
        async for step in generator:
            run.steps.append(step.to_dict())
        return _finish_child_run(run, getattr(run.inputs["state"], "_looplet_run_stats", None))
    except asyncio.CancelledError:
        run.error_reason = "cancelled"
        raise
    except BaseException:
        run.error_reason = "error"
        raise
    finally:
        try:
            if generator is not None:
                await generator.aclose()
        finally:
            await emit_event_async(
                run.observers, LifecycleEvent.SUBAGENT_STOP, **_child_stop_payload(run)
            )


async def run_sub_loops_async(
    requests: list[dict[str, Any]],
    *,
    max_concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Run child requests concurrently and return results in request order.

    Each request supplies the keyword arguments for ``run_sub_loop_async``.
    Failed children cancel their siblings. Shared resources remain caller-owned.
    """
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency < 1
    ):
        raise ValueError("max_concurrency must be a positive integer")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(request: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await run_sub_loop_async(**request)
            if result["status"] in {"failed", "cancelled"}:
                raise ChildRunFailed(result)
            return result

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(execute(request)) for request in requests]
    return [task.result() for task in tasks]


class _ChildBudgetStopHook:
    def __init__(self, policy: ChildRunPolicy) -> None:
        self.policy = policy

    def should_stop(self, state: Any, step_num: int, new_entities: int) -> Any:
        from looplet.hook_decision import HookDecision  # noqa: PLC0415

        if self.policy.cancel_token is not None and self.policy.cancel_token.is_cancelled:
            return HookDecision(stop="cancelled")
        if self.policy.deadline is not None and time.monotonic() >= self.policy.deadline:
            return HookDecision(stop="deadline_exceeded")
        if self.policy.model_budget is not None and self.policy.model_budget.remaining <= 0:
            return HookDecision(stop="model_budget_exhausted")
        return None


def _restrict_tools(registry: Any, allowed: frozenset[str]) -> Any:
    from looplet.tools import BaseToolRegistry  # noqa: PLC0415

    if not isinstance(registry, BaseToolRegistry):
        raise TypeError("child tool attenuation requires a BaseToolRegistry")
    restricted = BaseToolRegistry()
    restricted.set_resources(registry._resources)
    for name, spec in registry._tools.items():
        if name in allowed:
            restricted.register(replace(spec))
    return restricted


class _ParentHookForwarder:
    """Forwards a sub-loop's lifecycle events to the parent's hooks.

    Implements only ``on_event`` (not the full ``LoopHook`` interface):
    the sub-loop's per-step events are surfaced to parent observability
    (MetricsHook, StreamingHook, TrajectoryRecorder, …) without letting
    the parent's flow-control methods (``check_done``, ``pre_prompt``,
    etc.) re-enter on sub state.

    The forwarded :class:`EventPayload` is augmented with
    ``subagent_id`` in its ``extra`` dict so consumers can distinguish
    nested activity from the parent's own.
    """

    __slots__ = ("_parent_hooks", "_subagent_id", "_async_mode")

    def __init__(
        self, parent_hooks: list[Any], subagent_id: str, *, async_mode: bool = False
    ) -> None:
        self._parent_hooks = list(parent_hooks)
        self._subagent_id = subagent_id
        self._async_mode = async_mode

    def on_event(self, payload: Any) -> Any:
        # Tag the payload's extra dict so parent observers can filter /
        # nest sub-activity. Mutating the live payload is acceptable
        # here - it's about to be discarded by every other observer at
        # the end of this dispatch.
        try:
            extra = getattr(payload, "extra", None)
            if isinstance(extra, dict):
                extra.setdefault("subagent_id", self._subagent_id)
        except Exception:  # noqa: BLE001
            pass
        if self._async_mode:
            return self._dispatch_async(payload)
        for hook in self._parent_hooks:
            handler = getattr(hook, "on_event", None)
            if handler is None:
                continue
            try:
                handler(payload)
            except Exception:  # noqa: BLE001
                # Parent hook misbehaviour must never break the sub-loop.
                logger.warning(
                    "parent hook %r raised during sub-loop event forward "
                    "(subagent_id=%s); swallowing",
                    type(hook).__name__,
                    self._subagent_id,
                )

    async def _dispatch_async(self, payload: Any) -> None:
        for hook in self._parent_hooks:
            handler = getattr(hook, "on_event", None)
            if handler is None:
                continue
            try:
                response = handler(payload)
                if inspect.isawaitable(response):
                    await response
            except Exception:
                logger.exception("parent observer failed for child %s", self._subagent_id)


class _MinimalState:
    """Minimal agent state for sub-loops.

    Provides the interface the composable_loop expects from state:
    budget_remaining, step_count, steps, queries_used,
    context_summary(), snapshot().
    """

    def __init__(
        self, task: dict[str, Any] | None = None, max_steps: int = 5, **kwargs: Any
    ) -> None:
        self.task = task or {}
        self.max_steps = max_steps
        self.steps: list = []
        self.queries_used: int = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    def context_summary(self) -> str:
        if not self.steps:
            return "(no steps taken yet)"
        parts: list[str] = []
        for s in self.steps[-3:]:
            parts.append(s.summary())
        return "\n".join(parts)

    def snapshot(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "budget_remaining": self.budget_remaining,
        }


def clone_tools_excluding(parent_tools: Any, exclude: list[str]) -> Any:
    """Clone a tool registry, excluding specified tool names.

    Warns (via the module logger) when ``exclude`` contains names that
    are not present in ``parent_tools``. Typos here are silent
    correctness bugs - e.g. a caller passing
    ``exclude=["finish"]`` when the parent's done tool is named
    ``"finalize"`` would otherwise ship the state-mutating tool into
    the sub-agent without any signal.
    """
    from looplet.tools import BaseToolRegistry

    parent_names = set(parent_tools._tools.keys())
    missing = [name for name in exclude if name not in parent_names]
    if missing:
        logger.warning(
            "clone_tools_excluding: names %s are not registered on the parent "
            "(available: %s) - nothing to exclude for those entries. Typo?",
            missing,
            sorted(parent_names),
        )

    sub = BaseToolRegistry()
    for name, spec in parent_tools._tools.items():
        if name in exclude:
            continue
        cloned = copy(spec)
        cloned.requires = list(getattr(spec, "requires", ()))
        cloned.tags = list(getattr(spec, "tags", ()))
        cloned.render = dict(getattr(spec, "render", {}))
        cloned.capabilities = list(getattr(spec, "capabilities", ()))
        sub.register(cloned)
    return sub
