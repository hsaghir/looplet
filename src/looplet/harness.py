"""Shared, inspectable defaults for Python and cartridge agent runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from looplet.budget import ContextBudget, ThresholdCompactHook
from looplet.compact import DefaultCompactService
from looplet.limits import ToolCallBudgetHook
from looplet.result_store import FileToolResultStore
from looplet.stagnation import StagnationHook

if TYPE_CHECKING:
    from looplet.loop import LoopConfig

__all__ = ["ResolvedHarness", "resolve_harness"]

_active_registries: dict[int, object] = {}
_registry_lock = Lock()


def _release_registry(registry_id: int | None, token: object | None) -> None:
    if registry_id is None:
        return
    with _registry_lock:
        if token is not None and _active_registries.get(registry_id) is token:
            del _active_registries[registry_id]


@dataclass(frozen=True)
class ResolvedHarness:
    """Effective per-run configuration and hooks, without starting a run."""

    config: LoopConfig
    hooks: list[Any]

    def describe(self) -> dict[str, Any]:
        """Return a JSON-compatible explanation of the effective harness."""
        service = self.config.compact_service
        persistence = (
            self.config.harness_profile != "minimal"
            and "persistence" not in self.config.disabled_defaults
        )
        return {
            "profile": self.config.harness_profile,
            "disabled_defaults": list(self.config.disabled_defaults),
            "context_window": self.config.context_window,
            "max_steps": self.config.max_steps,
            "max_tokens": self.config.max_tokens,
            "max_turn_continuations": self.config.max_turn_continuations,
            "typed_history": self.config.uses_typed_history,
            "max_model_calls": self.config.max_model_calls,
            "max_tool_calls": next(
                (hook.max_calls for hook in self.hooks if isinstance(hook, ToolCallBudgetHook)),
                None,
            ),
            "compact_service": type(service).__name__ if service is not None else None,
            "hooks": [type(hook).__name__ for hook in self.hooks],
            "persistence": {
                "enabled": persistence,
                "run_dir": self.config.run_dir,
                "defaults": [
                    name
                    for name in (
                        "history",
                        "result_retention",
                        "provenance",
                        "checkpointing",
                        "branching",
                    )
                    if persistence and name not in self.config.disabled_defaults
                ],
            },
        }


def resolve_harness(
    config: LoopConfig | None = None,
    hooks: list[Any] | None = None,
) -> ResolvedHarness:
    """Compose standard defaults underneath explicit host configuration.

    ``execution-v1`` installs deterministic compaction without steering or
    derived tool quotas. ``standard-v1`` retains a non-blocking
    stagnation nudge, and a per-invocation tool-dispatch budget. ``minimal``
    keeps only explicitly supplied policies.
    Config and hook containers are not mutated; live host objects retain
    their identity. Resolution does not call a model or access the filesystem.
    """
    from looplet.loop import LoopConfig

    effective = replace(config) if config is not None else LoopConfig()
    resolved_hooks = list(hooks or ())
    if effective.harness_profile not in {"execution-v1", "standard-v1", "minimal"}:
        raise ValueError("harness_profile must be 'execution-v1', 'standard-v1', or 'minimal'")
    disabled = set(effective.disabled_defaults)
    unknown = disabled - {
        "compaction",
        "stagnation",
        "tool_budget",
        "persistence",
        "history",
        "result_retention",
        "provenance",
        "checkpointing",
        "branching",
    }
    if unknown:
        raise ValueError(f"unknown disabled_defaults: {sorted(unknown)}")
    if effective.harness_profile == "minimal":
        return ResolvedHarness(effective, resolved_hooks)

    if "compaction" not in disabled:
        if effective.compact_service is None:
            effective.compact_service = DefaultCompactService(use_llm_summary=False)
        if effective.context_window >= 4 and not any(
            callable(getattr(hook, "should_compact", None)) for hook in resolved_hooks
        ):
            window = effective.context_window
            reserved_output = effective.max_tokens if effective.max_tokens is not None else 3000
            window = max(4, window - reserved_output)
            buffer = min(13_000, max(0, (window - 1) // 10))
            budget = ContextBudget(
                context_window=window,
                warning_at=int(window * 0.6),
                error_at=int(window * 0.8),
                compact_buffer=buffer,
            )
            resolved_hooks.append(
                ThresholdCompactHook(
                    budget,
                    fire_tier="warning" if effective.harness_profile == "execution-v1" else "error",
                )
            )

    if effective.harness_profile == "execution-v1":
        return ResolvedHarness(effective, resolved_hooks)

    if "stagnation" not in disabled and not any(
        isinstance(hook, StagnationHook) for hook in resolved_hooks
    ):
        resolved_hooks.append(
            StagnationHook(
                fingerprint="result",
                threshold=6,
                ignore_tools={effective.done_tool, *effective.done_tools, "think"},
            )
        )
    if "tool_budget" not in disabled and not any(
        isinstance(hook, ToolCallBudgetHook) for hook in resolved_hooks
    ):
        resolved_hooks.append(ToolCallBudgetHook(max_calls=max(16, effective.max_steps * 4)))
    return ResolvedHarness(effective, resolved_hooks)


class _AutomaticResultStore(FileToolResultStore):
    """Keep completed results readable until the registry starts another run."""

    def __init__(self, directory: str, max_bytes: int | None = None, *, owner: Any = None) -> None:
        super().__init__(directory, max_bytes=max_bytes)
        self._owner = owner


class _AsyncPersistenceHook:
    def __init__(self, hook: Any) -> None:
        self.hook = hook

    def __getattr__(self, name: str) -> Any:
        callback = getattr(self.hook, name)
        if name not in {"post_dispatch", "on_event", "on_loop_end"}:
            return callback

        @wraps(callback)
        async def persist(*args: Any, **kwargs: Any) -> Any:
            from looplet.checkpoint import _run_persistence

            session_text = None
            if name == "on_loop_end" and hasattr(self.hook, "trajectory"):
                session_log = args[1]
                if session_log is not None and hasattr(session_log, "render"):
                    try:
                        session_text = str(session_log.render() or "")
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "Failed to render session log for trajectory", exc_info=True
                        )
                args = (args[0], None, *args[2:])
            try:
                return await _run_persistence(callback, *args, **kwargs)
            finally:
                if session_text is not None:
                    self.hook.trajectory.session_log_text = session_text

        return persist


@dataclass
class _HarnessRun:
    config: Any
    hooks: list[Any]
    llm: Any
    state: Any
    session_log: Any
    sink: Any = None
    tree: Any = None
    checkpoint_store: Any = None
    initial_checkpoint: Any = None
    compact_service: Any = None
    persistence_hooks: list[Any] = field(default_factory=list)
    registry_id: int | None = None
    registry_token: object | None = None
    run_dir: Path | None = None
    checkpoint_factory: Any = None
    resources: dict[str, Any] | None = None
    resources_changed: bool = False

    def bind_checkpoint(self, config: LoopConfig) -> LoopConfig:
        for hook in self.hooks:
            binder = getattr(hook, "bind_checkpoint_store", None)
            if getattr(hook, "checkpoint_owner", False) is True and callable(binder):
                binder(self.checkpoint_store, config.resolve_callable("checkpoint_state"))
                self.checkpoint_store = hook.checkpoint_store
                config = replace(config, checkpoint_state=hook.capture_state)
                break
        return config

    def enable_async_persistence(self) -> None:
        enable = getattr(self.checkpoint_store, "enable_async_persistence", None)
        if callable(enable):
            enable()
        owned = {id(hook) for hook in self.persistence_hooks}
        self.hooks = [
            _AsyncPersistenceHook(hook) if id(hook) in owned else hook for hook in self.hooks
        ]

    def persist(self) -> None:
        if self.sink is not None:
            self.sink.flush()
        if self.tree is not None:
            from looplet.session_tree import save_tree

            assert self.run_dir is not None
            save_tree(self.tree, self.run_dir / "session.jsonl")

    def release(self) -> None:
        _release_registry(self.registry_id, self.registry_token)

    def finish(self) -> None:
        try:
            self.persist()
        finally:
            self.release()


def _restore_step(value: dict[str, Any]) -> Any:
    from looplet.types import ErrorKind, Step, ToolCall, ToolError, ToolResult

    result = dict(value["tool_result"])
    result["args_summary"] = result.pop("args", "")
    detail = result.get("error_detail")
    if detail is not None:
        result["error_detail"] = ToolError(**{**detail, "kind": ErrorKind(detail["kind"])})
    return Step(
        number=value["step"],
        tool_call=ToolCall(**value["tool_call"]),
        tool_result=ToolResult(
            **{
                name: item
                for name, item in result.items()
                if name in ToolResult.__dataclass_fields__
            }
        ),
        metadata=value.get("metadata", {}),
    )


def prepare_harness_run(
    config: LoopConfig,
    hooks: list[Any],
    llm: Any,
    tools: Any,
    state: Any,
    session_log: Any,
    conversation: Any,
) -> _HarnessRun:
    """Bind missing persistence primitives without replacing explicit host choices."""
    prepared = _claim_harness_run(config, hooks, llm, tools, state, session_log)
    try:
        _initialize_harness_storage(prepared)
        return _bind_harness_run(prepared, tools)
    except BaseException:
        prepared.release()
        raise


async def prepare_harness_run_async(
    config: LoopConfig,
    hooks: list[Any],
    llm: Any,
    tools: Any,
    state: Any,
    session_log: Any,
    conversation: Any,
) -> _HarnessRun:
    """Initialize owned storage in a drained worker and bind host objects on the caller."""
    from looplet.checkpoint import _run_persistence

    prepared = _claim_harness_run(config, hooks, llm, tools, state, session_log)
    try:
        await _run_persistence(_initialize_harness_storage, prepared)
        return _bind_harness_run(prepared, tools)
    except BaseException:
        prepared.release()
        raise


def _claim_harness_run(
    config: LoopConfig,
    hooks: list[Any],
    llm: Any,
    tools: Any,
    state: Any,
    session_log: Any,
) -> _HarnessRun:
    prepared = _HarnessRun(config, hooks, llm, state, session_log)
    registry_id = id(tools) if tools is not None else None
    token = object()
    if registry_id is not None:
        with _registry_lock:
            if registry_id in _active_registries:
                raise RuntimeError(
                    "Tool registry already has an active loop; build a fresh registry "
                    "for concurrent runs"
                )
            _active_registries[registry_id] = token
    prepared.registry_id = registry_id
    prepared.registry_token = token
    try:
        return _configure_harness_run(prepared, tools)
    except BaseException:
        prepared.release()
        raise


def _configure_harness_run(
    prepared: _HarnessRun,
    tools: Any,
) -> _HarnessRun:
    from looplet.checkpoint import CheckpointHook, FileCheckpointStore, _IncrementalCheckpointStore

    config = prepared.config
    resolved = resolve_harness(config, prepared.hooks)
    prepared.hooks = resolved.hooks
    if config.compact_service is None:
        prepared.compact_service = resolved.config.compact_service
    disabled = set(config.disabled_defaults)
    if callable(getattr(tools, "set_resources", None)):
        resources = tools.resources
        prepared.resources = resources
        store = resources.get("tool_result_store")
        if isinstance(store, _AutomaticResultStore) and store._owner is tools:
            resources.pop("tool_result_store")
            prepared.resources_changed = True
    if config.checkpoint_dir is not None:
        prepared.checkpoint_factory = FileCheckpointStore
    if (
        config.harness_profile != "minimal"
        and "persistence" not in disabled
        and prepared.resources is not None
    ):
        prepared.run_dir = Path(config.run_dir or Path(".looplet/runs") / uuid4().hex)
        if (
            "checkpointing" not in disabled
            and prepared.checkpoint_factory is None
            and not any(
                isinstance(hook, CheckpointHook) or getattr(hook, "checkpoint_owner", False) is True
                for hook in prepared.hooks
            )
        ):
            prepared.checkpoint_factory = _IncrementalCheckpointStore
    return prepared


def _initialize_harness_storage(prepared: _HarnessRun) -> None:
    from looplet._storage import DiskBackedDict, DiskBackedList
    from looplet.checkpoint import _IncrementalCheckpointStore
    from looplet.provenance import ProvenanceSink, TrajectoryRecorder, _recording_backend
    from looplet.session import SessionLog
    from looplet.session_tree import SessionTree, SessionTreeRecorder, TreeNode
    from looplet.types import DefaultState

    config = prepared.config
    disabled = set(config.disabled_defaults)
    root = prepared.run_dir
    if root is not None:
        root = root.resolve()
        prepared.run_dir = root
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            policy = root / ".gitignore"
            if not policy.exists():
                policy.write_text("*\n", encoding="utf-8")
        except OSError as error:
            raise OSError(
                error.errno,
                "Cannot initialize run evidence. Set LoopConfig(run_dir=...) to a writable "
                "directory, or set LoopConfig(disabled_defaults=('persistence',)) to run "
                "without saved evidence.",
                error.filename,
            ) from error
    prepared.initial_checkpoint = config.initial_checkpoint
    if prepared.checkpoint_factory is not None:
        if prepared.checkpoint_factory is _IncrementalCheckpointStore:
            assert root is not None
            directory = root / "checkpoints"
        else:
            directory = config.checkpoint_dir
        prepared.checkpoint_store = prepared.checkpoint_factory(directory)
        if prepared.initial_checkpoint is None:
            prepared.initial_checkpoint = prepared.checkpoint_store.load_latest_for_resume(
                run_id=(config.run_envelope.run_id if config.run_envelope is not None else None)
            )
            if prepared.initial_checkpoint is not None:
                logging.getLogger(__name__).info(
                    "Resuming run from checkpoint at step %d",
                    prepared.initial_checkpoint.step_number,
                )
    if root is None:
        return
    if prepared.state is None:
        prepared.state = DefaultState(max_steps=config.max_steps)
    if type(prepared.state) is DefaultState and "history" not in disabled:
        prepared.state.steps = DiskBackedList(
            lambda step: step.to_full_dict(),
            _restore_step,
            prepared.state.steps,
            preserve_python=True,
            weight=DefaultState._step_charge,
        )
    if prepared.session_log is None:
        prepared.session_log = SessionLog()
    resources = prepared.resources
    assert resources is not None
    if "result_retention" not in disabled and "tool_result_store" not in resources:
        resources["tool_result_store"] = _AutomaticResultStore(str(root / "results" / uuid4().hex))
        prepared.resources_changed = True
    if "provenance" not in disabled and not any(
        isinstance(hook, TrajectoryRecorder) for hook in prepared.hooks
    ):
        trace_directory = root / "traces" / uuid4().hex
        if (recording := _recording_backend(prepared.llm)) is not None:
            prepared.sink = ProvenanceSink(
                dir=trace_directory,
                max_chars_per_call=recording._max_chars,
                redact=recording._redact,
                redact_upstream=False,
            )
        else:
            prepared.sink = ProvenanceSink(dir=trace_directory)
        prepared.llm = prepared.sink.wrap_llm(prepared.llm)
        recorder = prepared.sink.trajectory_hook()
        prepared.hooks.append(recorder)
        prepared.persistence_hooks.append(recorder)
    if "branching" not in disabled and not any(
        isinstance(hook, SessionTreeRecorder) for hook in prepared.hooks
    ):
        prepared.tree = SessionTree()
        nodes = DiskBackedDict(vars, TreeNode.from_dict)
        nodes.update(prepared.tree.nodes)
        children = DiskBackedDict(lambda value: value, lambda value: value)
        children.update(prepared.tree.children)
        prepared.tree.nodes = nodes
        prepared.tree.children = children
        recorder = SessionTreeRecorder(prepared.tree)
        prepared.hooks.append(recorder)
        prepared.persistence_hooks.append(recorder)


def _bind_harness_run(prepared: _HarnessRun, tools: Any) -> _HarnessRun:
    from looplet.checkpoint import _IncrementalCheckpointStore

    original_llm = (
        prepared.sink._recording_llm._backend if prepared.sink is not None else prepared.llm
    )
    effective = prepared.config
    if prepared.run_dir is not None:
        effective = replace(
            effective,
            run_dir=str(prepared.run_dir),
            compact_service=(
                prepared.compact_service
                if effective.compact_service is None
                else effective.compact_service
            ),
        )
        if prepared.checkpoint_factory is _IncrementalCheckpointStore:
            effective = replace(effective, checkpoint_dir=str(prepared.run_dir / "checkpoints"))
    effective = prepared.bind_checkpoint(effective)
    if prepared.run_dir is None:
        prepared.config = effective
    metadata = getattr(prepared.state, "metadata", None)
    if isinstance(metadata, dict) and prepared.run_dir is not None:
        metadata["run_dir"] = str(prepared.run_dir)
        if prepared.sink is not None:
            metadata["trace_dir"] = str(prepared.sink._dir)
    if prepared.sink is not None:
        from looplet.harness_snapshot import serialize_harness

        prepared.sink.trajectory_hook().trajectory.metadata["harness_snapshot"] = serialize_harness(
            config=effective,
            hooks=prepared.hooks,
            tools=tools,
            memory_sources=effective.memory_sources,
            llm=original_llm,
            extra={"resolved_harness": ResolvedHarness(effective, prepared.hooks).describe()},
        )
    if prepared.resources_changed:
        assert prepared.resources is not None
        store = prepared.resources.get("tool_result_store")
        if isinstance(store, _AutomaticResultStore):
            store._owner = tools
        tools.set_resources(prepared.resources)
    return prepared
