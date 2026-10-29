"""Shared, inspectable defaults for Python and cartridge agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
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


@dataclass
class _HarnessRun:
    config: Any
    hooks: list[Any]
    llm: Any
    state: Any
    session_log: Any
    sink: Any = None
    tree: Any = None
    restore_config: list[tuple[str, Any, Any]] = field(default_factory=list)

    def finish(self) -> None:
        try:
            if self.sink is not None:
                self.sink.flush()
            if self.tree is not None:
                from looplet.session_tree import save_tree

                save_tree(self.tree, Path(self.config.run_dir) / "session.jsonl")
        finally:
            for name, previous, installed in reversed(self.restore_config):
                if getattr(self.config, name) is installed:
                    setattr(self.config, name, previous)


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
    from looplet._storage import DiskBackedDict, DiskBackedList
    from looplet.checkpoint import CheckpointHook, FileCheckpointStore
    from looplet.provenance import ProvenanceSink, TrajectoryRecorder, _recording_backend
    from looplet.session import SessionLog
    from looplet.session_tree import SessionTree, SessionTreeRecorder, TreeNode
    from looplet.types import DefaultState

    prepared = _HarnessRun(config, list(hooks), llm, state, session_log)
    disabled = set(config.disabled_defaults)
    if callable(getattr(tools, "set_resources", None)):
        resources = tools.resources
        store = resources.get("tool_result_store")
        if isinstance(store, _AutomaticResultStore) and store._owner is tools:
            resources.pop("tool_result_store")
            tools.set_resources(resources)
    if config.harness_profile == "minimal" or "persistence" in disabled or tools is None:
        return prepared
    if not callable(getattr(tools, "set_resources", None)):
        return prepared
    root = Path(config.run_dir or Path(".looplet/runs") / uuid4().hex).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    policy = root / ".gitignore"
    if not policy.exists():
        policy.write_text("*\n", encoding="utf-8")
    effective = replace(config, run_dir=str(root))
    if (
        "checkpointing" not in disabled
        and effective.checkpoint_dir is None
        and not any(
            isinstance(hook, CheckpointHook) or getattr(hook, "checkpoint_owner", False) is True
            for hook in prepared.hooks
        )
    ):
        effective.checkpoint_dir = str(root / "checkpoints")
    if effective.initial_checkpoint is None and effective.checkpoint_dir is not None:
        effective.initial_checkpoint = FileCheckpointStore(
            effective.checkpoint_dir
        ).load_latest_for_resume(
            run_id=effective.run_envelope.run_id if effective.run_envelope is not None else None
        )
    prepared.config = effective
    prepared.state = state if state is not None else DefaultState(max_steps=effective.max_steps)
    if type(prepared.state) is DefaultState and "history" not in disabled:
        prepared.state.steps = DiskBackedList(
            lambda step: step.to_full_dict(),
            _restore_step,
            prepared.state.steps,
            preserve_python=True,
            weight=DefaultState._step_charge,
        )
    prepared.session_log = session_log if session_log is not None else SessionLog()
    resources = tools.resources
    result_store = None
    if "result_retention" not in disabled and "tool_result_store" not in resources:
        result_store = _AutomaticResultStore(str(root / "results" / uuid4().hex), owner=tools)
    metadata = getattr(prepared.state, "metadata", None)
    if isinstance(metadata, dict):
        metadata["run_dir"] = str(root)
    if "provenance" not in disabled and not any(
        isinstance(hook, TrajectoryRecorder) for hook in prepared.hooks
    ):
        prepared.sink = ProvenanceSink(dir=root / "traces" / uuid4().hex)
        if (recording := _recording_backend(llm)) is not None:
            recorder = TrajectoryRecorder(recording_llm=recording)
            prepared.sink._recording_llm = recording
            prepared.sink._hook = recorder
        else:
            prepared.llm = prepared.sink.wrap_llm(llm)
            recorder = prepared.sink.trajectory_hook()
        prepared.hooks.append(recorder)
        if isinstance(metadata, dict):
            metadata["trace_dir"] = str(prepared.sink._dir)
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
        prepared.hooks.append(SessionTreeRecorder(prepared.tree))
    for name in ("run_dir", "checkpoint_dir", "initial_checkpoint"):
        previous, installed = getattr(config, name), getattr(effective, name)
        if previous is not installed:
            prepared.restore_config.append((name, previous, installed))
            setattr(config, name, installed)
    prepared.config = config
    if result_store is not None:
        resources["tool_result_store"] = result_store
        tools.set_resources(resources)
    return prepared
