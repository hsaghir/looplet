"""Checkpoint - save and restore loop state for crash recovery and long-running tasks.

Provides:
  - Checkpoint: serializable snapshot of loop state at a given step
  - CheckpointStore: Protocol for checkpoint storage backends
  - FileCheckpointStore: JSON file-based storage
  - CheckpointHook: LoopHook that auto-saves checkpoints every N steps
  - resume_loop_state: reconstruct runnable state from a Checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from looplet.session import SessionLog
    from looplet.types import AgentState, LLMBackend, ToolCall, ToolResult

logger = logging.getLogger(__name__)

# ── Checkpoint dataclass ────────────────────────────────────────────


@dataclass
class Checkpoint:
    """Serializable snapshot of agent loop state at a given step.

    All fields are JSON-safe - no pickle, no binary formats.
    """

    step_number: int
    """The loop step at which this checkpoint was taken."""

    session_log_data: dict[str, Any]
    """Serialized SessionLog: {"entries": [...], "current_theory": str}."""

    conversation_data: dict[str, Any] | None
    """Serialized Conversation or None if not used."""

    config_snapshot: dict[str, Any]
    """JSON-safe LoopConfig fields: max_steps, max_tokens, temperature, done_tool, system_prompt."""

    tool_results_store: dict[str, Any]
    """Mapping of recall_key -> result data for stored tool outputs."""

    metadata: dict[str, Any]
    """Arbitrary metadata: task_id, version, timestamp, etc."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when this checkpoint was created."""

    run_status: str = "created"
    """Lifecycle status at checkpoint time."""

    run_phase: str = "starting"
    """Lifecycle phase at checkpoint time."""

    termination_reason: str | None = None
    """Terminal reason when the checkpoint represents loop shutdown."""

    domain_state: dict[str, Any] = field(default_factory=dict)
    """JSON-safe state supplied by a cartridge for crash-resume."""

    run_envelope: dict[str, Any] | None = None
    """JSON-safe host identity and policy context for this run."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        payload = {
            "step_number": self.step_number,
            "session_log_data": self.session_log_data,
            "conversation_data": self.conversation_data,
            "config_snapshot": self.config_snapshot,
            "tool_results_store": self.tool_results_store,
            "domain_state": self.domain_state,
            "run_envelope": self.run_envelope,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "run_status": self.run_status,
            "run_phase": self.run_phase,
            "termination_reason": self.termination_reason,
        }
        try:
            return json.loads(json.dumps(payload, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint contains non-JSON-safe data") from exc

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Deserialize from a dictionary produced by to_dict()."""
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be a JSON object")
        for field_name in (
            "session_log_data",
            "config_snapshot",
            "tool_results_store",
            "metadata",
            "domain_state",
        ):
            if field_name in data and not isinstance(data[field_name], dict):
                raise ValueError(f"checkpoint {field_name} must be a JSON object")
        if data.get("conversation_data") is not None and not isinstance(
            data.get("conversation_data"), dict
        ):
            raise ValueError("checkpoint conversation_data must be an object or null")
        if data.get("run_envelope") is not None and not isinstance(data.get("run_envelope"), dict):
            raise ValueError("checkpoint run_envelope must be an object or null")
        if "step_number" not in data:
            raise ValueError("checkpoint is missing step_number")
        if isinstance(data["step_number"], bool) or not isinstance(data["step_number"], int):
            raise ValueError("checkpoint step_number must be an integer")
        if isinstance(data.get("created_at", time.time()), bool) or not isinstance(
            data.get("created_at", time.time()), (int, float)
        ):
            raise ValueError("checkpoint created_at must be a number")
        for field_name in ("run_status", "run_phase"):
            if field_name in data and not isinstance(data[field_name], str):
                raise ValueError(f"checkpoint {field_name} must be a string")
        if data.get("termination_reason") is not None and not isinstance(
            data.get("termination_reason"), str
        ):
            raise ValueError("checkpoint termination_reason must be a string or null")
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint contains non-JSON-safe data") from exc
        return cls(
            step_number=data["step_number"],
            session_log_data=data.get("session_log_data", {}),
            conversation_data=data.get("conversation_data"),
            config_snapshot=data.get("config_snapshot", {}),
            tool_results_store=data.get("tool_results_store", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            run_status=str(data.get("run_status", "created")),
            run_phase=str(data.get("run_phase", "starting")),
            termination_reason=data.get("termination_reason"),
            domain_state=data.get("domain_state", {}),
            run_envelope=data.get("run_envelope"),
        )


# ── CheckpointStore Protocol ────────────────────────────────────────


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends.

    Any storage implementation must provide save() and load().
    """

    def save(self, checkpoint: Checkpoint, key: str) -> None:
        """Persist a checkpoint under the given key."""
        ...

    def load(self, key: str) -> Checkpoint | None:
        """Load a checkpoint by key; returns None if not found."""
        ...


# ── FileCheckpointStore ─────────────────────────────────────────────


class FileCheckpointStore:
    """Saves and loads checkpoints as JSON files in a directory.

    Files are named ``{key}.json`` inside the configured directory.
    The directory is created if it does not exist.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint, key: str) -> None:
        """Write checkpoint to ``{directory}/{key}.json``."""
        safe_key = Path(key).name  # strip any directory separators to prevent traversal
        path = self._dir / f"{safe_key}.json"
        fd, temporary_name = tempfile.mkstemp(
            dir=self._dir,
            prefix=f".{safe_key}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint.to_dict(), indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.debug("checkpoint saved: %s", path)

    def load(self, key: str) -> Checkpoint | None:
        """Read checkpoint from ``{directory}/{key}.json``; None if missing."""
        safe_key = Path(key).name  # strip any directory separators to prevent traversal
        path = self._dir / f"{safe_key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Checkpoint.from_dict(data)

    def load_latest(self) -> Checkpoint | None:
        """Load the checkpoint with the highest step number, or None.

        Scans all ``*.json`` files in the directory, parses each, and
        returns the one with the largest ``step_number``. Used by the
        loop for auto-resume when ``checkpoint_dir`` is set.
        """
        best: Checkpoint | None = None
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                cp = Checkpoint.from_dict(data)
                if cp.run_status == "completed" or cp.termination_reason == "done":
                    continue
                if best is None or cp.step_number > best.step_number:
                    best = cp
            except Exception:  # noqa: BLE001
                logger.warning("Skipping corrupt checkpoint: %s", path)
        return best


# ── CheckpointHook ─────────────────────────────────────────────────


class CheckpointHook:
    """Loop hook that auto-saves checkpoints every N steps.

    Implements the LoopHook duck-type interface. Only post_dispatch()
    is active - all other methods are no-ops that preserve loop behaviour.

    Args:
        store: CheckpointStore to save to.
        get_checkpoint_data: Callable(step_num) -> Checkpoint that
            extracts current loop state at save time.
        save_every_n_steps: Save interval (default 5). Saves when
            step_number % save_every_n_steps == 0.
    """

    def __init__(
        self,
        store: CheckpointStore,
        get_checkpoint_data: Callable[[int], Checkpoint],
        save_every_n_steps: int = 5,
    ) -> None:
        if save_every_n_steps < 1:
            raise ValueError("save_every_n_steps must be >= 1")
        self._store = store
        self._get_data = get_checkpoint_data
        self.save_every_n_steps = save_every_n_steps
        self._pending_steps: list[int] = []

    # ── LoopHook interface ─────────────────────────────────────────

    def pre_prompt(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        return None

    def pre_dispatch(
        self,
        state: AgentState,
        session_log: SessionLog,
        tool_call: ToolCall,
        step_num: int,
    ) -> None:
        return None

    def post_dispatch(
        self,
        state: AgentState,
        session_log: SessionLog,
        tool_call: ToolCall,
        tool_result: ToolResult,
        step_num: int,
    ) -> str | None:
        """Save a checkpoint if step_num is a multiple of save_every_n_steps."""
        n = step_num
        if n % self.save_every_n_steps == 0:
            if isinstance(getattr(state, "steps", None), list):
                self._pending_steps.append(n)
            else:
                self._save(n)
        return None

    def post_step(self, state: AgentState, session_log: SessionLog, step_num: int) -> None:
        """Flush a deferred checkpoint after the loop records the step."""
        if step_num in self._pending_steps:
            self._pending_steps.remove(step_num)
            self._save(step_num)

    def _save(self, step_num: int) -> None:
        cp = self._get_data(step_num)
        key = f"step_{step_num}"
        self._store.save(cp, key)
        logger.debug("auto-checkpoint at step %d → key=%s", step_num, key)

    def check_done(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        return None

    def should_stop(
        self,
        state: AgentState,
        step_num: int,
        new_entities: int,
    ) -> bool:
        return False

    def on_loop_end(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        llm: LLMBackend,
    ) -> int:
        return 0


# ── resume_loop_state ───────────────────────────────────────────────


def resume_loop_state(checkpoint: Checkpoint) -> dict[str, Any]:
    """Reconstruct runnable loop state from a checkpoint.

    Returns a dict with:
      - ``session_log``: reconstructed SessionLog
      - ``conversation``: reconstructed Conversation (message thread)
        if the checkpoint captured one; ``None`` otherwise
      - ``step_offset``: step number to continue from
      - ``state_counters``: dict with ``queries_used`` and
        ``budget_remaining`` if present in ``config_snapshot`` (so the
        loop can restore its budget/query accounting)
            - ``tool_results_store``: JSON-safe result-store snapshot
            - ``domain_state``: JSON-safe cartridge state from the checkpoint
      - ``metadata``: checkpoint metadata dict

    The returned dict can be passed to composable_loop to resume
    execution from where it left off. In particular, ``conversation``
    should be forwarded so multi-turn LLM context is preserved.
    """
    from looplet.session import SessionLog

    log = SessionLog()
    entries = checkpoint.session_log_data.get("entries", [])
    for entry_data in entries:
        log.record(
            step=entry_data["step"],
            theory=entry_data.get("theory", ""),
            tool=entry_data["tool"],
            reasoning=entry_data.get("reasoning", ""),
            entities=entry_data.get("entities_seen", []),
            findings=entry_data.get("findings", []),
            highlights=entry_data.get("highlights", []),
            recall_key=entry_data.get("recall_key", ""),
        )
    log.current_theory = checkpoint.session_log_data.get("current_theory", "")

    cfg = checkpoint.config_snapshot or {}
    state_counters = {k: cfg[k] for k in ("queries_used", "budget_remaining") if k in cfg}

    conv = None
    if checkpoint.conversation_data:
        from looplet.conversation import Conversation  # noqa: PLC0415

        conv = Conversation.deserialize(checkpoint.conversation_data)

    return {
        "session_log": log,
        "conversation": conv,
        "step_offset": checkpoint.step_number,
        "state_counters": state_counters,
        "tool_results_store": checkpoint.tool_results_store,
        "domain_state": checkpoint.domain_state,
        "run_envelope": checkpoint.run_envelope,
        "metadata": checkpoint.metadata,
    }
