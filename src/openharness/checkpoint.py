"""Checkpoint — save and restore loop state for crash recovery and long-running tasks.

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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from openharness.session import SessionLog
    from openharness.types import AgentState, LLMBackend, ToolCall, ToolResult

logger = logging.getLogger(__name__)

# ── Checkpoint dataclass ────────────────────────────────────────────

@dataclass
class Checkpoint:
    """Serializable snapshot of agent loop state at a given step.

    All fields are JSON-safe — no pickle, no binary formats.
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "step_number": self.step_number,
            "session_log_data": self.session_log_data,
            "conversation_data": self.conversation_data,
            "config_snapshot": self.config_snapshot,
            "tool_results_store": self.tool_results_store,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Deserialize from a dictionary produced by to_dict()."""
        return cls(
            step_number=data["step_number"],
            session_log_data=data.get("session_log_data", {}),
            conversation_data=data.get("conversation_data"),
            config_snapshot=data.get("config_snapshot", {}),
            tool_results_store=data.get("tool_results_store", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
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
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
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
                if best is None or cp.step_number > best.step_number:
                    best = cp
            except Exception:  # noqa: BLE001
                logger.warning("Skipping corrupt checkpoint: %s", path)
        return best

# ── CheckpointHook ─────────────────────────────────────────────────

class CheckpointHook:
    """Loop hook that auto-saves checkpoints every N steps.

    Implements the LoopHook duck-type interface. Only post_dispatch()
    is active — all other methods are no-ops that preserve loop behaviour.

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
        self._store = store
        self._get_data = get_checkpoint_data
        self.save_every_n_steps = save_every_n_steps

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
            cp = self._get_data(n)
            key = f"step_{n}"
            self._store.save(cp, key)
            logger.debug("auto-checkpoint at step %d → key=%s", n, key)
        return None

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
      - ``metadata``: checkpoint metadata dict

    The returned dict can be passed to composable_loop to resume
    execution from where it left off. In particular, ``conversation``
    should be forwarded so multi-turn LLM context is preserved.
    """
    from openharness.session import SessionLog

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
    state_counters = {
        k: cfg[k] for k in ("queries_used", "budget_remaining") if k in cfg
    }

    conv = None
    if checkpoint.conversation_data:
        from openharness.conversation import Conversation  # noqa: PLC0415
        conv = Conversation.deserialize(checkpoint.conversation_data)

    return {
        "session_log": log,
        "conversation": conv,
        "step_offset": checkpoint.step_number,
        "state_counters": state_counters,
        "metadata": checkpoint.metadata,
    }
