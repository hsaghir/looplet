"""Checkpoint - save and restore loop state for crash recovery and long-running tasks.

Provides:
  - Checkpoint: serializable snapshot of loop state at a given step
  - CheckpointStore: Protocol for checkpoint storage backends
  - FileCheckpointStore: JSON file-based storage
  - CheckpointHook: LoopHook that auto-saves checkpoints every N steps
  - resume_loop_state: reconstruct runnable state from a Checkpoint
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import glob
import inspect
import json
import logging
import ntpath
import os
import tempfile
import time
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from looplet.session import SessionLog
    from looplet.types import AgentState, LLMBackend, ToolCall, ToolResult

logger = logging.getLogger(__name__)
_DOCUMENT_SCHEMA = "looplet.checkpoint-documents.v1"


class CheckpointIdentityError(ValueError):
    """A checkpoint belongs to a different logical run."""


class CheckpointRestoreError(RuntimeError):
    """Participant restoration failed; discard the possibly partial host state."""


@dataclass(frozen=True)
class CheckpointParticipant:
    """One named, versioned host component with synchronous state callbacks.

    Validation must be side-effect free. Capture must describe a host-owned
    consistent snapshot; this helper does not make external systems atomic.
    """

    name: str
    version: str
    capture: Callable[[Any], dict[str, Any]]
    validate: Callable[[Any, dict[str, Any]], None]
    restore: Callable[[Any, dict[str, Any]], None]

    @classmethod
    def from_object(cls, name: str, version: str, component: Any) -> "CheckpointParticipant":
        """Bind context-free checkpoint/validate/restore methods, such as a budget."""
        capture = getattr(component, "checkpoint_state", None)
        validate = getattr(component, "validate_checkpoint_state", None)
        restore = getattr(component, "restore_checkpoint_state", None)
        if not callable(capture) or not callable(validate) or not callable(restore):
            raise ValueError("checkpoint participant requires capture, validation and restore")
        capture = cast(Callable[[], dict[str, Any]], capture)
        validate = cast(Callable[[dict[str, Any]], None], validate)
        restore = cast(Callable[[dict[str, Any]], None], restore)
        return cls(
            name,
            version,
            lambda ctx: capture(),
            lambda ctx, state: validate(state),
            lambda ctx, state: restore(state),
        )


class CheckpointParticipants:
    """Compose host state through existing LoopConfig checkpoint callbacks.

    Bind ``capture`` and ``restore`` to the corresponding config callbacks.
    Exact participant names, versions, host identity and every component state
    are validated before any restoration. A restore failure blocks this helper
    from reuse, but cannot roll back effects already performed by a callback.
    """

    _schema = "looplet.checkpoint-participants.v1"

    def __init__(self, participants: Iterable[CheckpointParticipant], *, identity: str) -> None:
        self.participants = tuple(participants)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("checkpoint participant identity must be a nonempty string")
        self.identity = identity
        names = set()
        for participant in self.participants:
            if (
                not isinstance(participant.name, str)
                or not participant.name.strip()
                or not isinstance(participant.version, str)
                or not participant.version.strip()
                or not all(
                    callable(callback)
                    for callback in (participant.capture, participant.validate, participant.restore)
                )
            ):
                raise ValueError("checkpoint participants require names, versions and callbacks")
            if participant.name in names:
                raise ValueError(f"duplicate checkpoint participant: {participant.name}")
            names.add(participant.name)
        self._failed_restore = False

    def _check_usable(self) -> None:
        if self._failed_restore:
            raise CheckpointRestoreError(
                "Previous restore failed; discard this helper and host state"
            )

    @staticmethod
    def _snapshot(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("checkpoint participant state must be an object")
        try:
            detached = json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint participant state must contain finite JSON values"
            ) from error
        if detached != value:
            raise ValueError("checkpoint participant state must contain lossless JSON values")
        return detached

    @staticmethod
    def _call(callback: Callable[..., Any], *args: Any) -> Any:
        result = callback(*args)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise ValueError("checkpoint participant callbacks must be synchronous")
        return result

    def capture(self, context: Any) -> dict[str, Any]:
        """Detach each component immediately; the host supplies a quiescent boundary."""
        self._check_usable()
        states = {}
        for participant in self.participants:
            state = self._snapshot(self._call(participant.capture, context))
            self._call(participant.validate, context, self._snapshot(state))
            states[participant.name] = {"version": participant.version, "state": state}
        return {"schema": self._schema, "identity": self.identity, "participants": states}

    def _prepare(self, context: Any, state: dict[str, Any]) -> dict[str, Any]:
        self._check_usable()
        saved = self._snapshot(state)
        if (
            set(saved) != {"schema", "identity", "participants"}
            or saved["schema"] != self._schema
            or saved["identity"] != self.identity
            or not isinstance(saved["participants"], dict)
            or set(saved["participants"]) != {item.name for item in self.participants}
        ):
            raise ValueError("incompatible checkpoint participant manifest or identity")
        for participant in self.participants:
            entry = saved["participants"][participant.name]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"version", "state"}
                or entry["version"] != participant.version
                or not isinstance(entry["state"], dict)
            ):
                raise ValueError(f"incompatible checkpoint participant: {participant.name}")
        for participant in self.participants:
            self._call(
                participant.validate,
                context,
                self._snapshot(saved["participants"][participant.name]["state"]),
            )
        return saved

    def validate(self, context: Any, state: dict[str, Any]) -> None:
        """Preflight the entire detached manifest without restoring any component."""
        self._prepare(context, state)

    def restore(self, context: Any, state: dict[str, Any]) -> None:
        """Restore only after prevalidation; block reuse if any restoration fails."""
        saved = self._prepare(context, state)
        try:
            for participant in self.participants:
                self._call(
                    participant.restore,
                    context,
                    self._snapshot(saved["participants"][participant.name]["state"]),
                )
        except BaseException as error:
            self._failed_restore = True
            if not isinstance(error, Exception):
                raise
            raise CheckpointRestoreError(
                f"Restoring participant {participant.name!r} failed; discard this helper and host state"
            ) from error


class UncertainModelCall(RuntimeError):
    """A started model request lacks an acknowledged durable response."""


class UncertainToolEffect(RuntimeError):
    """A pending dispatch must be reconciled before this run can resume."""

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        ids = ", ".join(str(call.get("call_id")) for call in calls)
        super().__init__(
            f"Tool effects unknown for calls {ids}; reconcile before resuming. "
            "Keep the pending checkpoint. See "
            "https://hsaghir.com/looplet/host-integration/#reconcile-an-interrupted-tool"
        )


def validate_checkpoint_identity(checkpoint: "Checkpoint", run_id: str | None) -> None:
    """Reject a checkpoint carrying an identity different from ``run_id``."""
    checkpoint_run_id = checkpoint.run_id
    if run_id is not None and checkpoint_run_id != run_id:
        raise CheckpointIdentityError(
            f"checkpoint belongs to run {checkpoint_run_id!r}, not {run_id!r}"
        )


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

    @property
    def run_id(self) -> str | None:
        """Return the logical run identity carried by this checkpoint."""
        if isinstance(self.run_envelope, dict):
            value = self.run_envelope.get("run_id")
            if value is not None:
                return str(value)
        return None

    @property
    def is_terminal(self) -> bool:
        """Whether this checkpoint is an authoritative terminal snapshot."""
        return self.run_status in {"completed", "failed", "cancelled"} or (
            self.run_phase == "terminal" or self.termination_reason is not None
        )

    def _payload(self) -> dict[str, Any]:
        """Return the payload before portable JSON normalization."""
        from looplet.assistant_turns import TURN_STATE_KEY, TURN_STATE_SCHEMA

        step_number: Any = self.step_number
        if TURN_STATE_KEY in self.metadata:
            step_number = {"tool_step": self.step_number, "required_semantics": TURN_STATE_SCHEMA}
        return {
            "step_number": step_number,
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        try:
            return json.loads(json.dumps(self._payload(), allow_nan=False))
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
        if isinstance(data["step_number"], dict):
            from looplet.assistant_turns import TURN_STATE_KEY, TURN_STATE_SCHEMA

            marker = data["step_number"]
            if marker.get(
                "required_semantics"
            ) != TURN_STATE_SCHEMA or TURN_STATE_KEY not in data.get("metadata", {}):
                raise ValueError("unsupported checkpoint required semantics")
            data = {**data, "step_number": marker.get("tool_step")}
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

    def load(self, key: str, *, run_id: str | None = None) -> Checkpoint | None:
        """Load a checkpoint by key; returns None if not found."""
        ...


# ── FileCheckpointStore ─────────────────────────────────────────────


class FileCheckpointStore:
    """Save plain JSON and load plain or automatic incremental checkpoints.

    Files are named ``{key}.json`` inside the configured directory.
    The directory is created if it does not exist.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _serialize(self, checkpoint: Checkpoint) -> dict[str, Any]:
        return checkpoint.to_dict()

    def _read_existing(self, path: Path) -> Checkpoint:
        return self._read(path)

    def _read(self, path: Path, *, summary_only: bool = False) -> Checkpoint:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == _DOCUMENT_SCHEMA:
            from looplet._checkpoint_storage import CheckpointDocuments

            if set(data) != {"schema", "root"}:
                raise ValueError("invalid checkpoint document manifest")
            documents = CheckpointDocuments(self._dir)
            fields = None
            if summary_only:
                from looplet.assistant_turns import TURN_STATE_KEY

                fields = {
                    "step_number": None,
                    "session_log_data": {},
                    "conversation_data": {},
                    "config_snapshot": {},
                    "tool_results_store": {},
                    "domain_state": {},
                    "run_envelope": {"run_id": None},
                    "metadata": {TURN_STATE_KEY: {"turn_count": None, "model_calls": None}},
                    "created_at": None,
                    "run_status": None,
                    "run_phase": None,
                    "termination_reason": None,
                }
            checkpoint = Checkpoint.from_dict(documents.load(data["root"], fields=fields))
            if _checkpoint_summary(checkpoint) != documents.load(data["root"], summary_only=True):
                raise ValueError("checkpoint summary does not match its payload")
            return checkpoint
        return Checkpoint.from_dict(data)

    def _verify_records(self, path: Path, verified: set[str]) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == _DOCUMENT_SCHEMA:
            from looplet._checkpoint_storage import CheckpointDocuments

            CheckpointDocuments(self._dir).verify(data["root"], verified)

    @staticmethod
    def _validate_run_id(run_id: str | None) -> None:
        if run_id is not None and (
            not isinstance(run_id, str)
            or not run_id
            or run_id in {".", ".."}
            or ntpath.splitdrive(run_id)[0]
            or any(separator in run_id for separator in ("/", "\\", "\0"))
        ):
            raise ValueError("checkpoint run_id must be a nonempty path component")

    @staticmethod
    def key_for(checkpoint: Checkpoint, key: str) -> str:
        """Return a run-scoped checkpoint key, preserving legacy keys."""
        FileCheckpointStore._validate_run_id(checkpoint.run_id)
        safe_key = Path(key).name
        return f"{checkpoint.run_id}__{safe_key}" if checkpoint.run_id else safe_key

    def save(self, checkpoint: Checkpoint, key: str) -> None:
        """Write checkpoint to ``{directory}/{key}.json``."""
        safe_key = self.key_for(checkpoint, key)
        path = self._dir / f"{safe_key}.json"
        fd, temporary_name = tempfile.mkstemp(
            dir=self._dir,
            prefix=f".{safe_key}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._serialize(checkpoint), indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self._read_existing(path)
                if existing.run_id != checkpoint.run_id:
                    raise ValueError("checkpoint path already belongs to a different run")
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.debug("checkpoint saved: %s", path)

    def load(self, key: str, *, run_id: str | None = None) -> Checkpoint | None:
        """Read checkpoint from ``{directory}/{key}.json``; None if missing."""
        self._validate_run_id(run_id)
        safe_key = Path(key).name  # strip any directory separators to prevent traversal
        path = (
            self._dir / f"{run_id}__{safe_key}.json"
            if run_id is not None
            else self._dir / f"{safe_key}.json"
        )
        if run_id is not None and not path.exists():
            path = self._dir / f"{safe_key}.json"
        if not path.exists():
            namespaced = sorted(self._dir.glob(f"*__{glob.escape(safe_key)}.json"))
            if len(namespaced) == 1:
                path = namespaced[0]
            elif len(namespaced) > 1:
                raise ValueError(f"checkpoint key {key!r} is ambiguous; pass run_id explicitly")
        if not path.exists():
            return None
        checkpoint = self._read(path)
        validate_checkpoint_identity(checkpoint, run_id)
        return checkpoint

    def load_latest(self, *, run_id: str | None = None) -> Checkpoint | None:
        """Load the highest-step checkpoint for one logical run.

        Scans all ``*.json`` files in the directory, parses each, and
        returns the one with the largest ``step_number``. If ``run_id`` is
        supplied, only checkpoints carrying that identity are considered.
        This is an inspection API and may return a terminal checkpoint;
        callers that want crash-resume semantics must use
        :meth:`load_latest_for_resume`.
        """
        self._validate_run_id(run_id)
        best: Checkpoint | None = None
        best_path: Path | None = None
        verified: set[str] = set()
        for path in sorted(self._dir.glob("*.json")):
            try:
                cp = self._read(path, summary_only=True)
                if run_id is not None and cp.run_id != run_id:
                    continue
                self._verify_records(path, verified)
                if best is None or _checkpoint_order(cp) > _checkpoint_order(best):
                    best = cp
                    best_path = path
            except Exception:  # noqa: BLE001
                logger.warning("Skipping corrupt checkpoint: %s", path)
        return self._read(best_path) if best_path is not None else None

    def load_latest_for_resume(self, *, run_id: str | None = None) -> Checkpoint | None:
        """Load the latest incomplete checkpoint unless the run is terminal.

        A terminal checkpoint is authoritative: once present, older running
        snapshots are not eligible for implicit auto-resume. ``run_id``
        scopes selection to one logical run; leaving it unset preserves the
        legacy identity-less checkpoint-directory behavior.
        Without a selected identity, every corrupt file blocks recovery.
        With an identity, other run namespaces remain isolated.
        """
        self._validate_run_id(run_id)
        best: Checkpoint | None = None
        terminal: Checkpoint | None = None
        best_path: Path | None = None
        terminal_path: Path | None = None
        verified: set[str] = set()
        for path in sorted(self._dir.glob("*.json")):
            try:
                cp = self._read(path, summary_only=True)
                if (run_id is None and cp.run_id is not None) or (
                    run_id is not None and cp.run_id != run_id
                ):
                    continue
                self._verify_records(path, verified)
                if cp.is_terminal:
                    if terminal is None or _checkpoint_order(cp) > _checkpoint_order(terminal):
                        terminal = cp
                        terminal_path = path
                    continue
                if best is None or _checkpoint_order(cp) > _checkpoint_order(best):
                    best = cp
                    best_path = path
            except (OSError, ValueError, TypeError) as error:
                if (
                    run_id is not None
                    and "__" in path.stem
                    and not path.stem.startswith(f"{run_id}__")
                ):
                    continue
                raise ValueError(
                    f"Cannot safely resume from checkpoint {path}. "
                    "Preserve the file and reconcile recovery state before retrying."
                ) from error
        if terminal_path is not None:
            from looplet.assistant_turns import TURN_STATE_KEY

            terminal = self._read(terminal_path)
            return terminal if TURN_STATE_KEY in terminal.metadata else None
        return self._read(best_path) if best_path is not None else None


def _checkpoint_summary(checkpoint: Checkpoint) -> dict[str, Any]:
    from looplet.assistant_turns import TURN_STATE_KEY

    metadata = {}
    if TURN_STATE_KEY in checkpoint.metadata:
        turns = checkpoint.metadata[TURN_STATE_KEY]
        metadata[TURN_STATE_KEY] = (
            {name: turns.get(name, 0) for name in ("turn_count", "model_calls")}
            if isinstance(turns, dict)
            else turns
        )
    return {
        "step_number": checkpoint.step_number,
        "session_log_data": {},
        "config_snapshot": {},
        "tool_results_store": {},
        "metadata": metadata,
        "created_at": checkpoint.created_at,
        "run_status": checkpoint.run_status,
        "run_phase": checkpoint.run_phase,
        "termination_reason": checkpoint.termination_reason,
        "run_envelope": {"run_id": checkpoint.run_id} if checkpoint.run_id is not None else None,
    }


class _IncrementalCheckpointStore(FileCheckpointStore):
    """Automatic manifests backed by immutable, shared JSON records."""

    def _read_existing(self, path: Path) -> Checkpoint:
        checkpoint = self._read(path, summary_only=True)
        self._verify_records(path, set())
        return checkpoint

    def _serialize(self, checkpoint: Checkpoint) -> dict[str, Any]:
        from looplet._checkpoint_storage import CheckpointDocuments

        root = CheckpointDocuments(self._dir).save(
            checkpoint._payload(), _checkpoint_summary(checkpoint)
        )
        return {"schema": _DOCUMENT_SCHEMA, "root": root}


def _checkpoint_order(checkpoint: Checkpoint) -> tuple[int, int, int, float]:
    from looplet.assistant_turns import TURN_STATE_KEY

    turns = checkpoint.metadata.get(TURN_STATE_KEY, {})
    if not isinstance(turns, dict):
        raise ValueError("invalid checkpoint assistant-turn state")
    counters = (turns.get("turn_count", 0), turns.get("model_calls", 0))
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("invalid checkpoint model-turn cursor")
    return (*counters, checkpoint.step_number, checkpoint.created_at)


def _capture_checkpoint(
    loop_ctx: Any,
    *,
    step_number: int,
    metadata: dict[str, Any],
    checkpoint_state: Callable[..., Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> Checkpoint:
    state = loop_ctx.state
    return Checkpoint(
        step_number=step_number,
        session_log_data={
            "entries": loop_ctx.session_log.to_list(),
            "current_theory": loop_ctx.session_log.current_theory,
        },
        conversation_data=loop_ctx.conversation.serialize(),
        config_snapshot=(
            config_snapshot
            if config_snapshot is not None
            else {
                "max_steps": loop_ctx.config.max_steps,
                "queries_used": getattr(state, "queries_used", 0),
                "budget_remaining": getattr(state, "budget_remaining", 0),
            }
        ),
        tool_results_store=loop_ctx.tools.checkpoint_results(),
        domain_state=checkpoint_state(loop_ctx) if checkpoint_state is not None else {},
        run_envelope=loop_ctx.run_envelope.to_dict() if loop_ctx.run_envelope is not None else None,
        metadata=metadata,
        run_status=loop_ctx.status.value,
        run_phase=loop_ctx.phase.value,
        termination_reason=loop_ctx.termination_reason,
    )


def _pending_dispatch_checkpoint(
    *,
    calls: list[ToolCall],
    dispatching: list[ToolCall],
    step_number: int,
    session_log: SessionLog,
    conversation_data: dict[str, Any],
    config_snapshot: dict[str, Any],
    tool_results_store: dict[str, Any],
    domain_state: dict[str, Any],
    run_envelope: dict[str, Any] | None,
    metadata: dict[str, Any],
    run_status: str,
    run_phase: str,
) -> Checkpoint:
    """Persist pending calls before execution, including the state needed for reconciliation."""
    dispatching_ids = {id(call) for call in dispatching}
    pending_calls = [
        {
            "call_id": call.call_id,
            "tool": call.tool,
            "proposed_args": call.metadata.get("proposed_args", call.args),
            "effective_args": call.args,
            "execution_status": (
                "effect_unknown" if id(call) in dispatching_ids else "not_executed"
            ),
        }
        for call in calls
    ]
    return Checkpoint(
        step_number=step_number,
        session_log_data={
            "entries": session_log.to_list(),
            "current_theory": session_log.current_theory,
        },
        conversation_data=conversation_data,
        config_snapshot=config_snapshot,
        tool_results_store=tool_results_store,
        domain_state=domain_state,
        run_envelope=run_envelope,
        metadata={**metadata, "pending_calls": pending_calls},
        run_status=run_status,
        run_phase=run_phase,
    )


def save_pending_dispatch(
    store: CheckpointStore,
    *,
    calls: list[ToolCall],
    dispatching: list[ToolCall],
    step_number: int,
    session_log: SessionLog,
    conversation_data: dict[str, Any],
    config_snapshot: dict[str, Any],
    tool_results_store: dict[str, Any],
    domain_state: dict[str, Any],
    run_envelope: dict[str, Any] | None,
    metadata: dict[str, Any],
    run_status: str,
    run_phase: str,
) -> None:
    """Persist pending calls before execution and preserve uncertain effects."""
    checkpoint = _pending_dispatch_checkpoint(
        calls=calls,
        dispatching=dispatching,
        step_number=step_number,
        session_log=session_log,
        conversation_data=conversation_data,
        config_snapshot=config_snapshot,
        tool_results_store=tool_results_store,
        domain_state=domain_state,
        run_envelope=run_envelope,
        metadata=metadata,
        run_status=run_status,
        run_phase=run_phase,
    )
    store.save(checkpoint, key=f"step_{step_number}_pending")


async def _run_persistence(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    pending = asyncio.get_running_loop().run_in_executor(
        None, functools.partial(contextvars.copy_context().run, operation, *args, **kwargs)
    )
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(pending)
            break
        except asyncio.CancelledError as error:
            if pending.cancelled():
                raise
            cancelled = error
    if cancelled is not None:
        raise cancelled
    return result


async def _save_pending_dispatch_async(store: CheckpointStore, **values: Any) -> None:
    checkpoint = _pending_dispatch_checkpoint(**values)
    await _run_persistence(store.save, checkpoint, key=f"step_{checkpoint.step_number}_pending")


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
            if isinstance(getattr(state, "steps", None), MutableSequence):
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


def resume_loop_state(checkpoint: Checkpoint, *, max_steps: int | None = None) -> dict[str, Any]:
    """Reconstruct runnable loop state from a checkpoint.

    When ``max_steps`` is supplied, saved spending applies to that total cap.

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

    if checkpoint.metadata.get("pending_calls"):
        raise UncertainToolEffect(checkpoint.metadata["pending_calls"])

    from looplet.assistant_turns import TURN_STATE_KEY

    turns = checkpoint.metadata.get(TURN_STATE_KEY, {})
    if isinstance(turns, dict) and turns.get("pending_model_call"):
        raise UncertainModelCall(
            "A model request may have started. Preserve this checkpoint and reconcile "
            "the request before resuming; retrying can duplicate inference and billing."
        )

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
    if max_steps is not None and "max_steps" in cfg and "budget_remaining" in cfg:
        budgets = (max_steps, cfg["max_steps"], cfg["budget_remaining"])
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in budgets
        ):
            raise ValueError("checkpoint step budgets must be nonnegative integers")
        state_counters["budget_remaining"] = max(
            0, cfg["budget_remaining"] + max_steps - cfg["max_steps"]
        )

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
