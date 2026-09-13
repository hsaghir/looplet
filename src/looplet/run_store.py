"""Small host-owned stores for run records and evidence references."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from looplet.checkpoint import Checkpoint
from looplet.run_records import ArtifactRef, RunEvent, RunRecord
from looplet.types import RunEnvelope, RunResult

__all__ = ["RunStore", "MemoryRunStore", "FileRunStore"]


@runtime_checkable
class RunStore(Protocol):
    """Persistence contract for a logical run; evidence remains separate."""

    def create(
        self,
        envelope: RunEnvelope,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunRecord: ...

    def append_event(self, event: RunEvent) -> None: ...

    def save_checkpoint(self, run_id: str, checkpoint: Checkpoint) -> str: ...

    def load_checkpoint(self, run_id: str, key: str) -> Checkpoint | None: ...

    def complete(
        self,
        run_id: str,
        result: RunResult,
        *,
        artifacts: Sequence[ArtifactRef] = (),
    ) -> RunRecord: ...

    def load(self, run_id: str) -> RunRecord | None: ...

    def events(self, run_id: str) -> tuple[RunEvent, ...]: ...


class MemoryRunStore:
    """Thread-safe in-memory store for tests and short-lived hosts."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._checkpoints: dict[tuple[str, str], Checkpoint] = {}
        self._lock = threading.RLock()

    def create(
        self, envelope: RunEnvelope, *, metadata: Mapping[str, Any] | None = None
    ) -> RunRecord:
        with self._lock:
            if envelope.run_id in self._records:
                raise FileExistsError(f"run already exists: {envelope.run_id}")
            record = RunRecord.create(envelope, metadata=metadata)
            self._records[envelope.run_id] = record
            return record

    def append_event(self, event: RunEvent) -> None:
        with self._lock:
            record = self._require(event.envelope.run_id if event.envelope else "")
            updated = record.with_updates(
                status="running",
                updated_at=event.timestamp,
            )
            self._records[record.run_id] = replace(updated, events=(*record.events, event))

    def save_checkpoint(self, run_id: str, checkpoint: Checkpoint) -> str:
        with self._lock:
            record = self._require(run_id)
            key = f"step_{checkpoint.step_number}"
            self._records[run_id] = record.with_updates(
                updated_at=checkpoint.created_at,
                checkpoint_keys=tuple(dict.fromkeys((*record.checkpoint_keys, key))),
            )
            self._checkpoints[(run_id, key)] = checkpoint
            return key

    def load_checkpoint(self, run_id: str, key: str) -> Checkpoint | None:
        with self._lock:
            return self._checkpoints.get((run_id, Path(key).name))

    def complete(
        self,
        run_id: str,
        result: RunResult,
        *,
        artifacts: Sequence[ArtifactRef] = (),
    ) -> RunRecord:
        with self._lock:
            record = self._require(run_id)
            updated = RunRecord(
                run_id=record.run_id,
                envelope=record.envelope,
                status=result.status.value,
                metadata=dict(record.metadata),
                created_at=record.created_at,
                updated_at=record.updated_at,
                result=result.to_dict(),
                checkpoint_keys=record.checkpoint_keys,
                artifact_refs=tuple(artifacts),
                events=record.events,
            )
            self._records[run_id] = updated
            return updated

    def load(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        record = self._require(run_id)
        return record.events

    def _require(self, run_id: str) -> RunRecord:
        record = self._records.get(run_id)
        if record is None:
            raise KeyError(f"unknown run: {run_id}")
        return record


class FileRunStore:
    """Simple durable store using ``run.json``, ``events.jsonl`` and checkpoints."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(
        self, envelope: RunEnvelope, *, metadata: Mapping[str, Any] | None = None
    ) -> RunRecord:
        with self._lock:
            root = self._root(envelope.run_id)
            root.mkdir(parents=True, exist_ok=True)
            path = root / "run.json"
            if path.exists():
                raise FileExistsError(f"run already exists: {envelope.run_id}")
            record = RunRecord.create(envelope, metadata=metadata)
            self._write(path, record.to_dict())
            (root / "events.jsonl").touch()
            (root / "checkpoints").mkdir(exist_ok=True)
            return record

    def append_event(self, event: RunEvent) -> None:
        with self._lock:
            if event.envelope is None:
                raise ValueError("stored run events require a run envelope")
            run_id = event.envelope.run_id
            record = self._require(run_id)
            with (self._root(run_id) / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write(
                self._root(run_id) / "run.json",
                record.with_updates(status="running", updated_at=event.timestamp).to_dict(),
            )

    def save_checkpoint(self, run_id: str, checkpoint: Checkpoint) -> str:
        with self._lock:
            record = self._require(run_id)
            key = f"step_{checkpoint.step_number}"
            path = self._root(run_id) / "checkpoints" / f"{key}.json"
            self._write(path, checkpoint.to_dict())
            updated = record.with_updates(
                updated_at=checkpoint.created_at,
                checkpoint_keys=tuple(dict.fromkeys((*record.checkpoint_keys, key))),
            )
            self._write(self._root(run_id) / "run.json", updated.to_dict())
            return key

    def load_checkpoint(self, run_id: str, key: str) -> Checkpoint | None:
        with self._lock:
            path = self._root(run_id) / "checkpoints" / f"{Path(key).name}.json"
            if not path.is_file():
                return None
            return Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def complete(
        self,
        run_id: str,
        result: RunResult,
        *,
        artifacts: Sequence[ArtifactRef] = (),
    ) -> RunRecord:
        with self._lock:
            record = self._require(run_id)
            updated = RunRecord(
                run_id=record.run_id,
                envelope=record.envelope,
                status=result.status.value,
                metadata=dict(record.metadata),
                created_at=record.created_at,
                updated_at=time.time(),
                result=result.to_dict(),
                checkpoint_keys=record.checkpoint_keys,
                artifact_refs=tuple(artifacts),
                events=self.events(run_id),
            )
            self._write(self._root(run_id) / "run.json", updated.to_dict())
            return updated

    def load(self, run_id: str) -> RunRecord | None:
        with self._lock:
            path = self._root(run_id) / "run.json"
            if not path.is_file():
                return None
            return RunRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8")), events=self.events(run_id)
            )

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._lock:
            path = self._root(run_id) / "events.jsonl"
            if not path.is_file():
                return ()
            events: list[RunEvent] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(RunEvent.from_dict(json.loads(line)))
            return tuple(events)

    def _root(self, run_id: str) -> Path:
        safe = Path(run_id).name
        if safe != run_id or not safe:
            raise ValueError("run_id must be a non-empty path-safe name")
        return self.directory / safe

    def _require(self, run_id: str) -> RunRecord:
        record = self.load(run_id)
        if record is None:
            raise KeyError(f"unknown run: {run_id}")
        return record

    @staticmethod
    def _write(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
