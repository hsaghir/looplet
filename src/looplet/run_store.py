"""Small host-owned stores for run records and evidence references."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from looplet.artifact_compat import _artifact_path
from looplet.checkpoint import Checkpoint, validate_checkpoint_identity
from looplet.run_records import ArtifactRef, RunEvent, RunRecord
from looplet.types import RunEnvelope, RunResult

__all__ = ["RunStore", "MemoryRunStore", "FileRunStore"]


class _InterProcessFileLock:
    """Small standard-library advisory lock for file-store transactions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any = None

    def __enter__(self) -> "_InterProcessFileLock":
        handle = _artifact_path(self._path.parent, self._path.name).open("a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            elif os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - unsupported platform fallback
                raise OSError(f"FileRunStore locking is unsupported on {os.name!r}")
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._handle.close()
            self._handle = None


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
        self._events: dict[str, list[RunEvent]] = {}
        self._checkpoints: dict[tuple[str, str], Checkpoint] = {}
        self._lock = threading.RLock()

    def create(
        self, envelope: RunEnvelope, *, metadata: Mapping[str, Any] | None = None
    ) -> RunRecord:
        with self._lock:
            if envelope.run_id in self._records:
                raise FileExistsError(f"run already exists: {envelope.run_id}")
            record = RunRecord.create(envelope, metadata=metadata)
            self._records[envelope.run_id] = copy.deepcopy(record)
            self._events[envelope.run_id] = []
            return record

    def append_event(self, event: RunEvent) -> None:
        with self._lock:
            record = self._require(event.envelope.run_id if event.envelope else "")
            updated = record.with_updates(
                status="running" if record.status == "created" else record.status,
                updated_at=event.timestamp,
            )
            self._records[record.run_id] = updated
            self._events.setdefault(record.run_id, []).append(copy.deepcopy(event))

    def save_checkpoint(self, run_id: str, checkpoint: Checkpoint) -> str:
        with self._lock:
            record = self._require(run_id)
            if checkpoint.run_id is None:
                checkpoint = replace(checkpoint, run_envelope={"run_id": run_id})
            else:
                validate_checkpoint_identity(checkpoint, run_id)
            key = f"step_{checkpoint.step_number}"
            self._records[run_id] = record.with_updates(
                updated_at=checkpoint.created_at,
                checkpoint_keys=tuple(dict.fromkeys((*record.checkpoint_keys, key))),
            )
            self._checkpoints[(run_id, key)] = copy.deepcopy(checkpoint)
            return key

    def load_checkpoint(self, run_id: str, key: str) -> Checkpoint | None:
        with self._lock:
            checkpoint = self._checkpoints.get((run_id, Path(key).name))
            if checkpoint is not None:
                validate_checkpoint_identity(checkpoint, run_id)
            return copy.deepcopy(checkpoint)

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
                result=result.to_dict(full=True),
                checkpoint_keys=record.checkpoint_keys,
                artifact_refs=tuple(artifacts),
                events=(),
            )
            self._records[run_id] = copy.deepcopy(updated)
            return replace(
                copy.deepcopy(updated),
                events=tuple(copy.deepcopy(self._events.get(run_id, ()))),
            )

    def load(self, run_id: str) -> RunRecord | None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return None
            return replace(
                copy.deepcopy(record),
                events=tuple(copy.deepcopy(self._events.get(run_id, ()))),
            )

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._lock:
            self._require(run_id)
            return tuple(copy.deepcopy(self._events.get(run_id, ())))

    def _require(self, run_id: str) -> RunRecord:
        record = self._records.get(run_id)
        if record is None:
            raise KeyError(f"unknown run: {run_id}")
        return record


class FileRunStore:
    """Durable run store using locked ``run.json``, events, and checkpoints.

    Each public operation takes both an in-process lock and an advisory
    process lock, so separate workers sharing a directory cannot lose a
    read/modify/write update. The lock file is an implementation detail and
    is safe to retain with the store directory.
    """

    def __init__(self, directory: str | Path) -> None:
        from collections import OrderedDict

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._header_cache: OrderedDict[str, tuple[int, int, RunRecord]] = OrderedDict()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            with _InterProcessFileLock(self.directory / ".run-store.lock"):
                yield

    def create(
        self, envelope: RunEnvelope, *, metadata: Mapping[str, Any] | None = None
    ) -> RunRecord:
        with self._locked():
            root = self._root(envelope.run_id)
            try:
                root.mkdir()
            except FileExistsError as error:
                raise FileExistsError(f"run already exists: {envelope.run_id}") from error
            path = self._file(envelope.run_id, "run.json")
            record = RunRecord.create(envelope, metadata=metadata)
            self._write_record(envelope.run_id, path, record)
            self._file(envelope.run_id, "events.jsonl").touch()
            (root / "checkpoints").mkdir(exist_ok=True)
            return record

    def append_event(self, event: RunEvent) -> None:
        with self._locked():
            if event.envelope is None:
                raise ValueError("stored run events require a run envelope")
            run_id = event.envelope.run_id
            record = self._require_unlocked(run_id)
            with self._file(run_id, "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if record.status == "created":
                updated = record.with_updates(status="running", updated_at=event.timestamp)
                self._write_record(run_id, self._file(run_id, "run.json"), updated)
            else:
                cached = self._header_cache.get(run_id)
                if cached is not None:
                    mtime_ns, size, hdr = cached
                    self._header_cache[run_id] = (
                        mtime_ns,
                        size,
                        hdr.with_updates(updated_at=max(hdr.updated_at, event.timestamp)),
                    )

    def save_checkpoint(self, run_id: str, checkpoint: Checkpoint) -> str:
        with self._locked():
            record = self._require_unlocked(run_id)
            if checkpoint.run_id is None:
                checkpoint = replace(checkpoint, run_envelope={"run_id": run_id})
            else:
                validate_checkpoint_identity(checkpoint, run_id)
            key = f"step_{checkpoint.step_number}"
            path = self._file(run_id, f"checkpoints/{key}.json")
            self._write(path, checkpoint.to_dict())
            updated = record.with_updates(
                updated_at=checkpoint.created_at,
                checkpoint_keys=tuple(dict.fromkeys((*record.checkpoint_keys, key))),
            )
            self._write_record(run_id, self._file(run_id, "run.json"), updated)
            return key

    def load_checkpoint(self, run_id: str, key: str) -> Checkpoint | None:
        with self._locked():
            path = self._file(run_id, f"checkpoints/{Path(key).name}.json")
            if not path.is_file():
                return None
            checkpoint = Checkpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
            validate_checkpoint_identity(checkpoint, run_id)
            return checkpoint

    def complete(
        self,
        run_id: str,
        result: RunResult,
        *,
        artifacts: Sequence[ArtifactRef] = (),
    ) -> RunRecord:
        with self._locked():
            record = self._require_unlocked(run_id)
            events = self._events_unlocked(run_id)
            updated = RunRecord(
                run_id=record.run_id,
                envelope=record.envelope,
                status=result.status.value,
                metadata=dict(record.metadata),
                created_at=record.created_at,
                updated_at=time.time(),
                result=result.to_dict(full=True),
                checkpoint_keys=record.checkpoint_keys,
                artifact_refs=tuple(artifacts),
                events=events,
            )
            self._write_record(run_id, self._file(run_id, "run.json"), updated)
            return updated

    def load(self, run_id: str) -> RunRecord | None:
        with self._locked():
            return self._load_unlocked(run_id)

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._locked():
            return self._events_unlocked(run_id)

    def _root(self, run_id: str) -> Path:
        safe = Path(run_id).name
        if safe != run_id or not safe or safe in {".", ".."}:
            raise ValueError("run_id must be a non-empty path-safe name")
        root = self.directory / safe
        if root.is_symlink():
            raise ValueError("run_id must identify a real directory, not a symlink")
        return root

    def _file(self, run_id: str, filename: str) -> Path:
        return _artifact_path(self._root(run_id), filename)

    def _write_record(self, run_id: str, path: Path, record: RunRecord) -> None:
        serialized = json.dumps(record.to_dict(), indent=2, default=str) + "\n"
        self._write_text(path, serialized)
        st = path.stat()
        detached = RunRecord.from_dict(json.loads(serialized), events=())
        self._header_cache.pop(run_id, None)
        self._header_cache[run_id] = (st.st_mtime_ns, st.st_size, detached)
        while len(self._header_cache) > 64:
            self._header_cache.popitem(last=False)

    def _load_unlocked(self, run_id: str, *, include_events: bool = True) -> RunRecord | None:
        path = self._file(run_id, "run.json")
        if not path.is_file():
            self._header_cache.pop(run_id, None)
            return None
        st = path.stat()
        cached = self._header_cache.get(run_id)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            self._header_cache.move_to_end(run_id)
            header = copy.deepcopy(cached[2])
        else:
            header = RunRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8")),
                events=(),
            )
            self._header_cache.pop(run_id, None)
            self._header_cache[run_id] = (st.st_mtime_ns, st.st_size, copy.deepcopy(header))
            while len(self._header_cache) > 64:
                self._header_cache.popitem(last=False)
        if not include_events:
            return header
        ev = self._events_unlocked(run_id)
        updated_at = (
            max(header.updated_at, max(e.timestamp for e in ev)) if ev else header.updated_at
        )
        return replace(header, events=ev, updated_at=updated_at)

    def _events_unlocked(self, run_id: str) -> tuple[RunEvent, ...]:
        path = self._file(run_id, "events.jsonl")
        if not path.is_file():
            return ()
        events: list[RunEvent] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    events.append(RunEvent.from_dict(json.loads(line)))
        return tuple(events)

    def _require_unlocked(self, run_id: str) -> RunRecord:
        record = self._load_unlocked(run_id, include_events=False)
        if record is None:
            raise KeyError(f"unknown run: {run_id}")
        return record

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    @staticmethod
    def _write(path: Path, value: Mapping[str, Any]) -> None:
        FileRunStore._write_text(path, json.dumps(value, indent=2, default=str) + "\n")
