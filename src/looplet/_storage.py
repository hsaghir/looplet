"""Private disk-backed and revision-tracked sequences with bounded overhead."""

from __future__ import annotations

import atexit
import copy
import json
import os
import pickle
import sqlite3
import tempfile
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSequence, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Generic, TypeVar, overload
from uuid import uuid4

Record = TypeVar("Record")


class _SharedTempSpool:
    """Process-local SQLite spool shared across temporary lists and dicts."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.directory = tempfile.TemporaryDirectory(prefix="looplet-spool-")
        self.db_path = Path(self.directory.name) / "spool.sqlite3"
        self.connection = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self.connection.execute("PRAGMA cache_size = -4096")
        self.connection.execute("PRAGMA journal_mode = MEMORY")
        self.connection.execute("PRAGMA synchronous = OFF")
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self.connection.execute(
            "CREATE TABLE list_records ("
            "cid INTEGER NOT NULL, position INTEGER NOT NULL, payload BLOB, "
            "weight INTEGER NOT NULL, revision INTEGER NOT NULL, "
            "PRIMARY KEY (cid, position)) WITHOUT ROWID"
        )
        self.connection.execute(
            "CREATE INDEX list_records_revision ON list_records (cid, revision, position)"
        )
        self.connection.execute(
            "CREATE TABLE dict_records ("
            "cid INTEGER NOT NULL, ord INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE (cid, key))"
        )
        self.lock = RLock()
        self.pending_deletes: deque[tuple[str, int]] = deque()
        self._next_cid = 0
        self._savepoint_seq = 0
        self.closed = False

    def drain_pending_deletes(self) -> None:
        """Delete finalized container rows safely under ``self.lock``."""
        while self.pending_deletes:
            table, cid = self.pending_deletes.popleft()
            if not self.closed:
                self.connection.execute(f"DELETE FROM {table} WHERE cid = ?", (cid,))

    def alloc_cid(self) -> int:
        with self.lock:
            self.drain_pending_deletes()
            self._next_cid += 1
            return self._next_cid

    def next_savepoint(self) -> str:
        with self.lock:
            self._savepoint_seq += 1
            return f"catalog_update_{self._savepoint_seq}"

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.pending_deletes.clear()
            try:
                self.connection.close()
            finally:
                self.directory.cleanup()


_SPOOL_LOCK = RLock()
_ACTIVE_SPOOL: _SharedTempSpool | None = None


def _cleanup_active_spool() -> None:
    global _ACTIVE_SPOOL
    with _SPOOL_LOCK:
        spool = _ACTIVE_SPOOL
        _ACTIVE_SPOOL = None
    if spool is not None and spool.pid == os.getpid():
        spool.close()


atexit.register(_cleanup_active_spool)


def _get_temp_spool() -> _SharedTempSpool:
    global _ACTIVE_SPOOL
    pid = os.getpid()
    spool = _ACTIVE_SPOOL
    if spool is not None and not spool.closed and spool.pid == pid and spool.db_path.exists():
        return spool
    with _SPOOL_LOCK:
        spool = _ACTIVE_SPOOL
        if spool is None or spool.closed or spool.pid != pid or not spool.db_path.exists():
            spool = _SharedTempSpool()
            _ACTIVE_SPOOL = spool
        return spool


def _release_spool_rows(spool: _SharedTempSpool, table: str, cid: int) -> None:
    if not spool.closed and spool.pid == os.getpid():
        spool.pending_deletes.append((table, cid))


class TrackedList(list):
    """Native list subclass tracking per-index revisions for O(delta) checkpointing."""

    def __init__(self, values: Iterable[Any] = ()) -> None:
        super().__init__(values)
        self._record_identity = uuid4().hex
        self._generation = 0
        n = len(self)
        self._revision = n
        self._item_revisions: list[int] = list(range(1, n + 1))
        self._change_log: list[tuple[int, int]] = [
            (rev, idx) for idx, rev in enumerate(self._item_revisions)
        ]
        self._detached = False

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(item, memo) for item in self]

    def _detach_if_needed(self) -> None:
        if self._detached:
            self._record_identity = uuid4().hex
            self._detached = False

    def _compact_change_log(self) -> None:
        if len(self._change_log) > max(64, len(self) * 4):
            self._change_log = sorted((rev, idx) for idx, rev in enumerate(self._item_revisions))

    def append(self, value: Any) -> None:
        self._detach_if_needed()
        pos = len(self)
        super().append(value)
        self._revision += 1
        self._item_revisions.append(self._revision)
        self._change_log.append((self._revision, pos))

    def extend(self, values: Iterable[Any]) -> None:
        items = list(values)
        if not items:
            return
        self._detach_if_needed()
        base_pos = len(self)
        super().extend(items)
        start = self._revision + 1
        self._revision += len(items)
        self._item_revisions.extend(range(start, self._revision + 1))
        self._change_log.extend((start + offset, base_pos + offset) for offset in range(len(items)))

    def insert(self, index: Any, value: Any) -> None:
        self._detach_if_needed()
        pos = max(0, min(int(index) if int(index) >= 0 else len(self) + int(index), len(self)))
        if pos == len(self):
            self.append(value)
            return
        super().insert(pos, value)
        self._generation += 1
        self._revision += 1
        self._item_revisions.insert(pos, self._revision)
        self._change_log = sorted((rev, idx) for idx, rev in enumerate(self._item_revisions))

    def pop(self, index: Any = -1) -> Any:
        self._detach_if_needed()
        idx = int(index)
        is_last = idx == -1 or idx == len(self) - 1
        val = super().pop(idx)
        self._item_revisions.pop(idx)
        if not is_last:
            self._generation += 1
            self._change_log = sorted((rev, i) for i, rev in enumerate(self._item_revisions))
        else:
            self._compact_change_log()
        self._revision += 1
        return val

    def clear(self) -> None:
        self._detach_if_needed()
        super().clear()
        self._item_revisions.clear()
        self._change_log.clear()
        self._generation += 1
        self._revision += 1

    def reverse(self) -> None:
        self._detach_if_needed()
        super().reverse()
        self._generation += 1
        self._revision += 1
        self._item_revisions = [self._revision] * len(self)
        self._change_log = [(self._revision, idx) for idx in range(len(self))]

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._detach_if_needed()
        super().sort(key=key, reverse=reverse)
        self._generation += 1
        self._revision += 1
        self._item_revisions = [self._revision] * len(self)
        self._change_log = [(self._revision, idx) for idx in range(len(self))]

    def mark_modified(self, index: int) -> None:
        if 0 <= index < len(self):
            self._detach_if_needed()
            self._revision += 1
            self._item_revisions[index] = self._revision
            self._change_log.append((self._revision, index))
            self._compact_change_log()

    def __setitem__(self, index: Any, value: Any) -> None:
        self._detach_if_needed()
        if isinstance(index, slice):
            positions = range(*index.indices(len(self)))
            vals = list(value)
            super().__setitem__(index, vals)
            if len(positions) == len(vals):
                for pos in positions:
                    self._revision += 1
                    self._item_revisions[pos] = self._revision
                    self._change_log.append((self._revision, pos))
                self._compact_change_log()
            else:
                self._generation += 1
                self._revision += 1
                self._item_revisions = [self._revision] * len(self)
                self._change_log = [(self._revision, idx) for idx in range(len(self))]
            return
        pos = int(index)
        if pos < 0:
            pos += len(self)
        super().__setitem__(pos, value)
        self._revision += 1
        self._item_revisions[pos] = self._revision
        self._change_log.append((self._revision, pos))
        self._compact_change_log()

    def __delitem__(self, index: Any) -> None:
        self._detach_if_needed()
        super().__delitem__(index)
        del self._item_revisions[index]
        self._generation += 1
        self._revision += 1
        self._change_log = sorted((rev, idx) for idx, rev in enumerate(self._item_revisions))

    def __iadd__(self, values: Iterable[Any]) -> TrackedList:
        self.extend(values)
        return self

    def snapshot_list(self) -> TrackedList:
        """Return an independent list copy preserving incremental checkpoint identity."""
        clone = TrackedList.__new__(TrackedList)
        list.__init__(clone, self)
        clone._record_identity = self._record_identity
        clone._generation = self._generation
        clone._revision = self._revision
        clone._item_revisions = self._item_revisions[:]
        clone._change_log = self._change_log[:]
        clone._detached = True
        return clone

    @contextmanager
    def changes_since(
        self, cursor: tuple[int, int, int], stop: int
    ) -> Iterator[tuple[int, int, Iterator[tuple[int, Any]]]]:
        from bisect import bisect_right

        if stop > len(self):
            raise IndexError("record index out of range")
        generation, revision, length = cursor
        if generation != self._generation or stop < length:
            changed = [(pos, self[pos]) for pos in range(stop)]
        else:
            limit = min(length, stop)
            log_pos = bisect_right(self._change_log, (revision, len(self) + 1))
            dirty_positions = sorted(
                {
                    idx
                    for _, idx in self._change_log[log_pos:]
                    if idx < limit and self._item_revisions[idx] > revision
                }
            )
            changed = [(pos, self[pos]) for pos in dirty_positions]
            if stop > length:
                changed.extend((pos, self[pos]) for pos in range(length, stop))
        yield self._generation, self._revision, iter(changed)


_HOT_MAX_BYTES = 65_536
_HOT_MAX_ITEMS = 128


class DiskBackedList(MutableSequence[Record], Generic[Record]):
    """Store detached records on private temporary disk, caching one live record.

    Indexing and iteration load records lazily. Explicit slices and copies can
    materialize multiple records. Assign an edited record to persist changes
    after it leaves the one-record cache. This is not a recovery journal.
    Python mode uses pickles only in this process's private temporary spool.
    """

    def __init__(
        self,
        encode: Callable[[Record], Any],
        decode: Callable[[Any], Record],
        values: Iterable[Record] = (),
        *,
        preserve_python: bool = False,
        weight: Callable[[Record], int] | None = None,
    ) -> None:
        spool = _get_temp_spool()
        self._spool = spool
        self._cid = spool.alloc_cid()
        self._finalizer = weakref.finalize(
            self, _release_spool_rows, spool, "list_records", self._cid
        )
        self._encode = encode
        self._decode = decode
        self._preserve_python = preserve_python
        self._weight = weight
        self._total_weight = 0
        self._cache_weight = 0
        self._length = 0
        self._cache: tuple[int, Record] | None = None
        self._cache_payload: str | bytes | None = None
        self._mem: list[tuple[str | bytes, int, int]] | None = []
        self._mem_bytes = 0
        self._lock = RLock()
        self._record_identity = uuid4().hex
        self._generation = 0
        self._revision = 0
        self.extend(values)

    def _spill_to_spool(self) -> None:
        if self._mem is None:
            return
        rows = self._mem
        self._mem = None
        self._mem_bytes = 0
        if rows:
            with self._spool.lock:
                self._spool.connection.executemany(
                    "INSERT INTO list_records VALUES (?, ?, ?, ?, ?)",
                    [
                        (self._cid, pos, payload, w, rev)
                        for pos, (payload, w, rev) in enumerate(rows)
                    ],
                )

    def _dumps(self, value: Record) -> str | bytes:
        encoded = self._encode(value)
        if self._preserve_python:
            return pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
        return json.dumps(encoded, default=str)

    def _loads(self, payload: Any) -> Record:
        encoded = pickle.loads(payload) if self._preserve_python else json.loads(payload)
        return self._decode(encoded)

    def __len__(self) -> int:
        return self._length

    def _sync_weight(self) -> None:
        if self._cache is not None and self._weight is not None:
            position, value = self._cache
            weight = self._weight(value)
            if weight != self._cache_weight:
                if self._mem is not None:
                    payload, _, rev = self._mem[position]
                    self._mem[position] = (payload, weight, rev)
                else:
                    with self._spool.lock:
                        self._spool.connection.execute(
                            "UPDATE list_records SET weight = ? WHERE cid = ? AND position = ?",
                            (weight, self._cid, position),
                        )
                self._total_weight += weight - self._cache_weight
                self._cache_weight = weight

    @property
    def total_weight(self) -> int | None:
        """Return the incremental total without decoding or serializing payloads."""
        with self._lock:
            self._sync_weight()
            return self._total_weight if self._weight is not None else None

    def _index(self, index: int) -> int:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError("record index out of range")
        return index

    def _flush(self) -> None:
        self._sync_weight()
        if self._cache is not None:
            index, value = self._cache
            payload = self._dumps(value)
            if payload != self._cache_payload:
                self._revision += 1
                if self._mem is not None:
                    old_payload, w, _ = self._mem[index]
                    new_bytes = self._mem_bytes - len(old_payload) + len(payload)
                    self._mem[index] = (payload, w, self._revision)
                    self._mem_bytes = new_bytes
                    if self._mem_bytes > _HOT_MAX_BYTES:
                        self._spill_to_spool()
                else:
                    with self._spool.lock:
                        self._spool.connection.execute(
                            "UPDATE list_records SET payload = ?, revision = ? "
                            "WHERE cid = ? AND position = ?",
                            (payload, self._revision, self._cid, index),
                        )
                self._cache_payload = payload

    @overload
    def __getitem__(self, index: int) -> Record: ...

    @overload
    def __getitem__(self, index: slice) -> list[Record]: ...

    def __getitem__(self, index: int | slice) -> Record | list[Record]:
        with self._lock:
            if isinstance(index, slice):
                return [self[position] for position in range(*index.indices(self._length))]
            position = self._index(index)
            if self._cache is None or self._cache[0] != position:
                self._flush()
                if self._mem is not None:
                    raw_payload, raw_weight, _ = self._mem[position]
                else:
                    with self._spool.lock:
                        row = self._spool.connection.execute(
                            "SELECT payload, weight FROM list_records WHERE cid = ? AND position = ?",
                            (self._cid, position),
                        ).fetchone()
                    raw_payload, raw_weight = row[0], row[1]
                self._cache = (position, self._loads(raw_payload))
                self._cache_payload = raw_payload
                self._cache_weight = raw_weight
            return self._cache[1]

    def __setitem__(self, index: Any, value: Any) -> None:
        with self._lock:
            if isinstance(index, slice):
                positions = range(*index.indices(self._length))
                values = list(value)
                if index.step not in (None, 1) and len(positions) != len(values):
                    raise ValueError("extended slice assignment must preserve length")
                if len(positions) == len(values):
                    for position, item in zip(positions, values, strict=True):
                        self[position] = item
                else:
                    start = positions.start
                    del self[index]
                    for offset, item in enumerate(values):
                        self.insert(start + offset, item)
                return
            position = self._index(index)
            self._flush()
            payload = self._dumps(value)
            weight = self._weight(value) if self._weight is not None else 0
            self._revision += 1
            if self._mem is not None:
                old_payload, previous_weight, _ = self._mem[position]
                self._mem[position] = (payload, weight, self._revision)
                self._mem_bytes += len(payload) - len(old_payload)
                if self._mem_bytes > _HOT_MAX_BYTES:
                    self._spill_to_spool()
            else:
                with self._spool.lock:
                    previous_weight = self._spool.connection.execute(
                        "SELECT weight FROM list_records WHERE cid = ? AND position = ?",
                        (self._cid, position),
                    ).fetchone()[0]
                    self._spool.connection.execute(
                        "UPDATE list_records SET payload = ?, weight = ?, revision = ? "
                        "WHERE cid = ? AND position = ?",
                        (payload, weight, self._revision, self._cid, position),
                    )
            self._total_weight += weight - previous_weight
            self._cache = (position, value)
            self._cache_payload = payload
            self._cache_weight = weight

    def __delitem__(self, index: Any) -> None:
        with self._lock:
            if isinstance(index, slice):
                for position in sorted(range(*index.indices(self._length)), reverse=True):
                    del self[position]
                return
            position = self._index(index)
            self._flush()
            self._cache = None
            if self._mem is not None:
                old_payload, weight, _ = self._mem.pop(position)
                self._mem_bytes -= len(old_payload)
            else:
                with self._spool.lock:
                    weight = self._spool.connection.execute(
                        "SELECT weight FROM list_records WHERE cid = ? AND position = ?",
                        (self._cid, position),
                    ).fetchone()[0]
                    self._spool.connection.execute(
                        "DELETE FROM list_records WHERE cid = ? AND position = ?",
                        (self._cid, position),
                    )
                    self._spool.connection.execute(
                        "UPDATE list_records SET position = -position WHERE cid = ? AND position > ?",
                        (self._cid, position),
                    )
                    self._spool.connection.execute(
                        "UPDATE list_records SET position = -position - 1 "
                        "WHERE cid = ? AND position < 0",
                        (self._cid,),
                    )
            self._generation += 1
            self._length -= 1
            self._total_weight -= weight

    def append(self, value: Record) -> None:
        with self._lock:
            self._flush()
            payload = self._dumps(value)
            weight = self._weight(value) if self._weight is not None else 0
            self._revision += 1
            position = self._length
            if self._mem is not None:
                if (
                    len(self._mem) < _HOT_MAX_ITEMS
                    and self._mem_bytes + len(payload) <= _HOT_MAX_BYTES
                ):
                    self._mem.append((payload, weight, self._revision))
                    self._mem_bytes += len(payload)
                else:
                    self._spill_to_spool()
                    with self._spool.lock:
                        self._spool.connection.execute(
                            "INSERT INTO list_records VALUES (?, ?, ?, ?, ?)",
                            (self._cid, position, payload, weight, self._revision),
                        )
            else:
                with self._spool.lock:
                    self._spool.connection.execute(
                        "INSERT INTO list_records VALUES (?, ?, ?, ?, ?)",
                        (self._cid, position, payload, weight, self._revision),
                    )
            self._length += 1
            self._total_weight += weight
            if len(payload) <= _HOT_MAX_BYTES:
                self._cache = (position, value)
                self._cache_payload = payload
                self._cache_weight = weight
            else:
                self._cache = None
                self._cache_payload = None
                self._cache_weight = 0

    def insert(self, index: int, value: Record) -> None:
        with self._lock:
            position = max(0, min(index if index >= 0 else self._length + index, self._length))
            if position == self._length:
                self.append(value)
                return
            self._flush()
            self._cache = None
            payload = self._dumps(value)
            weight = self._weight(value) if self._weight is not None else 0
            self._revision += 1
            self._generation += 1
            if self._mem is not None:
                if (
                    len(self._mem) < _HOT_MAX_ITEMS
                    and self._mem_bytes + len(payload) <= _HOT_MAX_BYTES
                ):
                    self._mem.insert(position, (payload, weight, self._revision))
                    self._mem_bytes += len(payload)
                else:
                    self._spill_to_spool()
                    with self._spool.lock:
                        self._spool.connection.execute(
                            "UPDATE list_records SET position = -position - 1 "
                            "WHERE cid = ? AND position >= ?",
                            (self._cid, position),
                        )
                        self._spool.connection.execute(
                            "UPDATE list_records SET position = -position "
                            "WHERE cid = ? AND position < 0",
                            (self._cid,),
                        )
                        self._spool.connection.execute(
                            "INSERT INTO list_records VALUES (?, ?, ?, ?, ?)",
                            (self._cid, position, payload, weight, self._revision),
                        )
            else:
                with self._spool.lock:
                    self._spool.connection.execute(
                        "UPDATE list_records SET position = -position - 1 "
                        "WHERE cid = ? AND position >= ?",
                        (self._cid, position),
                    )
                    self._spool.connection.execute(
                        "UPDATE list_records SET position = -position "
                        "WHERE cid = ? AND position < 0",
                        (self._cid,),
                    )
                    self._spool.connection.execute(
                        "INSERT INTO list_records VALUES (?, ?, ?, ?, ?)",
                        (self._cid, position, payload, weight, self._revision),
                    )
            self._length += 1
            self._total_weight += weight
            if len(payload) <= _HOT_MAX_BYTES:
                self._cache = (position, value)
                self._cache_payload = payload
                self._cache_weight = weight

    def clear(self) -> None:
        with self._lock:
            self._cache = None
            self._cache_payload = None
            self._cache_weight = 0
            if self._mem is not None:
                self._mem.clear()
                self._mem_bytes = 0
            else:
                with self._spool.lock:
                    self._spool.connection.execute(
                        "DELETE FROM list_records WHERE cid = ?", (self._cid,)
                    )
                self._mem = []
                self._mem_bytes = 0
            self._generation += 1
            self._length = 0
            self._total_weight = 0

    def sort(self, *, key: Callable[[Record], Any], reverse: bool = False) -> None:
        with self._lock:
            positions = sorted(
                range(self._length), key=lambda position: key(self[position]), reverse=reverse
            )
            self._flush()
            self._cache = None
            self._generation += 1
            if self._mem is not None:
                self._mem = [self._mem[orig] for orig in positions]
            else:
                with self._spool.lock:
                    self._spool.connection.execute(
                        "UPDATE list_records SET position = -position - 1 WHERE cid = ?",
                        (self._cid,),
                    )
                    for position, original in enumerate(positions):
                        self._spool.connection.execute(
                            "UPDATE list_records SET position = ? WHERE cid = ? AND position = ?",
                            (position, self._cid, -original - 1),
                        )

    @contextmanager
    def changes_since(
        self, cursor: tuple[int, int, int], stop: int
    ) -> Iterator[tuple[int, int, Iterator[tuple[int, Record]]]]:
        """Yield detached changed/new prefix records while holding the owning lock."""
        with self._lock:
            self._flush()
            if stop > self._length:
                raise IndexError("record index out of range")
            generation, revision, length = cursor
            if generation != self._generation or stop < length:
                revision, length = -1, 0
            if self._mem is not None:
                limit = min(length, stop)
                rows = [
                    (pos, self._mem[pos][0]) for pos in range(limit) if self._mem[pos][2] > revision
                ]
                if stop > length:
                    rows.extend((pos, self._mem[pos][0]) for pos in range(length, stop))
            else:
                with self._spool.lock:
                    rows = self._spool.connection.execute(
                        "SELECT position, payload FROM list_records INDEXED BY list_records_revision "
                        "WHERE cid = ? AND revision > ? AND position < ? "
                        "UNION ALL SELECT position, payload FROM list_records "
                        "WHERE cid = ? AND position >= ? AND position < ? ORDER BY position",
                        (self._cid, revision, min(length, stop), self._cid, length, stop),
                    ).fetchall()
            yield (
                self._generation,
                self._revision,
                ((position, self._loads(payload)) for position, payload in rows),
            )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(left == right for left, right in zip(self, other, strict=True))
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Record]:
        return [copy.deepcopy(value, memo) for value in self]

    def clone(self) -> DiskBackedList[Record]:
        """Return an independent mutable DiskBackedList copy without Python re-encoding."""
        with self._lock:
            self._flush()
            records = DiskBackedList(
                self._encode,
                self._decode,
                preserve_python=self._preserve_python,
                weight=self._weight,
            )
            if self._mem is not None:
                records._mem = list(self._mem)
                records._mem_bytes = self._mem_bytes
            else:
                records._mem = None
                records._mem_bytes = 0
                with self._spool.lock:
                    self._spool.connection.execute(
                        "INSERT INTO list_records (cid, position, payload, weight, revision) "
                        "SELECT ?, position, payload, weight, revision "
                        "FROM list_records WHERE cid = ?",
                        (records._cid, self._cid),
                    )
            records._length = self._length
            records._total_weight = self._total_weight
            records._generation = self._generation
            records._revision = self._revision
            return records

    def snapshot(self) -> DiskBackedSequence[Record]:
        """Return an independent read-only snapshot without loading all records."""
        return DiskBackedSequence(self.clone())


class DiskBackedSequence(Sequence[Record], Generic[Record]):
    """A read-only sequence owning a detached disk-backed snapshot."""

    def __init__(self, records: DiskBackedList[Record]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> Record: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Record, ...]: ...

    def __getitem__(self, index: int | slice) -> Record | tuple[Record, ...]:
        return tuple(self._records[index]) if isinstance(index, slice) else self._records[index]

    def __eq__(self, other: object) -> bool:
        return self._records == other


class DiskBackedDict(MutableMapping[str, Record], Generic[Record]):
    """Ordered records on disk with one cached value and a bounded database cache."""

    def __init__(
        self,
        encode: Callable[[Record], Any],
        decode: Callable[[Any], Record],
        path: Path | None = None,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self._cache: tuple[str, Record] | None = None
        self._cache_payload: str | None = None
        self._lock = RLock()
        if path is None:
            spool = _get_temp_spool()
            self._spool: _SharedTempSpool | None = spool
            self._cid: int | None = spool.alloc_cid()
            self._connection = spool.connection
            self._finalizer = weakref.finalize(
                self, _release_spool_rows, spool, "dict_records", self._cid
            )
        else:
            self._spool = None
            self._cid = None
            self._connection = sqlite3.connect(
                path, timeout=10.0, isolation_level=None, check_same_thread=False
            )
            self._connection.execute("PRAGMA cache_size = -1024")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, payload TEXT)"
            )
            self._finalizer = weakref.finalize(self, self._close_file, self._connection)
        self._mem: dict[str, tuple[int, str]] | None = {} if path is None else None
        self._mem_bytes = 0
        self._next_ord = 0

    def _spill_dict_to_spool(self) -> None:
        if self._mem is None or self._cid is None or self._spool is None:
            return
        items = list(self._mem.items())
        self._mem = None
        self._mem_bytes = 0
        if items:
            items.sort(key=lambda pair: pair[1][0])
            with self._db() as conn:
                conn.executemany(
                    "INSERT INTO dict_records (cid, key, payload) VALUES (?, ?, ?)",
                    [(self._cid, k, payload) for k, (_, payload) in items],
                )

    @staticmethod
    def _close_file(connection: sqlite3.Connection) -> None:
        connection.close()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        if self._spool is not None:
            with self._spool.lock:
                if self._spool.pending_deletes:
                    self._spool.drain_pending_deletes()
                yield self._spool.connection
        else:
            yield self._connection

    def flush(self) -> None:
        if self._cache is not None:
            key, value = self._cache
            payload = json.dumps(self._encode(value), default=str)
            if payload != self._cache_payload:
                if self._mem is not None and key in self._mem:
                    ord_val, old_p = self._mem[key]
                    self._mem[key] = (ord_val, payload)
                    self._mem_bytes += len(payload) - len(old_p)
                    if self._mem_bytes > _HOT_MAX_BYTES:
                        self._spill_dict_to_spool()
                else:
                    with self._db() as conn:
                        if self._cid is not None:
                            conn.execute(
                                "UPDATE dict_records SET payload = ? WHERE cid = ? AND key = ?",
                                (payload, self._cid, key),
                            )
                        else:
                            conn.execute(
                                "UPDATE records SET payload = ? WHERE key = ?", (payload, key)
                            )
                self._cache_payload = payload

    def __len__(self) -> int:
        with self._lock:
            if self._mem is not None:
                return len(self._mem)
            with self._db() as conn:
                if self._cid is not None:
                    return conn.execute(
                        "SELECT count(*) FROM dict_records WHERE cid = ?", (self._cid,)
                    ).fetchone()[0]
                return conn.execute("SELECT count(*) FROM records").fetchone()[0]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self.flush()
            if self._mem is not None:
                keys = [k for k, _ in sorted(self._mem.items(), key=lambda pair: pair[1][0])]
            else:
                with self._db() as conn:
                    if self._cid is not None:
                        rows = conn.execute(
                            "SELECT key FROM dict_records WHERE cid = ? ORDER BY ord", (self._cid,)
                        ).fetchall()
                    else:
                        rows = conn.execute("SELECT key FROM records ORDER BY rowid").fetchall()
                keys = [row[0] for row in rows]
            for key in keys:
                yield key

    def items_after(self, position: int) -> Iterator[tuple[int, str, Record]]:
        """Iterate newly appended catalog records from a stable row position."""
        with self._lock:
            self.flush()
            if self._mem is not None:
                entries = [
                    (ord_val, k, payload)
                    for k, (ord_val, payload) in self._mem.items()
                    if ord_val > position
                ]
                entries.sort(key=lambda item: item[0])
                rows = entries
            else:
                with self._db() as conn:
                    if self._cid is not None:
                        rows = conn.execute(
                            "SELECT ord, key, payload FROM dict_records "
                            "WHERE cid = ? AND ord > ? ORDER BY ord",
                            (self._cid, position),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT rowid, key, payload FROM records WHERE rowid > ? ORDER BY rowid",
                            (position,),
                        ).fetchall()
            for row in rows:
                yield row[0], row[1], self._decode(json.loads(row[2]))

    def __getitem__(self, key: str) -> Record:
        with self._lock:
            if self._cache is None or self._cache[0] != key:
                self.flush()
                if self._mem is not None:
                    item = self._mem.get(key)
                    if item is None:
                        raise KeyError(key)
                    raw_payload = item[1]
                else:
                    with self._db() as conn:
                        if self._cid is not None:
                            row = conn.execute(
                                "SELECT payload FROM dict_records WHERE cid = ? AND key = ?",
                                (self._cid, key),
                            ).fetchone()
                        else:
                            row = conn.execute(
                                "SELECT payload FROM records WHERE key = ?", (key,)
                            ).fetchone()
                    if row is None:
                        raise KeyError(key)
                    raw_payload = row[0]
                self._cache = (key, self._decode(json.loads(raw_payload)))
                self._cache_payload = raw_payload
            return self._cache[1]

    def __setitem__(self, key: str, value: Record) -> None:
        with self._lock:
            self.flush()
            payload = json.dumps(self._encode(value), default=str)
            if self._mem is not None:
                existing = self._mem.get(key)
                if existing is not None:
                    self._mem[key] = (existing[0], payload)
                    self._mem_bytes += len(payload) - len(existing[1])
                elif (
                    len(self._mem) < _HOT_MAX_ITEMS
                    and self._mem_bytes + len(payload) <= _HOT_MAX_BYTES
                ):
                    self._next_ord += 1
                    self._mem[key] = (self._next_ord, payload)
                    self._mem_bytes += len(payload)
                else:
                    self._spill_dict_to_spool()
                if self._mem is not None and self._mem_bytes > _HOT_MAX_BYTES:
                    self._spill_dict_to_spool()
            if self._mem is None:
                with self._db() as conn:
                    if self._cid is not None:
                        updated = conn.execute(
                            "UPDATE dict_records SET payload = ? WHERE cid = ? AND key = ?",
                            (payload, self._cid, key),
                        )
                        if not updated.rowcount:
                            conn.execute(
                                "INSERT INTO dict_records (cid, key, payload) VALUES (?, ?, ?)",
                                (self._cid, key, payload),
                            )
                    else:
                        conn.execute(
                            "INSERT INTO records VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET payload = excluded.payload",
                            (key, payload),
                        )
            if len(payload) <= _HOT_MAX_BYTES:
                self._cache = (key, value)
                self._cache_payload = payload
            else:
                self._cache = None
                self._cache_payload = None

    def __delitem__(self, key: str) -> None:
        with self._lock:
            self.flush()
            self._cache = None
            if self._mem is not None:
                removed = self._mem.pop(key, None)
                if removed is None:
                    raise KeyError(key)
                self._mem_bytes -= len(removed[1])
                return
            with self._db() as conn:
                if self._cid is not None:
                    deleted = conn.execute(
                        "DELETE FROM dict_records WHERE cid = ? AND key = ?", (self._cid, key)
                    ).rowcount
                else:
                    deleted = conn.execute("DELETE FROM records WHERE key = ?", (key,)).rowcount
            if not deleted:
                raise KeyError(key)

    def clear(self) -> None:
        with self._lock:
            self._cache = None
            self._cache_payload = None
            if self._mem is not None:
                self._mem.clear()
                self._mem_bytes = 0
                return
            with self._db() as conn:
                if self._cid is not None:
                    conn.execute("DELETE FROM dict_records WHERE cid = ?", (self._cid,))
                    self._mem = {}
                    self._mem_bytes = 0
                else:
                    conn.execute("DELETE FROM records")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        from contextlib import nullcontext

        with self._lock:
            self.flush()
            self._cache = None
            if self._mem is not None:
                saved_mem = dict(self._mem)
                saved_bytes = self._mem_bytes
                saved_ord = self._next_ord
                try:
                    yield
                    self.flush()
                except BaseException:
                    self._cache = None
                    if self._mem is None and self._cid is not None:
                        with self._db() as conn:
                            conn.execute("DELETE FROM dict_records WHERE cid = ?", (self._cid,))
                    self._mem = saved_mem
                    self._mem_bytes = saved_bytes
                    self._next_ord = saved_ord
                    raise
                finally:
                    self._cache = None
                return
            spool_guard = self._spool.lock if self._spool is not None else nullcontext()
            with spool_guard:
                savepoint = (
                    self._spool.next_savepoint()
                    if self._spool is not None
                    else "catalog_update_file"
                )
                with self._db() as conn:
                    conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    yield
                    self.flush()
                except BaseException:
                    self._cache = None
                    with self._db() as conn:
                        conn.execute(f"ROLLBACK TO {savepoint}")
                    raise
                finally:
                    self._cache = None
                    with self._db() as conn:
                        conn.execute(f"RELEASE {savepoint}")
