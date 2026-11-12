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
        self.connection.execute("PRAGMA journal_mode = OFF")
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
        self._next_cid = 0
        self.closed = False

    def alloc_cid(self) -> int:
        with self.lock:
            self._next_cid += 1
            return self._next_cid

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
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
    if spool.closed or spool.pid != os.getpid():
        return
    try:
        with spool.lock:
            if not spool.closed:
                spool.connection.execute(f"DELETE FROM {table} WHERE cid = ?", (cid,))
    except Exception:
        pass


class TrackedList(list):
    """Native list subclass tracking per-index revisions for O(delta) checkpointing."""

    def __init__(self, values: Iterable[Any] = ()) -> None:
        super().__init__(values)
        self._record_identity = uuid4().hex
        self._generation = 0
        n = len(self)
        self._revision = n
        self._item_revisions: list[int] = list(range(1, n + 1))
        self._detached = False

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(item, memo) for item in self]

    def _detach_if_needed(self) -> None:
        if self._detached:
            self._record_identity = uuid4().hex
            self._detached = False

    def append(self, value: Any) -> None:
        self._detach_if_needed()
        super().append(value)
        self._revision += 1
        self._item_revisions.append(self._revision)

    def extend(self, values: Iterable[Any]) -> None:
        items = list(values)
        if not items:
            return
        self._detach_if_needed()
        super().extend(items)
        start = self._revision + 1
        self._revision += len(items)
        self._item_revisions.extend(range(start, self._revision + 1))

    def insert(self, index: Any, value: Any) -> None:
        self._detach_if_needed()
        pos = max(0, min(int(index) if int(index) >= 0 else len(self) + int(index), len(self)))
        if pos == len(self):
            super().append(value)
            self._revision += 1
            self._item_revisions.append(self._revision)
            return
        super().insert(pos, value)
        self._generation += 1
        self._revision += 1
        self._item_revisions.insert(pos, self._revision)

    def pop(self, index: Any = -1) -> Any:
        self._detach_if_needed()
        idx = int(index)
        is_last = idx == -1 or idx == len(self) - 1
        val = super().pop(idx)
        self._item_revisions.pop(idx)
        if not is_last:
            self._generation += 1
        self._revision += 1
        return val

    def clear(self) -> None:
        self._detach_if_needed()
        super().clear()
        self._item_revisions.clear()
        self._generation += 1
        self._revision += 1

    def reverse(self) -> None:
        self._detach_if_needed()
        super().reverse()
        self._generation += 1
        self._revision += 1
        self._item_revisions = [self._revision] * len(self)

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._detach_if_needed()
        super().sort(key=key, reverse=reverse)
        self._generation += 1
        self._revision += 1
        self._item_revisions = [self._revision] * len(self)

    def mark_modified(self, index: int) -> None:
        if 0 <= index < len(self):
            self._detach_if_needed()
            self._revision += 1
            self._item_revisions[index] = self._revision

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
            else:
                self._generation += 1
                self._revision += 1
                self._item_revisions = [self._revision] * len(self)
            return
        pos = int(index)
        if pos < 0:
            pos += len(self)
        super().__setitem__(pos, value)
        self._revision += 1
        self._item_revisions[pos] = self._revision

    def __delitem__(self, index: Any) -> None:
        self._detach_if_needed()
        super().__delitem__(index)
        del self._item_revisions[index]
        self._generation += 1
        self._revision += 1

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
        clone._detached = True
        return clone

    @contextmanager
    def changes_since(
        self, cursor: tuple[int, int, int], stop: int
    ) -> Iterator[tuple[int, int, Iterator[tuple[int, Any]]]]:
        if stop > len(self):
            raise IndexError("record index out of range")
        generation, revision, length = cursor
        if generation != self._generation or stop < length:
            changed = [(pos, self[pos]) for pos in range(stop)]
        else:
            limit = min(length, stop)
            changed = [
                (pos, self[pos]) for pos in range(limit) if self._item_revisions[pos] > revision
            ]
            if stop > length:
                changed.extend((pos, self[pos]) for pos in range(length, stop))
        yield self._generation, self._revision, iter(changed)


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
        self._lock = RLock()
        self._record_identity = uuid4().hex
        self._generation = 0
        self._revision = 0
        self.extend(values)

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
                with self._spool.lock:
                    row = self._spool.connection.execute(
                        "SELECT payload, weight FROM list_records WHERE cid = ? AND position = ?",
                        (self._cid, position),
                    ).fetchone()
                self._cache = (position, self._loads(row[0]))
                self._cache_payload = row[0]
                self._cache_weight = row[1]
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

    def insert(self, index: int, value: Record) -> None:
        with self._lock:
            position = max(0, min(index if index >= 0 else self._length + index, self._length))
            self._flush()
            self._cache = None
            payload = self._dumps(value)
            weight = self._weight(value) if self._weight is not None else 0
            self._revision += 1
            with self._spool.lock:
                if position < self._length:
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
                    self._generation += 1
                self._spool.connection.execute(
                    "INSERT INTO list_records VALUES (?, ?, ?, ?, ?)",
                    (self._cid, position, payload, weight, self._revision),
                )
            self._length += 1
            self._total_weight += weight
            self._cache = (position, value)
            self._cache_payload = payload
            self._cache_weight = weight

    def clear(self) -> None:
        with self._lock:
            self._cache = None
            with self._spool.lock:
                self._spool.connection.execute(
                    "DELETE FROM list_records WHERE cid = ?", (self._cid,)
                )
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
            if self._spool is records._spool:
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
            else:
                records.extend(self)
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
            self._connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            self._connection.execute("PRAGMA cache_size = -1024")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, payload TEXT)"
            )
            self._finalizer = weakref.finalize(self, self._close_file, self._connection)

    @staticmethod
    def _close_file(connection: sqlite3.Connection) -> None:
        connection.close()

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        if self._spool is not None:
            with self._spool.lock:
                yield self._spool.connection
        else:
            yield self._connection

    def flush(self) -> None:
        if self._cache is not None:
            key, value = self._cache
            payload = json.dumps(self._encode(value), default=str)
            if payload != self._cache_payload:
                with self._db() as conn:
                    if self._cid is not None:
                        conn.execute(
                            "UPDATE dict_records SET payload = ? WHERE cid = ? AND key = ?",
                            (payload, self._cid, key),
                        )
                    else:
                        conn.execute("UPDATE records SET payload = ? WHERE key = ?", (payload, key))
                self._cache_payload = payload

    def __len__(self) -> int:
        with self._lock, self._db() as conn:
            if self._cid is not None:
                return conn.execute(
                    "SELECT count(*) FROM dict_records WHERE cid = ?", (self._cid,)
                ).fetchone()[0]
            return conn.execute("SELECT count(*) FROM records").fetchone()[0]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self.flush()
            with self._db() as conn:
                if self._cid is not None:
                    rows = conn.execute(
                        "SELECT key FROM dict_records WHERE cid = ? ORDER BY ord", (self._cid,)
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT key FROM records ORDER BY rowid").fetchall()
            for row in rows:
                yield row[0]

    def items_after(self, position: int) -> Iterator[tuple[int, str, Record]]:
        """Iterate newly appended catalog records from a stable row position."""
        with self._lock:
            self.flush()
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
                self._cache = (key, self._decode(json.loads(row[0])))
                self._cache_payload = row[0]
            return self._cache[1]

    def __setitem__(self, key: str, value: Record) -> None:
        with self._lock:
            self.flush()
            payload = json.dumps(self._encode(value), default=str)
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
            self._cache = (key, value)
            self._cache_payload = payload

    def __delitem__(self, key: str) -> None:
        with self._lock:
            self.flush()
            self._cache = None
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
            with self._db() as conn:
                if self._cid is not None:
                    conn.execute("DELETE FROM dict_records WHERE cid = ?", (self._cid,))
                else:
                    conn.execute("DELETE FROM records")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self.flush()
            self._cache = None
            savepoint = f"catalog_update_{self._cid or 0}"
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
