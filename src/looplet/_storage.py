"""Private disk-backed sequences for complete records with a bounded working set."""

from __future__ import annotations

import copy
import json
import pickle
import sqlite3
import tempfile
import weakref
from collections.abc import Callable, Iterable, Iterator, MutableMapping, MutableSequence, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Generic, TypeVar, overload

Record = TypeVar("Record")


def _close_records(connection: sqlite3.Connection, directory: tempfile.TemporaryDirectory) -> None:
    connection.close()
    directory.cleanup()


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
        directory = tempfile.TemporaryDirectory(prefix="looplet-records-")
        self._connection = sqlite3.connect(
            directory.name + "/records.sqlite3", isolation_level=None, check_same_thread=False
        )
        self._connection.execute("PRAGMA cache_size = -64")
        self._connection.execute("PRAGMA journal_mode = OFF")
        self._connection.execute(
            "CREATE TABLE records (position INTEGER PRIMARY KEY, payload BLOB, weight INTEGER NOT NULL)"
        )
        self._finalizer = weakref.finalize(self, _close_records, self._connection, directory)
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
                self._connection.execute(
                    "UPDATE records SET weight = ? WHERE position = ?", (weight, position)
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
                self._connection.execute(
                    "UPDATE records SET payload = ? WHERE position = ?", (payload, index)
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
                row = self._connection.execute(
                    "SELECT payload, weight FROM records WHERE position = ?", (position,)
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
            previous_weight = self._connection.execute(
                "SELECT weight FROM records WHERE position = ?", (position,)
            ).fetchone()[0]
            self._connection.execute(
                "UPDATE records SET payload = ?, weight = ? WHERE position = ?",
                (payload, weight, position),
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
            weight = self._connection.execute(
                "SELECT weight FROM records WHERE position = ?", (position,)
            ).fetchone()[0]
            self._connection.execute("DELETE FROM records WHERE position = ?", (position,))
            self._connection.execute(
                "UPDATE records SET position = -position WHERE position > ?", (position,)
            )
            self._connection.execute(
                "UPDATE records SET position = -position - 1 WHERE position < 0"
            )
            self._length -= 1
            self._total_weight -= weight

    def insert(self, index: int, value: Record) -> None:
        with self._lock:
            position = max(0, min(index if index >= 0 else self._length + index, self._length))
            self._flush()
            self._cache = None
            payload = self._dumps(value)
            weight = self._weight(value) if self._weight is not None else 0
            self._connection.execute(
                "UPDATE records SET position = -position - 1 WHERE position >= ?", (position,)
            )
            self._connection.execute("UPDATE records SET position = -position WHERE position < 0")
            self._connection.execute(
                "INSERT INTO records VALUES (?, ?, ?)",
                (position, payload, weight),
            )
            self._length += 1
            self._total_weight += weight
            self._cache = (position, value)
            self._cache_payload = payload
            self._cache_weight = weight

    def clear(self) -> None:
        with self._lock:
            self._cache = None
            self._connection.execute("DELETE FROM records")
            self._length = 0
            self._total_weight = 0

    def sort(self, *, key: Callable[[Record], Any], reverse: bool = False) -> None:
        with self._lock:
            positions = sorted(
                range(self._length), key=lambda position: key(self[position]), reverse=reverse
            )
            self._flush()
            self._cache = None
            self._connection.execute("UPDATE records SET position = -position - 1")
            for position, original in enumerate(positions):
                self._connection.execute(
                    "UPDATE records SET position = ? WHERE position = ?", (position, -original - 1)
                )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(left == right for left, right in zip(self, other, strict=True))
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Record]:
        return [copy.deepcopy(value, memo) for value in self]

    def snapshot(self) -> DiskBackedSequence[Record]:
        """Return an independent read-only snapshot without loading all records."""
        with self._lock:
            self._flush()
            records = DiskBackedList(
                self._encode,
                self._decode,
                preserve_python=self._preserve_python,
                weight=self._weight,
            )
            self._connection.backup(records._connection)
            records._length = self._length
            records._total_weight = self._total_weight
            return DiskBackedSequence(records)


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
        directory = None
        if path is None:
            directory = tempfile.TemporaryDirectory(prefix="looplet-catalog-")
            path = Path(directory.name) / "records.sqlite3"
        self._connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA cache_size = -64")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, payload TEXT)"
        )
        self._finalizer = weakref.finalize(self, self._close, self._connection, directory)
        self._encode = encode
        self._decode = decode
        self._cache: tuple[str, Record] | None = None
        self._cache_payload: str | None = None
        self._lock = RLock()

    @staticmethod
    def _close(connection: sqlite3.Connection, directory: Any) -> None:
        connection.close()
        if directory is not None:
            directory.cleanup()

    def flush(self) -> None:
        if self._cache is not None:
            key, value = self._cache
            payload = json.dumps(self._encode(value), default=str)
            if payload != self._cache_payload:
                self._connection.execute(
                    "UPDATE records SET payload = ? WHERE key = ?", (payload, key)
                )
                self._cache_payload = payload

    def __len__(self) -> int:
        with self._lock:
            return self._connection.execute("SELECT count(*) FROM records").fetchone()[0]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self.flush()
            rows = self._connection.execute("SELECT key FROM records ORDER BY rowid")
            for row in rows:
                yield row[0]

    def items_after(self, position: int) -> Iterator[tuple[int, str, Record]]:
        """Iterate newly appended catalog records from a stable row position."""
        with self._lock:
            self.flush()
            rows = self._connection.execute(
                "SELECT rowid, key, payload FROM records WHERE rowid > ? ORDER BY rowid",
                (position,),
            )
            for row in rows:
                yield row[0], row[1], self._decode(json.loads(row[2]))

    def __getitem__(self, key: str) -> Record:
        with self._lock:
            if self._cache is None or self._cache[0] != key:
                self.flush()
                row = self._connection.execute(
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
            self._connection.execute(
                "INSERT INTO records VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET payload = excluded.payload",
                (key, payload),
            )
            self._cache = (key, value)
            self._cache_payload = payload

    def __delitem__(self, key: str) -> None:
        with self._lock:
            self.flush()
            self._cache = None
            if not self._connection.execute("DELETE FROM records WHERE key = ?", (key,)).rowcount:
                raise KeyError(key)

    def clear(self) -> None:
        with self._lock:
            self._cache = None
            self._connection.execute("DELETE FROM records")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            self.flush()
            self._cache = None
            self._connection.execute("SAVEPOINT catalog_update")
            try:
                yield
                self.flush()
            except BaseException:
                self._cache = None
                self._connection.execute("ROLLBACK TO catalog_update")
                raise
            finally:
                self._cache = None
                self._connection.execute("RELEASE catalog_update")
