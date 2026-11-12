"""Immutable JSON records shared by automatic checkpoint manifests."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import weakref
from collections import OrderedDict
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from looplet._storage import TrackedList

_DIGEST = re.compile(r"[0-9a-f]{64}")
_FANOUT = 32
_CACHE_BYTES = 64 * 1024
_CACHE_ENTRIES = 1024


def _identity_encode(item: Any) -> Any:
    return item


class CheckpointSequence:
    """Capture a fixed count of owned records with a stable, pure encoder."""

    def __init__(self, values: Any, encode: Any) -> None:
        self.values = values
        self.encode = encode
        self.length = len(values)

    def __iter__(self):
        for index in range(self.length):
            yield self.encode(self.values[index])


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite checkpoint value: {value}")


class CheckpointDocuments:
    """Store JSON trees by content hash, sharing unchanged values and list chunks."""

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / ".checkpoint-records.sqlite3"
        self._write_connection: sqlite3.Connection | None = None
        self._file_identity: tuple[int, int] | None = None
        self._data_version: int | None = None
        self._encoders: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
        self._lock = RLock()
        self._record_cache: OrderedDict[str, bytes] | None = None
        self._record_cache_bytes = 0

    def _connect(self, *, write: bool) -> sqlite3.Connection:
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
            connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records (digest TEXT PRIMARY KEY, payload BLOB NOT NULL)"
            )
            return connection
        if not self.path.is_file():
            raise sqlite3.OperationalError("checkpoint database does not exist")
        try:
            connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            connection.execute("PRAGMA query_only=ON")
        except sqlite3.OperationalError:
            connection = sqlite3.connect(
                f"file:{quote(str(self.path.resolve()))}?mode=ro", uri=True
            )
        connection.execute("PRAGMA cache_size=-512")
        return connection

    def _put(self, connection: sqlite3.Connection, value: Any) -> str:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        digest = sha256(payload).hexdigest()
        cache = self._record_cache if connection is self._write_connection else None
        if cache is not None:
            previous = cache.pop(digest, None)
            if previous is not None:
                if previous != payload:
                    raise ValueError("checkpoint record hash mismatch")
                cache[digest] = previous
                return digest
        inserted = connection.execute(
            "INSERT OR IGNORE INTO records VALUES (?, ?)", (digest, payload)
        )
        if not inserted.rowcount:
            previous = connection.execute(
                "SELECT payload FROM records WHERE digest = ?", (digest,)
            ).fetchone()
            if previous is None or previous[0] != payload:
                raise ValueError("checkpoint record hash mismatch")
        if cache is not None and len(payload) <= _CACHE_BYTES:
            cache[digest] = payload
            self._record_cache_bytes += len(payload)
            while self._record_cache_bytes > _CACHE_BYTES or len(cache) > _CACHE_ENTRIES:
                _, removed = cache.popitem(last=False)
                self._record_cache_bytes -= len(removed)
        return digest

    def _sequence_set(
        self,
        connection: sqlite3.Connection,
        root: str | None,
        length: int,
        position: int,
        digest: str,
    ) -> str:
        height, capacity = 0, _FANOUT
        while length > capacity:
            height += 1
            capacity *= _FANOUT
        if length == capacity and position == length:
            root = self._put(connection, ["concat", [root]])
            height += 1

        def replace(reference: str | None, level: int, offset: int) -> str:
            kind = "concat" if level else "list"
            children = [] if reference is None else list(self._node(connection, reference)[1])
            if level:
                index, child_offset = divmod(offset, _FANOUT**level)
                child = replace(
                    children[index] if index < len(children) else None, level - 1, child_offset
                )
            else:
                index, child = offset, digest
            if index == len(children):
                children.append(child)
            else:
                children[index] = child
            return self._put(connection, [kind, children])

        return replace(root, height, position)

    def _encode_sequence(
        self, connection: sqlite3.Connection, value: CheckpointSequence, active: set[int]
    ) -> str:
        encoder = self._encoders.get(value.encode)
        if encoder is None:
            encoder = self._encoders[value.encode] = uuid4().hex
        source = value.values._record_identity
        previous = connection.execute(
            "SELECT generation, revision, length, root FROM checkpoint_sequences "
            "WHERE source = ? AND encoder = ?",
            (source, encoder),
        ).fetchone()
        cursor = (previous[0], previous[1], previous[2]) if previous is not None else (-1, -1, 0)
        with value.values.changes_since(cursor, value.length) as (generation, revision, records):
            if previous is not None and cursor[0] == generation and cursor[2] <= value.length:
                root, length = previous[3], previous[2]
                for position, record in records:
                    digest = self._encode(connection, value.encode(record), active)
                    root = self._sequence_set(connection, root, length, position, digest)
                    length = max(length, position + 1)
            else:
                root = self._encode_list(
                    connection, (value.encode(record) for _, record in records), active
                )
            connection.execute(
                "INSERT INTO checkpoint_sequences VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source, encoder) DO UPDATE SET generation = excluded.generation, "
                "revision = excluded.revision, length = excluded.length, root = excluded.root",
                (source, encoder, generation, revision, value.length, root),
            )
            return root

    def _encode_list(self, connection: sqlite3.Connection, values: Any, active: set[int]) -> str:
        chunks = []
        children = []
        for child in values:
            children.append(self._encode(connection, child, active))
            if len(children) == _FANOUT:
                chunks.append(self._put(connection, ["list", children]))
                children = []
        if children or not chunks:
            chunks.append(self._put(connection, ["list", children]))
        while len(chunks) > 1:
            chunks = [
                self._put(connection, ["concat", chunks[offset : offset + _FANOUT]])
                for offset in range(0, len(chunks), _FANOUT)
            ]
        return chunks[0]

    def _encode(self, connection: sqlite3.Connection, value: Any, active: set[int]) -> str:
        if isinstance(value, (dict, list, tuple, CheckpointSequence)):
            identity = id(value)
            if identity in active:
                raise ValueError("checkpoint contains circular data")
            active.add(identity)
            try:
                if isinstance(value, CheckpointSequence) and callable(
                    getattr(value.values, "changes_since", None)
                ):
                    return self._encode_sequence(connection, value, active)
                if isinstance(value, TrackedList):
                    return self._encode_sequence(
                        connection, CheckpointSequence(value, _identity_encode), active
                    )
                if isinstance(value, dict):
                    members = []
                    for key, child in value.items():
                        if isinstance(key, str):
                            name = key
                        elif key is None or isinstance(key, (bool, int, float)):
                            name = json.dumps(key, allow_nan=False)
                        else:
                            raise ValueError("checkpoint object keys must be JSON values")
                        members.append([name, self._encode(connection, child, active)])
                    return self._put(connection, ["object", members])
                return self._encode_list(connection, value, active)
            finally:
                active.remove(identity)
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError("checkpoint contains non-JSON-safe data")
        return self._put(connection, ["value", value])

    def save(self, payload: dict[str, Any], summary: dict[str, Any]) -> str:
        with self._lock:
            if self._write_connection is None:
                connection = self._connect(write=True)
                weakref.finalize(self, connection.close)
                connection.execute(
                    "CREATE TEMP TABLE checkpoint_sequences (source TEXT, encoder TEXT, "
                    "generation INTEGER, revision INTEGER, length INTEGER, root TEXT, "
                    "PRIMARY KEY (source, encoder))"
                )
                self._write_connection = connection
                identity = self.path.stat()
                self._file_identity = (identity.st_dev, identity.st_ino)
            connection = self._write_connection
            identity = self.path.stat()
            if (identity.st_dev, identity.st_ino) != self._file_identity:
                raise ValueError("checkpoint record database was replaced")
            self._record_cache = OrderedDict()
            try:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    version = connection.execute("PRAGMA data_version").fetchone()[0]
                    if version != self._data_version:
                        connection.execute("DELETE FROM checkpoint_sequences")
                    root = self._encode(
                        connection, {"checkpoint": payload, "summary": summary}, set()
                    )
                self._data_version = version
                return root
            finally:
                self._record_cache = None
                self._record_cache_bytes = 0

    @staticmethod
    def _node(connection: sqlite3.Connection, digest: str) -> list[Any]:
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError("invalid checkpoint record reference")
        row = connection.execute(
            "SELECT payload FROM records WHERE digest = ?", (digest,)
        ).fetchone()
        if row is None:
            raise ValueError("checkpoint record is missing")
        payload = row[0]
        if not isinstance(payload, bytes) or sha256(payload).hexdigest() != digest:
            raise ValueError("checkpoint record hash mismatch")
        node = json.loads(payload, parse_constant=_reject_constant)
        if not isinstance(node, list) or len(node) != 2 or not isinstance(node[0], str):
            raise ValueError("invalid checkpoint record")
        kind, value = node
        if kind == "value":
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError("invalid checkpoint scalar")
        elif kind == "object":
            if not isinstance(value, list) or any(
                not isinstance(member, list) or len(member) != 2 or not isinstance(member[0], str)
                for member in value
            ):
                raise ValueError("invalid checkpoint object")
        elif kind not in {"list", "concat"} or not isinstance(value, list):
            raise ValueError("invalid checkpoint container")
        return node

    def verify(self, root: str, verified: set[str]) -> None:
        def visit(connection: sqlite3.Connection, digest: str) -> None:
            if digest in verified:
                return
            kind, value = self._node(connection, digest)
            if kind == "object":
                for _, child in value:
                    visit(connection, child)
            elif kind in {"list", "concat"}:
                for child in value:
                    if kind == "concat" and self._node(connection, child)[0] not in {
                        "list",
                        "concat",
                    }:
                        raise ValueError("invalid checkpoint list chunk")
                    visit(connection, child)
            verified.add(digest)

        try:
            with closing(self._connect(write=False)) as connection:
                visit(connection, root)
        except sqlite3.Error as error:
            raise ValueError("cannot read checkpoint records") from error

    def _decode(
        self, connection: sqlite3.Connection, digest: str, fields: dict[str, Any] | None = None
    ) -> Any:
        kind, value = self._node(connection, digest)
        if kind == "value":
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError("invalid checkpoint scalar")
            return value
        if not isinstance(value, list):
            raise ValueError("invalid checkpoint container")
        if kind == "list":
            return [self._decode(connection, child) for child in value]
        if kind == "concat":
            result = []
            for child in value:
                chunk = self._decode(connection, child)
                if not isinstance(chunk, list):
                    raise ValueError("invalid checkpoint list chunk")
                result.extend(chunk)
            return result
        if kind == "object":
            result_object = {}
            for member in value:
                if (
                    not isinstance(member, list)
                    or len(member) != 2
                    or not isinstance(member[0], str)
                ):
                    raise ValueError("invalid checkpoint object member")
                if fields is None or member[0] in fields:
                    result_object[member[0]] = self._decode(
                        connection, member[1], fields.get(member[0]) if fields is not None else None
                    )
            return result_object
        raise ValueError("unknown checkpoint record kind")

    def load(
        self, root: str, *, summary_only: bool = False, fields: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            with closing(self._connect(write=False)) as connection:
                kind, members = self._node(connection, root)
                if kind != "object" or not isinstance(members, list):
                    raise ValueError("invalid checkpoint document")
                references = dict(members)
                if set(references) != {"checkpoint", "summary"}:
                    raise ValueError("invalid checkpoint document fields")
                selected = references["summary" if summary_only else "checkpoint"]
                result = self._decode(connection, selected, fields)
                if not isinstance(result, dict):
                    raise ValueError("checkpoint document must be an object")
                return result
        except sqlite3.Error as error:
            raise ValueError("cannot read checkpoint records") from error
