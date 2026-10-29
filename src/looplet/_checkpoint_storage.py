"""Immutable JSON records shared by automatic checkpoint manifests."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote

_DIGEST = re.compile(r"[0-9a-f]{64}")
_FANOUT = 32


class CheckpointSequence:
    """Capture a fixed record count and encode one owned record at a time."""

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

    def _connect(self, *, write: bool) -> sqlite3.Connection:
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
            connection = sqlite3.connect(self.path, timeout=10)
        else:
            connection = sqlite3.connect(
                f"file:{quote(str(self.path.resolve()))}?mode=ro", uri=True
            )
        connection.execute("PRAGMA cache_size=-64")
        if write:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records (digest TEXT PRIMARY KEY, payload BLOB NOT NULL)"
            )
        return connection

    @staticmethod
    def _put(connection: sqlite3.Connection, value: Any) -> str:
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        digest = sha256(payload).hexdigest()
        inserted = connection.execute(
            "INSERT OR IGNORE INTO records VALUES (?, ?)", (digest, payload)
        )
        if not inserted.rowcount:
            previous = connection.execute(
                "SELECT payload FROM records WHERE digest = ?", (digest,)
            ).fetchone()
            if previous is None or previous[0] != payload:
                raise ValueError("checkpoint record hash mismatch")
        return digest

    def _encode(self, connection: sqlite3.Connection, value: Any, active: set[int]) -> str:
        if isinstance(value, (dict, list, tuple, CheckpointSequence)):
            identity = id(value)
            if identity in active:
                raise ValueError("checkpoint contains circular data")
            active.add(identity)
            try:
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
                chunks = []
                children = []
                for child in value:
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
            finally:
                active.remove(identity)
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError("checkpoint contains non-JSON-safe data")
        return self._put(connection, ["value", value])

    def save(self, payload: dict[str, Any], summary: dict[str, Any]) -> str:
        with closing(self._connect(write=True)) as connection, connection:
            return self._encode(connection, {"checkpoint": payload, "summary": summary}, set())

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
                    visit(connection, child)
            verified.add(digest)

        try:
            with closing(self._connect(write=False)) as connection:
                visit(connection, root)
        except sqlite3.Error as error:
            raise ValueError("cannot read checkpoint records") from error

    def _decode(self, connection: sqlite3.Connection, digest: str) -> Any:
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
                result_object[member[0]] = self._decode(connection, member[1])
            return result_object
        raise ValueError("unknown checkpoint record kind")

    def load(self, root: str, *, summary_only: bool = False) -> dict[str, Any]:
        try:
            with closing(self._connect(write=False)) as connection:
                kind, members = self._node(connection, root)
                if kind != "object" or not isinstance(members, list):
                    raise ValueError("invalid checkpoint document")
                references = dict(members)
                if set(references) != {"checkpoint", "summary"}:
                    raise ValueError("invalid checkpoint document fields")
                selected = references["summary" if summary_only else "checkpoint"]
                result = self._decode(connection, selected)
                if not isinstance(result, dict):
                    raise ValueError("checkpoint document must be an object")
                return result
        except sqlite3.Error as error:
            raise ValueError("cannot read checkpoint records") from error
