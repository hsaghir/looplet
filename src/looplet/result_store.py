"""Optional host-owned storage for complete, content-addressed tool results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from itertools import islice, zip_longest
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from looplet._backend_contract import json_snapshot
from looplet._storage import DiskBackedDict

_CHECKPOINT_REFERENCE = "__looplet_file_result_store_v1__"


@dataclass
class FileToolResultStore:
    """Retain source results and reusable text views in a host-selected directory.

    ``max_bytes`` bounds encoded source and cache content, excluding the small
    catalog. Atomic replacement is not a write-ahead journal or an fsync guarantee.
    Raw steps, snapshots, and reading a selected JSON value can still use memory.
    Share one instance for concurrent calls; a directory has one active writer.
    """

    directory: str
    max_bytes: int | None = None
    _root: Path = field(init=False, repr=False, compare=False)
    _catalog: Any = field(init=False, repr=False, compare=False)
    _used_bytes: int = field(init=False, repr=False, compare=False)
    _lock: Any = field(init=False, repr=False, compare=False)
    _view_lengths: dict[str, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _checkpoint_log: Path | None = field(default=None, init=False, repr=False, compare=False)
    _checkpoint_rowid: int = field(default=0, init=False, repr=False, compare=False)
    _checkpoint_bytes: int = field(default=0, init=False, repr=False, compare=False)
    _checkpoint_digest: Any = field(
        default_factory=hashlib.sha256, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.max_bytes is not None and (type(self.max_bytes) is not int or self.max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer or None")
        self.directory = str(Path(self.directory).resolve())
        self._root = Path(self.directory)
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = RLock()
        catalog = self._root / ".catalog.json"
        self._catalog = DiskBackedDict(
            lambda value: value, lambda value: value, self._root / ".catalog.sqlite3"
        )
        if not self._catalog and catalog.exists():
            with self._catalog.transaction():
                self._catalog.update(json.loads(catalog.read_text(encoding="utf-8")))
        for key in self._catalog:
            self._path(key)
        self._used_bytes = sum(
            path.stat().st_size
            for path in self._root.glob("*.json")
            if re.fullmatch(r"[a-f0-9]{64}", path.stem)
        ) + sum(path.stat().st_size for path in (self._root / ".views").glob("*.txt"))

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or re.fullmatch(r"sha256:[a-f0-9]{64}", key) is None:
            raise ValueError("Expected an opaque sha256 result key")
        return self._root / (key.removeprefix("sha256:") + ".json")

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".write-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _save_catalog(self) -> None:
        self._catalog.flush()

    def _put(self, data: Any) -> str:
        snapshot = json_snapshot(data)
        payload = json.dumps(snapshot, sort_keys=True).encode("utf-8")
        key = "sha256:" + hashlib.sha256(payload).hexdigest()
        if key in self._catalog:
            return key
        path = self._path(key)
        if not path.exists():
            if self.max_bytes is not None and self._used_bytes + len(payload) > self.max_bytes:
                raise OSError("Tool result store content quota exceeded")
            self._write(path, payload)
            self._used_bytes += len(payload)
        elif path.read_bytes() != payload:
            raise ValueError("Retained result content does not match its key")
        entry: dict[str, Any] = {"result_key": key, "type": type(snapshot).__name__}
        if isinstance(snapshot, (str, list, dict)):
            entry["size"] = len(snapshot)
        self._catalog[key] = entry
        return key

    def store_result(self, data: Any) -> str:
        """Store a detached source value; identical values share a key."""
        with self._lock, self._catalog.transaction():
            key = self._put(data)
            self._save_catalog()
            return key

    def get_result(self, key: str) -> Any:
        with self._lock:
            self._path(key)
            if key not in self._catalog:
                raise KeyError(key)
            return self._read_source(key)

    def _read_source(self, key: str) -> Any:
        payload = self._path(key).read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != key:
            raise ValueError("Retained result content does not match its key")
        return json.loads(payload)

    def list_results(
        self, *, offset: int = 0, limit: int = 10, end_offset: int | None = None
    ) -> dict[str, Any]:
        from looplet.builtin_tools.tool_results import _integer

        _integer(offset, "offset")
        _integer(limit, "limit", maximum=20)
        if end_offset is not None:
            _integer(end_offset, "end_offset")
        if not limit:
            raise ValueError("limit must be positive")
        with self._lock:
            boundary = (
                len(self._catalog) if end_offset is None else min(len(self._catalog), end_offset)
            )
            entries = list(islice(self._catalog.values(), offset, min(offset + limit, boundary)))
            end = offset + len(entries)
            return {
                "results": [dict(entry) for entry in entries],
                "total_results": boundary,
                "end_offset": boundary,
                "next_offset": end if end < boundary else None,
            }

    def _text(self, key: str, pointer: str) -> str | Path:
        from looplet.builtin_tools.tool_results import _text

        with self._lock:
            self._path(key)
            if key not in self._catalog:
                raise KeyError(key)
            digest = hashlib.sha256(json.dumps([key, pointer]).encode("utf-8")).hexdigest()
            path = self._root / ".views" / (digest + ".txt")
            if path.exists():
                return path
            text = _text(self.get_result(key), pointer)
            payload = text.encode("utf-8")
            if self.max_bytes is None or self._used_bytes + len(payload) <= self.max_bytes:
                try:
                    path.parent.mkdir(exist_ok=True, mode=0o700)
                    self._write(path, payload)
                except OSError:
                    return text
                self._used_bytes += len(payload)
                if len(self._view_lengths) >= 64:
                    self._view_lengths.pop(next(iter(self._view_lengths)))
                self._view_lengths[path.name] = len(text)
                return path
            return text

    @staticmethod
    def _skip(stream: Any, characters: int) -> None:
        while characters > 0:
            content = stream.read(min(characters, 65536))
            if not content:
                break
            characters -= len(content)

    def _view_length(self, path: Path) -> int:
        if path.name not in self._view_lengths:
            if len(self._view_lengths) >= 64:
                self._view_lengths.pop(next(iter(self._view_lengths)))
            with path.open(encoding="utf-8", newline="") as stream:
                self._view_lengths[path.name] = sum(
                    len(chunk) for chunk in iter(lambda: stream.read(65536), "")
                )
        return self._view_lengths[path.name]

    def read_result(
        self, key: str, *, json_pointer: str = "", offset: int = 0, limit: int = 2000
    ) -> dict[str, Any]:
        from looplet.builtin_tools.tool_results import read_result

        result = read_result(
            "",
            result_key=key,
            json_pointer="",
            offset=offset,
            limit=limit,
        )
        view = self._text(key, json_pointer)
        if isinstance(view, Path):
            with view.open(encoding="utf-8", newline="") as stream:
                self._skip(stream, offset)
                content = stream.read(limit)
            total = self._view_length(view)
            end = offset + len(content)
            result.update(
                content=content, total_chars=total, next_offset=end if end < total else None
            )
        else:
            result = read_result(view, result_key=key, json_pointer="", offset=offset, limit=limit)
        result["json_pointer"] = json_pointer
        return result

    def search_result(
        self, key: str, query: str, *, json_pointer: str = "", offset: int = 0, limit: int = 5
    ) -> dict[str, Any]:
        from looplet.builtin_tools.tool_results import search_result

        result = search_result(
            "",
            result_key=key,
            query=query,
            json_pointer="",
            offset=offset,
            limit=limit,
        )
        result["json_pointer"] = json_pointer
        view = self._text(key, json_pointer)
        if not isinstance(view, Path):
            result = search_result(
                view, result_key=key, query=query, json_pointer="", offset=offset, limit=limit
            )
            result["json_pointer"] = json_pointer
            return result
        result["total_chars"] = self._view_length(view)
        base = max(0, offset - 80)
        cursor = offset
        continuation = None
        buffer = ""
        with view.open(encoding="utf-8", newline="") as stream:
            self._skip(stream, base)
            while True:
                chunk = stream.read(65536)
                buffer += chunk
                safe_end = max(0, len(buffer) - len(query) - 80) if chunk else len(buffer) + 1
                while True:
                    found = buffer.find(query, max(0, cursor - base))
                    if found < 0:
                        break
                    if len(result["matches"]) == limit:
                        result["next_offset"] = continuation
                        return result
                    if found >= safe_end:
                        break
                    result["matches"].append(
                        {
                            "offset": base + found,
                            "excerpt": buffer[max(0, found - 80) : found + len(query) + 80],
                        }
                    )
                    cursor = base + found + len(query)
                    continuation = cursor
                if not chunk:
                    break
                cursor = max(cursor, base + safe_end)
                discard = max(0, cursor - base - 80)
                buffer = buffer[discard:]
                base += discard
        return result

    def snapshot_results(self) -> dict[str, Any]:
        with self._lock:
            return {key: self.get_result(key) for key in self._catalog}

    def checkpoint_results(self) -> dict[str, Any]:
        """Bind an append-only catalog prefix without loading source values."""
        with self._lock:
            directory = self._root / ".snapshots"
            directory.mkdir(exist_ok=True, mode=0o700)
            if self._checkpoint_log is None:
                path = directory / ("catalog-" + uuid4().hex + ".jsonl")
                path.touch(mode=0o600, exist_ok=False)
                self._checkpoint_log = path
            digest = self._checkpoint_digest.copy()
            position = self._checkpoint_rowid
            length = self._checkpoint_bytes
            with self._checkpoint_log.open("ab") as stream:
                if stream.tell() != length:
                    raise ValueError("Checkpoint result catalog length changed")
                try:
                    for position, _key, entry in self._catalog.items_after(position):
                        payload = json.dumps(entry, sort_keys=True).encode("utf-8") + b"\n"
                        stream.write(payload)
                        digest.update(payload)
                        length += len(payload)
                    stream.flush()
                except BaseException:
                    stream.truncate(self._checkpoint_bytes)
                    raise
            self._checkpoint_rowid = position
            self._checkpoint_bytes = length
            self._checkpoint_digest = digest
            return {
                _CHECKPOINT_REFERENCE: {
                    "directory": self.directory,
                    "catalog": self._checkpoint_log.name,
                    "bytes": length,
                    "sha256": digest.hexdigest(),
                }
            }

    @staticmethod
    def _checkpoint_source(snapshot: dict[str, Any]) -> tuple[FileToolResultStore, Path, int]:
        reference = snapshot.get(_CHECKPOINT_REFERENCE)
        if not isinstance(reference, dict) or set(reference) not in (
            {"sha256", "directory"},
            {"sha256", "directory", "catalog", "bytes"},
        ):
            raise ValueError("Invalid checkpoint result store reference")
        checksum = reference["sha256"]
        if not isinstance(checksum, str) or re.fullmatch(r"[a-f0-9]{64}", checksum) is None:
            raise ValueError("Invalid checkpoint result catalog digest")
        if not isinstance(reference["directory"], str):
            raise ValueError("Invalid checkpoint result store directory")
        directory = Path(reference["directory"]).resolve()
        if not directory.is_dir():
            raise ValueError("Checkpoint result store is unavailable")
        if "catalog" in reference:
            name, length = reference["catalog"], reference["bytes"]
            if (
                not isinstance(name, str)
                or re.fullmatch(r"catalog-[a-f0-9]{32}\.jsonl", name) is None
            ):
                raise ValueError("Invalid checkpoint result catalog name")
            if type(length) is not int or length < 0:
                raise ValueError("Invalid checkpoint result catalog length")
            path = directory / ".snapshots" / name
        else:
            path = directory / ".snapshots" / (checksum + ".jsonl")
            length = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            remaining = length
            while remaining:
                payload = stream.read(min(remaining, 65536))
                if not payload:
                    raise ValueError("Checkpoint result catalog is incomplete")
                digest.update(payload)
                remaining -= len(payload)
            if digest.hexdigest() != checksum:
                raise ValueError("Checkpoint result catalog digest does not match")
        return FileToolResultStore(str(directory)), path, length

    @staticmethod
    def _checkpoint_entries(path: Path, length: int) -> Any:
        with path.open("rb") as stream:
            remaining = length
            while remaining:
                line = stream.readline(min(remaining, 65536))
                if not line.endswith(b"\n"):
                    raise ValueError("Invalid checkpoint result catalog entry")
                remaining -= len(line)
                yield json.loads(line)

    @staticmethod
    def checkpoint_values(snapshot: dict[str, Any]) -> Any:
        """Iterate verified source values for explicit restore into another store."""
        source, path, length = FileToolResultStore._checkpoint_source(snapshot)
        for entry in FileToolResultStore._checkpoint_entries(path, length):
            key = entry["result_key"]
            yield key, source._read_source(key)

    def _reset_checkpoint_catalog(self) -> None:
        self._checkpoint_log = None
        self._checkpoint_rowid = 0
        self._checkpoint_bytes = 0
        self._checkpoint_digest = hashlib.sha256()

    def restore_results(self, snapshot: dict[str, Any]) -> None:
        if set(snapshot) == {_CHECKPOINT_REFERENCE}:
            source, path, length = self._checkpoint_source(snapshot)
            if source._root == self._root:
                with self._lock:
                    keys = (entry["result_key"] for entry in self._checkpoint_entries(path, length))
                    if any(left != right for left, right in zip_longest(keys, self._catalog)):
                        raise ValueError("Restore an earlier catalog into a separate result store")
                    for key in self._catalog:
                        self.get_result(key)
                return
            with self._lock, self._catalog.transaction():
                self._catalog.clear()
                for entry in self._checkpoint_entries(path, length):
                    key = entry["result_key"]
                    if self._put(source._read_source(key)) != key:
                        raise ValueError("Checkpoint result content does not match its key")
                self._save_catalog()
                self._reset_checkpoint_catalog()
            return
        with self._lock, self._catalog.transaction():
            self._catalog.clear()
            for key, data in snapshot.items():
                if self._put(data) != key:
                    raise ValueError("Checkpoint result content does not match its key")
            self._save_catalog()
            self._reset_checkpoint_catalog()
