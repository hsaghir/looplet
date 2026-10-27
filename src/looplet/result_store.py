"""Optional host-owned storage for complete, content-addressed tool results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from looplet._backend_contract import json_snapshot


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
    _catalog: dict[str, dict[str, Any]] = field(init=False, repr=False, compare=False)
    _used_bytes: int = field(init=False, repr=False, compare=False)
    _lock: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_bytes is not None and (type(self.max_bytes) is not int or self.max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer or None")
        self.directory = str(Path(self.directory).resolve())
        self._root = Path(self.directory)
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = RLock()
        catalog = self._root / ".catalog.json"
        self._catalog = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else {}
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
        self._write(self._root / ".catalog.json", json.dumps(self._catalog).encode("utf-8"))

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
        with self._lock:
            previous = dict(self._catalog)
            try:
                key = self._put(data)
                self._save_catalog()
                return key
            except BaseException:
                self._catalog = previous
                raise

    def get_result(self, key: str) -> Any:
        with self._lock:
            path = self._path(key)
            if key not in self._catalog:
                raise KeyError(key)
            payload = path.read_bytes()
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
            entries = list(self._catalog.values())[offset : min(offset + limit, boundary)]
            end = offset + len(entries)
            return {
                "results": [dict(entry) for entry in entries],
                "total_results": boundary,
                "end_offset": boundary,
                "next_offset": end if end < boundary else None,
            }

    def _text(self, key: str, pointer: str) -> str:
        from looplet.builtin_tools.tool_results import _text

        with self._lock:
            self._path(key)
            if key not in self._catalog:
                raise KeyError(key)
            digest = hashlib.sha256(json.dumps([key, pointer]).encode("utf-8")).hexdigest()
            path = self._root / ".views" / (digest + ".txt")
            if path.exists():
                with path.open(encoding="utf-8", newline="") as stream:
                    return stream.read()
            text = _text(self.get_result(key), pointer)
            payload = text.encode("utf-8")
            if self.max_bytes is None or self._used_bytes + len(payload) <= self.max_bytes:
                try:
                    path.parent.mkdir(exist_ok=True, mode=0o700)
                    self._write(path, payload)
                except OSError:
                    return text
                self._used_bytes += len(payload)
            return text

    def read_result(
        self, key: str, *, json_pointer: str = "", offset: int = 0, limit: int = 2000
    ) -> dict[str, Any]:
        from looplet.builtin_tools.tool_results import read_result

        result = read_result(
            self._text(key, json_pointer),
            result_key=key,
            json_pointer="",
            offset=offset,
            limit=limit,
        )
        result["json_pointer"] = json_pointer
        return result

    def search_result(
        self, key: str, query: str, *, json_pointer: str = "", offset: int = 0, limit: int = 5
    ) -> dict[str, Any]:
        from looplet.builtin_tools.tool_results import search_result

        result = search_result(
            self._text(key, json_pointer),
            result_key=key,
            query=query,
            json_pointer="",
            offset=offset,
            limit=limit,
        )
        result["json_pointer"] = json_pointer
        return result

    def snapshot_results(self) -> dict[str, Any]:
        with self._lock:
            return {key: self.get_result(key) for key in self._catalog}

    def restore_results(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            previous = dict(self._catalog)
            try:
                for key, data in snapshot.items():
                    if self._put(data) != key:
                        raise ValueError("Checkpoint result content does not match its key")
                self._catalog = {key: self._catalog[key] for key in snapshot}
                self._save_catalog()
            except BaseException:
                self._catalog = previous
                raise
