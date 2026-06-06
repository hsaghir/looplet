"""todo tool — persistent checklist for coder.workspace sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from looplet.types import ToolContext

_VALID_STATUSES = {"not-started", "in-progress", "completed", "blocked"}


def _todo_path(workspace: str) -> Path:
    scratch = Path(workspace) / ".coder_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch / "todos.json"


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _save(path: Path, todos: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(todos, indent=2) + "\n")


def _normalize_items(items: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"todo item #{index} must be an object")
        title = str(raw.get("title", "")).strip()
        if not title:
            raise ValueError(f"todo item #{index} missing non-empty title")
        status = str(raw.get("status", "not-started")).strip() or "not-started"
        if status not in _VALID_STATUSES:
            raise ValueError(f"todo item #{index} has invalid status {status!r}")
        item: dict[str, Any] = {
            "id": int(raw.get("id") or index),
            "title": title,
            "status": status,
        }
        notes = str(raw.get("notes", "")).strip()
        if notes:
            item["notes"] = notes
        normalized.append(item)
    # Reassign ids so list state stays compact and deterministic.
    for index, item in enumerate(normalized, start=1):
        item["id"] = index
    return normalized


def _status_counts(todos: list[dict[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for item in todos if item.get("status") == status)
        for status in sorted(_VALID_STATUSES)
    }


def execute(
    ctx: ToolContext,
    *,
    operation: str = "list",
    todos: list | None = None,
    id: int = 0,
    title: str = "",
    status: str = "",
    notes: str = "",
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    path = _todo_path(workspace)
    op = operation.strip().lower()
    current = _load(path)

    try:
        if op == "list":
            updated = current
        elif op == "replace":
            updated = _normalize_items(list(todos or []))
            _save(path, updated)
        elif op == "add":
            item_title = title.strip()
            if not item_title:
                return {"error": "operation=add requires a non-empty title"}
            item_status = status.strip() or "not-started"
            if item_status not in _VALID_STATUSES:
                return {
                    "error": f"invalid status {item_status!r}",
                    "valid_statuses": sorted(_VALID_STATUSES),
                }
            updated = list(current)
            item: dict[str, Any] = {
                "id": len(updated) + 1,
                "title": item_title,
                "status": item_status,
            }
            item_notes = notes.strip()
            if item_notes:
                item["notes"] = item_notes
            updated.append(item)
            _save(path, updated)
        elif op == "update":
            if id <= 0:
                return {"error": "operation=update requires id > 0"}
            updated = list(current)
            target = next((item for item in updated if item.get("id") == id), None)
            if target is None:
                return {"error": f"todo id {id} not found", "todos": current}
            if title.strip():
                target["title"] = title.strip()
            if status.strip():
                item_status = status.strip()
                if item_status not in _VALID_STATUSES:
                    return {
                        "error": f"invalid status {item_status!r}",
                        "valid_statuses": sorted(_VALID_STATUSES),
                    }
                target["status"] = item_status
            if notes.strip():
                target["notes"] = notes.strip()
            _save(path, updated)
        elif op == "clear":
            updated = []
            _save(path, updated)
        else:
            return {
                "error": f"unknown operation {operation!r}",
                "valid_operations": ["list", "replace", "add", "update", "clear"],
            }
    except ValueError as exc:
        return {"error": str(exc)}

    return {
        "operation": op,
        "todos": updated,
        "count": len(updated),
        "status_counts": _status_counts(updated),
        "path": ".coder_scratch/todos.json",
    }
