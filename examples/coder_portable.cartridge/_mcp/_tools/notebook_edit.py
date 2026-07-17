"""notebook_edit tool - structural JSON edits for .ipynb files."""

from __future__ import annotations

import json
import uuid
from typing import Any

from coder_lib_tools import _resolve_safe_path, atomic_write_text

from looplet.types import ToolContext


def _as_source(value: str, like: Any | None = None) -> str | list[str]:
    if isinstance(like, list):
        if not value:
            return []
        return value.splitlines(keepends=True)
    return value


def _make_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell_id = uuid.uuid4().hex[:8]
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def execute(
    ctx: ToolContext,
    *,
    notebook_path: str,
    cell_id: str = "",
    new_source: str = "",
    cell_type: str = "",
    edit_mode: str = "replace",
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    cache = ctx.resources.get("file_cache")
    workspace = cfg.path if cfg is not None else "."
    p = _resolve_safe_path(workspace, notebook_path)
    if p is None:
        return {"error": f"Path '{notebook_path}' is outside the project directory."}
    if p.suffix != ".ipynb":
        return {"error": "notebook_edit only edits .ipynb files"}
    if not p.exists():
        return {"error": f"Notebook not found: {notebook_path}"}
    if cache is not None:
        if not cache.was_read(notebook_path):
            return {
                "error": f"Cannot edit {notebook_path!r}: notebook has not been read in this session.",
                "missing": "prior_read",
                "recovery": f"read_file(file_path={notebook_path!r})",
            }
        if not cache.is_unchanged(notebook_path):
            return {
                "error": f"Cannot edit {notebook_path!r}: notebook changed since last read.",
                "stale": True,
                "recovery": f"read_file(file_path={notebook_path!r})",
            }
    try:
        notebook = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"Notebook is not valid JSON: {exc}"}
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return {"error": "Notebook JSON missing cells list"}
    mode = edit_mode.strip().lower() or "replace"
    if mode not in {"replace", "insert", "delete"}:
        return {
            "error": f"unknown edit_mode {edit_mode!r}",
            "valid_modes": ["replace", "insert", "delete"],
        }
    normalized_type = cell_type.strip().lower()
    if normalized_type and normalized_type not in {"code", "markdown"}:
        return {"error": "cell_type must be code or markdown"}

    target_index = next(
        (i for i, cell in enumerate(cells) if isinstance(cell, dict) and cell.get("id") == cell_id),
        None,
    )
    if mode in {"replace", "delete"} and target_index is None:
        return {
            "error": f"cell_id {cell_id!r} not found",
            "available_cell_ids": [cell.get("id") for cell in cells if isinstance(cell, dict)],
        }

    original_cell_count = len(cells)
    if mode == "replace":
        cell = cells[target_index]
        if not isinstance(cell, dict):
            return {"error": f"cell_id {cell_id!r} is not an object cell"}
        final_type = normalized_type or str(cell.get("cell_type", "code"))
        cell["cell_type"] = final_type
        cell["source"] = _as_source(new_source, cell.get("source"))
        if final_type == "code":
            cell.setdefault("execution_count", None)
            cell.setdefault("outputs", [])
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
        changed_cell_id = cell.get("id")
    elif mode == "insert":
        final_type = normalized_type or "code"
        new_cell = _make_cell(final_type, new_source)
        insert_at = len(cells) if target_index is None else target_index + 1
        cells.insert(insert_at, new_cell)
        changed_cell_id = new_cell["id"]
    else:
        removed = cells.pop(target_index)
        changed_cell_id = removed.get("id") if isinstance(removed, dict) else cell_id

    atomic_write_text(p, json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    if cache is not None:
        cache.invalidate(notebook_path)
    return {
        "notebook_path": notebook_path,
        "edit_mode": mode,
        "cell_id": changed_cell_id,
        "cell_type": normalized_type or ("" if mode == "delete" else "code"),
        "old_cell_count": original_cell_count,
        "new_cell_count": len(cells),
    }
