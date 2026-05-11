"""worktree tool — managed git worktree helper."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from looplet.types import ToolContext


def _run(workspace: str, args: list[str]) -> dict:
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, timeout=60
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _safe_name(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        return None
    return cleaned


def _managed_root(workspace: str) -> Path:
    root = Path(workspace).resolve()
    return root.parent / f"{root.name}.worktrees"


def execute(
    ctx: ToolContext,
    *,
    operation: str = "list",
    name: str = "",
    base_ref: str = "HEAD",
    confirm: bool = False,
    force: bool = False,
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    probe = _run(workspace, ["rev-parse", "--is-inside-work-tree"])
    if probe["exit_code"] != 0:
        return {"error": "Not inside a git worktree", "stderr": probe["stderr"]}
    op = operation.strip().lower() or "list"
    managed = _managed_root(workspace)
    if op == "list":
        result = _run(workspace, ["worktree", "list", "--porcelain"])
        result.update({"operation": op, "managed_root": str(managed)})
        return result
    safe = _safe_name(name)
    if safe is None:
        return {"error": "name must contain only letters, numbers, dot, dash, and underscore"}
    target = (managed / safe).resolve()
    if managed.resolve() not in target.parents:
        return {"error": "resolved worktree path escapes managed root"}
    if op == "create":
        if target.exists():
            return {"error": f"managed worktree already exists: {target}"}
        managed.mkdir(parents=True, exist_ok=True)
        result = _run(workspace, ["worktree", "add", str(target), base_ref or "HEAD"])
        result.update({"operation": op, "path": str(target), "managed_root": str(managed)})
        return result
    if op == "remove":
        if not confirm:
            return {"error": "operation=remove requires confirm=true", "path": str(target)}
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(target))
        result = _run(workspace, args)
        result.update({"operation": op, "path": str(target), "managed_root": str(managed)})
        return result
    return {
        "error": f"unknown operation {operation!r}",
        "valid_operations": ["list", "create", "remove"],
    }
