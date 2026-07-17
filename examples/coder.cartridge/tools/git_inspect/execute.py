"""git_inspect tool - safe read-only git helpers."""

from __future__ import annotations

import subprocess

from looplet.types import ToolContext


def _run(workspace: str, args: list[str]) -> dict:
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, timeout=30
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def execute(
    ctx: ToolContext, *, operation: str = "status", pathspec: str = "", max_count: int = 10
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    probe = _run(workspace, ["rev-parse", "--is-inside-work-tree"])
    if probe["exit_code"] != 0:
        return {"error": "Not inside a git worktree", "stderr": probe["stderr"]}
    op = operation.strip().lower() or "status"
    if op == "status":
        args = ["status", "--short", "--branch"]
    elif op == "diff":
        args = ["diff"] + (["--", pathspec] if pathspec else [])
    elif op == "diff_stat":
        args = ["diff", "--stat"] + (["--", pathspec] if pathspec else [])
    elif op == "branch":
        args = ["branch", "--show-current"]
    elif op == "recent":
        args = ["log", "--oneline", "--decorate", "-n", str(max(1, int(max_count or 10)))]
    elif op == "changed_files":
        args = ["diff", "--name-only"] + (["--", pathspec] if pathspec else [])
    else:
        return {
            "error": f"unknown operation {operation!r}",
            "valid_operations": [
                "status",
                "diff",
                "diff_stat",
                "branch",
                "recent",
                "changed_files",
            ],
        }
    result = _run(workspace, args)
    result.update({"operation": op, "command": "git " + " ".join(args)})
    if len(result["stdout"]) > 20000:
        result["stdout"] = (
            result["stdout"][:10000]
            + "\n... [git output truncated] ...\n"
            + result["stdout"][-10000:]
        )
        result["truncated"] = True
    return result
