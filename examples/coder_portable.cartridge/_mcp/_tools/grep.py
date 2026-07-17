"""grep tool - rg-first recursive search with workspace-relative output paths.

Receives the workspace_config resource through ``ctx.resources``;
``tool.yaml`` declares ``requires: [workspace_config]``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from coder_lib_tools import _resolve_safe_path

from looplet.types import ToolContext


def _relativize(lines: list[str], workspace: str) -> list[str]:
    workspace_prefix = str(Path(workspace).resolve()) + os.sep
    return [
        line.removeprefix(workspace_prefix) if line.startswith(workspace_prefix) else line
        for line in lines
    ]


def _page(items: list[str], head_limit: int, offset: int) -> tuple[list[str], bool]:
    safe_offset = max(0, int(offset or 0))
    if head_limit == 0:
        return items[safe_offset:], False
    limit = max(1, int(head_limit or 250))
    sliced = items[safe_offset : safe_offset + limit]
    return sliced, len(items) - safe_offset > limit


def execute(
    ctx: ToolContext,
    *,
    pattern: str,
    path: str = ".",
    include: str = "",
    glob: str = "",
    output_mode: str = "content",
    type: str = "",
    context: int = 0,
    before: int = 0,
    after: int = 0,
    case_insensitive: bool = False,
    head_limit: int = 250,
    offset: int = 0,
    multiline: bool = False,
    max_count: int = 0,
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    target = _resolve_safe_path(workspace, path)
    if target is None:
        return {"error": f"Path '{path}' is outside the project directory."}
    if not target.exists():
        return {"error": f"Path not found: {path}", "matches": [], "count": 0}

    mode = (output_mode or "content").strip().lower()
    if mode not in {"content", "files_with_matches", "count"}:
        return {
            "error": f"unknown output_mode {output_mode!r}",
            "valid_modes": ["content", "files_with_matches", "count"],
        }

    rg = shutil.which("rg")
    file_glob = glob or include
    if rg:
        cmd = [rg, "--color", "never"]
        if mode == "content":
            cmd.append("--line-number")
            if context > 0:
                cmd.extend(["-C", str(context)])
            else:
                if before > 0:
                    cmd.extend(["-B", str(before)])
                if after > 0:
                    cmd.extend(["-A", str(after)])
        elif mode == "files_with_matches":
            cmd.append("--files-with-matches")
        else:
            cmd.append("--count-matches")
        if case_insensitive:
            cmd.append("-i")
        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])
        if file_glob:
            cmd.extend(["--glob", file_glob])
        if type:
            cmd.extend(["--type", type])
        if max_count > 0:
            cmd.extend(["--max-count", str(max_count)])
        cmd.extend(["--", pattern, str(target)])
        engine = "rg"
    else:
        cmd = ["grep", "-rn"]
        if case_insensitive:
            cmd.append("-i")
        if mode == "files_with_matches":
            cmd.append("-l")
        elif mode == "count":
            cmd.append("-c")
        elif context > 0:
            cmd.extend(["-C", str(context)])
        else:
            if before > 0:
                cmd.extend(["-B", str(before)])
            if after > 0:
                cmd.extend(["-A", str(after)])
        if file_glob:
            cmd.append(f"--include={file_glob}")
        cmd.extend(["--", pattern, str(target)])
        engine = "grep"
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=workspace,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": "Search timed out",
            "pattern": pattern,
            "matches": [],
            "count": 0,
            "engine": engine,
        }
    lines = _relativize(result.stdout.splitlines() if result.stdout else [], workspace)
    paged, truncated = _page(lines, head_limit=head_limit, offset=offset)
    data = {
        "pattern": pattern,
        "path": path,
        "mode": mode,
        "engine": engine,
        "matches": paged,
        "count": len(lines),
        "truncated": truncated,
        "offset": max(0, int(offset or 0)),
    }
    if mode == "files_with_matches":
        data["filenames"] = paged
    elif mode == "count":
        data["counts"] = paged
    else:
        data["content"] = "\n".join(paged)
    if result.returncode not in (0, 1):
        data["error"] = result.stderr.strip() or f"grep exited {result.returncode}"
    if truncated:
        data["recovery"] = (
            "Use a narrower path/glob/type, increase offset, or set head_limit=0 only when the result is known to be small."
        )
    return data
