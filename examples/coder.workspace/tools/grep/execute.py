"""grep tool — recursive grep with workspace-relative output paths."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from examples.coder.tools import _resolve_safe_path

WORKSPACE_CONFIG = None


def execute(*, pattern: str, path: str = ".", include: str = "") -> dict:
    workspace = WORKSPACE_CONFIG.path if WORKSPACE_CONFIG is not None else "."
    target = _resolve_safe_path(workspace, path)
    if target is None:
        return {"error": f"Path '{path}' is outside the project directory."}
    cmd = ["grep", "-rn"]
    if include:
        cmd.append(f"--include={include}")
    cmd.extend(["--", pattern, str(target)])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=workspace,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "pattern": pattern, "matches": [], "count": 0}
    lines = result.stdout.splitlines() if result.stdout else []
    workspace_prefix = str(Path(workspace).resolve()) + os.sep
    relative_lines = [
        line.removeprefix(workspace_prefix) if line.startswith(workspace_prefix) else line
        for line in lines
    ]
    data = {"pattern": pattern, "matches": relative_lines[:50], "count": len(relative_lines)}
    if result.returncode not in (0, 1):
        data["error"] = result.stderr.strip() or f"grep exited {result.returncode}"
    return data
