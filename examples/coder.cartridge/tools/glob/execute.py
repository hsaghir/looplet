"""glob tool - match files by glob pattern, return relative paths.

Receives the workspace_config resource through ``ctx.resources``;
``tool.yaml`` declares ``requires: [workspace_config]``.
"""

from __future__ import annotations

from pathlib import Path

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, pattern: str) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    return {
        "pattern": pattern,
        "matches": sorted(
            str(path.relative_to(workspace))
            for path in Path(workspace).glob(pattern)
            if path.is_file()
        )[:100],
    }
