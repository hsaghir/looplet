"""glob tool — match files by glob pattern, return relative paths."""

from __future__ import annotations

from pathlib import Path

WORKSPACE_CONFIG = None


def execute(*, pattern: str) -> dict:
    workspace = WORKSPACE_CONFIG.path if WORKSPACE_CONFIG is not None else "."
    return {
        "pattern": pattern,
        "matches": sorted(
            str(path.relative_to(workspace))
            for path in Path(workspace).glob(pattern)
            if path.is_file()
        )[:100],
    }
