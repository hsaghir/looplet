"""list_dir tool — tree view of a workspace path.

Receives the workspace_config resource through ``ctx.resources``;
``tool.yaml`` declares ``requires: [workspace_config]``.
"""

from __future__ import annotations

from pathlib import Path

from looplet.types import ToolContext

_SKIP = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


def execute(ctx: ToolContext, *, path: str = ".", depth: int = 2) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    target = Path(workspace) / path
    if not target.exists():
        return {"error": f"Not found: {path}"}
    if not target.is_dir():
        return {"error": f"Not a directory: {path}"}
    entries: list[str] = []

    def _walk(p: Path, prefix: str, d: int) -> None:
        if d > depth:
            return
        try:
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return
        for item in items:
            if item.name in _SKIP:
                continue
            if item.is_dir():
                entries.append(f"{prefix}{item.name}/")
                _walk(item, prefix + "  ", d + 1)
            elif len(entries) < 200:
                entries.append(f"{prefix}{item.name}")

    _walk(target, "", 0)
    return {"path": path, "entries": entries, "count": len(entries)}
