"""list_dir tool — tree view of a workspace path."""

from __future__ import annotations

from pathlib import Path

WORKSPACE_CONFIG = None

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


def execute(*, path: str = ".", depth: int = 2) -> dict:
    workspace = WORKSPACE_CONFIG.path if WORKSPACE_CONFIG is not None else "."
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
