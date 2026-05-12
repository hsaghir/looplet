"""Read a UTF-8 text file."""

from __future__ import annotations

from pathlib import Path


def execute(ctx, *, path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        runtime = ctx.resources.get("runtime") or {}
        root = runtime.get("project_root", ".")
        p = Path(root) / p
    if not p.is_file():
        return {"error": f"file not found: {p}"}
    text = p.read_text(encoding="utf-8", errors="replace")
    return {"content": text, "size": len(text), "lines": text.count("\n") + 1}
