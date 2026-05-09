"""Write a UTF-8 text file."""

from __future__ import annotations

from pathlib import Path


def execute(ctx, *, path: str, content: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        runtime = ctx.resources.get("runtime") or {}
        root = runtime.get("project_root", ".")
        p = Path(root) / p
    p.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data)}
