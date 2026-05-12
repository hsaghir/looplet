"""Write a UTF-8 text file."""

from __future__ import annotations

from pathlib import Path


def execute(ctx, *, path: str, content: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        runtime = ctx.resources.get("runtime") or {}
        # Resolve the project root via the standard helper so the
        # host doesn't have to pass any runtime kwargs when running
        # from inside the target project.
        from looplet.cartridge.runtime_helpers import resolve_project_root  # noqa: PLC0415

        root = resolve_project_root(runtime)
        p = Path(root) / p
    p.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data)}
