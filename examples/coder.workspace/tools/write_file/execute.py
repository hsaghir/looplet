"""write_file tool — create / overwrite a file inside the workspace.

Receives ``workspace_config`` and ``file_cache`` through
``ctx.resources``; ``tool.yaml`` declares
``requires: [workspace_config, file_cache]``.
"""

from __future__ import annotations

from coder_lib_tools import _resolve_safe_path, atomic_write_text

from looplet.types import ToolContext


def execute(
    ctx: ToolContext,
    *,
    file_path: str,
    content: str,
    overwrite: bool = False,
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    cache = ctx.resources.get("file_cache")
    workspace = cfg.path if cfg is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    if p.is_dir():
        return {
            "error": f"{file_path!r} is a directory, not a file.",
        }
    # Safer-overwrite: refuse to clobber an existing file unless
    # the caller explicitly opts in. This prevents the model from
    # accidentally wiping a file it should have edited instead.
    if p.exists() and not overwrite:
        return {
            "error": (
                f"Refused: {file_path!r} already exists. write_file "
                "is for NEW files; use edit_file to modify existing "
                "files (preserves the rest of the file + produces a "
                "reviewable diff). If you really intend to replace "
                "the entire file, pass `overwrite=True`."
            ),
            "exists": True,
            "recovery": (
                f"edit_file(file_path={file_path!r}, ...)  OR  write_file(..., overwrite=True)"
            ),
        }
    atomic_write_text(p, content)
    if cache is not None:
        cache.invalidate(file_path)
    return {
        "written": file_path,
        "lines": content.count("\n") + (0 if content.endswith("\n") or not content else 1),
        "overwritten": p.exists() and overwrite,
    }
