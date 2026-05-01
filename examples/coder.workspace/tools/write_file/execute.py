"""write_file tool — create / overwrite a file inside the workspace."""

from __future__ import annotations

from coder_lib_tools import _resolve_safe_path

WORKSPACE_CONFIG = None
FILE_CACHE = None


def execute(*, file_path: str, content: str) -> dict:
    workspace = WORKSPACE_CONFIG.path if WORKSPACE_CONFIG is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    if FILE_CACHE is not None:
        FILE_CACHE.invalidate(file_path)
    return {"written": file_path, "lines": content.count("\n") + 1}
