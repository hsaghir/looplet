"""write_file tool — create / overwrite a file inside the workspace.

Receives ``workspace_config`` and ``file_cache`` through
``ctx.resources``; ``tool.yaml`` declares
``requires: [workspace_config, file_cache]``.
"""

from __future__ import annotations

from coder_lib_tools import _resolve_safe_path

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, file_path: str, content: str) -> dict:
    cfg = ctx.resources.get("workspace_config")
    cache = ctx.resources.get("file_cache")
    workspace = cfg.path if cfg is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    if cache is not None:
        cache.invalidate(file_path)
    return {"written": file_path, "lines": content.count("\n") + 1}
