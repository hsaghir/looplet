"""read_file tool — read with line numbers + cache integration.

Receives ``workspace_config`` and ``file_cache`` through
``ctx.resources``; ``tool.yaml`` declares
``requires: [workspace_config, file_cache]``.
"""

from __future__ import annotations

from coder_lib_tools import _resolve_safe_path

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, file_path: str, start_line: int = 0, end_line: int = 0) -> dict:
    cfg = ctx.resources.get("workspace_config")
    cache = ctx.resources.get("file_cache")
    workspace = cfg.path if cfg is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    if not p.exists():
        return {"error": f"File not found: {file_path}"}
    # file_unchanged optimization shared with other coder tools.
    if cache is not None and start_line == 0 and end_line == 0 and cache.is_unchanged(file_path):
        return {
            "path": file_path,
            "file_unchanged": True,
            "note": "File has not changed since your last read. No need to re-read.",
        }
    try:
        lines = p.read_text().splitlines()
        if start_line > 0 and end_line > 0:
            selected = lines[start_line - 1 : end_line]
            numbered = [f"{start_line + i:>4} | {line}" for i, line in enumerate(selected)]
        elif start_line > 0:
            selected = lines[start_line - 1 :]
            numbered = [f"{start_line + i:>4} | {line}" for i, line in enumerate(selected)]
        else:
            numbered = [f"{i + 1:>4} | {line}" for i, line in enumerate(lines)]
        content = "\n".join(numbered)
        if len(content) > 20000:
            content = (
                content[:10000]
                + f"\n... [{len(content) - 20000} chars truncated] ...\n"
                + content[-10000:]
            )
        if cache is not None:
            cache.record(file_path)
        return {"path": file_path, "content": content, "total_lines": len(lines)}
    except Exception as e:
        return {"error": str(e)}
