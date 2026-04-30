"""read_file tool — read with line numbers + cache integration."""

from __future__ import annotations

from examples.coder.tools import _resolve_safe_path

WORKSPACE_CONFIG = None
FILE_CACHE = None


def execute(*, file_path: str, start_line: int = 0, end_line: int = 0) -> dict:
    workspace = WORKSPACE_CONFIG.path if WORKSPACE_CONFIG is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    if not p.exists():
        return {"error": f"File not found: {file_path}"}
    # file_unchanged optimization shared with other coder tools.
    if (
        FILE_CACHE is not None
        and start_line == 0
        and end_line == 0
        and FILE_CACHE.is_unchanged(file_path)
    ):
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
        if FILE_CACHE is not None:
            FILE_CACHE.record(file_path)
        return {"path": file_path, "content": content, "total_lines": len(lines)}
    except Exception as e:
        return {"error": str(e)}
