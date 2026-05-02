"""read_file tool — read with line numbers + cache integration.

Receives ``workspace_config`` and ``file_cache`` through
``ctx.resources``; ``tool.yaml`` declares
``requires: [workspace_config, file_cache]``.
"""

from __future__ import annotations

from coder_lib_tools import _resolve_safe_path, is_binary_file, read_text_with_fallback

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
    if p.is_dir():
        return {
            "error": f"{file_path!r} is a directory, not a file.",
            "recovery": f"list_dir(path={file_path!r})",
        }
    # Binary detection — prevents read_file from returning garbage
    # bytes that would blow up the model's tokenizer or look like a
    # legitimate edit target.
    is_binary, reason = is_binary_file(p)
    if is_binary:
        size = p.stat().st_size
        return {
            "error": (
                f"Refused: {file_path!r} appears to be a binary file "
                f"({reason}, {size} bytes). read_file only handles text."
            ),
            "binary": True,
            "size_bytes": size,
            "recovery": (
                "If you really need to inspect bytes, use bash with `xxd "
                f"{file_path} | head` (still subject to the cat-on-source "
                "refusal for project text files)."
            ),
        }
    # file_unchanged optimization shared with other coder tools.
    if cache is not None and start_line == 0 and end_line == 0 and cache.is_unchanged(file_path):
        return {
            "path": file_path,
            "file_unchanged": True,
            "note": "File has not changed since your last read. No need to re-read.",
        }
    text, encoding = read_text_with_fallback(p)
    if text is None:
        return {"error": f"Could not read {file_path!r}: {encoding}"}
    lines = text.splitlines()
    if start_line < 0 or end_line < 0:
        return {
            "error": "start_line and end_line must be non-negative (1-based).",
            "got": {"start_line": start_line, "end_line": end_line},
        }
    if start_line > 0 and end_line > 0 and end_line < start_line:
        return {
            "error": f"end_line ({end_line}) must be >= start_line ({start_line}).",
        }
    if start_line > 0 and end_line > 0:
        selected = lines[start_line - 1 : end_line]
        numbered = [f"{start_line + i:>4} | {line}" for i, line in enumerate(selected)]
    elif start_line > 0:
        selected = lines[start_line - 1 :]
        numbered = [f"{start_line + i:>4} | {line}" for i, line in enumerate(selected)]
    else:
        numbered = [f"{i + 1:>4} | {line}" for i, line in enumerate(lines)]
    content = "\n".join(numbered)
    truncated = False
    if len(content) > 20000:
        truncated = True
        content = (
            content[:10000]
            + f"\n... [{len(content) - 20000} chars truncated — re-read with start_line/end_line] ...\n"
            + content[-10000:]
        )
    if cache is not None:
        cache.record(file_path)
    result: dict = {
        "path": file_path,
        "content": content,
        "total_lines": len(lines),
    }
    if encoding != "utf-8":
        result["encoding"] = encoding
    if truncated:
        result["truncated"] = True
        result["recovery_hint"] = (
            "Use start_line/end_line to read specific ranges instead of the full file."
        )
    return result
