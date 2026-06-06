"""edit_file tool — exact-string replacement with fuzzy fallback hints.

Receives ``workspace_config`` and ``file_cache`` through
``ctx.resources``; ``tool.yaml`` declares
``requires: [workspace_config, file_cache]``.
"""

from __future__ import annotations

import difflib

from coder_lib_tools import _fuzzy_find, _resolve_safe_path

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, file_path: str, old_string: str, new_string: str) -> dict:
    cfg = ctx.resources.get("workspace_config")
    cache = ctx.resources.get("file_cache")
    workspace = cfg.path if cfg is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    if not p.exists():
        return {"error": f"File not found: {file_path}"}
    # Read-before-edit enforcement. Editing a file the model hasn't
    # read in this session is almost always a sign that the model is
    # guessing at content — the resulting old_string usually doesn't
    # match and the call wastes a turn. Refuse early with a message
    # that names the read tool so the model self-corrects.
    if cache is not None and not cache.was_read(file_path):
        return {
            "error": (
                f"Cannot edit {file_path!r}: this file has not been read in "
                f"the current session. Call read_file({file_path!r}) first so "
                f"you can copy old_string verbatim from its content. Editing "
                f"without reading first usually fails because the file's "
                f"actual contents differ from what was assumed."
            ),
            "missing": "prior_read",
            "recovery": f"read_file(file_path={file_path!r})",
        }
    if old_string == new_string:
        return {"error": "old_string and new_string are identical. No change needed."}
    text = p.read_text()
    count = text.count(old_string)
    if count == 1:
        new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text)
        if cache is not None:
            cache.invalidate(file_path)
        diff = difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=3,
        )
        diff_text = "".join(diff)
        if len(diff_text) > 2000:
            diff_text = diff_text[:2000] + "\n... [diff truncated]"
        return {"edited": file_path, "replacements": 1, "diff": diff_text}
    if count > 1:
        return {
            "error": (
                f"Matches {count} locations. Include more surrounding context for a unique match."
            ),
            "matches": count,
        }
    fuzzy = _fuzzy_find(text, old_string)
    if fuzzy:
        hints = [f"  line {n} ({r:.0%}): {t.strip()[:80]}" for n, r, t in fuzzy[:3]]
        return {
            "error": (
                "Exact match not found. Similar lines:\n"
                + "\n".join(hints)
                + "\n\nRECOVERY: read_file at those lines, then retry with exact text."
            ),
            "similar_lines": [f[0] for f in fuzzy[:3]],
        }
    return {"error": f"Not found in {file_path}. Use read_file to see exact content."}
