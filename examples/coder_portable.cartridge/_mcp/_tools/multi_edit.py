"""multi_edit tool — atomic batch of exact-string replacements on one file.

Many real edits to a single file (e.g. updating an import, changing a
function signature, and tweaking a call site) are naturally a batch.
Doing them as N separate ``edit_file`` calls means N×(read + edit +
verify) round-trips. ``multi_edit`` applies all edits in order
against an in-memory copy of the file and writes the result
atomically — either every edit succeeds or none do.

Receives ``workspace_config`` and ``file_cache`` through
``ctx.resources``; ``tool.yaml`` declares
``requires: [workspace_config, file_cache]``.
"""

from __future__ import annotations

import difflib

from coder_lib_tools import (
    _fuzzy_find,
    _resolve_safe_path,
    atomic_write_text,
    is_binary_file,
    read_text_with_fallback,
)

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, file_path: str, edits: list) -> dict:
    cfg = ctx.resources.get("workspace_config")
    cache = ctx.resources.get("file_cache")
    workspace = cfg.path if cfg is not None else "."
    p = _resolve_safe_path(workspace, file_path)
    if p is None:
        return {"error": f"Path '{file_path}' is outside the project directory."}
    if not p.exists():
        return {"error": f"File not found: {file_path}"}
    if p.is_dir():
        return {"error": f"{file_path!r} is a directory, not a file."}
    is_binary, reason = is_binary_file(p)
    if is_binary:
        return {"error": f"Refused: {file_path!r} is binary ({reason})."}
    # Same read-required-first discipline as edit_file.
    if cache is not None and not cache.was_read(file_path):
        return {
            "error": (
                f"Cannot edit {file_path!r}: file has not been read in "
                f"the current session. Call read_file({file_path!r}) first."
            ),
            "missing": "prior_read",
            "recovery": f"read_file(file_path={file_path!r})",
        }
    if not isinstance(edits, list) or not edits:
        return {
            "error": "edits must be a non-empty list of {old_string, new_string} dicts.",
        }
    # Validate edit shapes up front.
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return {"error": f"edit #{i + 1} must be a dict, got {type(edit).__name__}"}
        if "old_string" not in edit or "new_string" not in edit:
            return {
                "error": (
                    f"edit #{i + 1} missing required keys; need "
                    "{old_string, new_string} (replace_all is optional)."
                ),
                "missing_keys": [k for k in ("old_string", "new_string") if k not in edit],
            }
    text, encoding = read_text_with_fallback(p)
    if text is None:
        return {"error": f"Could not read {file_path!r}: {encoding}"}
    original = text
    applied: list[dict] = []
    for i, edit in enumerate(edits):
        old = edit["old_string"]
        new = edit["new_string"]
        replace_all = bool(edit.get("replace_all", False))
        if old == new:
            return {
                "error": f"edit #{i + 1}: old_string and new_string are identical.",
                "applied_so_far": applied,
            }
        if old not in text:
            fuzzy = _fuzzy_find(text, old)
            hints = (
                "\n".join(f"  line {n} ({r:.0%}): {t.strip()[:80]}" for n, r, t in fuzzy[:3])
                if fuzzy
                else "  (no similar lines found)"
            )
            return {
                "error": (
                    f"edit #{i + 1} not found in {file_path!r}. "
                    f"Similar lines:\n{hints}\n\nNo edits applied (atomic)."
                ),
                "failed_edit_index": i,
                "applied_so_far": applied,
                "similar_lines": [f[0] for f in fuzzy[:3]] if fuzzy else [],
            }
        count = text.count(old)
        if count > 1 and not replace_all:
            return {
                "error": (
                    f"edit #{i + 1} matches {count} locations in {file_path!r}. "
                    "Add more context for uniqueness, or pass replace_all=true."
                ),
                "failed_edit_index": i,
                "matches": count,
                "applied_so_far": applied,
            }
        if replace_all:
            text = text.replace(old, new)
            applied.append({"index": i, "replacements": count})
        else:
            text = text.replace(old, new, 1)
            applied.append({"index": i, "replacements": 1})
    # All edits succeeded — write atomically.
    atomic_write_text(p, text, encoding=encoding if encoding != "could not decode" else "utf-8")
    if cache is not None:
        cache.invalidate(file_path)
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        text.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=3,
    )
    diff_text = "".join(diff)
    if len(diff_text) > 4000:
        diff_text = diff_text[:4000] + "\n... [diff truncated]"
    return {
        "edited": file_path,
        "edits_applied": len(applied),
        "total_replacements": sum(a["replacements"] for a in applied),
        "diff": diff_text,
    }
