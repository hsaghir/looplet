"""bash tool — execute a shell command in the workspace root.

Receives the workspace_config resource through ``ctx.resources``
(workspace tool DI; ``tool.yaml`` declares ``requires:
[workspace_config]``). Top-level function (no closures) so it
round-trips losslessly through ``preset_to_workspace``.

Pre-dispatch checks (model-actionable, fail-loud rather than silent):

* **Destructive command detection.** A small allow-list of
  patterns (``rm -rf``, ``git push --force``, ``dd``, ``mkfs``,
  process killers) refuses execution and points at the safer
  alternative. Novel commands flow through unchanged — the
  permission engine is the principled gate.
* **sed -i refusal.** ``sed`` in-place edits bypass the file_cache,
  causing the next ``read_file`` to return stale content. Refuses
  with a pointer at ``edit_file``.
* **cd-outside-workspace warning.** Surfaces (does not block) a
  ``cd`` whose target escapes the project root.

The actual command-execution logic lives in ``coder_lib_tools._run``
so this file stays a thin adapter and all tools share one
battle-tested implementation.
"""

from __future__ import annotations

import re
from pathlib import Path

from coder_lib_tools import (
    _is_path_inside,
    _run,
    classify_bash_command,
    classify_sed_command,
    classify_view_command,
)

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, command: str) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."

    # Destructive-command pre-flight. Returns model-actionable error
    # naming each reason so the model can adjust the next call.
    classification = classify_bash_command(command)
    if classification["destructive"]:
        return {
            "error": (
                "Refused: destructive command pattern detected. "
                + " ".join(classification["reasons"])
                + ". If this is intentional, ask the user to confirm."
            ),
            "first_token": classification["first_token"],
            "reasons": classification["reasons"],
            "recovery": (
                "Pick a safer alternative (e.g. remove specific files instead "
                "of `rm -rf`; create a NEW commit instead of `git reset --hard`)."
            ),
        }

    # sed -i refusal: route in-place edits through edit_file so the
    # file_cache stays coherent.
    sed = classify_sed_command(command)
    if sed["in_place_edit"]:
        return {
            "error": (
                "Refused: `sed -i` in-place file edits bypass the file_cache. "
                + sed["recommendation"]
            ),
            "recovery": "edit_file(file_path=..., old_string=..., new_string=...)",
        }

    # cat/head/tail/less on a source file refusal: route file viewing
    # through read_file so the file_cache records the read and the
    # next edit_file can succeed.
    view = classify_view_command(command)
    if view["viewing_file"]:
        return {
            "error": (
                f"Refused: `{view['first_token']}` on a source file bypasses "
                "the file_cache. " + view["recommendation"]
            ),
            "first_token": view["first_token"],
            "recovery": "read_file(file_path=...)",
        }

    result = _run(command, workspace)

    # Detect cd outside workspace and surface a warning so the model
    # doesn't silently lose track of where commands run.
    parts = re.split(r"&&|\|\||;|\n", command)
    for part in parts:
        part = part.strip()
        if part.startswith("cd "):
            target = part[3:].strip().strip("'\"")
            resolved = Path(workspace) / target
            try:
                resolved = resolved.resolve()
                ws_resolved = Path(workspace).resolve()
                if not _is_path_inside(resolved, ws_resolved):
                    result["cwd_warning"] = (
                        f"Warning: 'cd {target}' points outside the project directory. "
                        f"All commands run in the project root. Use relative paths."
                    )
            except Exception:
                pass
    return result
