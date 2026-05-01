"""bash tool — execute a shell command in the workspace root.

Receives the workspace_config resource through ``ctx.resources``
(workspace tool DI; ``tool.yaml`` declares ``requires:
[workspace_config]``). Top-level function (no closures) so it
round-trips losslessly through ``preset_to_workspace``.

The actual command-execution logic lives in ``coder_lib_tools._run``
so this file stays a thin adapter and all tools share one
battle-tested implementation.
"""

from __future__ import annotations

import re
from pathlib import Path

from coder_lib_tools import _is_path_inside, _run

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, command: str) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
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
