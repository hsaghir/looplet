"""bash tool — execute a shell command in the workspace root.

Top-level function (no closures) so it round-trips losslessly through
``preset_to_workspace``. Reads its workspace dir from the
``WORKSPACE_CONFIG`` module global, which setup.py wires from the
shared ``@workspace_config`` resource at load time.

The actual command-execution logic lives in
``examples.coder.tools._run`` so this file stays a thin adapter and
all tools share one battle-tested implementation.
"""

from __future__ import annotations

import re
from pathlib import Path

from examples.coder.tools import _is_path_inside, _run

# Set by setup.py at workspace load time.
WORKSPACE_CONFIG = None


def execute(*, command: str) -> dict:
    workspace = WORKSPACE_CONFIG.path if WORKSPACE_CONFIG is not None else "."
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
