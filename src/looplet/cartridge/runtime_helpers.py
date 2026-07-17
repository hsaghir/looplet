"""Resolve the *target project root* from a cartridge runtime dict.

The "project root" is the directory the agent operates on - typically
the user's repository - distinct from the cartridge directory itself.
Cartridges that need it (coder, dep_doctor, git_detective, ...) used
to require the host to pass ``runtime["workspace"]`` explicitly. That
is fragile (host-specific) and overloads the term "workspace" which
this codebase reserves for the legacy cartridge name.

Resolution order (first wins):

1. ``runtime["project_root"]`` - the canonical key.
2. ``runtime["workspace"]`` - back-compat with pre-rename callers.
3. ``$LOOPLET_PROJECT_ROOT`` env var.
4. ``git rev-parse --show-toplevel`` (if cwd is inside a git repo).
5. ``Path.cwd()``.

Resource builders and tools should call :func:`resolve_project_root`
instead of reading any specific runtime key directly. That way a host
that simply runs the agent from inside the target repo doesn't have
to pass any runtime kwargs at all.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def resolve_project_root(runtime: dict[str, Any] | None = None) -> str:
    """Return the absolute path to the target project root.

    See module docstring for resolution order.
    """
    runtime = runtime or {}

    # 1 + 2: explicit runtime keys (canonical first, then legacy).
    for key in ("project_root", "workspace"):
        val = runtime.get(key)
        if val:
            return str(Path(val).expanduser().resolve())

    # 3: env override.
    env_val = os.environ.get("LOOPLET_PROJECT_ROOT")
    if env_val:
        return str(Path(env_val).expanduser().resolve())

    # 4: git toplevel - quiet, swallow any error and fall through.
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return str(Path(out.stdout.strip()).resolve())
    except (OSError, subprocess.SubprocessError):
        pass

    # 5: last resort.
    return str(Path.cwd().resolve())
