"""Project-context CallableMemorySource — inline so the closure's
``__module__`` becomes ``_chw_resource_project_memory`` and the
workspace round-trip can copy this builder verbatim instead of trying
to re-import a lambda from coder_lib_wiring.

Reads ``runtime['workspace']`` for the project root and
``runtime['max_steps']`` for the budget hint surfaced to the agent.
Falls back to ``"."`` and ``20`` respectively when unset so the
builder still works in standalone test loads.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from looplet import CallableMemorySource


def _project_context(workspace: str) -> str:
    """Lightweight project signature for the per-step memory line."""
    parts: list[str] = []
    try:
        branch = subprocess.run(
            ["git", "-C", workspace, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if branch:
            parts.append(f"branch={branch}")
    except Exception:  # noqa: BLE001
        pass
    for n in ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"]:
        if (Path(workspace) / n).exists():
            parts.append(n)
    return " ".join(parts) or "no project files"


def build(runtime=None) -> CallableMemorySource:
    runtime = runtime or {}
    workspace = str(runtime.get("workspace", "."))
    max_steps = int(runtime.get("max_steps", 20))
    project_ctx = _project_context(workspace)

    # The lambda closes over project_ctx + max_steps. Because this
    # function lives in ``resources/project_memory.py``, the closure's
    # ``__module__`` is ``_chw_resource_project_memory`` on workspace
    # load — so the round-trip writer recognises it and copies THIS
    # file verbatim instead of warning about a non-importable lambda.
    return CallableMemorySource(
        lambda state: f"[{project_ctx}] step {getattr(state, 'step_count', 0)}/{max_steps}"
    )
