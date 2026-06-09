"""Outcome collector for the coder cartridge — discovered by
``load_cartridge_evals`` (functions named ``collect_*`` in
``evals/collect_*.py``).

A collector runs once at end-of-loop and returns a dict merged into
``EvalContext.artifacts``, so graders can grade *what changed in the
world* (here: does the test suite pass?) instead of grepping the
trajectory. ``discover_collectors`` binds the ``runtime`` kwarg at
discovery time, so this re-runs the suite in the agent's own project
root without hard-coding any path — it relocates with the cartridge.
"""

from __future__ import annotations

import subprocess

from looplet.cartridge.runtime_helpers import resolve_project_root


def collect_tests(state, runtime) -> dict:
    """Re-run pytest in the agent's project root.

    Uses the same ``resolve_project_root`` the coder's tools use, so the
    collector grades exactly the directory the agent edited.
    """
    workspace = resolve_project_root(runtime)
    try:
        proc = subprocess.run(
            ["python", "-m", "pytest", "-q", "--tb=no"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=int((runtime or {}).get("test_timeout_s", 90)),
            check=False,
        )
    except FileNotFoundError:
        return {}
    except subprocess.TimeoutExpired:
        return {"tests_passing": False, "test_timeout": True}
    return {"tests_passing": proc.returncode == 0, "test_exit_code": proc.returncode}
