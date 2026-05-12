"""Outcome-grounded test-runner collector for the coder EvalHook.

Inlining the closure here (rather than delegating to a library
function) is what makes the workspace round-trip lossless: the
returned ``collect_test_results`` closure has
``__module__ == '_chw_resource_eval_collectors'`` on workspace load,
so the snapshot writer's ``_origin_chw_resource_file`` pre-pass
copies THIS file verbatim instead of falling back to a None-stub.

Reads the project root via
:func:`looplet.cartridge.runtime_helpers.resolve_project_root` and an
optional ``runtime['test_timeout_s']`` (default 60).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from looplet.cartridge.runtime_helpers import resolve_project_root


def build(runtime=None):
    runtime = runtime or {}
    workspace = resolve_project_root(runtime)
    timeout_s = int(runtime.get("test_timeout_s", 60))

    def collect_test_results(state) -> dict:
        ws = Path(workspace)
        if not (ws / "pyproject.toml").exists() and not (ws / "setup.py").exists():
            return {}
        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "-q", "--tb=no"],
                cwd=str(ws),
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return {}
        except subprocess.TimeoutExpired:
            return {"tests_passing": False, "test_runner": "pytest", "test_timeout": True}
        return {
            "tests_passing": proc.returncode == 0,
            "test_runner": "pytest",
            "test_exit_code": proc.returncode,
        }

    return [collect_test_results]
