"""Observability hooks for the coder example.

These four hooks are domain-specific (they know what a Python test
runner or a ruff lint failure looks like) but framework-generic:
each one implements the standard :class:`looplet.loop.LoopHook`
protocol and returns ``HookDecision``/``InjectContext`` values, so
they compose with the rest of the looplet ecosystem.

Design intent — *steer, don't restrict*:

* :class:`TestGuardHook` defaults to **observe-only** mode.
  Failures emit a briefing nudge; ``done()`` is never blocked.
  Outcome verification happens after the run via the eval hook
  built in :func:`examples.coder.wiring.build_eval_hook`. Pass
  ``strict=True`` to opt into the legacy hard-block.
* :class:`StaleFileHook` injects a re-read nudge when bash modifies
  a previously-read file (it does not block edits).
* :class:`LinterHook` runs ``ruff check`` after Python file edits
  and surfaces diagnostics; it does not refuse to dispatch.
* :class:`FileCacheHook` re-injects the file cache into the
  briefing after compaction kicks in.

See ``docs/evals.md`` ("Trajectory-blind evals") for the rationale
and ``docs/hooks.md`` for the broader contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from examples.coder.tools import FileCache

from looplet.hook_decision import HookDecision, InjectContext

__all__ = [
    "TestGuardHook",
    "FileCacheHook",
    "StaleFileHook",
    "LinterHook",
]


class TestGuardHook:
    """Watches for test runs and nudges the model on failures.

    By default (``strict=False``), the hook only **observes**:
    it injects a briefing nudge after each test invocation
    ("Tests passed" / "Tests FAILED — read the traceback…") and
    surfaces the outcome to evaluators via ``state``. It does NOT
    block ``done()``: the model may have a legitimate reason to
    finish without running tests (docs change, no test runner,
    flaky suite). Outcome-grading happens after the run via an
    ``EvalHook`` collector that re-checks the test suite.

    Pass ``strict=True`` to recover the old hard-block behavior
    (refuse ``done()`` if files were written but tests didn't pass).
    Use sparingly — see ``docs/evals.md`` "Trajectory-blind evals"
    for the rationale.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._tests_passed = False
        self._files_written: set[str] = set()
        self._strict = strict

    @property
    def tests_passed(self) -> bool:
        return self._tests_passed

    @property
    def files_written(self) -> set[str]:
        return set(self._files_written)

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool == "bash":
            cmd = tool_call.args.get("command", "")
            data = tool_result.data or {}
            if any(
                t in cmd
                for t in ["pytest", "python -m pytest", "npm test", "cargo test", "go test"]
            ):
                self._tests_passed = data.get("exit_code", 1) == 0
                if not self._tests_passed:
                    return InjectContext(
                        "⚠ Tests FAILED. Read the traceback. Find the exact file:line. Read that code. Fix the issue. Re-run tests."
                    )
                return InjectContext("✓ Tests passed.")
        if tool_call.tool in ("write_file", "edit_file"):
            self._files_written.add(tool_call.args.get("file_path", ""))
        return None

    def check_done(self, state, session_log, context, step_num):
        if not self._strict:
            # Observe-only: surface a nudge but allow `done()` to proceed.
            # Outcome is graded post-run via the EvalHook collector.
            if not self._tests_passed and self._files_written:
                return InjectContext(
                    "⚠ Finishing without a passing test run. "
                    "An eval will re-check the suite after the run."
                )
            return None
        # Strict mode: hard block (legacy / opt-in).
        if not self._tests_passed and self._files_written:
            return HookDecision(block="Run tests first. If no tests exist, create them.")
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class StaleFileHook:
    """Detects when bash commands modify files the model previously read.

    Mirrors Claude Code's staleReadFileStateHint: after each bash step,
    checks cached file mtimes and warns the model to re-read before editing.
    """

    def __init__(self, cache: FileCache):
        self._cache = cache

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool != "bash":
            return None
        # Check for stale files even on failed commands — a partial
        # build or interrupted write can still modify files.
        stale = self._cache.stale_files()
        if stale:
            return InjectContext(
                f"⚠ Stale files: {', '.join(stale)} were modified by this command. "
                f"Re-read them with read_file before editing."
            )
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class LinterHook:
    """Runs ruff check after Python file edits, injects diagnostics.

    Mirrors Claude Code's automatic LSP error reporting after file edits.
    Only runs when ruff is available in the workspace.
    """

    def __init__(self, workspace: str):
        self._workspace = workspace
        self._ruff_available: bool | None = None
        self._ruff_cmd: list[str] | None = None

    def _find_ruff(self) -> list[str] | None:
        workspace_path = Path(self._workspace)
        for candidate in (
            workspace_path / ".venv" / "bin" / "ruff",
            workspace_path / "venv" / "bin" / "ruff",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return [str(candidate)]
        path_ruff = shutil.which("ruff")
        if path_ruff:
            return [path_ruff]
        uv = shutil.which("uv")
        if uv and (
            (workspace_path / "pyproject.toml").exists() or (workspace_path / "uv.lock").exists()
        ):
            return [uv, "run", "ruff"]
        return None

    def _check_ruff(self) -> bool:
        if self._ruff_available is None:
            self._ruff_cmd = self._find_ruff()
            if self._ruff_cmd is None:
                self._ruff_available = False
                return False
            try:
                r = subprocess.run(self._ruff_cmd + ["--version"], capture_output=True, timeout=5)
                self._ruff_available = r.returncode == 0
            except Exception:
                self._ruff_available = False
        return self._ruff_available

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool not in ("edit_file", "write_file"):
            return None
        file_path = tool_call.args.get("file_path", "")
        if not file_path.endswith(".py"):
            return None
        if tool_result.error:
            return None
        if not self._check_ruff():
            return None
        try:
            r = subprocess.run(
                (self._ruff_cmd or ["ruff"]) + ["check", "--no-fix", file_path],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self._workspace,
            )
            if r.returncode != 0 and r.stdout.strip():
                lines = r.stdout.strip().splitlines()
                if len(lines) > 10:
                    lines = lines[:10] + [f"... and {len(lines) - 10} more issues"]
                return InjectContext(f"⚠ Lint issues in {file_path}:\n" + "\n".join(lines))
        except Exception:
            pass
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class FileCacheHook:
    def __init__(self, cache: FileCache):
        self._cache = cache

    def pre_prompt(self, state, session_log, context, step_num):
        if step_num > 3:
            return self._cache.render() or None
        return None

    def should_stop(self, state, step_num, new_entities):
        return False
