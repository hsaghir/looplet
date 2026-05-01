"""Composition layer for the coder example.

The reusable building block. Both the library entrypoint
(:mod:`examples.coder.agent`) and the runnable bundle
(``examples/coder/skill/looplet.py``) delegate to the helpers below
to construct identical hook stacks, memory sources, and evaluators.
That guarantees library/bundle parity is *structural* rather
than coincidental — the two surfaces cannot drift without someone
deliberately editing both call sites.

Defaults follow the "steer, don't restrict" principle (see
``docs/evals.md``):

* ``TestGuardHook`` runs in observe-only mode by default.
* ``StagnationHook`` uses :func:`looplet.stagnation.result_size_fingerprint`
  with a lenient threshold so legitimate retries (running pytest
  multiple times across edits) don't trigger a nudge.
* ``PerToolLimitHook`` exposes one high cap as a runaway safety
  net, not as a per-tool process budget.
* :func:`build_eval_hook` ships a ``collect_test_results``
  collector that re-runs pytest after the loop and surfaces the
  outcome via ``ctx.artifacts`` — outcome-grading rather than
  trajectory-grading.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from coder_lib_hooks import FileCacheHook, LinterHook, StaleFileHook, TestGuardHook
from coder_lib_tools import FileCache

from looplet import (
    CallableMemorySource,
    EvalContext,
    EvalHook,
    EvalResult,
    StaticMemorySource,
    StreamingHook,
)
from looplet.limits import PerToolLimitHook
from looplet.stagnation import StagnationHook, result_size_fingerprint
from looplet.streaming import CallbackEmitter

__all__ = [
    "SYSTEM_PROMPT",
    "scripted_responses",
    "build_default_hooks",
    "build_default_memory_sources",
    "make_test_collector",
    "build_eval_hook",
]


SYSTEM_PROMPT = """\
You are an expert software engineer. You solve tasks by understanding \
the codebase, planning carefully, making precise changes, and verifying \
with tests. You never guess — you read first, then act.

## Workflow
1. EXPLORE: list_dir to see structure. glob/grep to find relevant files.
2. READ: read_file on files you need to modify. Understand patterns and conventions.
3. PLAN: think() to plan approach. Break complex tasks into steps.
4. IMPLEMENT: edit_file for existing files, write_file for new files. One file at a time.
5. TEST: bash to run tests after EVERY change. Read failures. Fix and re-run.
6. DONE: done() with summary only after tests pass.

## Tool rules
- ALWAYS read_file before edit_file. Never edit blind.
- edit_file: copy-paste old_string from read_file output exactly. Include 3+ context lines.
- If edit fails "not found": read the hint lines, re-read file at those lines, retry with exact text.
- If edit fails "multiple matches": add more surrounding lines for uniqueness.
- write_file: NEW files only. Never overwrite files you should edit.
- bash: use relative paths. pytest -xvs for tests (stop on first failure).
- For bugs: write a failing test FIRST, then fix the code.

## Code quality
- Follow existing project style and conventions.
- Type hints on function signatures. Docstrings on public functions.
- Minimal changes. Don't refactor unrelated code.
- If stuck after 3 attempts: think() to reconsider approach.
"""


# ── Project context discovery ──────────────────────────────────────


def _discover_instructions(workspace: str) -> str:
    candidates = [
        "CLAUDE.md",
        ".claude.md",
        "AGENTS.md",
        ".cursorrules",
        "CODING_GUIDELINES.md",
        ".github/copilot-instructions.md",
    ]
    parts = []
    for name in candidates:
        p = Path(workspace) / name
        if p.exists():
            parts.append(f"## From {name}\n{p.read_text()[:4000]}")
    return "\n\n".join(parts)


def _project_context(workspace: str) -> str:
    parts = []
    try:
        branch = subprocess.run(
            ["git", "-C", workspace, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if branch:
            parts.append(f"branch={branch}")
    except Exception:
        pass
    for n in ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"]:
        if (Path(workspace) / n).exists():
            parts.append(n)
    return " | ".join(parts) if parts else "unknown"


# ── Scripted demo responses (used by --scripted and tests) ─────────


def _tool_call(tool_name: str, args: dict, reasoning: str) -> str:
    return json.dumps({"tool": tool_name, "args": args, "reasoning": reasoning})


def scripted_responses() -> list[str]:
    return [
        _tool_call("list_dir", {"path": ".", "depth": 1}, "inspect the workspace"),
        _tool_call(
            "write_file",
            {
                "file_path": "math_utils.py",
                "content": "def add(left: int, right: int) -> int:\n    return left + right\n",
            },
            "create the implementation",
        ),
        _tool_call(
            "write_file",
            {
                "file_path": "test_math_utils.py",
                "content": "from math_utils import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
            },
            "create a regression test",
        ),
        _tool_call("bash", {"command": "python -m pytest -q"}, "run the tests"),
        _tool_call(
            "done",
            {"summary": "Created math_utils.add with tests."},
            "finish after tests pass",
        ),
    ]


# ── Hook / memory / eval composition ───────────────────────────────


def build_default_hooks(
    workspace: str,
    file_cache: FileCache,
    *,
    require_tests: bool = True,
    test_strict: bool = False,
    stagnation_threshold: int = 6,
    per_tool_limit: int = 100,
    events: list | None = None,
) -> list:
    """Compose the canonical coder hook stack.

    The defaults follow the "steer, don't restrict" principle
    (see ``docs/evals.md``):

    * ``TestGuardHook`` is registered in **observe-only** mode
      (``test_strict=False``); failures inject a briefing nudge but
      ``done()`` is never blocked. Outcome is graded post-run via
      :func:`build_eval_hook`.
    * ``StagnationHook`` uses :func:`result_size_fingerprint` and a
      lenient threshold (6) — it ignores legitimate retries that
      change the world (e.g. running the test suite three times
      across edits) and only fires on truly unproductive loops.
    * ``PerToolLimitHook`` exposes a single high cap (``100``) as a
      runaway safety-net rather than per-tool process budgets.

    Set ``test_strict=True`` to recover the legacy hard-block on
    ``done()``; lower limits or change the fingerprint to harden
    further only when you have a specific reason to.
    """
    hooks: list = []
    if require_tests:
        hooks.append(TestGuardHook(strict=test_strict))
    hooks.append(FileCacheHook(file_cache))
    hooks.append(StaleFileHook(file_cache))
    hooks.append(LinterHook(workspace))
    hooks.append(
        StagnationHook(
            fingerprint=result_size_fingerprint,
            threshold=stagnation_threshold,
            nudge="[stagnation] Re-read the file, try a different approach, or think().",
        )
    )
    hooks.append(PerToolLimitHook(default_limit=per_tool_limit))
    if events is not None:
        hooks.append(StreamingHook(CallbackEmitter(events.append)))
    return hooks


def build_default_memory_sources(workspace: str, max_steps: int) -> list:
    """Project-context memory sources used by both library and bundle."""
    instructions = _discover_instructions(workspace)
    project_ctx = _project_context(workspace)
    sources: list = []
    if instructions:
        sources.append(StaticMemorySource(instructions))
    sources.append(
        CallableMemorySource(
            lambda state: f"[{project_ctx}] step {getattr(state, 'step_count', 0)}/{max_steps}"
        )
    )
    return sources


def make_test_collector(workspace: str, *, timeout_s: int = 60):
    """Return a collector that re-runs the project's test suite.

    Outcome-grounded eval data — the collector runs *after* the agent
    finishes and surfaces the result via ``ctx.artifacts``. Skipped
    silently when no Python test runner is detected.
    """

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

    return collect_test_results


def build_eval_hook(workspace: str) -> EvalHook:
    """Outcome-grounded post-run evaluation hook.

    Demonstrates the "trajectory-blind" eval pattern from
    ``docs/evals.md``: the collector re-runs the test suite, the
    evaluators read ``ctx.artifacts`` and ``ctx.completed`` rather
    than grepping the trajectory.
    """

    def eval_tests_passed(ctx: EvalContext):
        if "tests_passing" not in ctx.artifacts:
            return EvalResult(
                name="eval_tests_passed",
                label="skipped",
                explanation=(
                    "no Python project (pyproject.toml/setup.py) detected "
                    "in workspace; collector cannot re-run tests"
                ),
            )
        return bool(ctx.artifacts["tests_passing"])

    def eval_completed(ctx: EvalContext):
        return ctx.completed

    return EvalHook(
        evaluators=[eval_tests_passed, eval_completed],
        collectors=[make_test_collector(workspace)],
    )
