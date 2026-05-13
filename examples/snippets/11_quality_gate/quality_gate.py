"""Quality-gate hook — runs checks at ``done()`` time, not after every write.

This snippet demonstrates the looplet alternative to a built-in
"mid-edit linting" feature. We *deliberately* do not run linters or
type-checkers after every ``write`` call: the file is in flight, the
checker will report transient errors, and the agent will burn budget
chasing them.

Instead, write a tiny ``check_done`` hook that runs the gate exactly
once — when the agent thinks it's done. If the gate fails, the hook
blocks ``done()`` with an actionable message; the agent gets one more
turn to fix it.

Same pattern works for: pytest, ruff, mypy/pyright, custom
acceptance criteria, schema validation, etc. The point is the *gate
runs at the boundary*, not after every step.
"""

from __future__ import annotations

import subprocess
from typing import Any

from looplet import Block, HookDecision


class QualityGate:
    """Runs ``cmd`` exactly when the agent calls ``done``.

    Pass any shell command. Non-zero exit blocks ``done()`` and shows
    the agent the combined stdout/stderr so it knows what to fix.
    """

    def __init__(self, *, cmd: str, cwd: str | None = None) -> None:
        self._cmd = cmd
        self._cwd = cwd

    def check_done(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> HookDecision | None:
        result = subprocess.run(  # noqa: S602 — opt-in user-supplied command
            self._cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self._cwd,
            check=False,
        )
        if result.returncode == 0:
            return None  # gate passes — allow done()
        return Block(
            f"Quality gate failed (exit {result.returncode}). "
            f"Fix the issues below, then call done() again.\n\n"
            f"$ {self._cmd}\n{result.stdout}{result.stderr}".strip()
        )


# Usage:
#
#     from looplet import composable_loop, LoopConfig, DefaultState
#
#     hook = QualityGate(cmd="uv run pytest -q && uv run ruff check .")
#     for step in composable_loop(
#         llm=my_llm, tools=my_tools, state=DefaultState(max_steps=20),
#         config=LoopConfig(max_steps=20), hooks=[hook],
#         task={"goal": "Implement feature X with tests"},
#     ):
#         print(step.pretty())
#
# To ship this as a cartridge: drop this file at
# ``hooks/00_QualityGate/hook.py`` and add a sibling
# ``hooks/00_QualityGate/config.yaml`` with::
#
#     class_name: QualityGate
#     kwargs:
#       cmd: "uv run pytest -q && uv run ruff check ."
#
# That's it — same hook, packaged as files.
