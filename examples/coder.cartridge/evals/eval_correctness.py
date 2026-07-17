"""Graders for the coder cartridge - discovered by ``load_cartridge_evals``
(functions named ``eval_*`` in ``evals/eval_*.py``).

These ship *with the coder agent version*: a ``git diff`` of two coder
cartridges shows the grading change next to the prompt/tool change it
covers. They are case-AGNOSTIC predicates run N×M over every shipped
case (``evals/cases/*.json``), so the same three graders apply whether
the task is "implement a Stack" or "fix an off-by-one bug".

Three grader *kinds* are shown deliberately:
  * outcome-grounded  (eval_tests_pass) - reads ``ctx.artifacts``
  * trajectory        (eval_completed) - reads how the run ended
  * metric            (eval_step_count) - a number, NOT a pass/fail
"""

from __future__ import annotations

from looplet import EvalContext, EvalResult, eval_mark


@eval_mark("required")
def eval_tests_pass(ctx: EvalContext):
    """Did the shipped test suite actually pass after the run?

    Outcome-grounded: reads ``ctx.artifacts["tests_passing"]`` populated
    by the ``collect_*`` collector, NOT the trajectory - so it survives
    the agent changing its workflow. Skips loudly if no collector ran.
    """
    if "tests_passing" not in ctx.artifacts:
        return EvalResult(
            name="eval_tests_pass",
            label="skipped",
            explanation="no collector populated tests_passing",
        )
    return bool(ctx.artifacts["tests_passing"])


@eval_mark("required")
def eval_completed(ctx: EvalContext):
    """Did the agent finish by calling ``done()`` (not stall / get stopped)?"""
    return ctx.completed


def eval_step_count(ctx: EvalContext):
    """Efficiency metric - surfaced as data, not scored as pass/fail."""
    return {"steps": float(len(ctx.steps))}
