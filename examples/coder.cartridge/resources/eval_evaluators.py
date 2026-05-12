"""Evaluators for the coder workspace's EvalHook.

Re-uses the looplet.examples coder reference's ``build_eval_hook`` evaluators so the
workspace produces identical scores. The list lands in
``hooks/07_EvalHook/config.yaml`` via the ``@eval_evaluators`` ref.
"""

from looplet import EvalContext, EvalResult


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


def build(runtime=None):
    return [eval_tests_passed, eval_completed]
