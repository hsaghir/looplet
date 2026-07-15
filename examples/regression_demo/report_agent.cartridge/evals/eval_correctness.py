"""Behavioral contract for the report-agent demo."""

from __future__ import annotations

from looplet import EvalContext, eval_mark


@eval_mark("required")
def eval_profit_is_correct(ctx: EvalContext):
    expected = ctx.task.get("expected", {}).get("profit")
    return ctx.artifacts.get("observed_profit") == expected


@eval_mark("required")
def eval_completed(ctx: EvalContext):
    return ctx.completed
