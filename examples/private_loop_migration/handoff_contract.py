"""The host-owned contract: one case, one collector, one required grader.

The case carries its seed files and its expected result as data. Expected
data stays out of the task the agent receives, so the grader cannot be
satisfied by a model that read the answer from its own prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from looplet import EvalCase, EvalContext, EvalHook, EvalResult, eval_mark

from .handoff_tools import HANDOFF_FILE, INCIDENTS_FILE

SHIFT_LOG = {
    "shift": "2026-07-14-night",
    "incidents": [
        {"id": "INC-101", "summary": "checkout latency", "status": "resolved"},
        {"id": "INC-102", "summary": "stuck payout batch", "status": "open"},
        {"id": "INC-103", "summary": "search replica lag", "status": "monitoring"},
        {"id": "INC-104", "summary": "expired staging cert", "status": "resolved"},
    ],
}

CASE = EvalCase(
    id="night_shift_handoff",
    task={
        "goal": "Read the shift log, then write the handoff file for the day-shift owner.",
        "files": {INCIDENTS_FILE: json.dumps(SHIFT_LOG, indent=2) + "\n"},
    },
    expected={"open_count": 2, "open_ids": ["INC-102", "INC-103"]},
    marks=["smoke", "regression"],
    notes="Derived from a run that handed resolved incidents to the next owner as open work.",
)


def live_task() -> dict[str, Any]:
    """The task the agent sees: no seed files and no expected result."""
    return {key: value for key, value in CASE.task.items() if key != "files"}


def make_handoff_collector(workspace: str | Path) -> Callable[[Any], dict[str, Any]]:
    """Build a collector that reads the workspace the run acted on."""
    root = Path(workspace)

    def collect_handoff(state: Any) -> dict[str, Any]:
        """Report the observed handoff file, whatever route produced it."""
        # `state` carries the step list. This collector never reads it:
        # the verdict has to survive a model that takes another route to
        # the same correct outcome.
        path = root / HANDOFF_FILE
        if not path.is_file():
            return {"handoff_exists": False}
        try:
            handoff = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"handoff_exists": True, "handoff_error": str(exc)}
        return {
            "handoff_exists": True,
            "observed_open_count": handoff.get("open_count"),
            "observed_open_ids": handoff.get("open_ids"),
        }

    return collect_handoff


@eval_mark("required")
def eval_open_incidents_are_correct(ctx: EvalContext):
    """Compare the collected handoff with the grader-only expected result."""
    expected = ctx.task.get("expected", {})
    return ctx.artifacts.get("observed_open_count") == expected.get(
        "open_count"
    ) and ctx.artifacts.get("observed_open_ids") == expected.get("open_ids")


def build_eval_hook(workspace: str | Path) -> EvalHook:
    """Wire the collector and the required grader for one workspace."""
    return EvalHook(
        evaluators=[eval_open_incidents_are_correct],
        collectors=[make_handoff_collector(workspace)],
        expected=CASE.expected,
    )


def required_score(results: list[EvalResult]) -> float:
    """Score of the required grader, or raise when it did not run."""
    name = eval_open_incidents_are_correct.__name__
    for result in results:
        if result.name == name:
            return float(result.score if result.score is not None else 0.0)
    raise RuntimeError(f"required grader {name} did not run")
