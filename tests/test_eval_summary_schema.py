"""Stable state and manifest contract for `looplet eval run --json`."""

from __future__ import annotations

from pathlib import Path

from looplet import EvalCase, EvalContext, EvalResult, EvalRunRecord, eval_mark
from looplet.evals import (
    _EVAL_RESULT_STATES,
    _EVAL_SUMMARY_SCHEMA,
    _EVAL_SUMMARY_VERSION,
    _build_eval_summary,
    _eval_grader_manifest,
    _eval_summary_result,
)


def _result(state: str) -> tuple[EvalResult | None, dict[str, object]]:
    cases: dict[str, tuple[EvalResult | None, dict[str, object]]] = {
        "pass": (EvalResult(name="grader", score=1.0), {"score": 1.0}),
        "explicit_fail": (
            EvalResult(name="grader", score=0.0, label="fail"),
            {"score": 0.0, "label": "fail"},
        ),
        "threshold_fail": (EvalResult(name="grader", score=0.4), {"score": 0.4}),
        "skipped": (
            EvalResult(name="grader", label="skipped", explanation="requires judge"),
            {"label": "skipped", "explanation": "requires judge"},
        ),
        "missing": (None, {}),
        "collector_error": (
            EvalResult(name="collector:files", label="error", explanation="OSError: denied"),
            {"label": "error", "explanation": "OSError: denied", "error": "OSError: denied"},
        ),
        "grader_error": (
            EvalResult(name="grader", label="error", explanation="ValueError: bad"),
            {"label": "error", "explanation": "ValueError: bad", "error": "ValueError: bad"},
        ),
        "metric_only": (
            EvalResult(name="grader", metrics={"latency_ms": 12.0}),
            {"metrics": {"latency_ms": 12.0}},
        ),
    }
    return cases[state]


def test_every_eval_summary_state_matches_its_golden_record() -> None:
    assert _EVAL_RESULT_STATES == {
        "pass",
        "explicit_fail",
        "threshold_fail",
        "skipped",
        "missing",
        "collector_error",
        "grader_error",
        "metric_only",
    }
    for state in sorted(_EVAL_RESULT_STATES):
        result, extra = _result(state)
        collector = state == "collector_error"
        name = "collector:files" if collector else "grader"
        actual = _eval_summary_result(
            result,
            name=name,
            marks=set(),
            threshold=0.5,
            collector=collector,
        )
        assert actual == {
            "name": name,
            "marks": [],
            "required": False,
            "required_status": "not_required",
            "state": state,
            **extra,
        }


def test_summary_uses_the_pre_run_manifest_to_report_a_missing_grader() -> None:
    @eval_mark("accuracy")
    def eval_present(ctx):
        return True

    @eval_mark("required", "release")
    def eval_required_missing(ctx):
        return True

    record = EvalRunRecord(
        case=EvalCase(id="case-1", task={}, marks=["smoke"]),
        context=EvalContext(steps=[], stop_reason="done"),
        results=[EvalResult(name="eval_present", score=1.0, label="pass")],
        directory=Path("run"),
    )
    manifest = _eval_grader_manifest([eval_present, eval_required_missing])
    getattr(eval_required_missing, "_eval_marks").clear()

    report = _build_eval_summary(
        records=[record],
        grader_manifest=manifest,
        threshold=0.5,
        output_dir=None,
    )

    assert report["schema"] == _EVAL_SUMMARY_SCHEMA
    assert report["version"] == _EVAL_SUMMARY_VERSION
    assert report["state"] == "fail"
    assert report["passed"] is False
    assert report["cases"][0]["marks"] == ["smoke"]
    assert report["grader_manifest"] == [
        {"name": "eval_present", "marks": ["accuracy"], "required": False},
        {
            "name": "eval_required_missing",
            "marks": ["release", "required"],
            "required": True,
        },
    ]
    missing = next(item for item in report["cases"][0]["results"] if item["state"] == "missing")
    assert missing == {
        "name": "eval_required_missing",
        "marks": ["release", "required"],
        "required": True,
        "required_status": "failed",
        "state": "missing",
    }
    assert report["integrity_failures"] == [
        "case-1/eval_required_missing: grader missing from run result"
    ]


def test_required_metric_and_skipped_results_fail_without_changing_their_state() -> None:
    for result in (
        EvalResult(name="grader", metrics={"latency_ms": 12.0}),
        EvalResult(name="grader", label="skipped"),
    ):
        item = _eval_summary_result(
            result,
            name="grader",
            marks={"required"},
            threshold=0.0,
        )
        assert item["state"] in {"metric_only", "skipped"}
        assert item["required_status"] == "failed"


def test_required_score_can_satisfy_the_required_gate_but_fail_cli_threshold() -> None:
    item = _eval_summary_result(
        EvalResult(name="grader", score=0.7),
        name="grader",
        marks={"required"},
        threshold=0.9,
    )
    assert item["state"] == "threshold_fail"
    assert item["required_status"] == "satisfied"
