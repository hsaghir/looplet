"""Stable state and manifest contract for `looplet eval run --json`."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from looplet import EvalCase, EvalContext, EvalResult, EvalRunRecord, eval_assess, eval_mark
from looplet.evals import (
    _EVAL_RESULT_STATES,
    _EVAL_SUMMARY_SCHEMA,
    _EVAL_SUMMARY_VERSION,
    EvalReport,
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


def test_assessment_rejects_empty_cases_and_grader_inventory() -> None:
    report = EvalReport.from_records([], grader_manifest=[])
    assert report.version == 2
    assert not report.passed and not report.valid
    assert report.integrity_failures == ["No evaluation cases", "No graders selected"]
    with pytest.raises(AssertionError, match="No graders selected"):
        report.assert_passed()


@pytest.mark.parametrize("passed", [False, True])
@pytest.mark.parametrize("version", [1, 2])
def test_assessment_save_is_separate_private_and_exclusive(tmp_path, passed, version) -> None:
    def eval_quality(context):
        return passed

    report = eval_assess(
        [eval_quality],
        EvalContext(steps=[], task={"expected": {"secret": "grader-only"}}),
        version=version,
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    marker = capture / "evidence.txt"
    marker.write_text("original evidence\n")
    target = tmp_path / "assessments" / "strict.json"
    assert report.save(target, evidence={"run": capture}) == target
    payload = json.loads(target.read_text())
    assert payload["schema"] == "looplet.eval-assessment" and payload["version"] == 1
    assert payload["assessed_at"].endswith("Z")
    assert payload["assessment"] == report.to_dict()
    assert payload["assessment"]["passed"] is passed
    assert payload["evidence"] == {"run": "../capture"}
    assert "grader-only" not in target.read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        report.save(target)
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="outside saved evidence"):
        report.save(capture / "new.json", evidence={"run": capture})
    assert list(capture.iterdir()) == [marker]
    assert marker.read_text() == "original evidence\n"


def test_assessment_separates_failed_behavior_from_invalid_evaluation() -> None:
    record = EvalRunRecord(
        case=EvalCase(id="case-1"),
        context=EvalContext(steps=[], stop_reason="done"),
        results=[EvalResult(name="grader", score=0.0, label="fail")],
        directory=Path("run"),
    )
    inventory = [{"name": "grader", "marks": ["required"], "required": True}]
    report = EvalReport.from_records([record], grader_manifest=inventory)
    assert report.valid and not report.passed
    assert report.cases[0]["completed"]
    record = record._replace(results=[EvalResult(name="grader", label="error")])
    invalid = EvalReport.from_records([record], grader_manifest=inventory)
    assert not invalid.valid and not invalid.passed


def test_assessment_preserves_v1_and_returns_independent_snapshots() -> None:
    report = EvalReport.from_records([], grader_manifest=[], version=1)
    assert report.to_dict() == _build_eval_summary(
        records=[], grader_manifest=[], threshold=0.5, output_dir=None
    )
    snapshot = report.to_dict()
    snapshot["cases"].append({"id": "foreign"})
    assert report.cases == []


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_assessment_validates_threshold_before_running_grader(threshold):
    def eval_unused(ctx):
        raise AssertionError("grader must not run")

    with pytest.raises(ValueError, match="threshold"):
        eval_assess([eval_unused], EvalContext(steps=[]), threshold=threshold)


def test_assessment_requires_recorded_or_explicit_inventory():
    record = EvalRunRecord(
        None, EvalContext(steps=[]), [EvalResult(name="grader", score=1.0)], Path("run")
    )
    report = EvalReport.from_records([record])
    assert not report.valid and not report.passed
    with pytest.raises(ValueError, match="version"):
        EvalReport.from_records([record], grader_manifest=[], version=3)


def test_assessment_rejects_duplicate_results_and_keeps_result_snapshot():
    result = EvalResult(name="grader", score=1.0, metrics={"latency": 12.0})
    record = EvalRunRecord(None, EvalContext(steps=[]), [result], Path("run"))
    inventory = [{"name": "grader", "marks": [], "required": False}]
    report = EvalReport.from_records([record], grader_manifest=inventory)
    result.metrics["latency"] = 999.0
    inventory[0]["marks"].append("required")
    assert report.cases[0]["results"][0]["metrics"] == {"latency": 12.0}
    assert report.grader_manifest[0]["marks"] == []
    duplicate = EvalReport.from_records(
        [record._replace(results=[result, result])], grader_manifest=report.grader_manifest
    )
    assert not duplicate.valid and not duplicate.passed


def test_assessment_filter_cannot_remove_a_required_check():
    @eval_mark("required", "release")
    def eval_gate(ctx):
        return True

    @eval_mark("measurement")
    def eval_other(ctx):
        return True

    report = eval_assess([eval_gate, eval_other], EvalContext(steps=[]), include=["measurement"])
    assert not report.valid and not report.passed
    assert report.cases[0]["results"][0]["state"] == "missing"


def test_measurement_only_case_is_invalid_in_v2_but_retains_v1_state():
    record = EvalRunRecord(
        None,
        EvalContext(steps=[], stop_reason="done"),
        [EvalResult(name="measurement", metrics={"cost": 1.0})],
        Path("run"),
    )
    inventory = [{"name": "measurement", "marks": [], "required": False}]
    report = EvalReport.from_records([record], grader_manifest=inventory)
    assert not report.valid and not report.passed
    assert report.cases[0]["state"] == "fail"
    assert report.cases[0]["valid"] is False
    legacy = EvalReport.from_records([record], grader_manifest=inventory, version=1)
    assert legacy.passed
    assert legacy.cases[0]["state"] == "pass"
    assert "valid" not in legacy.cases[0]


@pytest.mark.parametrize("version", [1, 2])
def test_assessment_restores_observation_failures_omitted_from_results(version):
    error = EvalResult(name="collector:world", label="error", explanation="observation failed")
    record = EvalRunRecord(
        None,
        EvalContext(steps=[], metadata={"eval_collector_errors": [error.to_dict()]}),
        [EvalResult(name="grader", score=1.0)],
        Path("run"),
    )
    report = EvalReport.from_records(
        [record],
        grader_manifest=[{"name": "grader", "marks": [], "required": False}],
        version=version,
    )
    assert not report.passed and not report.valid
    assert report.cases[0]["results"][-1]["state"] == "collector_error"


@pytest.mark.parametrize(
    "item",
    [
        {"name": "grader", "marks": [], "required": True},
        {"name": "grader", "marks": ["required"], "required": False},
        {"name": "collector:world", "marks": [], "required": False},
        {"name": "", "marks": [], "required": False},
        {"name": "grader", "marks": [None], "required": False},
    ],
)
def test_assessment_refuses_inconsistent_or_reserved_manifest_entries(item):
    with pytest.raises(ValueError, match="Invalid grader manifest"):
        EvalReport.from_records([], grader_manifest=[item])
