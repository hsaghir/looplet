"""Per-case eval-run persistence: online grading and offline inspection
share one on-disk format.

``save_eval_run`` writes a per-case directory (trajectory.json +
artifacts.json + evals.json + case.json) and ``load_eval_run`` reads it
back as an ``EvalRunRecord`` (case + EvalContext with the full step
trajectory and outcome artifacts + persisted grader scores). This is
the mechanism for inspecting *exactly what happened* on each eval case,
and it keeps online (live EvalHook) and offline (reload + re-grade)
evals on the same files — they are NOT consolidated away.

Lives in ``looplet.evals`` (no cartridge dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import (
    EvalCase,
    EvalContext,
    EvalResult,
    EvalRunRecord,
    eval_run,
    load_eval_run,
    save_eval_run,
)


class _FakeRecorder:
    """Stand-in for TrajectoryRecorder.save — writes the trajectory.json
    + steps/ layout that the real recorder produces, without driving a
    live loop. Keeps the test fast and deterministic."""

    def __init__(self, steps: list[dict], task: dict, termination: str = "done") -> None:
        self._steps = steps
        self._task = task
        self._termination = termination

    def save(self, directory) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        traj = {
            "run_id": "test-run",
            "task": self._task,
            "termination_reason": self._termination,
            "steps": self._steps,
        }
        (root / "trajectory.json").write_text(json.dumps(traj, indent=2))
        steps_dir = root / "steps"
        steps_dir.mkdir(exist_ok=True)
        for i, s in enumerate(self._steps):
            (steps_dir / f"step_{i:02d}.json").write_text(json.dumps(s, indent=2))
        return root


class _FakeEvalHook:
    def __init__(self, artifacts: dict, results: list[EvalResult]) -> None:
        self.artifacts = artifacts
        self.results = results


def _sample_steps() -> list[dict]:
    return [
        {
            "tool_call": {"tool": "bash", "args": {"command": "pytest -q"}},
            "tool_result": {"data": {"exit_code": 0}},
        },
        {
            "tool_call": {"tool": "done", "args": {"summary": "fixed it"}},
            "tool_result": {"data": {"status": "completed"}},
        },
    ]


def _grader_tests_pass(ctx):
    return bool(ctx.artifacts.get("tests_passing", False))


def _grader_completed(ctx):
    return ctx.completed


def test_save_eval_run_writes_full_per_case_layout(tmp_path: Path) -> None:
    recorder = _FakeRecorder(_sample_steps(), task={"goal": "fix bug"})
    hook = _FakeEvalHook(
        artifacts={"tests_passing": True, "test_exit_code": 0},
        results=[EvalResult(name="eval_tests_pass", score=1.0, label="pass")],
    )
    case = EvalCase(id="bugfix_1", task={"goal": "fix bug"}, expected={"tests_passing": True})

    out = save_eval_run(tmp_path / "run", recorder=recorder, eval_hook=hook, case=case)

    assert (out / "trajectory.json").is_file()
    assert (out / "artifacts.json").is_file()  # the piece TrajectoryRecorder never wrote
    assert (out / "evals.json").is_file()
    assert (out / "case.json").is_file()
    assert (out / "steps").is_dir()
    # artifacts.json holds the outcome data, readable on its own.
    assert json.loads((out / "artifacts.json").read_text())["tests_passing"] is True


def test_load_eval_run_returns_trajectory_artifacts_scores(tmp_path: Path) -> None:
    recorder = _FakeRecorder(_sample_steps(), task={"goal": "fix bug"})
    hook = _FakeEvalHook(
        artifacts={"tests_passing": True},
        results=[
            EvalResult(name="eval_tests_pass", score=1.0, label="pass"),
            EvalResult(name="eval_completed", score=1.0, label="pass"),
        ],
    )
    case = EvalCase(id="bugfix_1", task={"goal": "fix bug"})
    save_eval_run(tmp_path / "run", recorder=recorder, eval_hook=hook, case=case)

    rec = load_eval_run(tmp_path / "run")
    assert isinstance(rec, EvalRunRecord)
    # case round-trips
    assert rec.case.id == "bugfix_1"
    # full trajectory is readable
    assert isinstance(rec.context, EvalContext)
    assert rec.context.tool_sequence == ["bash", "done"]
    assert rec.context.completed is True
    # outcome artifacts are readable
    assert rec.context.artifacts["tests_passing"] is True
    # persisted online scores round-trip
    assert {r.name: r.score for r in rec.results} == {
        "eval_tests_pass": 1.0,
        "eval_completed": 1.0,
    }


def test_same_graders_score_online_and_offline(tmp_path: Path) -> None:
    # The core "both online and offline" property: the SAME grader
    # functions produce the SAME scores whether run live (on the hook's
    # ctx) or offline (on the reloaded trajectory).
    recorder = _FakeRecorder(_sample_steps(), task={"goal": "x"})
    online_results = eval_run(
        [_grader_tests_pass, _grader_completed],
        EvalContext(steps=[], task={}, artifacts={"tests_passing": True}, stop_reason="done"),
    )
    hook = _FakeEvalHook(artifacts={"tests_passing": True}, results=online_results)
    save_eval_run(
        tmp_path / "run", recorder=recorder, eval_hook=hook, case=EvalCase(id="c", task={})
    )

    rec = load_eval_run(tmp_path / "run")
    offline_results = eval_run([_grader_tests_pass, _grader_completed], rec.context)

    online = {r.name: r.score for r in online_results}
    offline = {r.name: r.score for r in offline_results}
    assert online == offline == {"_grader_tests_pass": 1.0, "_grader_completed": 1.0}


def test_save_eval_run_without_hook_writes_empty_artifacts(tmp_path: Path) -> None:
    recorder = _FakeRecorder(_sample_steps(), task={})
    save_eval_run(tmp_path / "run", recorder=recorder)
    rec = load_eval_run(tmp_path / "run")
    assert rec.context.artifacts == {}
    assert rec.results == []
    assert rec.case is None


def test_explicit_results_override_hook(tmp_path: Path) -> None:
    recorder = _FakeRecorder(_sample_steps(), task={})
    hook = _FakeEvalHook(artifacts={}, results=[EvalResult(name="from_hook", score=0.0)])
    explicit = [EvalResult(name="from_offline", score=1.0)]
    save_eval_run(tmp_path / "run", recorder=recorder, eval_hook=hook, results=explicit)
    rec = load_eval_run(tmp_path / "run")
    assert [r.name for r in rec.results] == ["from_offline"]


def test_load_missing_trajectory_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_eval_run(tmp_path / "empty")
