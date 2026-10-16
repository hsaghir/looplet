"""Eval framework - pytest-style evaluation for agent runs.

Write functions named ``eval_*`` that take an :class:`EvalContext`
and return a score, label, dict, or :class:`EvalResult`. The
framework discovers them, runs them, and aggregates results.

Quick start::

    # eval_my_agent.py (anywhere in your project)

    def eval_task_completed(ctx):
        return "correct" if ctx.final_output.get("answer") == ctx.task.get("expected") else "wrong"

    def eval_tests_passed(ctx):
        # Outcome-grounded: read from artifacts the collector populated,
        # not from the trajectory.
        return ctx.artifacts.get("tests_passing", False)

    def eval_step_cost(ctx):
        # Cost metric, NOT a quality score. Surface it as data so you
        # can plot cost-vs-quality without conflating them.
        return {"steps": float(ctx.step_count)}

    def eval_reasoning_quality(ctx, llm):
        resp = llm.generate(f"Score 0-1: is {ctx.final_output} a well-supported answer given {ctx.session_log_text}?")
        return float(resp.strip())

Run evals::

    from looplet.evals import eval_discover, eval_run, EvalContext

    fns = eval_discover("eval_my_agent.py")
    ctx = EvalContext.from_trajectory_dir("traces/run_1/")
    results = eval_run(fns, ctx)
    for r in results:
        print(r.pretty())

Or attach to the loop for live scoring with outcome collectors::

    from looplet.evals import EvalHook

    def collect_test_results(state):
        # Re-run the test suite or read its last exit code from disk.
        return {"tests_passing": _tests_pass()}

    hook = EvalHook(
        evaluators=[eval_task_completed, eval_tests_passed],
        collectors=[collect_test_results],
    )
    for step in composable_loop(..., hooks=[hook]):
        ...
    print(hook.summary())

Prefer evaluating ``ctx.final_output`` and ``ctx.artifacts`` over
indexing ``ctx.tool_sequence`` or grepping ``ctx.steps`` - the
former survives the model changing its workflow, the latter
encodes today's expected trajectory as a permanent grade.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import shlex
import shutil
import sys
import tempfile
import time
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TextIO, cast

from looplet.artifact_compat import (
    begin_artifact_write,
    read_artifact_descriptor,
    write_artifact_descriptor,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from looplet.session import SessionLog
    from looplet.types import AgentState, LLMBackend

__all__ = [
    "EvalCase",
    "EvalContext",
    "EvalResult",
    "EvalReport",
    "EvalHook",
    "assert_evals_pass",
    "eval_discover",
    "eval_assess",
    "eval_run",
    "eval_run_batch",
    "eval_mark",
    "eval_cli",
    "load_cases",
    "parametrize_cases",
    "save_case",
    "save_cases",
    "pytest_param_cases",
    "CARTRIDGE_CASES_SUBPATH",
    "load_cartridge_cases",
    "save_cartridge_cases",
    "CARTRIDGE_EVALS_SUBPATH",
    "CartridgeEvals",
    "discover_collectors",
    "load_evals",
    "load_cartridge_evals",
    "EvalRunRecord",
    "save_eval_run",
    "load_eval_run",
    "seed_case_workspace",
    "promote_to_offline",
    "run_cartridge_evals",
]

logger = logging.getLogger(__name__)

# Where a cartridge's eval corpus lives, relative to the cartridge root.
# Evals are an *adjacent artifact* (per the "Agents Are Files" design):
# the case files travel *inside* the cartridge directory so they are
# version-controlled with the agent, but the cartridge *package* stays
# evals-agnostic. The convention (this subpath) is owned here, in
# ``looplet.evals`` - not in ``looplet.cartridge`` - so the dependency
# points evals → (a path string), never cartridge → evals.
CARTRIDGE_CASES_SUBPATH = "evals/cases"

# The cartridge's eval *bundle* root. Cases live in ``evals/cases/``
# (data); graders (``eval_*.py``) and outcome collectors
# (``collect_*.py``) live directly under ``evals/`` (code). One slot,
# everything an agent version needs to grade itself.
CARTRIDGE_EVALS_SUBPATH = "evals"

# ``required`` is a semantic use of the existing mark mechanism, not a
# second evaluator taxonomy. The CLI treats a skipped required grader as
# an integrity failure; unmarked judge graders may still be intentionally
# omitted when no judge backend is configured.
_REQUIRED_EVAL_MARK = "required"
_EVAL_SUMMARY_SCHEMA = "looplet.eval-summary"
_EVAL_SUMMARY_VERSION = 1
_EVAL_RESULT_STATES = {
    "pass",
    "explicit_fail",
    "threshold_fail",
    "skipped",
    "missing",
    "collector_error",
    "grader_error",
    "metric_only",
}


def _validate_case_id(case_id: str) -> str:
    """Return a safe case id or raise before it reaches a filesystem path."""
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("EvalCase id must be a non-empty string")
    has_control = any(ord(char) < 32 or ord(char) == 127 for char in case_id)
    if case_id in {".", ".."} or "/" in case_id or "\\" in case_id or has_control:
        raise ValueError(f"EvalCase id must be one path-safe component, got {case_id!r}")
    return case_id


def _contained_child(root: Path, relative: str, *, label: str) -> Path:
    """Join ``relative`` beneath ``root`` and reject traversal/symlink escapes."""
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or "\x00" in relative:
        raise ValueError(f"{label} {relative!r} escapes its root")
    target = root / rel
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} {relative!r} escapes its root") from exc
    if resolved_target == resolved_root:
        raise ValueError(f"{label} {relative!r} must name a file or child directory")
    return target


class CartridgeEvals(NamedTuple):
    """The complete eval bundle discovered inside a cartridge's ``evals/``.

    Unpacks as ``cases, graders, collectors`` and also offers named
    access. ``graders`` are case-agnostic ``eval_*`` predicates run N×M
    over every case; ``collectors`` are ``collect_*`` callables that
    populate :attr:`EvalContext.artifacts` for outcome-grounded grading.
    """

    cases: list["EvalCase"]
    graders: list[Callable]
    collectors: list[Callable]


class EvalRunRecord(NamedTuple):
    """One persisted eval-case run: case + trajectory/artifacts + scores.

    Returned by :func:`load_eval_run`. ``context`` is an
    :class:`EvalContext` carrying the full step trajectory AND the
    outcome ``artifacts``, so the same graders that scored the run live
    can re-score it offline. ``results`` are the persisted grader
    scores; ``case`` is the originating :class:`EvalCase` when the run
    dir recorded it.
    """

    case: "EvalCase | None"
    context: "EvalContext"
    results: list["EvalResult"]
    directory: Path
    cleanup_directory: Path | None = None

    def diagnostics(self) -> dict[str, Any]:
        """Inspect execution only; protected case data and scores stay separate."""
        return self.context.diagnostics()

    def cleanup(self) -> None:
        """Remove an owned no-output sandbox, if one exists.

        Persisted eval records never own their evidence directory. A no-output
        record remains inspectable until this method is called explicitly.
        """
        if self.cleanup_directory is None:
            return
        shutil.rmtree(self.cleanup_directory, ignore_errors=True)

    def __enter__(self) -> "EvalRunRecord":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.cleanup()


@dataclass(frozen=True)
class EvalReport:
    """A versioned assessment, without agent inputs or collected artifacts."""

    _data: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", deepcopy(self._data))

    @property
    def passed(self) -> bool:
        return self._data["passed"]

    @property
    def valid(self) -> bool:
        return not self._data["integrity_failures"]

    @property
    def version(self) -> int:
        return self._data["version"]

    @property
    def grader_manifest(self) -> list[dict[str, Any]]:
        return deepcopy(self._data["grader_manifest"])

    @property
    def cases(self) -> list[dict[str, Any]]:
        return deepcopy(self._data["cases"])

    @property
    def integrity_failures(self) -> list[str]:
        return list(self._data["integrity_failures"])

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-safe assessment snapshot."""
        return deepcopy(self._data)

    def assert_passed(self) -> None:
        """Raise with failed checks or invalid assessment inputs."""
        messages = list(self.integrity_failures)
        for case in self._data["cases"]:
            for result in case["results"]:
                if _eval_state_fails(result["state"], required=result["required"]):
                    if result["state"] == "missing":
                        kind = "required grader" if result["required"] else "grader"
                        detail = f"{kind} missing from run result"
                    else:
                        detail = result.get("explanation") or result.get("label") or result["state"]
                    messages.append(f"{case['id']}/{result['name']}: {detail}")
        assert self.passed, "\n".join(dict.fromkeys(messages)) or "Evaluation did not pass"

    @classmethod
    def from_records(
        cls,
        records: list[EvalRunRecord],
        *,
        grader_manifest: list[dict[str, Any]] | None = None,
        threshold: float = 0.5,
        output_dir: str | None = None,
        version: int = 2,
        integrity_failures: list[str] | None = None,
    ) -> "EvalReport":
        """Assess existing observations without executing agents or collectors."""
        if grader_manifest is None:
            manifests = [record.context.metadata.get("eval_grader_manifest") for record in records]
            grader_manifest = manifests[0] if manifests and isinstance(manifests[0], list) else []
            integrity_failures = list(integrity_failures or [])
            if not grader_manifest or any(item != grader_manifest for item in manifests):
                integrity_failures.append("Missing or inconsistent recorded grader inventory")
        return cls(
            _assess_eval_records(
                records=records,
                grader_manifest=grader_manifest,
                threshold=threshold,
                output_dir=output_dir,
                version=version,
                integrity_failures=integrity_failures,
            )
        )


# ── Core data types ──────────────────────────────────────────────


@dataclass
class EvalContext:
    """Everything an evaluator sees - the same data you see when debugging.

    Build from a live loop run (via :class:`EvalHook`) or from saved
    trajectories (via :meth:`from_trajectory_dir`).
    """

    steps: list[Any]
    """Full list of :class:`Step` objects from the run."""

    task: dict[str, Any] = field(default_factory=dict)
    """Original task dict passed to ``composable_loop``."""

    final_output: dict[str, Any] = field(default_factory=dict)
    """The ``done()`` tool's args - the agent's final answer."""

    session_log_text: str = ""
    """Rendered session log - the text the LLM saw."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra context: run_id, model, timestamp, etc."""

    artifacts: dict[str, Any] = field(default_factory=dict)
    """Outcome data collected from outside the trajectory.

    Populated by :class:`EvalHook` collectors at the end of a run, or
    loaded from ``artifacts.json`` via :meth:`from_trajectory_dir`.
    Use this slot to grade *what changed in the world* - tests
    passing, files modified, repo state - instead of grepping
    :attr:`steps` for tool calls. See ``docs/evals.md`` for the
    "trajectory-blind eval" recipe.
    """

    stop_reason: str | None = None
    """Why the loop terminated: ``\"done\"`` if the agent called ``done()``,
    otherwise a hook-supplied reason (``\"hook_stop\"``, ``\"budget\"``, ...)
    or ``None`` when unknown.  Evaluators should dispatch on this to
    handle early termination, e.g.::

        def eval_completed(ctx):
            return ctx.stop_reason == \"done\"
    """

    @property
    def expected(self) -> dict[str, Any]:
        """Grader-only expectations, with the existing task view kept compatible."""
        expected = self.task.get("expected", {})
        if not isinstance(expected, dict):
            raise ValueError("Expected grader data must be an object")
        return expected

    @property
    def completed(self) -> bool:
        """True when the agent called ``done()`` itself (not stopped by a hook)."""
        return self.stop_reason == "done"

    def diagnostics(self) -> dict[str, Any]:
        """Return the shared redacted view without artifacts or grader inputs."""
        from looplet.run_records import run_diagnostics

        return run_diagnostics(
            {
                "status": self.metadata.get("run_status"),
                "phase": self.metadata.get("run_phase"),
                "termination_reason": self.stop_reason,
                "steps": self.steps,
                "metadata": self.metadata,
                "run_envelope": self.metadata.get("run_envelope"),
            }
        )

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered list of tool names called during the run."""
        return [
            getattr(s.tool_call, "tool", "?")
            for s in self.steps
            if hasattr(s, "tool_call") and s.tool_call
        ]

    @property
    def errors(self) -> list[Any]:
        """Steps where the tool returned an error."""
        return [
            s
            for s in self.steps
            if hasattr(s, "tool_result") and s.tool_result and getattr(s.tool_result, "error", None)
        ]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @classmethod
    def from_trajectory_dir(cls, path: str | Path) -> "EvalContext":
        """Load an EvalContext from a saved trajectory directory.

        Expects ``trajectory.json`` (from :class:`TrajectoryRecorder`).
        """
        root = Path(path)
        read_artifact_descriptor(
            root,
            expected_kinds=("eval_run", "provenance"),
        )
        traj_path = root / "trajectory.json"
        if not traj_path.exists():
            raise FileNotFoundError(f"No trajectory.json in {root}")

        try:
            data = json.loads(traj_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid trajectory.json in {root}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid trajectory.json in {root}: expected object, got {type(data).__name__}"
            )
        steps = data.get("steps", [])
        if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
            raise ValueError(f"Invalid trajectory.json in {root}: steps must be a list of objects")
        raw_task = data.get("task", {})
        task: dict[str, Any]
        if not isinstance(raw_task, dict):
            task = {"description": str(raw_task)} if raw_task else {}
        else:
            task = dict(raw_task)
        expected = data.get("expected")
        expected_path = root / "expected.json"
        if expected_path.exists():
            try:
                persisted_expected = json.loads(expected_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid expected.json in {root}: {exc}") from exc
            if not isinstance(persisted_expected, dict):
                raise ValueError(
                    f"Invalid expected.json in {root}: expected object, "
                    f"got {type(persisted_expected).__name__}"
                )
            if expected is not None and expected != persisted_expected:
                raise ValueError(f"Conflicting grader expectations in {root}")
            expected = persisted_expected
        if expected is not None:
            if not isinstance(expected, dict):
                raise ValueError(
                    f"Invalid trajectory.json in {root}: expected must be an object, "
                    f"got {type(expected).__name__}"
                )
            if "expected" in task and task["expected"] != expected:
                raise ValueError(f"Conflicting grader expectations in {root}")
            # Keep the persisted agent-visible task honest while preserving
            # the documented grader API after reload.
            task["expected"] = expected
        # Pull through the trajectory's own metadata dict (which may
        # contain harness_snapshot from TrajectoryRecorder, plus any
        # user-attached fields) and overlay the available well-known
        # top-level fields at their documented keys.
        traj_metadata = data.get("metadata") or {}
        if not isinstance(traj_metadata, dict):
            traj_metadata = {}
        metadata: dict[str, Any] = dict(traj_metadata)
        top_level_metadata = {
            "run_id": data.get("run_id"),
            "started_at": data.get("started_at"),
            "ended_at": data.get("ended_at"),
            "termination_reason": data.get("termination_reason"),
        }
        metadata.update(
            {key: value for key, value in top_level_metadata.items() if value is not None}
        )

        stop_reason = data.get("termination_reason") or metadata.get("termination_reason")

        # A successful terminal call is always the final step, regardless
        # of whether the cartridge names it ``done``, ``submit``, or a
        # secondary sentinel such as ``escalate``. Fall back to the legacy
        # hard-coded name only for old trajectories with no stop reason.
        final_output: dict[str, Any] = {}
        terminal_steps = steps[-1:] if stop_reason == "done" else []
        if stop_reason is None:
            terminal_steps = [
                s
                for s in reversed(steps)
                if (s.get("tool_call", {}).get("tool") or s.get("tool", "")) == "done"
            ][:1]
        if terminal_steps:
            raw_output = terminal_steps[0].get("tool_call", {}).get("args", {})
            if isinstance(raw_output, dict):
                final_output = raw_output

        # Also load from metrics.json if available (ground-truth data)
        metrics_path = root / "metrics.json"
        if metrics_path.exists():
            try:
                metrics_data = json.loads(metrics_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid metrics.json in {root}: {exc}") from exc
            if not isinstance(metrics_data, dict):
                raise ValueError(
                    f"Invalid metrics.json in {root}: expected object, "
                    f"got {type(metrics_data).__name__}"
                )
            # Legacy benchmark traces place ground truth in metrics.json.
            # Preserve that reader contract, but never silently ignore a
            # malformed oracle file.
            for key, value in metrics_data.items():
                if key.startswith("expected_") and key not in task:
                    task[key] = value
            if not final_output and isinstance(metrics_data.get("output"), dict):
                final_output = metrics_data["output"]

        # Load artifacts.json if present - outcome data collected
        # outside the trajectory (test results, file diffs, repo
        # state, etc.). See EvalHook(collectors=...).
        artifacts: dict[str, Any] = {}
        artifacts_path = root / "artifacts.json"
        if artifacts_path.exists():
            try:
                loaded = json.loads(artifacts_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid artifacts.json in {root}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Invalid artifacts.json in {root}: expected object, "
                    f"got {type(loaded).__name__}"
                )
            artifacts = loaded

        return cls(
            steps=[_DictStep(s) for s in steps],
            task=task if isinstance(task, dict) else {"description": str(task)},
            final_output=final_output,
            session_log_text=str(data.get("session_log_text") or ""),
            metadata=metadata,
            artifacts=artifacts,
            stop_reason=stop_reason,
        )


@dataclass
class _DictStep:
    """Lightweight step wrapper for trajectories loaded from JSON.

    Supports both formats:
      - looplet: {"tool_call": {"tool": "x"}, "tool_result": {"data": {}}}
      - benchmark:   {"tool": "x", "args_summary": "...", "error": null}
    """

    _data: dict[str, Any]

    @property
    def tool_call(self) -> Any:
        tc = self._data.get("tool_call", {})
        if not tc and "tool" in self._data:
            # Flat format: tool name at top level
            tc = {"tool": self._data["tool"], "args": self._data.get("args", {})}
        return _DictView(tc)

    @property
    def tool_result(self) -> Any:
        tr = self._data.get("tool_result", {})
        if not tr and "error" in self._data:
            tr = {"error": self._data.get("error"), "data": self._data.get("data", {})}
        return _DictView(tr)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = self._data.get("metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}


class _DictView:
    """Attribute-access wrapper for dicts (so eval functions can use dot notation)."""

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d

    def __getattr__(self, name: str) -> Any:
        return self._d.get(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)


@dataclass
class EvalResult:
    """Result of one evaluator function.

    Evaluators can return any of: ``float``, ``str``, ``dict``,
    or ``EvalResult`` directly. The framework normalizes via
    :meth:`from_return`.
    """

    name: str = ""
    """Evaluator function name (set by the runner)."""

    score: float | None = None
    """Numeric score 0–1, if applicable."""

    label: str | None = None
    """Categorical label (e.g. 'correct', 'partial', 'wrong')."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Named numeric metrics (precision, recall, F1, etc.)."""

    details: list[str] = field(default_factory=list)
    """Specific findings (missed items, unsupported claims, etc.)."""

    explanation: str = ""
    """Human-readable summary of the evaluation."""

    duration_ms: float = 0.0
    """How long the evaluator took to run."""

    def __post_init__(self) -> None:
        if self.score is not None and (
            not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError(f"score must be finite and between 0 and 1, got {self.score!r}")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(f"label must be a string, got {type(self.label).__name__}")
        for key, metric in self.metrics.items():
            if not isinstance(metric, (int, float)) or not math.isfinite(metric):
                raise ValueError(f"metric {key!r} must be finite, got {metric!r}")

    @property
    def passed(self) -> bool:
        """Whether this evaluation passed.

        True when ``score`` is not ``None`` and ``>= 0.5``, OR when
        ``label`` is in ``{"pass", "correct", "yes"}`` (case-insensitive).
        Returns ``False`` otherwise, including when both ``score`` and
        ``label`` are unset.
        """
        return _eval_result_state(self, required=False, threshold=0.5) == "pass"

    @classmethod
    def from_return(cls, value: Any, *, name: str = "") -> "EvalResult":
        """Normalize any return type into an EvalResult."""
        if isinstance(value, EvalResult):
            if not value.name:
                value.name = name
            return value
        if isinstance(value, bool):
            return cls(name=name, score=1.0 if value else 0.0, label="pass" if value else "fail")
        if isinstance(value, (int, float)):
            score = float(value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                return cls(
                    name=name,
                    label="error",
                    explanation=f"score must be finite and between 0 and 1, got {score!r}",
                )
            return cls(name=name, score=score)
        if isinstance(value, str):
            return cls(name=name, label=value)
        if isinstance(value, dict):
            metrics = {k: float(v) for k, v in value.items() if isinstance(v, (int, float))}
            details = [f"{k}: {v}" for k, v in value.items() if not isinstance(v, (int, float))]
            invalid_metric = next(
                ((key, metric) for key, metric in metrics.items() if not math.isfinite(metric)),
                None,
            )
            if invalid_metric is not None:
                key, metric = invalid_metric
                return cls(
                    name=name,
                    label="error",
                    details=details,
                    explanation=f"metric {key!r} must be finite, got {metric!r}",
                )
            score_key = next(
                (key for key in ("score", "f1", "accuracy", "overall") if key in metrics),
                None,
            )
            score = metrics[score_key] if score_key is not None else None
            if score_key is not None and score_key != "score":
                warnings.warn(
                    f"Inferring an eval gate score from {score_key!r} is deprecated; "
                    "return EvalResult(metrics=...) for passive metrics or set score explicitly.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if score is not None and (not math.isfinite(score) or not 0.0 <= score <= 1.0):
                return cls(
                    name=name,
                    label="error",
                    metrics=metrics,
                    details=details,
                    explanation=f"primary score must be finite and between 0 and 1, got {score!r}",
                )
            return cls(name=name, score=score, metrics=metrics, details=details)
        return cls(
            name=name,
            label="error",
            explanation=f"unsupported evaluator return type: {type(value).__name__}",
        )

    def pretty(self) -> str:
        """One-line formatted output for terminal display."""
        parts = [f"{self.name:40s}"]
        if self.score is not None:
            parts.append(f"{self.score:.2f}")
        if self.label:
            parts.append(self.label)
        if self.metrics:
            metric_strs = [
                f"{k}={v:.2f}"
                for k, v in self.metrics.items()
                if k != "score" or self.score is None
            ]
            if metric_strs:
                parts.append(" ".join(metric_strs))
        if self.explanation:
            parts.append(self.explanation)
        result = " ".join(parts)
        if self.details:
            result += "\n" + "\n".join(f"  {d}" for d in self.details[:5])
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        d: dict[str, Any] = {"name": self.name}
        if self.score is not None:
            d["score"] = self.score
        if self.label:
            d["label"] = self.label
        if self.metrics:
            d["metrics"] = self.metrics
        if self.details:
            d["details"] = self.details
        if self.explanation:
            d["explanation"] = self.explanation
        if self.duration_ms:
            d["duration_ms"] = round(self.duration_ms, 1)
        return d


# ── Cases ────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    """A single eval case - task + expected outcomes + tags.

    Cases live as data: they round-trip to JSON via :meth:`to_dict`
    and :func:`load_cases`, so users can write them by hand and grow
    the corpus without touching Python.

    Field guidance:

    * ``id``: stable, human-readable (used as the file name and the
      pytest test id).
    * ``task``: passed verbatim to ``composable_loop(task=…)``.
    * ``expected``: free-form. Evaluators read what they need
      (``ctx.task["expected"]`` is populated by the runner). Keep
      keys outcome-oriented (``tests_passing``, ``files_created``)
      rather than trajectory-oriented (``read_file_count``).
    * ``marks``: same vocabulary as :func:`eval_mark`. Carried into
      pytest as native marks via :func:`pytest_param_cases`.
    * ``notes``: free-text - why this case was added, what regression
      it captures.
    """

    id: str
    task: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    marks: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_case_id(self.id)
        if not isinstance(self.task, dict):
            raise ValueError(f"EvalCase task must be an object, got {type(self.task).__name__}")
        if not isinstance(self.expected, dict):
            raise ValueError(
                f"EvalCase expected must be an object, got {type(self.expected).__name__}"
            )
        if not isinstance(self.marks, list) or any(
            not isinstance(mark, str) for mark in self.marks
        ):
            raise ValueError("EvalCase marks must be a list of strings")
        if not isinstance(self.notes, str):
            raise ValueError(f"EvalCase notes must be a string, got {type(self.notes).__name__}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "task": self.task}
        for key in ("expected", "marks", "notes"):
            if value := getattr(self, key):
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        if not isinstance(data, dict):
            raise ValueError(f"EvalCase must be an object, got {type(data).__name__}")
        if "id" not in data:
            raise ValueError("EvalCase requires 'id'")
        for name in data:
            if name not in cls.__dataclass_fields__:
                message = f"Unknown EvalCase field {name!r}"
                matches = get_close_matches(name, cls.__dataclass_fields__, n=1)
                if matches:
                    message += f"; did you mean {matches[0]!r}?"
                raise ValueError(message)
        return cls(**data)


def load_cases(path: str | Path) -> list["EvalCase"]:
    """Load eval cases from a file or directory.

    Accepts:

    * a single ``.json`` file containing one case dict, or a list of
      case dicts;
    * a single ``.jsonl`` file with one case dict per line;
    * a directory containing any mix of ``*.json`` / ``*.jsonl`` files
            (sorted by file name, then by their internal order);
        * an eval definitions root containing a ``cases/`` directory.

    Cases are returned in deterministic order so pytest IDs are stable.
    Raises :class:`FileNotFoundError` if ``path`` doesn't exist and
    :class:`ValueError` for malformed entries (so a bad case is loud).
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such case path: {root}")

    if root.is_dir() and (root / "cases").is_dir():
        root = root / "cases"
    if root.is_file():
        files = [root]
    else:
        files = sorted(
            (p for p in root.iterdir() if p.suffix in (".json", ".jsonl")),
            key=lambda p: p.name,
        )

    cases: list[EvalCase] = []
    seen: dict[str, Path] = {}
    for fpath in files:
        for location, raw in _iter_case_dicts(fpath):
            try:
                case = EvalCase.from_dict(raw)
            except ValueError as exc:
                raise ValueError(f"{location}: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"Duplicate case id {case.id!r} (in {fpath})")
            seen[case.id] = fpath
            cases.append(case)
    return cases


def _iter_case_dicts(fpath: Path):
    text = fpath.read_text()
    if fpath.suffix == ".jsonl":
        for ln, line in enumerate(text.split("\n"), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield f"{fpath}:{ln}", json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{fpath}:{ln}: invalid JSON ({e})") from e
        return
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{fpath}: invalid JSON ({e})") from e
    if isinstance(loaded, list):
        for index, raw in enumerate(loaded, 1):
            yield f"{fpath}: entry {index}", raw
    elif isinstance(loaded, dict):
        yield str(fpath), loaded
    else:
        raise ValueError(f"{fpath}: expected dict or list, got {type(loaded).__name__}")


def save_case(case: "EvalCase", path: str | Path) -> Path:
    """Write one case to ``path`` as pretty-printed JSON.

    Treated as a directory when ``path`` already exists as a directory,
    or when the string ends with a path separator (``"cases/"``).
    In that case writes to ``<path>/<case.id>.json``. Otherwise ``path``
    is the full target file path. Parent directories are created.

    Returns the written path.
    """
    raw = str(path)
    target = Path(path)
    treat_as_dir = (target.exists() and target.is_dir()) or raw.endswith(("/", "\\"))
    if treat_as_dir:
        case_id = _validate_case_id(case.id)
        target = _contained_child(target, f"{case_id}.json", label="case file")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(case.to_dict(), indent=2, sort_keys=True))
    return target


def save_cases(cases: list["EvalCase"], directory: str | Path) -> list[Path]:
    """Write a list of cases into ``directory`` as ``<id>.json`` files.

    Symmetric counterpart to :func:`load_cases` for the directory-based
    corpus pattern. Creates ``directory`` if it doesn't exist. Returns
    the list of written paths in input order.

    Raises ``ValueError`` when two cases share the same ``id`` (which
    would silently overwrite each other on disk).
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for c in cases:
        if c.id in seen:
            duplicates.add(c.id)
        seen.add(c.id)
    if duplicates:
        raise ValueError(f"duplicate case ids would overwrite each other: {sorted(duplicates)}")
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    return [save_case(c, target_dir) for c in cases]


def load_cartridge_cases(cartridge_dir: str | Path) -> list["EvalCase"]:
    """Load the eval corpus shipped inside a cartridge directory.

    Reads ``<cartridge_dir>/evals/cases/`` (see
    :data:`CARTRIDGE_CASES_SUBPATH`). This is how "evals ship with the
    agent version" is realised: the cases live in the cartridge
    directory and are version-controlled alongside the prompt / tools /
    hooks they protect, while the cartridge *loader* stays evals-
    agnostic. Pair the returned cases with a preset built by
    :func:`looplet.cartridge.cartridge_to_preset` to run an agent
    version against its own evals.

    Unlike :func:`load_cases`, a missing ``evals/cases/`` directory is
    not an error - it returns ``[]`` (a cartridge need not ship evals).
    Malformed case files still raise :class:`ValueError` (loud, never
    silently partial).
    """
    cases_dir = Path(cartridge_dir) / CARTRIDGE_CASES_SUBPATH
    if not cases_dir.is_dir():
        return []
    return load_cases(cases_dir)


def save_cartridge_cases(cartridge_dir: str | Path, cases: list["EvalCase"]) -> list[Path]:
    """Write an eval corpus into a cartridge directory's eval slot.

    Writes each case to ``<cartridge_dir>/evals/cases/<id>.json`` (see
    :data:`CARTRIDGE_CASES_SUBPATH`), creating the directory if needed.
    Symmetric counterpart to :func:`load_cartridge_cases`. Returns the
    written paths.

    Note this writes *only* the eval slot and never touches the rest of
    the cartridge - the eval corpus is managed independently of
    :func:`looplet.cartridge.preset_to_cartridge`, which by design does
    not serialise (or clobber) evals.
    """
    return save_cases(cases, Path(cartridge_dir) / CARTRIDGE_CASES_SUBPATH)


def discover_collectors(
    path: str | Path,
    *,
    runtime: dict[str, Any] | None = None,
    strict: bool = False,
) -> list[Callable]:
    """Discover outcome collectors (``collect_*`` functions) under ``path``.

    The sibling of :func:`eval_discover` for the *collector* role: where
    a grader reads ``EvalContext`` and returns a score, a collector runs
    once at end-of-loop and returns a ``dict`` merged into
    :attr:`EvalContext.artifacts` (so graders can grade *outcomes* -
    tests passing, files changed - instead of grepping the trajectory).

    A collector is a top-level function named ``collect_*`` whose
    signature is either ``(state) -> dict`` or ``(state, runtime) ->
    dict``. When the signature declares ``runtime``, the supplied
    ``runtime`` dict is bound here via :func:`functools.partial`, so the
    returned callables are uniformly ``(state) -> dict`` and drop
    straight into ``EvalHook(collectors=…)``. This mirrors how
    :func:`eval_run` passes ``llm`` only to evaluators whose signature
    asks for it - runtime is injected by introspection, not convention
    soup. Runtime-parameterised collectors are what let a collector
    re-run a test suite in the agent's sandbox without hard-coding the
    path.
    """
    raw = eval_discover(
        path,
        pattern="collect_*.py",
        prefix="collect_",
        strict=strict,
    )
    return _bind_collectors(raw, runtime)


def _bind_collectors(collectors: list[Callable], runtime: dict[str, Any] | None) -> list[Callable]:
    bound: list[Callable] = []
    for fn in collectors:
        try:
            needs_runtime = "runtime" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            needs_runtime = False
        if needs_runtime:
            bound.append(functools.partial(fn, runtime=dict(runtime or {})))
        else:
            bound.append(fn)
    return bound


def load_evals(
    path: str | Path,
    *,
    runtime: dict[str, Any] | None = None,
    strict: bool = True,
) -> "CartridgeEvals":
    """Load ordinary eval functions from a Python file or directory.

    ``eval_*`` graders and optional ``collect_*`` collectors can share one file.
    Directories also load JSON cases under ``cases/`` and preserve separate files.
    Each module loads once. This does not execute a run or call its functions.
    Strict loading rejects missing paths and broken definition imports.
    """
    root = Path(path)
    if not root.is_dir() and not root.is_file():
        if strict:
            raise ValueError(f"Eval definitions not found: {root}")
        return CartridgeEvals(cases=[], graders=[], collectors=[])
    cases_dir = root / "cases"
    cases = load_cases(cases_dir) if cases_dir.is_dir() else []
    definitions = _discover_functions(
        root, patterns=("eval_*.py", "collect_*.py"), prefixes=("eval_", "collect_"), strict=strict
    )
    return CartridgeEvals(
        cases=cases,
        graders=definitions["eval_"],
        collectors=_bind_collectors(definitions["collect_"], runtime),
    )


def load_cartridge_evals(
    cartridge_dir: str | Path,
    *,
    runtime: dict[str, Any] | None = None,
    strict: bool = False,
) -> "CartridgeEvals":
    """Discover the full eval bundle shipped inside a cartridge.

    Returns a :class:`CartridgeEvals` (``cases, graders, collectors``)
    read from ``<cartridge_dir>/evals/``:

    * ``cases`` - ``evals/cases/*.json`` / ``*.jsonl`` (data).
    * ``graders`` - ``eval_*`` functions in ``evals/eval_*.py``.
    * ``collectors`` - ``collect_*`` functions in ``evals/collect_*.py``
      (runtime-bound via :func:`discover_collectors`).

    This is the single call that makes "evals ship with the agent
    version" complete: one slot, one lookup, everything an agent
    version needs to grade itself travels with it. Pair the result with
    a preset from :func:`looplet.cartridge.cartridge_to_preset` and an
    :class:`EvalHook` to run a version against its own evals::

        evals = load_cartridge_evals(cdir, runtime={"workspace_dir": sb})
        preset = cartridge_to_preset(cdir, runtime={"workspace_dir": sb})
        hook = EvalHook(evaluators=evals.graders, collectors=evals.collectors)
        preset.hooks.append(hook)
        for _ in preset.run(llm, task=case.task):
            pass

    A cartridge with no ``evals/`` directory yields empty lists (never an
    error); malformed case files still raise loudly (see
    :func:`load_cartridge_cases`).
    """
    evals_root = Path(cartridge_dir) / CARTRIDGE_EVALS_SUBPATH
    if not evals_root.is_dir():
        return CartridgeEvals(cases=[], graders=[], collectors=[])
    return load_evals(evals_root, runtime=runtime, strict=strict)


def _eval_context_to_trajectory(
    ctx: "EvalContext",
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise an :class:`EvalContext` into the ``trajectory.json`` shape
    that :meth:`EvalContext.from_trajectory_dir` reads back.

    Works for BOTH a live context (real ``Step`` objects, captured by an
    online :class:`EvalHook`) and a reloaded context (``_DictStep``),
    using attribute access both support. This is what lets an online run
    be *promoted* to the offline on-disk format without a separate
    :class:`looplet.provenance.TrajectoryRecorder`.
    """
    steps_out: list[dict[str, Any]] = []
    for i, s in enumerate(ctx.steps):
        tc = getattr(s, "tool_call", None)
        tr = getattr(s, "tool_result", None)
        tool = getattr(tc, "tool", None) if tc is not None else None
        args = (getattr(tc, "args", None) if tc is not None else None) or {}
        data = getattr(tr, "data", None) if tr is not None else None
        error = getattr(tr, "error", None) if tr is not None else None
        steps_out.append(
            {
                "step_num": getattr(s, "number", i),
                "tool_call": {"tool": tool, "args": args},
                "tool_result": {"data": data, "error": error},
            }
        )
    task = dict(ctx.task)
    if expected is not None:
        task.pop("expected", None)
    return {
        "run_id": (ctx.metadata or {}).get("run_id"),
        "task": task,
        "termination_reason": ctx.stop_reason,
        "session_log_text": ctx.session_log_text,
        "metadata": dict(ctx.metadata or {}),
        "step_count": len(ctx.steps),
        "steps": steps_out,
    }


def _write_json(
    path: Path,
    value: Any,
    *,
    redact: Callable[[str], str] | None = None,
) -> None:
    """Write one JSON eval artifact through the recorder's redactor."""
    text = json.dumps(value, indent=2, default=str)
    if redact is not None:
        text = redact(text)
    path.write_text(text, encoding="utf-8")


def _merge_context_into_recorder(
    recorder: Any,
    ctx: "EvalContext",
    *,
    expected: dict[str, Any] | None = None,
) -> bool:
    """Merge context before a recorder's own serialization/redaction."""
    trajectory = getattr(recorder, "trajectory", None)
    if trajectory is None:
        return False
    required = ("session_log_text", "metadata", "task", "termination_reason")
    if not all(hasattr(trajectory, field_name) for field_name in required):
        return False
    trajectory.session_log_text = ctx.session_log_text
    context_metadata = dict(ctx.metadata or {})
    recorded_metadata = getattr(trajectory, "metadata", {}) or {}
    if isinstance(recorded_metadata, dict):
        context_metadata.update(recorded_metadata)
    trajectory.metadata = context_metadata
    if not getattr(trajectory, "task", None) and ctx.task:
        task = dict(ctx.task)
        if expected is not None:
            task.pop("expected", None)
        trajectory.task = task
    if getattr(trajectory, "termination_reason", None) in (None, "", "unknown"):
        if ctx.stop_reason:
            trajectory.termination_reason = ctx.stop_reason
    return True


def _merge_context_into_saved_trajectory(
    root: Path,
    ctx: "EvalContext | None",
    *,
    expected: dict[str, Any] | None = None,
    redact: Callable[[str], str] | None = None,
) -> None:
    """Add eval-only context fields without replacing recorder fidelity."""
    trajectory_path = root / "trajectory.json"
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid trajectory.json in {root}: {exc}") from exc
    if not isinstance(trajectory, dict):
        raise ValueError(
            f"Invalid trajectory.json in {root}: expected object, got {type(trajectory).__name__}"
        )

    if ctx is not None:
        trajectory["session_log_text"] = ctx.session_log_text
        context_metadata = dict(ctx.metadata or {})
        recorded_metadata = trajectory.get("metadata") or {}
        if isinstance(recorded_metadata, dict):
            context_metadata.update(recorded_metadata)
        trajectory["metadata"] = context_metadata
    if ctx is not None and not trajectory.get("task") and ctx.task:
        task = dict(ctx.task)
        if expected is not None:
            task.pop("expected", None)
        trajectory["task"] = task
    if (
        ctx is not None
        and trajectory.get("termination_reason") in (None, "", "unknown")
        and ctx.stop_reason
    ):
        trajectory["termination_reason"] = ctx.stop_reason

    _write_json(trajectory_path, trajectory, redact=redact)


def save_eval_run(
    directory: str | Path,
    *,
    recorder: Any = None,
    context: "EvalContext | None" = None,
    eval_hook: Any = None,
    case: "EvalCase | None" = None,
    results: list["EvalResult"] | None = None,
    redact: Callable[[str], str] | None = None,
) -> Path:
    """Persist ONE eval case's run into a directory the reader understands.

    This is the write side that closes the gap the dogfood exposed:
    :meth:`looplet.provenance.TrajectoryRecorder.save` writes
    ``trajectory.json`` + ``steps/`` but never the ``artifacts.json``
    that outcome-grounded grading depends on, and
    :meth:`EvalHook.save` bundles artifacts into a different shape.
    ``save_eval_run`` writes the per-case layout that
    :meth:`EvalContext.from_trajectory_dir` (hence :func:`load_eval_run`)
    reads - so **online** grading (the live :class:`EvalHook`) and
    **offline** inspection (reload + re-grade) share one on-disk format.

    Layout written under ``directory``::

        <dir>/
          trajectory.json   # full trajectory
          steps/step_NN.json
          artifacts.json    # outcome data (eval_hook.artifacts)  ← the gap
          evals.json        # grader scores (results / eval_hook.results)
          expected.json     # grader-only case expectations, when present
          case.json         # the EvalCase that produced this run

    Trajectory source - exactly one is used, in this priority:

    * ``recorder`` - a :class:`looplet.provenance.TrajectoryRecorder`
      (full fidelity incl. LLM calls). Preferred for a freshly captured
      run.
    * ``context`` - an :class:`EvalContext` serialised to the trajectory
      shape. Use when promoting an online run.
    * ``eval_hook.context`` - falls back to the hook's captured context
      (this is what :func:`promote_to_offline` rides on).

    Args:
        recorder: trajectory source (see above).
        context: trajectory source (see above).
        eval_hook: optional :class:`EvalHook` whose ``artifacts``,
            ``results`` (and, as a last resort, ``context``) are used.
        case: optional :class:`EvalCase` - written as ``case.json`` so
            the run directory is self-describing.
        results: optional explicit grader results, overriding
            ``eval_hook.results`` (e.g. when graders were run offline).
        redact: optional evidence redactor, including when no recorder is used.

    Returns the run directory path. Raises :class:`ValueError` when no
    trajectory source is available.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    # Resolve the trajectory source.
    src_context = context
    if src_context is None and eval_hook is not None:
        src_context = getattr(eval_hook, "context", None)
    if recorder is None and src_context is None:
        raise ValueError(
            "save_eval_run needs a trajectory source: pass recorder=, context=, "
            "or an eval_hook with a captured .context (run the loop first)."
        )
    begin_artifact_write(root)
    grader_expected: dict[str, Any] | None = None
    if case is not None:
        grader_expected = dict(case.expected)
    elif eval_hook is not None and getattr(eval_hook, "expected", None) is not None:
        grader_expected = dict(eval_hook.expected)

    raw_recorder_redact = getattr(recorder, "_redact", None) if recorder is not None else None
    recorder_redact = redact or (
        cast(Callable[[str], str], raw_recorder_redact) if callable(raw_recorder_redact) else None
    )
    has_model_calls = False

    if recorder is not None:
        # Full-fidelity record (trajectory.json + steps/ + LLM calls).
        merged_before_save = src_context is not None and _merge_context_into_recorder(
            recorder,
            src_context,
            expected=grader_expected,
        )
        save_parameters = inspect.signature(recorder.save).parameters
        if "publish_descriptor" in save_parameters:
            recorder.save(root, publish_descriptor=False)
        else:
            recorder.save(root)
        has_model_calls = (root / "manifest.jsonl").is_file()
        if src_context is not None and not merged_before_save:
            _merge_context_into_saved_trajectory(
                root,
                src_context,
                expected=grader_expected,
                redact=recorder_redact,
            )
    elif src_context is not None:
        traj = _eval_context_to_trajectory(src_context, expected=grader_expected)
        _write_json(root / "trajectory.json", traj, redact=recorder_redact)
        steps_dir = root / "steps"
        steps_dir.mkdir(exist_ok=True)
        for old_step in steps_dir.glob("step_*.json"):
            old_step.unlink()
        for i, step in enumerate(traj["steps"]):
            _write_json(steps_dir / f"step_{i:02d}.json", step, redact=recorder_redact)
    else:  # pragma: no cover - guarded above
        raise AssertionError("trajectory source disappeared during save")

    # Outcome artifacts - the piece TrajectoryRecorder never wrote.
    if eval_hook is not None:
        artifacts = dict(getattr(eval_hook, "artifacts", {}) or {})
    elif src_context is not None:
        artifacts = dict(src_context.artifacts or {})
    else:
        artifacts = {}
    _write_json(root / "artifacts.json", artifacts, redact=recorder_redact)

    # Grader scores.
    score_objs: list[Any] = (
        results
        if results is not None
        else (list(getattr(eval_hook, "results", []) or []) if eval_hook is not None else [])
    )
    _write_json(
        root / "evals.json",
        [r.to_dict() for r in score_objs],
        redact=recorder_redact,
    )

    if grader_expected is not None:
        _write_json(root / "expected.json", grader_expected, redact=recorder_redact)
    else:
        (root / "expected.json").unlink(missing_ok=True)

    # The case itself, so the directory is self-describing.
    if case is not None:
        _validate_case_id(case.id)
        _write_json(root / "case.json", case.to_dict(), redact=recorder_redact)
    else:
        (root / "case.json").unlink(missing_ok=True)

    components = ["artifacts", "eval_results", "trajectory"]
    if has_model_calls:
        components.append("model_calls")
    write_artifact_descriptor(root, kind="eval_run", components=components)
    return root


def promote_to_offline(
    directory: str | Path,
    *,
    eval_hook: Any,
    case: "EvalCase | None" = None,
    results: list["EvalResult"] | None = None,
) -> Path:
    """Save a live eval as an offline fixture through the canonical writer.

    This compatibility alias does not promote an agent candidate.

    The headline workflow: you run an agent with an online
    :class:`EvalHook`, spot an interesting case (a regression, a slow
    path, an edge case), and want to keep it forever as an offline eval
    you can replay and re-grade. This writes the hook's captured run -
    trajectory, tool calls, outcome ``artifacts``, and scores - to
    ``directory`` in the same layout :func:`load_eval_run` reads, so the
    promoted run is indistinguishable from one captured offline.

    Equivalent to ``save_eval_run(directory, eval_hook=eval_hook, …)``
    but named for intent and loud when the hook never ran::

        hook = EvalHook(evaluators=graders, collectors=collectors)
        for _ in preset.run(llm, task=case.task, ... hooks=[hook]):
            ...
        promote_to_offline("evals/runs/regression_42", eval_hook=hook, case=case)

    Raises :class:`ValueError` if ``eval_hook`` has no captured context
    (i.e. the loop has not run yet).
    """
    if getattr(eval_hook, "context", None) is None:
        raise ValueError(
            "eval_hook has no captured context to promote; run the loop with this "
            "EvalHook attached before calling promote_to_offline()."
        )
    return save_eval_run(directory, eval_hook=eval_hook, case=case, results=results)


def _case_seed_targets(
    case: "EvalCase", root: Path, *, files_key: str = "files"
) -> list[tuple[Path, Any]]:
    files = (getattr(case, "task", {}) or {}).get(files_key) or {}
    if not isinstance(files, dict):
        raise ValueError(
            f"case {getattr(case, 'id', '?')!r}: task[{files_key!r}] must be a "
            f"{{path: contents}} dict, got {type(files).__name__}"
        )
    return [
        (_contained_child(root, relative, label=f"case {case.id!r} seed path").resolve(), contents)
        for relative, contents in files.items()
    ]


def seed_case_workspace(
    case: "EvalCase",
    directory: str | Path,
    *,
    files_key: str = "files",
) -> Path:
    """Materialise a case's seed files into a workspace directory.

    Closes the last host-glue gap the coder dogfood exposed: a case can
    carry its starting files as data under ``case.task[files_key]`` (a
    ``{relative_path: file_contents}`` dict), and this helper writes them
    into ``directory`` (creating parent dirs) so a runner no longer
    hand-rolls the seeding. Formalises ``task["files"]`` as the seed
    convention. Returns the directory.

    Missing / empty ``files`` is a no-op (the directory is still
    created), so cases that operate on a pre-existing repo work too.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for target, contents in _case_seed_targets(case, root, files_key=files_key):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents if isinstance(contents, str) else str(contents))
    return root


def load_eval_run(directory: str | Path) -> "EvalRunRecord":
    """Read back one persisted eval-case run - trajectory + artifacts + scores.

    The READ mechanism for inspecting *exactly what happened* on each
    eval case. Returns an :class:`EvalRunRecord` bundling:

    * ``context`` - an :class:`EvalContext` (full step trajectory +
      outcome ``artifacts``), built via
      :meth:`EvalContext.from_trajectory_dir`, so the **same** graders
      that scored the run live can re-score it offline.
    * ``results`` - the grader scores persisted in ``evals.json``.
    * ``case`` - the :class:`EvalCase` from ``case.json`` when present.

    Pair with :func:`save_eval_run`. A directory missing
    ``trajectory.json`` raises :class:`FileNotFoundError` (loud).
    Unknown saved case fields remain ignored for format compatibility.
    """
    root = Path(directory)
    read_artifact_descriptor(root, expected_kinds=("eval_run",))
    context = EvalContext.from_trajectory_dir(root)

    results: list[EvalResult] = []
    evals_path = root / "evals.json"
    if evals_path.exists():
        try:
            raw = json.loads(evals_path.read_text())
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid evals.json in {root}: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError(
                f"Invalid evals.json in {root}: expected list, got {type(raw).__name__}"
            )
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Invalid evals.json in {root}: result {index} must be an object")
            fields = {
                key: value for key, value in item.items() if key in EvalResult.__dataclass_fields__
            }
            try:
                results.append(EvalResult(**fields))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid evals.json in {root}: result {index}: {exc}") from exc

    collector_errors = [
        result.to_dict()
        for result in results
        if result.name.startswith("collector:") and result.label == "error"
    ]
    if collector_errors:
        context.metadata["eval_collector_errors"] = collector_errors

    case: EvalCase | None = None
    case_path = root / "case.json"
    if case_path.exists():
        try:
            cd = json.loads(case_path.read_text())
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid case.json in {root}: {exc}") from exc
        if not isinstance(cd, dict):
            raise ValueError(
                f"Invalid case.json in {root}: expected object, got {type(cd).__name__}"
            )
        case = EvalCase.from_dict(
            {key: value for key, value in cd.items() if key in EvalCase.__dataclass_fields__}
        )
        context_expected = context.task.get("expected")
        if case.expected and context_expected is not None and context_expected != case.expected:
            raise ValueError(f"Conflicting grader expectations in {root}")
        if case.expected and context_expected is None:
            # Backward compatibility for run directories written before
            # grader-only expected data had its own trajectory field.
            context.task["expected"] = dict(case.expected)

    return EvalRunRecord(case=case, context=context, results=results, directory=root)


def _select_eval_cases(
    available_cases: list[EvalCase],
    case_ids: list[str] | None,
    *,
    case_marks: list[str] | None = None,
) -> list[EvalCase]:
    wanted = set(case_ids) if case_ids is not None else None
    if wanted is not None:
        available = {case.id for case in available_cases}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(
                f"Unknown eval case id(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(available)) or '(none)'}"
            )
    wanted_marks = set(case_marks) if case_marks is not None else None
    if wanted_marks is not None:
        available_marks = {mark for case in available_cases for mark in case.marks}
        unknown_marks = sorted(wanted_marks - available_marks)
        if unknown_marks:
            raise ValueError(
                f"Unknown eval case mark(s): {', '.join(unknown_marks)}; "
                f"available: {', '.join(sorted(available_marks)) or '(none)'}"
            )
    selected = [
        case
        for case in available_cases
        if (wanted is None or case.id in wanted)
        and (wanted_marks is None or wanted_marks.intersection(case.marks))
    ]
    if not selected:
        raise ValueError("No eval cases selected")
    return selected


def run_cartridge_evals(
    cartridge_dir: str | Path,
    *,
    llm: Any,
    output_dir: str | Path | None = None,
    runtime: dict[str, Any] | None = None,
    max_steps: int | None = None,
    judge_llm: "LLMBackend | None" = None,
    workspace_key: str = "project_root",
    cases: list[str] | None = None,
    case_marks: list[str] | None = None,
    capture: bool = True,
    redact: Callable[[str], str] | None = None,
    redact_upstream: bool = True,
) -> list["EvalRunRecord"]:
    """Run a cartridge against its shipped eval cases, end to end.

    The one call that ties the whole "evals ship with the agent
    version" story together: for each case shipped under
    ``<cartridge>/evals/`` it seeds a sandbox from the case's
    ``task["files"]``, loads the cartridge as a live agent, runs it with
    an online :class:`EvalHook` (graders + collectors discovered from the
    same ``evals/`` slot), and returns one :class:`EvalRunRecord` per
    case. When ``output_dir`` is given, each run is also persisted
    (trajectory + artifacts + scores + case) via :func:`save_eval_run`
    so it doubles as an offline fixture.

    Args:
        cartridge_dir: the cartridge to evaluate (must contain ``evals/``).
        llm: the backend driving the agent (any object with the
            :class:`looplet.types.LLMBackend` surface).
        output_dir: when set, persist ``<output_dir>/<case_id>/`` per
            case; the seeded sandbox lives at ``…/<case_id>/workspace``.
            Existing selected case paths raise ``FileExistsError`` before
            any case executes. Choose a fresh directory for each run.
            When ``None``, sandboxes are throwaway tempdirs.
        runtime: base runtime dict merged into every case's load
            (the per-case sandbox path is injected under
            ``workspace_key``, defaulting to the canonical
            ``project_root``).
        max_steps: optional per-case override of the cartridge's
            ``max_steps``.
        judge_llm: optional LLM for LLM-as-judge graders.
        workspace_key: runtime key the cartridge reads for its working
            directory (``project_root`` by convention).
        cases: optional list of case ids to run (default: all).
        case_marks: optional case marks; match any mark, then intersect with cases.
        capture: capture agent model calls when output_dir is set (default: True).
            False retains a grading-only record with no captured-response replay.
        redact: optional provenance redactor for captured/persisted evidence.
        redact_upstream: scrub model inputs as well as recorded evidence by default.

    Each case's ``expected`` mapping is injected into the evaluator's
    context only after the loop ends; it is not included in the task the
    agent sees. This runner owns evaluation: embedded :class:`EvalHook`
    observers are replaced by the case bundle's hook, so collectors run
    once. Other cartridge hooks remain unchanged.

    Returns one :class:`EvalRunRecord` per executed case (``context`` is
    the online run's :class:`EvalContext`, ``directory`` is the persisted
    dir when ``output_dir`` is set, else the sandbox).
    """
    from looplet.cartridge import cartridge_to_preset  # noqa: PLC0415 - evals→cartridge is allowed

    cdir = Path(cartridge_dir)
    base_runtime = dict(runtime or {})
    overview = load_cartridge_evals(cdir, runtime=base_runtime, strict=True)
    selected = _select_eval_cases(overview.cases, cases, case_marks=case_marks)
    if not overview.graders:
        raise ValueError("No graders selected")

    run_dirs: dict[str, Path] = {}
    if output_dir is not None:
        output_root = Path(output_dir)
        for case in selected:
            run_dir = _contained_child(
                output_root,
                _validate_case_id(case.id),
                label="eval run directory",
            )
            if run_dir.exists() or run_dir.is_symlink():
                raise FileExistsError(
                    f"Eval output already exists: {run_dir}. "
                    "Choose a new output_dir (CLI: --out) to preserve saved runs."
                )
            run_dirs[case.id] = run_dir

    records: list[EvalRunRecord] = []
    for case in selected:
        if "expected" in case.task:
            raise ValueError(
                f"Eval case {case.id!r} puts reserved grader data in task['expected']; "
                "use the top-level case expected field"
            )
        if output_dir is not None:
            run_dir = run_dirs[case.id]
            run_dir.mkdir(parents=True, mode=0o700)
            begin_artifact_write(run_dir)
            sandbox = run_dir / "workspace"
        else:
            run_dir = None
            sandbox = Path(tempfile.mkdtemp(prefix=f"evalcase_{case.id}_"))
        seed_case_workspace(case, sandbox)

        case_runtime = dict(base_runtime)
        case_runtime[workspace_key] = str(sandbox)

        bundle = load_cartridge_evals(cdir, runtime=case_runtime, strict=True)
        preset = cartridge_to_preset(cdir, runtime=case_runtime)
        if max_steps is not None:
            preset.config.max_steps = max_steps
            if hasattr(preset.state, "max_steps"):
                preset.state.max_steps = max_steps

        hook = EvalHook(
            evaluators=bundle.graders,
            collectors=bundle.collectors,
            judge_llm=judge_llm,
            expected=case.expected,
        )
        preset.hooks = [existing for existing in preset.hooks if not isinstance(existing, EvalHook)]
        preset.hooks.append(hook)

        task = {k: v for k, v in (case.task or {}).items() if k != "files"}
        recorder = None
        run_llm = llm
        if capture and run_dir is not None:
            from looplet.provenance import ProvenanceSink  # noqa: PLC0415

            sink = ProvenanceSink(dir=run_dir, redact=redact, redact_upstream=redact_upstream)
            run_llm = sink.wrap_llm(llm)
            recorder = sink.trajectory_hook()
        elif redact is not None:
            from looplet.provenance import RecordingLLMBackend  # noqa: PLC0415

            run_llm = RecordingLLMBackend(llm, redact=redact, redact_upstream=redact_upstream)
        failed = False
        interrupted_context = None
        try:
            for _ in preset.run(
                run_llm, task=task, extra_hooks=[recorder] if recorder is not None else []
            ):
                pass
        except BaseException as exc:
            failed = True
            interrupted_context = hook.context or EvalContext(
                steps=list(preset.state.steps), task=task, stop_reason="error"
            )
            interrupted_context.metadata["eval_run_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            try:
                final_context = interrupted_context or hook.context
                if final_context is not None:
                    state_metadata = getattr(preset.state, "metadata", {})
                    if isinstance(state_metadata, dict):
                        for key in (
                            "run_status",
                            "run_phase",
                            "termination_reason",
                            "llm_calls",
                            "usage_total",
                            "looplet_run_stats",
                        ):
                            if key in state_metadata:
                                final_context.metadata[key] = state_metadata[key]
                    envelope = getattr(preset.state, "run_envelope", None)
                    if envelope is not None:
                        final_context.metadata["run_envelope"] = envelope.to_dict()
                    stats = getattr(preset.state, "_looplet_run_stats", None)
                    if isinstance(stats, dict):
                        final_context.metadata["looplet_run_stats"] = dict(stats)
                    if failed:
                        final_context.metadata.update(run_status="failed", run_phase="terminal")
                if run_dir is not None:
                    save_eval_run(
                        run_dir,
                        recorder=recorder,
                        context=interrupted_context,
                        eval_hook=hook,
                        case=case,
                        redact=redact,
                    )
            finally:
                preset.close()
                if failed and run_dir is None:
                    shutil.rmtree(sandbox, ignore_errors=True)
                    for previous in records:
                        previous.cleanup()

        if run_dir is not None:
            directory = run_dir
        else:
            directory = sandbox

        # ``on_loop_end`` always sets the context once the loop runs; fall
        # back to an empty context defensively so the record type stays clean.
        ctx = hook.context if hook.context is not None else EvalContext(steps=[])
        records.append(
            EvalRunRecord(
                case=case,
                context=ctx,
                results=list(hook.results),
                directory=directory,
                cleanup_directory=None if run_dir is not None else sandbox,
            )
        )
    return records


def pytest_param_cases(cases: list["EvalCase"]) -> list[Any]:
    """Wrap cases for ``@pytest.mark.parametrize``.

    Each case becomes a ``pytest.param(case, id=case.id, marks=[…])``
    so pytest's ``-k <id>``, ``-m <mark>``, and report grouping all
    work out of the box. Falls back to plain cases (with no marks) if
    pytest is not importable.

    Example::

        import pytest
        from looplet.evals import load_cases, pytest_param_cases, eval_run

        CASES = load_cases("evals/cases")

        @pytest.mark.parametrize("case", pytest_param_cases(CASES))
        def test_coder(case, my_agent):
            ctx = my_agent.run(case)
            results = eval_run([eval_tests_passed], ctx)
            assert all(r.passed for r in results), "\\n".join(r.pretty() for r in results)
    """
    try:
        import pytest  # type: ignore
    except ImportError:
        return list(cases)
    return [
        pytest.param(c, id=c.id, marks=[getattr(pytest.mark, m) for m in c.marks]) for c in cases
    ]


def parametrize_cases(
    path: str | Path,
    *,
    argname: str = "case",
) -> Callable:
    """Pytest decorator: load cases from ``path`` and parametrize over them.

    Equivalent to::

        @pytest.mark.parametrize(
            argname, pytest_param_cases(load_cases(path))
        )

    but as a single import. Marks declared on each case (``smoke``,
    ``regression``, …) carry through, so ``-k <id>`` and ``-m <mark>``
    work as usual.

    Example::

        from looplet import parametrize_cases

        @parametrize_cases("evals/cases")
        def test_coder(case, my_agent):
            ctx = my_agent.run(case)
            assert_evals_pass(ctx, "evals/")
    """
    import pytest  # type: ignore

    return pytest.mark.parametrize(argname, pytest_param_cases(load_cases(path)))


# ── Discovery ────────────────────────────────────────────────────


@functools.cache
def _discover_cached(path: str) -> tuple[Callable, ...]:
    return tuple(eval_discover(path, strict=True))


def assert_evals_pass(
    ctx: "EvalContext",
    evals: "str | Path | list[Callable]",
    *,
    judge_llm: "LLMBackend | None" = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    threshold: float = 0.5,
    version: int = 2,
) -> None:
    """Run ``evals`` against ``ctx`` and assert nothing failed.

    Convenience wrapper that collapses the standard
    ``run → filter failed → assert with pretty failures`` idiom into
    one call. ``evals`` may be a list of evaluators or a path that's
    forwarded to :func:`eval_discover` (the discovery is cached, so
    calling this in a parametrized test does not re-import on every
    case).

    Raises :class:`AssertionError` with the shared assessment's failures.
    Version 2 rejects empty suites and gates without verdict-producing checks.
    """
    if isinstance(evals, (str, Path)):
        evaluators: list[Callable] = list(_discover_cached(str(Path(evals))))
    else:
        evaluators = list(evals)
    eval_assess(
        evaluators,
        ctx,
        judge_llm=judge_llm,
        include=include,
        exclude=exclude,
        threshold=threshold,
        version=version,
    ).assert_passed()


def eval_discover(
    path: str | Path,
    *,
    pattern: str = "eval_*.py",
    prefix: str = "eval_",
    strict: bool = False,
) -> list[Callable]:
    """Find evaluator functions in files matching ``pattern``.

    Discovers all functions whose name starts with ``prefix`` in
    all Python files matching ``pattern`` under ``path``. Works
    like pytest's test collection - no registration needed.

    Args:
        path: File or directory to search.
        pattern: Glob pattern for eval files (default: ``eval_*.py``).
        prefix: Function name prefix (default: ``eval_``).
        strict: Raise when a matching module cannot be loaded instead of
            logging and returning a partial suite.

    Returns:
        List of callable evaluator functions.
    """
    return _discover_functions(path, patterns=(pattern,), prefixes=(prefix,), strict=strict)[prefix]


def _discover_functions(
    path: str | Path, *, patterns: tuple[str, ...], prefixes: tuple[str, ...], strict: bool
) -> dict[str, list[Callable]]:
    root = Path(path)
    files = (
        [root]
        if root.is_file()
        else sorted({file for pattern in patterns for file in root.rglob(pattern)})
    )

    evaluators: dict[str, list[Callable]] = {prefix: [] for prefix in prefixes}
    seen_names: dict[tuple[str, str], Path] = {}
    for fpath in files:
        module_name = "_eval_" + hashlib.sha256(str(fpath.resolve()).encode()).hexdigest()[:16]
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                fpath,
            )
            if spec is None or spec.loader is None:
                if strict:
                    raise RuntimeError("no Python module loader available")
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod.__name__] = mod
            spec.loader.exec_module(mod)
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                prefix = next((value for value in prefixes if name.startswith(value)), None)
                if prefix is None:
                    continue
                # Only pick up functions defined in THIS module, not
                # re-exports (e.g. `from looplet import eval_mark`
                # silently turns the decorator itself into a discovered
                # eval).  Unwrap decorator chains first.
                target = inspect.unwrap(obj)
                if getattr(target, "__module__", None) != mod.__name__:
                    continue
                result_name = obj.__name__
                identity = (prefix, result_name)
                if strict and identity in seen_names:
                    raise RuntimeError(
                        f"duplicate evaluator name {result_name!r} in "
                        f"{seen_names[identity]} and {fpath}"
                    )
                seen_names[identity] = fpath
                evaluators[prefix].append(obj)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(module_name, None)
            if strict:
                raise RuntimeError(f"Failed to load eval file {fpath}: {exc}") from exc
            logger.warning("Failed to load eval file: %s", fpath, exc_info=True)

    return evaluators


# ── Runner ───────────────────────────────────────────────────────


def _recorded_collector_errors(ctx: EvalContext) -> list[EvalResult]:
    """Restore host observation failures before a grader can mutate context."""
    raw = ctx.metadata.get("eval_collector_errors", [])
    if not isinstance(raw, list):
        return [
            EvalResult(
                name="collector:recorded_errors",
                label="error",
                explanation="recorded collector errors must be a list",
            )
        ]
    failures: list[EvalResult] = []
    for item in raw:
        try:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"].startswith("collector:")
                or item.get("label") != "error"
            ):
                raise ValueError("expected a named collector error")
            failures.append(
                EvalResult(
                    **{
                        key: value
                        for key, value in item.items()
                        if key in EvalResult.__dataclass_fields__
                    }
                )
            )
        except (TypeError, ValueError) as exc:
            failures.append(
                EvalResult(
                    name="collector:recorded_errors",
                    label="error",
                    explanation=f"invalid recorded collector failure: {exc}",
                )
            )
    return failures


def eval_run(
    evaluators: list[Callable],
    ctx: EvalContext,
    *,
    judge_llm: LLMBackend | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[EvalResult]:
    """Run evaluators against an :class:`EvalContext`.

    Each evaluator is called with ``(ctx)`` or ``(ctx, llm)`` depending
    on its signature. Returns a list of :class:`EvalResult` in the
    same order as ``evaluators``.

    Args:
        evaluators: Functions to run (discovered via :func:`eval_discover`
            or passed directly).
        ctx: The evaluation context (trajectory + task + output).
        judge_llm: Optional LLM backend for LLM-as-judge evaluators.
            Only passed to evaluators whose signature includes an
            ``llm`` parameter.
        include: Only run evals with these marks (via ``@eval_mark``).
        exclude: Skip evals with these marks.
    """
    collector_errors = _recorded_collector_errors(ctx)
    filtered = _filter_evals(evaluators, include, exclude)
    results: list[EvalResult] = []
    for fn in filtered:
        name = fn.__name__
        t0 = time.time()
        try:
            sig = inspect.signature(fn)
            if "llm" in sig.parameters:
                if judge_llm is None:
                    logger.warning(
                        "Eval %s requires llm but no judge_llm provided; skipping",
                        name,
                    )
                    results.append(
                        EvalResult(
                            name=name,
                            label="skipped",
                            explanation="requires judge_llm",
                        )
                    )
                    continue
                raw = fn(ctx, judge_llm)
            else:
                raw = fn(ctx)
            result = EvalResult.from_return(raw, name=name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Eval %s raised: %s", name, e, exc_info=True)
            result = EvalResult(name=name, label="error", explanation=str(e))
        result.duration_ms = (time.time() - t0) * 1000
        results.append(result)
    results.extend(collector_errors)
    return results


def eval_assess(
    evaluators: list[Callable],
    ctx: EvalContext,
    *,
    judge_llm: LLMBackend | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    threshold: float = 0.5,
    version: int = 2,
    case: EvalCase | None = None,
) -> EvalReport:
    """Grade an observed outcome and assess it without running an agent or collector."""
    _validate_eval_policy(threshold=threshold, version=version)
    records, manifest = _rescore_eval_records(
        [EvalRunRecord(case, ctx, [], Path("run"))],
        evaluators,
        judge_llm=judge_llm,
        include=include,
        exclude=exclude,
    )
    return EvalReport.from_records(
        records, grader_manifest=manifest, threshold=threshold, version=version
    )


def _rescore_eval_records(
    records: list[EvalRunRecord],
    evaluators: list[Callable],
    *,
    judge_llm: LLMBackend | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[list[EvalRunRecord], list[dict[str, Any]]]:
    selected = _filter_evals(evaluators, include, exclude)
    selected_names = {fn.__name__ for fn in selected}
    manifest = _eval_grader_manifest(
        [
            fn
            for fn in evaluators
            if fn.__name__ in selected_names or _REQUIRED_EVAL_MARK in _get_marks(fn)
        ]
    )
    graded = []
    for record in records:
        results = eval_run(selected, record.context, judge_llm=judge_llm)
        record.context.metadata["eval_grader_manifest"] = deepcopy(manifest)
        graded.append(record._replace(results=results))
    return graded, manifest


def _format_summary(results: list[EvalResult]) -> str:
    """One-line summary of eval results."""
    scored = [r for r in results if r.score is not None]
    labeled = [r for r in results if r.label and r.score is None]
    parts = []
    if scored:
        avg = sum(r.score or 0.0 for r in scored) / len(scored)
        parts.append(f"{len(scored)} scored (avg {avg:.2f})")
    if labeled:
        parts.append(f"{len(labeled)} labeled")
    errors = [r for r in results if r.label == "error"]
    if errors:
        parts.append(f"{len(errors)} errors")
    return ", ".join(parts) if parts else "no results"


# ── Hook ─────────────────────────────────────────────────────────


class EvalHook:
    """LoopHook that runs evaluators at the end of each agent run.

    Builds :class:`EvalContext` from the loop's state, runs all
    evaluators, and stores results for :meth:`summary` / :meth:`save`.

    Usage::

        hook = EvalHook(
            evaluators=[my_eval_fn, my_other_eval],
            judge_llm=my_judge_model,  # optional
            collectors=[gather_test_results, gather_diff],  # outcome data
            verbose=True,              # print scores live
        )
        for step in composable_loop(..., hooks=[hook]):
            ...
        print(hook.summary())
        save_eval_run("eval-runs/run_1", eval_hook=hook)

    Collectors are callables ``(state) -> dict[str, Any]`` that run
    once at end-of-loop and merge their return values into
    :attr:`EvalContext.artifacts`. Use them to grade outcomes (tests
    passing, files modified, repo state) instead of grepping the
    trajectory. A collector that raises or returns a non-dict never
    breaks the agent run, but is retained as an ``error`` result so an
    eval CLI cannot report a false green.

    ``expected`` is optional grader-only case data. It is exposed as
    ``ctx.task["expected"]`` when this hook builds the context, but is
    never added to the task sent through the agent loop.
    """

    def __init__(
        self,
        evaluators: list[Callable],
        *,
        judge_llm: LLMBackend | None = None,
        collectors: list[Callable[[Any], dict[str, Any]]] | None = None,
        expected: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.evaluators = evaluators
        self.judge_llm = judge_llm
        self.collectors = list(collectors) if collectors else []
        self.expected = dict(expected) if expected is not None else None
        self.verbose = verbose
        self._results: list[EvalResult] = []
        self._task: dict[str, Any] = {}
        self._artifacts: dict[str, Any] = {}
        self._context: EvalContext | None = None
        self._grader_manifest: list[dict[str, Any]] = []

    def to_config(self) -> dict:
        """Cartridge round-trip: emit ``evaluators`` (and ``collectors``
        when present) as ``@ref`` strings.

        Original resource ref names are preserved when the evaluator /
        collector lists were produced by a workspace resource builder
        (e.g. ``resources/sql_evaluators.py`` round-trips as
        ``{"evaluators": "@sql_evaluators"}``). Otherwise the writer
        emits the generic ``"@evaluators"`` / ``"@collectors"`` names.
        """
        from looplet.refs import resource_ref_for  # noqa: PLC0415

        ev_ref = resource_ref_for(self.evaluators)
        cfg: dict[str, Any] = {"evaluators": ev_ref or "@evaluators"}
        if self.collectors:
            co_ref = resource_ref_for(self.collectors)
            cfg["collectors"] = co_ref or "@collectors"
        if self.verbose:
            cfg["verbose"] = True
        return cfg

    @property
    def results(self) -> list[EvalResult]:
        """Eval results from the most recent run."""
        return list(self._results)

    @property
    def artifacts(self) -> dict[str, Any]:
        """Outcome data gathered by collectors during the most recent run."""
        return dict(self._artifacts)

    @property
    def context(self) -> "EvalContext | None":
        """The :class:`EvalContext` built at the end of the most recent run.

        This is the ONLINE counterpart of
        :attr:`EvalRunRecord.context` (the offline read side): it carries
        the full step trajectory, tool sequence, final output, and
        outcome ``artifacts`` of the live run, so you can inspect tool
        calls and re-grade an online run with the *same* surface used
        for persisted offline runs. ``None`` until the loop ends.

        Promote it to a durable offline fixture with
        :func:`promote_to_offline` (or :func:`save_eval_run`).
        """
        return self._context

    def summary(self) -> str:
        """One-line summary of eval results."""
        return _format_summary(self._results)

    def assessment(self, *, threshold: float = 0.5, version: int = 2) -> EvalReport:
        """Assess the last run without repeating collection or grading."""
        records = (
            [EvalRunRecord(None, self._context, list(self._results), Path("run"))]
            if self._context is not None
            else []
        )
        return EvalReport.from_records(
            records,
            grader_manifest=self._grader_manifest,
            threshold=threshold,
            version=version,
        )

    def report(self) -> str:
        """Multi-line formatted report."""
        if not self._results:
            return "No eval results."
        lines = [r.pretty() for r in self._results]
        lines.append(f"\n{'overall':40s} {_format_summary(self._results)}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write the deprecated standalone report for compatibility.

        Use :func:`save_eval_run` for durable, reloadable evidence.
        """
        warnings.warn(
            "EvalHook.save() writes a deprecated standalone report; "
            "use save_eval_run(directory, eval_hook=hook) for durable evidence.",
            DeprecationWarning,
            stacklevel=2,
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        task = dict(self._task)
        if self.expected is not None:
            task.pop("expected", None)
        data: dict[str, Any] = {
            "task": task,
            "results": [r.to_dict() for r in self._results],
            "summary": _format_summary(self._results),
        }
        if self.expected is not None:
            data["expected"] = dict(self.expected)
        if self._artifacts:
            data["artifacts"] = self._artifacts
        p.write_text(json.dumps(data, indent=2, default=str))

    # ── LoopHook interface ─────────────────────────────────────

    def on_loop_end(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        llm: LLMBackend,
    ) -> int:
        """Run all evaluators after the loop finishes."""
        self._grader_manifest = _eval_grader_manifest(self.evaluators)
        steps = getattr(state, "steps", [])

        # Refresh every run: EvalHook instances are reusable and must not
        # retain the previous task when attached to another loop.
        state_task = getattr(state, "task", None)
        if isinstance(state_task, dict):
            self._task = dict(state_task)
        elif state_task is not None:
            self._task = {"description": str(state_task)}
        else:
            self._task = {}
        if self.expected is not None:
            self._task["expected"] = dict(self.expected)

        stop_reason = getattr(state, "_stop_reason", None)

        # A successfully completed run ends on its accepted terminal
        # sentinel. Its name is cartridge-defined, so do not hard-code
        # ``done``. Retain the old-name fallback for direct/manual hook
        # invocations that predate stop-reason tracking.
        final_output: dict[str, Any] = {}
        terminal_steps = steps[-1:] if stop_reason == "done" else []
        if stop_reason is None:
            terminal_steps = [
                step
                for step in reversed(steps)
                if getattr(getattr(step, "tool_call", None), "tool", "") == "done"
            ][:1]
        if terminal_steps:
            raw_output = getattr(getattr(terminal_steps[0], "tool_call", None), "args", {})
            if isinstance(raw_output, dict):
                final_output = raw_output

        log_text = ""
        if session_log is not None and hasattr(session_log, "render"):
            try:
                log_text = str(session_log.render() or "")
            except Exception:  # noqa: BLE001
                pass

        # Collectors observe and therefore never raise through the loop,
        # but harness failures remain explicit EvalResults for CI.
        artifacts: dict[str, Any] = {}
        collector_failures: list[EvalResult] = []
        for collector in self.collectors:
            collector_name = getattr(
                collector,
                "__name__",
                getattr(getattr(collector, "func", None), "__name__", type(collector).__name__),
            )
            try:
                produced = collector(state)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Eval collector %s raised; skipping",
                    collector_name,
                    exc_info=True,
                )
                collector_failures.append(
                    EvalResult(
                        name=f"collector:{collector_name}",
                        label="error",
                        explanation=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if isinstance(produced, dict):
                artifacts.update(produced)
            else:
                logger.warning(
                    "Eval collector %s returned %s; expected dict - ignored",
                    collector_name,
                    type(produced).__name__,
                )
                collector_failures.append(
                    EvalResult(
                        name=f"collector:{collector_name}",
                        label="error",
                        explanation=f"expected dict, got {type(produced).__name__}",
                    )
                )
        self._artifacts = artifacts

        state_metadata = getattr(state, "metadata", {}) or {}
        if not isinstance(state_metadata, dict):
            state_metadata = {}
        context_metadata = dict(state_metadata)
        stats = getattr(state, "_looplet_run_stats", None)
        if isinstance(stats, dict):
            context_metadata["looplet_run_stats"] = dict(stats)
        envelope = getattr(state, "run_envelope", None)
        if envelope is not None:
            context_metadata["run_envelope"] = envelope.to_dict()
        if stop_reason is not None:
            context_metadata["termination_reason"] = stop_reason
        context_metadata.pop("eval_collector_errors", None)
        if collector_failures:
            context_metadata["eval_collector_errors"] = [
                result.to_dict() for result in collector_failures
            ]

        ctx = EvalContext(
            steps=list(steps),
            task=self._task,
            final_output=final_output,
            session_log_text=log_text,
            metadata=context_metadata,
            artifacts=artifacts,
            stop_reason=stop_reason,
        )
        self._context = ctx

        self._results = eval_run(self.evaluators, ctx, judge_llm=self.judge_llm)
        ctx.metadata["eval_grader_manifest"] = deepcopy(self._grader_manifest)

        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Eval results:")
            for r in self._results:
                print(f"  {r.pretty()}")
            print(f"  {'overall':38s} {_format_summary(self._results)}")
            print(f"{'─' * 50}")

        return 0

    def pre_loop(self, state: AgentState, session_log: SessionLog, context: Any) -> None:
        """Capture the task from context for eval."""
        # The task is passed via composable_loop's task= kwarg and
        # threaded through the loop. We capture it from state or
        # context if available.
        return None

    # Protocol stubs


# ── Marks ────────────────────────────────────────────────────────


def eval_mark(*tags: str) -> Callable:
    """Tag an eval function with category marks for filtering.

    Like pytest.mark - lets you group and filter evals::

        @eval_mark("accuracy", "fast")
        def eval_answer_correct(ctx):
            ...

        @eval_mark("quality", "slow")
        def eval_reasoning_depth(ctx, llm):
            ...

        @eval_mark("required")
        def eval_release_gate(ctx):
            ...  # skipped or score < 0.5 makes the CLI fail

        # Run only "accuracy" evals:
        results = eval_run(evals, ctx, include=["accuracy"])

        # Skip "slow" evals in CI:
        results = eval_run(evals, ctx, exclude=["slow"])
    """

    def decorator(fn: Callable) -> Callable:
        fn._eval_marks = set(tags)
        return fn

    return decorator


def _get_marks(fn: Callable) -> set[str]:
    """Get eval marks from a function (empty set if unmarked)."""
    return getattr(fn, "_eval_marks", set())


def _result_is_integrity_failure(
    result: EvalResult | dict[str, Any],
    *,
    required: bool = False,
) -> bool:
    """Whether a non-numeric result makes a CLI run untrustworthy."""
    return _eval_state_fails(
        _eval_result_state(result, required=required, threshold=0.0),
        required=required,
    )


def _eval_result_state(
    result: EvalResult | dict[str, Any] | None,
    *,
    required: bool,
    threshold: float,
    collector: bool = False,
) -> str:
    """Classify one result using the stable eval-summary v1 vocabulary."""
    if result is None:
        return "missing"
    if isinstance(result, EvalResult):
        score, raw_label, metrics = result.score, result.label, result.metrics
    else:
        score, raw_label, metrics = result.get("score"), result.get("label"), result.get("metrics")
    label = str(raw_label).lower() if raw_label is not None else ""
    if collector and label == "error":
        return "collector_error"
    if label == "error":
        return "grader_error"
    if label == "skipped":
        return "skipped"
    if label in {"fail", "wrong", "no"}:
        return "explicit_fail"
    if score is not None:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError, OverflowError):
            return "grader_error"
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            return "grader_error"
        if required and numeric_score < 0.5:
            return "explicit_fail"
        if numeric_score < threshold:
            return "threshold_fail"
        return "pass"
    if metrics and not label:
        return "metric_only"
    if label in {"pass", "correct", "yes"}:
        return "pass"
    return "explicit_fail"


def _required_status(state: str, *, required: bool) -> str:
    if not required:
        return "not_required"
    if state in {"pass", "threshold_fail"}:
        return "satisfied"
    return "failed"


def _eval_state_fails(state: str, *, required: bool) -> bool:
    if state in {"pass", "metric_only", "skipped"}:
        return required and state != "pass"
    return True


def _eval_summary_result(
    result: EvalResult | None,
    *,
    name: str,
    marks: set[str],
    threshold: float,
    collector: bool = False,
) -> dict[str, Any]:
    required = _REQUIRED_EVAL_MARK in marks
    state = _eval_result_state(
        result,
        required=required,
        threshold=threshold,
        collector=collector,
    )
    payload: dict[str, Any] = {
        "name": name,
        "marks": sorted(marks),
        "required": required,
        "required_status": _required_status(state, required=required),
        "state": state,
    }
    if result is not None:
        payload.update(result.to_dict())
        if state in {"collector_error", "grader_error"}:
            payload["error"] = result.explanation or result.label or "unknown error"
    return payload


def _eval_grader_manifest(graders: list[Callable]) -> list[dict[str, Any]]:
    """Freeze grader identity and marks before candidate execution begins."""
    return [
        {
            "name": grader.__name__,
            "marks": sorted(_get_marks(grader)),
            "required": _REQUIRED_EVAL_MARK in _get_marks(grader),
        }
        for grader in sorted(graders, key=lambda item: item.__name__)
    ]


def _validate_eval_policy(*, threshold: float, version: int) -> None:
    if version not in (1, 2):
        raise ValueError("Eval report version must be 1 or 2")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Eval threshold must be finite and between 0 and 1")


def _assess_eval_records(
    *,
    records: list[EvalRunRecord],
    grader_manifest: list[dict[str, Any]],
    threshold: float,
    output_dir: str | None,
    version: int,
    integrity_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Apply one verdict policy to trusted results and a frozen grader inventory."""
    _validate_eval_policy(threshold=threshold, version=version)
    manifest = deepcopy(grader_manifest)
    if version == 2 and any(
        not isinstance(item, dict)
        or not isinstance(item.get("name"), str)
        or not item["name"]
        or item["name"].startswith("collector:")
        or not isinstance(item.get("marks"), list)
        or not all(isinstance(mark, str) for mark in item["marks"])
        or not isinstance(item.get("required"), bool)
        or item["required"] != (_REQUIRED_EVAL_MARK in item["marks"])
        for item in manifest
    ):
        raise ValueError(
            "Invalid grader manifest: expected named graders with consistent required marks"
        )
    manifest_names = {item["name"] for item in manifest}
    integrity_failures = list(integrity_failures or [])
    case_reports: list[dict[str, Any]] = []
    failed = bool(integrity_failures)
    if version == 2:
        if not records:
            integrity_failures.append("No evaluation cases")
        if not manifest:
            integrity_failures.append("No graders selected")
        if len(manifest_names) != len(manifest):
            integrity_failures.append("Duplicate grader names")
        failed = bool(integrity_failures)

    inventory_invalid = version == 2 and (not manifest or len(manifest_names) != len(manifest))
    for record in records:
        before_case = len(integrity_failures)
        case_id = record.case.id if record.case is not None else record.directory.name or "run"
        case_marks = sorted(record.case.marks) if record.case is not None else []
        observed_results = list(record.results)
        for error in _recorded_collector_errors(record.context):
            if not any(result.to_dict() == error.to_dict() for result in observed_results):
                observed_results.append(error)
        by_name = {result.name: result for result in observed_results}
        if version == 2 and len(by_name) != len(observed_results):
            integrity_failures.append(f"{case_id}: duplicate grader results")
            failed = True
        results: list[dict[str, Any]] = []
        for expected in manifest:
            result = by_name.get(expected["name"])
            item = _eval_summary_result(
                result,
                name=expected["name"],
                marks=set(expected["marks"]),
                threshold=threshold,
            )
            results.append(item)
            if _eval_state_fails(item["state"], required=item["required"]):
                failed = True
            if item["state"] == "missing":
                integrity_failures.append(
                    f"{case_id}/{item['name']}: grader missing from run result"
                )
            elif item["required_status"] == "failed" and (
                version == 1 or item["state"] in {"skipped", "metric_only", "grader_error"}
            ):
                integrity_failures.append(
                    f"{case_id}/{item['name']}: required grader {item['state']}"
                )
            elif item["state"] == "grader_error":
                integrity_failures.append(f"{case_id}/{item['name']}: grader error")

        for result in observed_results:
            if result.name in manifest_names:
                continue
            collector = result.name.startswith("collector:")
            item = _eval_summary_result(
                result,
                name=result.name,
                marks=set(),
                threshold=threshold,
                collector=collector,
            )
            results.append(item)
            if _eval_state_fails(item["state"], required=False):
                failed = True
            if item["state"] == "collector_error":
                integrity_failures.append(f"{case_id}/{item['name']}: collector error")
            elif item["state"] == "grader_error":
                integrity_failures.append(f"{case_id}/{item['name']}: unexpected grader error")
            elif version == 2 and not collector:
                integrity_failures.append(f"{case_id}/{item['name']}: unexpected grader result")
                failed = True

        if version == 2 and not any(
            item["state"] in {"pass", "explicit_fail", "threshold_fail"}
            for item in results
            if item["name"] in manifest_names
        ):
            integrity_failures.append(f"{case_id}: no verdict-producing grader ran")
            failed = True

        case_valid = not inventory_invalid and len(integrity_failures) == before_case
        case_failed = any(
            _eval_state_fails(item["state"], required=item["required"]) for item in results
        ) or (version == 2 and not case_valid)
        case_report = {
            "id": case_id,
            "marks": case_marks,
            "completed": record.context.completed,
            "state": "fail" if case_failed else "pass",
            "results": results,
        }
        if version == 2:
            case_report["valid"] = case_valid
        case_reports.append(case_report)

    report = {
        "schema": _EVAL_SUMMARY_SCHEMA,
        "version": version,
        "state": "fail" if failed else "pass",
        "passed": not failed,
        "threshold": threshold,
        "grader_manifest": manifest,
        "cases": case_reports,
        "integrity_failures": integrity_failures,
        "output_dir": output_dir,
    }
    if version == 2:
        report["valid"] = not integrity_failures
    assert {item["state"] for case in case_reports for item in case["results"]} <= (
        _EVAL_RESULT_STATES
    )
    return report


def _build_eval_summary(
    *,
    records: list[EvalRunRecord],
    grader_manifest: list[dict[str, Any]],
    threshold: float,
    output_dir: str | None,
) -> dict[str, Any]:
    """Retain the eval-summary v1 wire contract through the shared assessment."""
    return EvalReport.from_records(
        records,
        grader_manifest=grader_manifest,
        threshold=threshold,
        output_dir=output_dir,
        version=_EVAL_SUMMARY_VERSION,
    ).to_dict()


# ── Batch runner ─────────────────────────────────────────────────


def eval_run_batch(
    evaluators: list[Callable],
    contexts: list[EvalContext],
    *,
    judge_llm: LLMBackend | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run evaluators across multiple trajectories.

    Like pytest parametrize - same evals, different inputs::

        contexts = [EvalContext.from_trajectory_dir(d) for d in trace_dirs]
        table = eval_run_batch(evals, contexts)
        for row in table:
            print(f"{row['name']:30s} avg={row['avg_score']:.2f}")

    Args:
        evaluators: Eval functions to run.
        contexts: List of EvalContexts (one per trajectory).
        judge_llm: Optional LLM for LLM-as-judge evals.
        include: Only run evals with these marks.
        exclude: Skip evals with these marks.

    Returns:
        List of dicts, one per evaluator, with keys:
        name, scores, avg_score, min_score, max_score, per_run.
    """
    grader_names = [fn.__name__ for fn in _filter_evals(evaluators, include, exclude)]
    records, _manifest = _rescore_eval_records(
        [EvalRunRecord(None, ctx, [], Path(f"run_{index}")) for index, ctx in enumerate(contexts)],
        evaluators,
        judge_llm=judge_llm,
        include=include,
        exclude=exclude,
    )
    return _eval_batch_table(records, grader_names)


def _eval_batch_table(
    records: list[EvalRunRecord], grader_names: list[str]
) -> list[dict[str, Any]]:
    all_results = [record.results for record in records]
    summary: list[dict[str, Any]] = []
    names = grader_names + sorted(
        {result.name for results in all_results for result in results} - set(grader_names)
    )
    for name in names:
        per_run = [
            next(
                (result for result in results if result.name == name),
                EvalResult(
                    name=name,
                    label="error" if name in grader_names else "skipped",
                    explanation="grader missing from run result" if name in grader_names else "",
                ),
            )
            for results in all_results
        ]
        scores = [result.score for result in per_run if result.score is not None]
        entry: dict[str, Any] = {
            "name": name,
            "scores": scores,
            "runs": len(records),
        }
        if scores:
            entry["avg_score"] = round(sum(scores) / len(scores), 3)
            entry["min_score"] = round(min(scores), 3)
            entry["max_score"] = round(max(scores), 3)
        entry["per_run"] = [result.to_dict() for result in per_run]
        summary.append(entry)

    return summary


def _filter_evals(
    evaluators: list[Callable],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[Callable]:
    """Filter evaluators by include/exclude marks."""
    if not include and not exclude:
        return evaluators
    result = []
    for fn in evaluators:
        marks = _get_marks(fn)
        if include and not (marks & set(include)):
            continue
        if exclude and (marks & set(exclude)):
            continue
        result.append(fn)
    return result


# ── CLI runner ───────────────────────────────────────────────────


def _init_eval_definitions(path: Path, *, python_test: bool = False) -> list[Path]:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Eval directory is not a directory: {path}")
    sources = {
        "cases/hello.json": json.dumps(
            EvalCase(
                id="hello",
                task={"goal": "Write hello to result.txt in the project directory."},
                expected={"content": "hello"},
            ).to_dict(),
            indent=2,
        )
        + "\n",
        "eval_outcome.py": (
            '"""Check independently collected output against the case contract."""\n\n'
            "from pathlib import Path\n\n"
            "from looplet import eval_mark\n"
            "from looplet.cartridge.runtime_helpers import resolve_project_root\n\n\n"
            '@eval_mark("required")\n'
            "def eval_content(ctx):\n"
            '    return ctx.artifacts["content"] == ctx.expected["content"]\n'
            "\n\n"
            "def collect_content(state, runtime):\n"
            '    target = Path(resolve_project_root(runtime)) / "result.txt"\n'
            "    try:\n"
            '        content = target.read_text(encoding="utf-8").strip()\n'
            "    except FileNotFoundError:\n"
            "        content = None\n"
            '    return {"content": content}\n'
        ),
    }
    if python_test:
        existing_tests = sorted(path.glob("test_agent*.py"))
        if existing_tests:
            raise ValueError(f"Refusing to overwrite eval definition: {existing_tests[0]}")
        if (path / "cases").is_dir() and any(path.glob("eval_*.py")):
            sources = {}
        sources[f"test_agent_{uuid.uuid4().hex[:12]}.py"] = (
            '"""Run a Python agent and check its independently collected output."""\n\n'
            "import json\n"
            "from pathlib import Path\n\n"
            "import pytest\n\n"
            "from looplet import (\n"
            "    EvalHook,\n"
            "    MockLLMBackend,\n"
            "    load_evals,\n"
            "    minimal_preset,\n"
            "    pytest_param_cases,\n"
            "    seed_case_workspace,\n"
            "    tool,\n"
            ")\n\n\n"
            "@pytest.mark.parametrize(\n"
            '    "case", pytest_param_cases(load_evals(Path(__file__).parent).cases)\n'
            ")\n"
            "def test_agent(case, tmp_path):\n"
            "    definitions = load_evals(\n"
            '        Path(__file__).parent, runtime={"project_root": str(tmp_path)}\n'
            "    )\n"
            "    seed_case_workspace(case, tmp_path)\n\n"
            '    @tool(description="Write the result in the project directory.")\n'
            "    def write_result(content: str) -> dict:\n"
            '        (tmp_path / "result.txt").write_text(content, encoding="utf-8")\n'
            '        return {"written": True}\n\n'
            '    @tool(description="Finish the task.")\n'
            "    def done() -> dict:\n"
            "        return {}\n\n"
            "    grading = EvalHook(\n"
            "        evaluators=definitions.graders,\n"
            "        collectors=definitions.collectors,\n"
            "        expected=case.expected,\n"
            "    )\n"
            "    backend = MockLLMBackend([\n"
            '        json.dumps({"tool": "write_result", "args": {"content": "hello"}}),\n'
            '        json.dumps({"tool": "done", "args": {}}),\n'
            "    ])\n"
            "    task = {\n"
            '        key: value for key, value in case.task.items() if key not in {"files", "expected"}\n'
            "    }\n"
            "    with minimal_preset(tools=[write_result, done], max_steps=2, validate=True) as agent:\n"
            "        agent.hooks.append(grading)\n"
            "        result = agent.run_result(backend, task=task)\n"
            "    assert result.completed\n"
            "    grading.assessment().assert_passed()\n"
        )
    targets = {
        _contained_child(path, relative, label="eval starter file"): content
        for relative, content in sources.items()
    }
    for target in targets:
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to overwrite eval definition: {target}")
        parent = target.parent
        while parent != path:
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Eval definition parent is not a directory: {parent}")
            parent = parent.parent
    for target, content in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(content)
    return list(targets)


def _check_eval_definitions(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    bundle = CartridgeEvals([], [], [])
    manifest: list[dict[str, Any]] = []
    try:
        bundle = load_evals(path)
        if not bundle.graders:
            errors.append("No graders selected")
        manifest = _eval_grader_manifest(bundle.graders)
        if not all(isinstance(mark, str) for item in manifest for mark in item["marks"]):
            errors.append("Grader marks must be strings")
        if any(item["name"].startswith("collector:") for item in manifest):
            errors.append("Grader names cannot use the collector: namespace")
        for case in bundle.cases:
            if "expected" in case.task:
                errors.append(f"{case.id}: put grader data in the top-level expected field")
        with tempfile.TemporaryDirectory(prefix="eval-definition-check-") as workspace:
            for case in bundle.cases:
                _case_seed_targets(case, Path(workspace))
        for role, functions in (("grader", bundle.graders), ("collector", bundle.collectors)):
            for function in functions:
                target = getattr(function, "func", function)
                name = target.__name__
                if inspect.iscoroutinefunction(function):
                    errors.append(f"{name}: async {role}s are not supported")
                signature = inspect.signature(function)
                arguments = (
                    (object(), object())
                    if role == "grader" and "llm" in signature.parameters
                    else (object(),)
                )
                try:
                    signature.bind(*arguments)
                except TypeError as exc:
                    errors.append(f"{name}: unsupported {role} signature: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return {
        "schema": "looplet.eval-definitions",
        "version": 1,
        "valid": not errors,
        "path": str(path),
        "cases": [case.id for case in bundle.cases],
        "graders": manifest,
        "collectors": [
            getattr(function, "__name__", getattr(function, "func", function).__name__)
            for function in bundle.collectors
        ],
        "errors": errors,
    }


def _eval_definitions_cli(args: list[str]) -> int:
    import argparse

    action = args[0]
    parser = argparse.ArgumentParser(
        prog=f"looplet eval {action}",
        description=(
            "Create plain Python eval definitions without replacing existing files."
            if action == "init"
            else "Import trusted definitions and check their shape without running an agent."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("evals"),
        help=(
            "Eval directory (default: evals)"
            if action == "init"
            else "Python file or eval directory (default: evals)"
        ),
    )
    if action == "init":
        parser.add_argument(
            "--python",
            dest="python_test",
            action="store_true",
            help="Add a runnable pytest example without replacing existing definitions or tests.",
        )
    else:
        parser.add_argument("--json", action="store_true", help="Emit one definition check report")
    parsed = parser.parse_args(args[1:])
    if action == "init":
        try:
            files = _init_eval_definitions(parsed.path, python_test=parsed.python_test)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for path in files:
            print(f"created {path}")
        print(f"Check: looplet eval check {shlex.quote(str(parsed.path))}")
        if parsed.python_test:
            print("Install test dependency: python -m pip install pytest")
            print(f"Test: python -m pytest {shlex.quote(str(parsed.path))}")
        return 0

    from looplet.cli import execution_output  # noqa: PLC0415

    output = sys.stdout
    with execution_output(parsed.json):
        report = _check_eval_definitions(parsed.path)
    if parsed.json:
        print(json.dumps(report, indent=2), file=output)
    else:
        print(f"{'VALID' if report['valid'] else 'INVALID'} eval definitions: {parsed.path}")
        print(
            f"  {len(report['cases'])} cases, {len(report['graders'])} graders, {len(report['collectors'])} collectors"
        )
        for error in report["errors"]:
            print(f"  - {error}")
    return 0 if report["valid"] else 1


def eval_cli(args: list[str] | None = None) -> int:
    """CLI entry point for running evals.

    Usage::

        looplet eval traces/                          # score all runs
        looplet eval traces/ --evals eval_agent.py    # specific eval file
        looplet eval traces/ --threshold 0.7          # fail if any score < 0.7
        looplet eval traces/ --include accuracy       # only accuracy evals
        looplet eval traces/ --exclude slow           # skip slow evals

        looplet eval run <cartridge>                  # run a cartridge's shipped evals
        looplet eval run <cartridge> --out runs/      # …and persist each run
        looplet eval run <cartridge> --judge          # …with LLM-as-judge graders

        looplet eval cases ls evals/cases/            # list cases
        looplet eval cases show evals/cases/foo.json  # full case dump

    Returns 0 if all evals pass threshold, 1 otherwise.
    """
    import argparse

    # Lightweight subcommand dispatch - keep the existing flat surface
    # working when the first arg isn't a recognized subcommand.
    raw = list(args) if args is not None else sys.argv[1:]
    if raw and raw[0] in {"init", "check"}:
        return _eval_definitions_cli(raw)
    if raw and raw[0] == "cases":
        return _cases_cli(raw[1:])
    if raw and raw[0] == "run":
        return _run_cartridge_cli(raw[1:])

    parser = argparse.ArgumentParser(
        prog="looplet eval",
        description=(
            "Define, check, and run evals.\n\n"
            "Start with `looplet eval init evals/`, then `looplet eval check evals/`.\n"
            "Pass a traces directory below to grade saved agent trajectories."
        ),
        epilog=(
            "subcommands:\n"
            "  init [path]              create a plain Python eval starter\n"
            "  check [path]             validate trusted eval definitions\n"
            "  run <cartridge>          run a cartridge against its shipped eval cases\n"
            "  cases ls <path>          list eval cases (one line per case)\n"
            "  cases show <path>        show a single case in full\n"
            "\n"
            "Run `looplet eval run -h` / `looplet eval cases -h` for their flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("traces", help="Directory containing trajectory dirs")
    parser.add_argument(
        "--evals", default=None, help="Eval file or directory (default: discover in cwd)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fail if any scored result < threshold (default: 0.5)",
    )
    parser.add_argument(
        "--include", nargs="*", default=None, help="Only run evals with these marks"
    )
    parser.add_argument("--exclude", nargs="*", default=None, help="Skip evals with these marks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-run details")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable assessment")
    parser.add_argument(
        "--assessment-version",
        "--summary-version",
        dest="summary_version",
        type=int,
        choices=(1, 2),
        default=2,
        help="Gate policy and JSON schema version (default: 2; use 1 for legacy behavior)",
    )

    parsed = parser.parse_args(args)
    if not math.isfinite(parsed.threshold) or not 0.0 <= parsed.threshold <= 1.0:
        print("error: --threshold must be finite and between 0 and 1")
        return 1

    from looplet.cli import execution_output  # noqa: PLC0415

    output = sys.stdout
    with execution_output(parsed.json):
        return _run_saved_evaluation(parsed, output=output)


def _run_saved_evaluation(parsed: Namespace, *, output: TextIO) -> int:
    """Rescore saved observations through the shared assessment policy."""
    # Discover evals
    eval_path = parsed.evals or "."
    try:
        evaluators = eval_discover(eval_path, strict=True)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    if not evaluators:
        print(f"No eval_* functions found in {eval_path}")
        return 1

    # Discover trajectories
    traces_root = Path(parsed.traces)
    if not traces_root.is_dir():
        print(f"Traces directory not found: {traces_root}")
        return 1

    records: list[EvalRunRecord] = []
    names: list[str] = []
    load_failures: list[str] = []
    trace_dirs = (
        [traces_root]
        if (traces_root / "trajectory.json").is_file()
        else sorted(traces_root.iterdir())
    )
    for d in trace_dirs:
        if d.is_dir() and (d / "trajectory.json").exists():
            try:
                descriptor = read_artifact_descriptor(d, expected_kinds=("eval_run", "provenance"))
                is_eval_run = (
                    descriptor["kind"] == "eval_run"
                    if descriptor is not None
                    else (d / "evals.json").is_file()
                )
                records.append(
                    load_eval_run(d)
                    if is_eval_run
                    else EvalRunRecord(None, EvalContext.from_trajectory_dir(d), [], d)
                )
                names.append(d.name)
            except Exception as e:  # noqa: BLE001
                failure = f"{d.name}: {e}"
                load_failures.append(failure)
                print(f"  ERROR {failure}")

    if not records:
        print(f"No trajectories found in {traces_root}")
        return 1

    if not parsed.json:
        print(f"Found {len(evaluators)} evals, {len(records)} trajectories\n")

    selected_names = [
        fn.__name__ for fn in _filter_evals(evaluators, parsed.include, parsed.exclude)
    ]
    records, manifest = _rescore_eval_records(
        records,
        evaluators,
        include=parsed.include,
        exclude=parsed.exclude,
    )
    if not selected_names and not manifest:
        print("No evals selected after applying include/exclude filters")
        if not parsed.json:
            return 1

    assessment = EvalReport.from_records(
        records,
        grader_manifest=manifest,
        threshold=parsed.threshold,
        version=parsed.summary_version,
        integrity_failures=[f"trajectory load failed: {failure}" for failure in load_failures]
        + ([] if manifest else ["No graders selected"]),
    )
    report = assessment.to_dict()
    if parsed.json:
        print(json.dumps(report, indent=2), file=output)
        return 0 if assessment.passed else 1

    table = _eval_batch_table(records, selected_names)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for case in report["cases"]:
        for result in case["results"]:
            by_name.setdefault(result["name"], []).append(result)
    for row in table:
        results = by_name[row["name"]]
        avg = row.get("avg_score")
        if avg is not None:
            marker = (
                "✗"
                if any(
                    _eval_state_fails(item["state"], required=item["required"]) for item in results
                )
                else "✓"
            )
            print(
                f"  {marker} {row['name']:40s} avg={avg:.2f}  "
                f"min={row.get('min_score', 0):.2f}  "
                f"max={row.get('max_score', 0):.2f}  "
                f"({row['runs']} runs)"
            )
        else:
            print(f"  - {row['name']:40s} (no scores)")

        if parsed.verbose:
            for j, run in enumerate(row.get("per_run", [])):
                label = names[j] if j < len(names) else f"run_{j}"
                score = run.get("score", " - ")
                details = run.get("details", [])
                print(f"      {label}: {score}")
                for d in details[:3]:
                    print(f"        {d}")

    scored = [r for r in table if r.get("avg_score") is not None]
    if scored:
        overall = sum(r["avg_score"] for r in scored) / len(scored)
        print(f"\n  overall: {overall:.2f}")
    print(f"  threshold: {parsed.threshold:.2f}  → {'PASS' if assessment.passed else 'FAIL'}")

    if assessment.integrity_failures:
        print("\n  integrity failures:")
        for failure in assessment.integrity_failures:
            print(f"    - {failure}")
    failed_checks = [
        f"{case['id']}/{item['name']}: {item.get('label') or item['state']}"
        for case in report["cases"]
        for item in case["results"]
        if _eval_state_fails(item["state"], required=item["required"])
    ]
    if failed_checks:
        print("\n  failed checks:")
        for failure in failed_checks:
            print(f"    - {failure}")

    if load_failures:
        print("\n  trajectory load failures:")
        for failure in load_failures:
            print(f"    - {failure}")

    return 0 if assessment.passed else 1


def _run_cartridge_cli(args: list[str]) -> int:
    """``looplet eval run <cartridge> [--out DIR] [--model M] ...``.

    Run a cartridge against its own shipped eval cases (seed → run →
    grade), printing a cases × graders matrix and optionally persisting
    each run for offline inspection.
    """
    import argparse

    from looplet.cli import add_backend_options, execution_output

    parser = argparse.ArgumentParser(
        prog="looplet eval run",
        description="Run a cartridge against its shipped eval cases.",
    )
    parser.add_argument("cartridge", help="Cartridge directory (must contain evals/).")
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument(
        "--out",
        default=None,
        help="Save cases under <out>/<case_id>/ (default: a unique .looplet/evals/<cartridge>-<id>/ directory).",
    )
    persistence.add_argument(
        "--no-save",
        action="store_true",
        help="Run without saving eval evidence or captured model calls.",
    )
    add_backend_options(parser)
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps per case.")
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Persist grades/outcomes without agent model prompts/responses or captured-response replay.",
    )
    parser.add_argument(
        "--case", action="append", default=None, help="Run only this case id (repeatable)."
    )
    parser.add_argument(
        "--case-mark",
        action="append",
        default=None,
        help="Run cases matching any selected mark (repeatable; intersects with --case).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge graders (those with an ``llm`` parameter). "
        "Reuses the agent backend unless --judge-model is given.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for LLM-as-judge graders (implies --judge; else the agent model).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fail (exit 1) if any scored grader is below this on any case (default: 0.5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable eval report",
    )
    parser.add_argument(
        "--assessment-version",
        "--summary-version",
        dest="summary_version",
        type=int,
        choices=(1, 2),
        default=2,
        help="Gate policy and JSON schema version (default: 2; use 1 for legacy behavior)",
    )
    parsed = parser.parse_args(args)
    if not math.isfinite(parsed.threshold) or not 0.0 <= parsed.threshold <= 1.0:
        print("error: --threshold must be finite and between 0 and 1", file=sys.stderr)
        return 1
    if parsed.max_steps is not None and parsed.max_steps < 1:
        print("error: --max-steps must be positive", file=sys.stderr)
        return 1

    output = sys.stdout
    with execution_output(parsed.json):
        return _run_cartridge_evaluation(parsed, output=output)


def _run_cartridge_evaluation(parsed: Namespace, *, output: TextIO) -> int:
    from looplet.cli import build_cli_backend

    cdir = Path(parsed.cartridge)
    if (cdir.is_file() and cdir.suffix == ".py") or (
        not (cdir / "cartridge.json").is_file() and (cdir / "cases").is_dir()
    ):
        print(
            "error: `eval run` expects a cartridge, not a standalone Python eval directory or file.\n"
            "For Python agents, load definitions with load_evals() and run your pytest tests.\n"
            "Use `looplet eval check <path>` to validate definitions without running them.",
            file=sys.stderr,
        )
        return 1
    if not cdir.is_dir():
        print(f"error: cartridge not found: {cdir}", file=sys.stderr)
        return 1

    try:
        overview = load_cartridge_evals(cdir, strict=True)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not overview.cases:
        print(f"error: no eval cases under {cdir / CARTRIDGE_CASES_SUBPATH}", file=sys.stderr)
        return 1
    if not overview.graders:
        print(f"error: no graders (evals/eval_*.py) under {cdir / 'evals'}", file=sys.stderr)
        return 1
    try:
        selected_cases = _select_eval_cases(
            overview.cases, parsed.case, case_marks=parsed.case_mark
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    grader_manifest = _eval_grader_manifest(overview.graders)

    try:
        llm = build_cli_backend(model=parsed.model, base_url=parsed.base_url)
        judge_llm = None
        if parsed.judge_model:
            judge_llm = build_cli_backend(model=parsed.judge_model, base_url=parsed.base_url)
        elif parsed.judge:
            judge_llm = llm
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir = None
    if not parsed.no_save:
        output_dir = (
            Path(parsed.out)
            if parsed.out is not None
            else Path.cwd()
            / ".looplet"
            / "evals"
            / f"{cdir.name.removesuffix('.cartridge')}-{uuid.uuid4().hex[:12]}"
        )

    if not parsed.json:
        print(
            f"running {cdir.name}: {len(selected_cases)} case(s), "
            f"{len(overview.graders)} grader(s), {len(overview.collectors)} collector(s)"
            f"{'  [judge on]' if judge_llm is not None else ''}\n"
        )
    try:
        records = run_cartridge_evals(
            cdir,
            llm=llm,
            output_dir=output_dir,
            max_steps=parsed.max_steps,
            cases=parsed.case,
            case_marks=parsed.case_mark,
            judge_llm=judge_llm,
            capture=not parsed.no_capture,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"error: eval workspace or evidence output failed: {exc}\n"
            "Choose a writable --out directory or use --no-save to disable evidence persistence.",
            file=sys.stderr,
        )
        return 1

    report = EvalReport.from_records(
        records,
        grader_manifest=grader_manifest,
        threshold=parsed.threshold,
        output_dir=str(output_dir) if output_dir is not None else None,
        version=parsed.summary_version,
    ).to_dict()
    grader_names = [item["name"] for item in report["grader_manifest"]]
    if not parsed.json:
        rows = [["case", *grader_names]]
        for record in records:
            scores = {result.name: result for result in record.results}
            cells = [record.case.id if record.case is not None else "?"]
            for grader_name in grader_names:
                result = scores.get(grader_name)
                if result is None:
                    cells.append("-")
                elif result.score is not None:
                    cells.append(f"{result.score:.2f}")
                elif result.metrics:
                    cells.append(
                        "; ".join(
                            f"{metric_name}={metric_value:g}"
                            for metric_name, metric_value in result.metrics.items()
                        )
                    )
                else:
                    cells.append(result.label or "-")
            rows.append(cells)
        widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
        for row_number, row in enumerate(rows):
            print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
            if row_number == 0:
                print("-+-".join("-" * width for width in widths))

        required_names = {
            item["name"]
            for item in report["grader_manifest"]
            if "required" in item.get("marks", [])
        }
        printed_failure_details = False
        for record in records:
            for result in record.results:
                required = result.name in required_names
                state = _eval_result_state(result, required=required, threshold=parsed.threshold)
                if not _eval_state_fails(state, required=required):
                    continue
                if not printed_failure_details:
                    print("\nfailure details:")
                    printed_failure_details = True
                case_id = record.case.id if record.case is not None else "?"
                print(f"  {case_id} / {result.name}: {state}")
                if result.explanation:
                    print(f"    {result.explanation}")
                for detail in result.details:
                    print(f"    {detail}")

    failed = not report["passed"]
    if parsed.json:
        print(json.dumps(report, indent=2), file=output)
    elif output_dir is not None:
        print(f"\npersisted {len(records)} run(s) under {output_dir}/")
        for record in records:
            print(f"Inspect: looplet inspect {shlex.quote(str(record.directory))}")
    else:
        print("\nevidence not saved (--no-save); model calls cannot be replayed")
    if report["integrity_failures"] and not parsed.json:
        print("\nintegrity failures:")
        for failure in report["integrity_failures"]:
            print(f"  - {failure}")
    if parsed.threshold > 0 and not parsed.json:
        print(f"threshold {parsed.threshold:.2f} → {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def _cases_cli(args: list[str]) -> int:
    """``looplet eval cases ls|show <path>`` - read-only case browser."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="looplet eval cases",
        description="List or show eval cases.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p_ls = sub.add_parser("ls", help="List cases (one line per case).")
    p_ls.add_argument("path", help="Eval root, case directory, or case file.")
    p_ls.add_argument(
        "--case-mark",
        action="append",
        default=None,
        help="List cases matching any selected mark (repeatable).",
    )
    p_show = sub.add_parser("show", help="Show a case in full.")
    p_show.add_argument("path", help="Eval root, case directory, or case file.")
    p_show.add_argument(
        "case_id",
        nargs="?",
        default=None,
        help="Case id to show (required when path is a directory of multiple cases).",
    )

    parsed = parser.parse_args(args)

    try:
        cases = load_cases(parsed.path)
        if parsed.action == "ls" and parsed.case_mark is not None:
            cases = _select_eval_cases(cases, None, case_marks=parsed.case_mark)
    except FileNotFoundError as e:
        print(str(e))
        return 1
    except ValueError as e:
        print(f"  error: {e}")
        return 1

    if not cases:
        print(f"No cases found in {parsed.path}")
        return 1

    if parsed.action == "ls":
        import textwrap

        for c in cases:
            marks = ",".join(c.marks) or "-"
            desc = textwrap.shorten(
                str(c.task.get("description") or c.task.get("goal") or c.notes or "").replace(
                    "\n", " "
                ),
                width=60,
                placeholder="...",
            )
            print(f"  {c.id:30s} [{marks:20s}] {desc}")
        print(f"\n  {len(cases)} case(s)")
        return 0

    # show
    by_id = {c.id: c for c in cases}
    if parsed.case_id is not None:
        selected = by_id.get(parsed.case_id)
        if selected is None:
            print(f"  no case with id {parsed.case_id!r} (have: {', '.join(by_id)})")
            return 1
    elif len(cases) == 1:
        selected = cases[0]
    else:
        print(f"  multiple cases found ({len(cases)}); pass a case_id")
        return 1

    print(json.dumps(selected.to_dict(), indent=2, sort_keys=True))
    return 0
