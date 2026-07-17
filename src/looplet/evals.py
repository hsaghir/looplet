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
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, cast

if TYPE_CHECKING:
    from looplet.session import SessionLog
    from looplet.types import AgentState, LLMBackend

__all__ = [
    "EvalCase",
    "EvalContext",
    "EvalResult",
    "EvalHook",
    "assert_evals_pass",
    "eval_discover",
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
    def completed(self) -> bool:
        """True when the agent called ``done()`` itself (not stopped by a hook)."""
        return self.stop_reason == "done"

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
        if self.label is not None and self.label.lower() in {
            "error",
            "fail",
            "wrong",
            "no",
            "skipped",
        }:
            return False
        if self.score is not None:
            return math.isfinite(self.score) and 0.0 <= self.score <= 1.0 and self.score >= 0.5
        if self.label is not None:
            return self.label.lower() in {"pass", "correct", "yes"}
        return False

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
            # Test key presence rather than truthiness so a real 0.0 score
            # is not silently downgraded to a metric-only result.
            score = next(
                (metrics[key] for key in ("score", "f1", "accuracy", "overall") if key in metrics),
                None,
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
                if k not in ("score", "f1", "accuracy", "overall")
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
        if "id" not in data:
            raise ValueError("EvalCase requires 'id'")
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_cases(path: str | Path) -> list["EvalCase"]:
    """Load eval cases from a file or directory.

    Accepts:

    * a single ``.json`` file containing one case dict, or a list of
      case dicts;
    * a single ``.jsonl`` file with one case dict per line;
    * a directory containing any mix of ``*.json`` / ``*.jsonl`` files
      (sorted by file name, then by their internal order).

    Cases are returned in deterministic order so pytest IDs are stable.
    Raises :class:`FileNotFoundError` if ``path`` doesn't exist and
    :class:`ValueError` for malformed entries (so a bad case is loud).
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such case path: {root}")

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
        for raw in _iter_case_dicts(fpath):
            case = EvalCase.from_dict(raw)
            if case.id in seen:
                raise ValueError(f"Duplicate case id {case.id!r} (in {fpath})")
            seen[case.id] = fpath
            cases.append(case)
    return cases


def _iter_case_dicts(fpath: Path):
    text = fpath.read_text()
    if fpath.suffix == ".jsonl":
        for ln, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{fpath}:{ln}: invalid JSON ({e})") from e
        return
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{fpath}: invalid JSON ({e})") from e
    if isinstance(loaded, list):
        yield from loaded
    elif isinstance(loaded, dict):
        yield loaded
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
    bound: list[Callable] = []
    for fn in raw:
        try:
            needs_runtime = "runtime" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            needs_runtime = False
        if needs_runtime:
            bound.append(functools.partial(fn, runtime=dict(runtime or {})))
        else:
            bound.append(fn)
    return bound


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
    cases = load_cartridge_cases(cartridge_dir)
    graders = eval_discover(
        evals_root,
        pattern="eval_*.py",
        prefix="eval_",
        strict=strict,
    )
    collectors = discover_collectors(evals_root, runtime=runtime, strict=strict)
    return CartridgeEvals(cases=cases, graders=graders, collectors=collectors)


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

    Returns the run directory path. Raises :class:`ValueError` when no
    trajectory source is available.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    # Resolve the trajectory source.
    src_context = context
    if src_context is None and eval_hook is not None:
        src_context = getattr(eval_hook, "context", None)
    grader_expected: dict[str, Any] | None = None
    if case is not None:
        grader_expected = dict(case.expected)
    elif eval_hook is not None and getattr(eval_hook, "expected", None) is not None:
        grader_expected = dict(eval_hook.expected)

    raw_recorder_redact = getattr(recorder, "_redact", None) if recorder is not None else None
    recorder_redact = (
        cast(Callable[[str], str], raw_recorder_redact) if callable(raw_recorder_redact) else None
    )

    if recorder is not None:
        # Full-fidelity record (trajectory.json + steps/ + LLM calls).
        merged_before_save = src_context is not None and _merge_context_into_recorder(
            recorder,
            src_context,
            expected=grader_expected,
        )
        recorder.save(root)
        if src_context is not None and not merged_before_save:
            _merge_context_into_saved_trajectory(
                root,
                src_context,
                expected=grader_expected,
                redact=recorder_redact,
            )
    elif src_context is not None:
        traj = _eval_context_to_trajectory(src_context, expected=grader_expected)
        _write_json(root / "trajectory.json", traj)
        steps_dir = root / "steps"
        steps_dir.mkdir(exist_ok=True)
        for i, step in enumerate(traj["steps"]):
            _write_json(steps_dir / f"step_{i:02d}.json", step)
    else:
        raise ValueError(
            "save_eval_run needs a trajectory source: pass recorder=, context=, "
            "or an eval_hook with a captured .context (run the loop first)."
        )

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

    return root


def promote_to_offline(
    directory: str | Path,
    *,
    eval_hook: Any,
    case: "EvalCase | None" = None,
    results: list["EvalResult"] | None = None,
) -> Path:
    """Promote a live (ONLINE) eval run into a durable OFFLINE fixture.

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
    files = (getattr(case, "task", {}) or {}).get(files_key) or {}
    if not isinstance(files, dict):
        raise ValueError(
            f"case {getattr(case, 'id', '?')!r}: task[{files_key!r}] must be a "
            f"{{path: contents}} dict, got {type(files).__name__}"
        )
    for relpath, contents in files.items():
        # Resolve once after containment validation and write through that
        # exact path, so validation and the filesystem operation agree.
        target = _contained_child(
            root,
            relpath,
            label=f"case {case.id!r} seed path",
        ).resolve()
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
    """
    root = Path(directory)
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
        case = EvalCase.from_dict(cd)
        context_expected = context.task.get("expected")
        if case.expected and context_expected is not None and context_expected != case.expected:
            raise ValueError(f"Conflicting grader expectations in {root}")
        if case.expected and context_expected is None:
            # Backward compatibility for run directories written before
            # grader-only expected data had its own trajectory field.
            context.task["expected"] = dict(case.expected)

    return EvalRunRecord(case=case, context=context, results=results, directory=root)


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

    Each case's ``expected`` mapping is injected into the evaluator's
    context only after the loop ends; it is not included in the task the
    agent sees.

    Returns one :class:`EvalRunRecord` per executed case (``context`` is
    the online run's :class:`EvalContext`, ``directory`` is the persisted
    dir when ``output_dir`` is set, else the sandbox).
    """
    from looplet.cartridge import cartridge_to_preset  # noqa: PLC0415 - evals→cartridge is allowed

    cdir = Path(cartridge_dir)
    base_runtime = dict(runtime or {})
    overview = load_cartridge_evals(cdir, runtime=base_runtime, strict=True)
    wanted = set(cases) if cases is not None else None
    if wanted is not None:
        available = {case.id for case in overview.cases}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(
                f"Unknown eval case id(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(available)) or '(none)'}"
            )
    selected = [c for c in overview.cases if wanted is None or c.id in wanted]

    records: list[EvalRunRecord] = []
    for case in selected:
        if "expected" in case.task:
            raise ValueError(
                f"Eval case {case.id!r} puts reserved grader data in task['expected']; "
                "use the top-level case expected field"
            )
        if output_dir is not None:
            output_root = Path(output_dir)
            run_dir = _contained_child(
                output_root,
                _validate_case_id(case.id),
                label="eval run directory",
            )
            if run_dir.is_symlink():
                run_dir.unlink()
            elif run_dir.exists():
                shutil.rmtree(run_dir)
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
        preset.hooks = list(preset.hooks) + [hook]

        task = {k: v for k, v in (case.task or {}).items() if k != "files"}
        try:
            for _ in preset.run(llm, task=task):
                pass
        finally:
            preset.close()

        if run_dir is not None:
            save_eval_run(run_dir, eval_hook=hook, case=case)
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
) -> None:
    """Run ``evals`` against ``ctx`` and assert nothing failed.

    Convenience wrapper that collapses the standard
    ``run → filter failed → assert with pretty failures`` idiom into
    one call. ``evals`` may be a list of evaluators or a path that's
    forwarded to :func:`eval_discover` (the discovery is cached, so
    calling this in a parametrized test does not re-import on every
    case).

    Raises :class:`AssertionError` listing each failed eval's
    :meth:`EvalResult.pretty` block on its own line.
    """
    if isinstance(evals, (str, Path)):
        evaluators: list[Callable] = list(_discover_cached(str(Path(evals))))
    else:
        evaluators = list(evals)
    results = eval_run(evaluators, ctx, judge_llm=judge_llm, include=include, exclude=exclude)
    failed = [r for r in results if not r.passed]
    assert not failed, "\n".join(r.pretty() for r in failed)


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
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob(pattern))

    evaluators: list[Callable] = []
    seen_names: dict[str, Path] = {}
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
                if not name.startswith(prefix):
                    continue
                # Only pick up functions defined in THIS module, not
                # re-exports (e.g. `from looplet import eval_mark`
                # silently turns the decorator itself into a discovered
                # eval).  Unwrap decorator chains first.
                target = inspect.unwrap(obj)
                if getattr(target, "__module__", None) != mod.__name__:
                    continue
                result_name = obj.__name__
                if strict and result_name in seen_names:
                    raise RuntimeError(
                        f"duplicate evaluator name {result_name!r} in "
                        f"{seen_names[result_name]} and {fpath}"
                    )
                seen_names[result_name] = fpath
                evaluators.append(obj)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(module_name, None)
            if strict:
                raise RuntimeError(f"Failed to load eval file {fpath}: {exc}") from exc
            logger.warning("Failed to load eval file: %s", fpath, exc_info=True)

    return evaluators


# ── Runner ───────────────────────────────────────────────────────


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
    return results


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
        hook.save("evals/run_1.json")

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

    def report(self) -> str:
        """Multi-line formatted report."""
        if not self._results:
            return "No eval results."
        lines = [r.pretty() for r in self._results]
        lines.append(f"\n{'overall':40s} {_format_summary(self._results)}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save eval results to a JSON file."""
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
        if stop_reason is not None:
            context_metadata["termination_reason"] = stop_reason

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
        self._results.extend(collector_failures)

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
    if isinstance(result, EvalResult):
        score = result.score
        label = result.label
    else:
        score = result.get("score")
        label = result.get("label")
    normalized = str(label).lower() if label is not None else ""
    if normalized in {"error", "fail", "wrong", "no"}:
        return True
    if normalized == "skipped":
        return required
    if score is not None:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            return True
        return required and numeric_score < 0.5
    if label is None:
        metrics = result.metrics if isinstance(result, EvalResult) else result.get("metrics", {})
        return required or not bool(metrics)
    return normalized not in {"pass", "correct", "yes"}


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
    filtered = _filter_evals(evaluators, include, exclude)
    all_results: list[list[EvalResult]] = []

    for ctx in contexts:
        results = eval_run(filtered, ctx, judge_llm=judge_llm)
        all_results.append(results)

    # Pivot: per-evaluator aggregation
    summary: list[dict[str, Any]] = []
    for i, fn in enumerate(filtered):
        scores: list[float] = [
            s
            for s in (
                all_results[j][i].score for j in range(len(contexts)) if i < len(all_results[j])
            )
            if s is not None
        ]
        entry: dict[str, Any] = {
            "name": fn.__name__,
            "scores": scores,
            "runs": len(contexts),
        }
        if scores:
            entry["avg_score"] = round(sum(scores) / len(scores), 3)
            entry["min_score"] = round(min(scores), 3)
            entry["max_score"] = round(max(scores), 3)
        entry["per_run"] = [
            all_results[j][i].to_dict() for j in range(len(contexts)) if i < len(all_results[j])
        ]
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


def eval_cli(args: list[str] | None = None) -> int:
    """CLI entry point for running evals.

    Usage::

        looplet eval traces/                          # score all runs
        looplet eval traces/ --evals eval_agent.py    # specific eval file
        looplet eval traces/ --threshold 0.7          # fail if avg < 0.7
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
    if raw and raw[0] == "cases":
        return _cases_cli(raw[1:])
    if raw and raw[0] == "run":
        return _run_cartridge_cli(raw[1:])

    parser = argparse.ArgumentParser(
        prog="looplet eval",
        description="Run evals against saved agent trajectories.",
        epilog=(
            "subcommands:\n"
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
        default=0.0,
        help="Fail if any eval avg score < threshold (default: 0)",
    )
    parser.add_argument(
        "--include", nargs="*", default=None, help="Only run evals with these marks"
    )
    parser.add_argument("--exclude", nargs="*", default=None, help="Skip evals with these marks")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-run details")

    parsed = parser.parse_args(args)
    if not math.isfinite(parsed.threshold) or not 0.0 <= parsed.threshold <= 1.0:
        print("error: --threshold must be finite and between 0 and 1")
        return 1

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
    if not traces_root.exists():
        print(f"Traces directory not found: {traces_root}")
        return 1

    contexts: list[EvalContext] = []
    names: list[str] = []
    load_failures: list[str] = []
    for d in sorted(traces_root.iterdir()):
        if d.is_dir() and (d / "trajectory.json").exists():
            try:
                contexts.append(EvalContext.from_trajectory_dir(d))
                names.append(d.name)
            except Exception as e:  # noqa: BLE001
                failure = f"{d.name}: {e}"
                load_failures.append(failure)
                print(f"  ERROR {failure}")

    if not contexts:
        print(f"No trajectories found in {traces_root}")
        return 1

    print(f"Found {len(evaluators)} evals, {len(contexts)} trajectories\n")

    # Run batch
    table = eval_run_batch(
        evaluators,
        contexts,
        include=parsed.include,
        exclude=parsed.exclude,
    )
    if not table:
        print("No evals selected after applying include/exclude filters")
        return 1

    # Print results
    below_threshold = False
    integrity_failures: list[str] = []
    required_names = {fn.__name__ for fn in evaluators if _REQUIRED_EVAL_MARK in _get_marks(fn)}
    selected_names = {row["name"] for row in table}
    for omitted in sorted(required_names - selected_names):
        integrity_failures.append(f"{omitted}: required evaluator was filtered out")
    for row in table:
        avg = row.get("avg_score")
        if avg is not None:
            scores = row.get("scores", [])
            raw_avg = sum(scores) / len(scores) if scores else avg
            marker = "✓" if raw_avg >= parsed.threshold else "✗"
            if raw_avg < parsed.threshold:
                below_threshold = True
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

        for j, run in enumerate(row.get("per_run", [])):
            if _result_is_integrity_failure(
                run,
                required=row["name"] in required_names,
            ):
                run_name = names[j] if j < len(names) else f"run_{j}"
                integrity_failures.append(
                    f"{run_name}/{row['name']}: {run.get('label') or 'invalid result'}"
                )

    # Summary
    scored = [r for r in table if r.get("avg_score") is not None]
    if scored:
        overall = sum(r["avg_score"] for r in scored) / len(scored)
        print(f"\n  overall: {overall:.2f}")
        if parsed.threshold > 0:
            status = "FAIL" if below_threshold or integrity_failures else "PASS"
            print(f"  threshold: {parsed.threshold:.2f}  → {status}")

    if integrity_failures:
        print("\n  integrity failures:")
        for failure in integrity_failures:
            print(f"    - {failure}")

    if load_failures:
        print("\n  trajectory load failures:")
        for failure in load_failures:
            print(f"    - {failure}")

    return 1 if below_threshold or integrity_failures or load_failures else 0


def _run_cartridge_cli(args: list[str]) -> int:
    """``looplet eval run <cartridge> [--out DIR] [--model M] ...``.

    Run a cartridge against its own shipped eval cases (seed → run →
    grade), printing a cases × graders matrix and optionally persisting
    each run for offline inspection.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="looplet eval run",
        description="Run a cartridge against its shipped eval cases.",
    )
    parser.add_argument("cartridge", help="Cartridge directory (must contain evals/).")
    parser.add_argument(
        "--out",
        default=None,
        help="Persist each run under <out>/<case_id>/ (trajectory + artifacts + scores).",
    )
    parser.add_argument("--model", default=None, help="Model name (else $OPENAI_MODEL).")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL (else $OPENAI_BASE_URL)."
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps per case.")
    parser.add_argument(
        "--case", action="append", default=None, help="Run only this case id (repeatable)."
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
        default=0.0,
        help="Fail (exit 1) if any scored grader is below this on any case.",
    )
    parsed = parser.parse_args(args)
    if not math.isfinite(parsed.threshold) or not 0.0 <= parsed.threshold <= 1.0:
        print("error: --threshold must be finite and between 0 and 1", file=sys.stderr)
        return 1

    cdir = Path(parsed.cartridge)
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

    base_url = parsed.base_url or os.environ.get("OPENAI_BASE_URL")
    model = parsed.model or os.environ.get("OPENAI_MODEL")
    if not base_url and not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: no LLM configured. Set OPENAI_BASE_URL (local proxy) or "
            "OPENAI_API_KEY (cloud), or pass --base-url / --model.",
            file=sys.stderr,
        )
        return 1

    from looplet.backends import OpenAIBackend  # noqa: PLC0415

    llm = OpenAIBackend(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "x"),
        model=model or "gpt-4o",
    )

    # LLM-as-judge: graders whose signature has an ``llm`` param are only
    # run when a judge backend is supplied. --judge reuses the agent
    # backend; --judge-model builds a separate one (and implies --judge).
    judge_llm = None
    if parsed.judge or parsed.judge_model:
        if parsed.judge_model:
            judge_llm = OpenAIBackend(
                base_url=base_url,
                api_key=os.environ.get("OPENAI_API_KEY", "x"),
                model=parsed.judge_model,
            )
        else:
            judge_llm = llm

    print(
        f"running {cdir.name}: {len(overview.cases)} case(s), "
        f"{len(overview.graders)} grader(s), {len(overview.collectors)} collector(s)"
        f"{'  [judge on]' if judge_llm is not None else ''}\n"
    )
    try:
        records = run_cartridge_evals(
            cdir,
            llm=llm,
            output_dir=parsed.out,
            max_steps=parsed.max_steps,
            cases=parsed.case,
            judge_llm=judge_llm,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    grader_names = sorted(g.__name__ for g in overview.graders)
    header = f"{'case':22}" + "".join(f"{n:22}" for n in grader_names)
    print(header)
    print("-" * len(header))
    below_threshold = False
    integrity_failures: list[str] = []
    required_names = {
        grader.__name__ for grader in overview.graders if _REQUIRED_EVAL_MARK in _get_marks(grader)
    }
    for rec in records:
        scores = {r.name: r for r in rec.results}
        cells = ""
        for n in grader_names:
            r = scores.get(n)
            if r is None:
                cells += f"{'-':22}"
            elif r.score is not None:
                cells += f"{r.score:<22.2f}"
                if r.score < parsed.threshold:
                    below_threshold = True
            elif r.metrics:
                # Metric grader (no pass/fail score) - show the numbers.
                cells += f"{('; '.join(f'{k}={v:g}' for k, v in r.metrics.items())):22}"
            else:
                cells += f"{(r.label or '-'):22}"
        case_id = rec.case.id if rec.case is not None else "?"
        print(f"{case_id:22}{cells}")
        for result in rec.results:
            if _result_is_integrity_failure(
                result,
                required=result.name in required_names,
            ):
                integrity_failures.append(
                    f"{case_id}/{result.name}: {result.label or 'invalid result'}"
                )

    if parsed.out:
        print(f"\npersisted {len(records)} run(s) under {parsed.out}/")
    if integrity_failures:
        print("\nintegrity failures:")
        for failure in integrity_failures:
            print(f"  - {failure}")
    if parsed.threshold > 0:
        failed = below_threshold or bool(integrity_failures)
        print(f"threshold {parsed.threshold:.2f} → {'FAIL' if failed else 'PASS'}")
    return 1 if below_threshold or integrity_failures else 0


def _cases_cli(args: list[str]) -> int:
    """``looplet eval cases ls|show <path>`` - read-only case browser."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="looplet eval cases",
        description="List or show eval cases.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    p_ls = sub.add_parser("ls", help="List cases (one line per case).")
    p_ls.add_argument("path", help="Case file or directory.")
    p_show = sub.add_parser("show", help="Show a case in full.")
    p_show.add_argument("path", help="Case file or directory.")
    p_show.add_argument(
        "case_id",
        nargs="?",
        default=None,
        help="Case id to show (required when path is a directory of multiple cases).",
    )

    parsed = parser.parse_args(args)

    try:
        cases = load_cases(parsed.path)
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
                (c.task.get("description") or c.notes or "").replace("\n", " "),
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
