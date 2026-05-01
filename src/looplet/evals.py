"""Eval framework — pytest-style evaluation for agent runs.

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
indexing ``ctx.tool_sequence`` or grepping ``ctx.steps`` — the
former survives the model changing its workflow, the latter
encodes today's expected trajectory as a permanent grade.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
]

logger = logging.getLogger(__name__)

# ── Core data types ──────────────────────────────────────────────


@dataclass
class EvalContext:
    """Everything an evaluator sees — the same data you see when debugging.

    Build from a live loop run (via :class:`EvalHook`) or from saved
    trajectories (via :meth:`from_trajectory_dir`).
    """

    steps: list[Any]
    """Full list of :class:`Step` objects from the run."""

    task: dict[str, Any] = field(default_factory=dict)
    """Original task dict passed to ``composable_loop``."""

    final_output: dict[str, Any] = field(default_factory=dict)
    """The ``done()`` tool's args — the agent's final answer."""

    session_log_text: str = ""
    """Rendered session log — the text the LLM saw."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra context: run_id, model, timestamp, etc."""

    artifacts: dict[str, Any] = field(default_factory=dict)
    """Outcome data collected from outside the trajectory.

    Populated by :class:`EvalHook` collectors at the end of a run, or
    loaded from ``artifacts.json`` via :meth:`from_trajectory_dir`.
    Use this slot to grade *what changed in the world* — tests
    passing, files modified, repo state — instead of grepping
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

        data = json.loads(traj_path.read_text())
        steps = data.get("steps", [])
        task = data.get("task", {})
        if not isinstance(task, dict):
            task = {"description": str(task)} if task else {}
        # Pull through the trajectory's own metadata dict (which may
        # contain harness_snapshot from TrajectoryRecorder, plus any
        # user-attached fields) and overlay the well-known top-level
        # fields so they are always available at the documented keys.
        traj_metadata = data.get("metadata") or {}
        if not isinstance(traj_metadata, dict):
            traj_metadata = {}
        metadata: dict[str, Any] = dict(traj_metadata)
        metadata.update(
            {
                "run_id": data.get("run_id"),
                "started_at": data.get("started_at"),
                "ended_at": data.get("ended_at"),
                "termination_reason": data.get("termination_reason"),
            }
        )

        # Extract final_output from the last done() step
        final_output: dict[str, Any] = {}
        for s in reversed(steps):
            # Support both formats:
            #   looplet: {"tool_call": {"tool": "done", "args": {...}}}
            #   benchmark:   {"tool": "done", "args_summary": "..."}
            tc = s.get("tool_call", {})
            tool_name = tc.get("tool") or s.get("tool", "")
            if tool_name == "done":
                final_output = tc.get("args", {})
                break

        # Also load from metrics.json if available (ground-truth data)
        metrics_path = root / "metrics.json"
        if metrics_path.exists():
            try:
                metrics_data = json.loads(metrics_path.read_text())
                # Merge any ground-truth fields into task (so evaluators
                # can compare expected vs actual without knowing the
                # file layout). Only copies keys that don't already
                # exist in task to avoid overwriting user-supplied values.
                for key, value in metrics_data.items():
                    if key.startswith("expected_") and key not in task:
                        task[key] = value
                # If no done() output was found in the trajectory but
                # metrics.json has a top-level "output" dict, promote it.
                if not final_output and isinstance(metrics_data.get("output"), dict):
                    final_output = metrics_data["output"]
            except Exception:  # noqa: BLE001
                pass

        # Load artifacts.json if present — outcome data collected
        # outside the trajectory (test results, file diffs, repo
        # state, etc.). See EvalHook(collectors=...).
        artifacts: dict[str, Any] = {}
        artifacts_path = root / "artifacts.json"
        if artifacts_path.exists():
            try:
                loaded = json.loads(artifacts_path.read_text())
                if isinstance(loaded, dict):
                    artifacts = loaded
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load artifacts.json from %s", root, exc_info=True)

        return cls(
            steps=[_DictStep(s) for s in steps],
            task=task if isinstance(task, dict) else {"description": str(task)},
            final_output=final_output,
            session_log_text="",  # not saved in trajectory by default
            metadata=metadata,
            artifacts=artifacts,
            stop_reason=data.get("termination_reason") or metadata.get("termination_reason"),
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

    @property
    def passed(self) -> bool:
        """Whether this evaluation passed.

        True when ``score`` is not ``None`` and ``>= 0.5``, OR when
        ``label`` is in ``{"pass", "correct", "yes"}`` (case-insensitive).
        Returns ``False`` otherwise, including when both ``score`` and
        ``label`` are unset.
        """
        if self.score is not None:
            return self.score >= 0.5
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
            return cls(name=name, score=float(value))
        if isinstance(value, str):
            return cls(name=name, label=value)
        if isinstance(value, dict):
            metrics = {k: float(v) for k, v in value.items() if isinstance(v, (int, float))}
            details = [f"{k}: {v}" for k, v in value.items() if not isinstance(v, (int, float))]
            # Try to find a primary score
            score = (
                metrics.get("score")
                or metrics.get("f1")
                or metrics.get("accuracy")
                or metrics.get("overall")
            )
            return cls(name=name, score=score, metrics=metrics, details=details)
        return cls(name=name, explanation=str(value))

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
    """A single eval case — task + expected outcomes + tags.

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
    * ``notes``: free-text — why this case was added, what regression
      it captures.
    """

    id: str
    task: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    marks: list[str] = field(default_factory=list)
    notes: str = ""

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
        target = target / f"{case.id}.json"
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
    return tuple(eval_discover(path))


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
) -> list[Callable]:
    """Find evaluator functions in files matching ``pattern``.

    Discovers all functions whose name starts with ``prefix`` in
    all Python files matching ``pattern`` under ``path``. Works
    like pytest's test collection — no registration needed.

    Args:
        path: File or directory to search.
        pattern: Glob pattern for eval files (default: ``eval_*.py``).
        prefix: Function name prefix (default: ``eval_``).

    Returns:
        List of callable evaluator functions.
    """
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob(pattern))

    evaluators: list[Callable] = []
    for fpath in files:
        try:
            spec = importlib.util.spec_from_file_location(
                f"_eval_{fpath.stem}",
                fpath,
            )
            if spec is None or spec.loader is None:
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
                evaluators.append(obj)
        except Exception:  # noqa: BLE001
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
    trajectory. A collector that raises or returns a non-dict is
    silently skipped — collectors are observers and must never break
    a run.
    """

    def __init__(
        self,
        evaluators: list[Callable],
        *,
        judge_llm: LLMBackend | None = None,
        collectors: list[Callable[[Any], dict[str, Any]]] | None = None,
        verbose: bool = False,
    ) -> None:
        self.evaluators = evaluators
        self.judge_llm = judge_llm
        self.collectors = list(collectors) if collectors else []
        self.verbose = verbose
        self._results: list[EvalResult] = []
        self._task: dict[str, Any] = {}
        self._artifacts: dict[str, Any] = {}

    def to_config(self) -> dict:
        """Workspace round-trip: emit ``evaluators`` (and ``collectors``
        when present) as ``@ref`` strings so the workspace writer
        auto-generates ``resources/<name>.py`` builders. Lambda
        evaluators land in placeholder builders the user must replace;
        top-level callables ride straight through.
        """
        cfg: dict[str, Any] = {"evaluators": "@evaluators"}
        if self.collectors:
            cfg["collectors"] = "@collectors"
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
        data: dict[str, Any] = {
            "task": self._task,
            "results": [r.to_dict() for r in self._results],
            "summary": _format_summary(self._results),
        }
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

        # Capture task from state (stashed by composable_loop) if the
        # hook wasn't handed one explicitly.
        if not self._task:
            _state_task = getattr(state, "task", None)
            if isinstance(_state_task, dict):
                self._task = _state_task
            elif _state_task is not None:
                self._task = {"description": str(_state_task)}

        # Extract final_output from done() step
        final_output: dict[str, Any] = {}
        for s in reversed(steps):
            tc = getattr(s, "tool_call", None)
            if tc and getattr(tc, "tool", "") == "done":
                final_output = getattr(tc, "args", {})
                break

        log_text = ""
        if session_log is not None and hasattr(session_log, "render"):
            try:
                log_text = session_log.render() or ""
            except Exception:  # noqa: BLE001
                pass

        # Run collectors to populate outcome artifacts. A collector
        # that raises or returns a non-dict is skipped — collectors
        # observe, they must never break a run.
        artifacts: dict[str, Any] = {}
        for collector in self.collectors:
            try:
                produced = collector(state)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Eval collector %s raised; skipping",
                    getattr(collector, "__name__", repr(collector)),
                    exc_info=True,
                )
                continue
            if isinstance(produced, dict):
                artifacts.update(produced)
            else:
                logger.warning(
                    "Eval collector %s returned %s; expected dict — ignored",
                    getattr(collector, "__name__", repr(collector)),
                    type(produced).__name__,
                )
        self._artifacts = artifacts

        ctx = EvalContext(
            steps=list(steps),
            task=self._task,
            final_output=final_output,
            session_log_text=log_text,
            artifacts=artifacts,
            stop_reason=getattr(state, "_stop_reason", None),
        )

        self._results = eval_run(
            self.evaluators,
            ctx,
            judge_llm=self.judge_llm,
        )

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

    Like pytest.mark — lets you group and filter evals::

        @eval_mark("accuracy", "fast")
        def eval_answer_correct(ctx):
            ...

        @eval_mark("quality", "slow")
        def eval_reasoning_depth(ctx, llm):
            ...

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

    Like pytest parametrize — same evals, different inputs::

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

        looplet eval cases ls evals/cases/            # list cases
        looplet eval cases show evals/cases/foo.json  # full case dump

    Returns 0 if all evals pass threshold, 1 otherwise.
    """
    import argparse

    # Lightweight subcommand dispatch — keep the existing flat surface
    # working when the first arg isn't a recognized subcommand.
    raw = list(args) if args is not None else sys.argv[1:]
    if raw and raw[0] == "cases":
        return _cases_cli(raw[1:])

    parser = argparse.ArgumentParser(
        prog="looplet eval",
        description="Run evals against saved agent trajectories.",
        epilog=(
            "subcommands:\n"
            "  cases ls <path>          list eval cases (one line per case)\n"
            "  cases show <path>        show a single case in full\n"
            "\n"
            "Run `looplet eval cases -h` for case-browser flags."
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

    # Discover evals
    eval_path = parsed.evals or "."
    evaluators = eval_discover(eval_path)
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
    for d in sorted(traces_root.iterdir()):
        if d.is_dir() and (d / "trajectory.json").exists():
            try:
                contexts.append(EvalContext.from_trajectory_dir(d))
                names.append(d.name)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {d.name}: {e}")

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

    # Print results
    below_threshold = False
    for row in table:
        avg = row.get("avg_score")
        if avg is not None:
            marker = "✓" if avg >= parsed.threshold else "✗"
            if avg < parsed.threshold:
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
                score = run.get("score", "—")
                details = run.get("details", [])
                print(f"      {label}: {score}")
                for d in details[:3]:
                    print(f"        {d}")

    # Summary
    scored = [r for r in table if r.get("avg_score") is not None]
    if scored:
        overall = sum(r["avg_score"] for r in scored) / len(scored)
        print(f"\n  overall: {overall:.2f}")
        if parsed.threshold > 0:
            status = "PASS" if not below_threshold else "FAIL"
            print(f"  threshold: {parsed.threshold:.2f}  → {status}")

    return 1 if below_threshold else 0


def _cases_cli(args: list[str]) -> int:
    """``looplet eval cases ls|show <path>`` — read-only case browser."""
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
