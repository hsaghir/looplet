"""Eval framework — pytest-style evaluation for agent runs.

Write functions named ``eval_*`` that take an :class:`EvalContext`
and return a score, label, dict, or :class:`EvalResult`. The
framework discovers them, runs them, and aggregates results.

Quick start::

    # eval_investigation.py (anywhere in your project)

    def eval_evidence_density(ctx):
        claims = ctx.final_output.get("findings", [])
        evidenced = [c for c in claims if c.get("evidence")]
        return len(evidenced) / max(len(claims), 1)

    def eval_triage_correct(ctx):
        return "correct" if ctx.final_output.get("verdict") == ctx.task.get("expected") else "wrong"

    def eval_reasoning_gaps(ctx, llm):
        resp = llm.generate(f"Score 0-1: are claims in {ctx.final_output} supported by {ctx.session_log_text}?")
        return float(resp.strip())

Run evals::

    from openharness.evals import eval_discover, eval_run, EvalContext

    fns = eval_discover("eval_investigation.py")
    ctx = EvalContext.from_trajectory_dir("traces/run_1/")
    results = eval_run(fns, ctx)
    for r in results:
        print(r.pretty())

Or attach to the loop for live scoring::

    from openharness.evals import EvalHook

    hook = EvalHook(evaluators=[eval_evidence_density, eval_triage_correct])
    for step in composable_loop(..., hooks=[hook]):
        ...
    print(hook.summary())
"""

from __future__ import annotations

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
    from openharness.session import SessionLog
    from openharness.types import AgentState, LLMBackend

__all__ = [
    "EvalContext",
    "EvalResult",
    "EvalHook",
    "eval_discover",
    "eval_run",
    "eval_run_batch",
    "eval_mark",
    "eval_cli",
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
            s for s in self.steps
            if hasattr(s, "tool_result") and s.tool_result
            and getattr(s.tool_result, "error", None)
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
        metadata = {
            "run_id": data.get("run_id"),
            "started_at": data.get("started_at"),
            "ended_at": data.get("ended_at"),
            "termination_reason": data.get("termination_reason"),
        }

        # Extract final_output from the last done() step
        final_output: dict[str, Any] = {}
        for s in reversed(steps):
            # Support both formats:
            #   openharness: {"tool_call": {"tool": "done", "args": {...}}}
            #   benchmark:   {"tool": "done", "args_summary": "..."}
            tc = s.get("tool_call", {})
            tool_name = tc.get("tool") or s.get("tool", "")
            if tool_name == "done":
                final_output = tc.get("args", {})
                break

        # Also load from metrics.json if available (richer data)
        metrics_path = root / "metrics.json"
        if metrics_path.exists():
            try:
                metrics_data = json.loads(metrics_path.read_text())
                # Merge ground truth into task
                if "expected_verdict" in metrics_data and "expected_verdict" not in task:
                    task["expected_verdict"] = metrics_data["expected_verdict"]
                if "expected_iocs" in metrics_data and "expected_iocs" not in task:
                    task["expected_iocs"] = metrics_data["expected_iocs"]
                # Use predicted_iocs as final output if not already set
                if not final_output and "predicted_iocs" in metrics_data:
                    final_output = {
                        "promoted_iocs": metrics_data["predicted_iocs"],
                        "verdict": metrics_data.get("verdict", ""),
                    }
                elif "verdict" in metrics_data and "verdict" not in final_output:
                    final_output["verdict"] = metrics_data["verdict"]
            except Exception:  # noqa: BLE001
                pass

        # Also pull verdict from top-level trajectory data
        if "verdict" not in final_output and data.get("verdict"):
            final_output["verdict"] = data["verdict"]
        if "expected_verdict" not in task and data.get("expected_verdict"):
            task["expected_verdict"] = data["expected_verdict"]

        return cls(
            steps=[_DictStep(s) for s in steps],
            task=task if isinstance(task, dict) else {"description": str(task)},
            final_output=final_output,
            session_log_text="",  # not saved in trajectory by default
            metadata=metadata,
        )


@dataclass
class _DictStep:
    """Lightweight step wrapper for trajectories loaded from JSON.

    Supports both formats:
      - openharness: {"tool_call": {"tool": "x"}, "tool_result": {"data": {}}}
      - benchmark:   {"tool": "x", "args_summary": "...", "error": null}
    """

    _data: dict[str, Any]

    @property
    def tool_call(self) -> Any:
        tc = self._data.get("tool_call", {})
        if not tc and "tool" in self._data:
            # Flat format: tool name at top level
            tc = {"tool": self._data["tool"],
                  "args": self._data.get("args", {})}
        return _DictView(tc)

    @property
    def tool_result(self) -> Any:
        tr = self._data.get("tool_result", {})
        if not tr and "error" in self._data:
            tr = {"error": self._data.get("error"),
                  "data": self._data.get("data", {})}
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
    """Specific findings (missed IOCs, unsupported claims, etc.)."""

    explanation: str = ""
    """Human-readable summary of the evaluation."""

    duration_ms: float = 0.0
    """How long the evaluator took to run."""

    @classmethod
    def from_return(cls, value: Any, *, name: str = "") -> "EvalResult":
        """Normalize any return type into an EvalResult."""
        if isinstance(value, EvalResult):
            if not value.name:
                value.name = name
            return value
        if isinstance(value, bool):
            return cls(name=name, score=1.0 if value else 0.0,
                       label="pass" if value else "fail")
        if isinstance(value, (int, float)):
            return cls(name=name, score=float(value))
        if isinstance(value, str):
            return cls(name=name, label=value)
        if isinstance(value, dict):
            metrics = {k: float(v) for k, v in value.items()
                       if isinstance(v, (int, float))}
            details = [f"{k}: {v}" for k, v in value.items()
                       if not isinstance(v, (int, float))]
            # Try to find a primary score
            score = (metrics.get("score") or metrics.get("f1")
                     or metrics.get("accuracy") or metrics.get("overall"))
            return cls(name=name, score=score, metrics=metrics,
                       details=details)
        return cls(name=name, explanation=str(value))

    def pretty(self) -> str:
        """One-line formatted output for terminal display."""
        parts = [f"{self.name:40s}"]
        if self.score is not None:
            parts.append(f"{self.score:.2f}")
        if self.label:
            parts.append(self.label)
        if self.metrics:
            metric_strs = [f"{k}={v:.2f}" for k, v in self.metrics.items()
                           if k not in ("score", "f1", "accuracy", "overall")]
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


# ── Discovery ────────────────────────────────────────────────────


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
                f"_eval_{fpath.stem}", fpath,
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod.__name__] = mod
            spec.loader.exec_module(mod)
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                if name.startswith(prefix):
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
                    results.append(EvalResult(
                        name=name, label="skipped",
                        explanation="requires judge_llm",
                    ))
                    continue
                raw = fn(ctx, judge_llm)
            else:
                raw = fn(ctx)
            result = EvalResult.from_return(raw, name=name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Eval %s raised: %s", name, e, exc_info=True)
            result = EvalResult(name=name, label="error",
                                explanation=str(e))
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
            verbose=True,              # print scores live
        )
        for step in composable_loop(..., hooks=[hook]):
            ...
        print(hook.summary())
        hook.save("evals/run_1.json")
    """

    def __init__(
        self,
        evaluators: list[Callable],
        *,
        judge_llm: LLMBackend | None = None,
        verbose: bool = False,
    ) -> None:
        self.evaluators = evaluators
        self.judge_llm = judge_llm
        self.verbose = verbose
        self._results: list[EvalResult] = []
        self._task: dict[str, Any] = {}

    @property
    def results(self) -> list[EvalResult]:
        """Eval results from the most recent run."""
        return list(self._results)

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
        data = {
            "task": self._task,
            "results": [r.to_dict() for r in self._results],
            "summary": _format_summary(self._results),
        }
        p.write_text(json.dumps(data, indent=2, default=str))

    # ── LoopHook interface ─────────────────────────────────────

    def on_loop_end(
        self, state: AgentState, session_log: SessionLog, context: Any, llm: LLMBackend,
    ) -> int:
        """Run all evaluators after the loop finishes."""
        steps = getattr(state, "steps", [])

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

        ctx = EvalContext(
            steps=list(steps),
            task=self._task,
            final_output=final_output,
            session_log_text=log_text,
        )

        self._results = eval_run(
            self.evaluators, ctx, judge_llm=self.judge_llm,
        )

        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Eval results:")
            for r in self._results:
                print(f"  {r.pretty()}")
            print(f"  {'overall':38s} {_format_summary(self._results)}")
            print(f"{'─' * 50}")

        return 0

    def pre_loop(self, state: AgentState, session_log: SessionLog,
                 context: Any) -> None:
        """Capture the task from context for eval."""
        # The task is passed via composable_loop's task= kwarg and
        # threaded through the loop. We capture it from state or
        # context if available.
        return None

    # Protocol stubs
    def pre_prompt(self, *a: Any, **k: Any) -> None: return None
    def pre_dispatch(self, *a: Any, **k: Any) -> None: return None
    def post_dispatch(self, *a: Any, **k: Any) -> None: return None
    def check_done(self, *a: Any, **k: Any) -> None: return None
    def check_permission(self, *a: Any, **k: Any) -> None: return None
    def should_stop(self, *a: Any, **k: Any) -> bool: return False
    def should_compact(self, *a: Any, **k: Any) -> bool: return False
    def build_briefing(self, *a: Any, **k: Any) -> None: return None
    def build_prompt(self, **k: Any) -> None: return None
    def on_event(self, *a: Any, **k: Any) -> None: return None


# ── Marks ────────────────────────────────────────────────────────


def eval_mark(*tags: str) -> Callable:
    """Tag an eval function with category marks for filtering.

    Like pytest.mark — lets you group and filter evals::

        @eval_mark("verdict", "fast")
        def eval_verdict_correct(ctx):
            ...

        @eval_mark("ioc", "slow")
        def eval_ioc_quality(ctx, llm):
            ...

        # Run only "verdict" evals:
        results = eval_run(evals, ctx, include=["verdict"])

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
            s for s in (
                all_results[j][i].score
                for j in range(len(contexts))
                if i < len(all_results[j])
            ) if s is not None
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
            all_results[j][i].to_dict()
            for j in range(len(contexts))
            if i < len(all_results[j])
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

        openharness eval traces/                          # score all runs
        openharness eval traces/ --evals eval_agent.py    # specific eval file
        openharness eval traces/ --threshold 0.7          # fail if avg < 0.7
        openharness eval traces/ --include verdict        # only verdict evals
        openharness eval traces/ --exclude slow            # skip slow evals

    Returns 0 if all evals pass threshold, 1 otherwise.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="openharness eval",
        description="Run evals against saved agent trajectories.",
    )
    parser.add_argument("traces", help="Directory containing trajectory dirs")
    parser.add_argument("--evals", default=None,
                        help="Eval file or directory (default: discover in cwd)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Fail if any eval avg score < threshold (default: 0)")
    parser.add_argument("--include", nargs="*", default=None,
                        help="Only run evals with these marks")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Skip evals with these marks")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-run details")

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
        evaluators, contexts,
        include=parsed.include, exclude=parsed.exclude,
    )

    # Print results
    below_threshold = False
    for row in table:
        avg = row.get("avg_score")
        if avg is not None:
            marker = "✓" if avg >= parsed.threshold else "✗"
            if avg < parsed.threshold:
                below_threshold = True
            print(f"  {marker} {row['name']:40s} avg={avg:.2f}  "
                  f"min={row.get('min_score', 0):.2f}  "
                  f"max={row.get('max_score', 0):.2f}  "
                  f"({row['runs']} runs)")
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
