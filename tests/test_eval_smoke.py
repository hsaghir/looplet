"""Smoke tests for the eval framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.evals import (
    EvalContext,
    EvalHook,
    EvalResult,
    eval_discover,
    eval_mark,
    eval_run,
    eval_run_batch,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import Step, ToolCall, ToolResult

pytestmark = pytest.mark.smoke


# ── Helpers ──────────────────────────────────────────────────────


def _step(
    tool: str, args: dict | None = None, data: dict | None = None, error: str | None = None
) -> Step:
    tc = ToolCall(tool=tool, args=args or {})
    tr = ToolResult(tool=tool, args_summary="", data=data, error=error)
    return Step(number=1, tool_call=tc, tool_result=tr)


def _ctx(**kw) -> EvalContext:
    return EvalContext(
        steps=kw.get("steps", []), task=kw.get("task", {}), final_output=kw.get("final_output", {})
    )


# ── EvalResult.from_return ───────────────────────────────────────


class TestEvalResultFromReturn:
    def test_float(self):
        r = EvalResult.from_return(0.85, name="x")
        assert r.score == 0.85 and r.name == "x"

    def test_bool_true(self):
        r = EvalResult.from_return(True, name="y")
        assert r.score == 1.0 and r.label == "pass"

    def test_bool_false(self):
        r = EvalResult.from_return(False, name="z")
        assert r.score == 0.0 and r.label == "fail"

    def test_string(self):
        r = EvalResult.from_return("partial", name="t")
        assert r.label == "partial" and r.score is None

    def test_dict_with_metrics(self):
        r = EvalResult.from_return(
            {"precision": 0.9, "recall": 0.7, "f1": 0.8, "missed": ["a", "b"]},
            name="q",
        )
        assert r.score == 0.8  # picks f1
        assert r.metrics["precision"] == 0.9
        assert any("missed" in d for d in r.details)

    def test_eval_result_passthrough(self):
        orig = EvalResult(name="orig", score=0.5)
        r = EvalResult.from_return(orig, name="override")
        assert r.name == "orig"  # preserves original name

    def test_eval_result_unnamed(self):
        orig = EvalResult(score=0.5)
        r = EvalResult.from_return(orig, name="fallback")
        assert r.name == "fallback"


# ── eval_run ─────────────────────────────────────────────────────


class TestEvalRun:
    def test_runs_simple_evaluators(self):
        def eval_step_count(ctx: EvalContext) -> float:
            return min(ctx.step_count / 5, 1.0)

        def eval_no_errors(ctx: EvalContext) -> bool:
            return len(ctx.errors) == 0

        ctx = _ctx(steps=[_step("bash"), _step("done")])
        results = eval_run([eval_step_count, eval_no_errors], ctx)
        assert len(results) == 2
        assert results[0].name == "eval_step_count"
        assert results[0].score == pytest.approx(0.4)
        assert results[1].name == "eval_no_errors"
        assert results[1].score == 1.0

    def test_skips_llm_eval_without_judge(self):
        def eval_needs_llm(ctx, llm):
            return llm.generate("score this")

        results = eval_run([eval_needs_llm], _ctx())
        assert results[0].label == "skipped"

    def test_passes_llm_to_judge_eval(self):
        class FakeJudge:
            def generate(self, prompt, **kw):
                return "0.75"

        def eval_with_judge(ctx, llm):
            return float(llm.generate("x"))

        results = eval_run([eval_with_judge], _ctx(), judge_llm=FakeJudge())
        assert results[0].score == pytest.approx(0.75)

    def test_catches_eval_errors(self):
        def eval_broken(ctx):
            raise ValueError("oops")

        results = eval_run([eval_broken], _ctx())
        assert results[0].label == "error"
        assert "oops" in results[0].explanation

    def test_dict_return(self):
        def eval_multi(ctx):
            return {"precision": 0.9, "recall": 0.8, "notes": "good"}

        results = eval_run([eval_multi], _ctx())
        assert results[0].metrics["precision"] == 0.9
        assert results[0].metrics["recall"] == 0.8


# ── eval_discover ────────────────────────────────────────────────


class TestEvalDiscover:
    def test_discovers_eval_functions(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eval_test.py"
            p.write_text(
                "def eval_foo(ctx): return 1.0\n"
                "def eval_bar(ctx): return 0.5\n"
                "def helper(x): return x\n"  # not an eval
            )
            fns = eval_discover(d)
            names = [f.__name__ for f in fns]
            assert "eval_foo" in names
            assert "eval_bar" in names
            assert "helper" not in names

    def test_discovers_from_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eval_single.py"
            p.write_text("def eval_one(ctx): return 1.0\n")
            fns = eval_discover(p)
            assert len(fns) == 1

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            fns = eval_discover(d)
            assert fns == []


# ── EvalContext.from_trajectory_dir ──────────────────────────────


class TestEvalContextFromDir:
    def test_loads_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            traj = {
                "run_id": "test",
                "started_at": 1.0,
                "ended_at": 2.0,
                "steps": [
                    {
                        "tool_call": {"tool": "bash", "args": {"command": "ls"}},
                        "tool_result": {"tool": "bash", "data": {}},
                    },
                    {
                        "tool_call": {"tool": "done", "args": {"answer": "ok"}},
                        "tool_result": {"tool": "done", "data": {}},
                    },
                ],
                "task": {"goal": "test"},
            }
            (Path(d) / "trajectory.json").write_text(json.dumps(traj))
            ctx = EvalContext.from_trajectory_dir(d)
            assert ctx.step_count == 2
            assert ctx.tool_sequence == ["bash", "done"]
            assert ctx.final_output == {"answer": "ok"}


# ── EvalHook integration ────────────────────────────────────────


class TestEvalHookIntegration:
    def test_runs_evals_after_loop(self):
        def eval_completed(ctx: EvalContext) -> bool:
            return "done" in ctx.tool_sequence

        def eval_efficient(ctx: EvalContext) -> float:
            return min(3 / max(ctx.step_count, 1), 1.0)

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )

        hook = EvalHook(evaluators=[eval_completed, eval_efficient])
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                    ]
                ),
                tools=reg,
                state=DefaultState(max_steps=3),
                hooks=[hook],
                config=LoopConfig(max_steps=3),
            )
        )
        assert len(hook.results) == 2
        assert hook.results[0].score == 1.0  # completed
        assert hook.results[1].score == pytest.approx(1.0)  # 3/1 capped at 1

    def test_summary_and_report(self):
        def eval_a(ctx):
            return 0.8

        def eval_b(ctx):
            return "pass"

        hook = EvalHook(evaluators=[eval_a, eval_b])
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                    ]
                ),
                tools=reg,
                state=DefaultState(max_steps=3),
                hooks=[hook],
                config=LoopConfig(max_steps=3),
            )
        )
        assert "0.80" in hook.summary() or "scored" in hook.summary()
        assert "eval_a" in hook.report()
        assert "eval_b" in hook.report()

    def test_save(self):
        def eval_x(ctx):
            return 0.5

        hook = EvalHook(evaluators=[eval_x])
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                    ]
                ),
                tools=reg,
                state=DefaultState(max_steps=3),
                hooks=[hook],
                config=LoopConfig(max_steps=3),
            )
        )
        with tempfile.TemporaryDirectory() as d:
            hook.save(Path(d) / "results.json")
            data = json.loads((Path(d) / "results.json").read_text())
            assert data["results"][0]["score"] == 0.5


# ── EvalResult.pretty + to_dict ──────────────────────────────────


class TestEvalResultOutput:
    def test_pretty_score(self):
        r = EvalResult(name="test", score=0.85)
        assert "0.85" in r.pretty()

    def test_pretty_with_details(self):
        r = EvalResult(name="test", score=0.5, details=["missed: x", "missed: y"])
        p = r.pretty()
        assert "missed: x" in p
        assert "missed: y" in p

    def test_to_dict(self):
        r = EvalResult(name="t", score=0.5, label="partial", metrics={"p": 0.9}, details=["a"])
        d = r.to_dict()
        assert d["score"] == 0.5
        assert d["label"] == "partial"
        assert d["metrics"]["p"] == 0.9


# ── eval_mark + filtering ───────────────────────────────────────


class TestEvalMark:
    def test_marks_stored_on_function(self):
        @eval_mark("verdict", "fast")
        def eval_x(ctx):
            return 1.0

        assert eval_x._eval_marks == {"verdict", "fast"}

    def test_include_filter(self):
        @eval_mark("verdict")
        def eval_a(ctx):
            return 1.0

        @eval_mark("ioc")
        def eval_b(ctx):
            return 0.5

        results = eval_run([eval_a, eval_b], _ctx(), include=["verdict"])
        assert len(results) == 1
        assert results[0].name == "eval_a"

    def test_exclude_filter(self):
        @eval_mark("slow")
        def eval_slow(ctx):
            return 0.5

        def eval_fast(ctx):
            return 1.0

        results = eval_run([eval_slow, eval_fast], _ctx(), exclude=["slow"])
        assert len(results) == 1
        assert results[0].name == "eval_fast"

    def test_unmarked_passes_include(self):
        """Unmarked evals are excluded when include is set."""

        def eval_unmarked(ctx):
            return 1.0

        results = eval_run([eval_unmarked], _ctx(), include=["verdict"])
        assert len(results) == 0


# ── eval_run_batch ───────────────────────────────────────────────


class TestEvalRunBatch:
    def test_batch_across_contexts(self):
        def eval_steps(ctx):
            return min(ctx.step_count / 5, 1.0)

        ctx1 = _ctx(steps=[_step("a"), _step("b")])
        ctx2 = _ctx(steps=[_step("a")])
        table = eval_run_batch([eval_steps], [ctx1, ctx2])
        assert len(table) == 1
        assert table[0]["name"] == "eval_steps"
        assert table[0]["runs"] == 2
        assert table[0]["avg_score"] == pytest.approx(0.3)  # (0.4+0.2)/2

    def test_batch_with_marks_filter(self):
        @eval_mark("fast")
        def eval_a(ctx):
            return 1.0

        @eval_mark("slow")
        def eval_b(ctx):
            return 0.0

        table = eval_run_batch([eval_a, eval_b], [_ctx()], include=["fast"])
        assert len(table) == 1
        assert table[0]["name"] == "eval_a"

    def test_empty_batch(self):
        table = eval_run_batch([], [_ctx()])
        assert table == []


# ── Outcome-grading: EvalContext.artifacts ──────────────────────


class TestEvalContextArtifacts:
    def test_default_artifacts_is_empty_dict(self):
        ctx = EvalContext(steps=[])
        assert ctx.artifacts == {}

    def test_evaluator_can_read_artifacts(self):
        def eval_tests_passing(ctx):
            return ctx.artifacts.get("tests_passing", False)

        ctx = EvalContext(steps=[], artifacts={"tests_passing": True})
        results = eval_run([eval_tests_passing], ctx)
        assert results[0].score == 1.0

    def test_from_trajectory_dir_loads_artifacts_json(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "trajectory.json").write_text(
                json.dumps({"steps": [], "task": {}, "termination_reason": "done"})
            )
            (Path(d) / "artifacts.json").write_text(
                json.dumps({"tests_passing": True, "files_changed": ["a.py"]})
            )
            ctx = EvalContext.from_trajectory_dir(d)
            assert ctx.artifacts["tests_passing"] is True
            assert ctx.artifacts["files_changed"] == ["a.py"]

    def test_from_trajectory_dir_without_artifacts_json(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "trajectory.json").write_text(json.dumps({"steps": [], "task": {}}))
            ctx = EvalContext.from_trajectory_dir(d)
            assert ctx.artifacts == {}

    def test_from_trajectory_dir_preserves_trajectory_metadata(self):
        """Regression: ``trajectory.metadata`` (incl. harness_snapshot from
        TrajectoryRecorder) must survive the round-trip into
        ``EvalContext.metadata``. Previously only the four well-known
        top-level keys were copied, dropping harness_snapshot silently.
        """
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "trajectory.json").write_text(
                json.dumps(
                    {
                        "steps": [],
                        "task": {},
                        "run_id": "abc123",
                        "termination_reason": "done",
                        "metadata": {
                            "harness_snapshot": {
                                "schema_version": 1,
                                "extra": {"trial": "x"},
                            },
                            "user_field": "hello",
                        },
                    }
                )
            )
            ctx = EvalContext.from_trajectory_dir(d)
            # harness_snapshot must round-trip
            assert ctx.metadata["harness_snapshot"]["schema_version"] == 1
            assert ctx.metadata["harness_snapshot"]["extra"] == {"trial": "x"}
            # User-attached fields must round-trip
            assert ctx.metadata["user_field"] == "hello"
            # Top-level fields still take precedence
            assert ctx.metadata["run_id"] == "abc123"
            assert ctx.metadata["termination_reason"] == "done"


# ── EvalHook collectors ─────────────────────────────────────────


class TestEvalHookCollectors:
    def _run_hook(self, hook: EvalHook) -> None:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']
                ),
                tools=reg,
                state=DefaultState(max_steps=3),
                hooks=[hook],
                config=LoopConfig(max_steps=3),
            )
        )

    def test_collector_populates_artifacts(self):
        captured: dict = {}

        def collect_outcome(state):
            return {"tests_passing": True, "files_changed": 2}

        def eval_outcome(ctx):
            captured["artifacts"] = dict(ctx.artifacts)
            return ctx.artifacts.get("tests_passing", False)

        hook = EvalHook(evaluators=[eval_outcome], collectors=[collect_outcome])
        self._run_hook(hook)

        assert captured["artifacts"]["tests_passing"] is True
        assert captured["artifacts"]["files_changed"] == 2
        assert hook.results[0].score == 1.0

    def test_multiple_collectors_merge(self):
        def collect_a(state):
            return {"a": 1}

        def collect_b(state):
            return {"b": 2}

        seen: dict = {}

        def eval_seen(ctx):
            seen.update(ctx.artifacts)
            return 1.0

        hook = EvalHook(evaluators=[eval_seen], collectors=[collect_a, collect_b])
        self._run_hook(hook)
        assert seen == {"a": 1, "b": 2}

    def test_collector_exception_does_not_break_eval(self):
        def collect_broken(state):
            raise RuntimeError("boom")

        def collect_ok(state):
            return {"ok": True}

        def eval_ok(ctx):
            return ctx.artifacts.get("ok", False)

        hook = EvalHook(evaluators=[eval_ok], collectors=[collect_broken, collect_ok])
        self._run_hook(hook)
        # Broken collector skipped; good collector still applied.
        assert hook.results[0].score == 1.0

    def test_collector_must_return_dict(self):
        def collect_bad(state):
            return "not a dict"

        def eval_noop(ctx):
            return 1.0

        hook = EvalHook(evaluators=[eval_noop], collectors=[collect_bad])
        # Non-dict return is ignored, not raised.
        self._run_hook(hook)
        assert hook.results[0].score == 1.0
