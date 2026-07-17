# Behavioral evals: turn failures into release contracts

Looplet evals make harness behavior testable in the same workflow as normal
Python code:

1. **Case:** preserve the task that exposed a failure.
2. **Collector:** inspect the resulting world state independently of the
    agent's claim.
3. **Grader:** score the observed outcome against grader-only expectations.
4. **Required mark:** turn the behavior into a fail-closed CI gate.

Write functions named `eval_*` and Looplet discovers and runs them. Unlike
ordinary assertions, evals may return scores from 0–1 because some quality
dimensions are continuous. Required release contracts still produce ordinary
pass/fail exit codes.

The [network-free regression proof](regression-demo.md) shows this pipeline
end to end: one captured run, one tool fix, the same model responses, and a
required outcome grader moving from red to green.

```python
# eval_my_agent.py - discovered automatically by eval_discover()

def eval_tests_passed(ctx):
    """Did the agent get tests to pass?

    Outcome-grounded: read from `ctx.artifacts`, populated by a
    collector that re-runs the test suite (see "Trajectory-blind evals"
    below). Don't grep `ctx.steps` for "pytest" - a smarter model
    might use a different runner and pass tests anyway.
    """
    return ctx.artifacts.get("tests_passing", False)

def eval_step_cost(ctx):
    """Surface step count as a cost *metric*, not a quality score.

    `EvalResult.metrics` lets you plot cost-vs-quality without a
    fast-but-wrong run beating a slow-but-correct one.
    """
    return {"steps": float(ctx.step_count)}

def eval_ioc_quality(ctx):
    """Return multiple metrics at once."""
    return {"precision": 0.9, "recall": 0.75, "f1": 0.82}

def eval_reasoning_gaps(ctx, llm):
    """LLM-as-judge: are conclusions supported by data?"""
    resp = llm.generate(f"Score 0-1: {ctx.final_output} supported by {ctx.session_log_text}?")
    return float(resp.strip())
```

**Return anything:** `float`, `bool`, `str`, `dict`, or `EvalResult`.
The framework normalises. If your function takes an `llm` parameter,
the framework passes the judge LLM automatically.

## Trajectory-blind evals: grade outcomes, not process

The single biggest pitfall when writing evals is grading the *trajectory*
the model took instead of the *outcome* it produced. A real anecdote
from agent labs: a code-summarisation eval scored the model on whether
it read a specific list of files. The model was inferring those classes
from their usage elsewhere, which was a *better* strategy, and the eval marked
it down. The "restriction" preserved a 2024 trajectory as a permanent
ceiling.

**Rule of thumb:** if your eval indexes `ctx.tool_sequence` or greps
`ctx.steps` looking for tool calls by name, you are probably grading
process, not outcome. Two patterns to use instead.

### 1. Read `ctx.final_output` for the answer

```python
def eval_answer_correct(ctx):
    return ctx.final_output.get("answer") == ctx.task.get("expected")
```

The agent's `done()` arguments are the agent's own claim about the
result. Trust them at face value, then **check them against the world**
via artifacts.

### 2. Use collectors + `ctx.artifacts` for world-state

A *collector* is a callable `(state) -> dict[str, Any]` that runs once
at end-of-loop. Its return value is merged into `ctx.artifacts`,
where any evaluator can read it.

```python
from looplet import EvalHook

def collect_test_results(state):
    """Re-run the suite ourselves; don't ask the agent if it ran tests."""
    proc = subprocess.run(["pytest", "-q"], capture_output=True)
    return {"tests_passing": proc.returncode == 0}

def collect_repo_diff(state):
    """Snapshot what actually changed on disk."""
    diff = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True)
    return {"files_changed": diff.stdout.count("|"), "diff_text": diff.stdout}

def eval_tests_passed(ctx):
    return ctx.artifacts["tests_passing"]

def eval_changed_something(ctx):
    return ctx.artifacts["files_changed"] > 0

hook = EvalHook(
    evaluators=[eval_tests_passed, eval_changed_something],
    collectors=[collect_test_results, collect_repo_diff],
)
```

A collector that raises or returns a non-dict never breaks the agent run
because collectors are observers. `EvalHook` still records a synthetic
`collector:<name>` result with `label="error"`. CLI evaluation fails on
that result instead of reporting a false green. Multiple successful
collectors merge their dicts in order; later keys win.

For saved trajectories, drop an `artifacts.json` next to
`trajectory.json`. `EvalContext.from_trajectory_dir` loads it
automatically and fails loudly if it is malformed. New trajectories
also persist `session_log_text`, so an LLM judge sees the same evidence
online and after reload. Treat eval-run directories as sensitive prompt
evidence and apply the provenance recorder's `redact=` option where
needed:

```
traces/run_1/
├── trajectory.json
├── metrics.json
└── artifacts.json   ← {"tests_passing": true, "files_changed": 3}
```

For compatibility with older benchmark traces, a `metrics.json` file may
also contribute top-level `expected_*` keys to the offline grader task and
an `output` object when the trajectory has no terminal payload. This is a
legacy offline-harness convention, not data shown to a live agent. New eval
cases should use the top-level case `expected` field instead.

### When trajectory inspection *is* OK

Reading `ctx.tool_sequence` or `ctx.steps` is appropriate for:

- **Harness regression tests:** verify that *your hooks fired*, not
  that the model picked a particular tool.
- **Debugging:** find why a specific run went sideways.
- **Auditing:** record what the agent did without grading it.

If you find yourself writing `"pytest" in str(ctx.steps)` as a quality
signal, replace it with a collector that runs `pytest` and surfaces a
boolean artifact.

### Trust boundary: the agent must not own its oracle

Files under `case.task["files"]` are task inputs placed in the agent's
writable sandbox. They are **not** a protected release oracle: the agent
can read and modify them. Visible tests are useful guidance, but a serious
gate should invoke host-owned collector and grader code from the runner layer,
while keeping oracle data, paths, callables, and capabilities out of the task
and the `runtime` passed to `cartridge_to_preset()`. That runtime is available
to candidate resources and is not a secret channel.

Cartridge-shipped evals are the agent version's self-test contract. For
automatic cartridge evolution, keep a separate host-owned holdout suite and
runner: the candidate may edit its cartridge, but never receive or modify the
evaluator that decides promotion. Separate paths alone do not constrain
arbitrary same-user code; use OS or process isolation when the candidate is
untrusted.

## Attach to your loop

For live scoring during development:

```python
from looplet import EvalHook

hook = EvalHook(
    evaluators=[eval_tests_passed, eval_step_cost],
    collectors=[collect_test_results],   # populates ctx.artifacts
    verbose=True,                         # prints scores after each run
)
for step in composable_loop(..., hooks=[hook]):
    ...
print(hook.summary())          # "1 scored (avg 1.00), 1 labeled"
hook.save("evals/run_1.json")
```

## Discover and batch-run across saved trajectories

```python
from looplet import eval_discover, eval_run, EvalContext

evals = eval_discover("eval_my_agent.py")       # finds all eval_* functions
ctx = EvalContext.from_trajectory_dir("traces/run_1/")
results = eval_run(evals, ctx, judge_llm=my_judge)
for r in results:
    print(r.pretty())
```

The workflow: debug a run → preserve the case → collect the real outcome →
write a focused `eval_*` grader → require it in CI. Your debugging becomes a
behavioral contract instead of a trajectory snapshot.

> **Discovery scope.** `eval_discover` only collects functions *defined
> in* each `eval_*.py` file. Re-exports like `from looplet import
> eval_mark` are filtered out, so you can freely import decorators and
> helpers without them accidentally being run as evaluators.

## Distinguish "done" from hook-triggered early stops

Hooks that terminate the loop early (budget caps, source counters,
timeouts, quality gates) leave the agent without a `done()` call in the
trajectory. Evals should dispatch on `ctx.stop_reason`:

```python
def eval_completed_normally(ctx):
    """Agent called done() itself (not stopped by a hook)."""
    return ctx.completed          # shorthand for ctx.stop_reason == "done"

def eval_stopped_within_budget(ctx):
    """Either finished normally OR stopped by the budget hook (both are fine)."""
    return ctx.stop_reason in {"done", "budget_exceeded"}

def eval_not_hit_timeout(ctx):
    return ctx.stop_reason != "timeout"
```

`stop_reason` is populated from both live `EvalHook` runs (read from
state) and saved trajectories (read from `trajectory.json`). Hooks
should pass a meaningful label when they stop the loop:

```python
from looplet import HookDecision

class BudgetCap:
    def should_stop(self, state, step_num, new_entities):
        if self.tokens > self.budget:
            return HookDecision(stop="budget_exceeded")   # shows up as ctx.stop_reason
        return False
```

Returning a plain `True` from `should_stop` is still supported; it
records `stop_reason="hook_stop"`.

## Tag evals with marks for filtering

```python
from looplet import eval_mark

@eval_mark("verdict", "fast")
def eval_verdict_correct(ctx): ...

@eval_mark("ioc", "slow")
def eval_ioc_quality(ctx, llm): ...

# A skipped required grader is an integrity failure in the CLI.
@eval_mark("required")
def eval_release_gate(ctx): ...

# Run only "verdict" evals:
results = eval_run(evals, ctx, include=["verdict"])

# Skip "slow" evals in CI:
results = eval_run(evals, ctx, exclude=["slow"])
```

`required` uses the ordinary mark mechanism; it is not a second grader
type. In CLI runs, required graders must execute and meet the normal
`EvalResult.passed` boundary ($0.5$ for numeric scores), independently of
any stricter `--threshold` supplied for the run. Unmarked LLM judges remain
optional when no judge backend is configured. Evaluator/collector errors
and explicit failing labels always produce a non-zero CLI exit.

## Batch-run across multiple trajectories

```python
from looplet import eval_run_batch

contexts = [EvalContext.from_trajectory_dir(d) for d in trace_dirs]
table = eval_run_batch(evals, contexts)
for row in table:
    print(f"{row['name']:30s} avg={row['avg_score']:.2f}")
```

## Cases as data: write them by hand, run them with pytest

An **eval case** is just `task` + `expected` + tags. The cartridge runner
places the separate `expected` object at `ctx.task["expected"]` for every
grader **after the agent loop ends**; it is never included in the task
prompt sent to the agent. Persisted eval runs keep the agent-visible task
in `trajectory.json` and grader-only data in a sibling `expected.json`,
then restore the documented `ctx.task["expected"]` view when loaded. The
provenance record therefore remains honest about what the agent saw. Cases
live as JSON so you can hand-write the first few, grow the corpus from real
runs, and review them without a Python file.

`task["expected"]` is reserved in cartridge cases: put oracle data in the
top-level `expected` field. The runner rejects the ambiguous nested shape
rather than exposing it to the agent.

```json
// evals/cases/add_basic.json
{
  "id": "add_basic",
  "task": {"description": "Create math_utils.add() with a regression test"},
  "expected": {"tests_passing": true},
  "marks": ["smoke"],
  "notes": "Seed case; the simplest end-to-end coder run."
}
```

Browse the corpus from the CLI:

```bash
looplet eval cases ls evals/cases/
#   add_basic     [smoke      ] Create math_utils.add() with a regression test
#   multiply_fix  [regression ] Fix the multiply bug in calc.py
#
#   2 case(s)

looplet eval cases show evals/cases/ multiply_fix   # full JSON dump
```

Run them with stock pytest. The shortest path uses two helpers, with no
`pytest` import needed in your test file:

```python
# tests/test_evals.py
from looplet import assert_evals_pass, parametrize_cases


@parametrize_cases("evals/cases")
def test_coder(case, my_agent):           # `my_agent` = your own fixture
    ctx = my_agent.run(case)               # build a context however you like
    assert_evals_pass(ctx, "evals/")       # discovers eval_*.py and asserts
```

`parametrize_cases` carries each case's `marks` through, so `-k <id>`,
`-m <mark>`, `--lf`, IDE integration, and JUnit XML all work without a
custom plugin. `assert_evals_pass` runs the evaluators, collects any
failures, and raises `AssertionError` with each failed result's
`pretty()` block on its own line. Discovery is cached, so calling it
once per parametrized case is free.

If you want more control over selected evaluators, judge models, or individual
results, drop down to the primitives:

```python
import pytest
from looplet import (
    EvalContext, eval_discover, eval_run, load_cases, pytest_param_cases,
)

CASES = load_cases("evals/cases")
EVALS = eval_discover("evals/")


@pytest.mark.parametrize("case", pytest_param_cases(CASES))
def test_coder(case, my_agent):
    ctx: EvalContext = my_agent.run(case)
    results = eval_run(EVALS, ctx, judge_llm=my_agent.llm)
    failed = [r for r in results if not r.passed]
    assert not failed, "\n".join(r.pretty() for r in failed)
```

The same `EVALS` list also drives `EvalHook` for live grading and
`eval_cli` for CI batch runs. Write the eval once and use it three ways.

To save a case after a successful manual run:

```python
from looplet import EvalCase, save_case

save_case(
    EvalCase(
        id="multiply_fix",
        task={"description": "Fix the multiply bug in calc.py"},
        expected={"tests_passing": True},
        marks=["regression"],
        notes="Seen as a real failure on 2026-04-15.",
    ),
    "evals/cases/",
)
```

## CLI runner for CI

Like `pytest` with exit codes:

```bash
looplet eval traces/ --evals eval_agent.py --threshold 0.7 -v
```

```
  ✓ eval_verdict_correct           avg=1.00  min=1.00  max=1.00  (5 runs)
  ✗ eval_ioc_quality               avg=0.42  min=0.20  max=0.80  (5 runs)
  ✓ eval_no_tool_errors            avg=1.00  min=1.00  max=1.00  (5 runs)

  overall: 0.81
  threshold: 0.70  → PASS
```
