# Behavioral evals

An eval is a host-owned behavioral contract checked against an observed run outcome.
Define it as an ordinary Python function.
Your host owns execution and supplies the observed result.

## Start with one function

```python
from looplet import EvalContext, eval_assess

observed_context = EvalContext(
    steps=[],
    final_output={"answer": 42},
    task={"expected": {"answer": 42}},
)

def eval_answer(ctx):
    return ctx.final_output["answer"] == ctx.expected["answer"]

report = eval_assess([eval_answer], observed_context)
report.assert_passed()
```

This complete example grades a fixed observation; it does not execute an agent.
`observed_context` holds the observed output and grader-only expectations.
The [Quickstart](quickstart.md) builds this context from an actual run and adds pytest, capture, and replay.
A boolean is enough for a pass/fail contract.
Use `EvalResult` when you need scores, metrics, or explanations.
No files, saved cases, collectors, or cartridge are required.

The same functions serve Python, pytest, saved-run grading, and cartridge evaluation.
There is no second definition language or mandatory eval class.

## Keep a contract in one file

Put `eval_*` functions in a Python file when the contract needs version control or reuse.
Load that file without changing its functions:

```python
from looplet import eval_assess, load_evals

definitions = load_evals("eval_answer.py")
eval_assess(definitions.graders, observed_context).assert_passed()
```

Add ordinary `collect_*` functions to the same file when you need independent world-state evidence.
They run only when your host attaches them through an `EvalHook`.
Graders and collectors share one module import and its helper functions.
Collection remains separate from grading, so offline rescoring cannot repeat side effects.

For a growing corpus, keep JSON cases under `evals/cases/` and Python definitions under `evals/`.
`load_evals("evals/")` loads that directory through the same discovery code.
Existing separate grader and collector files remain supported.
Each case separates agent-visible `task` data from grader-only `expected` data.

Create a file-output example when you need a starter:

```bash
python -m pip install pytest
looplet eval init --python evals/
looplet eval check evals/ --json
python -m pytest evals/test_agent.py
```

```text
evals/
    cases/hello.json
    eval_outcome.py
    test_agent.py
```

`eval_outcome.py` contains a required grader and its outcome collector.
It checks the content of `result.txt`, not the agent's claim about that file.
Missing output fails the contract.
`test_agent.py` runs a Python agent with fixed model responses, then checks the collected file.
It needs no model key or service.
Replace its agent factory and backend with your own when you add your agent.
Keep tools and collectors bound to the same workspace.
Existing starter files are never overwritten.
Edit the functions and case data directly as the contract grows.

Omit `--python` to create definitions without a test.
The [Quickstart](quickstart.md#3-check-the-answer) provides complete executable files.
You can also check graders kept in an agent module with `looplet eval check first_loop.py`.
`looplet eval run` expects a cartridge directory, not the standalone `evals/` directory created above.
`--python` writes an ordinary pytest example; it does not add a Python runner to the CLI.

`eval check` accepts one file or a directory; saved cases are optional.
It imports trusted Python definitions.
Imports can execute authored code; use it only with definitions you trust.
The check does not call graders, collectors, models, or the agent.
It checks case shapes, seed paths, duplicate identities, imports, and supported signatures.
Its `looplet.eval-definitions` report describes definition validity, not whether an agent satisfies the contract.
It cannot prove that a grader is correct or that required checks were not deleted before discovery.

## Own execution explicitly

Use one owner for each operation:

| Situation | Recommended path |
| --- | --- |
| An observation needs grading | `eval_assess(graders, context)` |
| A live loop needs collection and grading | Attach `EvalHook`, then call `hook.assessment()` |
| Saved grades need a gate decision | `EvalReport.from_records(records)` |

The definition stays the same.
These paths differ because grading, collection, and reassessment are different operations.
Do not attach a grading hook and then run its graders again just to assert success.

For a case-driven loop with independent collection:

```python
from looplet import EvalHook, composable_loop, load_evals, save_eval_run, seed_case_workspace

bundle = load_evals("evals/", runtime={"project_root": str(project_root)})
case = bundle.cases[0]
seed_case_workspace(case, project_root)
hook = EvalHook(
    evaluators=bundle.graders,
    collectors=bundle.collectors,
    expected=case.expected,
)
for step in composable_loop(
    llm=llm,
    tools=tools,
    state=state,
    config=config,
    task={key: value for key, value in case.task.items() if key != "files"},
    hooks=[*hooks, hook],
):
    observe(step)
save_eval_run("eval-runs/hello", eval_hook=hook, case=case)
hook.assessment().assert_passed()
```

This example assumes your host already owns the backend, tools, state, configuration, hooks, and project directory.
Bind tools and collectors to the same project directory.
Use fresh state for each case.
The hook owns grading once; reassessment does not repeat collection or grading.
For an already observed, ungraded context, use `eval_assess(bundle.graders, ctx)` or `assert_evals_pass(ctx, bundle.graders)`.
For saved grades, use `EvalReport.from_records()`.
The existing primitives remain available for custom discovery and execution.

`load_evals()` uses strict imports by default and does not require a cartridge.
It returns the existing named tuple of `cases`, `graders`, and `collectors`.
Partial bundles remain useful for hosts that load only graders or collectors.
`load_cartridge_evals()` remains compatible and loads the same definitions from a cartridge's `evals/` directory.
Place this directory inside a cartridge to use the existing fresh-run CLI below.
The scaffold remains a self-test contract, not a protected promotion holdout.

## Choose one execution path

**Cartridge authors:** put cases, collectors, and graders under `evals/`, then
run the self-tests:

```bash
looplet eval run ./agent.cartridge --json --assessment-version 2
```

The Python equivalent is `run_cartridge_evals()`. This runner owns evaluation
and replaces embedded `EvalHook` observers with the self-test bundle's hook.
Policy hooks still run. The reference coder ships no implicit eval hook.

The CLI saves each invocation under a unique `.looplet/evals/<cartridge>-<id>/` directory in the current directory.
Human output prints the path; JSON reports it as `output_dir`.
Human output also prints failed graders' explanations and details beside their case and grader names.
Copy the printed per-case `Inspect:` commands to inspect individual traces; `output_dir` is their parent.
Use `--out eval-runs/run-001` for an explicit path, or `--no-save` to disable evidence persistence.
These options are mutually exclusive.
The Python API still requires `output_dir=` to persist runs.

Use a fresh explicit output directory for each run, such as `eval-runs/run-002` next.
The runner refuses existing selected case paths before any case executes.
This also applies with `capture=False` or `--no-capture`.
It preserves previous evidence and workspaces instead of deleting them.

Saved runs record agent model prompts/responses beside
trajectory, collected outcomes, grades, and grader-only expectations. Reload
grades with `load_eval_run()` and replay captured responses with `replay_loop()`
against a fresh tool registry/workspace. Replay executes fresh tool effects;
it is not deterministic simulation or checkpoint restoration. Judge-model calls
are separate host grading work and are not included in agent model-call capture.

Use `--no-capture` or `run_cartridge_evals(..., capture=False)` for a grading-only
record without persisted model calls. Such a record supports offline rescoring,
not captured-response replay. Persisted runs may contain sensitive data; the
Python API accepts `redact=...`, which scrubs upstream by default and redacts
saved evidence even with capture disabled. `redact_upstream=False` selects
record-only redaction. Redaction does not rewrite the sandbox's source files or
work products. Interrupted runs save available partial evidence before raising,
without inventing completion or missing grades.

On POSIX, captured case-run directories are created with owner-only permissions
(0700) before files are seeded or model calls begin. This does not prevent access
by same-user code or protect evidence uploaded to Git, artifact storage, or other
services. Keep captured runs out of public repositories and apply your host's
retention policy. Non-POSIX hosts must enforce equivalent access controls.

**Python hosts:** explicitly attach one `EvalHook` to your existing loop, then
save its record with `save_eval_run()`; see [Attach to your loop](#attach-to-your-loop).
Use the same `load_eval_run()` reader afterward. Live scoring, offline
rescoring, and pytest use the same graders, not different eval formats.

Case files, expected data, collectors, and graders belong to the evaluation
owner. Colocated cartridge cases are self-tests; protected promotion holdouts
remain host-owned. Fresh case workspaces are not an OS security boundary.
New persisted case directories own a wildcard `.gitignore` before seeding, including when model capture is disabled.
Existing ignore policies are preserved; tracked files and force-adds are not protected.
See [reviewed summaries](provenance.md#share-a-reviewed-summary) before sharing evidence.
Use the [CLI replay recipe](cli.md#replay-captured-responses) to replay one captured case in a new workspace.

The [network-free regression proof](regression-demo.md) shows this pipeline
end to end: one captured run, one tool fix, the same model responses, and a
required outcome grader moving from red to green.

```python
# eval_my_agent.py - discovered automatically by eval_discover()

from looplet import EvalResult, eval_mark

@eval_mark("required")
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
    return EvalResult(metrics={"steps": float(ctx.step_count)})

def eval_ioc_quality(ctx):
    """Return multiple metrics at once."""
    return EvalResult(metrics={"precision": 0.9, "recall": 0.75, "f1": 0.82})

def eval_reasoning_gaps(ctx, llm):
    """LLM-as-judge: are conclusions supported by data?"""
    resp = llm.generate(f"Score 0-1: {ctx.final_output} supported by {ctx.session_log_text}?")
    return float(resp.strip())
```

For new graders, return a **boolean** or **`EvalResult`**. Set `score=` when a
number should decide the gate; use `metrics=` for passive measurements.
`EvalResult(metrics={"f1": 0.82})` is not a gate, while
`EvalResult(score=0.82, metrics={"f1": 0.82})` is.

Numeric, string, and dictionary returns remain compatible. Dictionary score
inference from `f1`, `accuracy`, or `overall` is deprecated and warns; it still
preserves the legacy score. A dictionary's explicit `score` remains supported.
If a grader takes an `llm` parameter, the framework passes the judge backend
when supplied. Optional judges may be skipped; required judges may not.

## One assessment across interfaces

Use `eval_assess()` when you have an observed context that needs grading:

```python
from looplet import eval_assess, eval_discover, load_eval_run

graders = eval_discover("evals/", strict=True)
record = load_eval_run("eval-runs/run-001/case-1")
report = eval_assess(graders, record.context, case=record.case, threshold=0.7)
print(report.to_dict())
report.assert_passed()
```

This runs graders only. It does not execute the agent or repeat collection.
Use `hook.assessment()` after live grading to avoid running those graders twice.
Use `EvalReport.from_records(records, threshold=0.7)` to assess saved grades without rescoring.
These interfaces share the same result classification and gate policy.

A raw `ProvenanceSink` trace records execution, not a complete grading context.
Use `save_eval_run(..., eval_hook=hook, recorder=recorder)` to retain expectations, artifacts, grades, and captured responses.
Expectations stay in grader-only evidence, not the agent-visible task.
The [Quickstart](quickstart.md#4-capture-the-same-agent) shows capture, persistence, reload, and reassessment together.
The CLI accepts either a single saved run directory or a directory containing immediate child runs.

Version 2 is the default for Python assessment APIs, `assert_evals_pass()`, and both CLI routes.
It rejects empty observations, empty grader selections, duplicate results, and gates without any verdict-producing check.
Passive metrics and optional skipped judges remain visible but cannot establish a passing gate alone.
Required missing, skipped, or measurement-only checks fail closed.
Collector and grader errors also fail closed.

`report.valid` distinguishes usable grading from evaluation integrity failures.
A valid assessment can fail because the agent produced a wrong answer.
`report.passed` is the release decision.
Completion remains a separate case fact; it does not imply correctness or become an implicit requirement.
Require completion explicitly in a grader when your contract needs it.

Version 1 retains the previous gate semantics and JSON fields.
Pass `version=1` to Python assessment APIs for compatibility.
Use `--assessment-version 1` on either CLI route for the same legacy policy.
The existing `--summary-version` flag remains an alias.
These options change acceptance rules and JSON fields, not just output formatting.
An empty assertion suite now fails by default instead of silently succeeding.
Raw `eval_run()` lists, `eval_run_batch()` tables, and hook text reports keep their existing shapes.

Hooks preserve the pre-grading inventory in the existing run metadata.
`EvalReport.from_records()` uses that inventory when no explicit `grader_manifest` is supplied.
Older records need an explicit trusted manifest or fresh grading with `eval_assess()`.
Supply the release threshold and version explicitly when comparing assessments.
Snapshots returned by `to_dict()`, `cases`, and `grader_manifest` are independent copies.

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
    return ctx.final_output.get("answer") == ctx.expected.get("answer")
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

Collector failures remain in the run record when graders are re-run offline.
Rescoring does not repeat collection or clear a failed host observation.
`load_eval_run()` also restores collector failures from older records whose
trajectory metadata predates failure retention.

Persist the collected outcome with `save_eval_run()`, then reload through
`load_eval_run()`. Its `context.artifacts` has the same shape online and
offline; malformed evidence fails loudly. New trajectories
also persist `session_log_text`, so an LLM judge sees the same evidence
online and after reload. Treat eval-run directories as sensitive prompt
evidence and apply the provenance recorder's `redact=` option where
needed:

```text
eval-runs/run_1/
├── artifact.json
├── trajectory.json
├── evals.json
├── expected.json
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

`run_cartridge_evals()` keeps no-output sandboxes available for inspection and
returns their path in each `EvalRunRecord`. Call `record.cleanup()` when the
record is no longer needed, or use the record as a context manager. Records
created with `output_dir=` own persisted evidence and are never removed by
`cleanup()`.

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

Discovery freezes only the inventory it finds; it cannot detect a grader deleted before discovery.
Keep an expected grader inventory outside candidate control when deletion detection matters.
Recorded inventories are audit data, not authenticated authority against editable evidence.

## Attach to your loop

For a Python host, evaluation is explicit and attached once:

```python
from looplet import EvalHook, save_eval_run

hook = EvalHook(
    evaluators=[eval_tests_passed, eval_step_cost],
    collectors=[collect_test_results],   # populates ctx.artifacts
    verbose=True,                         # prints scores after each run
)
for step in composable_loop(..., hooks=[hook]):
    ...
print(hook.summary())          # "1 scored (avg 1.00), 1 labeled"
report = hook.assessment()
save_eval_run("eval-runs/run_1", eval_hook=hook)
report.assert_passed()
```

`EvalHook.save()` remains a deprecated, one-way standalone report for old
callers. It is not another durable format to choose from. `promote_to_offline()`
is a compatibility alias for saving a fixture. It does not promote an agent candidate.

## Discover and batch-run across saved trajectories

```python
from looplet import eval_discover, eval_run, load_eval_run

evals = eval_discover("eval_my_agent.py")       # finds all eval_* functions
record = load_eval_run("eval-runs/run_1")
results = eval_run(evals, record.context, judge_llm=my_judge)
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
exposes the separate `expected` object as `ctx.expected` for every
grader **after the agent loop ends**; it is never included in the task
prompt sent to the agent. Persisted eval runs keep the agent-visible task
in `trajectory.json` and grader-only data in a sibling `expected.json`,
then restore the documented `ctx.task["expected"]` view when loaded. The
provenance record therefore remains honest about what the agent saw. Cases
live as JSON so you can hand-write the first few, grow the corpus from real
runs, and review them without a Python file.

`ctx.expected` keeps the existing `ctx.task["expected"]` view compatible.
It does not add protected data to the agent's task.

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
looplet eval cases ls evals/
#   add_basic     [smoke      ] Create math_utils.add() with a regression test
#   multiply_fix  [regression ] Fix the multiply bug in calc.py
#
#   2 case(s)

looplet eval cases show evals/ multiply_fix   # full JSON dump
```

Use the same eval root as `eval init` and `eval check`, or pass a direct case directory or JSON/JSONL file.
Browsing does not import graders. Listings preview the task description, goal, or notes in that order.

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
failures, and raises `AssertionError` with the shared assessment's failures.
Path-based discovery is strict and cached.
Import failures or duplicate grader names cannot produce a partial passing suite.

The fixture's `run(case)` should return an observed, ungraded context when
`assert_evals_pass()` owns grading. Do not attach an eval hook and then invoke
the same graders again just to assert their results. Metric-only results and
optional skipped judges are neutral; required, errored, missing, or failing
graders fail the assertion. They cannot satisfy a version 2 gate without a verdict-producing check.
Numeric scores use the default $0.5$ boundary; `threshold=` selects a stricter gate.

If you want more control over selected evaluators, judge models, or individual
results, drop down to the primitives:

```python
import pytest
from looplet import (
    EvalContext, assert_evals_pass, eval_discover, load_cases, pytest_param_cases,
)

CASES = load_cases("evals/cases")
EVALS = eval_discover("evals/")


@pytest.mark.parametrize("case", pytest_param_cases(CASES))
def test_coder(case, my_agent):
    ctx: EvalContext = my_agent.run(case)
    assert_evals_pass(ctx, EVALS, judge_llm=my_agent.llm)
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

```text
  ✓ eval_verdict_correct           avg=1.00  min=1.00  max=1.00  (5 runs)
  ✗ eval_ioc_quality               avg=0.42  min=0.20  max=0.80  (5 runs)
  ✓ eval_no_tool_errors            avg=1.00  min=1.00  max=1.00  (5 runs)

  overall: 0.81
    threshold: 0.70  → FAIL
```

Both CLI paths default to `--threshold 0.5`, matching pytest. A scored result
below the threshold fails its run; an average cannot hide it. Aggregate scores
are descriptive only. An explicit `--threshold 0` retains the old numeric
default, but required graders still enforce their $0.5$ floor.

Both CLI routes emit the same transient `looplet.eval-summary` report with `--json`:

```bash
looplet eval run ./agent.cartridge --threshold 0.7 --json --assessment-version 2 > eval-summary.json
looplet eval eval-runs/run-001/ --evals evals/ --threshold 0.7 --json --assessment-version 2
jq -e '.schema == "looplet.eval-summary" and .version == 2 and .valid and .passed' \
    eval-summary.json
```

The first command executes fresh agent runs and collects outcomes.
The second rescores existing observations; it does not repeat execution or collection.
Ordinary grader diagnostic output goes to stderr in JSON mode.
Partial trajectory load failures remain visible in the assessment and produce a failing exit code.

The command itself exits non-zero exactly when `passed` is false. The report
includes the trusted pre-run `grader_manifest`, case IDs and marks, and one
result record per expected grader. Each result carries its grader marks,
`required`, `required_status`, and one closed `state` value:

| State | Meaning |
| --- | --- |
| `pass` | The result meets its required boundary and the CLI threshold. |
| `explicit_fail` | The grader returned a failing label, an invalid empty result, or a required numeric score below $0.5$. |
| `threshold_fail` | A numeric score is valid but below `--threshold`. |
| `skipped` | The grader did not run, normally because no judge model was configured. Required skipped graders fail the report. |
| `missing` | A grader discovered before execution has no run result. This is an integrity failure even when candidate-editable output omits it. |
| `collector_error` | An outcome collector raised or returned the wrong shape. |
| `grader_error` | A grader raised. |
| `metric_only` | The grader emitted metrics but no pass/fail score or label. Required metric-only graders fail the report. |

`required_status` is `not_required`, `satisfied`, or `failed`. A required score
can satisfy the ordinary $0.5$ required boundary while still receiving
`threshold_fail` under a stricter CLI threshold. Collector and grader errors
include an `error` string and can never serialize as passing.

Version 2 retains these result states and adds top-level and per-case `valid`.
Unlike version 1, a required behavioral failure is not itself an integrity failure.
Missing checks, skipped required checks, collector failures, and grader errors still invalidate evaluation.
Measurement-only assessments are not release gates in version 2.

This summary is a decision report, not a persisted run format.
Fresh CLI evals save durable evidence by default; find it through `output_dir`.
Python hosts use `save_eval_run()` or `run_cartridge_evals(..., output_dir=...)`.
Preserve the saved run's `artifact.json` descriptor. The JSON report intentionally omits task data,
expected data, prompts, responses, and artifact values.
Grader-authored explanations and details remain report content; graders must not disclose protected data there.
