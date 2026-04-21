# Evals — score your agent as you debug it

Agent evals work like pytest: write functions named `eval_*`, and the
framework discovers and runs them. The difference from tests: evals
return **scores** (0–1), not just pass/fail, because agent output
quality is a spectrum.

```python
# eval_my_agent.py — discovered automatically by eval_discover()

def eval_tests_passed(ctx):
    """Did the agent get tests to pass?"""
    for s in reversed(ctx.steps):
        if s.tool_call.tool == "bash" and "pytest" in s.tool_call.args.get("command", ""):
            return s.tool_result.data.get("exit_code") == 0
    return False

def eval_efficiency(ctx):
    """Score 0-1: fewer steps = better."""
    return min(5 / max(ctx.step_count, 1), 1.0)

def eval_ioc_quality(ctx):
    """Return multiple metrics at once."""
    return {"precision": 0.9, "recall": 0.75, "f1": 0.82}

def eval_reasoning_gaps(ctx, llm):
    """LLM-as-judge: are conclusions supported by data?"""
    resp = llm.generate(f"Score 0-1: {ctx.final_output} supported by {ctx.session_log_text}?")
    return float(resp.strip())
```

**Return anything** — `float`, `bool`, `str`, `dict`, or `EvalResult`.
The framework normalises. If your function takes an `llm` parameter,
the framework passes the judge LLM automatically.

## Attach to your loop

For live scoring during development:

```python
from openharness import EvalHook

hook = EvalHook(
    evaluators=[eval_tests_passed, eval_efficiency],
    verbose=True,   # prints scores after each run
)
for step in composable_loop(..., hooks=[hook]):
    ...
print(hook.summary())          # "2 scored (avg 0.90)"
hook.save("evals/run_1.json")
```

## Discover and batch-run across saved trajectories

```python
from openharness import eval_discover, eval_run, EvalContext

evals = eval_discover("eval_my_agent.py")       # finds all eval_* functions
ctx = EvalContext.from_trajectory_dir("traces/run_1/")
results = eval_run(evals, ctx, judge_llm=my_judge)
for r in results:
    print(r.pretty())
```

The workflow: debug a run → notice a failure pattern → write a 5-line
`eval_*` function → it runs automatically on every future run. Your
debugging becomes your eval suite.

## Tag evals with marks for filtering

```python
from openharness import eval_mark

@eval_mark("verdict", "fast")
def eval_verdict_correct(ctx): ...

@eval_mark("ioc", "slow")
def eval_ioc_quality(ctx, llm): ...

# Run only "verdict" evals:
results = eval_run(evals, ctx, include=["verdict"])

# Skip "slow" evals in CI:
results = eval_run(evals, ctx, exclude=["slow"])
```

## Batch-run across multiple trajectories

```python
from openharness import eval_run_batch

contexts = [EvalContext.from_trajectory_dir(d) for d in trace_dirs]
table = eval_run_batch(evals, contexts)
for row in table:
    print(f"{row['name']:30s} avg={row['avg_score']:.2f}")
```

## CLI runner for CI

Like `pytest` with exit codes:

```bash
openharness eval traces/ --evals eval_agent.py --threshold 0.7 -v
```

```
  ✓ eval_verdict_correct           avg=1.00  min=1.00  max=1.00  (5 runs)
  ✗ eval_ioc_quality               avg=0.42  min=0.20  max=0.80  (5 runs)
  ✓ eval_no_tool_errors            avg=1.00  min=1.00  max=1.00  (5 runs)

  overall: 0.81
  threshold: 0.70  → PASS
```
