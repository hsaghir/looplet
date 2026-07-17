# Choose the right experiment

Agent changes are easy to compare and hard to interpret. Before running an
eval, name the variable you want to study and decide what must remain fixed.

## Decision table

| Question | Use | Hold fixed | What it does not prove |
| --- | --- | --- | --- |
| Did changed tool or hook code fix this captured run? | Captured-response replay | Recorded model responses | That a future model run chooses the same actions |
| Does loop wiring dispatch, stop, and grade correctly? | Scripted backend | All model responses | Real model quality |
| Does a prompt or model change improve outcomes? | Fresh sampled runs | Cases, graders, environment, sampling policy where possible | Universal performance from one sample |
| Does code behave correctly against a controlled external system? | Mock, fixture, or sandbox | External side effects | Production service behavior outside the simulation |
| Is a release candidate safe to promote? | Fresh cases plus host-owned holdouts and graders | Promotion policy and isolated oracle | Protection from candidate code without an OS or process boundary |

Do not choose replay because it is cheap. Choose it when fixed model responses
are the correct control for the question.

## Captured-response replay

Replay is useful for changes below the model decision boundary:

- tool implementation;
- hook behavior;
- permission evaluation;
- state mutation;
- parsing and dispatch;
- loop runtime behavior.

It reuses recorded model responses while running fresh harness code. Tools,
clocks, networks, randomness, filesystem state, permissions, and other side
effects run again.

```python
result = replay_loop(
    "traces/failure/trajectory.json",
    tools=fixed_tools,
    hooks=fixed_hooks,
)
```

A red-to-green grader demonstrates that the changed execution produced a
different observed outcome under the same model decisions. It does not show
that a changed prompt causes better decisions. Run the
[network-free proof](regression-demo.md) for the complete example and its
limitations.

## Scripted backends

Use `MockLLMBackend` or `AsyncMockLLMBackend` for deterministic harness
mechanics:

```python
from looplet.testing import MockLLMBackend


llm = MockLLMBackend(responses=[tool_call_json, done_json])
```

This is appropriate for:

- schema validation and dispatch;
- hook order and decisions;
- stop reasons;
- collector and grader plumbing;
- persistence and reload behavior;
- required-check exit semantics.

It is not evidence that a live model will emit those responses. Keep scripted
harness tests fast and run them with ordinary pytest.

## Fresh sampled runs

Use new provider calls when changing anything that can alter model decisions:

- system prompt or task wording;
- model or serving connection;
- tool name, description, or schema;
- available context or memory;
- temperature or sampling policy;
- context compaction visible to the model.

Record at least:

- exact model identifier requested;
- serving endpoint or provider path;
- prompt and tool schema versions;
- sampling settings;
- case identifiers and grader versions;
- timestamp and relevant external-system version;
- raw per-case outcomes, not only an aggregate.

Matching a requested model family across different serving paths does not make
the underlying systems identical. State that limitation when comparing
harnesses.

One generation per case is useful for debugging but weak evidence for a
quality claim. Repeat noisy cases, preserve failures, and avoid presenting
small score differences as durable rankings without uncertainty analysis.

## Mocks and sandboxes

Use a mock when the contract is the request your code sends or the response it
handles. Use a sandbox when real side effects matter but must remain contained.

Examples:

| Outcome | Appropriate control |
| --- | --- |
| File contents | Fresh temporary directory |
| Git mutation | Disposable repository |
| SQL write | Transaction or isolated database |
| Shell command | Restricted process and temporary workspace |
| HTTP integration | Recorded fixture for mechanics; test service for integration |
| Untrusted candidate code | OS or process isolation with withheld oracle capabilities |

Directory separation is organization, not a security boundary, when arbitrary
same-user code can read the filesystem or inspect the runner process.

## Collect outcomes, not historical routes

A collector runs after the loop and inspects the resulting world. A grader
compares that evidence with grader-only expectations.

### File artifact

```python
import json
from pathlib import Path


def collect_report(state, runtime):
    path = Path(runtime["project_root"]) / "report.json"
    if not path.is_file():
        return {"report_exists": False, "report": None}
    return {
        "report_exists": True,
        "report": json.loads(path.read_text()),
    }


def eval_report(ctx):
    return ctx.artifacts["report"] == ctx.task["expected"]["report"]
```

### Command result

```python
import subprocess


def collect_tests(state, runtime):
    completed = subprocess.run(
        ["pytest", "-q"],
        cwd=runtime["project_root"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "tests_exit_code": completed.returncode,
        "tests_output": completed.stdout[-4000:],
    }


def eval_tests_pass(ctx):
    return ctx.artifacts["tests_exit_code"] == 0
```

### Structured product state

```python
def collect_order(state, runtime):
    order = runtime["orders"].get(runtime["order_id"])
    return {
        "order_status": order.status,
        "charge_count": len(order.charges),
    }


def eval_order_completed_once(ctx):
    return (
        ctx.artifacts["order_status"] == "completed"
        and ctx.artifacts["charge_count"] == 1
    )
```

The runner owns collectors and runtime capabilities. Do not expose a protected
database client, hidden path, grader callable, or expected result through the
agent task, cartridge resource, or candidate-visible runtime.

## Separate quality from cost

Steps, tokens, latency, and provider cost are useful metrics. They should not
let a fast but wrong run outrank a correct run.

Use required graders for release invariants, scored graders for continuous
quality, and `EvalResult.metrics` or operational telemetry for efficiency.
Compare cost only among runs that meet the required quality boundary.

## Layer CI by evidence type

A practical pipeline separates deterministic mechanics from sampled behavior:

```yaml title=".github/workflows/agent-evals.yml"
- name: Harness mechanics
  run: uv run pytest -q tests/test_harness.py

- name: Behavioral gate
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    OPENAI_MODEL: ${{ vars.OPENAI_MODEL }}
  run: >-
    uv run looplet eval run ./agent.cartridge
    --out ./eval-runs --threshold 1.0
```

Decide whether a provider outage should block a release, retry, or move a job
to a separate required environment. Do not convert unavailable evidence into a
passing score.

Persist enough evidence to answer:

1. what candidate and cases ran;
2. what the model and tools actually did;
3. what world state collectors observed;
4. which grader version made the decision;
5. why any case was skipped or unavailable.

## Holdout boundary

Cartridge-shipped evals are versioned self-tests. They are valuable, but a
candidate that can edit the cartridge can also edit those tests.

For promotion decisions, keep holdout cases, expected data, collectors,
graders, and capabilities in a host-owned runner. Withhold them from the task,
runtime, resources, tools, prompt, and writable files. If candidate code is
untrusted, enforce the boundary with OS or process isolation.

## Review checklist

Before accepting an experiment result:

1. Is the tested variable named precisely?
2. Is the chosen method capable of varying that variable?
3. Are important serving paths and external systems recorded?
4. Does the grader inspect an independent outcome?
5. Are required checks fail-closed on errors and missing evidence?
6. Are efficiency metrics separated from correctness?
7. Are sample size and uncertainty stated honestly?
8. Can another engineer inspect the cases, artifacts, and grader version?