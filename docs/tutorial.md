# Tutorial: build a testable harness

This tutorial builds the smallest useful Looplet artifact: a report
agent whose tool writes `report.json`, plus a behavioral contract that
checks the file independently.

The complete source ships under
[`examples/regression_demo/`](https://github.com/hsaghir/looplet/tree/master/examples/regression_demo).
Run it at any point with:

```bash
uv run python examples/regression_demo/run_demo.py
```

## 1. Start with a cartridge

```text
report_agent.cartridge/
├── cartridge.json
├── config.yaml
├── runtime.yaml
├── prompts/system.md
├── resources/project_dir.py
├── tools/
│   ├── publish_report/{tool.yaml, execute.py}
│   └── done/{tool.yaml, execute.py}
└── evals/
    ├── cases/profit_math.json
    ├── collect_outcome.py
    └── eval_correctness.py
```

The manifest identifies the artifact and schema:

```json title="cartridge.json"
{
  "name": "report_agent",
  "schema_version": 2,
  "description": "Write a small financial report."
}
```

Contract-level loop configuration stays small:

```yaml title="config.yaml"
max_steps: 3
done_tool: done
```

Host/runtime policy lives separately:

```yaml title="runtime.yaml"
use_native_tools: false
```

That split lets two hosts choose different provider or context behavior
without pretending they are different agent contracts.

## 2. Write the prompt and tool

```markdown title="prompts/system.md"
# Report agent

You publish a small financial report. Call `publish_report` with the
revenue and cost from the task, then call `done`.
```

The schema is data:

```yaml title="tools/publish_report/tool.yaml"
name: publish_report
description: Write report.json from integer revenue and cost inputs.
parameters:
  revenue:
    type: integer
  cost:
    type: integer
requires:
  - project_dir
```

The implementation is ordinary Python:

```python title="tools/publish_report/execute.py"
import json
from pathlib import Path

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, revenue: int, cost: int) -> dict:
    root = Path(ctx.resources["project_dir"])
    report = {
        "revenue": revenue,
        "cost": cost,
        "profit": revenue - cost,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"written": "report.json", "profit": report["profit"]}
```

`resources/project_dir.py` binds the runner's fresh sandbox:

```python
def build(runtime=None):
    return (runtime or {}).get("project_root", ".")
```

Inspect the loaded harness without calling a model:

```bash
looplet describe ./report_agent.cartridge
```

## 3. Express the scenario as data

```json title="evals/cases/profit_math.json"
{
  "id": "profit_math",
  "task": {
    "goal": "Publish report.json for revenue 120 and cost 80, then finish."
  },
  "expected": {
    "profit": 40
  },
  "marks": ["smoke", "regression"]
}
```

The top-level `expected` mapping is for graders. The cartridge runner
does not include it in the task sent to the agent. Persisted runs store
it in `expected.json`, separate from the agent-visible
`trajectory.json`.

If a case needs starting files, put a `{path: contents}` mapping under
`task.files`; the runner seeds those files in a fresh per-case
workspace and removes the mapping before building the model prompt.

## 4. Observe the world independently

The tool claims it wrote a correct report. Do not grade that claim.
Read the file:

```python title="evals/collect_outcome.py"
import json
from pathlib import Path


def collect_report(state, runtime) -> dict:
    root = Path(runtime["project_root"])
    path = root / "report.json"
    if not path.is_file():
        return {"report_exists": False}
    report = json.loads(path.read_text())
    return {
        "report_exists": True,
        "observed_profit": report.get("profit"),
    }
```

Collector functions are discovered from `evals/collect_*.py`. A
collector receives final state and may request the runtime mapping. Its
dictionary is merged into `EvalContext.artifacts`.

Collectors are observers, so an exception does not crash the live
agent. It does become an explicit error result, which prevents the CLI
from reporting a false green.

## 5. Define the behavioral contract

```python title="evals/eval_correctness.py"
from looplet import EvalContext, eval_mark


@eval_mark("required")
def eval_profit_is_correct(ctx: EvalContext):
    expected = ctx.task["expected"]["profit"]
    return ctx.artifacts.get("observed_profit") == expected


@eval_mark("required")
def eval_completed(ctx: EvalContext):
    return ctx.completed
```

Grader functions are discovered from `evals/eval_*.py`. `required`
uses the normal mark mechanism but adds fail-closed CLI semantics: the
grader must execute and pass. A skipped or errored required grader is a
failure, not missing data to ignore.

This contract permits any model trajectory that creates the correct
artifact and completes. It does not require a specific reasoning string
or tool order.

## 6. Run it without a network

Use `MockLLMBackend` in a normal test. The runner loads the cartridge,
seeds each case, runs the live loop, invokes collectors and graders,
and persists one self-contained record per case.

```python title="tests/test_report_agent.py"
import json

from looplet import run_cartridge_evals
from looplet.testing import MockLLMBackend


def test_report_contract(tmp_path):
    responses = [
        json.dumps({
            "tool": "publish_report",
            "args": {"revenue": 120, "cost": 80},
            "reasoning": "publish",
        }),
        json.dumps({
            "tool": "done",
            "args": {"summary": "published report.json"},
            "reasoning": "finished",
        }),
    ]

    records = run_cartridge_evals(
        "report_agent.cartridge",
        llm=MockLLMBackend(responses=responses),
        output_dir=tmp_path / "runs",
    )

    scores = {result.name: result.score for result in records[0].results}
    assert scores["eval_profit_is_correct"] == 1.0
    assert records[0].context.artifacts["observed_profit"] == 40
```

This is a harness regression test, not a model-quality benchmark. The
scripted responses make tool wiring and outcome grading deterministic.
Evaluate a real prompt or model change with fresh sampled model runs.

## 7. Run the contract with a model

Configure an OpenAI-compatible endpoint and execute the shipped cases:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=...
export OPENAI_MODEL=...

looplet eval run ./report_agent.cartridge \
  --out ./eval-runs \
  --threshold 1.0
```

Useful flags:

- `--case profit_math` runs one case;
- `--max-steps N` overrides the case budget;
- `--judge` enables graders whose signature requests an `llm`;
- `--judge-model NAME` uses a separate judge model;
- `--threshold VALUE` fails when any scored grader falls below the value.

Required graders, explicit failures, collector errors, malformed
records, unknown cases, and empty required suites produce a non-zero
exit.

## 8. Capture, change, replay

For an interesting live failure, attach `ProvenanceSink` while running
the preset. The shipped
[`run_demo.py`](https://github.com/hsaghir/looplet/blob/master/examples/regression_demo/run_demo.py)
shows the complete wiring:

1. wrap the model backend;
2. attach the trajectory recorder and `EvalHook`;
3. persist the v1 run with `save_eval_run()`;
4. load fresh v2 tools and collectors;
5. call `replay_loop()` on the v1 trace;
6. persist and compare the v2 outcome.

Review the harness change structurally:

```bash
looplet diff ./report-v1.cartridge ./report-v2.cartridge --show
```

Captured-response replay is useful here because the changed component
is tool code. It keeps the model calls fixed while that code executes
again. It would not establish that a prompt edit causes better model
decisions; that requires new runs.

## 9. Put it in CI

For deterministic harness mechanics, run the pytest test above. For
sampled agent behavior, run the CLI against the model and cases your
release process permits:

```yaml title=".github/workflows/agent-evals.yml"
- name: Harness regression tests
  run: uv run pytest -q tests/test_report_agent.py

- name: Behavioral gate
  env:
    OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    OPENAI_MODEL: ${{ vars.OPENAI_MODEL }}
  run: >-
    uv run looplet eval run ./report_agent.cartridge
    --out ./eval-runs --threshold 1.0
```

For serious promotion decisions, keep holdout collectors, graders, expected
data, and capabilities in a host-owned runner. Do not pass them through the
candidate task or runtime. If candidate code is untrusted, enforce the boundary
with OS or process isolation rather than directory layout alone.

## What to build next

- Add a [quality-gate hook](hooks.md) that blocks premature `done()`.
- Add [redacted provenance](provenance.md) for failures worth keeping.
- Grow cases from real incidents, not speculative taxonomies.
- Keep outcome graders separate from efficiency metrics such as steps,
  tokens, or latency.
- Read the [pitfalls](pitfalls.md) before adding concurrency,
  permissions, or long-context compaction.
