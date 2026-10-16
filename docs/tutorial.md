# Tutorial: check a report agent

This guide adds a test to an agent that writes `report.json`.
A collector reads the file after the loop.
A grader checks the report against an expected value.
These checks do not depend on the model's claim that it succeeded.

Complete the [Quickstart](quickstart.md) first.
This guide uses a cartridge to keep the agent and its tests in one directory.
A cartridge is optional for normal loop use.

The complete example is in
[`examples/regression_demo/`](https://github.com/hsaghir/looplet/tree/master/examples/regression_demo).
The installed package includes its files.
You do not need a repository checkout or a model key.

## 1. Use the example cartridge

Run these commands in your project directory.
Use new paths for the generated demo and the copied cartridge:

```bash
looplet-proof --out ./report-demo
cp -R ./report-demo/harness-v2.cartridge ./report_agent.cartridge
```

The first command runs the offline regression example using bundled files.
The second copies its corrected cartridge into your project.
The example includes the `done` tool and the full file structure.
The sections below show the files that define the report and its test.

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

The manifest names the agent and selects the schema version:

```json title="cartridge.json"
{
  "name": "report_agent",
  "schema_version": 2,
  "description": "Write a small financial report."
}
```

The loop configuration sets its step limit and completion tool:

```yaml title="config.yaml"
max_steps: 3
done_tool: done
```

Change the copied `runtime.yaml` to select the protocol for this example's fixed responses:

```yaml title="runtime.yaml"
use_native_tools: false
```

The test below uses the JSON-text protocol.
The library uses native tool calls by default.
The separate runtime file lets a host select its protocol without a change to the agent's task.

## 2. Write the prompt and tool

```markdown title="prompts/system.md"
# Report agent

You publish a small financial report. Call `publish_report` with the
revenue and cost from the task. Then call `done`.
```

The tool schema describes its arguments and resource:

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

The resource function returns the workspace that the host provides:

```python title="resources/project_dir.py"
def build(runtime=None):
    return (runtime or {}).get("project_root", ".")
```

Inspect the cartridge without a model call:

```bash
looplet describe ./report_agent.cartridge
```

## 3. Define the test case

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

The `expected` mapping is for graders.
The host does not send it in the model's task.
Saved runs put it in `expected.json`, separate from the agent's `trajectory.json`.

Put initial files in `task.files` as a `{path: contents}` mapping.
The host writes them in a fresh workspace for each case.
It removes the mapping before it builds the model prompt.

## 4. Read the result independently

Do not use the tool's success claim as the expected result.
Read the actual file:

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

The host loads collectors from `evals/collect_*.py`.
Each collector receives final state and can request the runtime mapping.
Its dictionary becomes part of `EvalContext.artifacts`.

A collector error does not crash the agent.
It becomes an error result, so the command cannot report success.

## 5. Define the required checks

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

The host loads graders from `evals/eval_*.py`.
A `required` grader must run and pass.
A skipped required grader or a grader error makes the command fail.

These checks permit any tool sequence that creates the correct report and completes the task.
They do not require specific model text or a fixed tool order.

## 6. Run it without a network

The test uses `MockLLMBackend` with fixed responses.
The host loads the cartridge and prepares a workspace for each case.
It runs the loop, collectors, and graders.
It saves one record per case.

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

Run the test from your project directory:

```bash
python -m pip install pytest
python -m pytest tests/test_report_agent.py
```

This test checks the tools and graders under fixed model responses.
It does not measure model quality.
A prompt or model change needs new model runs.

## 7. Check results with a real model

Install and configure a provider as described in [Install](install.md).
Set `use_native_tools: true` in `runtime.yaml` if the provider supports native calls.
Then run the case:

```bash
looplet eval run ./report_agent.cartridge \
  --out ./eval-runs \
  --threshold 1.0
```

| Option | Effect |
| --- | --- |
| `--case profit_math` | Run one case |
| `--max-steps N` | Override the step limit |
| `--judge` | Enable graders that request an `llm` |
| `--judge-model NAME` | Use a separate model for those graders |
| `--threshold VALUE` | Fail if a scored grader falls below the value |

The command returns a non-zero exit code if a required check fails.
Collector errors, invalid records, unknown cases, and empty required suites also make it fail.

## 8. Capture, change, replay

Attach `ProvenanceSink` to save a run that you need to investigate.
The example
[`run_demo.py`](https://github.com/hsaghir/looplet/blob/master/examples/regression_demo/run_demo.py)
shows these steps:

1. Wrap the model backend.
2. Attach the step recorder and `EvalHook`.
3. Save the first run with `save_eval_run()`.
4. Load new tools and collectors.
5. Call `replay_loop()` with the saved responses.
6. Save the new result.
7. Compare the two results.

Compare the agent directories:

```bash
looplet diff ./report-v1.cartridge ./report-v2.cartridge --show
```

Replay is useful here because the change affects tool code.
It keeps model responses fixed while the changed code runs again.
It does not prove that a prompt change produces better decisions.
That needs new model runs.
Tool effects also occur again unless the host isolates or mocks them.

## 9. Put it in CI

Use continuous integration (CI) to run the test above on code changes.
For model behavior, run the command with the model and cases that your release process permits:

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

An agent's own tests are not protected release checks.
Keep protected collectors, graders, and expected data outside the code that a candidate can change.
Do not give them to the candidate through its task or runtime.
Run untrusted code in an isolated process or operating system environment.
A separate directory does not provide that protection.

## What to build next

- Add a [hook](hooks.md) that blocks `done()` before the task is complete.
- [Save a run](provenance.md) and remove sensitive data from its records.
- Add cases from actual failures.
- Keep correctness checks separate from step, token, and time measurements.
- Read the [pitfalls](pitfalls.md) before you add concurrency or permissions.
