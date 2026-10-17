# Quickstart

Run one Python agent, get its answer, test it, and replay its saved model responses.
You need Python 3.11 or newer.
You do not need a model key until the optional real-model section.
Start with Python functions.
Cartridges and bundles are optional packaging formats, not prerequisites.

## 1. Install Looplet

Install the published package in your own project.
It contains the APIs used below and does not require Git or a repository checkout.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "looplet==0.4.0"
```

The core package has no third-party runtime dependencies.
The install needs network access; the fixed-response runs do not.
On Windows, activate `.venv\Scripts\Activate.ps1` in PowerShell.
See [Install](install.md) for virtual environments and model providers.

## 2. Run the agent and get its answer

This example uses fixed model responses.
It does not contact a model service.
The tool returns the team that owns a service.

```python title="first_loop.py"
from looplet import MockLLMBackend, minimal_preset


def lookup_owner(service: str) -> dict:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


def done(service: str, owner: str, summary: str = "") -> dict:
    """Finish with the service, its owner, and an optional explanation."""
    return {"service": service, "owner": owner, "summary": summary}


def scripted_backend():
    return MockLLMBackend([
        '{"tool":"lookup_owner","args":{"service":"payments"}}',
        '{"tool":"done","args":{"service":"payments","owner":"fintech-platform","summary":"The payments owner is fintech-platform."}}',
    ])


def run_agent(llm, *, hooks=(), use_native_tools=False, max_steps=2):
    with minimal_preset(tools=[lookup_owner, done], max_steps=max_steps, validate=True) as run:
        run.config.use_native_tools = use_native_tools
        run.hooks.extend(hooks)
        return run.run_result(llm, task={"goal": "Find the payments owner, then finish."})


if __name__ == "__main__":
    result = run_agent(scripted_backend())
    print(result.output)
    print(result.completed, result.termination_reason)
```

Run the example file:

```bash
python first_loop.py
```

The loop calls `lookup_owner`, then `done`.
`result.output["owner"]` contains the owner; `summary` is an optional explanation.
The last line prints `True done`.
This helper defaults to JSON-text for its fixed responses.
Pass `use_native_tools=True` when changing to a native-tool backend.
The [real-model section](#use-the-same-agent-with-a-real-model) shows the complete change.
The scripted run needs two calls; `max_steps` lets other runs use a different limit.

These tools remain ordinary Python functions.
Call them directly in your application or unit tests.
`minimal_preset` infers tool schemas from their signatures and descriptions from their docstrings.
Published `0.4.0` may show a false `ToolSpec` type error for these functions; the corrected annotation is unreleased.
The optional `@tool` decorator returns a `ToolSpec`, not a callable function.
Call a decorated tool's `.execute(...)` method to test its underlying function.
That direct call bypasses registry validation and hooks.

`AgentPreset` holds one run's mutable state and resources.
It is single-use, not a reusable configuration object.
`run_agent()` creates and closes a fresh instance for each call.
Reuse the function for repeated or concurrent tasks, not the instance inside it.

`validate=True` checks top-level tool argument types and enum choices before effects.
Nested schemas and other constraints remain tool-owned.
The base registry retains its lighter argument-name checks when this option is omitted.

## 3. Check the answer

Completion does not prove correctness.
Add these definitions above the main block in `first_loop.py`.
Replace the existing main block with the one at the end of this example.
The grader compares the observed answer with a host-owned expectation:

```python
from looplet import EvalContext, eval_assess


EXPECTED_OWNER = {"service": "payments", "owner": "fintech-platform"}


def eval_owner(ctx):
    return (
        ctx.completed
        and isinstance(ctx.final_output, dict)
        and ctx.final_output.get("service") == ctx.expected["service"]
        and ctx.final_output.get("owner") == ctx.expected["owner"]
    )


def assess_result(result):
    observed_context = EvalContext(
        steps=result.steps,
        final_output=result.output,
        stop_reason=result.termination_reason,
        task={"expected": dict(EXPECTED_OWNER)},
    )
    return eval_assess([eval_owner], observed_context)


if __name__ == "__main__":
    result = run_agent(scripted_backend())
    print(result.output)
    print(result.completed, result.termination_reason)
    report = assess_result(result)
    report.assert_passed()
    print(report.passed)
```

This prints `True` after checking completion and the answer.
The grader checks service and owner fields, not the explanation's exact wording.
The expected answer belongs to the grader; it is not part of the agent task.
For files or database changes, collect those effects independently before grading.
See [Behavioral evals](evals.md) for collectors and saved cases.

Put these tests beside `first_loop.py`:

```python title="test_first_loop.py"
from first_loop import assess_result, done, lookup_owner, run_agent, scripted_backend


def test_tools_are_regular_functions():
    assert lookup_owner("payments") == {"service": "payments", "owner": "fintech-platform"}
    assert done("payments", "fintech-platform") == {
        "service": "payments", "owner": "fintech-platform", "summary": ""
    }


def test_owner():
    result = run_agent(scripted_backend())
    assess_result(result).assert_passed()


def test_repeated_runs_are_independent():
    first = run_agent(scripted_backend())
    second = run_agent(scripted_backend())
    assert first.completed and second.completed
    assert first.steps[0] is not second.steps[0]
    assert first.output == second.output
```

Run the tests:

```bash
python -m pip install pytest
python -m pytest test_first_loop.py
```

Validate the same grader definitions without running the agent:

```bash
looplet eval check first_loop.py --json
```

Run Python agents with `python first_loop.py`; run their tests with pytest.
`looplet run` loads cartridges or bundles; `looplet eval run` executes cartridge eval cases.
`eval init --python evals/` creates definitions and a runnable pytest example.
Run it with `python -m pytest evals/`.
Omit `--python` to create definitions only.

## 4. Capture the same agent

Put this block in a separate `first_loop_capture.py` file.
Importing `first_loop` loads its functions without running its main block.
Capture records model calls and tool steps; it does not change the agent's tools or task.

```python title="first_loop_capture.py"
from looplet import EvalHook, ProvenanceSink, save_eval_run

from first_loop import EXPECTED_OWNER, eval_owner, run_agent, scripted_backend


def capture_agent(llm):
    sink = ProvenanceSink()
    grading = EvalHook(evaluators=[eval_owner], expected=EXPECTED_OWNER)
    try:
        captured_result = run_agent(
            sink.wrap_llm(llm),
            hooks=[sink.trajectory_hook(), grading],
        )
    finally:
        trace_dir = sink.flush()

    save_eval_run(trace_dir, recorder=sink.trajectory_hook(), eval_hook=grading)
    grading.assessment().assert_passed()
    return captured_result, trace_dir


if __name__ == "__main__":
    captured_backend = scripted_backend()
    captured_result, trace_dir = capture_agent(captured_backend)
    print(captured_result.output)
    print(f"Inspect with: looplet inspect {trace_dir}")
    print(f"Grade with: looplet eval {trace_dir} --evals first_loop.py --json")
    print(f"Rescore with: python first_loop_rescore.py {trace_dir}")
    print(f"Replay with: python first_loop_replay.py {trace_dir}")
```

Run one captured agent:

```bash
python first_loop_capture.py
```

Run the printed `inspect` command to read the saved run.
This command executes the agent once and saves one trajectory.
Each execution gets a unique directory under `.looplet/traces/`.
An explicit `ProvenanceSink(dir=...)` refuses existing evidence unless you pass `overwrite=True`.
Capture can contain secrets or personal data; review it before sharing or committing it.
New record directories own a wildcard `.gitignore`; it does not protect already tracked files or replace redaction.
See [reviewed summaries](provenance.md#share-a-reviewed-summary) before exporting evidence.

The grading hook owns the live assessment once.
`save_eval_run()` saves its expectations separately from the agent-visible task.
A raw trace alone does not preserve grader expectations or collected artifacts.
Reload the saved evidence when you want to rerun a grader without running the agent:

```python title="first_loop_rescore.py"
import sys

from looplet import eval_assess, load_eval_run

from first_loop import eval_owner


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python first_loop_rescore.py TRACE_DIR")
    saved_run = load_eval_run(sys.argv[1])
    saved_report = eval_assess([eval_owner], saved_run.context)
    saved_report.assert_passed()
    print(saved_report.passed)
```

Run the printed `Rescore` command after creating this file.
The printed `eval` command uses the same saved evidence and grader.
It accepts this single run directory or its parent directory.
Offline rescoring reruns graders, not collectors, tools, or agent model calls.

## 5. Replay and check the result

Put this block in a separate `first_loop_replay.py` file.
It uses the saved responses, the same tools, and fresh run state:

```python title="first_loop_replay.py"
import sys
from contextlib import closing

from looplet import DefaultState, RunResult, replay_loop, tools_from

from first_loop import assess_result, done, lookup_owner


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python first_loop_replay.py TRACE_DIR")
    replay_state = DefaultState(max_steps=2)
    with closing(replay_loop(
        sys.argv[1],
        tools=tools_from([lookup_owner, done], validate=True),
        state=replay_state,
        task={"goal": "Find the payments owner, then finish."},
    )) as steps:
        for step in steps:
            print(step.tool_call.tool, step.tool_result.data)

    replayed_result = RunResult.from_state(replay_state)
    assess_result(replayed_result).assert_passed()
```

Run the printed `Replay` command after creating this file.
Replay does not call the original model backend.
Tools and their effects run again.
Use isolated workspaces and controlled inputs for tools that write files or contact services.
A changed prompt or model needs a new model run, not replay.
For packaged agents, the [shell replay recipe](cli.md#replay-captured-responses) provides the same handoff through `looplet replay`.

## Inspect steps or stop early

`run_result()` consumes and closes the same iterator that `composable_loop()` returns.
Use the iterator directly when your application must inspect each step or stop between calls.
Put this block in `first_loop_stop.py` to stop after the lookup:

```python title="first_loop_stop.py"
from contextlib import closing

from looplet import DefaultState, LoopConfig, RunResult, composable_loop, tools_from

from first_loop import done, lookup_owner, scripted_backend


if __name__ == "__main__":
    state = DefaultState(max_steps=2)
    backend = scripted_backend()
    with closing(composable_loop(
        llm=backend,
        tools=tools_from([lookup_owner, done], validate=True),
        state=state,
        config=LoopConfig(max_steps=2, use_native_tools=False),
        task={"goal": "Find the payments owner, then finish."},
    )) as steps:
        for step in steps:
            print(step.tool_call.tool, step.tool_result.data)
            break

    print(RunResult.from_state(state).termination_reason)
```

```bash
python first_loop_stop.py
```

Only the payments lookup occurs.
The `break` statement stops your iteration.
`closing` closes the loop and runs its end hooks.
The last line prints `stream_closed`.
The `done` call does not occur.

Your loop receives a `Step` after each tool call.
The tool has already run when you receive this object.

| Field | Meaning |
| --- | --- |
| `step.tool_call.tool` | The tool name |
| `step.tool_call.args` | The arguments that the model supplied |
| `step.tool_result.data` | The data that the tool returned |
| `step.tool_result.error` | The error, if the tool call failed |

Use a [hook](hooks.md) to approve or reject a call before the tool runs.

Close resources that your code creates, such as database clients.
For async code, use `async_composable_loop()` with `contextlib.aclosing`.
See the [Python API](api.md#execution-core) for that form.

## Use the same agent with a real model

Keep `first_loop.py` unchanged and install the adapter for the same release:

```bash
python -m pip install "looplet[openai]==0.4.0"
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o
```

Put this in a separate `first_loop_live.py` file:

```python title="first_loop_live.py"
from looplet import OpenAIBackend

from first_loop import run_agent


if __name__ == "__main__":
    result = run_agent(OpenAIBackend.from_env(), use_native_tools=True, max_steps=8)
    print(result.output)
    print(result.completed, result.termination_reason)
    if not result.completed:
        raise SystemExit(f"Agent stopped before completion: {result.termination_reason}")
```

```bash
python first_loop_live.py
```

This makes real provider calls and may incur charges.
It uses the same tools and fresh-run factory, with native tool calls enabled explicitly.
The live example allows up to eight tool calls, including completion.
Adjust `max_steps` for your task; budget exhaustion means the run did not finish.
For Anthropic, install the `anthropic` extra and replace `OpenAIBackend` with `AnthropicBackend`.
Configure its key as described in [Install](install.md#configure-anthropic).
Use `use_native_tools=False` only when you intentionally select JSON-text responses.

## Choose the next guide

You do not need all of these features to use the loop.

| Goal | Guide |
| --- | --- |
| Use a real model | [Install and configure](install.md#configure-openai) |
| Change the loop's decisions | [Hooks](hooks.md) |
| Save and replay a run | [Capture and replay](provenance.md) |
| Check the result | [Tests and graders](evals.md) |
| Run the full regression example | [Regression example](regression-demo.md) |
| Put an agent in a directory | [Cartridges](cartridge.md) |
| Add a file-based test case | [Report-agent tutorial](tutorial.md) |
| Replace a loop that you have | [Migration guide](migrate.md) |
