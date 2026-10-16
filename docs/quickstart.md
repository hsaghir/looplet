# Quickstart

Run one Python agent, get its answer, test it, and replay its saved model responses.
You need Python 3.11 or newer.
You do not need a model key for these examples.
Start with Python functions.
Cartridges and bundles are optional packaging formats, not prerequisites.

## 1. Install the source version

These examples use the code in the repository.
The published PyPI package is `0.3.0`.
The result helpers below require this source install.

```bash
git clone https://github.com/hsaghir/looplet.git
cd looplet
python -m pip install -e .
```

The core package has no third-party runtime dependencies.
See [Install](install.md) for virtual environments and model providers.

## 2. Run the agent and get its answer

This example uses fixed model responses.
It does not contact a model service.
The tool returns the team that owns a service.

```python title="first_loop.py"
from looplet import MockLLMBackend, minimal_preset, tool


@tool(description="Find the team that owns a service.")
def lookup_owner(service: str) -> dict:
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


def scripted_backend():
    return MockLLMBackend([
        '{"tool":"lookup_owner","args":{"service":"payments"}}',
        '{"tool":"done","args":{"summary":"The payments owner is fintech-platform."}}',
    ])


def run_agent(llm, *, hooks=()):
    with minimal_preset(tools=[lookup_owner], max_steps=2) as run:
        run.config.use_native_tools = False
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
`result.output["summary"]` contains the answer.
The last line prints `True done`.
The example selects the JSON-text protocol for its fixed responses.
A real model uses native tool calls by default.
See [Install](install.md) before changing the backend.

`AgentPreset` holds one run's mutable state and resources.
It is single-use, not a reusable configuration object.
`run_agent()` creates and closes a fresh instance for each call.
Reuse the function for repeated or concurrent tasks, not the instance inside it.

## 3. Check the answer

Completion does not prove correctness.
Append this block to `first_loop.py` to compare the observed answer with a host-owned expectation:

```python
from looplet import EvalContext, eval_assess


def eval_owner(ctx):
    return ctx.completed and ctx.final_output["summary"] == ctx.expected["summary"]


def assess_result(result):
    observed_context = EvalContext(
        steps=result.steps,
        final_output=result.output,
        stop_reason=result.termination_reason,
        task={"expected": {"summary": "The payments owner is fintech-platform."}},
    )
    return eval_assess([eval_owner], observed_context)


if __name__ == "__main__":
    report = assess_result(result)
    report.assert_passed()
    print(report.passed)
```

This prints `True` after checking completion and the answer.
The expected answer belongs to the grader; it is not part of the agent task.
For files or database changes, collect those effects independently before grading.
See [Behavioral evals](evals.md) for collectors and saved cases.

Put these tests beside `first_loop.py`:

```python title="test_first_loop.py"
from first_loop import assess_result, run_agent, scripted_backend


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

## 4. Capture the same agent

Append this block to `first_loop.py`.
Capture records model calls and tool steps; it does not change the agent's tools or task.

```python
from looplet import ProvenanceSink

if __name__ == "__main__":
    sink = ProvenanceSink(dir="traces/owner")
    captured_backend = scripted_backend()
    try:
        captured_result = run_agent(
            sink.wrap_llm(captured_backend),
            hooks=[sink.trajectory_hook()],
        )
    finally:
        sink.flush()

    assess_result(captured_result).assert_passed()
```

Inspect the saved run:

```bash
looplet inspect traces/owner
```

Use a different trace directory for each run you want to keep.
Capture can contain secrets or personal data; review it before sharing or committing it.

## 5. Replay and check the result

Append this block to `first_loop.py`.
It uses the saved responses, the same tools, and fresh run state:

```python
from contextlib import closing

from looplet import DefaultState, RunResult, replay_loop, tools_from

if __name__ == "__main__":
    replay_state = DefaultState(max_steps=2)
    with closing(replay_loop(
        "traces/owner",
        tools=tools_from([lookup_owner], include_done=True),
        state=replay_state,
        task={"goal": "Find the payments owner, then finish."},
    )) as steps:
        for step in steps:
            print(step.tool_call.tool, step.tool_result.data)

    replayed_result = RunResult.from_state(replay_state)
    assess_result(replayed_result).assert_passed()
```

Replay does not call the original model backend.
Tools and their effects run again.
Use isolated workspaces and controlled inputs for tools that write files or contact services.
A changed prompt or model needs a new model run, not replay.

## Inspect steps or stop early

`run_result()` consumes and closes the same iterator that `composable_loop()` returns.
Use the iterator directly when your application must inspect each step or stop between calls.
Append this block to `first_loop.py` to stop after the lookup:

```python
from looplet import LoopConfig, composable_loop

if __name__ == "__main__":
    state = DefaultState(max_steps=2)
    with closing(composable_loop(
        llm=scripted_backend(),
        tools=tools_from([lookup_owner], include_done=True),
        state=state,
        config=LoopConfig(max_steps=2, use_native_tools=False),
        task={"goal": "Find the payments owner, then finish."},
    )) as steps:
        for step in steps:
            print(step.tool_call.tool, step.tool_result.data)
            break

    print(state.termination_reason)
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
