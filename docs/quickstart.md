# Quickstart

Run one loop, inspect its steps, and stop it from your code.
You need Python 3.11 or newer.
You do not need a model key for these examples.

## 1. Install the source version

These examples use the code in the repository.
Some features are not in the published package yet.

```bash
git clone https://github.com/hsaghir/looplet.git
cd looplet
python -m pip install -e .
```

The core package has no third-party runtime dependencies.
See [Install](install.md) for virtual environments and model providers.

## 2. Run one loop

This example uses fixed model responses.
It does not contact a model service.
The tool returns the team that owns a service.

```python title="first_loop.py"
from contextlib import closing

from looplet import DefaultState, LoopConfig, MockLLMBackend, composable_loop, tool, tools_from


@tool(description="Find the team that owns a service.")
def lookup_owner(service: str) -> dict:
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


llm = MockLLMBackend([
    '{"tool":"lookup_owner","args":{"service":"payments"}}',
    '{"tool":"done","args":{"summary":"The payments owner is fintech-platform."}}',
])
tools = tools_from([lookup_owner], include_done=True)
state = DefaultState(max_steps=2)

with closing(composable_loop(
    llm=llm,
    tools=tools,
    state=state,
    config=LoopConfig(max_steps=2, use_native_tools=False),
    task={"goal": "Find the payments owner, then finish."},
)) as steps:
    for step in steps:
        print(step.tool_call.tool, step.tool_result.data)

print(state.termination_reason)
```

Run the example file:

```bash
python first_loop.py
```

The loop calls `lookup_owner`, then `done`.
The last line prints `done`, the stop reason.
The example selects the JSON-text protocol for its fixed responses.
A real model uses native tool calls by default.

## 3. Inspect each step

Your `for` loop receives a `Step` after each tool call.
The tool has already run when you receive this object.

| Field | Meaning |
| --- | --- |
| `step.tool_call.tool` | The tool name |
| `step.tool_call.args` | The arguments that the model supplied |
| `step.tool_result.data` | The data that the tool returned |
| `step.tool_result.error` | The error, if the tool call failed |

`state.steps` contains the steps that have occurred.
Use these fields to display results or choose what your code does next.
Use a [hook](hooks.md) if you need to approve a call before the tool runs.

## 4. Stop the loop

Keep the imports and `lookup_owner` function from the first example.
Replace the rest of that example with this block.

```python
llm = MockLLMBackend([
    '{"tool":"lookup_owner","args":{"service":"payments"}}',
    '{"tool":"lookup_owner","args":{"service":"search"}}',
])
tools = tools_from([lookup_owner], include_done=True)
state = DefaultState(max_steps=2)

with closing(composable_loop(
    llm=llm,
    tools=tools,
    state=state,
    config=LoopConfig(max_steps=2, use_native_tools=False),
    task={"goal": "Find the payments and search owners."},
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
The search lookup does not occur.

Close resources that your code creates, such as database clients.
For async code, use `async_composable_loop()` with `contextlib.aclosing`.
See the [Python API](api.md#execution-core) for that form.

## What you now control

The model selects a tool.
Looplet checks the call, runs the tool, and records the result.
The loop then yields a step to your code.
Your code decides whether to request the next step.

`done` means that the loop accepted completion.
It does not prove that the answer is correct.

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
