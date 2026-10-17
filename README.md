# looplet

[![CI](https://github.com/hsaghir/looplet/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/looplet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hsaghir/looplet/branch/master/graph/badge.svg)](https://codecov.io/gh/hsaghir/looplet)
[![PyPI version](https://img.shields.io/pypi/v/looplet.svg)](https://pypi.org/project/looplet/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hsaghir/looplet/blob/master/LICENSE)

**A Python loop for agents that call tools.**

You provide the model, tools, task, and limits.
Looplet checks each tool call, runs the tool, and records the result.
It yields a `Step` after each call.
Your code can inspect each step and stop iteration.

Looplet also provides context management for longer runs and hooks to control each stage of the loop.
Add only the features that your agent needs.
Start in Python; use a cartridge when you need an agent stored and shared as files.
The [Quickstart](docs/quickstart.md) runs one agent, checks its answer, saves its calls, and replays them.

A short loop is enough if it already meets your needs.
Use Looplet when you need these controls without separate code for each agent.
The loop stays visible, and your code still controls iteration.

Looplet requires Python 3.11 or newer.
The core package has no third-party runtime dependencies.
It does not require a workflow graph, a hosted service, or a cartridge.

[Documentation](https://hsaghir.com/looplet/) | [Quickstart](docs/quickstart.md) | [Python API](docs/api.md)

## Run one loop

Install the current source build for the examples and fixes in these guides.
No Git installation or local repository checkout is needed.

```bash
python -m pip install "looplet @ https://github.com/hsaghir/looplet/archive/refs/heads/master.zip"
```

Run this inside a virtual environment; see [Install](docs/install.md).
This installs a non-editable package and needs network access.

Source builds identify as `0.4.1.dev0`; this is not a published patch release.
PyPI `0.4.0` lacks the [later safety fixes](docs/install.md#unreleased-safety-fixes).
The archive follows `master`; pin an approved full commit hash for deployment.

The example uses fixed model responses.
It runs without a model service or an API key.

```python
from contextlib import closing

from looplet import DefaultState, LoopConfig, MockLLMBackend, RunResult, composable_loop, tools_from


def lookup_owner(service: str) -> dict:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


llm = MockLLMBackend([
    '{"tool":"lookup_owner","args":{"service":"payments"}}',
    '{"tool":"done","args":{"summary":"The payments owner is fintech-platform."}}',
], cycle=False)
tools = tools_from([lookup_owner], include_done=True, validate=True)
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

print(RunResult.from_state(state).termination_reason)
```

The loop calls `lookup_owner`, then `done`.
The lookup returns `fintech-platform` as the payments owner.
The last line prints `done`, the stop reason.
`cycle=False` rejects unexpected extra model calls instead of repeating the script.

The tool remains a normal Python function.
You can also call `lookup_owner("payments")` directly.

Each yielded step follows a tool call.
The tool has already run when your code receives that step.
Use `break` to stop iteration before requesting another step.
For native batches, preventing later calls requires the [unreleased serial-dispatch fix](docs/install.md#unreleased-safety-fixes).
Published `0.4.0` may execute the batch before yielding its first step.
With `concurrent_dispatch=True`, the batch runs before its results are yielded.
`closing` closes the loop and runs its end hooks.
Your code must close resources that it creates.

`validate=True` checks top-level argument types, nullability, and enum choices before tools run.
Nested data and other constraints remain the tool's responsibility.
Without this option, the base registry checks argument names and required fields, not full types.

This example selects the JSON-text protocol for its fixed responses.
Real models use native tool calls by default.
See [Install](docs/install.md) for provider adapters and configuration.

The [Quickstart](docs/quickstart.md) adds a result helper, pytest checks, capture, and replay to this same agent.
For async code, use `async_composable_loop()` with `contextlib.aclosing`.

## Add only what you need

The same loop supports these optional features.
You do not need a new execution engine to use them.

| Need | Feature | Guide |
| --- | --- | --- |
| Keep a long run within context limits | Context budgets, summaries, and truncation | [Context management](docs/operations.md#context-pressure-and-compaction) |
| Add context or change the prompt | Hooks before the model call | [Hooks](docs/hooks.md) |
| Check or change calls and results | Hooks before and after tools run | [Hooks](docs/hooks.md) |
| Limit calls or detect repeated actions | Step limits, tool-call limits, and repetition checks | [Run controls](docs/operations.md#stagnation-and-call-limits) |
| Save prompts, responses, and steps | `ProvenanceSink` | [Capture and replay](docs/provenance.md) |
| Check files or other results | Collectors and graders | [Tests and graders](docs/evals.md) |
| Build a fresh agent run for each task | A Python factory returning `AgentPreset` | [Python API](docs/api.md#optional-preset-conveniences) |
| Store an agent in files | A cartridge directory | [Cartridges](docs/cartridge.md) |
| Use an agent from a shell | `new`, `run`, `inspect`, and `eval` | [Command-line guide](docs/cli.md) |
| Connect tools or other hosts | Protocol adapters | [Portability](docs/portability.md) |

Looplet `0.4.0` includes the Python helpers and unified CLI.
The guides mark later behavior as unreleased.
Older commands, such as `run-cartridge`, `describe`, and `show`, remain supported.

## Check results

A `done` step means that the loop accepted completion.
It does not prove that the answer is correct.

Use a collector to read the files or records that the agent produced.
Use a grader to compare that evidence with the expected result.
Required graders can fail a test or a continuous integration check.

Run the regression example after installation:

```bash
looplet-proof
```

The example records fixed model responses and an incorrect report.
It changes one tool line and replays the responses with the corrected tool.
A collector reads the new report.
The required grader changes from fail to pass.
See the [walkthrough](docs/regression-demo.md) for the files and limits.

Replay fixes model responses, not tool effects.
Tools, networks, clocks, and random values can change unless your host controls them.
A prompt or model change needs new model runs.

An agent's own tests are not protected acceptance checks.
Keep release checks outside the code that a candidate can change.
Run untrusted code in an isolated process or operating system environment.

## Use it in an agent that you have

Keep your provider and tool functions.
Replace the control loop first.
Check that the new loop preserves the behavior you need.
Then add capture, hooks, or result checks as needed.

See the [Migration guide](docs/migrate.md) for the full procedure.

## When Looplet fits

Use Looplet when:

- One model calls tools until the task ends.
- Your team uses Python, Git, and pytest.
- Your code must control when tools run.
- You need saved runs and tests for agent changes.

Use something else when:

- Your main task needs a durable workflow graph.
- You need a complete assistant, not a Python library.
- You need a managed dashboard as your main interface.
- A small loop that you wrote already meets your needs.

Looplet can run inside a workflow engine.
It can export evidence to other systems.
It does not replace those systems.

## Documentation

| Guide | Purpose |
| --- | --- |
| [Quickstart](docs/quickstart.md) | Run, inspect, and stop one loop |
| [Install](docs/install.md) | Choose a version and configure a model |
| [Python API](docs/api.md) | Find the classes and functions you need |
| [Migration](docs/migrate.md) | Replace a loop without new tool implementations |
| [Report-agent tutorial](docs/tutorial.md) | Add a case, collector, and grader to a file-based agent |
| [Operations](docs/operations.md) | Use async code, retries, cancellation, and checkpoints |
| [Examples](examples/) | Run complete examples |

See the [documentation site](https://hsaghir.com/looplet/) for the full guide list.

## Stability

Looplet follows semantic versioning.
Before version `1.0`, a minor release can change an API incompatibly.
Pin the minor version for a published install:

```toml
looplet>=0.4,<0.5
```

The latest PyPI release is `0.4.0`.
The recommended source build includes later fixes that are not yet on PyPI.
See the [changelog](CHANGELOG.md) and [release plan](ROADMAP.md#path-to-10).

## Contributing

Report bugs or propose examples and backend adapters.
A core feature must work across task domains.
Include checks for the behavior that your change affects.

See [CONTRIBUTING.md](CONTRIBUTING.md).
Report security issues as described in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
