# looplet

[![CI](https://github.com/hsaghir/looplet/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/looplet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hsaghir/looplet/branch/master/graph/badge.svg)](https://codecov.io/gh/hsaghir/looplet)
[![PyPI version](https://img.shields.io/pypi/v/looplet.svg)](https://pypi.org/project/looplet/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hsaghir/looplet/blob/master/LICENSE)

**A Python loop for agents that call tools.**

Looplet runs a tool-calling agent in your Python program.
You provide a model, a task, and Python functions.
It checks each call, runs the function, and yields a `Step` with the result.
Your code controls iteration and can stop before the next call.

Use hooks for policy, saved responses for replay, and independent checks for correctness.
These features use the same loop.
Start with functions; package an agent as files only when you need to share it.

Python 3.11 or newer is required.
The core has no third-party runtime dependencies.

[Documentation](https://hsaghir.com/looplet/) | [Quickstart](https://hsaghir.com/looplet/python/) | [Python API](https://hsaghir.com/looplet/api/)

## Run one loop

The Python-first API is an **unreleased source preview**.
It is not in PyPI `0.4.1`.
Install this fixed revision in a virtual environment:

```bash
python -m pip install "looplet @ https://github.com/hsaghir/looplet/archive/a866b0ad27be5d9b3119e41d2573503687ac794b.zip"
```

This example uses fixed model responses.
It needs no model key or model service:

```python
from looplet import MockLLMBackend, composable_loop


def lookup_owner(service: str) -> str:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return owners.get(service, "unknown")


responses = [
    '{"tool":"lookup_owner","args":{"service":"payments"}}',
    '{"tool":"done","args":{"summary":"fintech-platform"}}',
]
llm = MockLLMBackend(responses, cycle=False)
for step in composable_loop(llm, task="Who owns payments?", tools=[lookup_owner]):
    print(step.pretty())
```

The first tool result is `fintech-platform`.
The second call completes the run.
The function stays callable as ordinary Python.
Replace the scripted backend with your provider after this example works.
The [quickstart](https://hsaghir.com/looplet/python/) covers setup and this change.

## How the loop fits together

Each turn runs from left to right.
Hooks run around model and tool calls.
Updated state feeds the next turn.

[![Looplet's five-stage turn, a cartridge file layout on the right, and dashed output and artifact inputs to a grader below.](https://hsaghir.com/looplet/assets/loop-lifecycle.svg)](https://hsaghir.com/looplet/assets/loop-lifecycle.svg)

A cartridge defines this same loop declaratively as files.
It declares prompts, tools, hooks, and configuration.
Memory and context sources feed prompt construction.
Dashed lines connect output and artifacts to an optional grader after the run.
Its eval result is separate from the loop's completion decision.

Compaction shortens old context before the next prompt when needed.
Built-in compaction keeps original tool results available.
Replay reuses saved model responses and runs tools again.
Completion uses `check_done`, not ordinary tool permission checks.
See [hook timing and runtime controls](https://hsaghir.com/looplet/python/#how-the-loop-fits-together) for the details.

## Add controls to the same loop

Use `run_result()` when you need the final result instead of an iterator.
Keep the function and responses from above:

```python
from looplet import run_result

result = run_result(
    MockLLMBackend(responses, cycle=False),
    task="Who owns payments?",
    tools=[lookup_owner],
    max_steps=4,
)
print(result.output)
print(result.completed, result.termination_reason)
```

The iterator and result helper use the same engine.
`max_steps` limits charged tool steps, not requests or spending.
Use `LoopConfig` for other limits and runtime settings.

Add a [hook](https://hsaghir.com/looplet/hooks/) to check calls before execution or inspect results afterward.
For long runs, Looplet shortens old context and keeps original tool results separately.
Runs save private evidence under `.looplet/runs/<unique-id>` by default.
Compaction can discard details; you control evidence access and retention.

Explicit hooks and services take precedence over matching defaults.
Use `disabled_defaults` for individual opt-outs or `harness_profile="minimal"` to omit automatic additions.
See [runtime controls](https://hsaghir.com/looplet/operations/#standard-agent-harness) for the full policy.

The [quickstart](https://hsaghir.com/looplet/python/) adds a permission hook, captures a run, and replays this same agent.
Use [async APIs](https://hsaghir.com/looplet/operations/#asynchronous-loops) with an asynchronous backend.
Use `closing` or `aclosing` when stopping iteration early.

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

The example captures fixed model responses and an incorrect report.
It changes one tool line and replays those responses with the corrected tool.
A collector reads the new report.
The required grader changes from fail to pass.
See the [walkthrough](https://hsaghir.com/looplet/regression-demo/) for the files and limits.

Replay fixes model responses, not tool effects.
Tools, networks, clocks, and random values can change unless your host controls them.
A prompt or model change needs new model runs.

Keep release checks outside the code that a candidate can change.

## Keep your existing code

Keep your provider and tool functions.
Replace the control loop first.
Check that the new loop preserves the behavior you need.
Then add capture, hooks, or result checks as needed.

See the [Migration guide](https://hsaghir.com/looplet/migrate/) for the full procedure.

Use a [Python factory](https://hsaghir.com/looplet/api/#optional-preset-conveniences) to build fresh dependencies for each task.
Use a [cartridge](https://hsaghir.com/looplet/cartridge/) when you need an agent stored and shared as files.
The optional [`looplet new` generator](https://hsaghir.com/looplet/agent-factory/) drafts cartridges with a model.
It is separate from a Python factory.

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

Looplet does not create an operating system sandbox.
Run untrusted tools in a restricted container or VM.
A Git worktree separates edits, not privileges or credentials.
See [SECURITY.md](https://github.com/hsaghir/looplet/blob/master/SECURITY.md) for deployment controls.

## Published package and stability

The latest PyPI release is `0.4.1`.
Its API differs from the preview above.
Use the [released walkthrough](https://hsaghir.com/looplet/quickstart/) with this install:

```bash
python -m pip install "looplet==0.4.1"
```

See [installation and version guidance](https://hsaghir.com/looplet/install/) before choosing a build.
Read the [released safety fixes](https://hsaghir.com/looplet/install/#unreleased-safety-fixes) before sharing captures from `0.4.0`.

Looplet follows semantic versioning.
Before version `1.0`, a minor release can change an API incompatibly.
Pin your dependency version and artifact hashes for deployment.
See the [changelog](https://hsaghir.com/looplet/changelog/) and [release plan](https://hsaghir.com/looplet/roadmap/#path-to-10).

## Read next

- [Python quickstart](https://hsaghir.com/looplet/python/): run and extend one agent.
- [Install and configure](https://hsaghir.com/looplet/install/): choose a build and connect a provider.
- [Python API](https://hsaghir.com/looplet/api/): find functions, hooks, and configuration fields.
- [Operations](https://hsaghir.com/looplet/operations/): use retries, context limits, cancellation, and checkpoints.
- [Examples](https://github.com/hsaghir/looplet/tree/master/examples): run complete examples.

## Contributing

Report bugs or propose examples and backend adapters.
A core feature must work across task domains.
Include checks for the behavior that your change affects.

See [CONTRIBUTING.md](https://hsaghir.com/looplet/contributing/).
Report security issues as described in [SECURITY.md](https://github.com/hsaghir/looplet/blob/master/SECURITY.md).

## License

Apache License 2.0. See [LICENSE](https://github.com/hsaghir/looplet/blob/master/LICENSE).
