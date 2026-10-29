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
The source build supplies a versioned standard harness; replace or disable its policies as needed.
Start in Python; use a cartridge when you need an agent stored and shared as files.
The [Quickstart](docs/quickstart.md) runs one agent, checks its answer, saves its calls, and replays them.

A short loop is enough if it already meets your needs.
Use Looplet when you need these controls without separate code for each agent.
The loop stays visible, and your code still controls iteration.

Looplet requires Python 3.11 or newer.
The core package has no third-party runtime dependencies.
It does not require a workflow graph, a hosted service, or a cartridge.

[Documentation](https://hsaghir.com/looplet/) | [Quickstart](docs/python.md) | [Python API](docs/api.md)

## Run one loop

Write Python functions, give them to a loop, and inspect each step.
The Python-first API below is **unreleased**. Install this fixed preview:

```bash
python -m pip install "looplet[openai] @ https://github.com/hsaghir/looplet/archive/cf642eba3a0d4a2239db7400611222797e791540.zip"
```

Use a virtual environment. Set `OPENAI_API_KEY` and `OPENAI_MODEL` for your provider.
See the [Python quickstart](docs/python.md) for setup and a network-free test.
No Git installation or local repository checkout is needed.

```python
from looplet import OpenAIBackend, composable_loop


def lookup_owner(service: str) -> str:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return owners.get(service, "unknown")


llm = OpenAIBackend.from_env()
for step in composable_loop(llm, task="Who owns payments?", tools=[lookup_owner]):
    print(step.pretty())
```

The function stays callable as ordinary Python: `lookup_owner("payments")`.
Looplet builds a fresh registry, checks top-level argument types, and supplies a standard `done` tool.
It handles model retries, tool dispatch, context compaction, and private run evidence.
Nested validation, tool effects, security isolation, and evidence retention remain host responsibilities.

To collect the result instead of iterating:

```python
from looplet import run_result

result = run_result(llm, task="Who owns payments?", tools=[lookup_owner])
print(result.output)
print(result.completed, result.termination_reason)
```

These are two ways to consume the same engine.
Use `async_composable_loop()` or `await run_result_async()` with an asynchronous backend.
Use `contextlib.closing` or `aclosing` when stopping iteration early.
Add [configuration and hooks](docs/python.md#change-one-thing-at-a-time) only when needed.

### Published package

The latest PyPI release is `0.4.1`.
The Python-first additions above are not in that release.
For its complete, network-free walkthrough, use the [released guide](docs/quickstart.md):

```bash
python -m pip install "looplet==0.4.1"
```

It includes the [safety fixes](docs/install.md#unreleased-safety-fixes) missing from `0.4.0`.
Pin the version and artifact hashes in your dependency lockfile for deployment.

## Execution defaults (unreleased)

Source builds use `execution-v1` in Python loops and loaded cartridges.
The model chooses its method; the host controls authority and resources.

- Deterministic context compaction, without summary-model calls.
- Typed native-tool history without requiring an assistant completion policy.
- Original tool results retained separately from bounded conversation previews.
- Disk-backed history and provenance, retained results, checkpoints, and branch metadata under a private run directory.

There is no automatic stagnation nudge or tool quota derived from step count.
Select `standard-v1` to retain those policies from the earlier source profile,
or supply individual hooks and reviewed host limits explicitly.

The published `0.4.1` walkthrough above intentionally uses `minimal_preset`.
That helper remains an explicit minimal escape hatch in source builds.
For the ordinary library path, supply your tools directly to the loop:

```python
from looplet import LoopConfig, composable_loop, resolve_harness


config = LoopConfig(max_steps=8, context_window=32_000)
print(resolve_harness(config).describe())
for step in composable_loop(
    llm=llm, tools=[lookup_owner], config=config,
    task="Find the payments owner, then finish.",
):
    print(step.pretty())
```

Use `preset.resolve_harness().describe()` for the equivalent cartridge inspection.
Set `harness_profile="minimal"` to omit all automatic additions, or set
`disabled_defaults` selectively. For example, `["persistence"]` omits automatic
history spooling, evidence files, result-store binding, checkpoints, and branch recording.
Explicit services and hooks remain in place.

Context estimates are approximate and compaction can discard details.
An explicitly selected dispatch guard is not a provider-request or billing ceiling; set
`max_model_calls` explicitly when the host needs request admission.
The base registry records content-addressed originals for `get_result(result_key)`
and checkpoint result-store round trips. Hosts can disable new retention with
`BaseToolRegistry(retain_results=False)` or provide their own registry storage.
The host controls security isolation, permissions, grading, and evidence retention lifetime.
Run files default to `.looplet/runs/<unique-id>`; set `run_dir` to choose their location.
The state metadata exposes the resolved run and trace directories.
See [runtime controls](docs/operations.md#standard-agent-harness) for replacement rules.

The independently graded example exercises both entry paths and compaction:

```bash
python -m looplet.examples.default_harness --scripted --output ./harness-run-001
```

The settlement case retrieves evidence beyond a bounded preview, handles an
oversized page refusal, and independently grades the saved record:

```bash
python -m looplet.examples.default_harness --scripted --cases evidence --output ./evidence-run-001
```

Omit `--scripted` and supply `--base-url` and `--models` for a live endpoint.
See [bounded result views](docs/provenance.md#bounded-model-views) for shared
native/text budgets, retrieval tools, and optional file-backed retention.

Delivery refusals preserve the tool's execution status and provide retry details separately.
Hosts can also install [request admission](docs/operations.md#host-request-admission)
with their own token counter. Result-character limits and request-token limits remain separate controls.

## Customize what you need

The same loop supports these optional features.
You do not need a new execution engine to use them.

| Need                                   | Feature                                              | Guide                                                                    |
| -------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Keep a long run within context limits  | Context budgets, summaries, and truncation           | [Context management](docs/operations.md#context-pressure-and-compaction) |
| Add context or change the prompt       | Hooks before the model call                          | [Hooks](docs/hooks.md)                                                   |
| Check or change calls and results      | Hooks before and after tools run                     | [Hooks](docs/hooks.md)                                                   |
| Limit calls or detect repeated actions | Step limits, tool-call limits, and repetition checks | [Run controls](docs/operations.md#stagnation-and-call-limits)            |
| Save prompts, responses, and steps     | `ProvenanceSink`                                     | [Capture and replay](docs/provenance.md)                                 |
| Check files or other results           | Collectors and graders                               | [Tests and graders](docs/evals.md)                                       |
| Build a fresh agent run for each task  | A Python factory returning `AgentPreset`             | [Python API](docs/api.md#optional-preset-conveniences)                   |
| Store an agent in files                | A cartridge directory                                | [Cartridges](docs/cartridge.md)                                          |
| Use an agent from a shell              | `new`, `run`, `inspect`, and `eval`                  | [Command-line guide](docs/cli.md)                                        |
| Connect tools or other hosts           | Protocol adapters                                    | [Portability](docs/portability.md)                                       |

Looplet `0.4.0` includes the Python helpers and unified CLI.
The guides identify behavior added in `0.4.1`.
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

| Guide                                     | Purpose                                                 |
| ----------------------------------------- | ------------------------------------------------------- |
| [Quickstart](docs/quickstart.md)          | Run, inspect, and stop one loop                         |
| [Install](docs/install.md)                | Choose a version and configure a model                  |
| [Python API](docs/api.md)                 | Find the classes and functions you need                 |
| [Migration](docs/migrate.md)              | Replace a loop without new tool implementations         |
| [Report-agent tutorial](docs/tutorial.md) | Add a case, collector, and grader to a file-based agent |
| [Operations](docs/operations.md)          | Use async code, retries, cancellation, and checkpoints  |
| [Examples](examples/)                     | Run complete examples                                   |

See the [documentation site](https://hsaghir.com/looplet/) for the full guide list.

## Stability

Looplet follows semantic versioning.
Before version `1.0`, a minor release can change an API incompatibly.
Pin the minor version for a published install:

```toml
looplet>=0.4.1,<0.5
```

The latest PyPI release is `0.4.1`.
See the [changelog](CHANGELOG.md) and [release plan](ROADMAP.md#path-to-10).

## Contributing

Report bugs or propose examples and backend adapters.
A core feature must work across task domains.
Include checks for the behavior that your change affects.

See [CONTRIBUTING.md](CONTRIBUTING.md).
Report security issues as described in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
