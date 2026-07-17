# Troubleshooting

Start with the smallest failing layer. A provider probe does not test your
tools, a scripted run does not test a provider, and captured-response replay
does not test how a model will respond to a changed prompt.

## First checks

```bash
python --version
python -m pip show looplet
looplet doctor --no-backend
python -m looplet.examples.hello_world --scripted
```

Looplet requires Python 3.11 or newer. The scripted example verifies the
installed package, loop, tool dispatch, and eval path without a provider or
network.

For a machine-readable environment report:

```bash
looplet doctor --json --no-backend
```

Do not attach API keys, prompts, raw traces, or unredacted tool results to a
public issue.

## Installation and command failures

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError: looplet` | Confirm `python` and `pip` point at the same virtual environment. |
| `looplet: command not found` | Run `python -m looplet --help`; then inspect the environment's scripts directory. |
| Provider import fails | Install `looplet[openai]` or `looplet[anthropic]` in the active environment. |
| A new API is missing | Print `looplet.__version__` and compare it with the version you installed. |

The documentation may describe the next release before that release reaches
PyPI. Pin and inspect the package version rather than inferring it from the
website.

## OpenAI-compatible backend configuration

`OpenAIBackend.from_env()` needs either `OPENAI_API_KEY` or
`OPENAI_BASE_URL`. It reads `OPENAI_MODEL` when provided.

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.1:8b
export OPENAI_API_KEY=x
looplet doctor
```

If the endpoint accepts chat completions but not native tool schemas, the
probe reports that distinction. Select text-mode tools explicitly:

```python
from looplet import LoopConfig

config = LoopConfig(use_native_tools=False)
```

Do not silently fall back in a release harness. Record the selected protocol
in configuration and test it.

## Anthropic backend configuration

`AnthropicBackend.from_env()` requires `ANTHROPIC_API_KEY` and optionally reads
`ANTHROPIC_MODEL`:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

`looplet doctor` currently probes the OpenAI-compatible environment. For an
Anthropic-only deployment, construct `AnthropicBackend.from_env()` in a smoke
test and make one explicitly budgeted provider call.

## The loop produces no steps

`composable_loop()` is a generator. Calling it does not execute the loop until
you iterate:

```python
# This creates a generator but does not run it.
run = composable_loop(...)

# This executes the loop.
steps = list(run)
```

For live handling, prefer `for step in composable_loop(...):` so each dispatch
is visible as it happens.

## Tools are not called

Check these boundaries in order:

1. Confirm the registry contains the expected tool names.
2. Print or save the prompt and verify the tool schema is present.
3. Confirm `use_native_tools` matches the endpoint's actual capabilities.
4. Inspect the yielded error `Step` instead of relying on the final response.
5. Run the same tool directly with representative arguments.

Use `looplet doctor` for an OpenAI-compatible native-tool probe. Use
`looplet describe` for a cartridge's loaded structure.

## A hook does not fire

Hooks are protocol-based. The method name and signature must match the
lifecycle slot. Start with one hook and one test that exercises the exact
boundary. The [hook guide](hooks.md) lists every supported method and return
type.

Common mistakes include:

- returning a briefing string from a method that expects `HookDecision`;
- treating `should_stop` as if it ran before the current dispatch;
- registering a hook on a preset but running a different hooks list;
- swallowing exceptions and turning a failed policy check into a no-op.

## An eval passes when the product is wrong

Inspect what the grader reads. The final response and `done()` arguments are
agent claims, not independent evidence. A collector should inspect the file,
database record, command result, API response, or other world state that users
actually depend on.

Required graders, collector errors, malformed records, explicit failure
labels, and empty required suites fail closed in the CLI. If a grader was
skipped unexpectedly, verify its marks and the `--include` or `--exclude`
filters.

See [behavioral evals](evals.md) and [experiment design](experiments.md).

## Replay changed the result

Captured-response replay holds model responses constant. It deliberately runs
fresh tool code, hooks, permissions, state, clocks, networks, and side effects.
A changed result is expected when one of those surfaces changed.

Use a mock or sandbox when the external world must also remain fixed. Use new
sampled model calls when the prompt, model, sampling policy, or tool
description is the variable.

## Context grows until the provider rejects it

Configure both a compaction service and a threshold hook. The hook requests
compaction; the service performs it:

```python
from looplet import (
    ContextBudget,
    DefaultCompactService,
    LoopConfig,
    ThresholdCompactHook,
)

config = LoopConfig(compact_service=DefaultCompactService())
hooks = [ThresholdCompactHook(ContextBudget(context_window=128_000))]
```

The token estimate is approximate. Set thresholds below the provider's hard
limit and preserve output headroom. Read [runtime operations](operations.md)
before enabling summarization in a release harness.

## A run resumes old state

When `LoopConfig(checkpoint_dir=...)` points at a directory containing
checkpoints and no explicit `initial_checkpoint` is supplied, Looplet loads the
latest checkpoint automatically. Use a fresh directory for a new run, or
remove stale checkpoint files intentionally.

Checkpoint JSON may contain task and conversation data. Apply the same access,
retention, and redaction policy used for traces.

## MCP tools hang or fail at startup

Verify the server command independently before loading the cartridge. The MCP
adapter expects newline-delimited JSON over stdio, a bounded response timeout,
and a process that keeps stdout reserved for protocol messages. Send human
diagnostics to stderr.

Reduce the case to one server and one tool. Include the server command, Python
version, timeout, and redacted stderr in a bug report. Never include secrets or
unredacted tool payloads.

## Prepare a useful bug report

Include:

```bash
python --version
python -c "import looplet; print(looplet.__version__)"
looplet doctor --json --no-backend
```

Also include the smallest scripted reproduction, expected behavior, observed
behavior, and the relevant yielded `Step` or redacted trace fragment. State
whether the failure occurs in a live provider call, a scripted backend, replay,
or offline grading. Those are different execution paths.
