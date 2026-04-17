# openharness

[![CI](https://github.com/hsaghir/openharness/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/openharness/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/openharness.svg)](https://pypi.org/project/openharness/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#stability--versioning)

**The tool-calling loop you can actually step through.**

Every agent framework gives you `agent.run()`. `openharness` gives you
`for step in loop(...):` — and that's the whole product.

```python
from openharness import composable_loop

for step in composable_loop(llm=llm, tools=tools, task=task, ...):
    print(step.pretty())          # → "#1 ✓ search(query='…') → 12 items [182ms]"
    if step.tool_result.error:
        break                     # your loop, your control flow
```

> Not to be confused with [`pydantic-ai-harness`](https://github.com/pydantic/pydantic-ai-harness),
> which is a *capability* library for pydantic-ai. `openharness` is a *loop*
> library — it works with any LLM backend and has no framework dependency.

## Who this is for

- You're building an agent for a non-trivial domain (security, research,
  ops, robotics) and you've hit a wall where framework magic gets in your
  way.
- You want to iterate on *behavior at a single step* — add a hook, filter
  a result, veto a tool call — without learning a graph DSL.
- You need **vendor independence**: no Claude-only, no AWS-only, no
  "everything through Pydantic."
- You're a library author who wants to embed a reactive loop in your own
  package without pulling in dozens of transitive deps.

## Who this is **not** for

- You want Claude Code in a Python import → use
  [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/).
- You need a visual graph with branches and joins → use
  [`langgraph`](https://pypi.org/project/langgraph/).
- You have a multi-agent handoff system →
  [`openai-agents`](https://pypi.org/project/openai-agents/).
- You want Pydantic validation baked into every call →
  [`pydantic-ai`](https://pypi.org/project/pydantic-ai/).

## How it compares

| | openharness | claude-agent-sdk | strands-agents | pydantic-ai | langgraph |
|---|---|---|---|---|---|
| **You own the loop (iterator)** | ✅ `for step in loop(...)` | ❌ async message stream | ❌ closed `agent("...")` | ❌ closed `agent.run_sync()` | ❌ graph execution |
| **Provider-agnostic** | ✅ `LLMBackend` protocol | ❌ Claude only (bundled CLI) | ✅ | ✅ | ✅ |
| **No subprocess / bundled binary** | ✅ | ❌ CLI bundled in wheel | ✅ | ✅ | ✅ |
| **Hooks as `Protocol` objects** | ✅ `@runtime_checkable` | ⚠️ dict of callbacks | ⚠️ config + inheritance | ⚠️ `Capability` classes | ⚠️ nodes |
| **Sync ↔ async parity** | ✅ guaranteed | async only | mixed | mixed | mixed |
| **Fail-closed permissions engine** | ✅ built in | ⚠️ hooks only | ❌ | ⚠️ deferred tools | ❌ |
| **Crash-resume checkpoints + conversation** | ✅ | ❌ | ❌ | ⚠️ durable-execution add-on | ✅ |
| **OSI license** | Apache-2.0 | Anthropic commercial terms | Apache-2.0 | MIT | MIT |
| **Runtime deps (core)** | 1 (`pyyaml`) | CLI binary | several | many | many |

## Features

- **Composable loop** — `composable_loop` / `async_composable_loop` yield
  `Step`s you can observe or interrupt. Hooks (`pre_prompt`, `pre_dispatch`,
  `post_dispatch`, `check_done`, `should_stop`, `on_loop_end`) let you
  layer behavior without forking the loop.
- **Tool registry** — `BaseToolRegistry` + `ToolSpec` with JSON-schema
  catalog rendering, concurrent-safe batching, auto-`ctx` threading, and
  structured `ToolError` classification (`TIMEOUT`, `VALIDATION`,
  `PERMISSION_DENIED`, `RATE_LIMIT`, `CONTEXT_OVERFLOW`, `CANCELLED`, …).
- **Permissions** — declarative `PermissionEngine` with `ALLOW` / `DENY` /
  `ASK` / `DEFAULT` rules, fail-closed argument matchers, plug-in
  `ask_handler` for human-in-the-loop, and an append-only denial audit log.
- **Reactive recovery** — automatic re-prompting on JSON parse failures,
  prompt-too-long pre-flight detection with chained compaction strategies.
- **Streaming** — `StreamingHook` emits `LoopStart` / `StepStart` /
  `LLMCallStart` / `ToolDispatch` / `LoopEnd` events over an
  `EventEmitter`.
- **Checkpoints** — `FileCheckpointStore` + `resume_loop_state()` preserve
  session log, conversation, step offset, and budget counters across
  crash-resume.
- **Cooperative cancellation** — `CancelToken` is threaded through
  `LoopConfig` → `llm_call_with_retry` → `ToolContext`, so long-running
  tools stop on the next yield point.
- **Multi-block messages** — `Message.content` supports rich
  `ContentBlock`s (text, image, tool-use, …) with automatic
  `HEAVY_BLOCK_KINDS` stripping before summarization.
- **Backends** — sync + async + streaming adapters for Anthropic and
  OpenAI. Bring your own by implementing the `LLMBackend` /
  `AsyncLLMBackend` `Protocol`.
- **Sub-agents** — `run_sub_loop` spawns isolated child loops with their
  own tools / config while sharing the parent's tracer and telemetry.
- **Telemetry** — pluggable `Tracer` + `MetricsCollector` for OpenTelemetry
  or any other backend.

## Install

```bash
uv add openharness
# or
pip install openharness
```

Optional extras:

```bash
pip install "openharness[anthropic]"   # adds anthropic SDK
pip install "openharness[openai]"      # adds openai SDK
pip install "openharness[all]"         # both
```

## Quick start

```python
from anthropic import Anthropic
from openharness import (
    composable_loop, LoopConfig,
    BaseToolRegistry, ToolSpec,
    DefaultState, SessionLog,
    AnthropicBackend,
)

# 1. Define tools
tools = BaseToolRegistry()
tools.register(ToolSpec(
    name="add",
    description="Add two integers.",
    parameters={"a": "int", "b": "int"},
    execute=lambda a, b: {"sum": a + b},
))

# 2. Pick a backend (or bring your own implementing the LLMBackend protocol)
llm = AnthropicBackend(Anthropic(), model="claude-opus-4-1")

# 3. Configure the loop
config = LoopConfig(
    max_steps=10,
    max_tokens=1024,
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

# 4. Run
state = DefaultState(max_steps=10)
session = SessionLog()
for step in composable_loop(
    llm=llm,
    task={"question": "What is 2 + 3?"},
    tools=tools,
    config=config,
    state=state,
    session_log=session,
):
    print(f"Step {step.number}: {step.tool_call.tool} → {step.tool_result.data}")
```

## Adding behavior with hooks

Hooks are `@runtime_checkable` Protocols — any object with the right
method is a hook. No base class, no registry, no decorator:

```python
class ConsolePrinter:
    def post_dispatch(self, step, state, ctx):
        print(step.pretty())

class DenyShellCommands:
    def pre_dispatch(self, tool_call, state, ctx):
        if tool_call.tool == "shell":
            return "shell disabled for this task"    # non-empty string = veto
        return None

for step in composable_loop(..., hooks=[ConsolePrinter(), DenyShellCommands()]):
    ...
```

That's the whole hook API — `pre_prompt`, `pre_dispatch`, `post_dispatch`,
`check_done`, `should_stop`, `on_loop_end`. Each is optional; include only
the ones you need. See [HOOK_GUIDE.md](HOOK_GUIDE.md) for the full walkthrough.

**Testing without a real LLM.** `openharness.testing` ships a scripted
mock backend so you can unit-test hooks, tools, and your agent wiring
without hitting a provider:

```python
from openharness.testing import MockLLMBackend

llm = MockLLMBackend(responses=[
    '{"tool": "add", "args": {"a": 2, "b": 3}, "reasoning": "sum"}',
    '{"tool": "done", "args": {}, "reasoning": "finished"}',
])
```

See [`src/openharness/examples/`](src/openharness/examples/) for complete
examples including a calculator agent, research agent, and code review
agent.

## Observability

Wire a `Tracer` + `TracingHook` to capture per-step spans and feed them
to OpenTelemetry, Datadog, or any backend of your choice:

```python
from openharness import Tracer, TracingHook, MetricsCollector, MetricsHook

tracer = Tracer()
metrics = MetricsCollector()
hooks = [TracingHook(tracer), MetricsHook(metrics)]

for step in composable_loop(..., hooks=hooks):
    ...

# tracer.root_spans / metrics.snapshot() → export to OTel exporter
```

Every loop phase (`pre_prompt`, `pre_dispatch`, `post_dispatch`,
`check_done`, `on_loop_end`) emits events through `EventEmitter` so you
can also pipe live updates to a UI. See [HOOK_GUIDE.md](HOOK_GUIDE.md)
for concrete examples.

## Provenance — see exactly what your agent did

Debugging agent runs means answering two questions: *what did the LLM
actually see?* and *what trajectory did the loop take?* The
`openharness.provenance` module captures both in a few lines, with no
extra dependencies:

```python
from openharness import ProvenanceSink, composable_loop

sink = ProvenanceSink(dir="traces/run_1/")
llm = sink.wrap_llm(AnthropicBackend(...))            # capture every prompt + response
hooks = [sink.trajectory_hook()]                      # capture every step

for step in composable_loop(llm=llm, tool_registry=tools, hooks=hooks, ...):
    print(step.pretty())

sink.flush()
```

Writes a self-contained, diff-friendly directory:

```
traces/run_1/
  trajectory.json          # run_id, steps, termination reason, metadata
  steps/step_01.json       # per-step records — tool_call, tool_result, context
  steps/step_02.json
  call_00_prompt.txt       # the exact prompt sent to the LLM
  call_00_response.txt     # the raw response string (or content blocks)
  call_01_prompt.txt
  manifest.jsonl           # one LLMCall summary per line for machine parsing
```

Use `RecordingLLMBackend` / `AsyncRecordingLLMBackend` directly if you
only want LLM-call capture, or `TrajectoryRecorder` for trajectory-only.
Both accept a `redact=` callable for secret scrubbing and
`max_chars_per_call=` for bounded memory.

See [PROVENANCE_GUIDE.md](PROVENANCE_GUIDE.md) for the full API, recipes
(golden tests, cost accounting, bug-report bundles), and performance notes.

## Documentation

- [HOOK_GUIDE.md](HOOK_GUIDE.md) — writing and composing loop hooks
- [CHANGELOG.md](CHANGELOG.md) — release notes
- API reference: every public symbol is documented via docstrings (the
  package ships a `py.typed` marker).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how
to set up a development environment and submit pull requests. Please
follow the [Code of Conduct](CODE_OF_CONDUCT.md) when participating.

Security issues should be reported privately per [SECURITY.md](SECURITY.md).

## Development

```bash
uv sync
uv run pytest                 # full test suite (~4 s, 865 tests)
uv run pytest -m smoke        # smoke tests only
uv run ruff check .           # lint
uv run mypy src/openharness   # type-check
```

## Design philosophy

- **Composition over inheritance** — loops are built from hooks and
  configs, not subclassed.
- **Domain-agnostic core** — no assumption about what your agent *does*;
  you bring tools, prompts, and state shape.
- **Fail closed** — permissions, cancellation, parse recovery all default
  to the safe path.
- **Sync ↔ async parity** — `composable_loop` and `async_composable_loop`
  implement identical semantics; choose based on your backend.
- **Observable** — every loop phase emits events and records structured
  history; nothing happens inside a black box.

## Stability & versioning

`openharness` uses [semantic versioning](https://semver.org/). While the
package is pre-1.0 (currently `0.1.x`), minor versions may introduce
breaking changes to public APIs as the design stabilizes. Pin
conservatively in production:

```toml
# requirements.txt / pyproject.toml
openharness>=0.1.6,<0.2
```

Every breaking change is called out in [CHANGELOG.md](CHANGELOG.md). We
aim for a 1.0 once the loop/hook/permissions surface has a quarter of
field use without significant friction.

## License

Apache 2.0 — see [LICENSE](LICENSE).
