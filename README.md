# openharness

[![CI](https://github.com/hsaghir/openharness/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/openharness/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A composable, tool-calling LLM agent harness — the *inner loop* of an agent,
designed to be domain-agnostic. You bring the tools, prompts, and LLM
backend; `openharness` wires together the reactive loop, retries, parse
recovery, permissions, streaming, checkpoints, and telemetry.

> Extracted from [cadence](https://github.com/hsaghir/cadence) and hardened
> for reuse across different agent domains.

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
from openharness import (
    composable_loop, LoopConfig,
    BaseToolRegistry, ToolSpec,
    DefaultState, SessionLog,
)

# 1. Define tools
tools = BaseToolRegistry()
tools.register(ToolSpec(
    name="add",
    description="Add two integers.",
    parameters={"a": "int", "b": "int"},
    execute=lambda a, b: {"sum": a + b},
))

# 2. Configure the loop
config = LoopConfig(
    max_steps=10,
    max_tokens=1024,
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

# 3. Run
state = DefaultState(max_steps=10)
session = SessionLog()
for step in composable_loop(
    llm=my_llm_backend,          # implements LLMBackend protocol
    task={"question": "What is 2 + 3?"},
    tools=tools,
    config=config,
    state=state,
    session_log=session,
):
    print(f"Step {step.number}: {step.tool_call.tool} → {step.tool_result.data}")
```

See [`src/openharness/examples/`](src/openharness/examples/) for complete
examples including a calculator agent, research agent, and code review
agent.

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

## License

Apache 2.0 — see [LICENSE](LICENSE).
