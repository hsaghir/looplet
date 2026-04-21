# Roadmap

> Note: this is not a confused-with [`pydantic-ai-harness`](https://github.com/pydantic/pydantic-ai-harness)
> project — `openharness` is a framework-agnostic *loop* library. See
> [README.md](README.md#what-openharness-is) for the full positioning.

This document describes what `openharness` will and will **not** become.
Dates are aspirational; the only firm commitment is the [v1.0 API
contract](#v10-api-contract).

## Guiding principles

1. **One thing well.** The core product is the iterator-first
   tool-calling loop. Anything that dilutes that focus is out of scope.
2. **Composition over configuration.** New behaviour ships as hooks or
   protocols, not as flags on `LoopConfig`.
3. **Boring dependencies.** Core runtime stays at 1 dependency
   (`pyyaml`). New features land in optional extras or separate
   packages.
4. **Frozen public surface, fluid internals.** Once a symbol is in
   `openharness/__init__.py`, breaking it requires a major bump.

## Current status — `0.1.x` (Beta)

- Composable sync + async loop, hooks as `Protocol` objects
- Tool registry with JSON-schema rendering and concurrent batching
- Fail-closed permission engine with ALLOW/DENY/ASK rules
- Checkpoint + resume, cooperative cancellation, multi-block messages
- Anthropic + OpenAI backends (sync, async, streaming)
- Provenance capture (LLM prompts + trajectories)
- pytest-style eval framework with CLI runner
- MCP tool adapter + skills bundles

## Near-term (`0.2` — ~1 month out)

- **Gemini + Bedrock backends** (community contributions welcome — see
  [good-first-issues](docs/good-first-issues.md))
- **First-class Ollama recipe** with `examples/ollama_hello.py` and
  docs page
- **Structured-output helper** — optional `response_schema` support
  that threads through to providers that have it natively
- **Cost accounting hook** built on top of the provenance sink
- **Documentation site** on GitHub Pages (mkdocs-material)

## Mid-term (`0.3` — ~2 months out)

- **Loop-level retry policies** as composable objects (not config flags)
- **Deterministic replay** — given a saved trajectory + a deterministic
  LLM cassette, re-run the loop bit-for-bit for regression testing
- **Expanded eval library** — reusable `eval_*` recipes shipped as
  `openharness.evals.recipes` (efficiency, parse-quality, IOC coverage,
  tool-error rate)
- **OpenTelemetry exporter** as a first-party optional extra

## Path to `1.0` (~3 months out)

`1.0` is shipped when:

1. The v1.0 API contract (below) has been in production for at least a
   quarter across at least three independent codebases.
2. No open issue is tagged `api-design` or `breaking`.
3. Coverage ≥ 90 % and full pyright strict passes.
4. Documentation site is feature-complete.

## Explicitly **not** on the roadmap

These belong in *other* projects, not in `openharness`:

- **A graph DSL / branching orchestrator.** Use
  [`langgraph`](https://pypi.org/project/langgraph/) or
  [`burr`](https://pypi.org/project/burr/).
- **Multi-agent handoff protocols.** Use
  [`openai-agents`](https://pypi.org/project/openai-agents/) or
  [`crewai`](https://pypi.org/project/crewai/).
- **A prompt-templating DSL.** Use
  [`dspy`](https://pypi.org/project/dspy/) or plain f-strings.
- **A vector DB / memory store.** Memory is a tool; plug in your own.
- **A web UI / dashboard.** `openharness` emits events; wire any UI
  you want on top.
- **A CLI agent-in-a-box.** Use
  [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/).
- **Fine-tuning tooling, data pipelines, synthetic-data generation.**
  Out of scope.

## v1.0 API contract

These symbols and signatures are **frozen** from `1.0` onward. Breaking
any of them requires a major-version bump.

### Loop entry points

```python
composable_loop(
    llm: LLMBackend,
    *,
    tools: BaseToolRegistry,
    task: dict[str, Any],
    state: DefaultState | None = None,
    config: LoopConfig | None = None,
    hooks: Sequence[LoopHook] | None = None,
) -> Iterator[Step]

async_composable_loop(...)   # same signature, async iterator
```

### The `Step` record

```python
@dataclass(frozen=True)
class Step:
    number: int
    tool_call: ToolCall
    tool_result: ToolResult
    ...
```

The first four fields (`number`, `tool_call`, `tool_result`, `elapsed_ms`)
are frozen. Additional fields may be added in minor versions.

### The hook protocol

Six method names are frozen:

- `pre_loop(state, session_log, context)`
- `pre_prompt(state, session_log, context, step_num) -> str | None`
- `pre_dispatch(state, session_log, tool_call, step_num) -> ToolResult | None`
- `post_dispatch(state, session_log, tool_call, tool_result, step_num) -> str | None`
- `check_done(state, session_log, context, step_num) -> str | None`
- `should_stop(state, step_num, new_entities) -> bool`
- `on_loop_end(state, session_log, context, llm) -> int`

All methods remain optional (duck-typed). Minor versions may add
optional keyword arguments with defaults, never new required ones.

### The `LLMBackend` protocol

```python
class LLMBackend(Protocol):
    def generate(self, messages: list[Message], *, tools: list[dict] | None = None,
                 cancel_token: CancelToken | None = None) -> LLMResponse: ...
```

### Tool surface

`ToolSpec`, `ToolCall`, `ToolResult`, `BaseToolRegistry` — field names
and the `register` / `dispatch` / `catalog` method signatures are frozen.

### Error classification

`ToolError` categories are frozen: `TIMEOUT`, `VALIDATION`,
`PERMISSION_DENIED`, `RATE_LIMIT`, `CONTEXT_OVERFLOW`, `CANCELLED`,
`UNKNOWN`. New categories require a major bump.

## Release cadence

- Patch (`0.1.x`): as soon as bug fixes accumulate, weekly at most.
- Minor (`0.2`, `0.3`, …): roughly monthly, with a two-week release
  candidate on PyPI (`pip install openharness==0.2.0rc1`).
- Major: only when the v1.0 contract above changes, or every 12+
  months after `1.0`.

## How to influence the roadmap

- **File an issue** tagged `roadmap` with a concrete use case.
- **Open a discussion** under the *Ideas* category.
- **Send a PR.** The fastest way to move something forward is a
  working implementation behind an optional extra.
