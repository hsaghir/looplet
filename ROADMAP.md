# Roadmap

> Note: this is not a confused-with [`pydantic-ai-harness`](https://github.com/pydantic/pydantic-ai-harness)
> project — `looplet` is a framework-agnostic *loop* library. See
> [README.md](https://github.com/hsaghir/looplet#what-looplet-is) for the full positioning.

This document describes what `looplet` will and will **not** become.
Dates are aspirational; the only firm commitment is the [v1.0 API
contract](#v10-api-contract).

## Guiding principles

1. **One thing well.** The core product is the iterator-first
   tool-calling loop. Anything that dilutes that focus is out of scope.
2. **Composition over configuration.** New behaviour ships as hooks or
   protocols, not as flags on `LoopConfig`.
3. **Boring dependencies.** Core runtime has **zero** third-party
   packages — the standard library only. New features land in
   optional extras or separate packages.
4. **Frozen public surface, fluid internals.** Once a symbol is in
   `looplet/__init__.py`, breaking it requires a major bump.

## Current status — `0.1.x` (Beta)

- Composable sync + async loop, hooks as `Protocol` objects
- Tool registry with JSON-schema rendering and concurrent batching
- Fail-closed permission engine with ALLOW/DENY/ASK rules
- Checkpoint + resume, cooperative cancellation, multi-block messages
- Anthropic + OpenAI backends (sync, async, streaming)
- Provenance capture (LLM prompts + trajectories)
- pytest-style eval framework with CLI runner
- MCP tool adapter + skills bundles
- Decorator-first tool construction via `@tool` and `tools_from()`
- Native-tool protocol probing for OpenAI-compatible proxy mismatches
- `looplet doctor` diagnostics for local setup and backend checks

## A+ polish track

The next product goal is to make looplet feel obvious from the first
GitHub page through the first custom agent. The short pitch should stay
consistent everywhere:

> looplet exposes the agent loop as an iterator, makes every step
> observable, and lets users compose behavior with hooks.

### Custom-agent example to lead with

Lead with **Dependency Doctor**: an agent that audits a repository's
dependency files for security, license, and maintenance risk, then
produces a report card. It is more memorable than hello-world, useful to
most developers, and it demonstrates looplet's differentiation: every
lookup, warning, and conclusion is visible as a `Step` that users can
log, gate, replay, or evaluate.

Keep the other examples as second-line demos:

- **Git Detective** for repository-health analysis from commit history.
- **Threat Intel Briefing** for local-first security analysis.
- **Coder** as a reference implementation for tool-heavy agents, not as
  a claim to be a complete coding product.

### API consolidation

Keep the low-level modules, but make the common path feel smaller:

- Promote one front-door import story: `looplet` for essentials,
  submodules for advanced internals.
- Make presets match the best examples. `coding_agent_preset()` should
  eventually reuse the hardened file tools, stale-file hints, protocol
  probing, and test guardrails from `examples/coder/agent.py` instead of
  maintaining a simpler parallel version.
- Group production features into opinionated bundles: `debugging`,
  `safety`, `coding`, and `research` presets should assemble hooks,
  memory, compaction, provenance, and permissions with sane defaults.
- Add a `looplet doctor` command that verifies backend connectivity,
  native-tool behavior, model JSON compliance, and common config errors.

### Tool construction

Manual `ToolSpec(...)` construction should remain supported, but the
happy path should be decorator-first:

```python
from looplet import tool, tools_from

@tool(description="Search the docs by keyword.", concurrent_safe=True)
def search_docs(query: str, limit: int = 5) -> dict:
    return {"results": search(query, limit)}

tools = tools_from([search_docs])
```

The decorator should infer JSON Schema from type hints, mark parameters
with defaults as optional, use docstrings as descriptions when no
description is provided, preserve `ctx` injection, and still return a
plain `ToolSpec` so advanced users can inspect or mutate it.

## Near-term (`0.2` — ~1 month out)

- **Preset consolidation** — make `coding_agent_preset()` reuse the
  hardened file-tool and guardrail patterns from `examples/coder/`.
- **Production bundles** — opinionated `debugging`, `safety`, `coding`,
  and `research` preset bundles that assemble hooks and memory defaults.
- **Gemini + Bedrock backends** (community contributions welcome — see
  [good-first-issues](good-first-issues.md))
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
  `looplet.evals.recipes` (efficiency, parse-quality, IOC coverage,
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

These belong in *other* projects, not in `looplet`:

- **A graph DSL / branching orchestrator.** Use
  [`langgraph`](https://pypi.org/project/langgraph/) or
  [`burr`](https://pypi.org/project/burr/).
- **Multi-agent handoff protocols.** Use
  [`openai-agents`](https://pypi.org/project/openai-agents/) or
  [`crewai`](https://pypi.org/project/crewai/).
- **A prompt-templating DSL.** Use
  [`dspy`](https://pypi.org/project/dspy/) or plain f-strings.
- **A vector DB / memory store.** Memory is a tool; plug in your own.
- **A web UI / dashboard.** `looplet` emits events; wire any UI
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
  candidate on PyPI (`pip install looplet==0.2.0rc1`).
- Major: only when the v1.0 contract above changes, or every 12+
  months after `1.0`.

## How to influence the roadmap

- **File an issue** tagged `roadmap` with a concrete use case.
- **Open a discussion** under the *Ideas* category.
- **Send a PR.** The fastest way to move something forward is a
  working implementation behind an optional extra.
