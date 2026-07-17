# Python API map

Looplet exposes a small execution core and several optional layers around it.
This page is a curated map for choosing the right surface. It is not a dump of
every exported helper.

Most applications should begin with imports from `looplet`:

```python
from looplet import composable_loop, tool, tools_from
```

Provider-specific, streaming, and lower-level compatibility APIs may live in
their defining submodule. The package-level `looplet.__all__` is the canonical
list of re-exported names for a given release.

## Execution core

| API | Use it for |
| --- | --- |
| `composable_loop(...)` | Synchronous iterator-first tool loop that yields one `Step` per dispatch. |
| `async_composable_loop(...)` | Async generator with the same loop contract. |
| `LoopConfig` | Runtime limits, prompt settings, native tools, compaction, checkpointing, cancellation, and related policy. |
| `LoopContext` | Host context made available to loop and tool execution. |
| `DefaultState` | Default mutable loop state. Supply a compatible custom state when the domain needs more fields. |
| `Step` | One parsed tool call and its result, timing, classification, and related metadata. |
| `ToolCall` / `ToolResult` | Typed call and result records used by registries, hooks, and tests. |

The loop accepts explicit dependencies and returns control after each
dispatch:

```python
for step in composable_loop(
    llm=llm,
    tools=tools,
    task={"goal": "Inspect the repository and report one risk."},
    hooks=hooks,
    config=LoopConfig(max_steps=8),
):
    route(step)
```

Use `async for` with `async_composable_loop()`. Do not run the synchronous
generator in an async request handler when provider calls or tools may block
the event loop. See [runtime operations](operations.md#asynchronous-loops).

## Tools

| API | Use it for |
| --- | --- |
| `@tool(...)` | Build a `ToolSpec` from an ordinary typed callable. |
| `tools_from([...])` | Create a registry from tool specs and optionally add the standard `done` tool. |
| `ToolSpec` | Inspect or construct a tool name, description, schema, and implementation explicitly. |
| `BaseToolRegistry` | Register, validate, look up, and dispatch tools. |
| `register_done_tool(...)` | Add the completion tool to a custom registry. |
| `ToolContext` | Access host resources, workspace details, cancellation, progress, and the selected backend inside a tool. |
| `ToolError` / `ToolValidationError` | Return or test structured failures without parsing exception text. |

Prefer explicit schemas for public or high-risk tools. Decorator inference is
convenient, but a release contract should still test required fields, invalid
arguments, and side effects.

## Backends

| API | Import | Notes |
| --- | --- | --- |
| `OpenAIBackend` | `from looplet import OpenAIBackend` | Sync OpenAI and OpenAI-compatible adapter with native tool support. |
| `AnthropicBackend` | `from looplet import AnthropicBackend` | Sync Anthropic adapter with native tool support. |
| `AsyncOpenAIBackend` | `from looplet.backends import AsyncOpenAIBackend` | Async OpenAI-compatible adapter. |
| `OpenAIStreamingBackend` | `from looplet.backends import OpenAIStreamingBackend` | OpenAI adapter exposing token chunks. |
| `LLMBackend` | `from looplet import LLMBackend` | Runtime-checkable shape for custom synchronous backends. |
| `ResilientBackend` | `from looplet import ResilientBackend` | Sync retry, backoff, and caller-side timeout wrapper. |

Install provider extras separately. See [install and configure](install.md).

`ResilientBackend` is synchronous. Its timeout abandons a daemon worker from
the caller's perspective; it cannot guarantee that a provider SDK cancels the
underlying socket operation. Configure provider-level timeouts too.

## Hook decisions

Hooks are duck-typed objects. Implement only the lifecycle methods the policy
needs. Return values are normalized through these decisions:

| Decision | Meaning |
| --- | --- |
| `Continue()` | Proceed without changing the current operation. |
| `Allow()` / `Deny(reason)` | Resolve a permission boundary. |
| `Block(reason)` | Reject completion or another gated transition with feedback. |
| `Stop(reason)` | Terminate the loop with an explicit stop reason. |
| `InjectContext(text)` | Add host context to a subsequent prompt. |
| `RewriteThread(...)` | Apply a declarative thread or metadata rewrite. |

`HookDecision` is the underlying normalized form. The convenience constructors
make intent easier to review. The [hook guide](hooks.md) documents lifecycle
order, method signatures, composition, and error handling.

## Cartridges and presets

| API | Use it for |
| --- | --- |
| `AgentPreset` | In-memory composition of backend-independent harness pieces. |
| `Cartridge` / `CartridgeLayout` | Parsed file-native harness and its paths. |
| `cartridge_to_preset(...)` | Load a cartridge into the same preset used by Python callers. |
| `preset_to_cartridge(...)` | Write a supported preset surface as reviewable files. |
| `scaffold_cartridge(...)` | Create the initial cartridge structure programmatically. |
| `resource_ref_for(...)` | Resolve a host resource reference during round-trip serialization. |

Use a cartridge when prompts, tools, hooks, resources, and self-tests benefit
from one review and distribution unit. It is optional. Read
[cartridges](cartridge.md) for schema, inheritance, references, and trust
boundaries.

## Evidence and replay

| API | Use it for |
| --- | --- |
| `ProvenanceSink` | Capture model calls and trajectory records into readable files. |
| `TrajectoryRecorder` | Record step-level trajectory evidence directly. |
| `replay_loop(...)` | Feed captured model responses through fresh harness execution. |
| `serialize_harness(...)` | Preserve a reviewable snapshot of the active harness. |

Replay fixes recorded model responses only. It does not freeze tools, clocks,
networks, state, permissions, or randomness. Start with
[capture and replay](provenance.md), then use the
[experiment guide](experiments.md) to choose the right control.

## Behavioral evals

| API | Use it for |
| --- | --- |
| `EvalCase` | Task input, grader-only expected data, marks, and notes. |
| `EvalContext` | Final output, observed artifacts, steps, stop reason, and grader task view. |
| `EvalResult` | Normalized score, label, metrics, and error information. |
| `EvalHook` | Collect and score at the end of a live loop. |
| `eval_discover(...)` | Find locally defined `eval_*` functions. |
| `eval_run(...)` / `eval_run_batch(...)` | Execute graders for one or many contexts. |
| `eval_mark(...)` | Attach selection marks such as `required`, `smoke`, or `slow`. |
| `load_cases(...)` / `save_case(...)` | Read and write JSON case corpora. |
| `parametrize_cases(...)` | Turn case files into ordinary pytest parameters. |
| `assert_evals_pass(...)` | Run discovered graders and raise one useful assertion on failure. |
| `run_cartridge_evals(...)` | Execute a cartridge's cases, collectors, and graders end to end. |
| `save_eval_run(...)` / `load_eval_run(...)` | Persist and restore one self-contained eval record. |

Required graders fail closed when skipped, errored, or below the pass boundary.
Collector errors are explicit results rather than silent missing evidence. See
[behavioral evals](evals.md).

## Runtime controls

| Concern | Primary API |
| --- | --- |
| Context pressure | `ContextBudget`, `ThresholdCompactHook`, `DefaultCompactService` |
| Deterministic truncation | `TruncateCompact` |
| Model-assisted summary | `SummarizeCompact` |
| Large tool payloads | `PruneToolResults` |
| Permissions | `PermissionEngine`, `PermissionRule`, `PermissionHook` |
| Human approval boundary | `ApprovalHook` |
| Repeated actions | `StagnationHook` |
| Step budget warnings | `BudgetWarningHook` |
| Per-tool call caps | `PerToolLimitHook` |
| Cooperative cancellation | `CancelToken` |
| JSON checkpoints | `FileCheckpointStore` and `LoopConfig.checkpoint_dir` |
| Spans and aggregate metrics | `Tracer`, `TracingHook`, `MetricsCollector`, `MetricsHook` |
| Typed live events | `StreamingHook` and emitters from `looplet.streaming` |

These controls are opt-in. Add one because an observed failure or operational
requirement justifies it, then test that boundary. See
[runtime operations](operations.md).

## Testing helpers

`MockLLMBackend` and `AsyncMockLLMBackend` return scripted responses without a
network. `LLMResponsesExhausted` makes missing scripted responses explicit.
Use them for deterministic harness mechanics, not as evidence that a real
model will choose the same actions.

```python
from looplet.testing import MockLLMBackend

llm = MockLLMBackend(responses=[first_tool_call, done_call])
```

## Specialized surfaces

Looplet also exports bundle and blueprint analysis, MCP tools, native-tool
probing, state and model gateway clients, memory sources, skills, and preset
helpers. These support cartridge portability, factory output, and advanced
host integrations. Start from their task guide rather than selecting them by
name:

- [Skills and bundles](skills.md)
- [Agent factory](agent-factory.md)
- [Portability](portability.md)
- [Recipes](recipes.md)

Before `1.0`, minor releases may revise public APIs. Pin the minor line and
review the [changelog](changelog.md) when upgrading.