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
| `BaseToolRegistry.async_dispatch(...)` | Await coroutine tools directly and move blocking sync tools off the event loop. |
| `register_done_tool(...)` | Add the completion tool to a custom registry. |
| `ToolContext` | Access host resources, workspace details, cancellation, progress, and the selected backend inside a tool. |
| `ToolError` / `ToolValidationError` | Return or test structured failures without parsing exception text. |

Prefer explicit schemas for public or high-risk tools. Decorator inference is
convenient, but a release contract should still test required fields, invalid
arguments, and side effects.

`ToolSpec.to_json_schema()` is the shared parameter view for dispatch names
and requiredness, native-tool presentation, and registry introspection.
Python shorthand and cartridge descriptors remain supported; known type
aliases and optional markers are normalized without discarding defaults,
enums, nested schemas, or other declared constraints. MCP discovery preserves
the server's full `inputSchema`, rather than flattening it to descriptions.
New MCP servers should use standard JSON Schema `required` lists; legacy
`(optional)` type markers and JSON-string array/object repair remain supported.

The base registry checks argument names and required fields, not every JSON
Schema keyword. Keep semantic and richer constraint validation in the tool,
MCP server, or an explicit `ValidatingToolRegistry` contract.

`ValidatingToolRegistry.register_with_schema(spec, schema)` adds optional typed
argument validation to the same registry execution path. Its inherited
`dispatch`, `dispatch_batch`, `async_dispatch`, and `async_dispatch_batch` keep
the base cancellation, resource, timeout, result, and call-ID contracts. Tools
registered without a schema still use ordinary registry argument checks; model
visibility is not execution authority.

Bundle execution is an adapter over the preset's execution contract, not a
second runtime. `run_skill_bundle()` binds extra hooks and the gateway through
`AgentPreset.run()`, preserves the iterator's returned trace, and closes
helper-built presets even when an unstarted iterator is explicitly closed.
Caller-supplied presets remain caller-owned and follow the same single-use
rule as direct preset runs; use a fresh preset for another execution.

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

Recording, resilience, routing, and legacy cost wrappers share supported-option
forwarding and declared usage/state capabilities. Native generation is surfaced
only when the underlying backend supports it; unsupported arbitrary attributes
are not forwarded. Provider checkpoint methods apply to the active underlying
backend, not an invented cross-model recovery format.

When a resilience wrapper owns retries, the sync/async call helper does not retry
its exhausted call again. Nested resilience wrappers use the outer wrapper's
attempt policy, including in timeout workers. Each fallback backend may retain
its own resilience budget. Provider SDK retries and socket timeouts remain
separate; configure them explicitly when an exact HTTP attempt bound is needed.

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
order, method signatures, composition, and error handling. In
`async_composable_loop`, hook methods may be sync or async and are awaited in
registration order. Stream emitters passed through `stream=` remain ordinary
synchronous callbacks and should stay non-blocking.

## Cartridges and presets

| API | Use it for |
| --- | --- |
| `AgentPreset` | In-memory composition of backend-independent harness pieces. |
| `AgentBlueprint` / `blueprint_from_bundle(...)` | Inspect stable agent structure without retaining the temporary runtime preset. |
| `ContextProjection` | Immutable snapshot passed to `render_messages_override` for custom prompt rendering. |
| `validate_preset_contract(...)` | Validate a built preset at a bundle/host boundary before execution. |
| `bundled_cartridge_path(name)` | Resolve a reference cartridge shipped in the source tree or installed distribution. |
| `Cartridge` / `CartridgeLayout` | Parsed file-native harness and its paths. |
| `cartridge_to_preset(...)` | Load a cartridge into the same preset used by Python callers. |
| `preset_to_cartridge(...)` | Write a supported preset surface as reviewable files. |
| `scaffold_cartridge(...)` | Create the initial cartridge structure programmatically. |
| `resource_ref_for(...)` | Resolve a host resource reference during round-trip serialization. |

Use a cartridge when prompts, tools, hooks, resources, and self-tests benefit
from one review and distribution unit. It is optional. Read
[cartridges](cartridge.md) for schema, inheritance, references, and trust
boundaries.

## Prompt preparation

Both loop drivers share a prompt-preparation pipeline: render ordinary memory
for the current state, select model-visible tools, select scoped sources,
gather prompt inputs, run an optional advisory planner, choose the first
non-`None` hook builder or configured/default builder, render through a
detached `ContextProjection`, and check the final prompt budget.

Overflow recovery uses the same pipeline, selected native schemas, and budget
check. It rebuilds state-dependent inputs without repeating `pre_prompt`
side effects. Hook builder failures fall back in both drivers; async planners
and builders are awaited by the async driver. Scoped-source budgets and run
retention remain separate from the full rendered-prompt limit. An advisory
`ContextPlan` does not become an enforced policy, and tool visibility does not
grant execution authority. Existing custom builders and legacy keyword
renderers remain supported.

## Evidence and replay

| API | Use it for |
| --- | --- |
| `ProvenanceSink` | Capture model calls and trajectory records into readable files. |
| `TrajectoryRecorder` | Lower-level step recording for custom hosts; start with `ProvenanceSink`. |
| `replay_loop(...)` | Feed captured model responses through fresh harness execution. |
| `serialize_harness(...)` | Preserve a reviewable snapshot of the active harness. |

Replay fixes recorded model responses only. It does not freeze tools, clocks,
networks, state, permissions, or randomness. Start with
[capture and replay](provenance.md), then use the
[experiment guide](experiments.md) to choose the right control.

## Behavioral evals

Choose one evaluation owner: Python hosts attach `EvalHook` explicitly;
cartridge authors use `run_cartridge_evals()` or `looplet eval run`. Both write
the same durable record through `save_eval_run()` and read it through
`load_eval_run()`. Neither adds a second execution kernel.

| API | Use it for |
| --- | --- |
| `EvalCase` / `load_cases(...)` / `save_case(...)` | Task input, grader-only expected data, marks, and notes as reviewable JSON. |
| `EvalContext` | Final output, observed artifacts, steps, stop reason, and grader task view. |
| `EvalResult` | Explicit gate score, passive metrics, details, or error information. Boolean graders also work. |
| `EvalHook` | Host-attached collection and scoring at the end of a live loop. |
| `eval_mark(...)` | Attach selection marks such as `required`, `smoke`, or `slow`. |
| `run_cartridge_evals(...)` | Own collection and grading for a cartridge's self-tests, replacing embedded eval observers. |
| `save_eval_run(...)` / `load_eval_run(...)` | Persist and restore one self-contained eval record. |

For custom runners or pytest, compose `eval_discover()`, `eval_run()`,
`eval_run_batch()`, `parametrize_cases()`, and `assert_evals_pass()` as needed.
They reuse the same graders and verdict rules. No-output records expose
`cleanup()` and support `with`; persisted records retain their evidence.
`EvalHook.save()` and implicit dictionary score inference are deprecated
compatibility paths, not recommended alternatives.

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

## Optional host integration

These are not prerequisites for running or evaluating an agent. Add them when
the host needs lifecycle ownership; see [Host integration](host-integration.md).

| API | Use it for |
| --- | --- |
| `AgentRuntime` / `RunHandle` | Host lifecycle wrapper with cancellation, events, persistence, and cleanup. |
| `ExecutionSession` | Group related run IDs and forked host sessions. |
| `RunResult` | Host-facing summary of status, termination, output, steps, envelope, and metadata. |
| `RunEnvelope` | Host-supplied identity, deployment, policy, trace, and deadline context. |
| `RunStatus` / `RunPhase` | Lifecycle enums for host observers and result consumers. |
| `RunEvent` / `ArtifactRef` | Host event and artifact-reference records. |
| `MemoryRunStore` / `FileRunStore` | Logical run persistence over checkpoints and artifacts. |
| `ContextPlan` | Advisory prompt selection and budget plan through `ContextProjection`. |

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