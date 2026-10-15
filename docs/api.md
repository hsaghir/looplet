# Python API map

Start with the loop, tools, and a backend.
The other APIs are optional.
This page helps you choose an API; it does not list every helper.
Use the [Quickstart](quickstart.md) for a complete example.

Import the main APIs from `looplet`:

```python
from looplet import composable_loop, tool, tools_from
```

Some provider and advanced APIs need a submodule import.
`looplet.__all__` lists the names that the installed version exports from `looplet`.

## Execution core

| API | Use it for |
| --- | --- |
| `composable_loop(...)` | Yield a `Step` after each tool call. |
| `async_composable_loop(...)` | Use the same loop with async iteration. |
| `LoopConfig` | Set limits, prompts, tool protocols, context reduction, checkpoints, and cancellation. |
| `LoopContext` | Provide host context to the loop and tools. |
| `DefaultState` | Record steps and stop state. Use a compatible custom state if you need more fields. |
| `Step` | Read one tool call, its result, times, and metadata. |
| `ToolCall` / `ToolResult` | Represent a call and its result as typed records. |

The loop receives explicit dependencies and yields after each tool call.
These snippets assume that you have a backend, registry, state, and hooks:

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

Use `async for` with `async_composable_loop()`.
Do not use the synchronous loop in an async handler if provider calls or tools can block its event loop.
See [runtime operations](operations.md#asynchronous-loops).

Use Python's `contextlib.closing` for early exit.
Use `aclosing` for the async loop:

```python
from contextlib import closing


with closing(composable_loop(llm=llm, tools=tools, state=state, hooks=hooks, task=task)) as steps:
    for step in steps:
        route(step)
        if should_stop(step):
            break
```

Once the loop starts, it runs each initialized end hook once on completion, failure, or explicit close.
Early closure has status `stopped` and reason `stream_closed`.
The caller owns the tools, clients, and resources that it supplies.
`RunResult.from_state(state)` returns accepted output, steps, and stop status.
Completion does not prove that an answer is correct.

## Optional preset conveniences

`AgentPreset` holds `tools`, `state`, `config`, and `hooks`.
It is not another execution engine.
`preset.run(llm, task=task)` passes those objects to `composable_loop()` and returns its iterator.
`run_async()` does the same for the async loop.
You still supply the backend and task mapping explicitly.
You can inspect or change the objects before the run.

The source version has two result helpers that are not in PyPI `0.4.0`:

| API | Meaning |
| --- | --- |
| `preset.run_result(llm, **kwargs)` | Consume `preset.run(...)`, close the iterator, and return `RunResult.from_state(preset.state)`. |
| `await preset.run_result_async(llm, **kwargs)` | The same operation over `preset.run_async(...)`. |

```python
from looplet import minimal_preset


with minimal_preset(tools=[lookup_owner], max_steps=5) as preset:
    result = preset.run_result(llm, task={"goal": "Find the payments owner"})
print(result.output)
```

Both helpers pass arguments to the same loop.
The caller still owns resources.
The helpers raise setup errors and unhandled execution errors.
Normal stops return results.
`preset.close()` returns separate shutdown diagnostics.
Use each preset for one run.
Call your Python factory again for another run.

Load a cartridge with `cartridge_to_preset()` or a bundle with `load_skill_bundle()`.
File format and loop execution remain separate.
The optional `AgentRuntime` can report host errors as failed results.

## Tools

Use `run_sub_loop` for a child loop.
Use `run_sub_loop_async` or `run_sub_loops_async` for async or bounded parallel children.
They accept explicit backends, tools, state, hooks, and configuration.
See [bounded delegation](host-integration.md#bounded-delegation) for parent controls, result fields, and recovery.

| API | Use it for |
| --- | --- |
| `@tool(...)` | Build a `ToolSpec` from an ordinary typed callable. |
| `tools_from([...])` | Create a registry from ordinary typed functions or tool specs and optionally add the standard `done` tool. |
| `ToolSpec` | Inspect or construct a tool name, description, schema, and implementation explicitly. |
| `BaseToolRegistry` | Register, validate, look up, and dispatch tools. |
| `BaseToolRegistry.async_dispatch(...)` | Await coroutine tools directly and move blocking sync tools off the event loop. |
| `register_done_tool(...)` | Add the completion tool to a custom registry. |
| `ToolContext` | Access host resources, workspace details, cancellation, progress, and the selected backend inside a tool. |
| `ToolContext.async_llm` | Access the async backend inside async tools or child loops without the synchronous bridge. |
| `ToolError` / `ToolValidationError` | Return or test structured failures without parsing exception text. |

Prefer explicit schemas for public tools or tools with high-risk effects.
Check required fields, invalid arguments, and effects even if a decorator creates the schema.

`ToolSpec.to_json_schema()` provides the shared argument schema for the registry and native tool calls.
It supports Python shorthand and cartridge descriptors.
It retains defaults, enum values, nested schemas, and declared constraints.
Model Context Protocol (MCP) discovery retains the server's full `inputSchema`.
New MCP servers should use JSON Schema `required` lists.
Older `(optional)` markers and JSON-string array or object repair remain supported.

The base registry checks argument names and required fields, not every JSON Schema keyword.
Put other constraints in the tool, MCP server, or `ValidatingToolRegistry`.

A required argument must be present.
A nullable schema, including an inferred `int | None`, accepts `None`.
An omitted required argument still fails.
Older blank-string and non-nullable empty-input checks do not change.

`ValidatingToolRegistry.register_with_schema(spec, schema)` adds typed argument checks to the same registry.
Its dispatch methods preserve cancellation, resources, timeouts, results, and call IDs.
Tools without a schema still use the base argument checks.
A model can see a tool without permission to run it.

`run_skill_bundle()` passes extra hooks and the gateway through `AgentPreset.run()`.
It preserves the iterator's returned trace.
It closes presets that it built, even if you close an iterator before its first step.
The caller still owns presets that it supplies.
Use a fresh preset for another run.

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

`ResilientBackend` is synchronous.
After a timeout, the caller stops its wait for the worker.
The worker's socket call can continue.
Set provider timeouts too.

Backend wrappers forward supported options and declared usage or state capabilities.
They expose native tool calls only if the backend supports them.
They do not forward arbitrary unsupported attributes.
Provider checkpoint methods apply to the active backend.
They do not create a common recovery format across models.

If a resilience wrapper owns retries, the loop does not retry its exhausted call again.
Nested wrappers use the outer wrapper's attempt policy, including timeout workers.
Each fallback backend can have its own attempt limit.
Provider retries and socket timeouts remain separate.
Set them explicitly if you need an exact HTTP attempt limit.

## Hook decisions

Both loops permit a stop after normal steps, rejected completion, and parse errors.
A model event can request a stop.
The loop still records and yields the current step, but it does not make another model call.
`post_step` observers see recorded error steps too.
The async loop awaits async observers.
A fatal model error remains `llm_error`; an accepted completion remains `done`.

Hooks are ordinary objects.
Implement only the methods that you need.
Use these return values:

| Decision | Meaning |
| --- | --- |
| `Continue()` | Proceed without changing the current operation. |
| `Allow()` / `Deny(reason)` | Resolve a permission boundary. |
| `Block(reason)` | Reject completion or another gated transition with feedback. |
| `Stop(reason)` | Terminate the loop with an explicit stop reason. |
| `InjectContext(text)` | Add host context to a subsequent prompt. |
| `RewriteThread(...)` | Apply a declarative thread or metadata rewrite. |

`HookDecision` is the common decision type.
The constructors above express specific decisions.
The [hook guide](hooks.md) gives method signatures, call order, and error behavior.
`async_composable_loop()` accepts sync and async hook methods.
It awaits async methods in registration order.
Callbacks passed through `stream=` remain synchronous.
They should not block.

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

Use a cartridge when you want to review or distribute an agent as one directory.
It is optional.
See [Cartridges](cartridge.md) for the schema, inheritance, references, and trust boundaries.

## Prompt preparation

Both loops prepare prompts in this order:

1. Render memory for the current state.
2. Select tools and sources that the model can see.
3. Gather prompt inputs.
4. Run an optional advisory planner.
5. Select the first hook builder that returns a value, or the configured or default builder.
6. Render through a detached `ContextProjection`.
7. Check the final prompt limit.

Prompt overflow recovery uses the same process, native schemas, and limit check.
It rebuilds inputs without repeated `pre_prompt` effects.
Both loops use a fallback if a hook builder fails.
The async loop awaits async planners and builders.
Source limits and record retention remain separate from the full prompt limit.
An advisory `ContextPlan` is not an enforced policy.
Visible tools do not imply permission to run them.
Custom builders and older keyword renderers remain supported.

## Evidence and replay

| API | Use it for |
| --- | --- |
| `ProvenanceSink` | Capture model calls and trajectory records into readable files. |
| `TrajectoryRecorder` | Lower-level step recording for custom hosts; start with `ProvenanceSink`. |
| `replay_loop(...)` | Feed captured model responses through fresh harness execution. |
| `serialize_harness(...)` | Preserve a reviewable snapshot of the active harness. |

Replay fixes recorded model responses only.
Tools, clocks, networks, state, permissions, and random values can change.
See [Capture and replay](provenance.md) for the basic use.
Use the [experiment guide](experiments.md) to choose a suitable comparison.

## Behavioral evals

An eval checks a host-owned behavioral contract against an observed outcome.
It does not hide execution inside another runner.
Choose one owner for result checks.
Python hosts attach `EvalHook` explicitly.
Cartridge authors use `run_cartridge_evals()` or `looplet eval run`.
Both save the same record with `save_eval_run()` and read it with `load_eval_run()`.
They use the same loop.

| API | Use it for |
| --- | --- |
| `load_evals(path, ...)` | Load ordinary grader and collector functions from one file or directory, plus optional JSON cases. |
| `EvalCase` / `load_cases(...)` / `save_case(...)` | Task input, grader-only expected data, marks, and notes as reviewable JSON. |
| `EvalContext` | Final output, observed artifacts, steps, stop reason, and grader task view. |
| `ctx.expected` | Read grader-only expectations without changing the existing task view. |
| `EvalResult` | Explicit gate score, passive metrics, details, or error information. Boolean graders also work. |
| `EvalReport` | Versioned assessment with `passed`, `valid`, results, and integrity failures. |
| `eval_assess(graders, ctx, ...)` | Grade an existing observation and assess it without executing an agent or collector. |
| `EvalHook` | Host-attached collection and scoring at the end of a live loop. |
| `hook.assessment(...)` | Assess the latest grades without repeating collection or grading. |
| `EvalReport.from_records(records, ...)` | Assess saved grades using a recorded or explicit trusted grader inventory. |
| `eval_mark(...)` | Attach selection marks such as `required`, `smoke`, or `slow`. |
| `run_cartridge_evals(...)` | Own collection and grading for a cartridge's self-tests, replacing embedded eval observers. |
| `save_eval_run(...)` / `load_eval_run(...)` | Persist and restore one self-contained eval record. |

Use `looplet eval init` for a plain-file starter and `looplet eval check` for trusted definition validation.
Definition validation does not execute the agent or prove its correctness.
Custom hosts can use `eval_discover()`, `eval_run()`, and `eval_run_batch()`.
For pytest, use `parametrize_cases()` and `assert_evals_pass()` as needed.
These helpers use the same graders and result rules.
Version 2 assessment APIs reject empty or measurement-only gates.
`valid` distinguishes evaluation integrity from failed agent behavior; `passed` decides the release gate.
Completion remains a separate fact.
Use `version=1` for the previous assessment policy.
Both CLI paths support `--json` and `--summary-version 2`; their default remains version 1 for compatibility.
Raw result lists, batch tables, hook strings, and durable run records retain their existing shapes.
Records without a saved directory support `cleanup()` and `with`.
Saved records retain their evidence.
`EvalHook.save()` and implicit dictionary scores are deprecated.
`promote_to_offline()` remains a fixture-saving compatibility alias, not candidate promotion.

A required grader fails if it is skipped, has an error, or misses the pass threshold.
Collector errors become explicit results.
See [Tests and graders](evals.md).

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

These controls are optional.
Add a control for a failure or runtime requirement that you have observed.
Check the behavior that it changes.
See [runtime operations](operations.md).

## Testing helpers

`MockLLMBackend` and `AsyncMockLLMBackend` return fixed responses without a network.
`LLMResponsesExhausted` reports that no fixed responses remain.
Use these backends to check tools, hooks, and loop behavior.
They do not prove that a real model will select the same actions.

```python
from looplet.testing import MockLLMBackend

llm = MockLLMBackend(responses=[first_tool_call, done_call])
```

## Optional host integration

You do not need these APIs to run or check an agent.
Add them when a host must manage sessions, events, or saved results.
See [Host integration](host-integration.md).

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

Other APIs support bundle and blueprint inspection, MCP tools, protocol checks, gateways, memory, and skills.
Choose them from the guide for your task:

- [Skills and bundles](skills.md)
- [Agent factory](agent-factory.md)
- [Portability](portability.md)
- [Recipes](recipes.md)

Before version `1.0`, minor releases can change public APIs.
Pin the minor version.
Read the [changelog](changelog.md) before an upgrade.
