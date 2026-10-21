# Python API map

Start with the loop, tools, and a backend.
The other APIs are optional.
This page helps you choose an API; it does not list every helper.
Use the [Quickstart](quickstart.md) for a complete example.

Import the main APIs from `looplet`:

```python
from looplet import composable_loop, tools_from
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

Since `0.4.1`, loops reject terminal calls whose dispatch fails.
Invalid completion arguments do not produce accepted output.
The agent can correct the call while budget remains.

### Model-turn controls

**Unreleased:** Use a source checkout that includes these changes.
Explicit terminal-tool completion remains the default. To accept final assistant text instead:

```python
config = LoopConfig(
    allow_text_completion=True,
    max_tokens=4096,
    max_turn_continuations=1,
)
```

Assistant text becomes a primary terminal attempt with arguments `{"answer": text}`.
The existing `check_done` hooks and primary `output_schema` must still accept it.
Its call has an empty provider ID and `metadata["completion_source"] = "assistant_message"`.
The terminal tool's body is not executed, and no pending tool dispatch is checkpointed.
Ordinary terminal calls retain their existing execution path. Native tools take precedence over text.
Empty, refused, truncated, or structured-looking malformed responses are not promoted to completion.

For native responses, `max_turn_continuations` also bounds truncation recovery; its default is zero.
An extra request restarts the same undispatched turn with twice the previous positive `max_tokens`.
Recovery requires an explicit token limit, remaining deadline, and no cancellation.
It does not concatenate content blocks or execute tools proposed by a truncated response.
Only the complete replacement response can dispatch tools.
A stateful backend must expose synchronous `checkpoint_state` and `restore_checkpoint_state`
methods; the pre-request snapshot is deep-copied before reuse.
Backends exposing state without safe restoration do not restart. Custom stateful backends
must disclose their state through these capabilities rather than masquerading as stateless.
Existing JSON-text continuation behavior is unchanged.

Native truncation without safe recovery stops with `output_truncated`.
Refusal and incomplete provider turns stop with `model_refusal` and `model_incomplete`.
Neither is accepted completion. The sync and async loops apply the same policy.
See [structured capture](provenance.md#structured-model-call-capture) for per-request evidence and replay.

### Host-controlled assistant turns

**Unreleased:** `assistant_turn_policy=None` preserves strict tool-calling behavior.
A host can opt into valid assistant-only messages without inventing a tool call:

```python
from looplet import AssistantTurnContext, AssistantTurnDecision, LoopConfig


def decide(turn: AssistantTurnContext) -> AssistantTurnDecision:
    return AssistantTurnDecision("continue")


config = LoopConfig(
    max_steps=8,
    max_model_calls=12,
    max_assistant_turns=4,
    assistant_turn_policy=decide,
)
```

These limits are illustrative, not recommended defaults.
The policy receives detached text, turn identity, request counts, consecutive assistant count, and the next tool-step number.
The async loop also accepts an async policy. The sync loop rejects awaitable policy returns.

- `continue` retains the message and requests another model response within the remaining limits.
- `complete` proposes assistant-message completion. It requires `allow_text_completion=True`, existing completion hooks, and the primary output schema.
- `stop` ends without accepted completion or successful output.

Feedback is optional text, limited to 8,192 characters. It becomes one host message in working history.
Keep protected references, evaluator details, and unrelated private data out of feedback.
Invalid decisions and policy exceptions fail closed with `assistant_policy_error`; diagnostics retain only the exception type.
Cancellation, deadlines, and hook stops take precedence over a returned decision.
Host policies remain trusted code. Limits do not preempt a blocked synchronous callback.

The policy requires a positive integer `max_model_calls`.
This ceiling includes admitted retries, recovery, compaction summaries, and helpers using the supplied backend.
Custom backends must expose each underlying request rather than hide retries.
Built-in adapters disable SDK retries for bounded calls; cooperative retry wrappers return control between attempts.
An optional positive `max_assistant_turns` limits consecutive assistant-only turns.
Only a genuine tool proposal resets that counter. Rejected text completion does not reset it.
Tool-step budgets and `max_turn_continuations` retain their separate meanings.
Host aggregate budgets, child restrictions, and the original deadline remain authoritative.
Use an isolated backend for stateful helper or child inference, together with the host's shared budget.

Assistant continuation yields no `Step` and emits no tool-progress event.
Turn-aware context callbacks receive `model_turn_id` and `model_turn_number` when their signatures accept them.
LLM lifecycle events identify logical turns; request records and captures identify individual attempts.
Default wire progress omits prompts and intermediate response content. Trusted in-process hooks still receive their existing payloads.

Working conversation, selected prompt context, provider-held state, and immutable capture are different objects.
Built-in native adapters translate typed history without repeating rebuilt task prompts.
Custom adapters declare `conversation_mode` as `text`, `messages`, or `stateful`.
The latter two require `set_conversation_input(snapshot)`.
Stateful compaction also requires synchronous `apply_context_compaction(snapshot)` returning exactly `True`.
Missing capability or acknowledgement stops before inference with `unsupported_context`.
Provider-held context size remains unknown unless measured; local compaction alone does not prove that it shrank.
Overlapping requests through one bounded backend are rejected before a second invocation.

Checkpoints save message-only progress, copied backend state, counters, full tool receipts, and terminal output.
An uncertain admitted request is not retried implicitly after restart. Reconciliation remains the host's responsibility.
Required policy bindings must be supplied again, and saved ceilings cannot be removed or increased.
Completed checkpoint resume makes no model calls or tool effects.
This is not exactly-once inference or whole-process application recovery.
See [policy-aware replay](provenance.md#assistant-turn-capture-and-replay) for explicit replay binding.

Run the packaged offline example:

```bash
python -m looplet.examples.hello_world --scripted --assistant-turns --max-model-calls 5
```

### Step budgets

Since `0.4.1`, `DefaultState.max_steps` limits charged steps.
A successful nonterminal `ToolSpec(free=True)` call does not reduce `budget_remaining`.
Failed calls, intercepted calls, and terminal attempts still consume one step.
`step_count` includes every recorded step, including free calls.
Successful free steps record `metadata["budget_free"] = True`.
Native batches remain capped by the available step count before execution.
Custom states define their own budget accounting.
Use cancellation or a deadline to bound runs that keep making successful free calls.
See [bound free-tool runs](operations.md#bound-free-tool-runs) for a deadline example.
See the [published-version differences](install.md#unreleased-safety-fixes).

## Optional preset conveniences

`AgentPreset` holds `tools`, `state`, `config`, and `hooks`.
It represents one live run, not reusable configuration.
Call a Python factory for each task instead of caching a preset instance.
It is not another execution engine.
`preset.run(llm, task=task)` passes those objects to `composable_loop()` and returns its iterator.
`run_async()` does the same for the async loop.
You still supply the backend and task mapping explicitly.
You can inspect or change the objects before the run.

Looplet `0.4.0` includes two result helpers:

| API | Meaning |
| --- | --- |
| `preset.run_result(llm, **kwargs)` | Consume `preset.run(...)`, close the iterator, and return `RunResult.from_state(preset.state)`. |
| `await preset.run_result_async(llm, **kwargs)` | The same operation over `preset.run_async(...)`. |

```python
from looplet import minimal_preset


def run_agent(llm, task):
    with minimal_preset(tools=[lookup_owner], max_steps=5, validate=True) as run:
        return run.run_result(llm, task=task)


result = run_agent(llm, {"goal": "Find the payments owner"})
print(result.output)
```

Both helpers pass arguments to the same loop.
The caller still owns resources.
The helpers raise setup errors and unhandled execution errors.
Normal stops return results.
`preset.close()` returns separate shutdown diagnostics.
Use each preset for one run.
Call `run_agent()` again for another run with fresh state and resources.
See the [Quickstart](quickstart.md) for a complete executable example and repeated-run test.

Load a cartridge with `cartridge_to_preset()` or a bundle with `load_skill_bundle()`.
File format and loop execution remain separate.
The optional `AgentRuntime` can report host errors as failed results.

## Tools

Use `run_sub_loop` for a child loop.
Use `run_sub_loop_async` or `run_sub_loops_async` for async or bounded parallel children.
They accept explicit backends, tools, state, hooks, and configuration.
See [bounded delegation](host-integration.md#bounded-delegation) for parent controls, result fields, and recovery.

Since `0.4.1`, child loops install an isolated completion tool when cloning excludes the parent terminal.
That tool accepts a `summary` and does not invoke the parent's terminal tool.
Explicit `sub_tools` registries remain unchanged and must register their own terminal tool.

| API | Use it for |
| --- | --- |
| `tools_from([...])` | Create a registry from ordinary typed functions or tool specs and optionally add the standard `done` tool. |
| `@tool(...)` | Optionally replace a function with a non-callable `ToolSpec`. |
| `ToolSpec` | Inspect or construct a tool name, description, schema, and implementation explicitly. |
| `BaseToolRegistry` | Register, validate, look up, and dispatch tools. |
| `BaseToolRegistry.async_dispatch(...)` | Await coroutine tools directly and move blocking sync tools off the event loop. |
| `register_done_tool(...)` | Add the completion tool to a custom registry. |
| `ToolContext` | Access host resources, workspace details, cancellation, progress, and the selected backend inside a tool. |
| `ToolContext.async_llm` | Access the async backend inside async tools or child loops without the synchronous bridge. |
| `ToolError` / `ToolValidationError` | Return or test structured failures without parsing exception text. |

Start with ordinary typed functions and pass them to `tools_from` or `minimal_preset`.
Their names remain callable in application code and unit tests.
`minimal_preset` installs `done` automatically; `tools_from` does not.
Use `tools_from([...], include_done=True)` for the standard completion tool.
Register a matching terminal tool yourself when using a custom `done_tool` or `done_tools`.
Since `0.4.1`, loops warn before the first model call if no configured terminal tool is registered.
Hooks and host code can intentionally stop such runs; the warning does not reject them.
The registry infers schemas from signatures and descriptions from docstrings.
The optional `@tool` decorator replaces the function with a `ToolSpec`.
Call its `.execute(...)` method to invoke the underlying function directly.
Direct calls bypass registry validation and hooks; test dispatch separately when those controls matter.

Since `0.4.1`, tool inference also accepts `functools.partial` objects and callable instances.
A partial uses its underlying callable's name and docstring.
A callable instance uses its class name and its class or `__call__` docstring.
Use `tool(callable, name=..., description=...)` to override this metadata.
Give multiple instances or partials distinct explicit names before registering them together.
Schema inference uses the effective signature and resolves postponed annotations from the underlying function or `__call__` method.
Positionally bound parameters are absent from the schema.
Keyword-bound parameters keep their defaults and remain model-visible.
Use a wrapper or positional binding for dependencies that must not be exposed as tool arguments.

Since `0.4.1`, registration rejects positional-only parameters because dispatch uses keyword arguments.
Wrap existing positional-only functions with keyword-compatible functions.
Since `0.4.1`, schema inference also rejects `*args` and `**kwargs` parameters instead of silently omitting them.
Use explicit `list` or `dict` parameters in wrappers.
An explicit `ToolSpec` schema can expose named fields to an existing `**kwargs` implementation.
It does not convert JSON arrays into positional arguments.
JSON arrays arrive as Python lists.
Schema inference rejects `tuple`, `set`, and `frozenset` parameter annotations.
Accept a `list` parameter and convert it inside your tool or wrapper.
Explicit schemas do not convert lists into other Python containers.

Prefer explicit schemas for public tools or tools with high-risk effects.
Check required fields, invalid arguments, and effects even if a decorator creates the schema.

`ToolSpec.to_json_schema()` provides the shared argument schema for the registry and native tool calls.
It supports Python shorthand and cartridge descriptors.
It retains defaults, enum values, nested schemas, and declared constraints.
Model Context Protocol (MCP) discovery retains the server's full `inputSchema`.
New MCP servers should use JSON Schema `required` lists.
Older `(optional)` markers and JSON-string array or object repair remain supported.

Use `tools_from([...], validate=True)` or `minimal_preset(..., validate=True)` for typed Python tools.
These options check top-level types, nullability, and enum choices before effects.
They reject invalid values rather than coercing them.
Nested schemas and other constraints remain tool-owned.
These options are included in Looplet `0.4.0`.

Version `0.4.1` preserves these additional annotation contracts:

- `Annotated[T, ...]` uses the schema for `T`; metadata does not become a validation rule.
- `Any` adds no type constraint, including for explicit null values.
- Simple unions such as `str | int` retain all listed JSON types.
- Mixed or nullable `Literal` choices retain every declared JSON type, including `None`.
- Enum checks distinguish booleans from numeric values, so `True` does not match an allowed `1`.

Missing annotations retain the legacy string schema.
Required-field presence and blank-string checks still apply.
Unsupported annotations raise `TypeError` during tool construction instead of silently becoming strings.
Use an explicit `ToolSpec.parameters` schema for those contracts.
Since `0.4.1`, `minimal_preset` annotations also accept its existing callable-tool inputs and reusable sequences.

The base registry checks argument names and required fields, not full types or every JSON Schema keyword.
Without `validate=True`, type hints describe the model's schema but do not enforce values at dispatch.
Put other constraints in the tool, MCP server, or an explicit validation schema.

A required argument must be present.
A nullable schema, including an inferred `int | None`, accepts `None`.
An omitted required argument still fails.
Older blank-string and non-nullable empty-input checks do not change.

`ValidatingToolRegistry.register_with_schema(spec, schema)` adds typed argument checks to the same registry.
Its dispatch methods preserve cancellation, resources, timeouts, results, and call IDs.
Tools without a schema still use the base argument checks.
A model can see a tool without permission to run it.

Since `0.4.1`, cartridge export and load preserve supported per-tool argument
policies, including `validate=True`, without enabling validation for other tools.
Custom Python types and non-JSON enum values cannot round-trip and raise an
export error in either strict mode. See [round-trip guarantees](cartridge.md#round-trip-guarantees)
for supported policies, script dependencies, and receiving-host requirements.

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

### Optional backend contracts

Import capability protocols and wrapper helpers from `looplet.backend_contract`.
Existing simple backends remain valid. Capabilities do not require inheritance.

`ConversationInputBackend` accepts selected context. `CompactionAwareBackend` acknowledges compaction.
`ToolResultBackend` receives actual tool receipts. `CheckpointableLLMBackend` carries provider state.
`RequestAttemptBackend` reports individual provider attempts to host accounting callbacks.
`IsolatedBackend` creates fresh provider conversation state for a helper.

`BackendWrapper` forwards supported capabilities without copying live attributes.
It preserves generation signatures and sync or async methods.
Use callable attribute discovery for dynamic wrappers.
Python 3.12 and later use static inspection for runtime protocol checks.
An isolated factory returns a fresh backend. The host chooses that child's wrappers.

```python
from looplet.backend_contract import BackendRequestPolicy, BackendWrapper

wrapped = BackendWrapper(llm)
# A custom adapter can declare these fields without private flags.
request_policy = BackendRequestPolicy(
    manages_retries=True,
    supports_single_model_request=True,
)
```

An adapter's `request_policy` declares retry ownership and single-request support.
`single_model_request_active()` tells retry-capable adapters when the host owns each attempt.
Legacy retry flags remain supported. Conservative legacy declarations take precedence over permissive inherited defaults.
`TerminalLLMError` stops retries when repeating a request cannot recover safely.
The old private module retains identity-preserving aliases for existing integrations.

### Explicit provider options

```python
from looplet.backends import make_backend

llm = make_backend(
    provider="openai",
    model="your-model",
    request_options={"reasoning_effort": "high", "top_p": 0.9},
)
configuration = llm.request_configuration
```

`request_options` applies adapter-owned fields to text, native-tool and streaming requests.
`supported_request_options` lists supported adapter fields, not model or endpoint guarantees.
Unsupported fields fail explicitly. Options cannot replace prompts, tools, token limits or credentials.
Anthropic uses its own `thinking` and `output_config` objects. Effort names remain provider-specific.

`request_configuration` returns detached `requested`, `applied` and `reported` fields.
`applied` is `None` before submission. It then records fields submitted to the SDK.
`reported` remains `None`: sending a setting does not prove that a provider honored it.
The provider still validates model-specific constraints.

Cartridge model metadata remains a hint unless the host explicitly binds it to an adapter.
An explicit host backend remains authoritative. Core does not choose a reasoning policy.

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
| `replay_loop(...)` | Feed captured model responses through fresh synchronous harness execution. |
| `async_replay_loop(...)` | Replay captured responses while awaiting async hooks and tools. |
| `serialize_harness(...)` | Preserve a reviewable snapshot of the active harness. |

Replay fixes recorded model responses only.
Tools, clocks, networks, state, permissions, and random values can change.
Use the [async replay pattern](provenance.md#replay-async-agents) for async agents and early iterator cleanup.
`ProvenanceSink(redact=...)` scrubs provider prompt strings and saved evidence by default.
Use `redact_upstream=False` for record-only scrubbing when the provider is authorized to receive the original input.
Keep raw captures private; construct [reviewed synthetic fixtures](provenance.md#create-a-reviewed-replay-fixture) for versioned replay tests.
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

Graders and collectors are synchronous, including when the agent uses `async_composable_loop()`.
Await collection in your host before passing observed artifacts to synchronous graders.
See the [async-host pattern](evals.md#async-hosts) for a complete example and validation guidance.

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
| `EvalReport.save(path, evidence=...)` | Save a separate assessment to a new private JSON file, with optional relative evidence links. Unreleased. |
| `eval_mark(...)` | Attach selection marks such as `required`, `smoke`, or `slow`. |
| `run_cartridge_evals(...)` | Own collection and grading for a cartridge's self-tests, replacing embedded eval observers. |
| `save_eval_run(...)` / `load_eval_run(...)` | Persist and restore one self-contained eval record. |

Use `looplet eval init --python` for a runnable pytest starter and `looplet eval check` for trusted definition validation.
Definition validation does not execute the agent or prove its correctness.
Custom hosts can use `eval_discover()`, `eval_run()`, and `eval_run_batch()`.
For pytest, use `parametrize_cases()` and `assert_evals_pass()` as needed.
These helpers use the same graders and result rules.
Version 2 assessment APIs reject empty or measurement-only gates.
`valid` distinguishes evaluation integrity from failed agent behavior; `passed` decides the release gate.
Completion remains a separate fact.
Use `version=1` for the previous assessment policy.
Both CLI paths default to version 2 and support `--json`.
Use `--assessment-version 1` for legacy gates; `--summary-version` remains an alias.
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
