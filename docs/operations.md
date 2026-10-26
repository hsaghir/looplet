# Runtime operations

Operational controls should remain separate, composable layers around the
loop. Add one when a measured failure mode justifies it, and preserve a test
for the boundary it owns.

| Need | Start with |
| --- | --- |
| Async provider calls or async host | `async_composable_loop()` |
| Transient provider failures | `ResilientBackend` |
| Growing prompt context | `ContextBudget` plus a compaction service |
| Repeated or wasteful actions | `StagnationHook`, tool limits, budget warnings |
| Tool authorization | `PermissionEngine` and `PermissionHook` |
| Live events | `StreamingHook` and an emitter |
| Local traces and aggregate metrics | `TracingHook` and `MetricsHook` |
| Crash recovery | `LoopConfig(checkpoint_dir=...)` |
| Cooperative stop | `CancelToken` |
| Host-owned run lifecycle | `AgentRuntime` + `RunStore` |

## Asynchronous loops

Use the async loop when the host already owns an event loop or the provider
adapter is asynchronous:

```python
from looplet import async_composable_loop, tools_from
from looplet.backends import AsyncOpenAIBackend


llm = AsyncOpenAIBackend.from_env()

async for step in async_composable_loop(
    llm=llm,
    tools=tools_from([lookup_owner], include_done=True),
    task={"goal": "Find the payments owner, then finish."},
    max_steps=5,
):
    await publish_step(step)
```

The async loop yields the same `Step` contract. Coroutine tools are awaited
directly; synchronous tools are moved to a worker thread by the registry.
For long-running synchronous tools, still prefer an explicit host-owned
execution boundary when stronger isolation or resource limits are required.
Hook methods may be synchronous or `async def`; Looplet awaits them in order.
The existing compaction service, terminal output schemas, routing, context
budgets, and cancellation/deadline semantics apply to async runs as well.

`ResilientBackend` wraps the synchronous backend protocol. Do not use it as an
async retry layer. Configure retries and timeouts in the async provider client
or an async adapter instead.

## Provider retries and timeouts

```python
from looplet import OpenAIBackend, ResilientBackend


base = OpenAIBackend.from_env()
llm = ResilientBackend(
    base,
    retries=3,
    timeout_s=30,
    base_delay_s=0.5,
    max_delay_s=8,
    jitter=0.2,
    retry_on=lambda exc: isinstance(exc, (TimeoutError, ConnectionError)),
)
```

`retries` is the total number of attempts, so `retries=1` performs no retry.
A non-retriable exception is raised immediately. Exhausted retriable failures
raise `RetryExhausted`, whose `attempts` list preserves every exception.

The caller-side timeout uses a daemon worker thread. It stops waiting but
cannot force every provider SDK to cancel its underlying request. Set socket
and request timeouts in the provider client as well. Avoid retrying validation,
authentication, permission, and other permanent failures.

### Tool timeouts

`ToolSpec(timeout_s=...)` sets a caller-side timeout, not an execution or effect boundary.
A synchronous worker may continue and commit effects after the caller receives `TIMEOUT`.
Async cancellation is cooperative; a coroutine may delay or suppress it.
Neither outcome proves that the external effect did not occur.

Since `0.4.1`, tool timeout exceptions are marked `retriable=False`.
Their error text warns about continuing execution and unknown effects.
`error_detail.context` includes `effect_unknown=True`.
The recovery hint names the action `reconcile_tool_effect`.
Published `0.4.0` may instead return a blank, retryable timeout error.

These diagnostics do not stop the run or prevent another dispatch.
Stop new calls in your host before investigating.
Wait for the worker or use a host-owned process boundary when execution must be contained.
Inspect the external system's authoritative state before deciding whether a retry is safe.
Do not infer retry safety from a timeout, cancellation, or failure to observe a local result.

## Standard agent harness

**Unreleased source behavior:** sync, async, preset, and cartridge execution all
resolve `LoopConfig.harness_profile="execution-v1"` through `resolve_harness()`.
The resolver performs no I/O or model calls and does not mutate supplied containers.
`resolve_harness(config, hooks).describe()` explains the selected classes and limits;
`preset.resolve_harness().describe()` provides the same view for a loaded cartridge.
Execution preserves the supplied live config and host-object identities. The automatic
compact service is installed only for the invocation; explicit in-run replacements survive.

`execution-v1` installs deterministic compaction and native typed history, not a
completion policy, repetition nudge, or derived tool quota. Plain text backends
receive continuation context without pretending that tool results are system
instructions. Custom message renderers remain authoritative. Inspection includes
effective step, output, continuation, and model-call limits.

Automatic execution compaction starts at the warning tier, around 60% estimated history pressure.
The estimate excludes some request overhead. Configure output allowance and context headroom explicitly for small windows.
Deterministic summaries retain bounded tool receipts without making a model request.
Receipt excerpts are data, not verified conclusions or instructions. Full results remain in the registry and raw steps.
Built-in compaction services change the conversation projection, not those original results.

Portable model-gateway calls use the active loop's admitted backend.
Text and native helper requests share explicit host ceilings and accounting.
Native gateway replies preserve content blocks and call identifiers. Unsupported modes fail without a text fallback.

`LoopConfig.temperature` defaults to `None`, leaving sampling to the provider.
The built-in SDK adapters omit that wire field when unset; an explicit numeric
temperature is forwarded unchanged. Standalone adapter methods and recovery calls
also default to `None`.

The following policies remain available by explicitly selecting `standard-v1`:

| Capability           | `standard-v1` policy                                                                                                   | Replacement                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Context management   | `ThresholdCompactHook` at 80% estimated history pressure; deterministic `DefaultCompactService(use_llm_summary=False)` | Supply a `compact_service`; any explicit callable `should_compact` hook replaces the automatic trigger. |
| Repetition detection | `StagnationHook(fingerprint="result", threshold=6)`; terminal tools and `think` ignored                                | Supply a `StagnationHook` with your domain policy.                                                      |
| Dispatch budget      | `ToolCallBudgetHook(max_calls=max(16, 4 * max_steps))`, including free calls                                           | Supply a `ToolCallBudgetHook` with a reviewed cap.                                                      |

The result fingerprint compares actual arguments, result values, and errors, not
generated call IDs or just payload size. Legitimate changing poll results are distinct.
The stagnation policy only adds context; it does not stop or deny work.
The dispatch guard reserves before a batch executes and stops with `tool_call_budget`.
Reservations are conservative: later denials can still consume them. The guard is
per invocation, not a durable cross-resume quota or a provider-request ceiling.
Set `max_model_calls`, a deadline, and provider timeouts explicitly for host admission controls.

```python
from looplet import LoopConfig, StagnationHook, ToolCallBudgetHook, resolve_harness


config = LoopConfig(disabled_defaults=["compaction"])
hooks = [StagnationHook(threshold=8), ToolCallBudgetHook(max_calls=100)]
print(resolve_harness(config, hooks).describe())
```

`disabled_defaults` accepts `compaction`, `stagnation`, and `tool_budget`.
It omits automatic additions, not explicit hooks or services.
`LoopConfig(harness_profile="minimal")` and `minimal_preset()` omit all automatic additions.
Unknown profile or capability names fail before model calls. The profile and opt-outs
round-trip in a cartridge's contract-tier configuration.

Defaults do not create files, configure credentials, add approval prompts, enable tracing,
or provide an OS sandbox. Native tool protocol, registry validation, and reactive recovery
remain the loop's existing mechanisms. Prompt previews and compaction are lossy, but
do not mutate original results in run history. `BaseToolRegistry` retains detached,
content-addressed originals: `result.result_key` identifies the value returned by
`registry.get_result(key)`. Existing checkpoint result stores preserve those values
across resume. Lookup and snapshots return detached copies. A host can expose recall
as a domain tool, replace the registry store, or disable new retention with
`BaseToolRegistry(retain_results=False)`. Retention consumes host memory and checkpoint
storage; it is not an external archive or a substitute for host retention policy.
Configured `tool_result_persist_dir` previews still write large original outputs to disk.
Compaction does not guarantee that retained evidence is present in the current prompt.

See the [default-harness dogfood report](default-harness-dogfood.md) for live cases,
observed failures, artifact bindings, and reproduction commands.

## Context pressure and compaction

Thresholds classify an approximate history token count, including structured tool
arguments and results. Standard source builds install a trigger and service automatically.
Override the thresholds or allow summary-model calls explicitly when appropriate:

```python
from looplet import (
    ContextBudget,
    DefaultCompactService,
    LoopConfig,
    ThresholdCompactHook,
)


budget = ContextBudget(
    context_window=128_000,
    warning_at=76_000,
    error_at=102_000,
    compact_buffer=10_000,
)

config = LoopConfig(compact_service=DefaultCompactService())
hooks = [ThresholdCompactHook(budget, fire_tier="error")]
```

The ordering must remain `warning_at < error_at < blocking_at`, where
`blocking_at` is `context_window - compact_buffer`. Keep enough buffer for the
next response and provider-added framing.

### Choose a compaction strategy

| Service | Tradeoff |
| --- | --- |
| `DefaultCompactService` | Recommended starting chain: prune bulky results, preserve a summary where possible, and fall back safely. |
| `TruncateCompact(keep_recent=N)` | No model call and deterministic, but discarded middle context is gone. |
| `SummarizeCompact(keep_recent=N)` | Retains a compact narrative but spends a model call and can omit details. |
| `PruneToolResults(keep_recent=N)` | Clears old bulky payloads while preserving conversation structure. |

Compaction is a harness behavior. Test what must survive it, such as user
constraints, artifact paths, accepted decisions, and unresolved work. Observe
`PRE_COMPACT` and `POST_COMPACT` lifecycle events when production diagnostics
need the boundary.

The history estimate is intentionally approximate, not an exact full-request budget.
System prompts, tool schemas, provider tokenization, and hidden framing can add pressure.
Set `context_window` for the actual backend and keep thresholds conservative.
The separate preflight check and reactive recovery remain in place.

## Stagnation and call limits

Use a stagnation nudge when repeated actions indicate no progress:

```python
from looplet import StagnationHook, result_size_fingerprint


stagnation = StagnationHook(
    fingerprint=result_size_fingerprint,
    threshold=3,
    ignore_tools={"done", "note"},
    nudge="The last actions did not add evidence. Try a different approach.",
)
```

An explicitly constructed `StagnationHook()` compares tool name and arguments.
The standard harness instead selects the actual-value `result` fingerprint.
The result-size fingerprint above considers only coarse result shape. For domain progress,
provide a monotonic `progress(state)` counter, such as accepted artifacts or
distinct verified records.

Stagnation detection nudges; it does not stop the loop. Add hard limits when a
resource needs an enforceable cap:

```python
from looplet import BudgetWarningHook, PerToolLimitHook


hooks = [
    BudgetWarningHook(thresholds=(0.5, 0.2)),
    PerToolLimitHook(limits={"web_search": 5, "write_file": 3}),
]
```

Tune these limits from traces. Speculative low caps often prevent valid work.

## Permissions and approval

Rules are evaluated in order; the first match wins:

```python
from looplet import PermissionDecision, PermissionEngine, PermissionHook


def approve(call, rule):
    return prompt_operator(call.tool, call.args, rule.reason)


engine = PermissionEngine(
    default=PermissionDecision.DENY,
    ask_handler=approve,
)
engine.allow("read_file", reason="read-only workspace access")
engine.ask("write_file", reason="writes need operator approval")
engine.deny(
    "bash",
    arg_matcher=lambda args: "rm " in args.get("command", ""),
    reason="destructive shell command",
)

hooks = [PermissionHook(engine)]
```

An `ASK` rule without a handler denies, even when the engine default is allow.
An invalid handler result also collapses to deny. Matcher failures fail closed: a failing
deny matcher blocks, while a failing allow matcher does not grant access.

`engine.denials` is an append-only in-memory audit list with internal
scaffolding arguments removed. Export it through a host-controlled logger if a
durable audit record is required. Permission rules are policy, not a process
sandbox. Untrusted code still needs OS or process isolation.

## Structured events

Use a callback emitter for a local UI, logger, or adapter:

```python
from looplet import StreamingHook
from looplet.streaming import CallbackEmitter


events = []


def handle_event(event):
    events.append(event)


hooks = [StreamingHook(CallbackEmitter(handle_event))]
```

`QueueEmitter` bridges to a consumer thread. `CompositeEmitter` fans out to
multiple sinks. Events include loop, step, tool dispatch, tool result, hook,
recovery, and context-pressure records. Token-level `LLMChunkEvent` requires a
streaming backend and loop stream emitter.

Choose either the loop's `stream=` parameter or a `StreamingHook` for the same
emitter path. Installing both can duplicate lifecycle events.

## Traces and aggregate metrics

```python
from looplet import MetricsCollector, MetricsHook, Tracer, TracingHook


tracer = Tracer()
metrics = MetricsCollector()
hooks = [TracingHook(tracer), MetricsHook(metrics)]

for step in composable_loop(..., hooks=hooks):
    route(step)

print(metrics.report())
for root_span in tracer.root_spans:
    export_span(root_span)
```

`TracingHook` builds local loop and tool spans. `MetricsHook` counts steps,
tool calls, errors, durations, and LLM response events. Its token fields remain
zero unless the host records provider usage separately. These are in-memory
structures, not a hosted telemetry service. Export them through an adapter
owned by the application. The [OpenTelemetry recipe](recipes.md#opentelemetry)
shows one integration shape.

Use `ProvenanceSink` when the goal is durable, replayable run evidence. Use
streaming events, spans, and metrics when the goal is live operational
visibility. They can coexist but answer different questions.

## Checkpoints and cancellation

Set a checkpoint directory to save JSON checkpoints.
A later run can resume an incomplete checkpoint:

```python
from looplet import LoopConfig


config = LoopConfig(
    max_steps=100,
    checkpoint_dir=".looplet/checkpoints/task-42",
)
```

Use a unique directory per logical task.
With no explicit `initial_checkpoint`, Looplet selects the highest-step incomplete checkpoint for that run.
A terminal checkpoint prevents implicit resume of earlier snapshots.
Calling the loop again can still start fresh; it does not undo earlier effects.

Since `0.4.1`, resumed `DefaultState` preserves the saved charged-step budget.
Changing `max_steps` changes the total cap, not the number of additional steps.
Its `steps` list contains only steps from the current iterator.
Custom states must preserve cumulative charged budgets through their own state contract.

Automatic resume refuses unreadable or invalid checkpoint files for the selected run.
This includes legacy files whose run identity cannot be verified.
Preserve damaged files and reconcile possible effects before repairing recovery state.
Other named run namespaces remain isolated.
These recovery changes require [version `0.4.1` or newer](install.md#unreleased-safety-fixes).

An interrupted tool can leave an `effect_unknown` pending record.
Resume then raises `UncertainToolEffect` before another model or tool call.
This prevents a blind retry that could duplicate an external effect.
Inspect the external system before resolving the record.
See the [worked recovery example](host-integration.md#reconcile-an-interrupted-tool).

### Preserve application state

Automatic checkpoints save conversation history, session entries, tool-result storage, and budget counters.
They do not call `state.snapshot()` or save arbitrary `state.metadata` by default.
Provide both callbacks when application state must survive a restart.

Keep application-owned JSON-safe data under `state.metadata["app"]`.
This namespace separates application data from lifecycle fields that Looplet owns.

```python
from copy import deepcopy

from looplet import DefaultState, LoopConfig


def checkpoint_application_state(loop_ctx):
    return {"app": deepcopy(loop_ctx.state.metadata.get("app", {}))}


def restore_application_state(loop_ctx, saved):
    if "app" in saved:
        loop_ctx.state.metadata["app"] = deepcopy(saved["app"])


state = DefaultState(max_steps=4, metadata={"app": {"entities": ["customer-42"]}})
config = LoopConfig(
    max_steps=4,
    checkpoint_dir=".looplet/checkpoints/task-42",
    checkpoint_state=checkpoint_application_state,
    restore_checkpoint_state=restore_application_state,
)
```

Pass `state` and `config` to either loop.
Recreate this configuration with fresh state when restarting.
The callbacks copy nested values and leave other host metadata unchanged.
Older checkpoints without `app` leave existing application metadata unchanged.
Extend this namespace for hook progress or other application data.

Do not repopulate `state.steps`.
Recovery already supplies the prior step offset.
Adding old steps can count that offset twice.
Checkpoint files contain application data; apply the same privacy controls as traces.

### Cancellation

Cancellation is cooperative:

```python
from looplet import CancelToken, LoopConfig


cancel = CancelToken()
config = LoopConfig(cancel_token=cancel)

# Another host-controlled path may call:
cancel.cancel()
```

The loop checks between turns and passes the token through `ToolContext`.
Long-running tools must poll `ctx.cancel_token.is_cancelled` or call
`raise_if_cancelled()` at safe points. Cancellation does not terminate an
arbitrary blocking system call.

Checkpoint and trace files may contain prompts, tool results, and task data.
Apply explicit access, redaction, retention, and deletion policies.

## Bound free-tool runs

Since `0.4.1`, `max_steps` limits charged steps, not total tool or model calls.
Successful nonterminal free calls preserve that budget.
The source build's standard dispatch guard also bounds successful free calls.
Add a host deadline for elapsed-time control, including when automatic guards are disabled:

```python
from time import time

from looplet import DefaultState, LoopConfig, RunEnvelope, composable_loop


config = LoopConfig(
    max_steps=8,
    run_envelope=RunEnvelope(run_id="bounded-run", deadline_at=time() + 30),
)
state = DefaultState(max_steps=config.max_steps)
for step in composable_loop(llm=llm, tools=tools, state=state, config=config, task=task):
    route(step)
```

Use this configuration with `async_composable_loop` for async runs.
The deadline uses a Unix timestamp and applies even when charged budget remains.
The loop reports `deadline_exceeded` when it observes expiry at a cancellation boundary.
This control is cooperative; it does not interrupt a provider or tool call already running.
Configure provider timeouts and cooperative tool cancellation separately.
Use a supervised process when the host requires forced termination.

## Release checklist

Before enabling an operational control in a release harness:

1. name the observed failure or requirement it addresses;
2. add one focused test that can disprove the configuration;
3. record its thresholds, defaults, and failure mode in reviewable config;
4. preserve stop reasons and errors instead of turning them into success;
5. inspect redaction and secret exposure in logs, traces, and checkpoints;
6. keep acceptance graders independent from the model's own completion claim;
7. use process or OS isolation when candidate code is untrusted.
