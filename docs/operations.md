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

The async loop yields the same `Step` contract. A synchronous tool may still
block the event loop if its implementation performs slow I/O directly. Use an
async-aware tool boundary or move blocking work to a worker owned by the host.

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

## Context pressure and compaction

Thresholds classify an approximate token count. A threshold hook requests
compaction, while `LoopConfig.compact_service` decides how to compact:

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

The estimate is intentionally approximate. Provider tokenization, tool
schemas, and hidden framing may differ. Thresholds should be conservative.

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

The default fingerprint compares tool name and arguments. The result-size
fingerprint also considers the coarse result shape. For domain progress,
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

An `ASK` rule without a handler falls back to the engine default. An invalid
handler result collapses to deny. Matcher failures also fail closed: a failing
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

Set a checkpoint directory to save JSON checkpoints and automatically resume
the latest checkpoint on a later run:

```python
from looplet import LoopConfig


config = LoopConfig(
    max_steps=100,
    checkpoint_dir=".looplet/checkpoints/task-42",
)
```

Use a unique directory per logical task. If the directory already contains
checkpoints and `initial_checkpoint` is unset, Looplet loads the checkpoint
with the highest step number.

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

## Release checklist

Before enabling an operational control in a release harness:

1. name the observed failure or requirement it addresses;
2. add one focused test that can disprove the configuration;
3. record its thresholds, defaults, and failure mode in reviewable config;
4. preserve stop reasons and errors instead of turning them into success;
5. inspect redaction and secret exposure in logs, traces, and checkpoints;
6. keep acceptance graders independent from the model's own completion claim;
7. use process or OS isolation when candidate code is untrusted.
