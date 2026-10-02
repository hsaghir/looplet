# Host Integration

Looplet is the execution kernel. A host owns identity, deployment, storage,
policy selection, and worker scheduling, then supplies those decisions through
Looplet's public runtime contracts.

## Shared host setup

```python
from looplet import LoopConfig, RunEnvelope, RunResult


envelope = RunEnvelope(
    run_id="run-123",
    request_id="request-123",
    tenant_id="tenant-a",
    actor_id="operator-7",
    deployment="staging",
    model_id="customer-model",
    policy_version="policy-4",
    trace_id="trace-123",
)

config = LoopConfig(
    max_steps=20,
    run_envelope=envelope,
    # A host retrieval adapter should implement load(state) -> str | None.
    memory_sources=[authorized_memory_source],
    cancel_token=cancel_token,
)
```

The host should pass a fresh `RunEnvelope` and runtime configuration for each
run. Looplet carries that identity into hooks, checkpoints, policy records, and
provenance metadata.

## Inspect configuration

Sync and async entry points share configuration resolution. Explicit loop
arguments override the supplied `LoopConfig`; the same mutable config and live
control objects remain available to hooks. Flat domain callbacks override the
matching `DomainAdapter` callback, which overrides the built-in fallback.

```python
config = LoopConfig(max_steps=20, system_prompt="Private instructions")
explanation = config.explain()
assert explanation["max_steps"]["value"] == 20
assert explanation["system_prompt"]["redacted"]
```

Cartridge-loaded configs record the effective declaration tier: `config.yaml`,
`runtime.yaml`, or `prompts/system.md`. Loop argument overrides and observable
host changes are labelled separately. Inherited files report their merged tier,
not a claim about which ancestor supplied each nested value. Python construction
is labelled `config`: supplying a default explicitly cannot be distinguished
from omitting it. Unset options remain unset in this view; environment-backed
defaults are still resolved by their owning services.

`explain()` does not invoke callbacks, render prompts, load memory, start
resources, or call a model. Prompts, tool metadata, provider kwargs and host
envelopes are hidden by default. `include_sensitive=True` is an explicit host
decision, not a safe logging default. Opaque live objects expose only their
type; the view is not a serialization or a complete change audit.

## Sync and async hosts

Both loop variants yield the same `Step` objects. Consume the iterator, then
build the same `RunResult` from the live state:

```python
from looplet import RunResult, composable_loop

steps = list(composable_loop(llm=llm, tools=tools, state=state, config=config, task=task))
result = RunResult.from_state(state, steps=steps)
```

For an async backend:

```python
from looplet import RunResult, async_composable_loop

steps = []
async for step in async_composable_loop(
    llm=async_llm,
    tools=tools,
    state=state,
    config=config,
    task=task,
):
    steps.append(step)
result = RunResult.from_state(state, steps=steps)
```

`RunResult` is a host-facing summary built from existing Looplet state. It
contains status, phase, termination reason, accepted `done()` output, steps,
run envelope, and persistent metadata. It does not replace the iterator or
trajectory artifacts.

## Shared execution diagnostics

`RunResult.diagnostics()`, `RunRecord.diagnostics()`,
`EvalContext.diagnostics()`, and `EvalRunRecord.diagnostics()` expose the same
`looplet.run-diagnostics.v1` content-redacted view. It reports existing run
identity, lifecycle phase, terminal category, step and typed-error counts,
logical LLM calls, elapsed time, observed provider usage, persistence warning
counts, and available lifecycle-event counts. A store's active record can show
an unmatched `pre_llm_call` without pretending the response arrived.

```python
result = RunResult.from_state(state)
diagnostic = result.diagnostics()
saved_diagnostic = run_store.load(result.run_envelope.run_id).diagnostics()
```

Direct runs must receive the host's `RunEnvelope` for cross-record identity;
anonymous/older records remain explicit about missing IDs. Existing logical
counters remain private during execution and are copied into host result
metadata under `looplet_run_stats`; they are not injected into tool metadata.
These are not provider retry or billing counts. Runtime elapsed time includes
the host wrapper; direct elapsed time comes from the loop driver.

Missing usage and cost are `None`, never assumed zero. Measured zero remains a
valid measurement. Invalid, non-finite, negative, and boolean measurements are
excluded. Failed runtime results retain already observed state metadata.

The view omits prompts, arguments, outputs, error messages, arbitrary metadata,
artifacts, protected case data, and grader scores. Structural IDs and labels are
not anonymized, and custom termination text is categorized as `custom`.
Completion is execution status, not correctness. Full records and their
`to_dict()` serializers remain content-bearing, host-owned evidence; raw model
capture and replay retain their existing explicit policy and side-effect limits.

## Model-call telemetry

Attach `StreamingHook(CallbackEmitter(callback))` through `hooks`, or supply a
`stream` emitter to either loop variant. `LLMCallEndEvent.usage` contains only
nonnegative integer `input`, `output`, `cache_read`, and `cache_write` token
counts. It may be empty or all zero when a provider omits usage; neither shape
proves a call was free. The event also has response length and duration, not
response text. Prices and cost estimates belong to the host's versioned model
catalog, not the execution event.

Do not persist streaming events wholesale as a privacy control: tool argument
summaries and optional text chunks can include customer data. A host should
allowlist the fields it retains and protect full provenance separately.

For applications that need lifecycle ownership, use the optional runtime
wrapper. It keeps direct loop usage available while coordinating run events,
logical persistence, cancellation, and preset cleanup:

```python
from looplet import AgentRuntime, FileRunStore

with AgentRuntime(preset, store=FileRunStore(".looplet/runs")) as runtime:
    result = runtime.run(llm, task=task)
```

Set `checkpoint_every_n_steps` on `AgentRuntime` when the run store should
also retain recovery checkpoints. Checkpoints remain recovery state; traces,
evals, and application artifacts remain separate evidence files referenced by
the run record.

## Memory and policy

Use `LoopConfig.memory_sources` for authorized retrieval context. A source can
be static, callable, filesystem-backed, or backed by a customer vector/hybrid
search service. Filter by tenant and agent/deployment identity before creating
the source.

For task-dependent context, provide host-authorized sources with explicit
scopes and provenance. Looplet does not discover paths or infer trust:

```python
from looplet import ScopedContextSource

config = LoopConfig(
    scoped_context_sources=[
        ScopedContextSource(
            source_id="case-a",
            source=authorized_case_source,  # implements load(state) -> str | None
            scope=lambda *, task, state, step_num: task.get("case_id") == "a",
            origin="host/cases/a",
            trust="host-checked",
            retention="run",
        ),
    ],
    scoped_context_budget_tokens=1000,
)
```

Scopes are checked before loading. Eligible sources are selected in declaration
order within the scoped-source token budget; an oversized source may still be
loaded to estimate its size. Duplicate identical IDs are deduplicated, while
conflicting IDs fail early. `retention="run"` keeps the first included value
for later turns even when its predicate no longer matches, but never across
runs; `"turn"` rechecks scope and reloads each turn. The default prompt uses a
separate `SCOPED CONTEXT` section; custom prompt builders receive selected
content alongside persistent memory through their existing `memory` argument.
Message renderers can inspect `projection.scoped_context` and
`projection.scoped_context_plan`. Prompt messages record provenance and hashes
in `metadata["scoped_context_plan"]`, not a second copy of source text.
`trust` is a host-provided label, not a security guarantee. This budget does
not cap other prompt sections or `memory_sources`.

Use Looplet hooks for runtime policy. `PermissionHook` handles declarative
allow/deny rules. `ApprovalHook` turns a tool result containing
`needs_approval=True` into `waiting_for_approval`; the host persists the
pending request, obtains a decision, and resumes from a checkpoint.

For tool authority, supply an `ExecutionPolicy` from the host and declare
requirements on `ToolSpec.capabilities` such as `network`, `workspace.write`,
or `shell`. Undeclared tools retain legacy behavior; declared tools fail
closed when the host has not granted every capability. Cartridges may declare
tool requirements, but the policy itself remains host-owned and is never
authored in `config.yaml`.

An explicit `PermissionEngine.ask(...)` rule denies when there is no
`ask_handler`, even if the engine's default is `ALLOW`. Hosts that previously
relied on that fallback must install an approval handler or remove the rule.
An approval is valid only for the same call ID, tool, arguments, and policy at
dispatch; a hook that changes the approved call or policy causes a denial.

## Bounded delegation

Pass a host-owned `ChildRunPolicy` to `run_sub_loop(..., policy=policy)` when a
child must inherit a restricted tool set, a step cap, a monotonic deadline, or
the parent's cancellation token. Supply a `SharedModelBudget` and wrap the
parent backend with `budget.wrap(llm)` to count parent and child model calls
against the same allowance. Child results and stop events carry `parent_id`.
Without a policy, direct sub-loops retain their existing independent behavior.

The bundled `subagent` tool accepts the same policy in the host-supplied
`ToolContext.metadata["child_run_policy"]`. It cannot exceed the invoking
loop's remaining step budget, even when a child requests more steps. Tool
allowlists restrict the child's executable registry; they are not merely
model-visible tool descriptions. The built-in recursion guard remains in place.

## Per-turn tool visibility

Set `LoopConfig.tool_view_selector` to a host callback receiving `step_num`,
`state`, `tools`, and `task` and returning model-visible tool names. Return
`None` for all registered tools or an empty sequence for none. Each turn uses
one `ToolView` for prompt text, native tool schemas, and cache markers; the
conversation records the selected names and a stable view version.

This is a presentation choice, not an authority boundary: a model can still
request a tool absent from its view. Use `ExecutionPolicy` or `PermissionHook`
to restrict what the child or parent loop may actually execute.

## Cancellation and checkpoints

Cancellation is cooperative and shared with LLM calls and tools:

```python
cancel_token.cancel()
```

For durable recovery, configure a `CheckpointStore` and `CheckpointHook`.
Checkpoint payloads already carry lifecycle status, phase, run envelope,
domain state, session data, and conversation data. A worker can reconstruct
resume inputs with `resume_loop_state(checkpoint)` and continue with the same
agent version and host envelope.

When `checkpoint_dir` is configured, sync and async loops persist a pending
dispatch before calling a tool. An interrupted dispatch records the call ID,
proposed and effective arguments, and `effect_unknown` status. A later resume
raises `UncertainToolEffect` before calling the model or tool again. The host
must reconcile the external effect and persist a consistent checkpoint before
resuming; clearing the pending record without reconciliation can repeat an
effect. This guard is not exactly-once execution.

## Evaluation and evidence

Choose one evaluation owner. Python hosts explicitly attach one `EvalHook`
for live collection and grading. Cartridge authors use `run_cartridge_evals()`
for shipped case suites; it replaces embedded eval observers while preserving
policy hooks. The reference coder no longer grades normal runs implicitly.

Use `save_eval_run()` / `load_eval_run()` for the same durable record in either
path, and persist that directory intact in the host's artifact store. Use
explicit `EvalResult(score=...)` for gates and `metrics=...` for measurements.
The host may use the evaluated result as a release gate; Looplet does not own
deployment promotion. Protected holdouts still stay outside the candidate's
task, runtime, and writable cartridge.

`AgentPreset` is a single-use execution plan. Build a fresh preset from the
cartridge or factory for each sequential or concurrent run; its mutable state,
hooks, and configuration are not copied between runs. Use `AgentRuntime` when
the host needs lifecycle, identity, cancellation, and persistence. Always
close the preset and any host-owned resources. `AgentPreset.close()` and
`EvalRunRecord.cleanup()` distinguish clean-up of temporary execution state
from persisted evidence.

`preset.run()` and `preset.run_async()` share validation and protocol binding,
including envelope-aware extra hooks. A setup failure releases the preset
claim so the host can repair the binding and retry before execution. Gateway
binding errors fail before execution in both modes; the sync path no longer
continues with an unbound or stale gateway. Closing an unstarted iterator also
releases its claim. Once execution starts, use a fresh preset for another run.
The synchronous iterator preserves the loop's return trace in
`StopIteration.value`, so manual iteration remains available.

The optional runtime shares event/checkpoint setup, config restoration,
persistence warnings, and claim release across sync and async drivers.
External async task cancellation still propagates as `CancelledError` while
restoring the runtime's temporary config. Background handles and bounded
shutdown remain optional host capabilities, not requirements for direct loops.
