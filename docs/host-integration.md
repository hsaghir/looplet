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

Attach `EvalHook` for live collectors and graders, or use
`run_cartridge_evals()` for shipped case suites. Persist the resulting
`EvalRunRecord` and trajectory/provenance directory in the host's artifact
store. The host may use the evaluated result as a release gate; Looplet does
not own deployment promotion.

`AgentPreset` is a single-use execution plan. Build a fresh preset from the
cartridge or factory for each sequential or concurrent run; its mutable state,
hooks, and configuration are not copied between runs. Use `AgentRuntime` when
the host needs lifecycle, identity, cancellation, and persistence. Always
close the preset and any host-owned resources. `AgentPreset.close()` and
`EvalRunRecord.cleanup()` distinguish clean-up of temporary execution state
from persisted evidence.
