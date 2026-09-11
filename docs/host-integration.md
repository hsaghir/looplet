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

## Memory and policy

Use `LoopConfig.memory_sources` for authorized retrieval context. A source can
be static, callable, filesystem-backed, or backed by a customer vector/hybrid
search service. Filter by tenant and agent/deployment identity before creating
the source.

Use Looplet hooks for runtime policy. `PermissionHook` handles declarative
allow/deny rules. `ApprovalHook` turns a tool result containing
`needs_approval=True` into `waiting_for_approval`; the host persists the
pending request, obtains a decision, and resumes from a checkpoint.

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

## Evaluation and evidence

Attach `EvalHook` for live collectors and graders, or use
`run_cartridge_evals()` for shipped case suites. Persist the resulting
`EvalRunRecord` and trajectory/provenance directory in the host's artifact
store. The host may use the evaluated result as a release gate; Looplet does
not own deployment promotion.

Always close the preset and any host-owned resources. `AgentPreset.close()`
and `EvalRunRecord.cleanup()` distinguish clean-up of temporary execution
state from persisted evidence.
