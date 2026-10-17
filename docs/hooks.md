# Hook Authoring Guide

Hooks are the primary extension mechanism in `looplet`. They let you
inject domain-specific behavior into the generic loop without modifying
any framework code.

Hooks are [`@runtime_checkable`](https://docs.python.org/3/library/typing.html#typing.runtime_checkable)
Protocols - **any object with the right methods is a hook**. No base
class, no registry, no decorator:

```python
class ConsolePrinter:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        print(f"#{step_num} {tool_call.tool} → {tool_result.data}")
        return None

for step in composable_loop(..., hooks=[ConsolePrinter()]):
    ...
```

That's the entire contract. Include only the hook methods you need; the
loop calls the ones you define and ignores the rest.

## The Hook Protocol

A hook is any Python object that implements one or more of these methods:

```python
class MyHook:
    def pre_loop(self, state, session_log, context, tools=None):
        """Called once at loop start. Setup, logging, initial state."""
        pass

    def build_briefing(self, state, session_log, context) -> str | None:
        """Return the base briefing. First non-None hook result wins."""
        return None

    def pre_prompt(self, state, session_log, context, step_num) -> str | None:
        """Called before each LLM prompt. Return text to inject into briefing."""
        return None

    def build_prompt(self, **prompt_parts) -> str | None:
        """Return a complete custom prompt, or None for the default builder."""
        return None

    def pre_dispatch(self, state, session_log, tool_call, step_num) -> ToolResult | None:
        """Called before each tool. Return a ToolResult to skip execution (cache hit)."""
        return None

    def check_permission(self, tool_call, state) -> bool:
        """Return False to deny a tool call."""
        return True

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num) -> str | None:
        """Called after each tool. Return text to inject into next prompt."""
        return None

    def check_done(self, state, session_log, context, step_num) -> str | None:
        """Called when agent calls done(). Return a string to reject (quality gate)."""
        return None

    def should_stop(self, state, step_num, new_entities) -> bool:
        """Called after each step. Return True to force loop termination."""
        return False

    def should_compact(self, state, session_log, conversation, step_num) -> bool:
        """Return True to request configured compaction before the next prompt."""
        return False

    def on_event(self, payload):
        """Observe a lifecycle event; return a HookDecision or None."""
        return None

    def on_loop_end(self, state, session_log, context, llm) -> int:
        """Called once after loop exits. Return count of extra LLM calls made."""
        return 0
```

**All methods are optional.** Implement only the ones you need. The loop checks
for method existence with `hasattr()` before calling.

Hook methods may be synchronous or `async def` when using
`async_composable_loop()`. `pre_loop` may declare `tools=` (or `**kwargs`) to
register or inspect derived tools before the run starts. The `build_prompt`
callable receives the prompt-part keyword arguments documented in the API map.

## Effect Applicability

Return one `HookDecision` from decision slots; the slot determines which
fields apply. Builders, compaction votes, and cleanup keep their dedicated
return shapes. `hook_effect_fields(slot)` in `looplet.hook_decision` exposes
the contract, and `decision.ignored_effects(slot)` reports unsupported fields.
Normalization logs ignored field names without their values.
At tool gates, it converts `Block` into a denial without changing the original decision object.

| Slot | Applied Effects |
| --- | --- |
| `pre_prompt` | `additional_context` |
| `pre_dispatch` / `pre_tool_use` | `updated_args`, `updated_result`, denial, and context |
| `check_permission` | `Deny` or `Block` denies the call |
| `post_dispatch` / `post_tool_use` / `post_tool_failure` | `updated_result`, context, and `stop` |
| `check_done` | `Block` or denial rejects completion; `Stop` is ignored |
| `should_stop` | `stop`, including its original reason |
| `pre_llm_call` / `post_llm_response` | Context and stop-after-step |
| `pre_compact` / `post_compact` | Abort compaction / `rewrite_thread`, respectively |

Other lifecycle events, including terminal `stop`, are observation-only.
Metadata and `policy_decision` are audit data, not execution authority.
Use `Deny` or `Block` to prevent tool execution at a tool gate.
`Block` also rejects completion in `check_done`.
A block takes precedence over `permission="allow"` in the same decision.
Use `Stop` in a stopping slot to end the run after the current step.
Accepted completion remains distinct from passing an outcome evaluation.

LEP hooks use this same host contract without discarding supported effects.
The adapter's directly called methods retain their existing boolean/string returns.
`Block` and `Deny` prevent tool execution through both Python and LEP hooks.

## Hook Composition

Hooks are passed as a list. All hooks fire in order for each hook point:

```python
hooks = [LoggingHook(), QualityGateHook(), MetricsHook()]
composable_loop(llm, tools=reg, hooks=hooks, ...)
```

- `pre_prompt`: All non-None returns are concatenated into the briefing
- `pre_dispatch`: Argument rewrites flow in order; the first denial or cached result short-circuits dispatch.
- `check_permission`: All permission gates must allow; an allow does not override another denial.
- `post_dispatch`: Result rewrites flow in order; context accumulates and stop requests retain their reasons.
- `check_done`: The first blocking decision rejects completion; a nonblocking decision does not.
- `should_stop`: The first stop decision or legacy True stops the loop.

Checking cached `pre_dispatch` results requires the unreleased permission fix.
See [source installation](install.md#unreleased-safety-fixes).

> When a hook terminates the loop via `should_stop`, return
> `HookDecision(stop="my_reason")` instead of a plain `True`. The reason
> string surfaces as `EvalContext.stop_reason` in saved trajectories, so
> evaluators can distinguish "agent called done()" from "budget hook
> stopped us" from "timeout hook stopped us." See the
> [evals guide](evals.md#distinguish-done-from-hook-triggered-early-stops).

## Common Hook Patterns

### 1. Progress Tracking

```python
class ProgressHook:
    def __init__(self):
        self.tool_counts: dict[str, int] = {}

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        name = tool_call.tool
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1

        if tool_result.error:
            return f"⚠ {name} failed: {tool_result.error[:100]}"
        return None  # no injection when things go well
```

### 2. Quality Gate

Prevent the agent from stopping prematurely:

```python
class MinimumEvidenceGate:
    def __init__(self, min_tools: int = 3, *, done_tools: tuple[str, ...] = ("done",)):
        self.min_tools = min_tools
        self.done_tools = done_tools

    def post_step(self, state, session_log, step_num):
        step = state.steps[-1]
        if step.tool_result.error is None and step.tool_call.tool not in self.done_tools:
            progress = state.metadata.setdefault("app", {})
            progress["successful_calls"] = progress.get("successful_calls", 0) + 1

    def check_done(self, state, session_log, context, step_num):
        successful_calls = state.metadata.get("app", {}).get("successful_calls", 0)
        if successful_calls < self.min_tools:
            return (
                f"Insufficient evidence: only {successful_calls} successful non-completion calls. "
                f"Make at least {self.min_tools} successful non-completion calls before concluding."
            )
        return None  # allow done()
```

List every completion tool in `done_tools` when using custom completion names.
`post_step` records progress before the committed checkpoint is saved.
It excludes failed calls and completion attempts, including rejected completions.
Use one gate per task and configure both
[application-state callbacks](operations.md#preserve-application-state)
to preserve this counter across restarts.
`checkpoint_dir` alone does not save application metadata.
After recovery, `state.steps` contains only the current iterator's steps;
do not reload old steps to reconstruct the counter.

This example counts successful calls; it does not judge evidence quality.
Use a collector and grader to check the task's outputs.

`check_done` accepts an optional `tool_call` kwarg carrying the
candidate `done()` invocation (and its proposed final answer). Hooks
that opt in can reject `done()` based on the answer itself, not just
the surrounding state:

```python
class GroundedAnswerGate:
    def check_done(self, state, session_log, context, step_num, tool_call):
        if "fabricated" in str(tool_call.args.get("answer", "")):
            return "answer mentions a value the agent never observed"
        return None
```

The loop dispatches with or without `tool_call` based on the hook's
signature, so legacy 4-arg implementations keep working unchanged.

### 3. Deduplication Warning

Warn when the agent repeats the same tool call:

```python
class DedupHook:
    def __init__(self):
        self._seen: set[str] = set()

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        key = f"{tool_call.tool}:{sorted(tool_call.args.items())}"
        if key in self._seen:
            return (
                f"⚠ DUPLICATE: You already called {tool_call.tool} with "
                f"these args. Try different parameters or a different tool."
            )
        self._seen.add(key)
        return None
```

### 4. Theory Tracking

Maintain and inject a running theory/hypothesis:

```python
class TheoryHook:
    def __init__(self):
        self.current_theory = "No theory yet"

    def pre_prompt(self, state, session_log, context, step_num):
        return f"Current working theory: {self.current_theory}"

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        # Update theory based on findings
        theory = tool_call.args.get("__theory__")
        if theory:
            self.current_theory = theory
        return None
```

### 5. Timeout / Budget Warning

Inject urgency as the budget runs low:

```python
class BudgetWarningHook:
    def pre_prompt(self, state, session_log, context, step_num):
        remaining = state.budget_remaining
        if remaining <= 2:
            return f"⚠ CRITICAL: Only {remaining} steps left. Wrap up NOW."
        if remaining <= 5:
            return f"Budget alert: {remaining} steps remaining."
        return None
```

### 6. Domain-Specific Guidance

Inject domain knowledge at specific stages:

```python
class CodeReviewGuidance:
    def pre_prompt(self, state, session_log, context, step_num):
        if step_num == 1:
            return "Start by listing all changed files to understand scope."
        if step_num == 3:
            return "Focus on security-sensitive patterns: auth, crypto, SQL."
        return None
```

## Using Built-In Hooks

looplet ships with several ready-to-use hooks and hook-compatible helpers:

```python
from looplet import (
    ContextBudget,
    EvalHook,
    MetricsHook,
    PermissionHook,
    ProvenanceSink,
    ThresholdCompactHook,
    TracingHook,
)
from looplet.streaming import CallbackEmitter, StreamingHook
```

### ThresholdCompactHook

Warns or compacts as context pressure rises:

```python
hook = ThresholdCompactHook(
    ContextBudget(
        context_window=128_000,
        warning_at=80_000,
        error_at=110_000,
    )
)
```

### StreamingHook

Emits typed events for real-time observability:

```python
from looplet.streaming import CallbackEmitter, StreamingHook

def on_event(event):
    print(f"[{event.event_type}] {event}")

hook = StreamingHook(CallbackEmitter(on_event))
```

### Hook decision events

`LifecycleEvent.HOOK_DECISION` fires whenever a hook returns a non-noop
`HookDecision`, including decisions returned from `on_event`. The payload
sets `hook_slot` and `hook_name`, and stores the serialized decision at
`payload.extra["decision"]`; `on_event` decisions also include the
originating lifecycle event name.

```python
from looplet import LifecycleEvent


class DecisionAudit:
    def on_event(self, payload):
        if payload.event == LifecycleEvent.HOOK_DECISION:
            print(payload.hook_slot, payload.hook_name, payload.extra["decision"])
```

Event-style hooks can also observe `LifecycleEvent.DONE_ACCEPTED` to
record the accepted `done()` payload after `check_done` has passed, e.g.
`if payload.event == LifecycleEvent.DONE_ACCEPTED: audit(payload.tool_call, payload.tool_result)`.

### ProvenanceSink

Captures prompts, responses, and trajectory files for later replay and
evaluation:

```python
from looplet import ProvenanceSink

sink = ProvenanceSink(dir="traces/run_1", redact=lambda s: s.replace("secret", "[REDACTED]"))
llm = sink.wrap_llm(llm)
hooks = [sink.trajectory_hook()]
```

## Async Hooks

The async loop accepts the same hook slots as the synchronous loop. Hook
methods may be ordinary functions or `async def`; awaitable results are
resolved sequentially in registration order so decision precedence remains
deterministic:

```python
class AsyncProgressHook:
    async def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        await save_to_database(tool_call, tool_result)
        return None
```

Sync hooks remain valid in async runs, and async hooks can return the same
`HookDecision`, text, or legacy values as sync hooks. Stream emitters remain
ordinary synchronous callbacks; keep them non-blocking when used from an
async host.

## Testing Hooks

Hooks are plain Python classes - test them in isolation:

Reuse `MinimumEvidenceGate` from above.
Record each successful step and notify `post_step` before checking completion.

```python
from looplet import DefaultState
from looplet.types import Step, ToolCall, ToolResult


def record_successful_work(hook, count):
    state = DefaultState(max_steps=count + 1)
    for number in range(1, count + 1):
        state.steps.append(
            Step(
                number=number,
                tool_call=ToolCall(tool="lookup"),
                tool_result=ToolResult(tool="lookup", args_summary="", data={"observed": True}),
            )
        )
        hook.post_step(state, None, number)
    return state


def test_quality_gate_rejects_early():
    hook = MinimumEvidenceGate(min_tools=3)
    state = record_successful_work(hook, 2)
    result = hook.check_done(state, None, None, 2)
    assert result is not None
    assert "Insufficient" in result


def test_quality_gate_allows_enough():
    hook = MinimumEvidenceGate(min_tools=3)
    state = record_successful_work(hook, 3)
    result = hook.check_done(state, None, None, 3)
    assert result is None
```

---

## Tool-internal LLM access: `ctx.llm`

Tools that accept a `ctx` parameter receive `ctx.llm` - the same LLM
backend the loop is using. This lets tools make internal LLM calls
(summarize, classify, extract) without closing over the backend:

```python
def search(*, query: str, ctx: ToolContext) -> dict:
    raw = external_api(query)
    if len(raw) > 10_000 and ctx.llm is not None:
        summary = ctx.llm.generate(f"Summarize in 3 bullets:\n{raw[:8000]}")
        return {"summary": summary, "raw_chars": len(raw)}
    return {"results": raw}
```

Tool-internal calls are:

- **Tracked** - when a `RecordingLLMBackend` is in use, internal calls
  appear in the same `manifest.jsonl` with `scope: "tool:<name>"`
- **Cost-accounted** - `CostTracker` sees them
- **Debuggable** - `python -m looplet show traces/` shows them alongside
  loop-level calls

`ctx.llm` is `None` when no LLM is available (e.g. in unit tests
without a backend). Always guard with `if ctx.llm is not None`.

For multi-step sub-tasks, use `run_sub_loop()` instead - `ctx.llm`
is for single-call internal operations.

---

## Hook-to-hook communication: `step_context`

Hooks can share data within one step through `state.step_context`.
The loop clears this dictionary before each step's `pre_prompt` hooks.
A later hook in the same step can read values written by an earlier hook.

Use `state.metadata` when data must survive into a later turn:

```python
class EntityExtractor:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        entities = extract_from(tool_result.data)
        state.metadata["entities"] = entities
        return None

class BriefingBuilder:
    def pre_prompt(self, state, session_log, context, step_num):
        entities = state.metadata.get("entities", [])
        if entities:
            return f"Entities found so far: {', '.join(entities)}"
        return None

hooks = [EntityExtractor(), BriefingBuilder()]
```

**`step_context` vs `metadata`:**

| | `state.step_context` | `state.metadata` |
|---|---|---|
| **Lifetime** | Cleared every step | Persists across turns |
| **Use for** | Ephemeral hook-to-hook data | Persistent agent metadata |
| **Cleanup** | Automatic (loop clears it) | Manual |
| **In `snapshot()`** | No | Yes |

---

## Related: provenance hook

`TrajectoryRecorder` (in `looplet.provenance`) is a hook that
captures a complete structured record of every loop run -
`pre_loop` → `pre_prompt` → `post_dispatch` → `on_loop_end` - and
serialises it to disk alongside per-LLM-call prompt/response files.
Use it whenever you want a git-diffable audit trail of what your
agent did. See [provenance.md](provenance.md).
