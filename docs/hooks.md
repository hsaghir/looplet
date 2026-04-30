# Hook Authoring Guide

Hooks are the primary extension mechanism in `looplet`. They let you
inject domain-specific behavior into the generic loop without modifying
any framework code.

Hooks are [`@runtime_checkable`](https://docs.python.org/3/library/typing.html#typing.runtime_checkable)
Protocols — **any object with the right methods is a hook**. No base
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
    def pre_loop(self, state, session_log, context):
        """Called once at loop start. Setup, logging, initial state."""
        pass

    def pre_prompt(self, state, session_log, context, step_num) -> str | None:
        """Called before each LLM prompt. Return text to inject into briefing."""
        return None

    def pre_dispatch(self, state, session_log, tool_call, step_num) -> ToolResult | None:
        """Called before each tool. Return a ToolResult to skip execution (cache hit)."""
        return None

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num) -> str | None:
        """Called after each tool. Return text to inject into next prompt."""
        return None

    def check_done(self, state, session_log, context, step_num) -> str | None:
        """Called when agent calls done(). Return a string to reject (quality gate)."""
        return None

    def should_stop(self, state, step_num, new_entities) -> bool:
        """Called after each step. Return True to force loop termination."""
        return False

    def on_loop_end(self, state, session_log, context, llm) -> int:
        """Called once after loop exits. Return count of extra LLM calls made."""
        return 0
```

**All methods are optional.** Implement only the ones you need. The loop checks
for method existence with `hasattr()` before calling.

## Hook Composition

Hooks are passed as a list. All hooks fire in order for each hook point:

```python
hooks = [LoggingHook(), QualityGateHook(), MetricsHook()]
composable_loop(llm, tools=reg, hooks=hooks, ...)
```

- `pre_prompt`: All non-None returns are concatenated into the briefing
- `pre_dispatch`: First hook to return non-None wins (intercepts the call)
- `post_dispatch`: All non-None returns are accumulated for the next prompt
- `check_done`: First hook to return non-None rejects the done() call
- `should_stop`: First True stops the loop

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
    def __init__(self, min_tools: int = 3):
        self.min_tools = min_tools

    def check_done(self, state, session_log, context, step_num):
        if len(state.steps) < self.min_tools:
            return (
                f"Insufficient evidence: only {len(state.steps)} tools called. "
                f"Call at least {self.min_tools} before concluding."
            )
        return None  # allow done()
```

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
            return "⚠ CRITICAL: Only {remaining} steps left. Wrap up NOW."
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

For the async loop, implement the same methods as `async def`:

```python
class AsyncProgressHook:
    async def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        await save_to_database(tool_call, tool_result)
        return None
```

The async loop checks `asyncio.iscoroutinefunction()` and properly `await`s
async hooks while also supporting sync hooks in the same hook list.

## Testing Hooks

Hooks are plain Python classes — test them in isolation:

```python
def test_quality_gate_rejects_early():
    hook = MinimumEvidenceGate(min_tools=3)
    state = MockState(steps=[1, 2])  # only 2 steps
    result = hook.check_done(state, None, None, 2)
    assert result is not None  # should reject
    assert "Insufficient" in result

def test_quality_gate_allows_enough():
    hook = MinimumEvidenceGate(min_tools=3)
    state = MockState(steps=[1, 2, 3])  # enough
    result = hook.check_done(state, None, None, 3)
    assert result is None  # should allow
```

---

## Tool-internal LLM access: `ctx.llm`

Tools that accept a `ctx` parameter receive `ctx.llm` — the same LLM
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

- **Tracked** — when a `RecordingLLMBackend` is in use, internal calls
  appear in the same `manifest.jsonl` with `scope: "tool:<name>"`
- **Cost-accounted** — `CostTracker` sees them
- **Debuggable** — `python -m looplet show traces/` shows them alongside
  loop-level calls

`ctx.llm` is `None` when no LLM is available (e.g. in unit tests
without a backend). Always guard with `if ctx.llm is not None`.

For multi-step sub-tasks, use `run_sub_loop()` instead — `ctx.llm`
is for single-call internal operations.

---

## Hook-to-hook communication: `step_context`

Hooks sometimes need to share data within the same step — e.g. an
entity-extraction hook writes entities in `post_dispatch`, and a
briefing hook reads them in the next step's `pre_prompt`.

Use `state.step_context` for this. The loop clears it to `{}` at the
start of every step, so it's ephemeral — no manual cleanup needed:

```python
class EntityExtractor:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        entities = extract_from(tool_result.data)
        state.step_context["entities"] = entities
        return None

class BriefingBuilder:
    def pre_prompt(self, state, session_log, context, step_num):
        entities = state.step_context.get("entities", [])
        if entities:
            return f"Entities found so far: {', '.join(entities)}"
        return None

# Both hooks share data within each step:
hooks = [EntityExtractor(), BriefingBuilder()]
```

**`step_context` vs `metadata`:**

| | `state.step_context` | `state.metadata` |
|---|---|---|
| **Lifetime** | Cleared every step | Persists entire run |
| **Use for** | Ephemeral hook-to-hook data | Persistent agent metadata |
| **Cleanup** | Automatic (loop clears it) | Manual |
| **In `snapshot()`** | No | Yes |

---

## Related: provenance hook

`TrajectoryRecorder` (in `looplet.provenance`) is a hook that
captures a complete structured record of every loop run —
`pre_loop` → `pre_prompt` → `post_dispatch` → `on_loop_end` — and
serialises it to disk alongside per-LLM-call prompt/response files.
Use it whenever you want a git-diffable audit trail of what your
agent did. See [provenance.md](provenance.md).
