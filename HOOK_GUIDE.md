# Hook Authoring Guide

Hooks are the primary extension mechanism in `openharness`. They let you
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

Cadence ships with several ready-to-use hooks:

```python
from cadence import (
    ContextManagerHook,  # Progressive context management
    StreamingHook,       # Event emission for observability
    CheckpointHook,      # Auto-save checkpoints
    TracingHook,         # Span-based tracing
    MetricsHook,         # Metrics collection
)
```

### ContextManagerHook

Prevents context window overflow with three tiers:

```python
hook = ContextManagerHook(
    llm,                          # for LLM-based compaction
    context_window=128_000,       # your model's window size
    result_max_age_full=3,        # steps before result aging
    per_result_chars=50_000,      # max chars per result
    aggregate_chars=500_000,      # max total chars
)
```

### StreamingHook

Emits typed events for real-time observability:

```python
from cadence import StreamingHook, CallbackEmitter

def on_event(event):
    print(f"[{event.event_type}] {event}")

hook = StreamingHook(CallbackEmitter(on_event))
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

## Related: provenance hook

`TrajectoryRecorder` (in `openharness.provenance`) is a hook that
captures a complete structured record of every loop run —
`pre_loop` → `pre_prompt` → `post_dispatch` → `on_loop_end` — and
serialises it to disk alongside per-LLM-call prompt/response files.
Use it whenever you want a git-diffable audit trail of what your
agent did. See [PROVENANCE_GUIDE.md](PROVENANCE_GUIDE.md).
