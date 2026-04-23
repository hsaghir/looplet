# Pitfalls

Ten sharp edges worth knowing before you build on looplet. Every one of
them has a principled fix in the library — the notes below are the
"right way" for each.

## 1. `max_steps` must match in config and state

```python
# ✓ do this
N = 20
config = LoopConfig(max_steps=N)
state  = DefaultState(max_steps=N)
```

The loop warns and syncs to the `LoopConfig` value if the two differ,
but matching them silences the warning and makes intent clear.

## 2. `redact=` in provenance scrubs upstream BY DEFAULT

```python
# ✓ do this — PII never reaches Anthropic OR the trace file
sink = ProvenanceSink(dir="traces/", redact=scrub_pii)
llm  = sink.wrap_llm(AnthropicBackend(...))
```

Do not double-wrap the LLM in a separate redactor outside the sink.
The sink already scrubs the prompt before forwarding to the wrapped
backend. If you want the legacy record-only behaviour (scrub the trace
but forward the raw prompt to the provider), opt out:

```python
sink = ProvenanceSink(dir="traces/", redact=scrub_pii, redact_upstream=False)
```

## 3. Use `HookDecision(stop="reason")` in `should_stop`

```python
# ✓ do this — the reason string appears in EvalContext.stop_reason
def should_stop(self, state, step_num, new_entities):
    if self.tokens > self.cap:
        return HookDecision(stop="budget_exceeded")
    return False
```

A bare `return True` works but records `stop_reason="hook_stop"`,
which makes evaluators unable to distinguish a budget stop from a
timeout stop.

## 4. `eval_discover` only collects functions defined in the eval file

```python
# eval_my_agent.py
from looplet import eval_mark            # decorator — not collected
from my_helpers import eval_count_tools  # helper from another file — not collected

@eval_mark("verdict")
def eval_correct(ctx):                   # collected
    return ctx.final_output.get("answer") == ctx.task.get("expected")
```

This is intentional. Do not work around it by defining pass-through
wrappers. Just import normally; the `__module__` filter handles the
rest.

## 5. `should_stop` fires AFTER the current step

If a hook stops the loop, the trajectory may not end with a `done()`
call. Trajectory evaluators must handle this via `ctx.stop_reason`,
not by assuming a terminal `done()` step:

```python
def eval_finished_cleanly(ctx):
    return ctx.completed         # True iff stop_reason == "done"

def eval_no_hard_timeout(ctx):
    return ctx.stop_reason != "timeout"
```

## 6. Tool errors should carry remediation

The LLM reads `tool_result.error` and `tool_result.data` verbatim. A
good error includes both what went wrong and what to try next:

```python
# ✓ do this
return {
    "error": "File not found: x.py",
    "remediation": "Use glob to list existing files, or write to create one.",
}

# ✗ not this
return {"error": "ENOENT"}
```

## 7. Do not swallow exceptions in hooks

A hook that eats `KeyError` can mask a real bug — for example, a
missing `tool_call.args` key that should have surfaced as a prompt for
the model. Let exceptions propagate unless you have a specific recovery.

## 8. `composable_loop` is a generator

```python
# ✓ do this
for step in composable_loop(...):
    ...

# ✓ or this if you do not care about streaming
list(composable_loop(...))

# ✗ this does nothing — the loop never runs
composable_loop(...)
```

## 9. `generate_with_tools` is surfaced via hasattr

If you wrap an LLM backend yourself, forward `generate_with_tools`
when the wrapped backend has it:

```python
class MyWrapper:
    def __init__(self, inner):
        self._inner = inner
        if hasattr(inner, "generate_with_tools"):
            self.generate_with_tools = inner.generate_with_tools

    def generate(self, prompt, **kw):
        return self._inner.generate(prompt, **kw)
```

Otherwise native tool-calling silently falls back to JSON parsing.

## 10. Prefer Protocol-conforming classes over inheritance

All hooks, LLM backends, and states are `@runtime_checkable` Protocols.
Any object with the right methods works. Do not subclass `LoopHook` or
register anywhere — just implement the methods you need:

```python
# ✓ do this
class MyHook:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        ...

# ✗ do not do this
class MyHook(LoopHook):         # unnecessary
    ...
```
