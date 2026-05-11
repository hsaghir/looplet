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

## 11. Don't run linters / type-checkers / LSP after every `write`

A good editing trajectory looks like: `write A`, `write B`, `edit C`,
then `done()`. The intermediate states almost always fail to compile
or type-check — that's normal, the work isn't finished yet.

If you wire a `post_dispatch` hook that runs `mypy` / `tsc` / LSP
diagnostics after every edit and injects the errors as
`InjectContext(...)`, the model receives a constant stream of "you
broke it" feedback during a sequence of edits that, taken together,
would have been correct. Models then abandon multi-step refactors and
revert to single-edit-then-verify patterns that are objectively worse.

```python
# ✗ do not do this
class LSPFeedback:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool in {"write", "edit"}:
            errors = run_typechecker()        # noisy mid-sequence
            if errors:
                return InjectContext(f"Type errors:\n{errors}")
        return None

# ✓ do this — only check at natural sync points (done() or explicit checkpoints)
class LSPFeedback:
    def check_done(self, state, session_log, context, step_num):
        errors = run_typechecker()
        if errors:
            return Block(f"Type errors before done():\n{errors}")
        return None
```

Credit to Mario Zechner's [Pi](https://github.com/earendil-works/pi)
write-up for naming this anti-pattern crisply.

## 12. Aggressive compaction silently destroys prompt caching

Anthropic and OpenAI cache prompt prefixes; the cache breakpoint moves
forward as the conversation grows. If your compaction strategy
*rewrites* the prefix on every turn (e.g. by pruning all tool results
older than N tokens, or summarising older messages in place), the
cache hit rate collapses and per-turn cost can rise 5–10×.

```python
# ✗ silently cache-hostile — every turn rewrites the prefix
config = LoopConfig(
    compact_service=PruneToolResults(keep_recent_tool_results=2),
    cache_policy=CachePolicy(...),
)

# ✓ keep enough recent results to stay behind the cache breakpoint,
#    and only summarise on overflow, not every turn
config = LoopConfig(
    compact_service=DefaultCompactService(
        keep_recent=4,
        keep_recent_tool_results=10,    # ≥ what cache_policy expects to keep stable
    ),
    cache_policy=CachePolicy(...),
)
```

If you're unsure, run with `MetricsHook` for a few turns and inspect
the `usage.cache_read_input_tokens` reported by your provider. A
healthy run shows that number climbing; a cache-hostile run shows it
flat near zero.


## 13. Mid-loop PII redaction causes confident hallucinations

A natural-looking pattern is to write a hook that scrubs PII (emails,
SSNs, names) from `tool_result.data` in `post_dispatch` so the LLM
"never sees" sensitive values. This works for the trace file. **It
does not work for the LLM.** When the model receives `[EMAIL]`
instead of `j.smith@example.com`, it doesn't treat it as opaque — it
*invents* a plausible-looking replacement (`bhansen@corp.local`) and
then continues building a story around the invention. By the time
the agent writes a structured report, every downstream value can be
hallucinated.

This was found in dogfood round 15 (a SOC triage cartridge): the
agent invented an entire username, host IP, and lateral-movement
narrative, then confidently labelled it `severity: critical /
recommended_action: isolate_host` against a host that didn't exist.

```python
# ✗ Scrubs the LLM's own view → it invents replacements
class PIIRedactionHook:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        tool_result.data = scrub_pii(tool_result.data)   # don't do this
        return None
```

The right pattern is to scrub **at the boundary, not in the loop**:

* **Trace-only redaction.** Use `ProvenanceSink(redact=scrub_pii)` —
  this rewrites only what hits disk; the LLM sees the real values
  and reasons over them.
* **Stable token substitution.** If the LLM truly should never see
  the raw value, replace it with a *consistent* opaque token
  (`USER_AC42F1`, generated by hashing the original) BEFORE it ever
  enters the loop, and look the original up post-hoc when reporting.
  The model treats the token as an opaque reference instead of
  inventing a plausible-looking string.
* **Deny tools that return PII.** If a tool's output is too
  sensitive for the LLM to ever see, the right move is a permission
  rule that denies the tool entirely, not a hook that mutates its
  output.

The general principle: **never silently change what the LLM sees
mid-conversation.** The model doesn't know the substitution happened
and will reconstruct what it thinks the missing piece "should be."
