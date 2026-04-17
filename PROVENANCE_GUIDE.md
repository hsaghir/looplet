# Provenance Guide

`openharness.provenance` captures exactly what your agent saw and did —
every prompt, every response, every step — and writes them to a
diff-friendly directory you can `cat`, `grep`, and check into git.

**The loop:** capture → read → replay.

```python
# 1. Capture
sink = ProvenanceSink(dir="traces/run_1/")
llm = sink.wrap_llm(AnthropicBackend(...))
for step in composable_loop(llm=llm, tools=tools, state=state,
                            hooks=[sink.trajectory_hook()]):
    print(step.pretty())
sink.flush()

# 2. Read (from the shell)
$ python -m openharness show traces/run_1/

# 3. Replay (change hooks/tools, no LLM cost)
from openharness import replay_loop
for step in replay_loop("traces/run_1/", tools=tools):
    print(step.pretty())
```

**Two primitives, one facade, two extras:**

| Class / function | What it does | Implements |
|---|---|---|
| `RecordingLLMBackend` | Every call to `generate` / `generate_with_tools` | `LLMBackend` (wraps any backend) |
| `AsyncRecordingLLMBackend` | Same, async | `AsyncLLMBackend` |
| `TrajectoryRecorder` | Every loop step, with context-before and linked LLM calls | `LoopHook` |
| `ProvenanceSink` | Both of the above in a 3-line drop-in | — |
| `replay_loop(dir, ...)` | Rerun the loop against cached LLM output | generator |
| `python -m openharness show <dir>` | One-page readable summary | CLI |

All of it is zero-dependency, safe by default, and preserves the
`NativeToolBackend` capability surface of whatever they wrap.

---

## Quick start — `ProvenanceSink`

```python
from openharness import ProvenanceSink, composable_loop, DefaultState
from openharness.backends import AnthropicBackend

sink = ProvenanceSink(dir="traces/run_1/")

llm = sink.wrap_llm(AnthropicBackend(api_key=...))
hooks = [sink.trajectory_hook()]
state = DefaultState(max_steps=15)

for step in composable_loop(llm=llm, tools=tools, state=state, hooks=hooks):
    print(step.pretty())

sink.flush()
```

After `flush()`, the directory contains:

```
traces/run_1/
├── trajectory.json         # run_id, steps, termination_reason, metadata
├── steps/
│   ├── step_01.json        # tool_call, tool_result, context_before, linked LLM indices
│   └── step_02.json
├── call_00_prompt.txt      # exact prompt sent (system + user + tools)
├── call_00_response.txt    # raw response
├── call_01_prompt.txt
├── call_01_response.txt
└── manifest.jsonl          # one LLMCall summary per line
```

Every file is human-readable. `trajectory.json` and `manifest.jsonl` are
machine-parseable. You can diff, grep, version-control, attach to bug
reports, or feed to a separate analysis pipeline.

---

## LLM-call provenance only

If you don't need the full trajectory (e.g. you're debugging prompt
rendering in isolation), just wrap the backend:

```python
from openharness.provenance import RecordingLLMBackend

llm = RecordingLLMBackend(MyBackend())

# Use `llm` anywhere a backend is expected — `composable_loop`, a sub-agent,
# a one-off `llm.generate(...)` call. Every invocation is captured.

response = llm.generate("hello", system_prompt="be brief", max_tokens=50)

# Inspect in memory:
assert len(llm.calls) == 1
c = llm.calls[0]
print(c.prompt, c.response, c.duration_ms, c.error)

# Or dump to disk:
llm.save("traces/debug_run/")
```

### The `LLMCall` dataclass

Each entry in `llm.calls` exposes:

| Field | Type | Notes |
|---|---|---|
| `index` | `int` | 0-based, monotonic |
| `timestamp` | `float` | Unix time when the call started |
| `duration_ms` | `float` | Wall-clock around the wrapped call |
| `method` | `str` | `"generate"` or `"generate_with_tools"` |
| `prompt` | `str` | Exact text passed in (after redaction/truncation) |
| `system_prompt` | `str` | Same |
| `response` | `str \| list[dict]` | String for `generate`; content blocks for `generate_with_tools` |
| `tools` | `list[dict] \| None` | Tool schemas passed, when applicable |
| `temperature` | `float` | As passed |
| `max_tokens` | `int` | As passed |
| `step_num` | `int \| None` | Set by `TrajectoryRecorder` to link calls to loop steps |
| `error` | `str \| None` | `f"{type(e).__name__}: {e}"` if the wrapped backend raised |

### Safety knobs

```python
llm = RecordingLLMBackend(
    backend,
    max_chars_per_call=200_000,     # truncate huge prompts with an elision marker
    redact=lambda s: SECRET_RE.sub("[REDACTED]", s),  # scrub before storage
)
```

### `generate_with_tools` is preserved

```python
rec = RecordingLLMBackend(native_backend)
assert hasattr(rec, "generate_with_tools")   # surfaced only when wrapped supports it

blocks = rec.generate_with_tools("...", tools=[...])
assert rec.calls[-1].method == "generate_with_tools"
assert rec.calls[-1].tools == [...]
```

---

## Trajectory provenance only

Install `TrajectoryRecorder` as a hook. It captures a structured
`Trajectory` for the whole run using the standard hook surface —
`pre_loop`, `pre_prompt`, `post_dispatch`, `on_loop_end`. No changes to
the loop are needed.

```python
from openharness.provenance import TrajectoryRecorder

hook = TrajectoryRecorder()
for step in composable_loop(llm=llm, tools=tools, state=state, hooks=[hook]):
    ...
hook.save("traces/run_1/")     # writes trajectory.json + steps/*.json
```

### Linking steps to LLM calls

Pair `TrajectoryRecorder` with `RecordingLLMBackend` and every
`StepRecord.llm_call_indices` will point into
`Trajectory.llm_calls` — so you always know which prompt produced a
given tool call:

```python
rec_llm = RecordingLLMBackend(MyBackend())
hook = TrajectoryRecorder(recording_llm=rec_llm)

for step in composable_loop(llm=rec_llm, tools=tools, state=state, hooks=[hook]):
    ...

# Which prompts produced step 3?
indices = hook.trajectory.steps[2].llm_call_indices
for i in indices:
    print(hook.trajectory.llm_calls[i].prompt)
```

### What a `StepRecord` captures

| Field | Notes |
|---|---|
| `step_num` | 1-based loop step index |
| `timestamp`, `duration_ms` | When the step ran and how long |
| `pretty` | `Step.pretty()` — `#N ✓ tool(args) → result [Xms]` |
| `tool_call` | `ToolCall.to_dict()` |
| `tool_result` | `ToolResult.to_dict()` (truncated safely) |
| `context_before` | The briefing shown to the LLM before this step's prompt |
| `llm_call_indices` | Into `trajectory.llm_calls` (empty if no recording backend) |

### Termination inference

`on_loop_end` inspects the last step and sets `termination_reason`:

- `"done"` — the agent called the `done` tool successfully
- `"error"` — the last step returned an error
- `"max_steps_or_stop"` — budget exhausted or a `should_stop` hook fired
- `"no_steps"` — the loop produced nothing

The loop's `done` path bypasses `post_dispatch`, so `TrajectoryRecorder`
sweeps `state.steps` at `on_loop_end` to catch that final step.

### Embedded `Tracer`

`TrajectoryRecorder` embeds a `Tracer` by default — no extra hook
needed. Pass your own if you want to export spans elsewhere:

```python
from openharness import Tracer
my_tracer = Tracer()
hook = TrajectoryRecorder(tracer=my_tracer)
...
# Export my_tracer.root_spans to OTel, Datadog, etc.
```

---

## Replay a captured run

```python
from openharness import replay_loop

for step in replay_loop("traces/run_1/", tools=my_tools):
    print(step.pretty())
```

`replay_loop` reads `manifest.jsonl` + `call_NN_response.txt` from the
trace directory and feeds them back into `composable_loop` in order.
The LLM is **not** called again — your tools, hooks, permission
engine, and state are fresh. This is the whole point: change any of
those, diff the step output, and you get a cost-free A/B.

**Common uses:**

- **Change a hook, re-run.** Add a new permission rule or logging
  hook, replay, see how step outputs differ.
- **Upgrade openharness and diff the loop.** If the replay produces
  different steps with the same LLM output, the loop behavior changed.
- **Golden tests.** Capture once, check the directory into git,
  `replay_loop` in CI.

**Constraints:**

- If your loop now asks for **more** LLM calls than were recorded,
  replay raises `RuntimeError` — reduce `max_steps` or re-record.
- If a call was recorded as `generate_with_tools` but the replay loop
  uses `generate` (or vice versa), replay raises `RuntimeError` — the
  divergence is almost certainly a bug you want to see.
- Tools **do** execute at replay time. Replace with mocks if you don't
  want side effects.
- If `manifest.jsonl` is missing, replay falls back to the
  `call_NN_response.txt` files alone (every call treated as
  `generate`).

---

## Inspect a trace from the CLI

```
$ python -m openharness show traces/run_1/
154a1edb893c  ✓ done  4 steps  4 LLM calls  0ms

#1  ✓ add(a=1, b=2)       → 1 keys    [   12ms] call 0
#2  ✓ add(a=3, b=4)       → 1 keys    [   10ms] call 1
#3  ✓ done(answer=ok)     → 1 keys    [    8ms]

LLM: 4 calls, 1,582 in / 230 out chars, 0 errors
```

Zero dependencies, stdlib-only. Useful when you just want a one-glance
answer to "did this run do what I expected?" without loading an IDE or
a notebook.

Exit codes: `0` success, `1` missing / malformed directory.

---

## Recipes

### 1. Golden-trajectory regression tests

```python
# Capture the golden run once:
sink = ProvenanceSink(dir="tests/goldens/login_flow/")
llm = sink.wrap_llm(backend)
for step in composable_loop(llm=llm, tools=tools, state=state, hooks=[sink.trajectory_hook()]):
    ...
sink.flush()

# In CI, replay against the stored manifest:
expected = json.loads((Path("tests/goldens/login_flow") / "trajectory.json").read_text())
assert actual["step_count"] == expected["step_count"]
assert [s["tool_call"]["tool"] for s in actual["steps"]] == \
       [s["tool_call"]["tool"] for s in expected["steps"]]
```

### 2. Cost accounting

```python
from collections import Counter

llm = RecordingLLMBackend(AnthropicBackend(...))
# ... run the loop ...

total_prompt_chars = sum(len(c.prompt) for c in llm.calls)
total_response_chars = sum(
    len(c.response) if isinstance(c.response, str) else len(json.dumps(c.response))
    for c in llm.calls
)
by_method = Counter(c.method for c in llm.calls)
print(f"{len(llm.calls)} calls, {total_prompt_chars:,} in / {total_response_chars:,} out, {by_method}")
```

### 3. Secret scrubbing

```python
import re
SECRET_RE = re.compile(r"(api_key|token|password)\s*[:=]\s*\S+", re.I)

sink = ProvenanceSink(
    dir="traces/run_1/",
    redact=lambda s: SECRET_RE.sub(r"\1=[REDACTED]", s),
)
```

### 4. Per-scenario benchmarks

```python
for scenario in scenarios:
    sink = ProvenanceSink(dir=f"traces/{scenario.name}/")
    llm = sink.wrap_llm(backend)
    for step in composable_loop(llm=llm, tools=tools, state=scenario.fresh_state(),
                                hooks=[sink.trajectory_hook()]):
        ...
    sink.flush()
# Now you have traces/<name>/trajectory.json for every scenario —
# diff across runs, feed to a notebook, attach to PR descriptions.
```

### 5. Bug-report bundles

```python
# On failure, flush whatever was captured so far:
try:
    for step in composable_loop(llm=llm, tools=tools, state=state, hooks=[sink.trajectory_hook()]):
        ...
except Exception:
    sink.flush()   # partial trace written; safe to attach to the issue
    raise
```

---

## Performance

- Capture overhead is one dict append + one `str()` per LLM call; per
  step: one `StepRecord` append.
- Memory is bounded by `max_chars_per_call` (default 200,000 chars) on
  each `prompt`/`response`/`system_prompt`. Tool schemas and response
  blocks are stored as-is — if you pass giant tool schemas, consider
  redacting or truncating before the call.
- `save()` is the only disk I/O. Call it once at loop end (or inside
  an `except` handler) — not inside the loop.
- Both sync and async variants are correct under
  `asyncio.gather(...)` concurrency at the call site, but note that
  `TrajectoryRecorder` is designed for a single loop at a time; use
  one sink per concurrent run.

---

## When not to use

- **Production hot paths** with throughput > ~100 LLM calls/sec:
  `RecordingLLMBackend` is cheap but not free. Either use a sampling
  wrapper in front of it or use OTel via `TracingHook` which is designed
  for that.
- **PII-sensitive deployments** without a `redact=` callable. The
  recorder writes prompts verbatim — if your prompts contain PII and
  you can't scrub, pick a different tool.

For everything else — development, CI, benchmarks, debugging, golden
tests — this is what you want.
