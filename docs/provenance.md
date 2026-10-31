# Capture and replay: evidence for every harness change

`looplet.provenance` captures exactly what your agent saw and did: every
prompt, response, and step. It writes them to a
diff-friendly directory for private inspection with tools such as `cat` and `grep`.
Keep raw captures out of version control; share only reviewed exports.

**The loop:** capture → inspect → turn the failure into a contract → replay
captured responses against a changed harness.

Replay here always means **captured-response replay**. The model is not called
again, but tools, hooks, state, permissions, clocks, networks, randomness, and
other side effects are fresh. This isolates model-output variation; it is not
bit-for-bit deterministic simulation.

```python
# 1. Capture
sink = ProvenanceSink()
llm = sink.wrap_llm(AnthropicBackend(...))
for step in composable_loop(llm=llm, tools=tools, state=state,
                            hooks=[sink.trajectory_hook()]):
    print(step.pretty())
trace_dir = sink.flush()
print(trace_dir)

# 2. Read (from the shell)
$ python -m looplet show <printed-trace-directory>

# 3. Replay (change hooks/tools, no LLM cost)
from looplet import replay_loop
for step in replay_loop(trace_dir, tools=tools):
    print(step.pretty())
```

**Source preview:** Default replay restores recorded primary and alternate terminal tool names.
An explicit `config` overrides those defaults.

**Two primitives, one facade, two extras:**

| Class / function               | What it does                                              | Implements                       |
| ------------------------------ | --------------------------------------------------------- | -------------------------------- |
| `RecordingLLMBackend`          | Every call to `generate` / `generate_with_tools`          | `LLMBackend` (wraps any backend) |
| `AsyncRecordingLLMBackend`     | Same, async                                               | `AsyncLLMBackend`                |
| `TrajectoryRecorder`           | Every loop step, with context-before and linked LLM calls | `LoopHook`                       |
| `ProvenanceSink`               | Both of the above in a 3-line drop-in                     | Not a protocol                   |
| `replay_loop(dir, ...)`        | Rerun the loop against cached LLM output                  | generator                        |
| `async_replay_loop(dir, ...)`  | Replay with awaited async tools and hooks                 | async generator                  |
| `python -m looplet show <dir>` | One-page readable summary                                 | CLI                              |

All of it is zero-dependency and preserves the `NativeToolBackend` capability
surface of whatever it wraps. Prompts and responses may contain sensitive
data, so configure `redact=` before capture when the run crosses a privacy or
security boundary.

**Choose the redaction scope before capture.** `ProvenanceSink(redact=...)` scrubs prompt strings before the provider receives them and in saved evidence.
Adding this wrapper can therefore change the model's decisions.
Use `redact_upstream=False` only when the provider is authorized to receive the original input and only saved evidence should be scrubbed.
This option does not protect data sent to the provider.

```python title="capture_redaction.py"
from looplet import ProvenanceSink


def scrub_account(text):
  return text.replace("acct-example-42", "[REDACTED]")


provider_and_record = ProvenanceSink(redact=scrub_account)
record_only = ProvenanceSink(redact=scrub_account, redact_upstream=False)
```

Wrap the backend with the selected sink before executing the run.
The same scope applies to sync and async backends.
Prompt strings and recorded response copies are scrubbed; live responses and native tool schemas are not rewritten.
Review exports separately; a redaction callback is not proof that every sensitive value was removed.

**Version scope:** Native-response redaction and the Python replay defaults below require `0.4.1` or newer.
Published `0.4.0` can save unredacted native response blocks even when `redact=` is configured.
See [Safety fixes in `0.4.1`](install.md#unreleased-safety-fixes) before relying on these behaviors.
`async_replay_loop()` also requires `0.4.1` or newer.

---

## Quick start: `ProvenanceSink`

```python
from looplet import ProvenanceSink, composable_loop, DefaultState
from looplet.backends import AnthropicBackend

sink = ProvenanceSink()

llm = sink.wrap_llm(AnthropicBackend(api_key=...))
hooks = [sink.trajectory_hook()]
state = DefaultState(max_steps=15)

for step in composable_loop(llm=llm, tools=tools, state=state, hooks=hooks):
    print(step.pretty())

trace_dir = sink.flush()
print(trace_dir)
```

You may call `trajectory_hook()` before or after `wrap_llm()`; both setup
orders capture the same model-call evidence for sync and async loops. Build
the recording wrapper before executing the run. Replacing it during a run
does not merge separate call streams. Prefer a fresh sink for each run, or
call `reset()` before reusing the sink and obtain its new trajectory hook.

After `flush()`, the directory contains:

```
.looplet/traces/<unique-id>/
├── trajectory.json         # run_id, steps, stop reason, session log, metadata
├── steps/
│   ├── step_01.json        # tool_call, tool_result, context_before, linked LLM indices
│   └── step_02.json
├── call_00_prompt.txt      # bounded human-readable request projection
├── call_00_response.txt    # bounded human-readable response projection
├── call_01_prompt.txt
├── call_01_response.txt
└── manifest.jsonl          # one LLMCall summary per line
```

Every file is human-readable. `trajectory.json` and `manifest.jsonl` are
machine-parseable. You can diff, grep, or feed them to a private analysis pipeline.
Review and redact evidence before attaching it to bug reports or uploading it elsewhere.

Omitting `dir` gives each sink a unique directory under `.looplet/traces/`.
`flush()` returns its path; `reset()` selects a new automatic path.
An explicit directory refuses existing evidence unless you pass `overwrite=True`.
Repeated flushes of the same run remain supported.
Low-level recorder `save()` methods still replace their owned call and step files.
They preserve files that do not match those artifact names.
Do not use one explicit trace directory for concurrent writers.

A trace alone does not preserve grader expectations or collected artifacts.
Use `save_eval_run()` for a grading record that you can reload and reassess.
See the [Quickstart](quickstart.md#4-capture-the-same-agent) for the complete handoff.

### Full tool results and model context

**Unreleased:** Full evidence and model-visible context have separate representations.

| Surface                                    | Content                                                            | Persistence                              |
| ------------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------- |
| `state.steps` and yielded `Step`           | Complete dispatched `ToolCall` and `ToolResult` objects            | In memory                                |
| Registry result store                      | Detached original JSON-compatible output, keyed by `sha256:...`    | In memory by default                     |
| Checkpoint `tool_results_store`            | Snapshot of that result store                                      | Written when checkpointing is configured |
| Conversation and prompt                    | Selected, projected history, with previews and result references   | Recorded with model calls                |
| `trajectory.json` and `steps/step_NN.json` | Complete tool arguments, data, IDs, warnings, errors, and metadata | Written by sink `flush()`                |

`ToolResult.to_dict()` remains a compact view, not a lossless audit serializer.
It limits direct lists to 20 items and other direct values to 2,000 characters.
Already projected model results are not clipped a second time by this serializer.
`ToolResult.to_full_dict()` preserves complete result data for evidence capture.
`Step.to_full_dict()` exposes complete calls and results to custom host trace writers.
The recorder snapshots full results before later state changes can alter them.
Explicit redaction still applies to saved evidence.

Prompt projection and context compaction do not delete the registry's retained original.
`registry.get_result(result_key)` returns a detached copy for host code.
Checkpoint restoration restores this store for later retrieval.
`BaseToolRegistry(retain_results=False)` disables registry retention, not trace capture or raw `Step` objects.

Optional `LoopConfig(tool_result_persist_dir=...)` also writes large serialized outputs to `tool-output-<hash>.txt`.
Its preview includes the path, UTF-8 byte size, and character count.
This option is unset by default.
A saved path alone does not grant filesystem access or provide a model tool.

#### Bounded Model Views

Native and JSON-text calls share one result projection. The character budget
counts the serialized envelope, including nested data, the result reference,
warnings, errors, and JSON escaping. Original tool outputs are not changed.

`LoopConfig.context_inline_per_step_chars` defaults to 3,000 characters.
Source previews also respect `LOOPLET_TOOL_RESULT_MAX_CHARS` and the existing
row limit. An explicit `ToolSpec.render["max_chars"]` overrides the per-result
default; `render["preview"]` selects the source row limit. Cartridge exports
put these host policies in `runtime.yaml: tool_render_hints`, not `tool.yaml`.

Native result batches share `context_window_total_chars` (20,000 by default).
The largest previews shrink first, and admitted previews remain stable on later
turns. Text history uses the same aggregate limit. These are character budgets,
not exact token counts or a replacement for whole-conversation compaction.

Retrieval tools return views of existing evidence. A view is delivered intact
or replaced by a delivery refusal, without page content or a continuation cursor.
`delivery.status` is `"refused"`; `delivery.reason` identifies the boundary, such as `result_view_too_large`.
`error` still describes tool execution. A successful read does not become an execution failure because its page cannot fit.
Retry the same offset with a smaller `limit`. When available, `delivery.suggested_limit` accounts for the complete serialized page envelope.
The hint does not advance the offset or deliver hidden content. An 8,000-character read
is an allowed request, not a promise that every configured envelope budget can
deliver it. Exceptionally small budgets that cannot fit a control envelope
raise a context overflow instead of silently exceeding the limit.

Failed tool results keep their execution error and source reference.
If the error alone is too large, `delivery.error_truncated` marks its shortened model-facing text.
Complete errors and warnings remain in raw evidence. Conversation serialization preserves selected warnings, references, and delivery status.
Text continuation and native messages use the same result envelope.
Backends can observe original results through `record_tool_receipt(call, result)` before projection.
This explicit capability does not invoke a host's legacy `record_tool_result(call_id, data, error)` callback.

Custom read/search tools can opt into this contract with
`@tool(result_is_view=True)` or `ToolSpec(result_is_view=True, ...)`.
Cartridges use `result_is_view: true` in `tool.yaml`. Views still appear in raw
steps and traces, but are not retained as new source results. The flag is not
a privacy control: the tool owner must ensure it really returns a view and
that the model has authority to inspect its underlying evidence.

#### Model-accessible retrieval

The host can grant retained-evidence access without granting filesystem or shell access:

```python
from looplet import tools_from

tools = tools_from(my_tools, include_done=True, include_result_tools=True)
```

Cartridges select the same built-ins:

```yaml
builtin_tools:
  - list_tool_results
  - read_tool_result
  - search_tool_result
```

These are ordinary native tools and also support JSON-text calls.
They use the current registry, not a global store or arbitrary file paths.
Hosts can explicitly provide a narrower `tool_result_store` resource.
Enabling retrieval grants the model access to that store's full retained values.
Do not enable it when the model may see only redacted previews.

- `list_tool_results(offset=0, limit=10, end_offset=None)` discovers references after compaction. The maximum limit is 20.
- `read_tool_result(result_key, json_pointer="", offset=0, limit=2000)` reads a page. The maximum page is 8,000 characters.
- `search_tool_result(result_key, query, json_pointer="", offset=0, limit=5)` performs literal search. The maximum is 10 matches.

`json_pointer` follows RFC 6901. For example, `/stdout` selects a command's stdout.
Strings are read directly. Other selected values use JSON text.
Offsets count Unicode characters, not UTF-8 bytes.
Responses include `next_offset`; `null` means no further page or match.
Search returns bounded excerpts with offsets suitable for a later read.
Retrieval neither invokes an LLM nor chooses an analytical method.

Listing returns an `end_offset`. Carry the first page's boundary into subsequent
calls to finish that catalog even if new source results arrive during paging.
Listing, reading, and searching do not add their own replies to the catalog.
A later listing without the boundary can discover newly retained sources.

#### File-Backed Retention

`FileToolResultStore` implements the same result-store resource with opaque
content-addressed keys. Standard execution profiles bind one automatically under
the run directory. Attach an explicit store to the preset to override that default
for both Python and cartridge runs:

```python
from looplet import FileToolResultStore

preset.resources["tool_result_store"] = FileToolResultStore(
  "./private-tool-results", max_bytes=100_000_000
)
```

Preset execution binds its resources to the registry before inference.
Preset resources take precedence over duplicate registry bindings; registry-only resources remain available.
Enabled retrieval tools require the store's corresponding list, read, or search method to be callable.
An invalid binding fails before inference. Python and cartridge execution use the same check.

Direct registry users can instead call
`tools.set_resources({"tool_result_store": store})` before running their loop.
A writable resource receives new source results; a read-only resource can
continue exposing a narrower host-selected collection without receiving writes.
Do not share a directory between independent writers. Share one store instance
for concurrent calls within a process.

Source files and selected serialized views are reused across pages and store
reopens. `max_bytes` bounds encoded source and cache content, excluding the small
catalog. It is not a filesystem-wide disk quota. Cache writes are optional;
when they do not fit, reads can still use the retained source. A source retention
I/O or quota failure leaves the successful raw tool output intact, omits its
result key, and adds a visible warning plus `metadata.retention_error`.

Cached views use incremental reads and literal searches, including across chunk boundaries.
Unicode offsets and line endings remain unchanged. Late offsets and uncached length checks still require linear scans.
The first extraction of a selected JSON value can still materialize the complete source.

The host owns directory selection, permissions, cleanup, and lifetime. Keys
are references, not secrets or access-control tokens. Enabling retrieval grants
access to all results in that resource; use a separate store for each authority
boundary. Files are atomically replaced, but writes are not fsynced and there is
no write-ahead journal or cross-process writer coordination.

The default harness also spools raw steps and provenance to disk and streams
provenance export. Automatic checkpoints reference an immutable, verified catalog;
they require the original result directory to remain available. Restoring to a fresh
store verifies and copies one source value at a time, preserving the original store.
Explicit `snapshot_results()` still exports complete values for a portable snapshot
and may allocate memory proportional to all results.

Lazy history and automatic checkpointing avoid retaining every output in RAM.
This is not a total-process memory limit: one active result, a concurrent batch,
selected JSON extraction, custom host state, or an explicit full export can allocate
more. A caller that collects every yielded step also retains those objects.
See [default persistence and memory](operations.md#default-persistence-and-memory)
for replacement rules, opt-outs, file lifetime, and the bounded return summary.

#### Inspecting the model's view

Inspect each model call's structured `capture.request`, not a raw `Step`, to see its selected input.
It records the prompt, system prompt, schemas, options, and selected `conversation_input` when available.
This includes native tool-result previews and their original call IDs.
`capture.provider_request` contains backend-provided request metadata when the backend exposes it.
It is not a universal HTTP wire capture.
The human-readable call files and legacy summary fields can be bounded independently.

Call `sink.flush()` in `finally` to save evidence when a run raises or is canceled.
Recording is not a write-ahead journal; an abrupt process or host failure can lose unflushed records.
Use checkpoints for persisted result retention between runs.
Capture covers values tools return, not output they discarded before returning.
Use a separate sink for each child loop and retain its parent/run linkage.

### Structured model-call capture

**Unreleased:** New captures include a `capture` object on each model-call record in
`manifest.jsonl` and `trajectory.json`. Its schema is `looplet.model-call.v1`:

```json
{
  "schema": "looplet.model-call.v1",
  "request": {
    "method": "generate_with_tools",
    "prompt": "Finish the task.",
    "system_prompt": "Use the supplied tools.",
    "temperature": 0.2,
    "max_tokens": 4096,
    "tools": [],
    "options": {}
  },
  "response": [{"type": "text", "text": "Observed assistant output"}],
  "provider_request": null,
  "provider_response": null,
  "stop_reason": "stop",
  "usage": {},
  "error": null
}
```

The request, normalized response, and available provider metadata are independent JSON snapshots.
Built-in non-streaming adapters expose their full request and response objects. Custom backends
can expose `last_request`, `last_response`, `last_stop_reason`, and `last_usage`; absent metadata
remains unavailable. Failed calls retain their error without borrowing a previous response or usage.
Each recording-wrapper invocation has its own record, including native recovery attempts.
SDK-internal retries and streaming chunks are not captured by this interface.

Returned reasoning, thinking, signature, redacted-thinking, and provider extension fields are retained
when supplied by the adapter. These are observed provider data, not proof of private or causal reasoning.
Unreturned reasoning cannot be reconstructed. Opaque signed fields are evidence, not a guarantee
that the saved record can resume a live provider session.

`max_chars_per_call` bounds the readable projections, not the canonical `capture` object.
Configured redaction applies recursively before retaining the structured values.
Full snapshots can be large and contain sensitive data, including provider fields absent from the
readable projection. Review the entire structured capture before sharing it; a bounded text file
is not a privacy or storage limit.

Step records link to every model call that contributed to their turn, including discarded truncated
responses. A native batch's tool steps can share one model-call index. Tool arguments and results
are snapshotted when observed, and assistant-message completion retains its origin marker.
Trajectory metadata includes native per-call accounting and the scalar completion/recovery settings.
The cost hook counts each native request and its reported usage; missing usage is not invented.

### Assistant-turn capture and replay

**Unreleased:** Bounded assistant-turn runs use `looplet.model-call.v2` captures.
Their execution metadata declares `looplet.assistant-turns.v1`, turn identity, request identity, and whether a host policy is required.
Strict legacy runs retain `looplet.model-call.v1`.
Readers reject unknown required semantics instead of silently applying legacy completion rules.

Canonical capture remains separate from working history and bounded progress events.
Recursive record redaction still applies; configured upstream redaction also covers typed context and compaction reset inputs.
Upstream redaction that changes opaque signed blocks is rejected before inference.
Review full captures before sharing them. Opaque provider data can contain sensitive material.
Returned reasoning is provider evidence, not proof of private reasoning or answer quality.

Rebind the host policy explicitly during Python replay:

```python
from contextlib import closing

from looplet import replay_loop


with closing(replay_loop(trace_dir, tools=tools, assistant_turn_policy=decide)) as steps:
  for step in steps:
    print(step.pretty())
```

The host supplies `trace_dir`, `tools`, and the reviewed `decide` callable.
Replay checks recorded decisions in order, including the action and a SHA-256 feedback digest.
Missing or mismatched bindings fail closed. Callable code is never reconstructed from a capture.
To intentionally change policy, also provide a nonempty `policy_override_name` of at most 128 characters.
The fresh trajectory records that override name. A `stop` decision still does not become success.
The same contract applies to `async_replay_loop`, including async host policies.
CLI replay cannot reconstruct a required callable; use the Python API to supply it.

Fresh tools and effects still execute during replay. Captured responses do not restore a live provider session.
Bounded runs disable supported SDK retries and admit cooperative wrapper attempts separately.
Default strict captures still cannot identify hidden SDK-internal attempts or streaming chunks.

### Save a fresh replay record

**Unreleased:** `recording_sink` and the direct `max_steps` replay option require the
[pinned preview](install.md#try-unreleased-factory-commands).
The [one-file starter](quickstart.md#make-the-complete-example-your-own) owns this workflow for routine experiments.

Pass a fresh sink to the existing replay function when your host owns the tool registry:

```python
from contextlib import closing

from looplet import ProvenanceSink, replay_loop

replay_sink = ProvenanceSink()
try:
  with closing(replay_loop(
    trace_dir,
    tools=tools,
    recording_sink=replay_sink,
    max_steps=3,
  )) as steps:
    for step in steps:
      print(step.pretty())
finally:
  print(replay_sink.flush())
```

Your host supplies `trace_dir` and `tools` from its captured run and current agent.
The new record contains fresh replay prompts, captured responses, and observed tool steps.
The original capture is unchanged.
The same options work with `async_replay_loop()` and `contextlib.aclosing`.
An explicit limit overrides the recorded budget without mutating a supplied `LoopConfig`.
Omit it to use the recorded budget and protocol.
Flush the sink after closing the iterator, including interrupted runs.
This records replay evidence; attach an `EvalHook` and use `save_eval_run()` to persist grades too.
Tools and other effects execute again; a new record is not a sandbox or checkpoint restoration.

The one-file starter saves an `EvalCase` with each capture.
It keeps the task, case identity, and grader-only expectations together through Python and CLI replay.
CLI replay restores the captured native-tool or JSON-text protocol instead of the factory default.
Older eval captures with saved expectations can be replayed without first adding a case file or recapturing model calls.
The new CLI replay record preserves those expectations for offline grading.
Raw traces without expectations remain ungraded.
The CLI `--evals` option supplies collectors; run the printed grading command separately.

### Share a reviewed summary

New trace and eval-record directories contain a wildcard `.gitignore` before evidence is written.
This protects new captures even when the surrounding project has no ignore policy.
Looplet preserves an existing directory policy and does not edit your project's ignore file.
Git ignore rules do not untrack existing files or prevent `git add -f`.
They are not redaction or access controls.

Export only the fields that a reviewer needs, outside the raw capture directory:

```python
import json
from pathlib import Path

from looplet import load_eval_run

record = load_eval_run("eval-runs/run-001/hello")
summary = {
  "completed": record.context.completed,
  "step_count": record.context.step_count,
}
with Path("reviewed-summary.json").open("x", encoding="utf-8") as output:
  json.dump(summary, output, indent=2)
  output.write("\n")
```

Review that file before committing or uploading it.
This summary omits prompts, responses, task data, expectations, and artifacts.
It is not replay evidence, a correctness verdict, or automatic permission to disclose run information.
Configure `redact=` before capture when a fuller export is needed, and review that export separately.

### Create a reviewed replay fixture

Keep incident captures private.
For a shareable golden test, construct a separate fixture with synthetic, reviewed inputs and responses.
Do not force-add the raw capture or replace its ignore policy.
A summary export cannot substitute for the model-call files that replay requires.

Run this example from your consumer project with a new fixture path:

```python title="reviewed_replay_fixture.py"
import json
from pathlib import Path

from looplet import ProvenanceSink
from looplet.testing import MockLLMBackend


fixture = Path("tests/fixtures/reviewed-done")
fixture.mkdir(parents=True, exist_ok=False)
allowed = [
  ".gitignore",
  "artifact.json",
  "manifest.jsonl",
  "call_00_prompt.txt",
  "call_00_response.txt",
]
(fixture / ".gitignore").write_text(
  "*\n" + "".join(f"!/{name}\n" for name in allowed),
  encoding="utf-8",
)
response = json.dumps({
  "tool": "done",
  "args": {"summary": "Synthetic fixture complete."},
})
sink = ProvenanceSink(dir=fixture)
recorder = sink.wrap_llm(MockLLMBackend([response], cycle=False))
recorder.generate("Complete this synthetic test.", system_prompt="Synthetic data only.")
sink.flush()
```

This makes no provider or tool calls.
The pre-existing allowlist is preserved; unlisted files remain ignored.
The filenames describe this one-response fixture, not a frozen artifact-format guarantee.
Review every allowlisted file before staging it:

```bash
git add -- tests/fixtures/reviewed-done
git diff --cached --name-only
git diff --cached -- tests/fixtures/reviewed-done
```

Use this directory with `replay_loop()` or `async_replay_loop()` and your trusted fresh tools.
Git ignore rules are not access controls or automatic approval to disclose the files.

### Replay from the shell

Use `looplet replay` with an explicit trusted cartridge and a new workspace.
The [CLI recipe](cli.md#replay-captured-responses) captures, inspects, and replays one agent without a model service.
Saved eval cases supply their agent-visible task and seed files; raw traces reuse saved trajectory tasks when available.
Raw traces do not snapshot starting files. Use `--seed-from ./starting-inputs` to copy a reviewed input directory into the new workspace without modifying the source.
The workspace must be outside that directory. Git and Looplet metadata and the selected trace and cartridge trees are excluded; included symlinks and special files are rejected.
Saved cases reject `--seed-from` because they already define their seed files.
Pass `--task` when no task was saved, or to replace the saved task with a new goal.
The default replay budget uses recorded tool steps, with captured response count as a fallback.
Use `--max-steps` to override that budget.
The command records replay evidence separately and leaves the original capture unchanged.
Saved-case replay reruns the cartridge's collectors and saves fresh artifacts with the original grader-only expectations and case.
Human output prints a grading command; JSON output returns the new record's `trace_dir` for a separate `looplet eval` call.
New trajectories link to the source through `parent_run_id` and, for saved cases, `source_case_id` metadata.
Graders do not run automatically, and old collected artifacts are not reused as new observations.
Fresh tool and hook effects still execute; it is not a sandbox or checkpoint restoration.
Shell replay accepts cartridges only; use the [Python bundle recipe](cli.md#replay-skill-bundles) for skill bundles.

`looplet show` and `replay_loop()` fail before rendering or execution when the
evidence structures they consume are malformed. Replay also requires
contiguous manifest indices and the corresponding response file for every
manifest record. These checks protect one run from mixed or corrupt evidence;
they do not make the pre-1.0 artifact format a frozen schema.

### Link related traces

Seed trajectory metadata when a host wants to record evidence lineage:

```python
parent_hook = parent_sink.trajectory_hook()
child_sink = ProvenanceSink(
    dir="traces/child/",
    metadata={"parent_run_id": parent_hook.trajectory.run_id},
)
```

This is an annotation only. It does not add scheduling, traversal, replay, or
state inheritance semantics to the loop.

---

## LLM-call provenance only

If you don't need the full trajectory (e.g. you're debugging prompt
rendering in isolation), just wrap the backend:

```python
from looplet.provenance import RecordingLLMBackend

llm = RecordingLLMBackend(MyBackend())

# Use `llm` anywhere a backend is expected - `composable_loop`, a sub-agent,
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

| Field           | Type                 | Notes                                                           |
| --------------- | -------------------- | --------------------------------------------------------------- |
| `index`         | `int`                | 0-based, monotonic                                              |
| `timestamp`     | `float`              | Unix time when the call started                                 |
| `duration_ms`   | `float`              | Wall-clock around the wrapped call                              |
| `method`        | `str`                | `"generate"` or `"generate_with_tools"`                         |
| `prompt`        | `str`                | Exact text passed in (after redaction/truncation)               |
| `system_prompt` | `str`                | Same                                                            |
| `response`      | `str \| list[dict]`  | String for `generate`; content blocks for `generate_with_tools` |
| `tools`         | `list[dict] \| None` | Tool schemas passed, when applicable                            |
| `temperature`   | `float`              | As passed                                                       |
| `max_tokens`    | `int`                | As passed                                                       |
| `step_num`      | `int \| None`        | Set by `TrajectoryRecorder` to link calls to loop steps         |
| `error`         | `str \| None`        | `f"{type(e).__name__}: {e}"` if the wrapped backend raised      |
| `metadata`      | `dict[str, Any]`     | Free-form annotations for external hooks; defaults to `{}`      |

### Safety knobs

```python
llm = RecordingLLMBackend(
    backend,
    max_chars_per_call=200_000,     # truncate huge prompts with an elision marker
    redact=lambda s: SECRET_RE.sub("[REDACTED]", s),  # scrub PII/secrets
)
```

**By default, `redact` scrubs prompt and `system_prompt` strings before
they reach the wrapped backend.** The same callable scrubs recorded text responses.
**Since `0.4.1`:** Native response string values and keys are scrubbed in a separate recorded copy.
Published `0.4.0` does not scrub native response blocks.
Live responses and tool arguments remain unchanged.
Replay uses the saved, redacted values; it does not recover the original values.
Review captured evidence before sharing or replaying it.

**Since `0.4.1`:** Trajectory and step captures scrub string keys and values before JSON encoding.
JSON-sensitive replacements remain valid JSON.
Recorded tool schemas, backend error text, and manifest metadata also use the callback.
Live schemas and raised exceptions remain unchanged.
Upstream scrubbing applies to prompt strings, not native tool schemas.

`ProvenanceSink(redact=...)` accepts the same callable and inherits the
same semantics:

```python
sink = ProvenanceSink(dir="traces/", redact=redact_pii)
llm  = sink.wrap_llm(AnthropicBackend(...))
# Prompt strings are scrubbed upstream. Captures use a separate scrubbed copy.
```

If you want the legacy record-only behaviour (scrub the trace but
forward the raw prompt to the provider), which is useful when the provider
itself needs the PII to respond correctly), opt out:

```python
sink = ProvenanceSink(dir="traces/", redact=redact_pii, redact_upstream=False)
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
`Trajectory` for the whole run using the standard hook surface:
`pre_loop`, `pre_prompt`, `post_dispatch`, `on_loop_end`. No changes to
the loop are needed.

```python
from looplet.provenance import TrajectoryRecorder

hook = TrajectoryRecorder()
for step in composable_loop(llm=llm, tools=tools, state=state, hooks=[hook]):
    ...
hook.save("traces/run_1/")     # writes trajectory.json + steps/*.json
```

### Recording the harness snapshot

Use `serialize_harness()` to record the harness genome that produced a
trajectory without changing the trajectory schema:

```python
from looplet import serialize_harness
from looplet.provenance import TrajectoryRecorder

snapshot = serialize_harness(config=config, tools=tools, hooks=hooks, llm=llm)
hook = TrajectoryRecorder(harness_snapshot=snapshot)

for step in composable_loop(llm=llm, tools=tools, state=state, hooks=[hook]):
    ...

assert hook.trajectory.metadata["harness_snapshot"]["schema_version"] == 1
```

### Linking steps to LLM calls

Pair `TrajectoryRecorder` with `RecordingLLMBackend` and every
`StepRecord.llm_call_indices` will point into
`Trajectory.llm_calls`, so you always know which prompt produced a
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

| Field                      | Notes                                                               |
| -------------------------- | ------------------------------------------------------------------- |
| `step_num`                 | 1-based loop step index                                             |
| `timestamp`, `duration_ms` | When the step ran and how long                                      |
| `pretty`                   | `Step.pretty()` - `#N ✓ tool(args) → result [Xms]`                  |
| `tool_call`                | `ToolCall.to_dict()`                                                |
| `tool_result`              | `ToolResult.to_full_dict()`; complete result data and call identity |
| `context_before`           | The briefing shown to the LLM before this step's prompt             |
| `llm_call_indices`         | Into `trajectory.llm_calls` (empty if no recording backend)         |
| `metadata`                 | Free-form annotations for external hooks; defaults to `{}`          |

### Termination inference

`on_loop_end` inspects the last step and sets `termination_reason`:

- `"done"`: the agent called the `done` tool successfully
- `"error"`: the last step returned an error
- `"max_steps_or_stop"`: budget exhausted or a `should_stop` hook fired
- `"no_steps"`: the loop produced nothing

The loop's `done` path bypasses `post_dispatch`, so `TrajectoryRecorder`
sweeps `state.steps` at `on_loop_end` to catch that final step.

### Embedded `Tracer`

`TrajectoryRecorder` embeds a `Tracer` by default, so no extra hook is
needed. Pass your own if you want to export spans elsewhere:

```python
from looplet import Tracer
my_tracer = Tracer()
hook = TrajectoryRecorder(tracer=my_tracer)
...
# Export my_tracer.root_spans to OTel, Datadog, etc.
```

---

## Replay captured responses through a fresh harness

Use `replay_loop()` for synchronous tools and hooks.
Use `async_replay_loop()` when a tool or hook is asynchronous.

```python
from looplet import replay_loop

for step in replay_loop("traces/run_1/", tools=my_tools):
    print(step.pretty())
```

`replay_loop` reads `manifest.jsonl` from the trace directory and feeds captured responses
back into `composable_loop` in order. New captures use the validated structured snapshot;
legacy captures fall back to `call_NN_response.txt`. Required readable artifact files are
still validated, even when replay does not parse their contents.
The LLM is **not** called again. Your tools, hooks, permission
engine, and state are fresh. This is useful when a tool, hook, permission,
or loop-runtime change is the variable under review. It is not an A/B test of
prompt or model changes because the recorded decisions stay fixed.

**`0.4.1` defaults:** The defaults below require version `0.4.1` or newer.
Published `0.4.0` uses captured model-call count as the budget and defaults to an empty task.
Set `config` and `task` explicitly when using that release.

Without `config`, replay uses the saved tool-step count and the first captured call's protocol.
New trajectories also restore the recorded `allow_text_completion`, `done_tool`, `max_tokens`,
and `max_turn_continuations` settings. Call-only captures count native tool calls or text
responses instead. An explicit `LoopConfig` overrides these defaults.
Your host still supplies trusted tools, hooks, output schemas, and application state.
Saved scalar settings do not reconstruct executable policy or grader expectations.

Without `task`, replay reuses the saved trajectory task, or `{}` when no task was saved.
Pass `task={}` to request an empty task, or another dictionary to replace the saved task.
Replay does not restore checkpoints or copy starting files.

### Inspect replay failures

Replay reports model-call failures through the run state, not caller-visible exceptions.
If the loop asks for more captured responses, it ends with status `failed` and
`termination_reason="replay_exhausted"`. If the configured protocol differs from
the captured protocol, the run fails with `termination_reason="llm_error"`.
The final step's error describes the mismatch.

Pass state to inspect the outcome after iteration:

```python
from looplet import DefaultState, RunResult, replay_loop

state = DefaultState(max_steps=20)
steps = list(replay_loop("traces/run_1/", tools=my_tools, state=state, max_steps=20))
result = RunResult.from_state(state)
print(result.status.value, result.termination_reason)
```

This also applies to `async_replay_loop`. Tools that ran before failure keep their
fresh side effects. Replay does not roll them back or alter the source capture.
Missing or invalid capture files can still raise before the loop starts.

### Replay async agents

The async entry point uses the same evidence validation, saved task, protocol, and budget defaults.
It awaits async hooks and tools; synchronous implementations remain supported.
Use `aclosing` to release loop-owned resources when iteration stops early:

```python title="replay_async.py"
from contextlib import aclosing

from looplet import async_replay_loop


async def replay_agent(trace_dir, tools, hooks=()):
  steps = []
  async with aclosing(async_replay_loop(trace_dir, tools=tools, hooks=list(hooks))) as iterator:
    async for step in iterator:
      steps.append(step)
  return steps
```

Call `await replay_agent(trace_dir, tools, hooks=hooks)` from your async host.
Fresh tools and hooks still execute; use an isolated workspace for side effects.
The shell replay command remains synchronous.

**Common uses:**

- **Change a hook, re-run.** Add a new permission rule or logging
  hook, replay, see how step outputs differ.
- **Upgrade looplet and diff the loop.** If the replay produces
  different steps with the same LLM output, the loop behavior changed.
- **Golden tests.** Create a [reviewed synthetic replay fixture](#create-a-reviewed-replay-fixture), then replay it in CI.
- **Reproduce a tool failure.** Hold model responses fixed, change one tool
  implementation, then let a collector grade the fresh world state.

**Constraints:**

- If your loop now asks for **more** LLM calls than were recorded,
  replay fails with `replay_exhausted`; reduce `max_steps` or re-record.
- If a call was recorded as `generate_with_tools` but the replay loop
  requests `generate`, replay fails through normal model-call error handling.
  Keep the captured protocol unless the divergence is the subject of the test.
- Tools **do** execute at replay time. Replace with mocks if you don't
  want side effects.
- Prompt edits do not affect the already-captured responses. Record fresh
  sampled runs to evaluate whether a prompt changes model decisions.
- If `manifest.jsonl` is missing, replay falls back to the
  `call_NN_response.txt` files alone (every call treated as
  `generate`).

---

## Inspect a trace from the CLI

```
$ python -m looplet show traces/run_1/
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
# Now you have traces/<name>/trajectory.json for every scenario -
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

- Capture snapshots and redacts the request, response, and available provider objects on each
  model call, plus one `StepRecord` per tool step.
- `max_chars_per_call` (default 200,000 chars) bounds readable prompt and response projections.
  Full structured snapshots remain in memory and are written to both the manifest and trajectory.
  Set host storage limits and review/redact provider data for large or sensitive workloads.
- `save()` is the only disk I/O. Call it once at loop end (or inside
  an `except` handler), not inside the loop.
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
  recorder writes prompts verbatim. If your prompts contain PII and
  you can't scrub, pick a different tool.

For development, CI, benchmarks, debugging, and golden tests, this is the
intended workflow.
