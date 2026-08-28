# Saved artifact reference

Looplet writes ordinary text and JSON so a run can be reviewed without a
proprietary viewer. This page identifies each file, who should own it, and
which reader to use.

Saved run directories are evidence, not harmless logs. They can contain full
prompts, tool schemas, model responses, task data, filesystem observations,
and grader expectations.

## Provenance trace

`ProvenanceSink` combines model-call capture with step-level trajectory
capture:

```text
traces/run-42/
├── artifact.json
├── trajectory.json
├── steps/
│   ├── step_00.json
│   └── step_01.json
├── manifest.jsonl
├── call_00_prompt.txt
├── call_00_response.txt
├── call_01_prompt.txt
└── call_01_response.txt
```

| Path | Contents | Primary reader |
| --- | --- | --- |
| `artifact.json` | Format version, artifact kind, producing Looplet version, and declared components. | All supported directory readers |
| `trajectory.json` | Run metadata, task view, steps, stop reason, timing, and captured context. | `EvalContext.from_trajectory_dir()` or JSON tooling |
| `steps/step_NN.json` | One review-friendly copy of each step record. | Humans, diffs, JSON tooling |
| `manifest.jsonl` | One structured summary per model call. | `looplet show`, replay loader, line-oriented tooling |
| `call_NN_prompt.txt` | Exact recorded system prompt, user prompt, tool schemas, and call settings. | Humans and replay diagnostics |
| `call_NN_response.txt` | Exact recorded model response or captured error. | Replay loader and humans |

Create and inspect a trace:

```python
from looplet import ProvenanceSink


sink = ProvenanceSink(dir="traces/run-42", redact=scrub_secrets)
recorded_llm = sink.wrap_llm(llm)

for step in composable_loop(
    llm=recorded_llm,
    tools=tools,
    hooks=[sink.trajectory_hook()],
    task=task,
):
    route(step)

sink.flush()
```

```bash
looplet show traces/run-42
```

The prompt and response text files are deliberately readable. Do not publish
them without inspection and redaction.

## Persisted eval run

`save_eval_run()` and `looplet eval run --out` add independent outcome data,
grader results, and case identity to the trajectory:

```text
eval-runs/regression-42/
├── artifact.json
├── trajectory.json
├── steps/
├── manifest.jsonl              # when a recording backend was attached
├── call_NN_prompt.txt          # when model calls were recorded
├── call_NN_response.txt
├── artifacts.json
├── evals.json
├── expected.json               # when the case has expectations
└── case.json                   # when the case was supplied
```

| Path | Contents | Trust role |
| --- | --- | --- |
| `artifact.json` | Declares a versioned `eval_run` and its required components. | Compatibility boundary |
| `artifacts.json` | Collector-observed world state used by graders. | Host observation |
| `evals.json` | Normalized grader scores, labels, metrics, and errors. | Decision evidence |
| `expected.json` | Grader-only expected data restored into `ctx.task["expected"]` after the run. | Promotion oracle input |
| `case.json` | The source case: id, task, expected data, marks, and notes. | Corpus identity and review |

Load the complete record through the supported reader:

```python
from looplet import load_eval_run


record = load_eval_run("eval-runs/regression-42")
print(record.context.artifacts)
print([result.pretty() for result in record.results])
print(record.case.id if record.case else "no case metadata")
```

A missing `trajectory.json` or malformed JSON fails loudly. Collector errors
remain explicit eval results rather than disappearing as absent data.

### Writer inventory

| Writer | Files it owns |
| --- | --- |
| `RecordingLLMBackend.save()` and its async twin | `artifact.json`, `manifest.jsonl`, and indexed `call_NN_{prompt,response}.txt` pairs |
| `TrajectoryRecorder.save()` | `artifact.json`, `trajectory.json`, `steps/step_NN.json`, and the model-call files when a recording backend is attached |
| `ProvenanceSink.flush()` | The union selected by its attached recorder/backend; an unused sink may create only its directory |
| `EvalHook.save()` | One legacy standalone JSON report with `task`, `results`, `summary`, and optional `expected` / `artifacts` |
| `save_eval_run()` | A versioned eval-run directory: required trajectory, artifacts, and eval results; optional case, expectations, and recorded calls |
| `promote_to_offline()` | The same layout as `save_eval_run()` |
| `run_cartridge_evals(..., output_dir=...)` | One `save_eval_run()` directory per case plus a `workspace/` owned by the evaluated application |

`EvalHook.save()` remains an unversioned compatibility report. It has no
supported round-trip reader and should not be used as a long-lived CI wire
format. Use `save_eval_run()` for durable evidence; use the documented
`looplet eval run --json` schema for transient CI decisions.

### Agent-visible and grader-only data

During a cartridge eval, only `case.task` is sent to the agent. The top-level
`case.expected` object is withheld, persisted separately as `expected.json`,
and restored for graders after execution.

That separation prevents accidental prompt leakage. It is not a security
sandbox. Candidate code running with the same filesystem or process authority
may still inspect runner files or memory. Keep promotion cases, expected data,
collectors, graders, and capabilities in a host-owned runner, and use OS or
process isolation for untrusted candidates.

## Checkpoints

`LoopConfig(checkpoint_dir=...)` writes one JSON checkpoint per completed step:

```text
.looplet/checkpoints/task-42/
├── step_1.json
├── step_2.json
└── step_3.json
```

A checkpoint stores the step number, session log, conversation, selected
configuration fields, tool-result store, metadata, and creation timestamp.
When the same checkpoint directory is used again and `initial_checkpoint` is
unset, Looplet resumes the highest-step valid checkpoint.

Checkpoints are recovery state, not provenance or eval evidence. Use a unique
directory per logical task and apply the same access and retention policy as
traces.

## Legacy compatibility inputs

`EvalContext.from_trajectory_dir()` can read selected top-level fields from a
legacy `metrics.json` used by older benchmark traces. New eval runs should
write collector output to `artifacts.json` and top-level case expectations to
`expected.json`. Do not create new dependencies on the legacy convention.

## Compatibility policy

The cartridge schema and saved-artifact schema are independent. New provenance
and eval-run directories carry this descriptor:

```json
{
  "schema": "looplet.saved-artifact",
  "version": 1,
  "kind": "provenance",
  "producer": {"name": "looplet", "version": "0.4.0"},
  "components": ["model_calls", "trajectory"]
}
```

`version` governs the artifact layout; `producer.version` identifies the
package that wrote it for diagnostics. They do not advance together. `kind` is
`provenance` or `eval_run`. Version 1 components use the closed vocabulary
`trajectory`, `model_calls`, `artifacts`, and `eval_results`.

An absent `artifact.json` means the supported unversioned legacy shape (called
v0), not “the latest version.” Looplet keeps concrete legacy fixtures for that
shape, including the older `metrics.json` input. An unknown schema, unsupported
explicit version, unknown kind/component, malformed descriptor, or missing
declared component fails before rendering, replay, or grading.

### Stable and optional fields

Required means a current writer emits the field and a v1 reader may rely on its
meaning. Optional means consumers must accept its absence and producers may add
it without advancing the format. Unknown object fields are ignored; this makes
additive metadata forward-compatible. Unknown component names are refused
because components declare completeness.

| File | Required stable | Optional stable | Internal or application-owned |
| --- | --- | --- | --- |
| `artifact.json` | `schema`, `version`, `kind`, `producer.name`, `producer.version`, `components` | Additional object fields | None |
| `trajectory.json` | `run_id`, `termination_reason`, `steps`; each step's `tool_call` and `tool_result` | `task`, timestamps, counts, session text, metadata, context, call linkage, call summaries | In-memory span objects |
| `manifest.jsonl` | Contiguous `index`, supported `method`; matching indexed prompt/response files and replay response section | Timing, sizes, tool count, step link, error, scope, metadata | None |
| `artifacts.json` | A JSON object | Application-defined keys and values | The meaning of application keys |
| `evals.json` | An array; each result's `name` | Present `score`, `label`, `metrics`, `details`, `explanation`, `duration_ms` | None |
| `expected.json` | A JSON object when present | The whole file is optional | Application-defined keys and values |
| `case.json` | `id` and `task` when present | `expected`, `marks`, `notes`; the whole file is optional | None |
| `steps/step_NN.json` | None | None | Redundant review copy; `trajectory.json` is authoritative |
| `workspace/` | None | None | Evaluated application state, outside the artifact schema |

Current writers also emit `task`, `results`, and `summary` in the standalone
`EvalHook.save()` report; `expected` and `artifacts` are optional there. Those
v0 fields retain their current meaning, but new automation should use the
versioned directory or CI summary instead.

Malformed decision-bearing JSON fails closed. Missing optional fields use the
defaults documented by the supported readers. Conflicting expected data is an
error. Consumers should call `load_eval_run()` and
`EvalContext.from_trajectory_dir()` rather than reconstructing internal
dataclasses from JSON.

### Publication and incomplete directories

Saved directories are not transactional databases, and concurrent reads while
a writer is active are unsupported. Before clearing an old descriptor, writers
atomically publish `.artifact.json.pending`. Readers refuse any directory with
that marker, so a crash cannot turn a partial v1 replacement into an
ordinary-looking legacy v0 artifact. Writers then replace the payload, publish
`artifact.json` with an atomic file replacement, and remove the pending marker.
Once a v1 descriptor is present, every declared component file must be present
or the directory is rejected as incomplete.

Legacy v0 has no completion marker, so its historical best-effort behavior is
retained. Required malformed files still fail loudly. Reusing a directory
removes only exact Looplet-owned names; use one directory per logical run and
inspect it only after the writer returns.

For automation, preserve the artifact directory intact, tolerate additional
object fields, fail on missing required evidence, and review the changelog
before advancing the producing or reading Looplet version. There is no promise
to read every historical or future shape.

## Redaction and retention

Use `ProvenanceSink(redact=...)` to transform persisted prompt and result text.
By default, the sink also applies the redactor before forwarding captured
content upstream. Verify that behavior against the application's privacy
requirements rather than assuming storage-only redaction.

A production policy should state:

- which prompts, responses, tool results, and task fields may be retained;
- which values are removed before provider calls and before disk writes;
- who can read traces, eval expectations, and checkpoints;
- how long each artifact type is retained;
- whether CI artifacts cross repository or organizational boundaries;
- how deletion requests and incident response apply to saved runs.

Never place credentials in case files, command arguments, or grader notes.
Do not upload unreviewed traces to a public issue.

## Which artifact should I use?

| Need | Artifact |
| --- | --- |
| Inspect what the model saw and returned | Provenance trace |
| Re-execute recorded responses through changed harness code | Provenance trace with recorded calls |
| Re-grade an observed product outcome | Persisted eval run |
| Review one step in a pull request | `steps/step_NN.json` |
| Resume an interrupted logical task | Checkpoint directory |
| Compare prompt or model quality | Fresh sampled runs plus persisted eval records |

Read [capture and replay](provenance.md) for execution semantics,
[behavioral evals](evals.md) for graders and cases, and
[experiment design](experiments.md) before choosing replay as a control.
