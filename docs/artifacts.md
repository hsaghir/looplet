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

The cartridge format has its own schema version. Saved provenance and eval
run directories in `0.3` do not yet declare a separate stable artifact schema
version.

For automation:

1. pin the Looplet minor line;
2. prefer `load_eval_run()` and `EvalContext.from_trajectory_dir()` over
   reconstructing dataclasses from JSON fields;
3. tolerate additional object fields when consuming JSON;
4. fail on missing required evidence rather than substituting success;
5. preserve the producing Looplet version and cartridge hash beside exported
   artifacts;
6. review the changelog before upgrading the reader or producer.

The roadmap includes explicit schema-version and compatibility guarantees for
saved evidence. Until those land, readable files aid review and portability
but are not a frozen cross-version wire contract.

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
