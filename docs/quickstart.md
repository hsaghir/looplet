# Quickstart

Run one Python agent, get its answer, test it, and replay its saved model responses.
You need Python 3.11 or newer.
You do not need a model key until the optional real-model section.
Start with Python functions.
Keep the agent in Python while you change and test it.
Use a Python factory for shell commands; package it only when you need a directory artifact.

## 1. Install Looplet

Use this pinned preview for the whole guide, including the editable starter and factory commands.
Those commands are unreleased and are not in published `0.4.1`.
The install does not require Git or a repository checkout.

```bash
python -m venv .venv-looplet-preview
source .venv-looplet-preview/bin/activate
python -m pip install "looplet @ https://github.com/hsaghir/looplet/archive/270ebab43f0be9deea4b2a039ded2bbd3d94acfd.zip"
looplet doctor --no-backend --json
```

The core package has no third-party runtime dependencies.
The install needs network access; the fixed-response runs do not.
This is a non-editable install of commit `270ebab`.
Keep this environment active for the complete guide; no second install is needed for the starter.
Both the preview and published package report version `0.4.1`.
The preview's `looplet --version` also prints the doctor's source fingerprint, distinguishing it from the published build.
Keep the commit and the doctor's library source fingerprint with your results.
Pin artifact hashes in your dependency lockfile for deployment.
On Windows, activate `.venv-looplet-preview\Scripts\Activate.ps1` in PowerShell.
See the [preview details](install.md#try-unreleased-factory-commands) for the same install.
The [published fallback](install.md#install-the-published-package) supports the incremental Python examples, but not the editable starter workflow.

## Run the complete example

The package includes the complete example below in one Python module.
These commands need no authored Python files, model key, or provider service:

```bash
python -m looplet.examples.quickstart
```

The command runs the agent and checks its answer.
Use a new trace directory to capture the same run:

```bash
python -m looplet.examples.quickstart --capture ./quickstart-trace
looplet inspect ./quickstart-trace
```

**Unreleased source behavior:** failed captures print their stop reason and an inspection command before returning nonzero.
The starter's grader explains incomplete runs and shows expected and observed values for mismatched fields.
Capture, rescoring, and inspection preserve that explanation without sending expectations to the model.
The saved trace remains available for inspection.
For `budget_exhausted`, increase `--max-steps` and choose a new capture directory for the next run.
Published `0.4.1` saves this evidence but may print only the failed grader name.

Rescore saved evidence without running tools or calling a model:

```bash
python -m looplet.examples.quickstart --rescore ./quickstart-trace --out ./quickstart-assessment.json
looplet inspect ./quickstart-assessment.json
```

This saves a new assessment using the current grader without replacing capture-time grades.
Choose a new report filename for each assessment; existing files are refused.
Without `--out`, rescoring prints a transient result.

Replay captured model responses with fresh tools, then check the new result:

```bash
python -m looplet.examples.quickstart --replay ./quickstart-trace
```

Replay runs tool effects again.
This example only reads an in-memory lookup table.
Use isolated workspaces for tools that modify files or contact services.

Stop after the lookup without calling `done`:

```bash
python -m looplet.examples.quickstart --stop-after-lookup
```

Read the [complete Python source](https://github.com/hsaghir/looplet/blob/master/src/looplet/examples/quickstart.py) to edit or reuse its functions.

### Make the complete example your own

Keep using the environment from step 1.
It already contains the complete starter and factory commands.

Use one editable file for your tools, capture, rescoring, and replay.
Export the installed companion once; an existing file is never replaced:

```bash title="copy_agent_starter.sh"
python - <<'PY'
from importlib.resources import files
from pathlib import Path

starter = files("looplet.examples").joinpath("quickstart.py")
with Path("my_agent.py").open("x", encoding="utf-8") as output:
    output.write(starter.read_text(encoding="utf-8"))
PY
```

Edit the tools, `EXPECTED_OWNER`, grader, and fixed responses in that file.
The default `owner_case()` derives the task from `EXPECTED_OWNER["service"]` and keeps the expected answer separate.
No other authored Python files are needed.
Run this complete workflow again whenever you change the agent:

```bash title="custom_agent_workflow.sh"
python my_agent.py --experiment
```

Each invocation creates a new directory under `.looplet/experiments/` and preserves earlier evidence.
It captures and grades the run, rescores saved evidence, then replays and grades fresh tool results.
The command prints inspection commands for the experiment and available records.
Inspect the printed experiment directory or its `experiment.json` file:

```bash
looplet inspect PATH_TO_EXPERIMENT
looplet inspect PATH_TO_EXPERIMENT/experiment.json --json
```

The report records the last stage, each stage's status, and any error.
Stage statuses are `passed`, `failed`, or `not_run`; the last stage is `complete` after success.
Trace paths are `null` when their directories were not created.
Otherwise, paths are relative to `experiment.json`, not your current directory.
Copy or move the whole experiment directory to keep its trace links working.
Inspection also prefers colocated capture and replay directories in copied legacy reports.
Failed runs keep their evidence; inspecting a readable failed report returns zero without rerunning anything.
Use the printed trace commands for steps, results, and saved grades.
Use `--out NEW_DIRECTORY` to choose a new experiment directory explicitly.
Existing directories are refused.
Experiment directories store evidence; they do not isolate tool effects.
The starter prints human-readable output; it does not support `--json`.
Unknown flags are rejected rather than abbreviated.

Inspect the same exported factory without calling a model:

```bash
looplet inspect my_agent:build --mode runtime
```

Human output lists tools, the completion tool, step limit, tool protocol, and hook count.
Use its printed `--json` command for the full blueprint with the same workspace and step limit.
The default workspace and step limit also support this JSON form:

```bash title="inspect_agent.sh"
looplet inspect my_agent:build --mode runtime --json
```

Runtime inspection imports trusted Python, builds and validates one preset, then closes its resources.
It checks structural wiring, not behavioral correctness.
Without `--mode runtime`, inspection reports only the factory reference and does not import the module.
Use `module:factory`, not the Python file path.

Use one case file when the task needs different wording or expectations:

```json title="owner_case.json"
{
    "id": "payments-owner",
    "task": {"goal": "Look up who owns the payments service."},
    "expected": {"service": "payments", "owner": "fintech-platform"}
}
```

```bash title="one_file_case.sh"
python my_agent.py --case owner_case.json --experiment
```

The task goes to the agent; expectations stay with the grader.
Capture and replay save the same case.
`--case` applies only to fresh runs; replay and rescoring use their saved case.
Fixed responses are fixtures; update them when changing the expected tool calls or answers.
For multiple cases or seed files, use the [factory CLI](#use-the-same-factory-in-the-shell).
The one-file command refuses seeded cases instead of writing their files into the current directory.

Use the individual operations when diagnosing a run:

```bash
python my_agent.py --capture ./original-capture
python my_agent.py --rescore ./original-capture --out ./current-assessment.json
looplet inspect ./current-assessment.json
python my_agent.py --replay ./original-capture --out ./replayed-capture --max-steps 2
looplet inspect ./replayed-capture
```

Choose unused paths for captures, assessment reports, and replays.
Rescoring saves its assessment before returning nonzero for a failed grade.
Trace inspection still shows capture-time grades; report inspection shows the separately saved assessment.
Replay saves fresh prompts, responses, steps, and grades without changing the original record.
It reuses the saved task and expectations, then runs current tools and graders.
Without `--max-steps`, replay uses the recorded tool-step count.
An explicit replay limit is honored.
A larger limit can exhaust the recorded responses.
Replay does not call the original provider or restore historical workspace state.
Use isolated workspaces when tools modify files or contact services.
Malformed saved input returns nonzero with concise recovery guidance.
Keep damaged files; restore an intact copy or capture a fresh run in a new directory.

Use the same saved capture and exported factory from the CLI:

```bash
looplet replay ./original-capture --factory my_agent:build --evals my_agent.py --workspace ./cli-replay
```

Choose a new workspace path each time.
CLI replay preserves the captured protocol, task, and expectations.
It collects fresh evidence but does not run graders.
Run its printed `Grade` command to assess the new record.
Older eval captures without a case file retain their saved expectations too.

#### Use the one-file agent with a real model

Keep the exported file and install the provider extra from the same pinned preview:

```bash title="one_file_live.sh"
python -m pip install "looplet[openai] @ https://github.com/hsaghir/looplet/archive/270ebab43f0be9deea4b2a039ded2bbd3d94acfd.zip"
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o
python my_agent.py --backend openai --max-steps 8 --experiment
```

This command makes real provider calls and may incur charges.
The same file still owns the tools, factory, graders, and captured-response replay.
For Anthropic, install the `anthropic` extra from that preview and use `--backend anthropic`.
Configure its key with the [provider settings](install.md#configure-anthropic).
Add `--json-text` only for an endpoint that needs JSON-text instead of native tools.
`--json-text` selects the model protocol; it does not request JSON output.
Missing provider settings or SDKs produce a concise setup error before creating experiment output.
Check the named variables, install the matching provider extra, and rerun the command.
Replay and rescoring reject provider options because neither calls that provider.
The file also exports `build(runtime)` for the existing `my_agent:build` factory CLI interface.

The remaining sections explain the pieces incrementally.
They are optional; the one-file starter does not import their example files.

## 2. Run the agent and get its answer

This example uses fixed model responses.
It does not contact a model service.
The tool returns the team that owns a service.

```python title="first_loop.py"
from looplet import MockLLMBackend, minimal_preset


def lookup_owner(service: str) -> dict:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


def done(service: str, owner: str, summary: str = "") -> dict:
    """Finish with the service, its owner, and an optional explanation."""
    return {"service": service, "owner": owner, "summary": summary}


def scripted_backend():
    return MockLLMBackend([
        '{"tool":"lookup_owner","args":{"service":"payments"}}',
        '{"tool":"done","args":{"service":"payments","owner":"fintech-platform","summary":"The payments owner is fintech-platform."}}',
    ], cycle=False)


def run_agent(llm, *, hooks=(), use_native_tools=False, max_steps=2):
    with minimal_preset(tools=[lookup_owner, done], max_steps=max_steps, validate=True) as run:
        run.config.use_native_tools = use_native_tools
        run.hooks.extend(hooks)
        return run.run_result(llm, task={"goal": "Find the payments owner, then finish."})


if __name__ == "__main__":
    result = run_agent(scripted_backend())
    print(result.output)
    print(result.completed, result.termination_reason)
```

Run the example file:

```bash
python first_loop.py
```

The loop calls `lookup_owner`, then `done`.
`result.output["owner"]` contains the owner; `summary` is an optional explanation.
The last line prints `True done`.
This helper defaults to JSON-text for its fixed responses.
Pass `use_native_tools=True` when changing to a native-tool backend.
The [real-model section](#use-the-same-agent-with-a-real-model) shows the complete change.
The scripted run needs two calls; `max_steps` lets other runs use a different limit.
`cycle=False` rejects unexpected extra model calls instead of repeating the script.

These tools remain ordinary Python functions.
Call them directly in your application or unit tests.
`minimal_preset` infers tool schemas from their signatures and descriptions from their docstrings.
Version `0.4.0` may show a false `ToolSpec` type error for these functions; `0.4.1` corrects the annotation.
The optional `@tool` decorator returns a `ToolSpec`, not a callable function.
Call a decorated tool's `.execute(...)` method to test its underlying function.
That direct call bypasses registry validation and hooks.

`AgentPreset` holds one run's mutable state and resources.
It is single-use, not a reusable configuration object.
`run_agent()` creates and closes a fresh instance for each call.
Reuse the function for repeated or concurrent tasks, not the instance inside it.

`validate=True` checks top-level tool argument types and enum choices before effects.
Nested schemas and other constraints remain tool-owned.
The base registry retains its lighter argument-name checks when this option is omitted.

## Use the same agent with a real model

This step is optional.
The remaining sections keep using the fixed-response backend.

Keep `first_loop.py` unchanged and add the adapter from the same pinned preview in your active environment:

```bash
python -m pip install "looplet[openai] @ https://github.com/hsaghir/looplet/archive/270ebab43f0be9deea4b2a039ded2bbd3d94acfd.zip"
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o
```

Keep the package and provider extras on the same commit.

Put this in a separate `first_loop_live.py` file:

```python title="first_loop_live.py"
from looplet import OpenAIBackend

from first_loop import run_agent


if __name__ == "__main__":
    result = run_agent(OpenAIBackend.from_env(), use_native_tools=True, max_steps=8)
    print(result.output)
    print(result.completed, result.termination_reason)
    if not result.completed:
        raise SystemExit(f"Agent stopped before completion: {result.termination_reason}")
```

```bash
python first_loop_live.py
```

This makes real provider calls and may incur charges.
It uses the same tools and fresh-run factory, with native tool calls enabled explicitly.
The live example allows up to eight tool calls, including completion.
Adjust `max_steps` for your task; budget exhaustion means the run did not finish.
For Anthropic, install the `anthropic` extra from the same preview.
Replace `OpenAIBackend` with `AnthropicBackend`.
Configure its key as described in [Install](install.md#configure-anthropic).
Use `use_native_tools=False` only when you intentionally select JSON-text responses.

## 3. Check the answer

Completion does not prove correctness.
Add these definitions above the main block in `first_loop.py`.
Replace the existing main block with the one at the end of this example.
The grader compares the observed answer with a host-owned expectation:

```python
from looplet import EvalContext, eval_assess


EXPECTED_OWNER = {"service": "payments", "owner": "fintech-platform"}


def build(runtime):
    return minimal_preset(tools=[lookup_owner, done], max_steps=runtime.max_steps, validate=True)


def eval_owner(ctx):
    return (
        ctx.completed
        and isinstance(ctx.final_output, dict)
        and ctx.final_output.get("service") == ctx.expected["service"]
        and ctx.final_output.get("owner") == ctx.expected["owner"]
    )


def assess_result(result):
    observed_context = EvalContext(
        steps=result.steps,
        final_output=result.output,
        stop_reason=result.termination_reason,
        task={"expected": dict(EXPECTED_OWNER)},
    )
    return eval_assess([eval_owner], observed_context)


if __name__ == "__main__":
    result = run_agent(scripted_backend())
    print(result.output)
    print(result.completed, result.termination_reason)
    report = assess_result(result)
    report.assert_passed()
    print(report.passed)
```

This prints `True` after checking completion and the answer.
The grader checks service and owner fields, not the explanation's exact wording.
The expected answer belongs to the grader; it is not part of the agent task.
For files or database changes, collect those effects independently before grading.
See [Behavioral evals](evals.md) for collectors and saved cases.

Put these tests beside `first_loop.py`:

```python title="test_first_loop.py"
from first_loop import assess_result, done, lookup_owner, run_agent, scripted_backend


def test_tools_are_regular_functions():
    assert lookup_owner("payments") == {"service": "payments", "owner": "fintech-platform"}
    assert done("payments", "fintech-platform") == {
        "service": "payments", "owner": "fintech-platform", "summary": ""
    }


def test_owner():
    result = run_agent(scripted_backend())
    assess_result(result).assert_passed()


def test_repeated_runs_are_independent():
    first = run_agent(scripted_backend())
    second = run_agent(scripted_backend())
    assert first.completed and second.completed
    assert first.steps[0] is not second.steps[0]
    assert first.output == second.output
```

Run the tests:

```bash
python -m pip install pytest
python -m pytest test_first_loop.py
```

Validate the same grader definitions without running the agent:

```bash
looplet eval check first_loop.py --json
```

Run Python agents with `python first_loop.py`; run their tests with pytest.
`eval init --python evals/` creates definitions and a runnable pytest example.
Run it with `python -m pytest evals/`.
Omit `--python` to create definitions only.

### Use the same factory in the shell

**Unreleased:** the following factory commands are not in published `0.4.1`.
The Python and pytest commands above work with that release.
Use the [pinned preview install](install.md#try-unreleased-factory-commands) before trying these shell commands.

`build(runtime)` returns a fresh preset with the same tools.
The CLI supplies `SkillRuntime` and closes the preset after each run.
Factories with file effects should use `runtime.workspace` for their working files.
Factory imports use the current directory and installed Python environment.
These commands execute trusted Python; they do not provide an operating-system sandbox.
No cartridge or bundle is required.

Create `cases/owner.json` for the same host-owned expectation:

```json title="cases/owner.json"
{
    "id": "owner",
    "task": {"goal": "Find the payments owner, then finish."},
    "expected": {"service": "payments", "owner": "fintech-platform"}
}
```

Run, evaluate, and replay the same factory without a provider:

The one-file starter above handles routine experiments with `python my_agent.py --experiment`.
The following manual CLI recipe is optional.
It uses the incremental `first_loop` example and exposes each operation separately.

```bash title="python_factory_cli.sh"
set -eu
mkdir -p .looplet/experiments
experiment_root="$(mktemp -d .looplet/experiments/owner.XXXXXXXX)"
printf 'Experiment: %s\n' "$experiment_root"

looplet run first_loop:build "Find the payments owner, then finish." \
    --scripted-response '{"tool":"lookup_owner","args":{"service":"payments"}}' \
    --scripted-response '{"tool":"done","args":{"service":"payments","owner":"fintech-platform"}}' \
    --max-steps 2 --json > "$experiment_root/run.json"

looplet eval run --factory first_loop:build --evals first_loop.py \
    --cases cases/owner.json --out "$experiment_root/evals" --max-steps 2 \
    --scripted-response '{"tool":"lookup_owner","args":{"service":"payments"}}' \
    --scripted-response '{"tool":"done","args":{"service":"payments","owner":"fintech-platform"}}' \
    --json > "$experiment_root/eval.json"

case_dir="$(python - "$experiment_root/eval.json" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as source:
    report = json.load(source)
print(Path(report["output_dir"]) / report["cases"][0]["id"])
PY
)"

looplet replay "$case_dir" --factory first_loop:build --evals first_loop.py \
    --workspace "$experiment_root/replay" --json > "$experiment_root/replay.json"

replay_trace="$(python - "$experiment_root/replay.json" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8") as source:
    print(json.load(source)["trace_dir"])
PY
)"

looplet inspect "$experiment_root/evals"
looplet inspect "$replay_trace"
looplet eval "$replay_trace" --evals first_loop.py --out "$experiment_root/assessment.json" \
    --json > "$experiment_root/grade.json"
looplet inspect "$experiment_root/assessment.json"
python -m json.tool "$experiment_root/grade.json"
```

Repeat this whole recipe after changing the agent; each invocation creates a fresh experiment directory.
The reports supply the case and replay paths, so no path substitutions are needed.
The printed directory contains the run, eval, replay, and grade reports, plus the separate assessment.
Suite inspection lists saved case grades without rerunning graders or guessing the original gate policy.
This single-case example replays the first reported case; select each case when expanding your suite.
Earlier evidence and workspaces are never overwritten.

`run` reports accepted completion; without eval hooks, its `grading_status` is `not_graded`.
`eval check` reports definition validity and `behavior_run: false`; it does not grade an answer.
`eval run` executes the graders and gates on their assessment.
Replay reuses model responses and reruns tools and collectors in the new workspace.
It does not rerun graders; grade its new trace with `looplet eval <trace-dir> --evals first_loop.py`.
Replay is not checkpoint restoration or deterministic execution.
Keep the original module and dependencies available; these commands do not copy them.

## 4. Capture the same agent

Put this block in a separate `first_loop_capture.py` file.
Importing `first_loop` loads its functions without running its main block.
Capture records model calls and tool steps; it does not change the agent's tools or task.

```python title="first_loop_capture.py"
from shlex import quote

from looplet import EvalHook, ProvenanceSink, save_eval_run

from first_loop import EXPECTED_OWNER, eval_owner, run_agent, scripted_backend


def capture_agent(llm):
    sink = ProvenanceSink()
    grading = EvalHook(evaluators=[eval_owner], expected=EXPECTED_OWNER)
    try:
        captured_result = run_agent(
            sink.wrap_llm(llm),
            hooks=[sink.trajectory_hook(), grading],
        )
    finally:
        trace_dir = sink.flush()

    save_eval_run(trace_dir, recorder=sink.trajectory_hook(), eval_hook=grading)
    try:
        grading.assessment().assert_passed()
    except AssertionError as error:
        raise AssertionError(
            f"{error}\nStop reason: {captured_result.termination_reason}\n"
            f"Inspect with: looplet inspect {quote(str(trace_dir))}"
        ) from error
    return captured_result, trace_dir


if __name__ == "__main__":
    captured_backend = scripted_backend()
    captured_result, trace_dir = capture_agent(captured_backend)
    print(captured_result.output)
    print(f"Inspect with: looplet inspect {quote(str(trace_dir))}")
    print(f"Grade with: looplet eval {trace_dir} --evals first_loop.py --json")
    print(f"Rescore with: python first_loop_rescore.py {trace_dir}")
    print(f"Replay with: python first_loop_replay.py {trace_dir}")
```

Run one captured agent:

```bash
python first_loop_capture.py
```

Run the printed `inspect` command to read the saved run.
This command executes the agent once and saves one trajectory.
Each execution gets a unique directory under `.looplet/traces/`.
An explicit `ProvenanceSink(dir=...)` refuses existing evidence unless you pass `overwrite=True`.
Capture can contain secrets or personal data; review it before sharing or committing it.
New record directories own a wildcard `.gitignore`; it does not protect already tracked files or replace redaction.
See [reviewed summaries](provenance.md#share-a-reviewed-summary) before exporting evidence.

The grading hook owns the live assessment once.
`save_eval_run()` saves its expectations separately from the agent-visible task.
A raw trace alone does not preserve grader expectations or collected artifacts.
Reload the saved evidence when you want to rerun a grader without running the agent:

```python title="first_loop_rescore.py"
import sys
from pathlib import Path
from shlex import quote

from looplet import eval_assess, load_eval_run

from first_loop import eval_owner


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python first_loop_rescore.py TRACE_DIR [NEW_REPORT.json]")
    saved_run = load_eval_run(sys.argv[1])
    saved_report = eval_assess([eval_owner], saved_run.context, case=saved_run.case)
    report_path = (
        Path(sys.argv[2]) if len(sys.argv) == 3
        else saved_run.directory.with_name(saved_run.directory.name + "-assessment.json")
    )
    saved_report.save(report_path, evidence={saved_run.directory.name: saved_run.directory})
    print(saved_report.passed)
    print(f"New assessment; capture-time grades are unchanged. Inspect: looplet inspect {quote(str(report_path))}")
    saved_report.assert_passed()
```

Run the printed `Rescore` command after creating this file.
The printed `eval` command uses the same saved evidence and grader.
It accepts this single run directory or its parent directory.
Offline rescoring reruns graders, not collectors, tools, or agent model calls.
The Python script saves a sibling assessment before asserting its verdict.
Pass a new report filename as the second argument when rescoring the same capture again.
Keep reports private; grader explanations can contain sensitive evidence.

## 5. Replay and check the result

Put this block in a separate `first_loop_replay.py` file.
It uses the saved responses, the same tools, and fresh run state:

```python title="first_loop_replay.py"
import sys
from contextlib import closing

from looplet import DefaultState, RunResult, replay_loop, tools_from

from first_loop import assess_result, done, lookup_owner


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python first_loop_replay.py TRACE_DIR")
    replay_state = DefaultState(max_steps=2)
    with closing(replay_loop(
        sys.argv[1],
        tools=tools_from([lookup_owner, done], validate=True),
        state=replay_state,
        task={"goal": "Find the payments owner, then finish."},
    )) as steps:
        for step in steps:
            print(step.tool_call.tool, step.tool_result.data)

    replayed_result = RunResult.from_state(replay_state)
    assess_result(replayed_result).assert_passed()
```

Run the printed `Replay` command after creating this file.
Replay does not call the original model backend.
Tools and their effects run again.
Use isolated workspaces and controlled inputs for tools that write files or contact services.
A changed prompt or model needs a new model run, not replay.
For packaged agents, the [shell replay recipe](cli.md#replay-captured-responses) provides the same handoff through `looplet replay`.

## Inspect steps or stop early

`run_result()` consumes and closes the same iterator that `composable_loop()` returns.
Use the iterator directly when your application must inspect each step or stop between calls.
Put this block in `first_loop_stop.py` to stop after the lookup:

```python title="first_loop_stop.py"
from contextlib import closing

from looplet import DefaultState, LoopConfig, RunResult, composable_loop, tools_from

from first_loop import done, lookup_owner, scripted_backend


if __name__ == "__main__":
    state = DefaultState(max_steps=2)
    backend = scripted_backend()
    with closing(composable_loop(
        llm=backend,
        tools=tools_from([lookup_owner, done], validate=True),
        state=state,
        config=LoopConfig(max_steps=2, use_native_tools=False),
        task={"goal": "Find the payments owner, then finish."},
    )) as steps:
        for step in steps:
            print(step.tool_call.tool, step.tool_result.data)
            break

    print(RunResult.from_state(state).termination_reason)
```

```bash
python first_loop_stop.py
```

Only the payments lookup occurs.
The `break` statement stops your iteration.
`closing` closes the loop and runs its end hooks.
The last line prints `stream_closed`.
The `done` call does not occur.

Your loop receives a `Step` after each tool call.
The tool has already run when you receive this object.

| Field | Meaning |
| --- | --- |
| `step.tool_call.tool` | The tool name |
| `step.tool_call.args` | The arguments that the model supplied |
| `step.tool_result.data` | The data that the tool returned |
| `step.tool_result.error` | The error, if the tool call failed |

Use a [hook](hooks.md) to approve or reject a call before the tool runs.

Close resources that your code creates, such as database clients.
For async code, use `async_composable_loop()` with `contextlib.aclosing`.
See the [Python API](api.md#execution-core) for that form.

## Choose the next guide

Keep running this agent with Python and testing it with pytest.
Use the [same Python factory](#use-the-same-factory-in-the-shell) for shell run, eval, and replay.
Choose [cartridges](cartridge.md) for a reviewable agent directory.
You do not need a skill bundle for that transition.
See the [CLI guide](cli.md#choose-an-agent-format) for the Python factory workflow.
Add the other features below only when your agent needs them.

| Goal | Guide |
| --- | --- |
| Use a real model | [Install and configure](install.md#configure-openai) |
| Change the loop's decisions | [Hooks](hooks.md) |
| Save and replay a run | [Capture and replay](provenance.md) |
| Check the result | [Tests and graders](evals.md) |
| Run the full regression example | [Regression example](regression-demo.md) |
| Put an agent in a directory | [Cartridges](cartridge.md) |
| Add a file-based test case | [Report-agent tutorial](tutorial.md) |
| Replace a loop that you have | [Migration guide](migrate.md) |
