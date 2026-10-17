# CLI reference

The `looplet` command exposes the same harness, evidence, and eval surfaces as
the Python package. Use `python -m looplet` when an environment does not place
console scripts on `PATH`.

```bash
looplet --help
looplet --help-all
looplet <command> --help
```

## Choose an agent format

Python agents do not need packaging: use `python first_loop.py` and `python -m pytest test_first_loop.py`.
See the [Quickstart](quickstart.md) for both files.
The `run` command below loads packaged agents, not arbitrary Python scripts.

There are two runnable file formats:

- a **cartridge** is a reviewable agent harness containing prompts, tools,
  hooks, resources, and optional evals;
- a **skill bundle** is an executable packaged skill with a Python entrypoint.

Choose a cartridge when you need shell replay or shipped CLI eval cases.
Choose a skill bundle when you need its Python packaging and entrypoint.
Both use the same underlying loop, but their shell workflows differ:

| Workflow | Python agent | Cartridge | Skill bundle |
| --- | --- | --- | --- |
| Create | Write Python | `looplet new` | Write a bundle entrypoint |
| Run | `python first_loop.py` | `looplet run` | `looplet run` |
| Inspect declarations | Read Python | `looplet inspect` | `looplet inspect` |
| Replay captured responses | `replay_loop()` in Python | `looplet replay` | [`replay_loop()` in Python](#replay-skill-bundles) |
| Run behavioral tests | `python -m pytest` | `looplet eval run` | `python -m pytest` |

Bundle replay requires Python; `eval run` requires cartridge cases.
These are format limits, not extra flags to enable on a bundle.

Use `new`, `run`, `inspect`, and `eval`; add `replay` when comparing captured responses. `run` detects either
format; shell `replay` and `eval run` accept cartridges only.
`inspect` defaults to cartridge declarations, bundle metadata, or a saved
trace without importing authored agent code. The formats remain distinct.
Default help shows the main workflow. `--help-all` reveals advanced
commands and compatibility aliases; their individual help remains available.
`run-cartridge` / `run-workspace`, `run-bundle`, `describe`, and `show` remain
available when a script needs an explicit format.

Unified `run`, `inspect`, `run-bundle`, and `new --offline` are included in Looplet `0.4.0`.
Use the [recommended install](install.md#install-the-current-build) for later fixes described in this guide.
Add the provider extra when using a real model.

## Diagnose and inspect runs

### `looplet --version`

Print the installed package version without selecting or contacting a provider:

```bash
looplet --version
python -m looplet --version
```

This flag is available in unreleased builds.
For published `0.4.0`, use `looplet doctor --no-backend` instead.

### `looplet doctor`

Check Python, package version, the selected provider, and native-tool support.

```bash
looplet doctor
looplet doctor --no-backend
looplet doctor --json
looplet doctor --strict
```

`--no-backend` performs no provider call. `--strict` makes warnings produce a
non-zero exit, which is useful for CI configuration checks. Live commands
use the same provider resolver as Python: a cloud key, a local compatible URL,
or an Anthropic key is enough. Models use provider defaults unless configured;
set `LOOPLET_PROVIDER` when multiple providers are available. See
[install and configure](install.md).

Automatic selection warns when Anthropic and OpenAI settings are both present.
The existing Anthropic-first precedence remains unchanged.
`doctor` reports a `provider_selection` warning; `--strict` fails until you choose a provider explicitly.

The `endpoint` check shows the effective scheme, host, port, and matching configuration variable or provider client default.
It displays only the origin; user info, paths, queries, and fragments are omitted because they may contain credentials.
This identifies the configured destination without claiming connectivity when `--no-backend` is used.

In unreleased builds, the `build` check fingerprints library Python sources without requiring Git.
It excludes bundled examples and bytecode caches.
Compare the version and fingerprint together; same-version local builds may contain different source code.
This fingerprint is not an official release identifier or a security attestation.

### `looplet inspect <path>`

Inspect a cartridge, skill bundle, or saved trace without running it:

```bash
looplet inspect ./agent.cartridge --json
looplet inspect ./skills/code-review --json
looplet inspect traces/incident-42 --json
looplet inspect ./agent.cartridge --mode runtime --json
looplet inspect ./skills/code-review --mode runtime --workspace . --json
```

Cartridge output is declaration-only (`runtime_validated: false`). Bundle
output contains metadata and metadata errors, not a loaded blueprint. Trace
output preserves the `show` JSON contract, and rejects runtime mode.
`--mode declaration` is the safe default. `--mode runtime` explicitly imports
authored code, builds one preset, validates its executable contract, and closes
its resources. Reports label `mode` and `runtime_validated`; invalid wiring
returns nonzero. This is structural validation, not a behavioral eval.
Cartridge declaration content remains redacted unless `--include-sensitive`
is passed. Runtime bundle blueprints may include authored instructions/config;
review them before sharing. Harness diagnostics go to stderr in JSON mode.

`describe` also accepts `--mode declaration|runtime`, including explicit runtime
JSON; legacy `--instantiate --json` remains mutually exclusive. `blueprint`
accepts the same `--mode` vocabulary but retains its legacy live-building
default. Its explicit runtime mode validates wiring; the historical default
only builds and describes it. Legacy flags remain supported and conflicting
mode choices are rejected.

### `looplet show <trace-dir>`

Print a one-page summary of `trajectory.json` and `manifest.jsonl`, including
steps, failures, LLM-call counts, and timing when recorded.

```bash
looplet show traces/incident-42
looplet show traces/incident-42 --json | jq '.trajectory.termination_reason'
```

The command returns non-zero for a missing, malformed, or empty trace
directory. JSON output contains the parsed `trajectory.json` object and
`manifest.jsonl` records; consumers should tolerate added fields. Read
[saved artifacts](artifacts.md) for the complete layout.
Missing or null timings do not contribute to human timing totals; JSON output
and the stored evidence retain their original values.

Unreleased builds also print the complete recorded result under `result:` for completed runs.
Rejected completions and incomplete runs have no final-result section.
Both `show` and trace `inspect` use this view.
Plain-trace JSON output and saved files are unchanged.
Inspect traces before sharing them; complete results may contain sensitive data.

Unreleased builds also show recorded grades under `saved evals:` and collected evidence under `artifacts:` for saved eval runs.
Their JSON output adds `evals` and `artifacts`; ordinary traces retain the existing JSON shape.
Inspection does not rerun graders, collectors, tools, or models.
The `done` header means loop completion, not correctness.
Inspection preserves recorded scores and does not apply a release threshold.
A readable failed eval still exits zero from inspection.
Use `looplet eval <trace-dir> --evals ./evals/ --threshold 0.7` to rescore with your gate policy.
Saved grades and artifacts may include sensitive data; review both before sharing.

### Replay captured responses

This complete example needs no model service or API key after package installation:

```bash
looplet new "Finish with a short summary" ./replay-demo.cartridge --offline
looplet run ./replay-demo.cartridge "Say hello" --project-root . \
  --trace-dir ./original-trace \
  --scripted-response '{"tool":"done","args":{"summary":"Hello"}}' \
  --require-completed
looplet inspect ./original-trace
looplet replay ./original-trace --cartridge ./replay-demo.cartridge \
  --workspace ./replayed-workspace --require-completed --json
```

Choose unused paths when repeating this example; existing evidence and replay workspaces are refused.
The final command uses recorded model responses and writes a separate trace under the new workspace's `.looplet/traces/` directory.
Its JSON and `--require-completed` behavior match `run`.
It does not grade correctness or modify the original capture.

Raw captures do not contain a snapshot of the original workspace. Supply a reviewed starting input tree explicitly when tools need files:

```bash
looplet replay ./original-trace --cartridge ./agent.cartridge \
  --workspace ./seeded-replay-workspace --seed-from ./starting-inputs
```

The new workspace must be outside the seed directory. Nested files, binary data, empty directories, and file permissions are copied without modifying the inputs.
Git metadata, `.looplet` directories, and the selected trace and cartridge trees are excluded.
Symlinks and special files in the included tree are rejected before cartridge loading.
Choose agent-visible inputs only; this option does not reconstruct historical files or isolate tool effects.
Saved eval cases already supply their seed files and reject `--seed-from` rather than mixing input sources.

For a saved eval case, pass its individual case directory instead of a parent run directory:

```bash
looplet replay eval-runs/run-001/hello --cartridge ./agent.cartridge \
  --workspace ./replay-workspaces/hello-001 --require-completed
```

Saved cases supply agent-visible task data and seed files; grader-only expectations are not forwarded to the agent.
Replay reruns the selected cartridge's collectors against the new workspace.
It saves a new eval record with those fresh artifacts, the saved case, and grader-only expectations.
It does not copy old collected artifacts or run graders automatically.
Run the printed `Grade:` command to assess the new record with the cartridge's graders.
With `--json`, use the returned `trace_dir` as the input to `looplet eval <trace_dir> --evals ./agent.cartridge/evals --json`.
The new trajectory records `parent_run_id` and `source_case_id`; raw traces with a trajectory record only the parent link.
Saved expectations remain private grading evidence, not a protected oracle against tools with unrestricted filesystem access.
Raw traces reuse agent-visible task data from `trajectory.json` when available.
Pass `--task` only when no task was saved or when replacing the saved task with a new goal.
Replay defaults to the recorded tool-step count, or captured response count when step metadata is absent.
This preserves budget-stopped captures without requesting extra responses by default and accounts for native multi-tool turns.
`--max-steps` overrides this replay budget; a larger value can exhaust the recorded responses.
Use only trusted cartridges: loading imports authored code, and tools, hooks, networks, and other effects execute again.
A fresh directory is not an operating-system sandbox, checkpoint restoration, or a guarantee of identical results.
Authored tools and resources may contact services even though replay does not construct the original provider backend.

### Replay skill bundles

Shell `replay --cartridge` does not accept skill bundles, even though `run` and `inspect` support both formats.
Use the public Python replay function with a fresh bundle preset instead:

```python
from pathlib import Path

from looplet import RunResult, replay_loop
from looplet.bundles import SkillRuntime, load_skill_bundle

workspace = Path("replay-workspaces/bundle-001")
workspace.mkdir(parents=True, mode=0o700)
bundle = load_skill_bundle("skills/coder")
runtime = SkillRuntime(
  workspace=workspace,
  max_steps=20,
  options={"require_tests": False, "use_native_tools": True},
)
with bundle.import_context(), bundle.build_preset(runtime) as preset:
  for step in replay_loop(
    "bundle-trace",
    tools=preset.tools,
    state=preset.state,
    hooks=preset.hooks,
    config=preset.config,
    task={"description": "Review this repository"},
  ):
    print(step.pretty())
  result = RunResult.from_state(preset.state)
  if not result.completed:
    raise SystemExit(result.termination_reason)
  print(result.output)
```

Replace the bundle path, capture path, task, and bundle-specific options with those from your original run.
Match the captured tool protocol; this example uses native tools and disables the coder bundle's test-completion gate.
The recipe runs fresh tools and hooks; it does not automatically save replay evidence or grade correctness.
Use only trusted bundle code inside an appropriate operating-system sandbox.

## Build and run cartridges

### `looplet new <description> [target]`

Use the configured provider to scaffold a cartridge draft from a brief.
Generated files are a starting point, not a release-ready agent.

```bash
looplet new \
  "Inspect a repository and report dependency risks" \
  ./dependency-review.cartridge \
  --tool read_file \
  --tool run_tests
```

Live creation accepts the same model and endpoint overrides as `run` and `eval run`:

```bash
looplet new "Inspect local inputs" ./local-agent.cartridge \
  --model local-model --base-url http://localhost:11434/v1
```

For a draft with placeholder tools and no provider or API key:

```bash
looplet new "Look up service owners" ./owner.cartridge --offline --tool lookup_owner
looplet inspect ./owner.cartridge
```

Offline creation refuses a nonempty destination. Implement the placeholder
tool code before running it. Useful options include repeatable `--tool`,
`--name`, `--max-steps`, `--quiet`, and `--pretty`. `--max-steps` bounds the
live factory's work, not the generated agent's budget; offline creation does
not run the factory. Combining `--offline` with `--max-steps` fails before creating files.
Set `max_steps` in the generated `config.yaml` to limit the agent itself.
Review the prompt, schemas, implementations, and runtime
policy, then add an outcome contract before release.

The scaffold command prints the prompt and tool paths to edit next.
Inspection lists generated tool stubs under `unfinished_tools` and labels them `readiness: draft`.
Runtime wiring can be valid while these stubs remain unfinished.
`readiness: not_assessed` means no scaffold stub was found; it does not certify behavior.

### `looplet run <path> <task>`

Load a cartridge and run one task:

```bash
looplet run ./agent.cartridge \
  "Inspect the current change" \
  --project-root . \
  --max-steps 20 \
  --require-completed
```

Use `--require-completed` in automation so an incomplete task returns nonzero.
Without this flag, exit code `0` means the loop ran, not that the task completed.
Budget exhaustion and hook stops can return `0` with `completed: false`.
Tool errors can also exhaust the budget without accepted completion.
Accepted completion still does not prove correctness; use an outcome grader for that.

In unreleased builds, `--max-steps` overrides the charged-step budget for `DefaultState`.
Successful nonterminal free calls do not consume it; total tool and model calls can exceed this value.
See [bound free-tool runs](operations.md#bound-free-tool-runs) for a cooperative deadline.

Live `new`, `run`, its explicit cartridge and bundle aliases, and `eval run` accept the same provider options:

```bash
looplet run ./agent.cartridge "Inspect the current change" \
  --model local-model --base-url http://localhost:11434/v1 --require-completed
```

`--model` overrides the selected provider's model.
`--base-url` explicitly selects an OpenAI-compatible endpoint, even when another provider is configured.
Omitted options keep the existing environment-based defaults; overrides do not change environment variables.
Bundle runs with explicit provider options use the shared preset runner instead of a legacy presentation callback.
Scripted runs do not construct a provider backend.

Use `-` as the task to read it from standard input:

```bash
git diff | looplet run ./review.cartridge - --project-root . --require-completed
```

Link a follow-up run to an earlier trace without changing execution:

```bash
looplet run ./repair.cartridge "Repair the finding" \
  --parent-trace .looplet/traces/review-a1b2c3 --require-completed
```

Emit one completion object for a shell pipeline:

```bash
looplet run ./agent.cartridge "Inspect the change" --require-completed --json \
  | jq -e '.completed'
```

Require accepted completion directly in the command's exit status:

```bash
looplet run ./agent.cartridge "Inspect the change" --require-completed --json > completion.json
```

`--require-completed` returns nonzero for budget exhaustion, hook stops, or other incomplete results.
It works in human and JSON modes, including the explicit cartridge and bundle commands.
It checks completion, not correctness; use an outcome grader for that.
The flag is included in Looplet `0.4.0`.

Unreleased builds reject failed terminal dispatches, including invalid `done` arguments.
Published `0.4.0` can falsely report those calls as completed, even with `--require-completed`.
See the [unreleased safety fixes](install.md#unreleased-safety-fixes).

Unreleased cartridge and generic bundle runners show the last recorded error when a run does not complete.
Their JSON completion object adds a nullable `error` field with `step`, `tool`, and `message`.
Messages are limited to 2,000 characters; full error evidence remains in the trace.
The field excludes tool arguments and result payloads.
It is `null` after accepted completion or when no error was recorded.
Review error messages before sharing output; tool-authored text can contain sensitive data.
Custom bundle `run()` callbacks still own their human presentation.
JSON mode, strict completion, or explicit provider options select the generic bundle runner.

`--project-root` (`--workspace`, `-w`) controls the directory available to
project-aware tools. It
defaults to `LOOPLET_PROJECT_ROOT`, the current Git repository, or the current
directory, and must already exist. Each run also writes a provenance trace under
`.looplet/traces/<cartridge>-<id>/` in that project root. Pass `--trace-dir`
to choose an explicit location or `--no-trace` to disable capture.
An existing trace returns an error without replacing evidence or loading the cartridge.
Choose a new `--trace-dir`, or omit it for an automatic unique path.
Traces may
contain full prompts, responses, and tool results; inspect them before sharing.
`--parent-trace` reads the parent's `run_id` and records it as
`metadata.parent_run_id`; it does not resume, replay, or orchestrate that run.
`--json` suppresses human progress output and emits `completed`,
`termination_reason`, `steps`, `duration_ms`, the terminal `result`, and
`trace_dir` (`null` with `--no-trace`). `completed` is true only when the agent
reaches its configured terminal tool and the completion is accepted; budget
and hook stops remain successful CLI executions but are
reported as incomplete unless `--require-completed` is set.
Human output likewise prints `stopped (<reason>)`
instead of `done` for incomplete runs. Fatal execution failures, including
provider errors, return a non-zero exit code in both human and JSON modes.
In JSON mode, ordinary Python output from loaded harness code goes to standard
error instead of corrupting the completion object. That output is not redacted.
Errors remain on standard error; consumers should
tolerate added fields. Cartridge execution through either `run` or
`run-cartridge` accepts `--quiet`, `--pretty`, and `--parent-trace`; these
options are rejected for bundles before loading authored code.
`--json` cannot be combined with `--pretty`.
`run-cartridge` and its `run-workspace` alias retain their existing behavior.
`run-bundle` forces bundle routing. If a directory has both `cartridge.json`
and `SKILL.md`, `run` refuses the ambiguity; choose the explicit command.

### Review commands

| Command | Purpose |
| --- | --- |
| `looplet describe <cartridge>` | Print tools, hooks, config, and a prompt preview. |
| `looplet diff <before> <after> [--show]` | Group changes by manifest, config, runtime, prompt, tool, hook, resource, memory, or eval. |
| `looplet hash <cartridge> [--show-files]` | Hash every versioned regular file except documented runtime/cache exclusions. |
| `looplet portability <cartridge> [--json]` | Classify protocol, standard-library, runtime, and Python-host dependencies. |
| `looplet conform [fixtures] [-v]` | Run Cartridge Spec conformance fixtures against the loader. |
| `looplet migrate <cartridge> [--dry-run]` | Upgrade a v1 cartridge to schema version 2. |

Use `diff` in review, `hash` in deployment metadata, and `portability
--require-portable` as a CI gate when a protocol-portable cartridge is a hard
requirement. That gate exits with code 2 when the profile is not portable.
Always run `migrate --dry-run` first and review the resulting files.

## Run behavioral evals

### Check Python definitions and run Python tests

Use the agent and tests from the [Quickstart](quickstart.md#3-check-the-answer):

```bash
looplet eval check first_loop.py --json
python -m pytest test_first_loop.py
```

`eval check` imports trusted definitions and validates their shape; it does not run an agent.
`eval init --python evals/` creates plain grader, collector, case, and pytest files.
Run the generated test with `python -m pytest evals/`.
Omit `--python` to create definitions only.
Use `load_evals("evals")` in your Python test to load those definitions.
`eval run evals/` is not the next command: the runner below requires a cartridge.

### Run shipped cartridge cases

```bash
looplet eval run ./agent.cartridge \
  --threshold 1.0 \
  --json | jq -e '.passed'
```

Each invocation saves cases under a unique `.looplet/evals/<cartridge>-<id>/` directory in the current directory.
Human output prints that directory; JSON includes it as `output_dir`.
Human output also prints a shell-quoted `Inspect:` command for each saved case.
Use those commands directly; the parent `output_dir` groups cases and is not an individual trace.
Failed grades retain the same evidence as passing grades.
Use `--out` to choose another directory, or `--no-save` to disable persistence explicitly.
`--out` and `--no-save` cannot be combined.

Notable options:

| Option | Effect |
| --- | --- |
| `--case ID` | Run one case; repeat to select several. |
| `--case-mark MARK` | Run cases with any selected mark; repeat to select several marks. Unreleased. |
| `--max-steps N` | Override the per-case charged-step budget. Successful free calls do not count in unreleased builds. |
| `--model NAME` | Override the selected provider's model. |
| `--base-url URL` | Explicitly select an OpenAI-compatible endpoint for agent and judge. |
| `--judge` | Enable graders whose signature requests an LLM. |
| `--judge-model NAME` | Use a separate judge model and imply `--judge`. |
| `--out DIR` | Choose the parent directory for saved case evidence. |
| `--no-save` | Run without saving eval evidence or captured model calls. |
| `--no-capture` | Save grades and outcomes without agent model prompts or responses. |
| `--threshold VALUE` | Fail when any scored grader falls below the value. |
| `--json` | Emit one report with the overall verdict and per-case grader results. |

Unreleased builds accept the same mark filter when listing cases:

```bash
looplet eval cases ls ./agent.cartridge/evals --case-mark smoke
looplet eval run ./agent.cartridge --case-mark smoke --case-mark regression
```

Repeated marks match any listed value.
Combine `--case` with marks to run their intersection.
Unknown marks and empty selections fail before backend construction.
Case marks select cases, not graders.
Saved-run grading uses `--include` and `--exclude` for grader marks instead.

Required graders, explicit failures, collector errors, malformed records,
unknown cases, and empty required suites also fail the command. Persisted case
runs use the [eval artifact layout](artifacts.md#persisted-eval-run). JSON output
contains the threshold verdict, case completion state, serialized grader
results, integrity failures, and output directory. It deliberately omits
artifacts and grader-only expected data; read those through the persisted run.
Consumers should tolerate added fields.
For an explicit `--out`, choose a fresh directory for each invocation, such as `eval-runs/run-002` next.
Existing selected case paths are refused before any case executes, including with `--no-capture`.
Previous evidence and workspaces are not deleted.
Each saved case owns a wildcard `.gitignore`, including its workspace, so new evidence stays out of ordinary Git staging.
Existing ignore policies are preserved; already tracked files and force-adds remain the host's responsibility.
Human tables size columns for grader names and values and separate them with visible delimiters.
Ordinary Python diagnostics from loaded graders, collectors, and harness code
go to stderr in JSON mode and are not redacted.

### Grade saved trajectories

Grade one saved run, using the same path that `inspect` accepts:

```bash
looplet eval traces/incident-42 --evals eval_agent.py --json
```

Or grade the immediate child runs of a directory:

```bash
looplet eval traces/ \
  --evals eval_agent.py \
  --include required smoke \
  --threshold 1.0 \
  --verbose
```

Use `--exclude slow` to omit marked graders. This path grades existing traces;
it does not execute a cartridge or make new agent model calls unless a grader
requests a judge backend supplied by the host.
Raw capture does not include grader-only expectations or collected artifacts.
Use `save_eval_run()` when preserving those inputs; see the [Quickstart](quickstart.md#4-capture-the-same-agent).

### Browse case data

```bash
looplet eval cases ls evals/
looplet eval cases show evals/ regression_42
```

Both commands accept an eval root, a direct case directory, or a JSON/JSONL file without importing graders.
Listings preview `task.description`, then `task.goal`, then case notes.
Cases are JSON source data. Keep agent-visible task input under `task` and
protected expectations in the top-level `expected` object.

## Work with skill bundles

### Bundle-specific execution

Run a packaged skill bundle. Provenance capture is enabled by default:

```bash
looplet run ./skills/code-review \
  "Review this repository" \
  --project-root . \
  --trace-dir traces/code-review
```

Use `--scripted` for bundle-provided deterministic responses, repeat
`--scripted-response` to supply responses directly, or `--no-trace` when the
host deliberately disables capture. Stdin tasks, `--json`, existing-directory
validation, stop reasons, and completion fields match cartridge execution.
`--no-tests` passes `require_tests=False` to bundles that support it and is
rejected for cartridges. Generic bundle runs default to 20 steps; cartridge
runs retain their configured budget unless `--max-steps` overrides it.
Run help groups bundle-only `--scripted` and `--no-tests` separately from cartridge-only options.
For cartridges, use repeatable `--scripted-response` with explicit responses instead of `--scripted`.

A bundle's custom `run(...) -> int` and renderers still own human execution
when provided. With `--json` or `--require-completed`, the CLI instead executes the required
`build(runtime) -> AgentPreset` contract and emits the shared completion
result in the selected output format.
This lets the CLI observe completion instead of trusting a custom callback's exit code.
Without those flags, custom runners remain responsible for their own status and effects.

### Bundle inspection and packaging

| Command | Purpose |
| --- | --- |
| `looplet list-bundles <roots...> [--json]` | Discover runnable bundles under one or more roots. |
| `looplet blueprint <bundle>` | Print the loaded bundle blueprint as JSON. |
| `looplet export-code <bundle> <file>` | Export exact Python wrapper code for a bundle. |
| `looplet package <module:factory> <dir>` | Package an importable `AgentPreset` factory as a bundle. |
| `looplet wrap-claude-skill <skill> <dir>` | Wrap a Claude or Agent Skills directory as a Looplet bundle. |

`blueprint` retains live inspection as its default: it imports authored code,
builds a temporary preset, and closes its resources. `--instantiate` makes this
mode explicit. Use `--declaration-only` to avoid authored imports and resource
construction:

```bash
looplet blueprint ./skills/code-review --declaration-only
looplet blueprint ./skills/code-review --instantiate --workspace .
```

The declaration view reports `runtime_validated: false` and lists unresolved
runtime components; it is incomplete and does not establish behavioral
equivalence. `--workspace` supplies the live factory's working directory, not
the bundle or cartridge path.

Blueprint JSON remains on stdout; ordinary Python diagnostics from live import,
build, and resource cleanup go to stderr and are not redacted.

Packaging requires `--name` and `--description`; repeat `--tag` to attach
searchable tags. Validate and run the output before distributing it.

## Automation conventions

- Exit code 0 means command success, not agent completion or a correct answer.
- Use `run --require-completed` to require accepted completion, or check `completed=true` in JSON output.
- Use a passing eval to check correctness.
- Use `--json` only on commands that advertise it; do not parse human output.
- Pin the Looplet minor version when a script depends on output fields.
- Capture stdout, stderr, command arguments, package version, and artifact hash
  in release automation.
- Keep API keys out of command arguments and persisted logs.
- Prefer the Python API when the host needs structured objects that a command
  does not expose as JSON.

Before `1.0`, command options and machine-readable fields may change in a minor
release. For PyPI installations, pin `looplet>=0.4,<0.5`.
Review the [changelog](changelog.md) before upgrading.
