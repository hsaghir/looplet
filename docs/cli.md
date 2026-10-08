# CLI reference

The `looplet` command exposes the same harness, evidence, and eval surfaces as
the Python package. Use `python -m looplet` when an environment does not place
console scripts on `PATH`.

```bash
looplet --help
looplet <command> --help
```

There are two runnable file formats:

- a **cartridge** is a reviewable agent harness containing prompts, tools,
  hooks, resources, and optional evals;
- a **skill bundle** is an executable packaged skill with a Python entrypoint.

Use one workflow: `new`, `run`, `inspect`, then `eval`. `run` detects either
format; `inspect` reads cartridge declarations, bundle metadata, or a saved
trace without importing authored agent code. The formats remain distinct.
`run-cartridge` / `run-workspace`, `run-bundle`, `describe`, and `show` remain
available when a script needs an explicit format.

Unified `run`, `inspect`, `run-bundle`, and `new --offline` are unreleased
source additions, not part of PyPI `0.4.0`. Install this checkout with
`pip install -e ".[openai]"` (or the appropriate provider extra) to use them.

## Diagnose and inspect runs

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

### `looplet inspect <path>`

Inspect a cartridge, skill bundle, or saved trace without running it:

```bash
looplet inspect ./agent.cartridge --json
looplet inspect ./skills/code-review --json
looplet inspect traces/incident-42 --json
```

Cartridge output is declaration-only (`runtime_validated: false`). Bundle
output contains metadata and metadata errors, not a loaded blueprint. Trace
output preserves the `show` JSON contract. Use `blueprint` or a run when
runtime validation is required; those operations import authored Python code.

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

For a draft with placeholder tools and no provider or API key:

```bash
looplet new "Look up service owners" ./owner.cartridge --offline --tool lookup_owner
looplet inspect ./owner.cartridge
```

Offline creation refuses a nonempty destination. Implement the placeholder
tool code before running it. Useful options include repeatable `--tool`,
`--name`, `--max-steps`, `--quiet`, and `--pretty`. `--max-steps` bounds the
live factory's work, not the generated agent's budget; offline creation does
not run the factory. Review the prompt, schemas, implementations, and runtime
policy, then add an outcome contract before release.

### `looplet run <path> <task>`

Load a cartridge and run one task:

```bash
looplet run ./agent.cartridge \
  "Inspect the current change" \
  --project-root . \
  --max-steps 20
```

Use `-` as the task to read it from standard input:

```bash
git diff | looplet run ./review.cartridge - --project-root .
```

Link a follow-up run to an earlier trace without changing execution:

```bash
looplet run-cartridge ./repair.cartridge "Repair the finding" \
  --parent-trace .looplet/traces/review-a1b2c3
```

Emit one completion object for a shell pipeline:

```bash
looplet run ./agent.cartridge "Inspect the change" --json \
  | jq -e '.completed'
```

`--project-root` (`--workspace`, `-w`) controls the directory available to
project-aware tools. It
defaults to `LOOPLET_PROJECT_ROOT`, the current Git repository, or the current
directory, and must already exist. Each run also writes a provenance trace under
`.looplet/traces/<cartridge>-<id>/` in that project root. Pass `--trace-dir`
to choose an explicit location or `--no-trace` to disable capture. Traces may
contain full prompts, responses, and tool results; inspect them before sharing.
`--parent-trace` reads the parent's `run_id` and records it as
`metadata.parent_run_id`; it does not resume, replay, or orchestrate that run.
`--json` suppresses human progress output and emits `completed`,
`termination_reason`, `steps`, `duration_ms`, the terminal `result`, and
`trace_dir` (`null` with `--no-trace`). `completed` is true only when the agent
reaches its configured terminal tool and the completion is accepted; budget
and hook stops remain successful CLI executions but are
reported as incomplete. Human output likewise prints `stopped (<reason>)`
instead of `done` for incomplete runs. Fatal execution failures, including
provider errors, return a non-zero exit code in both human and JSON modes.
Errors remain on standard error; consumers should
tolerate added fields. The explicit cartridge command retains `--quiet`,
`--pretty`, and `--parent-trace`; `--json` cannot be combined with `--pretty`.
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

### Run shipped cartridge cases

```bash
looplet eval run ./agent.cartridge \
  --out ./eval-runs \
  --threshold 1.0 \
  --json | jq -e '.passed'
```

Notable options:

| Option | Effect |
| --- | --- |
| `--case ID` | Run one case; repeat to select several. |
| `--max-steps N` | Override the per-case tool-call budget. |
| `--model NAME` | Override the selected provider's model. |
| `--base-url URL` | Explicitly select an OpenAI-compatible endpoint for agent and judge. |
| `--judge` | Enable graders whose signature requests an LLM. |
| `--judge-model NAME` | Use a separate judge model and imply `--judge`. |
| `--out DIR` | Persist each case under `DIR/<case-id>/`. |
| `--threshold VALUE` | Fail when any scored grader falls below the value. |
| `--json` | Emit one report with the overall verdict and per-case grader results. |

Required graders, explicit failures, collector errors, malformed records,
unknown cases, and empty required suites also fail the command. Persisted case
runs use the [eval artifact layout](artifacts.md#persisted-eval-run). JSON output
contains the threshold verdict, case completion state, serialized grader
results, integrity failures, and output directory. It deliberately omits
artifacts and grader-only expected data; read those through the persisted run.
Consumers should tolerate added fields.

### Grade saved trajectories

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

### Browse case data

```bash
looplet eval cases ls evals/cases/
looplet eval cases show evals/cases/ regression_42
```

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

A bundle's custom `run(...) -> int` and renderers still own human execution
when provided. In `--json` mode, the CLI instead executes the required
`build(runtime) -> AgentPreset` contract and emits the shared completion
object. Custom runners remain responsible for their own status and effects.

### Bundle inspection and packaging

| Command | Purpose |
| --- | --- |
| `looplet list-bundles <roots...> [--json]` | Discover runnable bundles under one or more roots. |
| `looplet blueprint <bundle>` | Print the loaded bundle blueprint as JSON. |
| `looplet export-code <bundle> <file>` | Export exact Python wrapper code for a bundle. |
| `looplet package <module:factory> <dir>` | Package an importable `AgentPreset` factory as a bundle. |
| `looplet wrap-claude-skill <skill> <dir>` | Wrap a Claude or Agent Skills directory as a Looplet bundle. |

Packaging requires `--name` and `--description`; repeat `--tag` to attach
searchable tags. Validate and run the output before distributing it.

## Automation conventions

- Treat exit code 0 as success and any non-zero code as a failed operation.
- Use `--json` only on commands that advertise it; do not parse human output.
- Pin the Looplet minor version when a script depends on output fields.
- Capture stdout, stderr, command arguments, package version, and artifact hash
  in release automation.
- Keep API keys out of command arguments and persisted logs.
- Prefer the Python API when the host needs structured objects that a command
  does not expose as JSON.

Before `1.0`, command options and machine-readable fields may change in a minor
release. Pin `looplet>=0.4,<0.5` and review the [changelog](changelog.md) when
upgrading.
