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

Use `run-cartridge` for the first and `run` for the second. The formats serve
different jobs and the commands are not aliases.

## Diagnose and inspect runs

### `looplet doctor`

Check Python, package version, provider environment, and OpenAI-compatible
native-tool support.

```bash
looplet doctor
looplet doctor --no-backend
looplet doctor --json
looplet doctor --strict
```

`--no-backend` performs no provider call. `--strict` makes warnings produce a
non-zero exit, which is useful for CI configuration checks. The doctor command
currently probes OpenAI-compatible configuration; see
[install and configure](install.md) for Anthropic smoke testing.

### `looplet show <trace-dir>`

Print a one-page summary of `trajectory.json` and `manifest.jsonl`, including
steps, failures, LLM-call counts, and timing when recorded.

```bash
looplet show traces/incident-42
```

The command returns non-zero for a missing, malformed, or empty trace
directory. Read [saved artifacts](artifacts.md) for the complete layout.

## Build and run cartridges

### `looplet new <description> [target]`

Use an OpenAI-compatible backend to scaffold a cartridge draft from a brief.
Generated files are a starting point, not a release-ready agent.

```bash
looplet new \
  "Inspect a repository and report dependency risks" \
  ./dependency-review.cartridge \
  --tool read_file \
  --tool run_tests
```

Useful options include repeatable `--tool`, `--name`, `--max-steps`, `--quiet`,
and `--pretty`. Review the generated prompt, schemas, implementations, and
runtime policy, then add an outcome contract before release.

### `looplet run-cartridge <cartridge> <task>`

Load a cartridge and run one task:

```bash
looplet run-cartridge ./agent.cartridge \
  "Inspect the current change" \
  --project-root . \
  --max-steps 20
```

`--project-root` controls the directory available to project-aware tools. It
defaults to `LOOPLET_PROJECT_ROOT`, the current Git repository, or the current
directory. `run-workspace` remains a compatibility alias.

### Review commands

| Command | Purpose |
| --- | --- |
| `looplet describe <cartridge>` | Print tools, hooks, config, and a prompt preview. |
| `looplet diff <before> <after> [--show]` | Group cartridge changes by prompt, tool, hook, resource, or config. |
| `looplet hash <cartridge> [--show-files]` | Compute a stable SHA-256 hash over content-bearing harness files. |
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
  --threshold 1.0
```

Notable options:

| Option | Effect |
| --- | --- |
| `--case ID` | Run one case; repeat to select several. |
| `--max-steps N` | Override the per-case tool-call budget. |
| `--model NAME` / `--base-url URL` | Override OpenAI-compatible environment configuration. |
| `--judge` | Enable graders whose signature requests an LLM. |
| `--judge-model NAME` | Use a separate judge model and imply `--judge`. |
| `--out DIR` | Persist each case under `DIR/<case-id>/`. |
| `--threshold VALUE` | Fail when any scored grader falls below the value. |

Required graders, explicit failures, collector errors, malformed records,
unknown cases, and empty required suites also fail the command. Persisted case
runs use the [eval artifact layout](artifacts.md#persisted-eval-run).

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

### `looplet run <bundle> <task>`

Run a packaged skill bundle. Provenance capture is enabled by default:

```bash
looplet run ./skills/code-review \
  "Review this repository" \
  --workspace . \
  --trace-dir traces/code-review
```

Use `--scripted` for bundle-provided deterministic responses, repeat
`--scripted-response` to supply responses directly, or `--no-trace` when the
host deliberately disables capture.

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
release. Pin `looplet>=0.3,<0.4` and review the [changelog](changelog.md) when
upgrading.
