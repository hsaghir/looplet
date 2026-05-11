# Cartridge Spec v1.0

> **Status.** Reference implementation in this repository (Looplet).
> Companion artifact: [`cartridge.schema.json`](cartridge.schema.json).
> **Audience.** Loader implementers in any language; agent builders.
> **Versioning.** Semantic. v1.x is additive. Slot renames or breaking
> shape changes require a v2 RFC.

A **cartridge** (a.k.a. *workspace*) is a small directory of files
that fully describes an LLM agent's behavioural contract. The
runtime that loads and runs a cartridge is interchangeable. The
cartridge is the artifact; the runtime is the engine.

This document specifies what files a v1 cartridge contains, what
each file means, and what a conformant loader must do with them.

## Core principles

1. **The cartridge is complete.** Loading the cartridge is sufficient
   to reproduce the agent's surface. No hidden registries, no
   runtime patches, no environment-only state.
2. **The cartridge is runtime-agnostic.** No file in the cartridge
   names a particular runtime. Tool bodies are ordinary functions in
   the host language; hooks are ordinary classes implementing a small
   protocol. The runtime binds them.
3. **Every editable concern has a fixed location.** Reviewers,
   admission gates, and diff tools can determine the *category* of an
   edit from the *path* alone, without reading file contents.
4. **Slots may be empty, never ambiguous.** A loader sees either a
   recognised file at a known path or nothing.

## Layout (canonical)

```
my_agent.workspace/
├── workspace.json              # required: name, schema_version
├── config.yaml                 # required: loop config + declarative slots
├── prompts/
│   └── system.md               # required: the system prompt, alone
├── tools/
│   └── <name>/
│       ├── tool.yaml           # required per tool
│       └── execute.py          # required per tool (host language body)
├── hooks/                      # optional
│   └── NN_<name>/
│       ├── config.yaml         # optional kwargs for the hook class
│       └── hook.py             # required when the hook is local code
├── resources/                  # optional
│   └── <name>.py               # optional: shared singletons (def build())
├── memory/                     # optional
│   ├── long_term.md            # optional: long-term memory (v1.0 slot)
│   └── *.md / *.py             # optional: ordered memory sources
└── setup.py                    # optional: imperative escape hatch
```

The `.workspace` directory suffix is conventional but not load-bearing
(loaders MUST accept any directory containing a valid `workspace.json`).

## Manifest — `workspace.json`

```json
{
  "name": "my_agent",
  "schema_version": 1,
  "description": "what the agent does",
  "version": "1.0.0"
}
```

Required fields: `name`, `schema_version`. v1.0 cartridges set
`schema_version: 1`. `description` and `version` are optional and
declarative.

## Configuration — `config.yaml`

The configuration file declares loop budgets, model binding, slot
references, and inheritance. All fields are optional except as
noted. Loaders MUST accept any v1.0 cartridge with an empty
`config.yaml` (defaults apply).

### Loop budgets

```yaml
max_steps: 20                  # default 15
max_tokens: 2000               # max tokens per LLM call
recovery_temperature: 0.1
context_window: 128000
max_briefing_tokens: 4000      # null = unbounded
use_native_tools: false
concurrent_dispatch: false
reactive_recovery: true
done_tool: done                # name of the completion sentinel tool
```

### Model binding (v1.0 slot)

A structured `model:` block declares the LLM the cartridge is
designed for. Loaders that bind a different model SHOULD warn.

```yaml
model:
  provider: anthropic              # openai | anthropic | azure | …
  name: claude-sonnet-4.6          # provider-specific id
  reasoning_effort: high           # minimal | low | medium | high | xhigh
  max_tokens: 4096
  temperature: 0.2
  top_p: 1.0
  extra:                           # provider-specific pass-through
    cache_control: ephemeral
```

**Backwards compatibility.** Pre-v1.0 cartridges set the flat
`temperature:` and `max_tokens:` at the top level. Loaders MUST
continue to accept the flat form; when both are present, the
`model:` block wins.

### Permissions (v1.0 slot)

A declarative permissions block compiles into a `PermissionEngine`
hook. Tools without an explicit rule fall through to `default`.

```yaml
permissions:
  default: allow                   # allow | deny | ask
  deny:
    - tool: bash
      contains:
        command: "rm -rf"
      reason: "destructive shell"
    - tool: write
      contains:
        file_path: "/etc"
      reason: "system path"
  ask:
    - tool: bash
  allow:
    - read
    - glob
    - grep
```

A bare string entry (e.g. `- read`) is shorthand for `{ tool: read }`.

### Memory (v1.0 slot)

```yaml
memory:
  long_term: memory/long_term.md   # default if file exists
  include:                         # additional ordered files
    - memory/lessons.md
  max_tokens: 1500                 # truncate combined memory
```

When `memory/long_term.md` exists and no `memory:` block is given,
loaders MUST auto-load it as a long-term memory source.

### Inheritance

```yaml
extends: ../base.workspace        # one parent
```

The loader resolves the chain top-down: ancestor first, child wins on
any conflict. Inheritance applies to tools (by directory name),
hooks (by directory name), resources (by file name), and config
fields (last write wins).

### Built-in registries

```yaml
builtin_tools:
  - search_skills
  - activate_skill

builtin_hooks:
  - skill_activation
  - stagnation: { threshold: 6, ignore_tools: [think, done] }
```

A loader MAY ship a built-in registry of tools and hooks. The
contents of the registry are not part of v1.0 (they are
implementation-defined). The directives themselves are part of v1.0.

### Reference grammar

Three forms work in any string-valued YAML field:

| Form                     | Meaning                                       |
|--------------------------|-----------------------------------------------|
| `${ref:name}`            | Resolve from `resources/<name>.py::build()`.  |
| `${py:module:symbol}`    | Import a Python object by dotted path.        |
| `${runtime.field}`       | Per-invocation runtime dict; supports `.a.b`. |
| `${runtime.x:-default}`  | Default if the runtime key is absent.         |

The legacy `@name` form is an alias for `${ref:name}`.

## System prompt — `prompts/system.md`

A single Markdown file containing the agent's system prompt verbatim.
No templating. The whole file is the prompt.

## Tools — `tools/<name>/`

Each tool is a directory with a `tool.yaml` manifest and an
`execute.py` body. Loaders MAY accept additional language extensions
in future spec versions.

```yaml
# tools/bash/tool.yaml
name: bash
description: Run a shell command and return stdout, stderr, exit code.
parameters:
  command:
    type: string
    description: The shell command to run.
requires:
  - workspace_config
concurrent_safe: false
free: false
timeout_s: 60
```

```python
# tools/bash/execute.py
import subprocess

def execute(ctx, *, command: str) -> dict:
    workspace = ctx.resources["workspace_config"].workspace
    proc = subprocess.run(command, shell=True, cwd=workspace, capture_output=True, text=True, timeout=60)
    return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode}
```

The `done` tool is required. The loader treats it as the loop's
completion sentinel; `done_tool:` in `config.yaml` defaults to it.

### Output contract on `done` (v1.0 slot)

When `tools/done/tool.yaml` declares `output_schema:`, loaders MUST
validate the agent's `done` arguments against it before terminating
the loop. This makes the agent's *return shape* part of the
cartridge contract, not an implicit convention.

```yaml
# tools/done/tool.yaml
name: done
description: Mark the task complete with a structured summary.
parameters:
  summary: { type: string }
  pass:    { type: boolean }
output_schema:
  type: object
  required: [summary, pass]
  properties:
    summary: { type: string, minLength: 1 }
    pass:    { type: boolean }
```

## Hooks — `hooks/NN_<name>/`

A hook directory contains either a local `hook.py` plus optional
`config.yaml` (kwargs passed to the hook class), or just `config.yaml`
that points at an importable class via the reference grammar.

```yaml
# hooks/01_CodingGuardrailHook/config.yaml
class_name: CodingGuardrailHook
order: 1
kwargs:
  max_warnings: 3
```

The `NN_` prefix sorts hook ordering. The hook must implement the
`LoopHook` protocol: at minimum one of `pre_dispatch`, `post_dispatch`,
`check_done`, or `should_stop`.

## Resources — `resources/<name>.py`

Each resource module exports a `build(runtime=None)` function that
returns a singleton instance. Tools and hooks request the singleton
through `requires:`; the loader injects it via `ctx.resources[<name>]`.

```python
# resources/workspace_config.py
def build(runtime=None):
    return WorkspaceConfig(workspace=runtime["workspace"])
```

The reserved resource name `runtime` is auto-injected from the host
runtime dict. Tools may declare `requires: [runtime]` directly without
a builder file.

## Memory — `memory/`

Files in this directory contribute to the agent's persistent context.

| File                 | Role                                              |
|----------------------|---------------------------------------------------|
| `long_term.md`       | Long-term memory; auto-loaded as a v1.0 slot.     |
| `*.md`               | Ordered static memory sources (filename = order). |
| `*.py`               | A module with `def load(state) -> str`.           |

Memory sources are concatenated in filename order, prefixed with
`memory/long_term.md` if present.

## Setup — `setup.py` (optional escape hatch)

A `setup(preset, resources, *, runtime=None)` function may mutate the
preset before it is returned. Most cartridges should not need this;
the declarative slots above cover the published examples.

## Loader contract (what a conformant runtime promises)

A loader implementation MUST:

1. **Validate the manifest.** Reject cartridges with no
   `workspace.json` or with `schema_version` greater than the loader
   supports.
2. **Resolve `extends:`.** Apply ancestors top-down before reading
   the leaf.
3. **Construct an `AgentPreset`** with: the parsed `LoopConfig`, an
   ordered tool registry, an ordered hook list, the resource registry,
   and (if the cartridge declares one) an LLM backend factory.
4. **Honour the reference grammar** in every string-valued YAML field.
5. **Auto-load `memory/long_term.md`** when present and not overridden.
6. **Validate `done` arguments** against `output_schema:` when
   declared.
7. **Fail loudly with a structured error** that names the offending
   file path on any malformed slot. No silent defaults that hide
   schema breaks.

A loader MAY also implement: hot-reload, parallel tool dispatch,
prompt-cache reuse, and provider-specific extras (`extra:` in the
model block). Conformance does not require these.

## Backwards compatibility

v1.0 introduces three new slots: structured `model:`, declarative
`permissions:`, and `memory.long_term`. All three are optional and
co-exist with the pre-v1.0 forms (flat `temperature`, hook-based
permissions, `memory/*.md` files). When both old and new shapes
appear, the new wins. Loaders SHOULD log a one-line deprecation
notice when only the old shape is present.

## Conformance fixtures

`tests/conformance/` (in this repository) contains a small set of
cartridges paired with expected loader outputs. Any v1.0 loader is
expected to produce equivalent outputs against the same fixtures.
The conformance suite will grow with v1.x; v2 will mandate it.

## Repository status

Today the schema lives inside the reference implementation. Once a
second loader exists (or the cartridge registry product reaches
beta), the spec, the JSON schema, and the conformance suite will
move to a neutral `cartridge-spec` repository so they have a stable
home independent of any one runtime. This is planned, not done.

## Changelog

- **v1.0** (2026-05-09) — first numbered version. New slots:
  `model:`, `permissions:`, `memory.long_term`, `output_schema` on
  `done`. Conformance fixture seed introduced.
- **v0.x** — implementation-defined; everything was already
  declarative but slots were not numbered.
