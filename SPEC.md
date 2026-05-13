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
my_agent.cartridge/                # or my_agent.cartridge/
├── cartridge.json              # required: name, schema_version (alias: cartridge.json)
├── config.yaml                 # required: contract — what the agent does
├── runtime.yaml                # optional: runtime knobs — how this host runs it
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

The `.workspace` and `.cartridge` directory suffixes are conventional
but not load-bearing (loaders MUST accept any directory containing a
valid manifest file). The manifest may be named `cartridge.json`
(historical) or `cartridge.json` (spec terminology); they are
equivalent. If both are present, `cartridge.json` wins.

## Manifest — `cartridge.json` / `cartridge.json`

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

### Field tiers (spec v2 preview)

LoopConfig fields fall into three tiers:

- **CONTRACT** — *what the agent does.* Lives in `config.yaml`.
  `max_steps`, `system_prompt`, `done_tool`, `done_tools`,
  `acceptance_criteria`, `tool_metadata`, `permissions`, `memory`,
  `model`, `extends`, `builtin_tools`, `builtin_hooks`, etc. These
  travel with the cartridge across hosts and SHOULD round-trip
  identically.
- **RUNTIME** — *how this host runs it.* Lives in the sibling
  `runtime.yaml`. `max_tokens`, `temperature`, `recovery_temperature`,
  `max_turn_continuations`, `use_native_tools`, `concurrent_dispatch`,
  `reactive_recovery`, `context_window`, `context_window_steps`,
  `context_inline_per_step_chars`, `context_window_total_chars`,
  `max_briefing_tokens`, `router`, `tracer`, `recovery_registry`,
  `compact_service`, `cache_policy`, `checkpoint_dir`,
  `initial_checkpoint`, `tool_result_persist_dir`. Different hosts
  MAY override freely.
- **HOST** — *runtime-supplied callables.* Never serialised:
  `approval_handler`, `cancel_token`, `render_messages_override`.

**Backwards compatibility (v1.x).** Loaders MUST still accept
RUNTIME-tier keys in `config.yaml` and SHOULD emit a deprecation
warning naming the offending keys and the target `runtime.yaml`
path. **v2.0 will hard-fail** on RUNTIME keys appearing in
`config.yaml`.

### Runtime configuration — `runtime.yaml`

`runtime.yaml` is an optional sibling of `config.yaml` containing
only RUNTIME-tier fields. Same YAML shape and reference grammar as
`config.yaml` (`@<name>`, `${ref:name}`, `${py:module:symbol}`,
`${runtime.field}`).

```yaml
# runtime.yaml
max_tokens: 2000
temperature: 0.2
context_window: 128000
compact_service: "@compact_service"
```

Merge order under `extends:`: parent `runtime.yaml` is loaded
first, then child overrides via shallow merge (top-level scalars
and lists replaced wholesale; mappings recursively merged) — same
rules as `config.yaml`. Keys outside the RUNTIME or HOST tier
appearing in `runtime.yaml` MUST raise a load-time error.

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
done_tools: []                 # v1.1: additional terminal sentinels (additive)
```

### Multiple terminal sentinels (v1.1)

By default the loop ends only when the agent invokes `done_tool`
(default `done`). Cartridges with multiple distinct outcomes (e.g. a
SOC-triage agent that finishes via either `report` or `escalate`)
declare additional sentinels via `done_tools:`:

```yaml
done_tool: report              # primary sentinel; carries any output_schema
done_tools: [escalate]         # additional terminal sentinels
```

The list is **additive**: the loop terminates when the agent invokes
`done_tool` OR any name in `done_tools`. The primary `done_tool`
remains the sentinel for `output_schema:` validation; secondary
sentinels are unvalidated by the loop (their `tool.yaml` may still
declare `output_schema:` as metadata, and a hook may enforce
cross-sentinel checks).

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

**`ask:` rules require a host-supplied handler.** A cartridge that
declares any `ask:` rule is announcing a human-in-the-loop contract.
Loaders MUST refuse to load such a cartridge unless the host supplies
an `ask_handler` callable (passed as `runtime={"ask_handler": fn}` to
`cartridge_to_preset`). The handler receives `(ToolCall,
PermissionRule)` and MUST return `PermissionDecision.ALLOW` or
`PermissionDecision.DENY`. Without this fail-loud check, ASK rules
silently fall back to the engine's `default` (typically `allow`),
defeating the intent of asking.

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
extends: ../base.cartridge        # one parent
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

### Optional prompt files (v1.1)

Two additional optional files in `prompts/` get auto-attached as
hooks when present:

* **`prompts/briefing.md`** — auto-prepended to every step's
  briefing section (via `pre_prompt`). Use for short reminders that
  should appear in every prompt without bloating the system prompt.
  Other hooks may add their own briefing output; all are concatenated.
* **`prompts/recovery.md`** — injected into the prompt that follows
  any tool error (via `post_dispatch` + `InjectContext`). Use for
  general remediation guidance that applies broadly when something
  goes wrong.

Both are absent by default. Loaders MUST attach the corresponding
hook (e.g. `StaticBriefingHook`, `RecoveryHintHook` in the reference
implementation) when the file is present, and skip silently when it
isn't.

No other prompt files are recognised in v1.1. Cartridges that need
more elaborate prompt templating use plain Python in a hook or
resource — the cartridge format does not include a templating DSL.

## Tools — `tools/<name>/`

Each tool is either a directory with a `tool.yaml` manifest and an
`execute.py` body **or** (v1.1) a single Python file at
`tools/<name>.py`. Loaders MAY accept additional language extensions
in future spec versions.

### Multi-file form (canonical)

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
tags: [shell, mutating]                # v1.1: free-form labels (advisory)
render:                                # v1.1: prompt-rendering hints (advisory)
  preview: 5                           #   if data is a list, show first 5 items
  max_chars: 4000                      #   per-step cap for THIS tool's results
```

```python
# tools/bash/execute.py
import subprocess

def execute(ctx, *, command: str) -> dict:
    workspace = ctx.resources["workspace_config"].workspace
    proc = subprocess.run(command, shell=True, cwd=workspace, capture_output=True, text=True, timeout=60)
    return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode}
```

### Single-file form (v1.1)

For tools whose body is short and whose metadata is light, a single
`tools/<name>.py` declares everything via module-level dunders:

```python
# tools/echo.py
"""Echo back what you got."""

__name__ = "echo"
__description__ = "Echo back what you got."
__parameters__ = {"text": {"type": "string"}}
__tags__ = ["test"]
__render__ = {"preview": 5}            # optional; same shape as tool.yaml render:
__requires__ = []                      # optional; same shape as tool.yaml requires:
# Optional: __concurrent_safe__, __free__, __timeout_s__

def execute(ctx, *, text: str) -> dict:
    return {"echoed": text}
```

The two forms are equivalent: a tool authored in either shape
produces the same `ToolSpec` and round-trips identically through
`preset_to_cartridge`. Both forms can co-exist in the same
`tools/` directory; a `tools/foo.py` file and a `tools/bar/`
directory both register tools.

### Tool fields

| Field | Type | Required | Where |
|---|---|---|---|
| `name` | string | yes | `tool.yaml: name` / `__name__` |
| `description` | string | yes | `tool.yaml: description` / `__description__` |
| `parameters` | dict | yes | `tool.yaml: parameters` / `__parameters__` |
| `requires` | list[str] | no | `tool.yaml: requires` / `__requires__` |
| `concurrent_safe` | bool | no, default false | `tool.yaml: concurrent_safe` / `__concurrent_safe__` |
| `free` | bool | no, default false | `tool.yaml: free` / `__free__` |
| `timeout_s` | float | no | `tool.yaml: timeout_s` / `__timeout_s__` |
| `tags` (v1.1) | list[str] | no, advisory | `tool.yaml: tags` / `__tags__` |
| `render` (v1.1) | dict | no, advisory | `tool.yaml: render` / `__render__` |
| `output_schema` | dict | no, only for done sentinels | `tool.yaml: output_schema` |

`tags` and `render` are **advisory hints**: loaders MAY honor them,
but no behaviour beyond storing them on `ToolSpec` is mandated by
v1.1. A second runtime that ignores `render.preview` is still
conformant; tool semantics are unchanged.

The `done` tool is required by default. The loader treats it as the
loop's completion sentinel; `done_tool:` in `config.yaml` defaults
to it. Cartridges can declare additional terminal sentinels via
v1.1's `done_tools:` (see Configuration section).

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
   `cartridge.json` or with `schema_version` greater than the loader
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
cartridges paired with locked-down expected loader outputs. Each
fixture pairs a cartridge with an `expected.json` describing the
tools, hooks, and config a v1.0 loader MUST produce. The
`looplet conform` driver runs all fixtures against any
`cartridge_to_preset`-shaped callable and reports per-fixture
pass/fail. Run `python -m looplet conform` from this repository to
exercise the suite against the reference loader.

## Versioning policy

- **`schema_version`** in `cartridge.json` is the cartridge's
  declared spec major. v1.x is additive: a v1.1 loader MUST load
  any v1.0 cartridge with no behaviour change. v2 is reserved for
  breaking shape changes (slot rename, slot removal, semantics flip)
  and requires a migration guide.
- **Loaders** advertise the highest `schema_version` they support.
  Loading a cartridge with a higher `schema_version` than the
  loader supports MUST fail with a structured error.
- **Forward compatibility.** A v1.0 cartridge that uses *only* v1.0
  slots is guaranteed to load on every future v1.x loader.
- **Cartridge `version`** (in `cartridge.json`) is independent of
  `schema_version`. Cartridge authors choose semver,
  content-addressed, calendar, or anything else; loaders MUST treat
  it as opaque.

## Changelog

- **v1.1** (2026-05-12) — additive: tool `tags:` (advisory metadata),
  tool `render:` (advisory rendering hints with `preview:` and
  `max_chars:`), single-file tool form (`tools/<name>.py` with
  module-level dunders), `done_tools: [a, b]` plural sentinels
  (additive to `done_tool:`), and two optional prompt files
  (`prompts/briefing.md` auto-prepended to the briefing section,
  `prompts/recovery.md` injected after tool errors). All five are
  optional; v1.0 cartridges load on a v1.1 loader unchanged.
- **v1.0** (2026-05-09) — first numbered version. New slots:
  `model:`, `permissions:`, `memory.long_term`, `output_schema` on
  `done`. Conformance fixture seed introduced.
- **v0.x** — implementation-defined; everything was already
  declarative but slots were not numbered.
