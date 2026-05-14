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

### Cartridge vs. skill

A **cartridge is the agent**: a self-contained behavioural contract
(identity, system prompt, tools, hooks, permissions, memory). A
**skill is procedural knowledge that any agent may load on demand**
(a snippet of instructions plus optional helper tools that the agent
discovers and activates mid-run via `search_skills` / `activate_skill`).
A cartridge composes skills by declaring `builtin_tools:
[search_skills, activate_skill]` and a `skill_manager` resource;
skills do not stand alone as agents.

When in doubt: if it has a system prompt and a `done` sentinel, it is
a cartridge. If it adds tools/instructions to *another* agent's loop,
it is a skill. The two are complementary, not competing — see
[`examples/skillful_analyst.cartridge/`](examples/skillful_analyst.cartridge/)
for a cartridge that loads skills at runtime.

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
  `permissions`, `memory`, `model`, `extends`, `builtin_tools`,
  `builtin_hooks`, etc. These travel with the cartridge across hosts
  and SHOULD round-trip identically. (`tool_metadata` rides along
  but is auto-populated by the loader — not authored by hand.)
- **RUNTIME** — *how this host runs it.* Lives in the sibling
  `runtime.yaml`. `max_tokens`, `temperature`, `recovery_temperature`,
  `max_turn_continuations`, `generate_kwargs`, `use_native_tools`,
  `concurrent_dispatch`, `reactive_recovery`, `context_window`,
  `context_window_steps`, `context_inline_per_step_chars`,
  `context_window_total_chars`, `max_briefing_tokens`, `router`,
  `tracer`, `recovery_registry`, `compact_service`, `cache_policy`,
  `checkpoint_dir`, `initial_checkpoint`, `tool_result_persist_dir`.
  Different hosts MAY override freely.
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

# Per-tool render overrides (cartridge spec v2). Shallow-merged onto
# each tool's ``render:`` block from tool.yaml; runtime keys win.
# This is the principled-exclusion answer for "I want render hints":
# the cartridge declares the agent's default; the host shifts the
# truncation policy without editing tools/<name>/tool.yaml.
tool_render_hints:
  bash:
    preview: 5
    max_chars: 4000
```

Unknown tool names in `tool_render_hints:` are a load-time error
(under `strict=True`) or a logged warning otherwise.

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
use_native_tools: true         # default true; auto-falls-back when backend lacks support
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

### Optional prompt files (v1.1, deprecated in v2)

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

**Cartridge spec v2 deprecation.** The magic-filename auto-load is
being replaced by an explicit declarative form via `builtin_hooks:`:

```yaml
builtin_hooks:
  - static_briefing:
      path: prompts/briefing.md   # or text: |- inline body
  - recovery_hint:
      path: prompts/recovery.md
```

Each hook accepts `text:` (inline body) xor `path:` (resolved
relative to the cartridge root via the loader-injected
`cartridge_root` resource). v1.x loaders MUST continue to honour
the magic-filename auto-load with a `DeprecationWarning`; v2.0 MUST
drop the auto-load. The benefit of the explicit form is that every
hook a cartridge installs is visible in `config.yaml` rather than
discovered by filename.

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

### Observable-behavior conformance (trajectory parity)

Loader-shape parity (above) is necessary but not sufficient for
portability. A second, stronger claim — **observable-behavior
conformance** — is exercised by trajectory-fixture pairs:

- **Fixture.** A cartridge plus a scripted sequence of tool calls
  plus an `expected_trajectory.json` listing the
  `{step, tool, args, result, error}` sequence the loop MUST emit.
- **Driver.** Both the reference loader (looplet's
  `composable_loop`) and a from-scratch second runtime
  (`examples/alt_runtime/tinyloop.py`) execute the same script and
  must produce the same trajectory modulo timing and call-id
  formatting.
- **Tier semantics.** MUST = same tool, same args, same result, same
  error, same order. SHOULD = same step number / step count
  (loaders that compress retries are non-conforming for trajectory
  fixtures but may still be loader-conformant). MAY = anything not
  in the spec-pinned subset (durations, request IDs, tracing spans).

Reference fixture: `tests/conformance/fixtures/08_trajectory_two_tools/`.
Reference test: `tests/conformance/test_trajectory_conformance.py`.

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

## Cartridge identity (v2 prep)

A cartridge has a deterministic content hash computed by hashing each
content-bearing file's contents and folding them into a single
SHA-256 digest in canonical order:

1. Walk the cartridge directory recursively.
2. Skip files whose path contains any of `__pycache__/`, `.git/`,
   `.venv/`, `seed/`, `.pytest_cache/`, `.mypy_cache/`.
3. Skip files with suffix `.pyc` or `.pyo`.
4. For every remaining file, compute its SHA-256.
5. Sort the `(relative_posix_path, file_sha256)` pairs by path.
6. Fold into the overall digest by writing
   `<rel_path>\0<file_sha256>\n` for each pair into a SHA-256
   accumulator.

The reference loader exposes this as `looplet hash <cartridge>` and
as `looplet.cli.spec_commands.cartridge_hash(root)`. The exclusion
list is part of the spec; changing it is a versioned change.

`seed/` is excluded so a cartridge that ships starter data the agent
may overwrite at runtime keeps a stable identity across runs. Hosts
that want runtime data to count against identity should write it
elsewhere.

## Resource concurrency declaration (v2 prep)

A `resources/<name>.py` module MAY declare a module-level
`THREAD_SAFE = True` or `THREAD_SAFE = False` constant. Loaders MUST
record this in a parallel registry keyed under the reserved name
`_resource_thread_safety` so the runtime can refuse
`concurrent_dispatch` of tools whose `requires:` includes an unsafe
resource. Resources that omit the declaration are treated as
"unknown" — the default runtime behaviour is to allow with a
warning; stricter hosts may choose to fail.

`THREAD_SAFE` MUST be a Python `bool`. Any other value is a
load-time error.

## Deferred design decisions

These were considered for v2 and explicitly deferred. Each entry
records the option, why it was deferred, and the workaround in v1.x.

### Multi-extends (`extends: [a, b]`)

**Status.** Deferred. Single-parent `extends:` is the canonical
form; lists are not accepted.

**Rationale.** Diamond-inheritance resolution (C3 linearisation,
MRO) adds complexity without a corresponding capability gain in
practice. Cartridges that "extend two parents" are nearly always
composing two independent concerns (e.g. a coding base + a security
profile), which is better expressed as composition (`builtin_hooks:`
+ shared `resources/`) than as inheritance.

**Workaround.** Chain single-parent extends (A extends B extends C)
or compose via `builtin_hooks:` + a shared resource module.

### Sub-agent isolation rules (v1.x today)

The composition primitive for "agent calls another agent" is
`run_sub_loop()` (in code) and the `subagent` builtin tool (in
cartridges). The isolation contract today:

| Concern | Default | Rationale |
|---|---|---|
| **State** | Isolated. Sub-loop gets a fresh `_MinimalState` keyed on its own `task`. | The sub-agent's step counter, query budget, and entities are its own. |
| **Tool registry** | Isolated by clone. Parent's tools are cloned, with `state_mutating_tools` (default `["done"]`) excluded so the sub-agent can't end the parent's loop. | The sub-agent sees the same tools as the parent except for sentinels. |
| **Conversation / session log** | Isolated. Sub-loop forks a new `SessionLog`; if a `conversation` is passed it's also forked. | The parent's history is not perturbed by the sub-agent's intermediate calls. |
| **Hooks** | Isolated by default. Pass `parent_hooks=ctx.hooks` to forward `on_event` to the parent's observability stack (MetricsHook, StreamingHook, TrajectoryRecorder). Full hook protocol (`check_done`, `should_stop`) stays sub-only. | Observability composes; control does not. |
| **Resources** | **Shared.** Sub-agent sees the parent's `ctx.resources`. | Forking would break the common case of reusing expensive clients (DB, vector store, SIEM). |
| **Budget (`max_steps`, `max_tokens`)** | Per sub-agent. The sub-loop has its own budget; the parent's `max_steps` does NOT decrement on sub-loop steps. | Sub-budgets must be sized at the call site, not inherited. |
| **`done` sentinel** | Sub-agent must end via its own `done` (or fall through `max_steps`). The parent's terminal sentinels are excluded from the sub's tool registry. | Each loop owns its own completion. |

**Working example.** [`examples/planner.cartridge/`](examples/planner.cartridge/)
demonstrates the principled-exclusion answer for "I want plan mode":
the parent has only two tools (`subagent`, `done`), delegates to a
child cartridge (`planner_child.cartridge/`), then summarises the
returned plan via `done`. No phase, no state machine, no special
loop feature.

**Workaround for resource isolation.** Pass `sub_tools=` with a
registry whose tools reference fresh resource instances, or wrap
the parent's resources in a copy-on-write dict. v2 will likely add
an opt-in `subagent: { resources: fork }` argument; until then,
shared is the default.

### Sub-agent resource isolation by default

**Status.** Deferred to v2.0. Today, sub-agents (`run_sub_loop`)
share the parent's resource registry; tools, state, conversation,
and session log are already isolated per sub-loop.

**Rationale.** Forking the resource registry by default would break
the common case of a sub-agent reusing the parent's expensive
clients (DB connections, vector stores). The right default is
sharing; opt-in forking can be added without a breaking change.

**Workaround.** Pass `sub_tools=` with a tool registry whose
`ctx.resources` reference fresh instances, or wrap the parent's
resources in a copy-on-write dict.

### Discriminated-union `output_schema` on a single `done`

**Status.** Deferred. The current model (`done_tool: report` +
`done_tools: [escalate, ...]`, each with its own `tool.yaml` and
optional `output_schema:`) is retained — and as of cartridge spec v2
**every sentinel listed in `done_tools:` whose `tool.yaml` declares
`output_schema:` is validated** by the loop, just like the primary
`done_tool`. The loader records secondary-sentinel schemas in
`LoopConfig.done_tool_schemas[<name>]`; a sentinel without an
`output_schema:` block is unvalidated (preserves v1.1 behaviour).

**Rationale.** Multiple sentinel tools give the LLM a clearer
contract (each tool name has a distinct shape) than a single tool
whose payload depends on a discriminator field. The LLM picks an
outcome by calling the right tool; collapsing into one `done` would
require it to embed the discriminator and inflate prompt complexity
for no gain.

**Workaround.** Use `done_tools:` (v1.1+) when an agent has multiple
distinct terminal outcomes; declare an `output_schema:` on each
sentinel's `tool.yaml` to validate every branch.

### Compaction tier (`compact_service`)

**Status.** Resolved (v2). `compact_service` stays in the RUNTIME
tier (`runtime.yaml`), not the CONTRACT tier (`config.yaml`).

**Rationale.** Compaction is a *runtime concern*: it depends on the
host's context window, the chosen LLM's cost/latency profile, and
the operator's preference for losslessness vs. recall. The cartridge
contract should describe *what the agent does* (its tools, memory,
permissions, system prompt), not *how aggressively a particular host
recycles tokens*. The same cartridge ought to run with no compaction
in a 1M-token-window deployment and with `DefaultCompactService` in
a 32k-token one — without editing `config.yaml`.

If a cartridge requires compaction for correctness (e.g. it expects
its conversation to fit inside a fixed budget), document that in
`prompts/system.md` and add a smoke test, not in CONTRACT. Hosts
that ignore the runtime-tier `compact_service` setting are
responsible for ensuring the conversation fits their window.

## v2.0 — hard removals (this release)

Setting `schema_version: 2` in `cartridge.json` opts the cartridge
into the v2 contract. The loader hard-fails (instead of emitting a
`DeprecationWarning`) when:

1. `config.yaml` declares any runtime-tier key (`max_tokens`,
   `temperature`, `context_window`, `compact_service`, etc.).
   Move them to `runtime.yaml`.
2. `prompts/briefing.md` exists without a matching
   `builtin_hooks: - static_briefing: { path: prompts/briefing.md }`
   entry. Same for `prompts/recovery.md` / `recovery_hint`.
3. `setup.py` is present. Express the wiring via `resources/` +
   `builtin_hooks:` + `@ref` strings instead.

`schema_version: 1` (the implicit default) continues to load with
deprecation warnings.

### Migration

`looplet migrate <cartridge>` mechanically performs the rewrite:

* splits runtime-tier keys out of `config.yaml` into `runtime.yaml`,
* converts magic `prompts/briefing.md` / `prompts/recovery.md`
  into explicit `builtin_hooks:` entries (the file content stays
  on disk; only the declaration moves),
* bumps `schema_version` to 2 in `cartridge.json`.

The tool refuses to run when `setup.py` is present, since there is
no general mechanical rewrite for opaque Python wiring; port that
manually, delete `setup.py`, then re-run `looplet migrate`.

`looplet migrate --dry-run` previews the changes without writing.
The migration is idempotent: running it on an already-v2 cartridge
is a no-op.

## Language-agnostic loader

`looplet`'s Python loader is one implementation. A conforming loader
in any language MUST honour the following clauses, in order:

1. **Manifest probe.** Find `cartridge.json` (or the legacy
   `workspace.json`) at the cartridge root. Read `schema_version`,
   `name`, `description`, `metadata`. Reject `schema_version`
   greater than the loader's max supported version.
2. **Extends.** If `config.yaml` declares `extends: <path>`,
   recursively load the parent cartridge first; layer the child's
   `tools/`, `hooks/`, `resources/`, `prompts/`, and `memory/`
   files over the parent's (child wins on filename collision).
3. **Config split.** Read `config.yaml` (CONTRACT tier) and the
   sibling `runtime.yaml` (RUNTIME tier). For `schema_version >= 2`,
   reject any RUNTIME key found in `config.yaml`. For
   `schema_version == 1`, accept with a deprecation diagnostic.
4. **Resources.** Build the resource registry: for each
   `resources/<name>.py` (or equivalent in the host language),
   call its `build()` factory. Expose the registry as a mapping
   that tools / hooks can index into via the `@<name>` or
   `${ref:name}` reference grammar.
5. **Tools, hooks, memory.** Materialise `tools/<name>/`,
   `hooks/<name>/`, and `memory/*.{md,py}` into the host's loop
   primitives. Resolve `requires:` against the resource registry;
   missing entries are load-time errors.
6. **Prompts.** Read `prompts/system.md` as the system prompt.
   For `schema_version == 1`, attach `prompts/briefing.md` and
   `prompts/recovery.md` as auto-loaded hooks with a deprecation
   diagnostic; for `schema_version >= 2`, reject their presence
   unless a matching `builtin_hooks:` entry exists.

The reference grammar (`@name`, `${ref:name}`, `${py:module:symbol}`,
`${runtime.field}`) is part of the spec; loaders MUST resolve all
four forms.
