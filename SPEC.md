# Cartridge Spec v2.0

> **Status.** Current. Implemented by Looplet 0.4.0 and later.
> **Companion schema.** [`cartridge.schema.json`](cartridge.schema.json).
> **Audience.** Agent builders and loader implementers.
> **Versioning.** `schema_version` is a spec major. Version 2 changes are
> additive unless this document explicitly says otherwise.

A cartridge is a directory of reviewable files describing one agent harness.
It separates the behavioral contract from host execution policy while keeping
every author-controlled component visible in source control.

This document specifies schema version 2 only. The Looplet loader rejects every
other declared schema version. `looplet migrate` remains available to rewrite
legacy schema-v1 directories.

## Scope

A cartridge describes:

- identity and body language;
- one system prompt;
- tools and terminal output shape;
- hooks and declarative policy;
- shared resources and memory sources;
- contract and runtime configuration;
- optional out-of-process tool, hook, state, and model boundaries.

A cartridge does not define a graph runtime, hosted control plane, annotation
system, sandbox, or promotion service. An `evals/` directory may travel beside
the harness as a versioned self-test bundle, but eval discovery and protected
holdouts are adjacent concerns rather than part of the loader contract.

## Canonical layout

```text
agent.cartridge/
├── cartridge.json
├── config.yaml
├── runtime.yaml
├── prompts/
│   └── system.md
├── tools/
│   ├── <name>/
│   │   ├── tool.yaml
│   │   ├── description.md
│   │   └── execute.py
│   └── <simple_name>.py
├── hooks/
│   └── <order>_<name>/
│       ├── config.yaml
│       └── hook.py
├── resources/
│   └── <name>.py
├── memory/
│   ├── long_term.md
│   ├── <order>_<name>.md
│   └── <order>_<name>.py
└── evals/
    ├── cases/*.json
    ├── collect_*.py
    └── eval_*.py
```

`cartridge.json`, `config.yaml`, and `prompts/system.md` are the canonical
identity, contract, and prompt files. Tool and hook directories are optional,
but a runnable cartridge normally provides the configured terminal tool.

The `.cartridge` directory suffix is conventional, not load-bearing. Looplet
also recognizes the historical `workspace.json` manifest filename so migrated
artifacts can retain stable paths. `cartridge.json` is canonical and wins when
both names exist.

Root-level `setup.py` is forbidden. Wiring belongs in resources, hooks,
built-in declarations, or host-supplied runtime values.

## Manifest

`cartridge.json` is JSON:

```json
{
  "name": "report_agent",
  "schema_version": 2,
  "description": "Publish a checked report",
  "language": "python",
  "version": "1.4.0",
  "metadata": {}
}
```

Required fields:

| Field | Contract |
| --- | --- |
| `name` | Non-empty artifact name matching `[A-Za-z0-9_.-]+`. |
| `schema_version` | Integer `2`. |

Optional fields:

| Field | Contract |
| --- | --- |
| `description` | Human-readable purpose. |
| `language` | One body language for local tools, hooks, resources, and memory modules. Defaults to `python`. |
| `version` | Artifact version chosen by the author. It is independent of `schema_version`. |
| `metadata` | Free-form JSON metadata. Loaders must preserve unknown keys. |

A runtime must reject a body language it cannot execute before importing any
body. Looplet's reference runtime executes `language: python` only.

## Configuration tiers

Schema v2 separates configuration into three tiers.

### Contract tier: `config.yaml`

Contract values define what the agent is. Common fields include:

```yaml
max_steps: 20
done_tool: done

model:
  provider: openai
  name: gpt-5.5
  reasoning_effort: high

permissions:
  default: allow
  deny:
    - tool: bash
      contains: { command: "rm -rf" }
      reason: destructive shell command

memory:
  long_term: memory/long_term.md

builtin_tools:
  - subagent

builtin_hooks:
  - stagnation: { threshold: 6, ignore_tools: [think, done] }
```

Other contract directives include `extends`, `state`, `memory_sources`,
`mcp_servers`, `state_services`, and `llm_gateway`.

`done_tools` is not a schema-v2 cartridge field. A cartridge declares one
terminal tool with `done_tool`. Distinct outcomes belong in that tool's payload
and output schema. Host code may still configure additional terminal tools
programmatically through `LoopConfig`; that is outside this file format.

### Runtime tier: `runtime.yaml`

Runtime values control how one host executes the same contract:

```yaml
max_tokens: 4096
temperature: 0.2
use_native_tools: true
concurrent_dispatch: false
context_window: 128000
compact_service: ${ref:compact_service}
checkpoint_dir: ${runtime.checkpoint_dir:-.looplet/checkpoints}

tool_render_hints:
  bash:
    preview: 5
    max_chars: 4000
```

Runtime-tier fields include sampling settings, provider kwargs, native-tool
selection, concurrent dispatch, reactive recovery, context budgets,
compaction, cache policy, tracing/router integrations, checkpoints, and tool
result persistence. `tool_render_hints` is a loader-level runtime override.

A schema-v2 loader must reject a runtime-tier field in `config.yaml`, even when
the same key also appears in `runtime.yaml`. It must reject contract or host
fields in `runtime.yaml`.

For an `extends` chain, parent configuration is read first and the child wins.
Mappings merge recursively; scalars and lists replace the parent value.

### Host tier

Host capabilities are never serialized. They include approval handlers,
cancellation tokens, message render overrides, API credentials, and live
trajectory sinks. The caller supplies them through code, environment, or the
`runtime` argument to its loader.

## Reference grammar

String values in configuration and hook kwargs may use:

| Form | Meaning |
| --- | --- |
| `${ref:name}` | Resolve the object built by `resources/name.py`. |
| `${runtime.field}` | Resolve a host-supplied runtime value. Nested fields are allowed. |
| `${runtime.field:-default}` | Use a scalar default when the runtime value is absent. |
| `@name` | Historical alias for `${ref:name}`. |

`${py:module:symbol}` is not part of schema v2. A cartridge must not import an
arbitrary host symbol through YAML. Wrap that import in a small
`resources/<name>.py` builder and reference the resulting resource.

An unresolved reference is a load error that must identify the source file and
known resource or runtime names.

## System prompt

`prompts/system.md` contains the system prompt verbatim. The cartridge format
does not define a prompt-template language.

`prompts/briefing.md` and `prompts/recovery.md` have no magic behavior. A
cartridge may keep text at those paths only when it explicitly declares the
corresponding built-in hook:

```yaml
builtin_hooks:
  - static_briefing: { path: prompts/briefing.md }
  - recovery_hint: { path: prompts/recovery.md }
```

A loader must reject either file when its matching hook is absent.

## Tools

### Multi-file form

The canonical form uses `tools/<name>/tool.yaml` and
`tools/<name>/execute.py`:

```yaml
name: lookup_owner
description: Look up one service owner.
parameters:
  service:
    type: string
requires:
  - ownership_store
concurrent_safe: true
free: false
timeout_s: 30
```

```python
def execute(ctx, *, service: str) -> dict:
    store = ctx.resources["ownership_store"]
    return {"service": service, "owner": store.lookup(service)}
```

`description.md` is optional. When present, it replaces the short YAML
description and supports longer capability, usage, and limitation text.

Supported tool fields are `name`, `description`, `parameters`, `requires`,
`concurrent_safe`, `free`, `timeout_s`, `render`, and `output_schema`.
Schema v2 rejects per-tool `tags`; consuming policy owns its tool categories.

Every `requires` name must resolve before execution. Tool directory and tool
name should match; strict loaders must reject a mismatch.

### Single-file form

Small tools may use `tools/<name>.py`:

```python
"""Echo one value."""

__description__ = "Echo one value."
__parameters__ = {"text": {"type": "string"}}
__concurrent_safe__ = True

def execute(ctx, *, text: str) -> dict:
    return {"echoed": text}
```

The single-file form is intentionally limited. It may declare name,
description, parameters, concurrency, free-call, and timeout metadata. It must
not declare `__requires__`, `__render__`, or `__tags__`. Use the multi-file form
when a tool needs those capabilities.

`tools/foo.py` and `tools/foo/` collide and must not coexist.

### Terminal output contract

The tool named by `done_tool` may declare `output_schema` in its `tool.yaml`:

```yaml
name: done
description: Finish with one explicit outcome.
parameters:
  outcome: { type: string, enum: [report, escalate] }
  summary: { type: string }
  blocked_on: { type: string }
output_schema:
  type: object
  required: [outcome]
  properties:
    outcome: { type: string, enum: [report, escalate] }
    summary: { type: string }
    blocked_on: { type: string }
```

The reference runtime supports a flat JSON Schema subset: top-level
`type: object`, `properties`, and `required`; each property may declare
`type`, `enum`, and `description`. The runtime validates those fields before
accepting completion. Conditional relationships such as "report requires
summary" belong in a `check_done` hook and an outcome grader; `oneOf`, `const`,
and length constraints are not currently enforced by `OutputSchema`.

## Hooks

Hook directories are ordered by `config.yaml: order` when present and then by
directory name. Numeric filename prefixes provide a readable default order.

An in-process hook directory contains `hook.py` and optional `config.yaml`:

```yaml
class_name: RequireArtifact
order: 10
enabled: true
kwargs:
  artifact_dir: ${ref:artifact_dir}
```

The class implements one or more Looplet lifecycle methods. The loader resolves
constructor kwargs and instantiates the class once. `enabled: false` supports
explicit ablation without deleting inherited files.

An out-of-process LEP hook uses `kind: lep` plus a command or server path in
`config.yaml`. Its configuration declares subscribed lifecycle slots, the view
sent across the boundary, and failure policy. It does not require `hook.py`.

## Resources and state

Each `resources/<name>.py` exports `build(runtime=None)`. The loader calls it
once and stores the returned object under `name`. Tools consume resources
through `requires`; hooks and configuration consume them through `${ref:name}`.

```python
def build(runtime=None):
    return WorkspaceConfig(root=runtime["project_root"])
```

The reserved `runtime` resource exposes the host runtime mapping. The loader
also exposes `cartridge_root` so bundled subprocess commands can resolve paths.

`config.yaml: state` may reference a built resource. Without a state directive
or host-supplied state factory, the runtime constructs its default state.

A resource may declare `THREAD_SAFE = True` or `False`. The runtime records the
declaration when deciding whether concurrent tools can share it. A non-boolean
declaration is a load error.

## Memory

Files under `memory/` are loaded in filename order:

| File | Behavior |
| --- | --- |
| `*.md` | Static persistent context. |
| `*.py` | Dynamic source exporting `load(state) -> str`. |
| `long_term.md` | Default long-term file when `memory.long_term` is not set. |

An explicit `memory.long_term` path wins over the default filename.

## Inheritance

`config.yaml` may declare one parent:

```yaml
extends: ../base.cartridge
```

Loaders resolve ancestors first and overlay the child. Child files and values
win on collision. Cycles are errors. Multiple parents are not supported; use a
single chain plus built-in hooks and shared resources for orthogonal concerns.

## Out-of-process services

`mcp_servers` declares tool servers. Each entry includes a string `command` and
may include a string-to-string `env` mapping, positive `timeout_s`, and a
`tools` list. An absent list registers every discovered tool; an empty list
registers none. The loader initializes the server, discovers tools, and keeps
its handle on the loaded preset for cleanup. Environment entries extend rather
than replace the host environment.

`state_services` declares shared mutable state servers. Each entry has the same
string `command`, string-to-string `env`, and positive `timeout_s` requirements.
The loader starts each service, injects its client into the resource registry
under the service name, and exports the service socket to sibling processes.

Boolean `llm_gateway` controls host model access for out-of-process components.
It is enabled by default when MCP servers are present. A host binds the live
backend at run time.

Commands and protocols are contract data, but their process language and
transport support remain runtime capabilities. Hosts must close all spawned
handles when a run ends.

## Loader contract

A conforming loader must:

1. find and validate the manifest;
2. reject any schema major other than 2;
3. reject unsupported body languages before importing bodies;
4. resolve the single-parent inheritance chain and reject cycles;
5. enforce the contract, runtime, and host field split;
6. reject `setup.py`, magic prompt files, `${py:...}`, cartridge `done_tools`,
   and disallowed tool metadata;
7. build resources once and resolve references with source-attributed errors;
8. load tools and hooks in deterministic order;
9. validate required resources and terminal output schemas;
10. materialize the configured state, permissions, memory, and service handles;
11. expose enough structured state for the host to close spawned processes;
12. fail loudly on malformed required components rather than silently changing
    the agent surface.

A loader may add hot reload, provider integrations, richer diagnostics, and
other host features without changing cartridge semantics.

## Content identity

`looplet hash <cartridge>` computes a canonical SHA-256 digest:

1. walk the directory recursively;
2. skip paths containing `__pycache__`, `.git`, `.venv`, `seed`,
   `.pytest_cache`, or `.mypy_cache`;
3. skip `.pyc` and `.pyo` files;
4. hash every remaining file;
5. sort `(relative_path, file_hash)` pairs by POSIX path;
6. hash each `<relative_path>\0<file_hash>\n` record in order.

`seed/` is excluded because runtime starter data may be overwritten. Hosts that
need data included in identity must store it elsewhere. Signatures belong in a
sibling registry artifact and target this digest; they do not belong inside the
content-addressed cartridge.

## Conformance

`tests/conformance/fixtures/` contains loader-shape, rejection, and
scripted-trajectory fixtures. `looplet conform` runs loader-shape and required
rejection cases against the reference loader. The test suite executes the
scripted-trajectory fixture separately. Loader shape parity is necessary but
does not establish full behavioral equivalence across providers, operating
systems, or process implementations.

For trajectory fixtures, conforming runtimes must agree on tool name, arguments,
result, error, and order. Timing, request identifiers, and tracing spans are not
part of trajectory parity.

## Versioning and migration

`schema_version` is the cartridge format major. A loader must fail on an
unsupported major rather than guess. Additive fields within major 2 must not
change the meaning of existing valid cartridges.

The manifest's optional `version` is the cartridge artifact version and remains
opaque to loaders.

`looplet migrate <cartridge>` upgrades legacy schema-v1 directories by:

1. moving runtime-tier fields from `config.yaml` to `runtime.yaml`;
2. declaring briefing and recovery prompt files through built-in hooks;
3. setting `schema_version` to 2.

Migration refuses opaque `setup.py` wiring. Port that code to resources and
declarative hooks, delete the file, and rerun migration. Use `--dry-run` to
preview changes. Loading is v2-only; migration is the compatibility boundary.