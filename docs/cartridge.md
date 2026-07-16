# Cartridges — the harness as reviewable files

`looplet.cartridge` makes the agent harness an editable, testable artifact on
disk. A cartridge round-trips with an `AgentPreset` for the JSON-able subset
of the harness and provides a clean Python escape hatch for the rest.

This is the review unit for test-driven harness engineering. Prompt changes,
tool implementations, hook policy, runtime wiring, and self-test cases become
ordinary diffs instead of hidden framework state.

This is the missing direction. With it, you can:

```python
from looplet import preset_to_cartridge, cartridge_to_preset

# Looplet preset → editable directory on disk
preset_to_cartridge(my_preset, "agent.cartridge")

# … edit prompts/system.md, tools/*/tool.yaml, hooks/*/config.yaml, … …

# Edited directory → fresh AgentPreset ready to run
preset = cartridge_to_preset("agent.cartridge")
for step in composable_loop(
    llm=llm, tools=preset.tools, state=preset.state,
    config=preset.config, hooks=preset.hooks,
):
    print(step.pretty())
```

A cartridge is a normal directory; everything below is plain text or
Python. Diff-friendly. Git-friendly. Editor-friendly. Agent-friendly.

---

## Principled exclusions — what cartridges deliberately don't do

The cartridge format is intentionally narrow. Every feature in the
table below was considered and *deliberately left out* of the format,
because it can be expressed cleanly as a composition of existing
primitives (subagent, hook, tool, builtin). Adding it to the format
would force every cartridge author to pay the complexity tax whether
they use it or not, and would foreclose the alternative compositions
each user might prefer.

This table is the **canonical** version. `AGENTS.md` ("Anti-features")
and `SPEC.md` reference it; the paper (`paper/boundary.tex`,
`paper/principled_cartridge_v2.md`) builds on it. If you change the
exclusion stance, update this table first and propagate.

| Excluded feature | Principle | Use this instead | Working example |
| --- | --- | --- | --- |
| Built-in **plan mode** (loop-level plan/execute split) | Planning is one composition of `subagent + done`, not a loop phase. | Parent cartridge calls `subagent` against a child planner cartridge, then executes the returned plan. | [`examples/planner.cartridge/`](https://github.com/hsaghir/looplet/blob/master/examples/planner.cartridge/) |
| Built-in **mid-edit linting** (run linters after every `write`) | Transient errors during edits waste budget. Run gates *at the boundary*. | A `check_done` hook that runs pytest/ruff/etc. exactly when the agent calls `done()`; failure blocks `done()` with an actionable message. | [`examples/snippets/11_quality_gate/quality_gate.py`](https://github.com/hsaghir/looplet/blob/master/examples/snippets/11_quality_gate/quality_gate.py) |
| Built-in **to-do list** (loop-tracked task ledger) | They confuse models and duplicate the session log. | Have the agent read/write a plain `TODO.md` file with the same tools it uses for everything else, OR compose a `Todo` tool. | (any tool with `read`/`write` over `TODO.md`) |
| Built-in **approval popups** by default | Approval fatigue degrades into security theatre. The right boundary is a sandbox. | Run inside a container/worktree (default). Opt in to `PermissionHook(engine)` + `ApprovalHook` only when there is a real human supervising. | [`src/looplet/permissions.py`](https://github.com/hsaghir/looplet/blob/master/src/looplet/permissions.py) |
| Built-in **TUI / chat shell** | looplet is a library that yields `Step`s. Shells and TUIs are downstream. | Build any TUI on top of the `Step` stream, or use the separate `looplet new` shell. | (consumer code) |
| Built-in **background daemons** (long-running shell tasks) | Lifecycle, signals, and cleanup are not loop concerns. | tmux, systemd, a job queue — anything that already solves daemon supervision. | (out of scope) |
| Built-in **DSL / magic globals** | Cartridges should be plain Python data + plain functions. | YAML for declarative slots; importable Python for code. No DSL. | (every shipped cartridge) |
| Built-in **mandatory inheritance** | Hooks/backends/states are `@runtime_checkable` Protocols. | Any object with the right methods works — no base class to import. | [`src/looplet/loop.py` `LoopHook`](https://github.com/hsaghir/looplet/blob/master/src/looplet/loop.py) |
| **Phases / state machines** | Phases turn the LLM into a slot-filler and re-introduce the rigidity cartridges replaced. | Write the SOP into `prompts/system.md` and let the LLM follow it. For genuine state-machine routing, write a host application that orchestrates multiple cartridges. | [`examples/coder.cartridge/prompts/system.md`](https://github.com/hsaghir/looplet/blob/master/examples/coder.cartridge/prompts/system.md) |
| **Wider context window / cache policy / compaction knobs in the cartridge** | Two hosts can legitimately disagree about token budgets, retention, and cache TTLs without changing what the agent does. | Sibling `runtime.yaml` carries every RUNTIME-tier knob (`context_window`, `max_tokens`, `cache_policy`, `compact_service`, etc.); the same cartridge runs unchanged on a 32k-window host and a 1M-window host. | [`examples/coder.cartridge/runtime.yaml`](https://github.com/hsaghir/looplet/blob/master/examples/coder.cartridge/runtime.yaml) |
| **Magic prompt files** (`prompts/briefing.md`, `prompts/recovery.md` auto-loaded by filename) | Magic filenames create hidden behaviour discovered only by reading the loader. Every hook a cartridge installs should be visible in `config.yaml`. | Declare the file via `builtin_hooks: - static_briefing: { path: ... }` / `recovery_hint`. Hard-rejected in v2 unless declared. | [docs below: `static_briefing` / `recovery_hint`](#builtin-hooks) |
| **Tool `tags:` for cross-tool filtering** (treating tags as routing input) | Tags spread categorisation across every `tool.yaml`; the consuming hook should own its categorisation so tools stay decoupled. | Put the tool list in the hook's `kwargs:` (`enrichment_tools: [a, b, c]`). Tools advertise capabilities via their schema, not labels. | (any hook with a `kwargs.<role>_tools:` field) |
| **Render / truncation hints on tool schemas** | Two hosts can legitimately disagree on truncation policy. The cartridge should not pre-decide. | Tool body returns small-by-default plus an `expand` parameter; the agent learns the affordance from the result. Per-host truncation overrides live in `runtime.yaml: tool_render_hints:`. | (see `docs/recipes.md`) |
| **Multi-`extends:`** (`extends: [a, b]`) | Diamond-inheritance / C3 linearisation buys nothing in practice; cartridges that "extend two parents" are composing two concerns. | Single `extends:` chain + `builtin_hooks:` + a shared `resources/` module. For genuinely independent concerns, host the two cartridges side-by-side. | [`examples/snippets/01_inheritance/`](https://github.com/hsaghir/looplet/blob/master/examples/snippets/01_inheritance/) |
| **Polyglot tool bodies in one cartridge** (Python + TypeScript + Rust under one `tools/<name>/`) | A runtime cannot execute tool bodies in a language it does not host; mixing languages forecloses portability instead of enabling it. | Pick one body language per cartridge. Two-runtime portability ships the *same-language* cartridge to both runtimes; cross-language reuse is a registry concern. | (out of scope for v1.x) |
| **Signed cartridges** (signature embedded in the cartridge body) | Signing is a registry concern; the cartridge must remain content-addressable so the signature can target a stable hash. | Sibling `<name>.cartridge.sig` over the canonical content hash from [SPEC.md §"Cartridge identity"](https://github.com/hsaghir/looplet/blob/master/SPEC.md#cartridge-identity-v2-prep). Computed by the registry / signer, not by the loader. | (out of scope for v1.x) |
| **Host-owned release holdouts inside the cartridge** | Colocated evals are useful self-tests, but a candidate that can edit its own promotion oracle can manufacture a false green. | Ship versioned self-tests under `evals/`; keep protected holdouts outside the cartridge and inject host-owned paths through `runtime`. | [`examples/regression_demo/report_agent.cartridge/evals/`](https://github.com/hsaghir/looplet/tree/master/examples/regression_demo/report_agent.cartridge/evals) |
| **API keys / model secrets / approval handlers / cancel tokens / trajectory sinks** | These are HOST-tier — never serialised. The cartridge declares *intent* (`model:`, `permissions: ask:`); the host supplies *capability*. | API keys via host env. Approval handlers via `runtime={"ask_handler": fn}` (load-time fail-loud if `ask:` rules are present without one). Trajectory sinks via `ProvenanceSink` in the runner. | [SPEC.md §"Permissions"](https://github.com/hsaghir/looplet/blob/master/SPEC.md#permissions-v10-slot) |

If a feature in this table turns out to be wrong, the bar to add it
to the format is: (a) it cannot be expressed as a hook / tool /
preset / subagent, AND (b) every cartridge author pays for it whether
they want it or not. **Both** must hold.

---

## Layout

```text
agent.cartridge/
├── cartridge.json           # schema_version, name, description, free-form metadata
├── prompts/
│   └── system.md            # config.system_prompt (file body)
├── config.yaml              # LoopConfig JSON-able subset
├── runtime.yaml             # optional host/runtime-tier defaults
├── tools/
│   └── grep/
│       ├── tool.yaml        # name, description, parameters, optional flags
│       └── execute.py       # def execute(*, ...) -> Any
├── hooks/
│   └── 00_DemoCounter/      # leading number = sort order = hook list order
│       ├── hook.py          # exposes `class HookClass`
│       └── config.yaml      # class (or class_name) + kwargs for HookClass(**kwargs)
├── resources/
│   └── project_dir.py       # optional build(runtime) dependency
├── evals/                   # optional, versioned self-test contract
│   ├── cases/
│   │   └── smoke.json       # agent-visible task + grader-only expected
│   ├── collect_outcome.py   # collect_* functions inspect world state
│   └── eval_correctness.py  # eval_* graders score collected outcomes
└── memory/
    └── 00_static.md         # one StaticMemorySource per file
```

Sort order matters for hooks: directories are loaded alphabetically,
which becomes the hook-list order at execution time. Use `00_`, `10_`,
`20_` prefixes to keep room for inserts.

### Self-tests versus protected holdouts

`load_cartridge_evals()` discovers all three parts of a colocated `evals/`
bundle: case data, `collect_*` outcome collectors, and `eval_*` graders.
`run_cartridge_evals()` can execute the complete bundle in fresh per-case
workspaces and persist trajectories, artifacts, scores, and grader-only
expected data.

Those evals define what the cartridge claims about itself and should travel
with its version. They are analogous to package tests. They are not a secure
promotion oracle when the candidate or generator can modify the cartridge.
Keep release holdouts in host-owned storage, keep their expected data out of
the agent sandbox, and bind protected collectors through the runner's
`runtime` dictionary. See [Behavioral evals](evals.md#trust-boundary-the-agent-must-not-own-its-oracle).

---

## Reference grammar

Any string value in `config.yaml`, hook `config.yaml`, or `tool.yaml`
can use the cartridge reference grammar to resolve to a Python
object at load time. Three forms, one resolver, one mental model:

| Form | Resolves to |
| --- | --- |
| `${ref:name}` | The resource built by `resources/name.py::build()` |
| `${py:module:symbol}` | `importlib.import_module(module).symbol` (dotted symbols allowed: `${py:my.app:Class.factory}`) |
| `${runtime.field}` | The value of `runtime[field]` passed to `cartridge_to_preset(cartridge_path, runtime=...)`. Supports nested lookup (`${runtime.a.b.c}`) and defaults (`${runtime.x:-15}`) |

The legacy `"@name"` form is still accepted as an alias for
`${ref:name}` so older cartridges keep loading unchanged.

References work uniformly:

```yaml
# config.yaml
max_steps: ${runtime.max_steps:-15}
compact_service: ${ref:compact_service}
state: ${py:my.app.state:MyAgentState}
memory_sources:
  - ${ref:project_memory}

# hooks/00_MyHook/config.yaml
class: ${py:my.app.hooks:MyHook}
kwargs:
  llm: ${ref:llm}
  threshold: ${runtime.threshold:-0.85}
```

After load, every built resource is exposed on `preset.resources`
keyed by name, so callers (benchmarks, evidence-bundle writers, SDK
shims) can reach live objects without going through `setup.py`.

---

## Composition with `extends:` { #extends }

A cartridge may inherit one parent by declaring a relative or absolute path in
`config.yaml`:

```yaml
extends: ../coder.cartridge
max_steps: 20
```

Looplet resolves the parent first, then layers the child over it. Parent tools,
hooks, resources, prompts, memory, and runtime defaults are inherited; child
files and config keys win on collision. Inheritance can be transitive, cycles
fail, and multiple parents are deliberately unsupported. Use built-in hooks or
shared resources for orthogonal concerns instead of diamond inheritance.

## Built-in registries { #built-in-registries }

`builtin_tools:` and `builtin_hooks:` are spec-portable directives
(part of v1.0) but the **contents** of each registry are
runtime-defined. The looplet runtime ships the following.

### `builtin_tools:` (looplet) { #builtin-tools }

| Name | Purpose |
| --- | --- |
| `subagent` | Spawn a sub-loop with its own tools/system_prompt and return the structured result; used to compose agents-as-tools without manual orchestration. |
| `scaffold_cartridge` | Write a fresh cartridge skeleton (`cartridge.json` + `config.yaml` + `prompts/system.md` + `tools/done/`) into a target directory; used by `agent_factory.cartridge` to let an agent bootstrap another agent. |
| `search_skills` | Query the active `SkillManager` for skills matching a description; returns ranked candidates. |
| `activate_skill` | Activate a discovered skill into the current loop, registering its tools and adding its instructions to the briefing. |

Opt in from a cartridge:

```yaml
# config.yaml
builtin_tools:
  - subagent
  - scaffold_cartridge
```

Unknown names raise `CartridgeSerializationError` at load time;
canonical list lives at `looplet.builtin_tools.AVAILABLE`.

### `builtin_hooks:` (looplet) { #builtin-hooks }

| Name | Purpose |
| --- | --- |
| `skill_activation` | Pairs with `search_skills` / `activate_skill`: tracks active skills across steps and threads their instructions into the briefing. Requires the `skill_manager` resource. |
| `stagnation` | Stops the loop with a structured reason when the same `(tool, args)` pair repeats N times in a row (`threshold:`, `ignore_tools:`). The principled alternative to baking "are we stuck?" detection into the loop. |
| `per_tool_limit` | Blocks further calls to a named tool once a per-tool budget is exhausted (`limits: { write: 50 }`); returns `Block(...)` with the limit in the reason so the model can self-correct. |
| `threshold_compact` | Triggers compaction when prompt tokens cross a fraction of the context window. Pairs with `compact_service:` in `runtime.yaml`. |
| `static_briefing` | Loads a fixed file (e.g. `prompts/briefing.md`) and prepends it to every step's prompt. The explicit replacement for the v1.x magic `prompts/briefing.md` auto-load (which is hard-rejected in v2). |
| `recovery_hint` | Loads a fixed file (e.g. `prompts/recovery.md`) and injects it after a tool error. The explicit replacement for the v1.x magic `prompts/recovery.md` auto-load (hard-rejected in v2). |

Opt in with optional kwargs:

```yaml
# config.yaml
builtin_hooks:
  - stagnation: { threshold: 6, ignore_tools: [think, done] }
  - per_tool_limit: { limits: { write: 50, bash: 200 } }
  - threshold_compact: { fraction: 0.85 }
  - static_briefing: { path: prompts/briefing.md }
```

Each entry is either a bare string (no kwargs) or a single-key
dict (`name: kwargs`). Unknown names raise
`CartridgeSerializationError`; canonical list lives at
`looplet.builtin_hooks.AVAILABLE`.

**Working examples in the repo:**

- `examples/coder.cartridge/` — `subagent`, `stagnation`,
  `per_tool_limit`, `threshold_compact`.
- `examples/agent_factory.cartridge/` — `scaffold_cartridge` (extends
  `coder.cartridge`).
- `examples/skillful_analyst.cartridge/` — `search_skills`,
  `activate_skill`, `skill_activation`.

---

## Round-trip guarantees

### What round-trips losslessly

| Component | How |
| --- | --- |
| CONTRACT-tier `LoopConfig` fields (`max_steps`, `done_tool`, `done_tools`, `permissions`, `memory`, `model`, `extends`, `builtin_tools`, `builtin_hooks`; plus `tool_metadata` auto-populated by the loader) | Serialised via `config.yaml` |
| RUNTIME-tier `LoopConfig` fields (`max_tokens`, `temperature`, `recovery_temperature`, `max_turn_continuations`, `generate_kwargs`, `use_native_tools`, `concurrent_dispatch`, `reactive_recovery`, `context_window`, `max_briefing_tokens`, `compact_service`, `cache_policy`, `checkpoint_dir`, `initial_checkpoint`, `tool_result_persist_dir`, `router`, `tracer`, `recovery_registry`) | Serialised via sibling `runtime.yaml` (spec v2). v1.x cartridges placing these in `config.yaml` continue to load with a `DeprecationWarning`; v2.0 will hard-fail. |
| `system_prompt` | Written to `prompts/system.md` |
| Tools whose `execute` is a top-level function | `tools/<name>/{tool.yaml, execute.py}`; the source is preserved verbatim and an `execute = <orig_name>` alias is appended so the loader finds it under the canonical name |
| Hooks with an opt-in `to_config(self) -> dict` method | `hooks/NN_<ClassName>/{hook.py, config.yaml}`; class source is preserved, kwargs come from `to_config()` |
| Hooks whose constructor takes no kwargs | Same as above; `kwargs: {}` |
| `StaticMemorySource` instances | One markdown file per source under `memory/` |

### What does not round-trip (and what happens)

The non-serialisable `LoopConfig` fields are callables and runtime
objects: `build_briefing`, `extract_entities`, `build_trace`,
`build_prompt`, `extract_step_metadata`, `domain`, `router`, `tracer`,
`recovery_registry`, `compact_service`, `output_schema`,
`initial_checkpoint`, `cache_policy`, `cancel_token`,
`approval_handler`, `render_messages_override`.

Behaviour controlled by `strict`:

- `strict=False` (default) — they are silently omitted from the
  serialised config. Each skipped field appends a string to
  `Cartridge.serialization_warnings` so callers can audit what was
  dropped.
- `strict=True` — `CartridgeSerializationError` is raised on the first
  non-round-trippable field.

Tools whose `execute` is a closure or lambda fall into the same
bucket: a placeholder `execute()` is written and a warning is recorded
(or raised under `strict=True`). The fix is to refactor the tool's
`execute` into a top-level function.

Hooks whose source cannot be retrieved by `inspect.getsource` (e.g.
defined dynamically) get a placeholder class and a warning.

---

## Hook patterns

### Pattern 1: opt-in `to_config()` (recommended)

```python
class DemoCounter:
    def __init__(self, *, threshold: int = 3) -> None:
        self.threshold = threshold

    def to_config(self) -> dict:
        return {"threshold": self.threshold}

    def post_dispatch(self, *args, **kwargs):
        ...
```

Round-trips perfectly. After load, `loaded.hooks[0].threshold == 5`.

### Pattern 2: dataclass hook

If the hook is a dataclass, you can wire `to_config` to
`dataclasses.asdict(self)` once and forget about it.

### Pattern 3: code-only hook

Hooks without `to_config()` still round-trip *structurally* (their
class source is preserved on disk, and `kwargs={}` is used at load
time). For hooks with required constructor arguments, you must add
`to_config()` or hand-edit `config.yaml`.

---

## Usage examples

### From a built preset

```python
from looplet import (
    BaseToolRegistry, DefaultState, LoopConfig,
    preset_to_cartridge,
)
from looplet.tools import ToolSpec
from looplet.presets import AgentPreset

def lookup(*, key: str) -> dict:
    return {"key": key, "value": {"x": 1, "y": 2}.get(key)}

reg = BaseToolRegistry()
reg.register(ToolSpec(name="lookup", description="lookup",
                      parameters={"key": "str"}, execute=lookup))
preset = AgentPreset(
    config=LoopConfig(max_steps=10, system_prompt="lookup agent"),
    hooks=[],
    tools=reg,
    state=DefaultState(max_steps=10),
)

ws = preset_to_cartridge(preset, "agent.cartridge")
print(ws.serialization_warnings)   # [] for a clean preset
```

### From an existing cartridge

```python
from looplet import cartridge_to_preset, composable_loop

preset = cartridge_to_preset("agent.cartridge")
for step in composable_loop(
    llm=llm, tools=preset.tools, state=preset.state,
    config=preset.config, hooks=preset.hooks,
):
    ...
```

### Inspecting metadata only

```python
from looplet import Cartridge

cartridge = Cartridge.from_directory("agent.cartridge")
print(cartridge.name, cartridge.description, cartridge.schema_version)
```

---

## When to use a Cartridge vs. SkillBundle

| You want to … | Use |
| --- | --- |
| Ship a runnable bundle as a Python package with a custom `build()` factory | `SkillBundle` |
| Edit prompt / tool / hook content as text files, version-control diffs, and re-execute | `Cartridge` |
| Review or modify a harness from another process or agent | `Cartridge` |
| Snapshot the live preset of a running agent for later inspection | `Cartridge` |
| Both: ship a bundle whose `build()` simply loads a cartridge | Both — bundle's `looplet.py` calls `cartridge_to_preset(__file__).to_preset()` |

---

## Schema versioning

`cartridge.json` carries a `schema_version` integer. The current
schema is `1`. Forward-incompatible layout changes will bump this; a
loader can detect the version before reading and choose how to handle
mismatches.
