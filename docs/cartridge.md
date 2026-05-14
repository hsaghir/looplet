# Workspace

`looplet.cartridge` makes the agent harness an editable artifact on
disk. It is the bidirectional, lossless inverse of
`looplet.bundles.SkillBundle`: a cartridge round-trips with an
`AgentPreset` for the JSON-able subset of the harness, and provides a
clean code-escape hatch for the rest.

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
|---|---|---|---|
| Built-in **plan mode** (loop-level plan/execute split) | Planning is one composition of `subagent + done`, not a loop phase. | Parent cartridge calls `subagent` against a child planner cartridge, then executes the returned plan. | [`examples/planner.cartridge/`](../examples/planner.cartridge/) |
| Built-in **mid-edit linting** (run linters after every `write`) | Transient errors during edits waste budget. Run gates *at the boundary*. | A `check_done` hook that runs pytest/ruff/etc. exactly when the agent calls `done()`; failure blocks `done()` with an actionable message. | [`examples/snippets/11_quality_gate/quality_gate.py`](../examples/snippets/11_quality_gate/quality_gate.py) |
| Built-in **to-do list** (loop-tracked task ledger) | They confuse models and duplicate the session log. | Have the agent read/write a plain `TODO.md` file with the same tools it uses for everything else, OR compose a `Todo` tool. | (any tool with `read`/`write` over `TODO.md`) |
| Built-in **approval popups** by default | Approval fatigue degrades into security theatre. The right boundary is a sandbox. | Run inside a container/worktree (default). Opt in to `PermissionHook(engine)` + `ApprovalHook` only when there is a real human supervising. | [`src/looplet/permissions.py`](../src/looplet/permissions.py) |
| Built-in **TUI / chat shell** | looplet is a library that yields `Step`s. Shells and TUIs are downstream. | Build any TUI on top of the `Step` stream, or use the separate `looplet new` shell. | (consumer code) |
| Built-in **background daemons** (long-running shell tasks) | Lifecycle, signals, and cleanup are not loop concerns. | tmux, systemd, a job queue — anything that already solves daemon supervision. | (out of scope) |
| Built-in **DSL / magic globals** | Cartridges should be plain Python data + plain functions. | YAML for declarative slots; importable Python for code. No DSL. | (every shipped cartridge) |
| Built-in **mandatory inheritance** | Hooks/backends/states are `@runtime_checkable` Protocols. | Any object with the right methods works — no base class to import. | [`src/looplet/loop.py` `LoopHook`](../src/looplet/loop.py) |

If a feature in this table turns out to be wrong, the bar to add it
to the format is: (a) it cannot be expressed as a hook / tool /
preset / subagent, AND (b) every cartridge author pays for it whether
they want it or not. **Both** must hold.

---

## Layout

```
agent.cartridge/
├── cartridge.json           # schema_version, name, description, free-form metadata
├── prompts/
│   └── system.md            # config.system_prompt (file body)
├── config.yaml              # LoopConfig JSON-able subset
├── tools/
│   └── grep/
│       ├── tool.yaml        # name, description, parameters, optional flags
│       └── execute.py       # def execute(*, ...) -> Any
├── hooks/
│   └── 00_DemoCounter/      # leading number = sort order = hook list order
│       ├── hook.py          # exposes `class HookClass`
│       └── config.yaml      # class (or class_name) + kwargs for HookClass(**kwargs)
└── memory/
    └── 00_static.md         # one StaticMemorySource per file
```

Sort order matters for hooks: directories are loaded alphabetically,
which becomes the hook-list order at execution time. Use `00_`, `10_`,
`20_` prefixes to keep room for inserts.

---

## Reference grammar

Any string value in `config.yaml`, hook `config.yaml`, or `tool.yaml`
can use the cartridge reference grammar to resolve to a Python
object at load time. Three forms, one resolver, one mental model:

| Form | Resolves to |
|---|---|
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

## Round-trip guarantees

### What round-trips losslessly

| Component | How |
|---|---|
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
  `Workspace.serialization_warnings` so callers can audit what was
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
from looplet import Workspace

ws = Workspace.from_directory("agent.cartridge")
print(ws.name, ws.description, ws.schema_version)
```

---

## When to use Workspace vs. SkillBundle

| You want to … | Use |
|---|---|
| Ship a runnable bundle as a Python package with a custom `build()` factory | `SkillBundle` |
| Edit prompt / tool / hook content as text files, version-control diffs, and re-execute | `Workspace` |
| Mutate the harness from another agent (search, GEPA-style evolution, code review) | `Workspace` |
| Snapshot the live preset of a running agent for later inspection | `Workspace` |
| Both: ship a bundle whose `build()` simply loads a cartridge | Both — bundle's `looplet.py` calls `cartridge_to_preset(__file__).to_preset()` |

---

## Schema versioning

`cartridge.json` carries a `schema_version` integer. The current
schema is `1`. Forward-incompatible layout changes will bump this; a
loader can detect the version before reading and choose how to handle
mismatches.
