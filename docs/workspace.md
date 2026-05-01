# Workspace

`looplet.workspace` makes the agent harness an editable artifact on
disk. It is the bidirectional, lossless inverse of
`looplet.bundles.SkillBundle`: a workspace round-trips with an
`AgentPreset` for the JSON-able subset of the harness, and provides a
clean code-escape hatch for the rest.

This is the missing direction. With it, you can:

```python
from looplet import preset_to_workspace, workspace_to_preset

# Looplet preset → editable directory on disk
preset_to_workspace(my_preset, "agent.workspace")

# … edit prompts/system.md, tools/*/tool.yaml, hooks/*/config.yaml, … …

# Edited directory → fresh AgentPreset ready to run
preset = workspace_to_preset("agent.workspace")
for step in composable_loop(
    llm=llm, tools=preset.tools, state=preset.state,
    config=preset.config, hooks=preset.hooks,
):
    print(step.pretty())
```

A workspace is a normal directory; everything below is plain text or
Python. Diff-friendly. Git-friendly. Editor-friendly. Agent-friendly.

---

## Layout

```
agent.workspace/
├── workspace.json           # schema_version, name, description, free-form metadata
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
│       └── config.yaml      # class_name + kwargs for HookClass(**kwargs)
└── memory/
    └── 00_static.md         # one StaticMemorySource per file
```

Sort order matters for hooks: directories are loaded alphabetically,
which becomes the hook-list order at execution time. Use `00_`, `10_`,
`20_` prefixes to keep room for inserts.

---

## Round-trip guarantees

### What round-trips losslessly

| Component | How |
|---|---|
| Primitive scalar `LoopConfig` fields (`max_steps`, `max_tokens`, `temperature`, `done_tool`, `use_native_tools`, `concurrent_dispatch`, `reactive_recovery`, `context_window`, `max_briefing_tokens`, `checkpoint_dir`, `acceptance_criteria`, `tool_metadata`, `generate_kwargs`) | Serialised via `config.yaml` |
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
- `strict=True` — `WorkspaceSerializationError` is raised on the first
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
    preset_to_workspace,
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

ws = preset_to_workspace(preset, "agent.workspace")
print(ws.serialization_warnings)   # [] for a clean preset
```

### From an existing workspace

```python
from looplet import workspace_to_preset, composable_loop

preset = workspace_to_preset("agent.workspace")
for step in composable_loop(
    llm=llm, tools=preset.tools, state=preset.state,
    config=preset.config, hooks=preset.hooks,
):
    ...
```

### Inspecting metadata only

```python
from looplet import Workspace

ws = Workspace.from_directory("agent.workspace")
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
| Both: ship a bundle whose `build()` simply loads a workspace | Both — bundle's `looplet.py` calls `workspace_to_preset(__file__).to_preset()` |

---

## Schema versioning

`workspace.json` carries a `schema_version` integer. The current
schema is `1`. Forward-incompatible layout changes will bump this; a
loader can detect the version before reading and choose how to handle
mismatches.
