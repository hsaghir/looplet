# Skills

Skills are optional, lazy capability bundles. They let an agent discover
domain instructions without putting every domain manual, checklist, or
script into every prompt.

The core loop does not know about skills. A skill compiles into existing
looplet primitives:

- `SkillActivationHook` injects active instructions through `pre_prompt`.
- `make_skill_tools()` creates ordinary `ToolSpec`s for discovery and activation.
- Existing `Skill` objects can still register concrete tools and memory sources.

## On-disk format

`FileSkillStore` reads Claude/Agent Skills-style folders:

```text
skills/
  pdf/
    SKILL.md
    scripts/
    examples/
```

`SKILL.md` uses YAML-style frontmatter followed by markdown instructions:

```markdown
---
name: pdf
description: Use this skill whenever the user wants to work with PDF files.
---

# PDF Processing Guide

Use pypdf for structural edits and pdfplumber for text/table extraction.
```

Only `SKILL.md` is parsed. Scripts and resources are inert unless you wrap
them as normal looplet tools.

## Lazy activation

```python
from looplet import BaseToolRegistry, FileSkillStore, SkillActivationHook, SkillManager
from looplet.skills import make_skill_tools

store = FileSkillStore("./skills")
manager = SkillManager(store)
hooks = [SkillActivationHook(manager)]

tools = BaseToolRegistry()
for spec in make_skill_tools(manager):
    tools.register(spec)
```

The agent can call `search_skills` to see lightweight cards, then
`activate_skill` to load the full instructions. Only active skills are
injected into future prompts.

## Direct activation

For product flows where the UI or manifest decides the domain up front:

```python
store = FileSkillStore("./skills")
manager = SkillManager(store)
manager.activate("pdf")
hooks = [SkillActivationHook(manager)]
```

This keeps product ergonomics outside the main loop while preserving the
same observable, hook-based execution path.

## Runnable Bundles

A simple skill is an instruction payload. A runnable bundle is a full
domain bundle: `SKILL.md` plus a trusted Python entrypoint that builds
normal looplet primitives.

The loop story does not change when you use a bundle: the bundle
builds tools, hooks, config, and state; the LLM proposes a tool call; the
registry dispatches it; hooks observe or steer; state records the step; and
the loop yields a `Step`.

```text
skills/
  coder/
    SKILL.md
    looplet.py
```

`SKILL.md` declares the entrypoint:

```markdown
---
name: coder
description: Build, edit, test, and iterate on software projects.
entrypoint: looplet.py
---
```

The entrypoint exposes `build(runtime) -> AgentPreset`:

```python
from looplet import AgentPreset, SkillRuntime

def build(runtime: SkillRuntime) -> AgentPreset:
  ...
```

Load, validate, and run the bundle:

```python
from looplet import SkillRuntime, load_skill_bundle, run_skill_bundle, validate_skill_bundle

runtime = SkillRuntime(workspace=".", max_steps=20)
assert validate_skill_bundle("./skills/coder", runtime).ok

bundle = load_skill_bundle("./skills/coder")
for step in run_skill_bundle(bundle, llm=my_llm, task="Fix the tests", runtime=runtime):
  print(step.pretty())
```

The core loop still does not know about bundles. The bundle compiles into
an `AgentPreset` containing tools, hooks, config, and state.

`run_skill_bundle()` records provenance by default. Unless you provide an
explicit trace directory, traces are written under
`.looplet/traces/<skill-name>-<id>/` inside the runtime workspace. Pass
`provenance=False` to disable this when you need an unrecorded run.

The CLI can run a bundle directly:

```bash
python -m looplet run ./skills/coder "Fix the tests" --workspace .
python -m looplet run ./skills/coder "Fix the tests" --workspace . --trace-dir .looplet-trace
python -m looplet run ./skills/coder "Fix the tests" --workspace . --no-trace
```

Bundles can also be discovered without importing their Python
entrypoints. This is the lightweight index path for product UIs and agent
menus: it reads `SKILL.md`, checks whether the declared entrypoint exists,
and returns `BundleCard` records.

```python
from looplet import discover_skill_bundles

for card in discover_skill_bundles("./skills"):
  print(card.name, card.description, card.path)
```

The same discovery path is available from the CLI:

```bash
python -m looplet list-bundles ./skills
python -m looplet list-bundles ./skills --json
python -m looplet list-bundles ./skills --include-invalid
```

By default, instruction-only Claude Skills are skipped because they do not
have a runnable entrypoint yet. Pass `--include-invalid` when you want to
surface folders that look like bundles but need repair or wrapping.

Bundles may optionally expose `scripted_responses()` for deterministic
dogfood runs and `render_step(step)` for domain-specific terminal output.
For product bundles that need exact terminal behavior, a bundle can also
expose `run(...)`. In that case `looplet run` delegates the whole shell to
the bundle while still passing the default provenance setting and trace
directory options through. The delegated callable receives keyword-only
runtime inputs:

```python
def run(
    *,
    task: str,
    workspace: str | Path,
    max_steps: int,
    scripted: bool,
    scripted_responses: list[str],
    require_tests: bool,
    trace_dir: str | Path | None,
    provenance: bool,
) -> int:
  ...
```

## Blueprints and round trips

Bundles can be inspected as versioned `AgentBlueprint` records. A
blueprint is a stable structural view of the built `AgentPreset`: config,
tool schemas, hook order, memory source types, state type, metadata, and
source location. It is intentionally not a Python decompiler; arbitrary
closures stay opaque, while importable tools and components carry stable
`module:qualname` references.

```python
from looplet import SkillRuntime, blueprint_from_bundle

runtime = SkillRuntime(workspace=".", max_steps=20)
blueprint = blueprint_from_bundle("./skills/coder", runtime)
print(blueprint.fingerprint())
```

Use `export_bundle_to_library_code()` when a beginner wants to move from a
bundle command to normal Python call sites. The generated module is an
exact local wrapper around the bundle and includes the captured blueprint
for inspection. Because it preserves behavior by loading the original
bundle path, keep that bundle available or re-export after moving it:

```python
from looplet import export_bundle_to_library_code

export_bundle_to_library_code("./skills/coder", "coder_agent.py", function_name="build_agent")
```

The same conversion is available from the CLI:

```bash
python -m looplet blueprint ./skills/coder --workspace . --max-steps 20
python -m looplet export-code ./skills/coder coder_agent.py --function-name build_agent
```

Use `package_agent_factory_as_bundle()` when an advanced user has an
importable looplet factory and wants a runnable bundle entrypoint tied
to that factory reference:

```python
from looplet import package_agent_factory_as_bundle

package_agent_factory_as_bundle(
    "my_agent:build",
    "./skills/my-agent",
    name="my-agent",
    description="Run my custom looplet agent.",
)
```

Or from the CLI:

```bash
python -m looplet package my_agent:build ./skills/my-agent \
  --name my-agent \
  --description "Run my custom looplet agent."
```

`compare_blueprints()` checks whether two factories or bundles build the
same looplet structure. This is the programmatic equivalent of the coder
parity checks: compare the generated wrapper or packaged bundle against
the original before you trust the conversion.

Generated factory packages preserve exact behavior by importing the factory
reference. To move them to another environment, ship the referenced module
and its dependencies alongside the bundle or install them in that
environment.

## Claude Skill compatibility

looplet can load Claude/Agent Skills-style folders as lazy skills today.
It can also wrap instruction-only Claude Skills as runnable looplet
bundles:

```python
from looplet import claude_skill_compatibility, wrap_claude_skill_as_bundle

report = claude_skill_compatibility("./claude-skills/pdf")
assert report.can_wrap
wrap_claude_skill_as_bundle("./claude-skills/pdf", "./skills/pdf")
```

From the CLI:

```bash
python -m looplet wrap-claude-skill ./claude-skills/pdf ./skills/pdf
```

Compatibility levels are explicit:

- `instruction-only`: can be wrapped and run as a minimal looplet bundle.
- `resources-present`: resources are copied, but remain inert unless exposed
  as normal looplet tools.
- `scripts-present`: instructions can be wrapped, but scripts require an
  explicit tool adapter before looplet can claim exact behavior.
- `looplet-bundle`: the folder already declares a looplet entrypoint;
  wrapping copies the bundle as-is and preserves that entrypoint.

This means looplet can run a useful subset of Claude Skills directly as
bundles, while surfacing adapter gaps instead of pretending Claude-specific
runtime behavior is automatically reproducible.
