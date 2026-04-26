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