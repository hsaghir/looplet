# 01 — Cartridge inheritance

A cartridge inherits tools, hooks, resources, and config from a parent
via `extends:`. Local fields override inherited ones with the same name.
Inheritance chains arbitrarily deep — exactly like classes.

This snippet ships a **two-hop chain**:

```text
coder.cartridge                              (bundled)
  ↑ extends
refactorer.cartridge                         (this snippet)
  ↑ extends
pytest_refactorer.cartridge                  (this snippet)
```

The middle cartridge narrows `coder` to a Python-only refactoring
stance with a tighter step budget. The leaf adds a stricter
test-discipline prompt and a slightly higher budget. **Every leaf
inherits all 17 tools and all 10 hooks** from `coder` without copying
or forking anything.

## Look at the files

```text
01_inheritance/
├── refactorer.cartridge/
│   ├── cartridge.json
│   ├── config.yaml          # extends ../../../coder.cartridge
│   └── prompts/system.md    # narrower mission
└── pytest_refactorer.cartridge/
    ├── cartridge.json
    ├── config.yaml          # extends ../refactorer.cartridge
    └── prompts/system.md    # adds pytest discipline
```

`refactorer.cartridge/config.yaml` in full:

```yaml
extends: ../../../coder.cartridge
max_steps: 12
```

`pytest_refactorer.cartridge/config.yaml` in full:

```yaml
extends: ../refactorer.cartridge
max_steps: 16
```

That's the entire mechanism. No fork, no class hierarchy, no
copy-pasted tool registrations.

## Run it

```bash
uv run python -c "
from looplet import cartridge_to_preset
for ws in [
    'examples/coder.cartridge',
    'examples/snippets/01_inheritance/refactorer.cartridge',
    'examples/snippets/01_inheritance/pytest_refactorer.cartridge',
]:
    p = cartridge_to_preset(ws, runtime={'project_root': '.'})
    n_tools = len(p.tools.introspect()['tools'])
    print(f'{ws:<60} max_steps={p.config.max_steps:>3}  tools={n_tools}')
"
```

Expected output:

```text
examples/coder.cartridge                                     max_steps= 20  tools=17
examples/snippets/01_inheritance/refactorer.cartridge        max_steps= 12  tools=17
examples/snippets/01_inheritance/pytest_refactorer.cartridge max_steps= 16  tools=17
```

All three levels share the same tool registry; each level overrides
its own prompt and budget.

## Why this matters

The same operation in a code-defined framework would be a
class-hierarchy decision (subclass? mixin? composition?), spread
across multiple files, and brittle the moment someone refactors the
parent. With cartridges it is two YAML lines per level, and every
override lives in the file path that names what it overrides.
Composition is filesystem-mechanical, not codebase-organisational.
This is the layering OOP gave to classes, applied to agents.
