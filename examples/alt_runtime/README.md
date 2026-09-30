# `tinyloop` - a second runtime for the cartridge format

A stand-alone Python script that loads a small declarative cartridge subset
without importing `looplet`. It checks that this subset can be represented by
a second loader that does not share code with the reference one. It is not
a replacement for the current Looplet runtime or its hooks, schemas, and evidence.

## What it implements

- Manifest parsing (`workspace.json` / `cartridge.json`).
- A tiny YAML reader for the subset of YAML the conformance fixtures
  use (`max_steps:` / `max_tokens:` / `temperature:` / `done_tool:`,
  inline `{ ... }` in `tool.yaml`).
- Tool discovery: `tools/<name>/{tool.yaml, execute.py}` pairs.
- A `conformance_summary()` matching the pinned declarative fixtures, plus
  current schema-v2 hard-rejection fixtures.
- A scripted loop that dispatches a hard-coded list of tool calls
  against the loaded tool bodies.

## What it deliberately does NOT implement

`extends:`, hooks, resources, permissions, model binding, memory,
output schemas, hot-reload, native tool calling, recovery,
compaction, provenance. Each is a documented loader extension point
in [`SPEC.md`](../../SPEC.md), not a precondition for the
identity / shape / portability properties this script demonstrates.

## Running it

Print the conformance summary:

```bash
python examples/alt_runtime/tinyloop.py conform \
    tests/conformance/fixtures/01_minimal/cartridge
```

Compare against the expected summary:

```bash
python examples/alt_runtime/tinyloop.py conform \
    tests/conformance/fixtures/01_minimal/cartridge \
    --expected tests/conformance/fixtures/01_minimal/expected.json
```

Run a scripted loop:

```bash
python examples/alt_runtime/tinyloop.py run \
    tests/conformance/fixtures/01_minimal/cartridge \
    '[{"tool": "done", "args": {"summary": "ok"}}]'
```

## Why this matters

`tinyloop` independently checks loader output for the documented fixture
subset. Matching that subset does not establish full behavior equivalence for
arbitrary cartridges, MCP/LEP/SSP/MGP components, model calls, or side effects.
Use Looplet's example composition tests for those supported runtime paths.
