# `tinyloop` — a second runtime for the cartridge format

A 200-line stand-alone Python script that loads and runs Cartridge
Spec v1.0 cartridges without importing `looplet`. Its purpose is
narrow: prove that the cartridge is portable by demonstrating a
second loader that doesn't share code with the reference one.

## What it implements

- Manifest parsing (`workspace.json` / `cartridge.json`).
- A tiny YAML reader for the subset of YAML the conformance fixtures
  use (`max_steps:` / `max_tokens:` / `temperature:` / `done_tool:`,
  inline `{ ... }` in `tool.yaml`).
- Tool discovery: `tools/<name>/{tool.yaml, execute.py}` pairs.
- A `conformance_summary()` matching the v1.0 spec-pinned subset.
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

The Cartridge Spec v1.0 claims:

> Because the cartridge declares no runtime, any conformant runtime
> can execute it.

The way to verify that claim is to write a second runtime that
shares no code with the first and check that it produces the same
loader output for the spec-pinned subset. `tinyloop` is that second
runtime, kept deliberately tiny so it can be read end-to-end.
