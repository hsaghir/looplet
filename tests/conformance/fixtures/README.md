# Conformance Fixtures for Cartridge-Spec v1.0

Each subdirectory is a minimal cartridge plus an `expected.json`
describing the loader output a v1.0 conformant runtime must produce.
The four seed fixtures cover the new v1.0 slots.

## Fixtures

| Directory | What it pins |
|---|---|
| `01_minimal/`                  | Smallest legal cartridge: manifest, prompt, done tool. |
| `02_permissions/`              | Declarative `permissions:` compiles into a `PermissionHook`. |
| `03_model_and_output_schema/`  | Structured `model:` block + `output_schema:` on done. |
| `04_long_term_memory/`         | `memory/long_term.md` is auto-loaded as a memory source. |

## Adding fixtures

1. Create `tests/conformance/fixtures/<NN>_<name>/cartridge/` with the
   cartridge files.
2. Create `tests/conformance/fixtures/<NN>_<name>/expected.json` using
   the same key set as the existing fixtures.
3. The parametric test in `test_conformance.py` will pick it up
   automatically.

The summary the test compares against is intentionally narrow.
Implementation details (live Python objects, hook ordering beyond
what the cartridge declares, etc.) are NOT pinned by v1.0.

## Status

Seed only — v1.0 conformance is documentation, not enforcement. v2 will
mandate this suite as a release criterion.
