# Contributing to looplet

Thanks for your interest in improving `looplet`! This document lays
out how to set up a dev environment, the conventions we follow, and how
to submit changes.

## Development setup

`looplet` uses [`uv`](https://docs.astral.sh/uv/) for dependency
management. If you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the repo root:

```bash
make install           # uv sync --all-extras (matches CI)
make check             # everything CI runs: lint + format + pyright + pytest
make test              # just the test suite
make install-hooks     # one-time: install a pre-push git hook that runs `make check`
```

The rule is simple: if `make check` passes locally, CI passes.
Run it before every push, or let the pre-push hook do it for you.

## Branching & commits

- Create a feature branch off `master`.
- Write small, focused commits. We aim for commits that tell a story;
  squash-merges are fine on the PR side.
- Include a test for any behavior change; bug fixes should come with a
  regression test.

## Coding conventions

- **Python**: target 3.11+; use modern syntax (`X | Y`, `match`,
  `dataclasses`, `typing.Protocol`).
- **Domain-agnostic core**: the harness must not assume what the agent
  *does*. Domain-specific helpers belong in the user's code, not here.
- **Generalize before adding**: prefer a plain hook, tool, collector, grader,
  cartridge, recipe, or downstream package over a new loop concept. Search,
  statistics, prompt optimization, and domain policy do not belong in core.
- **Evidence over demos**: a behavior claim needs a network-free regression
  test or a reproducible case + collector + grader. Do not grade a preferred
  trajectory when the real outcome can be inspected independently.
- **Protect the oracle**: cartridge evals are useful self-tests. Promotion
  holdouts must remain host-owned when a candidate can edit its cartridge.
- **Fail closed**: permissions, cancellation, and parse recovery must
  default to the safe path. If in doubt, deny / cancel / re-prompt.
- **Sync ↔ async parity**: any behavior added to `composable_loop` must
  have the equivalent in `async_composable_loop`, and vice-versa. Add
  tests for both paths.
- **Avoid bare `except`**: catch the narrowest exception you can.
- **No new runtime dependencies without discussion** - the core runtime
  depends only on the standard library. Optional extras are
  fine under `[project.optional-dependencies]`.
- **Public API surface** - if you add a public symbol, export it from
  `looplet/__init__.py`, document it with a docstring, and add a
  test.

## Testing

- Unit tests live under `tests/`, mirroring the module layout of
  `src/looplet/`.
- Mark fast tests with `@pytest.mark.smoke` (module-level `pytestmark`
  is fine) and slow / network-bound tests with `@pytest.mark.integration`
  or `@pytest.mark.slow`.
- Tests should not require network access by default. Mock the LLM
  backend via the `LLMBackend` protocol.
- Keep individual tests under 1 second where possible. If a test
  takes longer, mark it `slow`.

## Pull request checklist

Before opening a PR, please verify:

- [ ] `make check` passes (lint + format + pyright + pytest).
- [ ] New public API has docstrings and tests.
- [ ] Behavioral changes include an outcome-grounded regression contract (or
  the PR explains why a unit assertion is the right level).
- [ ] The change belongs in core rather than a hook, tool, cartridge, recipe,
  optional extra, or downstream package.
- [ ] `CHANGELOG.md` has an entry under `## Unreleased` describing your
      change (unless it's a docs-only or internal refactor with no
      user-visible effect).
- [ ] Sync and async loops stay in parity (if applicable).

## Reporting bugs

Open an issue on GitHub with:

- What you expected to happen.
- What actually happened (stack trace, log output).
- A minimal reproduction - ideally a single `pytest` test case.
- Your Python version and `looplet` version.

## Security issues

Please **do not** open a public issue for vulnerabilities; see
[SECURITY.md](https://github.com/hsaghir/looplet/blob/master/SECURITY.md) for the private disclosure channel.

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0, the same as the project.
