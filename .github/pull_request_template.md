<!--
Thanks for contributing! Please fill out the sections below.
Delete any section that doesn't apply.
-->

## Summary

<!-- What does this PR do, in one or two sentences? -->

## Motivation

<!-- Why is this change needed? Link any related issues. -->

Fixes #

## Changes

<!-- Bullet list of the most important changes. -->

-

## Behavioral contract

<!--
What observable outcome is preserved or changed? Link the regression test,
case + collector + grader, or explain why this is not a behavioral change.
Do not use a preferred tool sequence as a proxy when the outcome is inspectable.
-->

## Core-boundary check

<!--
If this adds a core concept, why can it not be a hook, tool, cartridge, recipe,
optional extra, or downstream package? Write "Not applicable" when it does not.
-->

## Sync ↔ async parity

<!-- If this touches the loop, how did you keep sync and async in sync? -->

- [ ] Not applicable (no loop change).
- [ ] Behavior mirrored in both `composable_loop` and `async_composable_loop`.
- [ ] Tests cover both paths.

## Checklist

- [ ] `uv run pytest` passes.
- [ ] `uv run ruff check .` reports no new issues.
- [ ] New public API has docstrings and tests.
- [ ] Behavioral changes have outcome-grounded regression evidence.
- [ ] Any candidate-editable eval is not treated as a protected release oracle.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` (or change is docs-only).
- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md).
