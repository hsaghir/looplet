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

## Sync ↔ async parity

<!-- If this touches the loop, how did you keep sync and async in sync? -->

- [ ] Not applicable (no loop change).
- [ ] Behavior mirrored in both `composable_loop` and `async_composable_loop`.
- [ ] Tests cover both paths.

## Checklist

- [ ] `uv run pytest` passes.
- [ ] `uv run ruff check .` reports no new issues.
- [ ] New public API has docstrings and tests.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` (or change is docs-only).
- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md).
