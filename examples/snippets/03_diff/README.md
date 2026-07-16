# 03 — Semantic diff

When the agent is files, *the path of an edit reveals its category*.
A cartridge pull request that touches `tools/lint/` adds a tool; one
that touches `hooks/08_PermissionHook/` adds a permission policy; one
that touches `prompts/system.md` edits the contract.

This snippet diffs two cartridges and prints a one-line summary per
changed file, grouped by category derived from the path.

## Try it

Compare two standalone cartridges shipped with looplet:

```bash
uv run python examples/snippets/03_diff/diff_workspaces.py \
    examples/threat_intel.cartridge \
    examples/dep_doctor.cartridge
```

Output is grouped by `# manifest`, `# config`, `# prompt`, `# tool`,
`# hook`, `# resource` — you see what changed *categorically* before
reading any code.

> Note: when one of the cartridges uses `extends:` (inheritance), the
> diff compares physical directory contents, not the resolved view.
> If you want to see what an extending cartridge overrides, run it on
> two materialised cartridges (the four standalone examples above) or
> first materialise the extender (`preset_to_cartridge(...)`).

The same `diff -ruN` against an equivalent code-defined agent would
land in one monolithic Python file (see the position paper, §6.1).
