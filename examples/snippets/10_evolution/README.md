# 10 — Observability-driven autonomous evolution

The strongest demonstration that the artifact boundary matters: an
agent can **edit another agent's cartridge**, evaluate the result, and
keep the change if it improves a metric. AHE (Lin et al., 2026)
showed this lifts pass@1 on Terminal-Bench~2 by 7+ percentage
points; their first "observability pillar" is the file-level
component representation we name as the cartridge.

This snippet is a **minimal proof-of-concept**: a 100-line loop that
mutates one cartridge component per iteration (prompt edit, tool add,
hook add, config tweak), runs a tiny in-process eval, and records
which mutations improved a static score.

```bash
uv run python examples/snippets/10_evolution/evolve.py \
    examples/hello.workspace 5
```

The eval here is intentionally trivial (number of tools + prompt
clarity + max_steps in a healthy range) so the snippet is
**offline and runs in seconds**. The mechanism is what matters: an
*agent action* is a *cartridge mutation*, and the action space is
**explicit** because the cartridge is files. Drop in a real eval
harness (pytest, SWE-bench, etc.) and the same loop becomes AHE.
