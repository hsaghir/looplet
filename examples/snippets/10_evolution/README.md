# 10 — Downstream cartridge-search mechanics

A separate optimization system can edit a cartridge, evaluate a candidate,
and decide whether to keep it. This snippet shows only that file-copy/mutation
control flow. Search algorithms, statistics, candidate generation, and
promotion policy are deliberately **not** Looplet core concerns.

The script is a small, offline toy. It changes one cartridge component per
iteration and ranks candidates with a hand-written static heuristic.

```bash
uv run python examples/snippets/10_evolution/evolve.py \
    examples/hello.cartridge 5
```

## What this does not prove

The heuristic counts tools, prompt words, and a `max_steps` range. It does not
run the agent or observe a task outcome, so a higher score is **not evidence of
a better harness**. The candidate also carries its own files, so a real
promotion system must keep holdout cases, collectors, and expected data in
host-owned storage the candidate cannot edit.

Use this snippet to understand the downstream mutation loop, then replace the
heuristic with repeated task runs, outcome-grounded graders, an explicit
statistical policy, and protected holdouts. Keep that optimization system in a
downstream project; promote only generic evidence/cartridge primitives into
Looplet after they prove useful across strategies.
