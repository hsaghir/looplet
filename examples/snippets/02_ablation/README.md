# 02 - Cartridge ablation

An *ablation matrix* sweeps several cartridge mutations across several
tasks and records what changes. With agents-as-files, each cell is a
function that mutates a cartridge directory; the matrix is mechanical.

This snippet is a **70-line driver** that ablates the shipped coder
cartridge 5 ways and prints a small markdown table. It does **not**
run an LLM; it only loads each mutated cartridge and reports static
properties (number of tools, prompt length, max_steps). That keeps
the snippet runnable in CI in under a second while making the
mechanism legible.

```bash
uv run python examples/snippets/02_ablation/ablate.py
```

For a real ablation matrix that runs the LLM and scores via pytest,
see `paper/data/ablation_experiment/run.py` (private; described in
the position paper).
