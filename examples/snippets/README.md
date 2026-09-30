# Cartridge snippets

Each subdirectory shows one operation that becomes routine once an
agent is a cartridge artifact rather than a Python program.

Each snippet states its prerequisites for a fresh checkout of the
[looplet](https://github.com/hsaghir/looplet) repo. Most are 20–60
lines. They are designed to be read in 30 seconds each.

## The list

| # | Snippet | What it shows |
|---|---|---|
| 01 | [inheritance](01_inheritance/) | Compose a new agent from an existing one with `extends:` |
| 02 | [ablation](02_ablation/) | Compare static properties of copied cartridge mutations; no performance claim |
| 03 | [diff](03_diff/) | Diff two cartridges and see the change *category* in the path |
| 04 | [subagent](04_subagent/) | Invoke an entire cartridge as a single tool call |
| 05 | [factory](05_factory/) | Have an agent build another agent from a one-paragraph brief |
| 06 | [registry](06_registry/) | Pull and list cartridges from a directory or git repo |
| 07 | [admission](07_admission/) | Refuse to load a cartridge that violates a policy |
| 08 | [portability](08_portability/) | Compare preset, sub-loop, and fresh scripted execution with hooks and resources intact |
| 09 | [lifecycle](09_lifecycle/) | Tag trajectories with cartridge identity for reproducibility |
| 10 | [evolution](10_evolution/) | Toy static mutation search; not evidence of behavioral improvement |
| 11 | [quality gate](11_quality_gate/quality_gate.py) | Reject completion when a host-owned check fails |

Each subdirectory contains its own README explaining what to read,
what to run, and which property of the artifact boundary it
demonstrates.
