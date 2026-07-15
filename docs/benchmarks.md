# Evidence and benchmarks

Looplet's primary claim is not that a loop library makes a model smarter. It
is that an owned harness can be **inspected, changed, and regression-tested**.
The evidence is organized accordingly: a controlled behavioral proof first,
then model-controlled harness comparisons, then package-cost measurements.

Every study below names its variable and limitations. Historical numbers are
snapshots, not timeless leaderboard claims.

## 1. Behavioral proof: failure → regression contract

The repository includes a network-free experiment with two harness versions:

- the scripted model makes the same `publish_report` and `done` decisions;
- v1 contains one arithmetic bug and the required outcome grader fails;
- captured-response replay executes the fixed v2 tool in a fresh workspace;
- an independent collector reads `report.json`; the same grader passes.

| Variable | Held fixed / changed |
| --- | --- |
| Model calls | Fixed captured responses; no provider or network |
| Tool decisions | Fixed: `publish_report → done` |
| Harness code | One line changes: addition → subtraction |
| Outcome | Profit changes from 200 → 40 |
| Required eval | 0.00 fail → 1.00 pass |

Run it with `uv run python examples/regression_demo/run_demo.py`. Read the
[full proof and limitations](regression-demo.md) before treating replay as an
experiment design: tools and side effects execute again, so this is
captured-response replay, not deterministic simulation.

## 2. Model-controlled harness comparison

The optional
[coder harness benchmark](https://github.com/hsaghir/looplet/tree/master/benchmarks/coder_vs_agents)
compares `examples/coder.cartridge` with GitHub Copilot CLI while holding the
underlying model family fixed (`claude-sonnet-4.6` in the recorded snapshot).
The purpose is to compare scaffolding—prompt, tools, loop control, context, and
startup—not model intelligence.

| Recorded suite | Looplet coder | Copilot CLI | What the snapshot supports |
| --- | ---: | ---: | --- |
| 9 short coding/non-coding tasks with deterministic verifiers | 9/9 | 9/9 | Correctness parity on this small suite; Looplet averaged 18.3s vs 23.1s. |
| 4 hard coding tasks with hidden verifiers | 4/4 | 4/4 | No observed correctness separation. |
| 6 open-ended tasks, blind LLM judges | 18.0/20 after a one-line prompt change | 18.6/20 | Copilot retained a small prose-quality edge; the result is judge- and sample-sensitive. |

Reports preserve task-level results and methodology:

- [short-suite report](https://github.com/hsaghir/looplet/blob/master/benchmarks/coder_vs_agents/REPORT.md)
- [hard-suite report](https://github.com/hsaghir/looplet/blob/master/benchmarks/coder_vs_agents/HARD_REPORT.md)
- [open-ended report](https://github.com/hsaghir/looplet/blob/master/benchmarks/coder_vs_agents/SOFT_REPORT.md)
- [prompt comparison](https://github.com/hsaghir/looplet/blob/master/benchmarks/coder_vs_agents/PROMPTS_COMPARISON.md)

### Limits of this comparison

- Same model family does not guarantee identical serving stacks, hidden
    provider policy, caching, or sampling implementation.
- Nine and four tasks are demonstration-sized samples, not population
    estimates. There are no confidence intervals.
- Looplet token counts are `chars / 4` estimates because the proxy omitted
    usage; Copilot counts are self-reported. Compare direction, not precision.
- Open-ended scores are single-generation LLM-judge results. Small differences
    may be noise even with blind, counterbalanced judging.
- The suites do not represent huge repositories, external services, or
    multi-hour autonomous operation.

The honest result is modest: a small, owned cartridge was competitive on these
tasks. It does not establish that Looplet is universally faster, cheaper, or
more capable than a turnkey coding agent.

## 3. Runtime footprint

Core Looplet 0.2.0 declares **zero third-party runtime dependencies**. Provider
SDKs are optional extras. This is a current package property, not a benchmark
inference.

The table below is an archived environment snapshot from 2026-04-21 (Python
3.11.13, Linux x86_64, fresh uv-managed environments, then-current PyPI
versions). "Packages installed" includes the target package and excludes pip,
setuptools, and wheel.

| Install in that snapshot | Packages installed |
| --- | ---: |
| `pip install looplet` | **1** (Looplet itself; zero runtime dependencies) |
| `pip install looplet[all]` | 20 |
| `pip install claude-agent-sdk` | 30 |
| `pip install langgraph` | 31 |
| `pip install strands-agents` | 49 |
| `pip install pydantic-ai` | 144 |

In the same archived run, median cold import over nine fresh subprocesses was
289 ms for Looplet 0.1.8. The comparison packages ranged from 1,885 ms to
3,975 ms. Those latency numbers are historical: hardware, OS caches, Python,
wheel versions, and package releases all affect them. Re-run before using them
in a current engineering decision.

## Reproduce instead of repeating the headline

```bash
# Current cold-import snapshot (choose an explicit interpreter/environment).
python scripts/bench_cold_import.py --runs 9 --markdown

# Current dependency snapshot (creates throwaway environments).
python scripts/bench_dep_footprint.py --markdown

# Behavioral proof: no model, network, or external CLI.
uv run python examples/regression_demo/run_demo.py
```

The model-controlled coder comparison has external prerequisites and
environment knobs documented in its
[benchmark README](https://github.com/hsaghir/looplet/tree/master/benchmarks/coder_vs_agents#readme).
Commit the raw environment metadata and task-level output when publishing a new
snapshot; do not silently replace historical results.

## What these results do not claim

They do not show that Looplet improves model reasoning, guarantees safer
agents, makes arbitrary replays deterministic, or outperforms graph runtimes
and turnkey agents on their intended workloads. They show three narrower
properties:

1. a harness failure can become an executable behavioral contract;
2. a compact, owned harness can be competitive in a small same-model study;
3. the core package imposes no third-party runtime dependency tree.
