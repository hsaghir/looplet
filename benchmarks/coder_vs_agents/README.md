# coder.cartridge vs commercial coding agents — benchmark

A small, reproducible, **model-controlled** comparison of looplet's
[`examples/coder.cartridge`](../../examples/coder.cartridge) against shipping
coding-agent CLIs (primarily **GitHub Copilot CLI**), plus a read-through of
their system prompts.

The point is to compare **harnesses, not models**: every agent runs on the
*same* underlying model (`claude-sonnet-4.6`), so differences come from the
scaffolding — system prompt, tool set, loop, startup cost — not raw model IQ.

> This is an optional research/eval artifact. It is **not** part of looplet's
> runtime and pulls in external tools; nothing here is imported by `looplet`.

## What's inside

| File | What it is |
|---|---|
| `bench.py` | Orchestrator — runs each agent on each task in an isolated workspace, verifies, records metrics |
| `tasks.py` | 9 short tasks (5 coding + 4 non-coding), each with a deterministic verifier |
| `hard_tasks.py` | 4 hard, multi-file/long-horizon coding tasks graded by hidden test suites |
| `soft_tasks.py` | 6 open-ended tasks (design / explanation / reasoning / writing) |
| `looplet_runner.py` | Drives the coder cartridge via `composable_loop` (as `run-workspace` does) + token instrumentation |
| `judge.py` | Blind LLM-as-judge for the open-ended tasks (two neutral judges, counterbalanced A/B order) |
| `report.py` | Renders a results JSON into a Markdown table |
| [`REPORT.md`](REPORT.md) | Short-coding + non-coding suite (9 tasks) |
| [`HARD_REPORT.md`](HARD_REPORT.md) | Hard / long-horizon coding suite (4 tasks) |
| [`SOFT_REPORT.md`](SOFT_REPORT.md) | Non-coding suite with blind judging (6 tasks) |
| [`PROMPTS_COMPARISON.md`](PROMPTS_COMPARISON.md) | System-prompt comparison: coder vs Copilot / Codex / Claude Code / Pi |

Generated artifacts (`runs_*/`, `results_*.json`, `TABLES_*.md`) are
git-ignored — re-running regenerates them.

## Headline findings (same model, `claude-sonnet-4.6`)

| Suite | looplet coder | Copilot CLI | takeaway |
|---|:--:|:--:|---|
| Short coding (9) | 9/9, ~21% faster, leaner context | 9/9 | looplet edge: startup + tokens |
| Hard coding (4) | 4/4 | 4/4 | peers; the model does the work |
| Non-coding (6) | 17.2/20 → **18.0** after a 1-line prompt tweak | 18.6/20 | Copilot edge on open-ended prose; a single answer-quality line closed ~62% of the gap |

Full detail, methodology, and caveats are in the four report files above.

## Requirements

- **GitHub Copilot CLI** on `PATH` (`copilot`), logged in.
- An OpenAI-compatible endpoint for looplet's LLM. To hold the model constant
  we point looplet at a local **[copilot-lm-proxy](https://github.com/hsaghir/copilot-lm-proxy)**
  (default `http://127.0.0.1:19823/v1`), which exposes Copilot's models over
  the OpenAI API. Any OpenAI-compatible endpoint works.
- looplet installed in the repo's `.venv` (the runner uses `.venv/bin/python`).

## Running

```bash
cd benchmarks/coder_vs_agents

# Short suite (9 tasks, both agents)
python bench.py

# Hard suite
BENCH_TASKS=hard_tasks BENCH_OUT=results_hard.json BENCH_RUNS=runs_hard \
  BENCH_MAX_STEPS=40 BENCH_TIMEOUT=600 python bench.py

# Non-coding suite, then blind judging
BENCH_TASKS=soft_tasks BENCH_OUT=results_soft.json BENCH_RUNS=runs_soft python bench.py
python judge.py

# Render a table from any results file
python report.py results_hard.json
```

Environment knobs: `BENCH_MODEL` (default `claude-sonnet-4.6`), `BENCH_TOOLS`
(`looplet,copilot`), `BENCH_PROXY`, `BENCH_MAX_STEPS`, `BENCH_TIMEOUT`,
`BENCH_TASKS`, `BENCH_OUT`, `BENCH_RUNS`. `judge.py` reads the same proxy and
uses `gemini-3.1-pro-preview` + `gpt-5.5` as neutral judges.

## Caveats

- **Harness comparison, not a model comparison** — the model is held constant.
- Token figures for looplet are chars/4 estimates (the proxy strips `usage`);
  Copilot self-reports real tokens. Use them for order-of-magnitude only.
- The open-ended scores come from an LLM judge; they are blind and
  counterbalanced but still single-generation, so treat small deltas as noise.
- Tasks are self-contained and objectively checkable; they don't exercise huge
  existing codebases, external services, or multi-hour autonomy.
