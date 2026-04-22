# Benchmarks

Honest, reproducible measurements. Every number here can be regenerated
with a single command and a clean Python 3.11 venv.

> **What this page is:** a defence of the one-dependency, loop-first
> design choice against frameworks that give up more to do more.
>
> **What this page is not:** an end-to-end agent benchmark. None of
> these numbers say anything about *how well* any framework solves a
> task. They measure *what you pay for loading it*.

All runs: Python 3.11.13, Linux x86_64, uv-managed venvs, PyPI wheels
from 2026-04-21. See the `scripts/` directory for the scripts —
they're short enough to read.

## Cold import time

The time from `python -c "import <pkg>"` to interpreter exit. Measured
as median wall-clock of 9 fresh subprocess runs (so no warm bytecode
cache).

```bash
python scripts/bench_cold_import.py --runs 9 --markdown
```

| Framework | Version | Median cold import | vs looplet |
| --- | --- | ---: | ---: |
| `looplet` | 0.1.7 | **289 ms** | — |
| `strands-agents` | 1.36.0 | **1 885 ms** | 6.5× |
| `langgraph` | 1.1.9 | **2 294 ms** | 7.9× |
| `claude-agent-sdk` | 0.1.65 | **2 409 ms** | 8.3× |
| `pydantic-ai` | 1.85.1 | **3 975 ms** | 13.8× |

Why it matters: agents are increasingly invoked as CLI tools,
serverless functions, and hot-reload dev loops. A 3-second import tax
is the difference between "snappy" and "go get coffee" for every
invocation that doesn't reuse a warm process. `looplet`'s
single-dependency core (`typing_extensions` for Python <3.12) leaves
no room for surprises.

## Dependency footprint

Count of third-party packages installed into a fresh venv by
`pip install <pkg>`, minus `pip`, `setuptools`, and `wheel`.

```bash
python scripts/bench_dep_footprint.py --markdown
```

| Install | Packages installed |
| --- | ---: |
| `pip install looplet` | **2** |
| `pip install looplet[all]` | **20** |
| `pip install claude-agent-sdk` | **30** |
| `pip install langgraph` | **31** |
| `pip install strands-agents` | **49** |
| `pip install pydantic-ai` | **144** |

`looplet[all]` pulls in the official `openai` and `anthropic` SDKs
plus their transitive deps. If you bring your own HTTP client, stay on
core `looplet` and write a 20-line `LLMBackend` adapter — see
[`docs/recipes.md`](recipes.md).

Why it matters: every package in your environment is a potential
supply-chain surface, a potential version-conflict, and a potential
wheel-download delay for your container or Lambda. 144 transitive
dependencies is not a free choice — it's an ambient cost.

## What we don't claim

`looplet` is not *faster at running tools*, *better at prompting*, or
*more accurate at any task* than the frameworks above. Per-step
latency is dominated by the LLM round-trip; per-task accuracy depends
on prompts, tools, and the model you choose.

What `looplet` *is* is small, cold-starts in under a third of a second,
and hands you a `for step in loop(...):` iterator so you can observe
and interrupt any step without learning a new graph DSL.

If that's the trade-off you want, these numbers are your defence
when someone asks why.

## Reproducing

```bash
# One-time: clean Python 3.11 venv with all four frameworks.
uv venv /tmp/bench_env --python 3.11 -q
uv pip install --python /tmp/bench_env/bin/python \
    looplet langgraph claude-agent-sdk pydantic-ai strands-agents

# Cold-import numbers.
/tmp/bench_env/bin/python scripts/bench_cold_import.py \
    --python /tmp/bench_env/bin/python --runs 9 --markdown

# Dependency footprint (creates its own throwaway venvs).
python scripts/bench_dep_footprint.py --markdown
```

Both scripts exit non-zero on install/import failure; that's the only
signal they're broken.

## History

| Date | looplet | claude-agent-sdk | pydantic-ai | langgraph | strands-agents |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-04-21 | 289 ms / 2 pkg | 2 409 ms / 30 pkg | 3 975 ms / 144 pkg | 2 294 ms / 31 pkg | 1 885 ms / 49 pkg |

Re-run these when any dependency's major version ships — deps tend to
move up, not down.
