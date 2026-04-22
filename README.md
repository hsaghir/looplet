# looplet

![demo — 4-tool data-cleanup loop with a DebugHook trace and a human approval pause](docs/demo.gif)

[![CI](https://github.com/hsaghir/looplet/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/looplet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hsaghir/looplet/branch/master/graph/badge.svg)](https://codecov.io/gh/hsaghir/looplet)
[![PyPI version](https://img.shields.io/pypi/v/looplet.svg)](https://pypi.org/project/looplet/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](ROADMAP.md)

**A small, framework-agnostic Python library for building LLM agents that call tools in a loop.**
It hands you a `for step in loop(...):` iterator so you can observe, filter, or interrupt
*any* step — no graph DSL, no subclassing, no vendor lock-in. **Zero runtime dependencies.**

```python
from looplet import composable_loop

for step in composable_loop(llm=llm, tools=tools, task=task, config=cfg, state=state):
    print(step.pretty())          # → "#1 ✓ search(query='…') → 12 items [182ms]"
    if step.tool_result.error:
        break                     # your loop, your control flow
```

```bash
pip install looplet               # core — zero third-party packages pulled in
pip install "looplet[openai]"     # works with OpenAI, Ollama, Together, Groq, vLLM, …
pip install "looplet[anthropic]"  # or Anthropic directly
```

---

## Why it exists

Most agent frameworks give you `agent.run(task)` and a black box. When the
agent does something wrong at step 7, you can't step in between step 6 and
step 8. You end up forking the library or writing a second agent to babysit
the first.

`looplet` does the opposite: **the loop is the whole product, and hooks are
the whole API.** Every tool call is a `Step` object you can print, save, or
diff. Every decision the loop makes — what goes in the next prompt, whether
to compact context, whether to dispatch a dangerous tool, whether to stop —
is a `Protocol` method you implement in 3 lines. Hooks compose without
inheritance. Nothing is hidden.

That one design choice is where the library's three practical superpowers
come from:

* **Shape agent behaviour** without forking — a 10-line hook can redact PII
  from every prompt, inject retrieved docs, rewrite tool arguments, or
  rate-limit calls to a single tool. Hooks are the extension point the
  framework *can't* close off because the loop itself is built on them.
* **Manage context on your terms** — `compact_chain(Prune, Summarize,
  Truncate)` is three hooks you wire together. Swap the strategy, change
  the budget, fire on a different threshold — no monkey-patching.
* **Debug and eval without a second tool** — `step.pretty()` is a
  human-readable trace, `ProvenanceSink` dumps every prompt the LLM saw
  plus every tool result into a diff-friendly directory, and pytest-style
  `eval_*` functions turn that trace into a regression suite. Your debug
  output *is* your eval harness.

It's what you'd build if you wrote an agent once, got tired of fighting
the framework, and decided the framework was the problem.

---

## Your first agent (60 seconds)

```python
from looplet import (
    BaseToolRegistry, DefaultState, LoopConfig, ToolSpec, composable_loop,
)
from looplet.backends import OpenAIBackend
from openai import OpenAI
import os

llm = OpenAIBackend(
    OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    ),
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
)

tools = BaseToolRegistry()
tools.register(ToolSpec(
    name="greet", description="Greet someone.",
    parameters={"name": "str"},
    execute=lambda *, name: {"greeting": f"Hello, {name}!"},
))
tools.register(ToolSpec(
    name="done", description="Finish.",
    parameters={"answer": "str"},
    execute=lambda *, answer: {"answer": answer},
))

for step in composable_loop(
    llm=llm, tools=tools,
    state=DefaultState(max_steps=5),
    config=LoopConfig(max_steps=5),
    task={"goal": "Greet Alice and Bob, then finish."},
):
    print(step.pretty())
```

Works out of the box with any OpenAI-compatible endpoint. No Claude-only
SDK, no pydantic schema gymnastics, no LangChain memory objects.

Try it on your laptop against a local Ollama in three lines:

```bash
OPENAI_BASE_URL=http://127.0.0.1:11434/v1 \
OPENAI_API_KEY=ollama OPENAI_MODEL=llama3.1 \
python -m looplet.examples.hello_world
```

---

## What you get

| Capability | Why it matters |
| --- | --- |
| **Composable loop** — `composable_loop` yields `Step`s, hooks layer behaviour per-phase. | You can see every tool call *and* intercept it before it happens. |
| **Tool registry** — `ToolSpec` + JSON-schema, concurrent batching, auto-`ctx` threading, typed `ToolError`. | Tools are data, not classes. Adding one is one `register()` call. |
| **Permissions** — declarative `ALLOW/DENY/ASK` rules with arg matchers, audit log. | You can fail-closed on destructive tools without editing the tool. |
| **Context management** — `compact_chain(Prune, Summarize, Truncate)` fires on budget pressure. | Long sessions don't crash with *prompt too long*. You choose the strategy. |
| **Checkpoints** — step-by-step JSON snapshots; `resume_loop_state()` reconstructs the run. | Kill the process, restart, resume at step N. Works with crashes *and* approvals. |
| **Human approval** — `ApprovalHook` suspends the loop for out-of-band sign-off, resumes with injected context. | CI-safe automation, Slack-bot sign-off, real audit trails. |
| **Provenance** — `ProvenanceSink` dumps every prompt the LLM saw + every tool result, in a diff-friendly directory. | Replay a failing run exactly. Diff two runs at the prompt level. |
| **Evals** — pytest-style `eval_*` functions discovered and batched by a CLI. | Your debug output becomes your regression suite. |
| **MCP + skills** — `MCPToolAdapter` without the official SDK; `Skill` bundles tools + prompt + memory. | Use MCP servers without pulling in their dependency graph. |
| **Backends** — sync / async / streaming adapters for OpenAI and Anthropic; `LLMBackend` protocol for your own. | Swap providers in one line. Local-first is a first-class citizen. |

Every public symbol has a docstring and the package ships a `py.typed`
marker so your editor knows the types.

---

## How it compares

|                                          | looplet | claude-agent-sdk | strands-agents | pydantic-ai | langgraph |
| ---------------------------------------- | ----------- | ---------------- | -------------- | ----------- | --------- |
| **You own the loop (iterator)**          | ✅ `for step in loop(...)` | ❌ async stream | ❌ closed `agent()` | ❌ `run_sync()` | ❌ graph |
| **Provider-agnostic**                    | ✅ | ❌ Claude-only | ✅ | ✅ | ✅ |
| **No subprocess / bundled binary**       | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Hooks as `Protocol` objects**          | ✅ | ⚠️ dict callbacks | ⚠️ inheritance | ⚠️ `Capability` | ⚠️ nodes |
| **Fail-closed permissions**              | ✅ built in | ⚠️ hooks only | ❌ | ⚠️ deferred tools | ❌ |
| **Crash-resume checkpoints**             | ✅ | ❌ | ❌ | ⚠️ add-on | ✅ |
| **Built-in evals**                       | ✅ pytest-style | ❌ | ❌ | ❌ | ❌ |
| **OSI license**                          | Apache-2.0 | Anthropic terms | Apache-2.0 | MIT | MIT |
| **Core runtime deps**                    | **0** | CLI binary | several | many | many |

Numbers on import time and dependency footprint: [docs/benchmarks.md](docs/benchmarks.md).
On a fresh Python 3.11 venv, `looplet` cold-imports in 289 ms against 1.9–4.0 s
for the alternatives.

---

## Examples

All three real-LLM examples read `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
`OPENAI_MODEL` from the environment. Point them at Ollama or any
OpenAI-compatible endpoint.

```bash
python -m looplet.examples.hello_world                            # 30-line starter
python -m looplet.examples.coding_agent "implement fizzbuzz"      # bash/read/write/edit/grep
python -m looplet.examples.coding_agent --trace ./traces/         # save full trajectory
python -m looplet.examples.data_agent --clean                     # approval + compact + checkpoints
python -m looplet.examples.data_agent --resume                    # resume from last checkpoint
```

Plus [`scripted_demo.py`](src/looplet/examples/scripted_demo.py) —
a scripted `MockLLMBackend` run used only to record the GIF above.
Not a usage reference.

---

## Learn more

| Doc | What's in it |
| --- | --- |
| [docs/tutorial.md](docs/tutorial.md) | Build your first agent in 5 steps |
| [docs/hooks.md](docs/hooks.md) | Writing and composing hooks |
| [docs/evals.md](docs/evals.md) | pytest-style agent evaluation |
| [docs/provenance.md](docs/provenance.md) | Capturing prompts + trajectories |
| [docs/recipes.md](docs/recipes.md) | Ollama, OTel, MCP, cost accounting, checkpoints |
| [docs/benchmarks.md](docs/benchmarks.md) | Cold-import time & dep footprint vs alternatives |
| [ROADMAP.md](ROADMAP.md) | What's planned, what's frozen, what's out of scope |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, conventions, PR checklist |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

---

## Stability

`looplet` follows [SemVer](https://semver.org/). Pre-`1.0`, minor versions
may introduce breaking changes as the design stabilises — pin conservatively:

```toml
looplet>=0.1.7,<0.2
```

See [ROADMAP.md § v1.0 API contract](ROADMAP.md#v10-api-contract) for the
frozen surface and the path to `1.0`.

## Contributing

Contributions welcome — bug reports, docs, backends, examples, evals.
Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/good-first-issues.md](docs/good-first-issues.md). Security issues
go through [SECURITY.md](SECURITY.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
