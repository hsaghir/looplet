# openharness

[![CI](https://github.com/hsaghir/openharness/actions/workflows/ci.yml/badge.svg)](https://github.com/hsaghir/openharness/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hsaghir/openharness/branch/master/graph/badge.svg)](https://codecov.io/gh/hsaghir/openharness)
[![PyPI version](https://img.shields.io/pypi/v/openharness.svg)](https://pypi.org/project/openharness/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](ROADMAP.md)

> **Not [`pydantic-ai-harness`](https://github.com/pydantic/pydantic-ai-harness)** — that's a *capability* library for pydantic-ai.
> `openharness` is a framework-agnostic *loop* library. Works with any LLM backend, one dependency.

**The tool-calling loop you can actually step through.**

Every agent framework gives you `agent.run()`. `openharness` gives you
`for step in loop(...):` — and that's the whole product.

```python
from openharness import composable_loop

for step in composable_loop(llm=llm, tools=tools, task=task, ...):
    print(step.pretty())          # → "#1 ✓ search(query='…') → 12 items [182ms]"
    if step.tool_result.error:
        break                     # your loop, your control flow
```

## Who this is for

- You're building an agent for a non-trivial domain (security,
  research, ops, robotics) and framework magic is in your way.
- You want to iterate on *behaviour at a single step* — add a hook,
  filter a result, veto a tool call — without learning a graph DSL.
- You need **vendor independence**: no Claude-only, no AWS-only, no
  "everything through Pydantic."
- You're a library author who wants to embed a reactive loop in your
  own package without pulling in dozens of transitive deps.

## How it compares

|                                          | openharness | claude-agent-sdk | strands-agents | pydantic-ai | langgraph |
| ---------------------------------------- | ----------- | ---------------- | -------------- | ----------- | --------- |
| **You own the loop (iterator)**          | ✅ `for step in loop(...)` | ❌ async stream | ❌ closed `agent()` | ❌ `run_sync()` | ❌ graph |
| **Provider-agnostic**                    | ✅ | ❌ Claude-only | ✅ | ✅ | ✅ |
| **No subprocess / bundled binary**       | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Hooks as `Protocol` objects**          | ✅ | ⚠️ dict callbacks | ⚠️ inheritance | ⚠️ `Capability` | ⚠️ nodes |
| **Fail-closed permissions**              | ✅ built in | ⚠️ hooks only | ❌ | ⚠️ deferred tools | ❌ |
| **Crash-resume checkpoints**             | ✅ | ❌ | ❌ | ⚠️ add-on | ✅ |
| **Built-in evals**                       | ✅ pytest-style | ❌ | ❌ | ❌ | ❌ |
| **OSI license**                          | Apache-2.0 | Anthropic terms | Apache-2.0 | MIT | MIT |
| **Core runtime deps**                    | **1** | CLI binary | several | many | many |

## Install

```bash
pip install openharness                    # core only
pip install "openharness[openai]"          # + OpenAI / Ollama / any OAI-compat
pip install "openharness[anthropic]"       # + Anthropic
pip install "openharness[all]"             # both
```

## 60-second example

```python
from openharness import composable_loop, LoopConfig, DefaultState, BaseToolRegistry, ToolSpec
from openharness.backends import OpenAIBackend
from openai import OpenAI

llm = OpenAIBackend(OpenAI(), model="gpt-4o-mini")

tools = BaseToolRegistry()
tools.register(ToolSpec(name="greet", description="Greet someone.",
                        parameters={"name": "str"},
                        execute=lambda *, name: {"greeting": f"Hello, {name}!"}))
tools.register(ToolSpec(name="done", description="Finish.",
                        parameters={"answer": "str"},
                        execute=lambda *, answer: {"answer": answer}))

for step in composable_loop(
    llm=llm, tools=tools,
    state=DefaultState(max_steps=5),
    config=LoopConfig(max_steps=5),
    task={"goal": "Greet Alice, then finish."},
):
    print(step.pretty())
```

Runs against any OpenAI-compatible endpoint (OpenAI, Ollama, Together,
Groq, vLLM, …). Set `OPENAI_BASE_URL` and `OPENAI_MODEL` to your
provider.

## What `openharness` gives you

- **Composable loop** — `composable_loop` yields `Step`s you can
  observe or interrupt. Hooks (`pre_prompt`, `pre_dispatch`,
  `post_dispatch`, `check_done`, `should_stop`, `on_loop_end`) layer
  behaviour without forking the loop.
- **Tool registry** — `ToolSpec` + JSON-schema catalog, concurrent
  batching, auto-`ctx` threading, structured `ToolError` categories.
- **Permissions** — declarative `PermissionEngine` with ALLOW/DENY/ASK
  rules, argument matchers, human-in-the-loop handler, audit log.
- **Context management** — `compact_chain` of prune / summarise /
  truncate strategies triggered on budget pressure.
- **Checkpoints** — `FileCheckpointStore` + `resume_loop_state()`
  preserve session log, conversation, step offset, and budgets across
  crash-resume.
- **Provenance** — `ProvenanceSink` captures the exact prompts the LLM
  saw and the trajectory the loop took, in a diff-friendly directory.
- **Evals** — pytest-style `eval_*` functions discovered, batched, and
  run from the CLI. Your debug output becomes your regression suite.
- **MCP + skills** — `MCPToolAdapter` bridges MCP servers without the
  MCP SDK; `Skill` bundles tools + prompt fragment + context.
- **Backends** — sync / async / streaming adapters for OpenAI and
  Anthropic. Bring your own via the `LLMBackend` protocol.

## Learn more

| Doc                                                    | What's in it                                         |
| ------------------------------------------------------ | ---------------------------------------------------- |
| [docs/tutorial.md](docs/tutorial.md)                   | Build your first agent in 5 steps                    |
| [HOOK_GUIDE.md](HOOK_GUIDE.md)                         | Writing and composing hooks                          |
| [docs/evals.md](docs/evals.md)                         | pytest-style agent evaluation                        |
| [PROVENANCE_GUIDE.md](PROVENANCE_GUIDE.md)             | Capturing prompts + trajectories                     |
| [docs/recipes.md](docs/recipes.md)                     | Ollama, OTel, MCP, cost accounting, checkpoints      |
| [ROADMAP.md](ROADMAP.md)                               | What's planned, what's frozen, what's out of scope   |
| [CONTRIBUTING.md](CONTRIBUTING.md)                     | Dev setup, conventions, PR checklist                 |
| [CHANGELOG.md](CHANGELOG.md)                           | Release notes                                        |

Every public symbol has a docstring and the package ships a `py.typed`
marker.

## Examples

```bash
python -m openharness.examples.hello_world                            # 30-line starter
python -m openharness.examples.coding_agent "implement fizzbuzz"      # bash/read/write/edit/grep
python -m openharness.examples.coding_agent --trace ./traces/         # save full trajectory
```

## Stability

`openharness` follows [SemVer](https://semver.org/). Pre-`1.0`, minor
versions may introduce breaking changes as the design stabilises —
pin conservatively:

```toml
openharness>=0.1.6,<0.2
```

See [ROADMAP.md § v1.0 API contract](ROADMAP.md#v10-api-contract) for
what's frozen and the path to `1.0`.

## Contributing

Contributions welcome — bug reports, docs, backends, examples, evals.
Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/good-first-issues.md](docs/good-first-issues.md). Security
issues go through [SECURITY.md](SECURITY.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
