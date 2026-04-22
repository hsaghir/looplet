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

## What you get — one diagram

`looplet` is just a `for`-loop you own, three loop phases, and a handful
of **`Protocol` methods** you can implement in a few lines to change any
part of the loop. That's the whole mental model:

```mermaid
%%{init: {'theme':'base', 'themeVariables': {
  'primaryColor':'#1e40af',
  'primaryTextColor':'#f8fafc',
  'primaryBorderColor':'#60a5fa',
  'lineColor':'#64748b',
  'fontSize':'14px'
}}}%%
flowchart TB
    classDef user fill:#0f172a,stroke:#475569,stroke-width:2px,color:#f1f5f9
    classDef phase fill:#1e40af,stroke:#60a5fa,stroke-width:2px,color:#f8fafc
    classDef done fill:#065f46,stroke:#34d399,stroke-width:2px,color:#ecfdf5
    classDef hook fill:#fde68a,stroke:#b45309,stroke-width:1.5px,color:#451a03
    classDef step fill:#4338ca,stroke:#a5b4fc,stroke-width:2px,color:#eef2ff

    User["🧑‍💻 <b>your</b> <tt>for step in loop(...)</tt><br/><i>you own the control flow</i>"]:::user

    subgraph LL[" ⚙️  composable_loop "]
      direction TB
      P("🗣️  prompt LLM"):::phase
      D("🛠️  dispatch tool"):::phase
      DD{{"🎯 done?"}}:::done
      P --> D --> DD
      DD -- "no" --> P
    end

    User ==> LL
    DD == "yes" ==> Step["📦 <b>Step</b> — <tt>step.pretty()</tt>"]:::step
    Step ==> User

    H1["🧩 <b>pre_prompt</b><br/>─────────────<br/>redact · inject context<br/>compact · retry"]:::hook
    H2["🛡️ <b>pre_dispatch</b><br/>─────────────<br/>permissions · approval<br/>rewrite args · cache"]:::hook
    H3["📝 <b>post_dispatch</b><br/>─────────────<br/>trace · metrics<br/>checkpoint · provenance"]:::hook
    H4["🏁 <b>check_done / should_stop</b><br/>─────────────<br/>custom stop rules<br/>max steps · budget"]:::hook

    H1 -.-> P
    H2 -.-> D
    H3 -.-> D
    H4 -.-> DD

    linkStyle 0,1,2,3 stroke:#60a5fa,stroke-width:2px
    linkStyle 4,5,6 stroke:#a5b4fc,stroke-width:3px
    linkStyle 7,8,9,10 stroke:#d97706,stroke-width:2px,stroke-dasharray: 6 4
```

Every amber box is a `Protocol` method. A hook is any object that
implements one or more of them — no base class, no inheritance:

```python
class RedactPII:
    def pre_prompt(self, state, log, ctx, step):
        return _scrub_emails(ctx)          # mutates the next LLM prompt

class RetryFlakyTool:
    def pre_dispatch(self, state, log, tc, step):
        if tc.tool == "web_search" and state.last_error:
            return Deny("retry with backoff", retry=True)

for step in composable_loop(..., hooks=[RedactPII(), RetryFlakyTool()]):
    ...
```

Ship-ready hooks already wired in: `ApprovalHook`, `PermissionHook`,
`CheckpointHook`, `ContextPressureHook`, `ThresholdCompactHook`,
`ProvenanceSink`, `TracingHook`, `MetricsHook`, `EvalHook`, plus the
`compact_chain(Prune, Summarize, Truncate)` context strategy. Use any,
all, or none — and [drop in your own](docs/hooks.md) in 10 lines.

---

## When should you reach for `looplet`?

**Use it when you want to build your own agent loop and actually own
the details.** Concretely:

* You need to **insert logic at an exact phase** of the loop — before
  the prompt is built, before a tool is dispatched, after a tool
  returns — without forking a framework.
* You need to **swap context-management strategy at runtime** (prune,
  summarize, truncate, your own) without losing the rest of your stack.
* You need the loop to **pause for human approval**, then resume where
  it left off when approval arrives.
* You want **first-class debugging and evaluation** — a printable
  `Step`, a prompt-level provenance dump, pytest-style `eval_*`
  functions — without bolting on a second tool.
* You want **zero runtime dependencies** and a loop that cold-imports
  in ~300 ms (numbers in [docs/benchmarks.md](docs/benchmarks.md)).

**Don't reach for `looplet` if** you want `agent.run(task)` to handle
everything and return a string, or if you want a visual graph DSL — a
higher-level framework will feel more natural and the overlap in
features won't be worth the extra control `looplet` gives you.

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
