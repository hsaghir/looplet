---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<p class="hero-eyebrow">Zero deps · Four hooks · One iterator</p>

# The loop is the product.

<p class="hero-sub" markdown>
**looplet** is a `for`-loop you own for LLM tool-calling agents. Yield
every step, intercept every tool call, redact every prompt, eval every
trajectory. No graph DSL, no agent runtime, no hidden state. Works with
any OpenAI-compatible endpoint or Anthropic directly.
</p>

<p class="hero-badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/looplet?label=pypi&color=4F46E5)](https://pypi.org/project/looplet/)
[![Python](https://img.shields.io/pypi/pyversions/looplet?color=4F46E5)](https://pypi.org/project/looplet/)
[![License](https://img.shields.io/badge/license-MIT-4F46E5)](https://github.com/hsaghir/looplet/blob/master/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/hsaghir/looplet/ci.yml?branch=master&label=CI&color=4F46E5)](https://github.com/hsaghir/looplet/actions)
[![GitHub stars](https://img.shields.io/github/stars/hsaghir/looplet?color=4F46E5)](https://github.com/hsaghir/looplet)
</p>

<div class="hero-cta">
  <a href="quickstart/" class="md-button md-button--primary">Quickstart →</a>
  <a href="tutorial/" class="md-button">5-step tutorial</a>
  <a href="https://github.com/hsaghir/looplet" class="md-button">GitHub ⭐</a>
</div>

</div>

=== "Run one"

    ```python
    from looplet import composable_loop

    for step in composable_loop(llm=llm, tools=tools, task=task, config=cfg, state=state):
        print(step.pretty())          # "#1 search(query='…') → 12 items [340ms]"
        if step.usage.total_tokens > BUDGET:
            break                      # your loop, your control flow
    ```

=== "Install"

    ```bash
    pip install looplet                  # core — zero third-party packages
    pip install "looplet[openai]"        # OpenAI, Ollama, Groq, Together, vLLM
    pip install "looplet[anthropic]"     # Anthropic
    ```

=== "See it"

    ![looplet demo — three-step investigation loop](demo.gif){ loading=lazy }

---

## How it works

```mermaid
flowchart LR
    subgraph Loop["composable_loop()"]
        direction LR
        P[pre_prompt] --> L[llm.generate]
        L --> D[pre_dispatch]
        D --> T[tool.execute]
        T --> R[post_dispatch]
        R --> C{check_done}
        C -- no --> P
        C -- yes --> Y([yield Step])
    end
    Y --> Caller[Your for-loop]
    Caller -. "stop / log / score" .-> Loop

    classDef hook fill:#EEF2FF,stroke:#4F46E5,stroke-width:1.5px,color:#1E1B4B
    class P,D,R,C hook
```

Four hook methods on any Python object. Implement only the ones you
need. The loop uses `hasattr` — no base class, no registration.

---

## Why looplet?

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Fast to start, fast to run**

    ---

    289 ms cold import. Zero runtime dependencies. `pip install` stays
    snappy on serverless and short-lived scripts.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

-   :material-puzzle:{ .lg .middle } **Composable by Protocol**

    ---

    Four `@runtime_checkable` hook methods. Any object implementing
    one or more is a hook. No base classes, no registration.

    [:octicons-arrow-right-24: Hooks](hooks.md)

-   :material-eye:{ .lg .middle } **Observable by default**

    ---

    `step.pretty()`, `ProvenanceSink`, and `eval_*` all read the
    same `Step` dataclass. One artifact, three uses.

    [:octicons-arrow-right-24: Provenance](provenance.md)

-   :material-shield-lock:{ .lg .middle } **Safe by design**

    ---

    `redact=` scrubs PII **before** the provider sees it *and* before
    the trace is written. No wrapping-order footguns.

    [:octicons-arrow-right-24: Pitfalls](pitfalls.md)

-   :material-arrow-decision:{ .lg .middle } **Compose agents as tools**

    ---

    Any looplet agent is a function that returns a result. Wrap it
    in a `ToolSpec` and plug it into the next agent.

    [:octicons-arrow-right-24: Recipes](recipes.md)

-   :material-test-tube:{ .lg .middle } **Debugging is evaluation**

    ---

    What you do while debugging (`print(step.pretty())`) is a
    trajectory. Evals are pytest-style functions over the same data.

    [:octicons-arrow-right-24: Evals](evals.md)

</div>

---

## See the difference

=== "Hidden loop (most frameworks)"

    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model=llm,
        tools=[search, fetch],
        state_schema=State,
    )
    result = agent.invoke({"messages": [task]})
    # Where does the loop stop?
    # Where does the tool call happen?
    # How do I intercept it?
    # → read the framework source.
    ```

=== "Loop-is-product (looplet)"

    ```python
    for step in composable_loop(
        llm=llm, tools=tools, task=task,
        hooks=[BudgetCap(10_000), Redactor()],
    ):
        if step.tool_call.tool == "delete":
            if not approve(step):           # (1)
                break
        log(step)                            # (2)
    ```

    1.  Intercept any tool call with ordinary Python — no custom graph
        node needed.
    2.  One `Step` object is the trace, the eval context, and the
        checkpoint unit.

---

## Honest benchmarks

All numbers regenerate in one command on a fresh Python 3.11 venv.
See [Benchmarks](benchmarks.md) for the full methodology.

| Framework | Cold import | PyPI deps | vs looplet |
| --- | ---: | ---: | ---: |
| **looplet** | **289 ms** | **0** | — |
| strands-agents | 1,885 ms | 6 | 6.5× slower |
| LangGraph | 2,294 ms | 31 | 7.9× slower |
| Claude Agent SDK | 2,409 ms | 13 | 8.3× slower |
| Pydantic AI | 3,975 ms | 12 | 13.8× slower |

<small>Median of 9 fresh subprocess runs. Python 3.11.13, Linux x86_64, PyPI wheels from 2026-04-21.</small>

---

## Start here

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } **[Quickstart](quickstart.md)**

    ---

    Install. Run. Understand the loop in five minutes.

-   :material-school:{ .lg .middle } **[Tutorial](tutorial.md)**

    ---

    Build an agent with hooks, context management, crash-resume, and
    approval — in five steps.

-   :material-book-open-variant:{ .lg .middle } **[Hooks](hooks.md)**

    ---

    The four extension points. Recipes for every common pattern.

-   :material-chart-bar:{ .lg .middle } **[Evals](evals.md)**

    ---

    pytest-style scoring over the same trajectory you debug with.

-   :material-database-eye:{ .lg .middle } **[Provenance](provenance.md)**

    ---

    Capture every prompt and step to a diff-friendly directory.

-   :material-book-open-page-variant:{ .lg .middle } **[Recipes](recipes.md)**

    ---

    Ollama, OTel, MCP, cost accounting, checkpoints.

-   :material-alert-circle:{ .lg .middle } **[Pitfalls](pitfalls.md)**

    ---

    Ten sharp edges worth knowing — with the "right way" for each.

-   :material-speedometer-medium:{ .lg .middle } **[Benchmarks](benchmarks.md)**

    ---

    Cold import and dependency footprint, with the harness.

</div>

---

## Extension points at a glance

<div class="grid cards" markdown>

-   **`pre_prompt`**

    Inject context into the next prompt. Context managers,
    retrieval-augmented briefings, step-specific guidance.

-   **`pre_dispatch`**

    Intercept a tool call before it runs. Cache hits, permission
    gates, argument rewriting, approval flows.

-   **`post_dispatch`**

    React to a tool result. Duplicate-call warnings, error remediation
    messages, metric collection, streaming events.

-   **`check_done`**

    Reject premature completion. Quality gates, minimum-evidence
    thresholds, required-tool checks.

</div>

---

## Talks and writing

- :material-post: **[Blog: "The loop is the product"](https://hsaghir.github.io/engineering/the-loop-is-the-product/)** — the design argument behind the library.
- :material-github: **[THIRD_PARTY_USERS.md](https://github.com/hsaghir/looplet/blob/master/THIRD_PARTY_USERS.md)** — who is building on looplet.

---

<p class="home-footer" markdown>
[GitHub](https://github.com/hsaghir/looplet){ .md-button }
[PyPI](https://pypi.org/project/looplet/){ .md-button }
[Blog](https://hsaghir.github.io/engineering/the-loop-is-the-product/){ .md-button }
</p>
---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<p class="hero-eyebrow">Zero deps · Four hooks · One iterator</p>

# The loop is the product.

<p class="hero-sub" markdown>
**looplet** is a `for`-loop you own for LLM tool-calling agents. Yield
every step, intercept every tool call, redact every prompt, eval every
trajectory. No graph DSL, no agent runtime, no hidden state. Works with
any OpenAI-compatible endpoint or Anthropic directly.
</p>

<div class="hero-cta">
  <a href="quickstart/" class="md-button md-button--primary">Quickstart →</a>
  <a href="tutorial/" class="md-button">5-step tutorial</a>
  <a href="https://github.com/hsaghir/looplet" class="md-button">GitHub ⭐</a>
</div>

</div>

```python
from looplet import composable_loop

for step in composable_loop(llm=llm, tools=tools, task=task, config=cfg, state=state):
    print(step.pretty())          # "#1 search(query='…') → 12 items [340ms]"
    if step.usage.total_tokens > BUDGET:
        break                      # your loop, your control flow
```

```bash
pip install looplet                  # core — zero third-party packages
pip install "looplet[openai]"        # OpenAI, Ollama, Groq, Together, vLLM
pip install "looplet[anthropic]"     # Anthropic
```

---

## Why looplet?

<div class="grid cards" markdown>

-   **:material-rocket-launch: Fast to start, fast to run**

    289 ms cold import. Zero runtime dependencies. `pip install` stays
    snappy on serverless and short-lived scripts.

-   **:material-puzzle: Composable by Protocol**

    Four `@runtime_checkable` hook methods. Any object implementing one
    or more is a hook. No base classes, no registration, no decorators.

-   **:material-eye: Observable by default**

    `step.pretty()` trace, `ProvenanceSink` to disk, and `eval_*`
    helpers all read the same `Step` dataclass. One artifact, three uses.

-   **:material-shield-lock: Safe by design**

    `redact=` scrubs PII **before** the provider sees it *and* before
    the trace is written. No wrapping-order footguns.

-   **:material-arrow-decision: Compose agents as tools**

    Any looplet agent is a function that returns a result. Wrap it in a
    `ToolSpec` and plug it into the next agent. Sub-agents are just
    nested loops.

-   **:material-test-tube: Debugging === evaluation**

    What you do while debugging (`print(step.pretty())`) is a
    trajectory. What you write as an eval is a function over the same
    trajectory. No separate logging pipeline.

</div>

---

## See the difference

<div class="grid" markdown>

```python title="Hidden loop (LangGraph style)"
agent = create_agent(
    llm=llm,
    tools=[search, fetch],
    state_schema=State,
)
result = agent.invoke({"input": task})
# where does the loop stop?  where does the
# tool call happen?  how do I intercept it?
# → read the framework source.
```

```python title="Loop-is-product (looplet)"
for step in composable_loop(
    llm=llm, tools=tools, task=task,
    hooks=[BudgetCap(10_000), Redactor()],
):
    if step.tool_call.tool == "delete":
        if not approve(step): break   # (1)
    log(step)                         # (2)
```

1.  Intercept any tool call with ordinary Python.
2.  One `Step` object is the trace, the eval context, and the
    checkpoint unit.

</div>

---

## Honest benchmarks

All numbers regenerate in one command on a fresh Python 3.11 venv. See
[Benchmarks](benchmarks.md) for the full methodology.

| Framework | Cold import | PyPI deps | vs looplet |
| --- | ---: | ---: | ---: |
| **looplet** | **289 ms** | **0** | — |
| strands-agents | 1,885 ms | 6 | 6.5× slower |
| LangGraph | 2,294 ms | 31 | 7.9× slower |
| Claude Agent SDK | 2,409 ms | 13 | 8.3× slower |
| Pydantic AI | 3,975 ms | 12 | 13.8× slower |

<small>Median of 9 fresh subprocess runs. Python 3.11.13, Linux x86_64, PyPI wheels from 2026-04-21.</small>

---

## Start here

<div class="grid cards" markdown>

-   **[:material-speedometer: Quickstart](quickstart.md)**

    Install. Run. Understand the loop in five minutes.

-   **[:material-school: Tutorial](tutorial.md)**

    Build an agent with hooks, context management, crash-resume, and
    approval — in five steps.

-   **[:material-book-open-variant: Hooks](hooks.md)**

    The four extension points: `pre_prompt`, `pre_dispatch`,
    `post_dispatch`, `check_done`. Recipes for every common pattern.

-   **[:material-chart-bar: Evals](evals.md)**

    pytest-style scoring that reads the same trajectory you debug with.
    Your debugging becomes your eval suite.

-   **[:material-database-eye: Provenance](provenance.md)**

    Capture every prompt, every step, to a diff-friendly directory you
    can `cat`, `grep`, and check into git. Replay against cached LLM
    output.

-   **[:material-book-open-page-variant: Recipes](recipes.md)**

    Ollama, OTel, MCP, cost accounting, checkpoints. Copy-paste
    solutions for common integrations.

-   **[:material-alert-circle: Pitfalls](pitfalls.md)**

    Ten sharp edges worth knowing before you start — the "right way" for
    each.

-   **[:material-speedometer-medium: Benchmarks](benchmarks.md)**

    Cold import, dependency footprint, and why they matter for
    serverless and CI.

</div>

---

## Extension points at a glance

<div class="grid cards" markdown>

-   **`pre_prompt`**

    Inject context into the next prompt. Context managers,
    retrieval-augmented briefings, guidance at specific steps.

-   **`pre_dispatch`**

    Intercept a tool call before it runs. Cache hits, permission
    gates, argument rewriting, approval flows.

-   **`post_dispatch`**

    React to a tool result. Duplicate-call warnings, error remediation
    messages, metric collection, streaming events.

-   **`check_done`**

    Reject premature completion. Quality gates ("tests must pass"),
    minimum-evidence thresholds, required-tool checks.

</div>

---

## Talks and writing

- **[:material-post: Blog: "The loop is the product"](https://hsaghir.github.io/engineering/the-loop-is-the-product/)** — the design argument behind the library.
- **[:material-github: THIRD_PARTY_USERS.md](https://github.com/hsaghir/looplet/blob/master/THIRD_PARTY_USERS.md)** — who is building on looplet.

---

<p align="center" style="margin-top: 2rem; color: var(--md-default-fg-color--light);">
<a href="https://github.com/hsaghir/looplet"><strong>GitHub</strong></a>
·
<a href="https://pypi.org/project/looplet/"><strong>PyPI</strong></a>
·
<a href="https://hsaghir.github.io/engineering/the-loop-is-the-product/"><strong>Blog</strong></a>
</p>
# looplet

![demo -- 3-step investigation loop](demo.gif)

**A `for`-loop you own for LLM tool-calling agents.** Zero runtime
dependencies. Four Protocol hooks. Works with any OpenAI-compatible
endpoint or Anthropic directly.

```python
from looplet import composable_loop

for step in composable_loop(llm=llm, tools=tools, task=task, config=cfg, state=state):
    print(step.pretty())   # "#1 search(query='...') -> 12 items [340ms]"
    if step.usage.total_tokens > budget:
        break               # your loop, your control flow
```

```bash
pip install looplet               # core -- zero third-party packages
pip install "looplet[openai]"     # OpenAI, Ollama, Groq, Together, vLLM
pip install "looplet[anthropic]"  # Anthropic
```

## Why looplet?

Most agent frameworks give you `agent.run(task)` and a black box.
looplet gives you the loop itself. Each iteration yields a `Step`
dataclass with the full prompt, tool call, result, token usage, and
timing. You decide when to stop, what to show the model, and whether
to let a tool call proceed.

Behaviour injection uses Python's Protocol pattern: four hook points
(`pre_prompt`, `pre_dispatch`, `post_dispatch`, `check_done`) that
any object can implement without inheriting from anything. Hooks
compose by stacking in a list.

The debug trace and the eval harness are the same artifact:
`step.pretty()` is the trace, `ProvenanceSink` dumps it to disk,
and the `eval_*` helpers read it directly. No separate pipeline.

| Metric | looplet | LangGraph | Claude SDK | Pydantic AI |
|--------|--------:|----------:|-----------:|------------:|
| Cold import | 289 ms | 2,294 ms | 2,409 ms | 3,975 ms |
| PyPI deps | 0 | 31 | 13 | 12 |

## Start here

- **[Tutorial](tutorial.md)** -- build your first agent in 5 steps
- **[Hooks](hooks.md)** -- the per-phase extension points that replace subclassing
- **[Benchmarks](benchmarks.md)** -- cold-import and dependency numbers vs alternatives
- **[Recipes](recipes.md)** -- Ollama, OTel, MCP, cost accounting, checkpoints

## Reference

- **[Evals](evals.md)** -- pytest-style agent evaluation
- **[Provenance](provenance.md)** -- capture prompts and trajectories
- **[FAQ](faq.md)** -- including "why not LangGraph?"
- **[Roadmap](roadmap.md)** -- planned, frozen, and out-of-scope features

## Project

- **[Contributing](contributing.md)** -- dev setup, conventions, PR checklist
- **[Good first issues](good-first-issues.md)** -- curated tasks for first-time contributors
- **[Changelog](changelog.md)** -- release notes

**[GitHub](https://github.com/hsaghir/looplet)** |
**[PyPI](https://pypi.org/project/looplet/)** |
**[Blog post: "The loop is the product"](https://hsaghir.github.io/engineering/the-loop-is-the-product/)**
