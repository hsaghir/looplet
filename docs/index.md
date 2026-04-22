# looplet

![demo — 3-step investigation loop](demo.gif)

**A small, framework-agnostic Python library for building LLM agents
that call tools in a loop.** It hands you a `for step in loop(...):`
iterator so you can observe, filter, or interrupt *any* step — no
graph DSL, no subclassing, no vendor lock-in. One runtime dependency.

```python
from looplet import composable_loop

for step in composable_loop(llm=llm, tools=tools, task=task, config=cfg, state=state):
    print(step.pretty())
    if step.tool_result.error:
        break
```

```bash
pip install "looplet[openai]"     # OpenAI-compatible (Ollama, Groq, Together, vLLM)
pip install "looplet[anthropic]"  # Anthropic
```

## Start here

- **[Tutorial](tutorial.md)** — build your first real agent in 5 steps.
- **[Hooks](hooks.md)** — the per-phase extension points that replace
  subclassing.
- **[Benchmarks](benchmarks.md)** — honest cold-import and dependency
  numbers against the alternatives.
- **[Recipes](recipes.md)** — Ollama, OTel, MCP, cost accounting,
  checkpoints — short snippets for common needs.

## Reference

- **[Evals](evals.md)** — pytest-style agent evaluation.
- **[Provenance](provenance.md)** — capture the exact prompts the LLM
  saw and the trajectory the loop took.
- **[Roadmap](roadmap.md)** — what's planned, what's frozen, what's
  explicitly out of scope.

## Project

- **[Contributing](contributing.md)** — dev setup, conventions, PR
  checklist.
- **[Good first issues](good-first-issues.md)** — curated, scoped
  tasks for first-time contributors.
- **[Changelog](changelog.md)** — release notes.

The full README lives [on GitHub](https://github.com/hsaghir/looplet).
