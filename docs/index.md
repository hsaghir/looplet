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
