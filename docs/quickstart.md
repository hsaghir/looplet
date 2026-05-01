# Quickstart

Five minutes from zero to a running agent you understand every line of.

The whole looplet mental model fits in one turn:

1. The LLM proposes a tool call.
2. The registry validates and dispatches it.
3. Hooks observe or steer the turn.
4. State records the step.
5. The loop yields a `Step` back to your code.

Everything below is ordinary Python around that mechanism.

## 1. Install

```bash
pip install "looplet[openai]"
# or
pip install "looplet[anthropic]"
```

!!! speed "Cold import: 289 ms"
    looplet has zero required runtime dependencies. The `[openai]` /
    `[anthropic]` extras are imported lazily only when you instantiate
    a backend.

## 2. Point it at any OpenAI-compatible endpoint

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1   # or Ollama, Groq, Together, vLLM, …
export OPENAI_API_KEY=sk-…
export OPENAI_MODEL=gpt-4o-mini
```

Run the bundled hello-world to sanity-check the wiring:

```bash
python -m looplet.examples.hello_world
```

You should see three lines of `#1 greet(name='…') → {…} [Xms]` trace
followed by a final `#N ✓ done(...)`. If that works, you are ready.

## 3. Write your first loop

```python title="my_agent.py"
from looplet import (
    composable_loop, LoopConfig, DefaultState,
    OpenAIBackend, tool, tools_from,
)

llm = OpenAIBackend(base_url="https://api.openai.com/v1",
                    api_key="sk-...", model="gpt-4o-mini")

@tool(description="Search the docs.")
def search(query: str) -> dict:
    return {"results": [f"result for {query}"]}

tools = tools_from([search], include_done=True)

# Run.  You own the iteration.
for step in composable_loop(
    llm=llm,
    tools=tools,
    state=DefaultState(max_steps=5),
    config=LoopConfig(max_steps=5),
    task={"goal": "What is looplet?"},
):
    print(step.pretty())
```

That's it. The whole agent is 30 lines.

The objects map directly to the mental model:

- `tools_from(...)` builds the registry that validates and dispatches tool calls.
- `hooks=[...]` lets plain Python objects observe or steer the loop.
- `DefaultState` records the steps and remaining budget.
- `composable_loop(...)` yields each `Step` so your code can print, test, stop, or route it.

## 4. Add a hook

Hooks are plain classes. Implement only the methods you want — the loop
checks with `hasattr` before calling.

```python
from looplet import HookDecision

class StepBudget:
    """Cap the loop at N productive steps; block ``done()`` until
    we've gathered enough evidence."""

    def __init__(self, max_productive: int) -> None:
        self.cap = max_productive
        self.productive = 0

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if not tool_result.error and tool_call.tool != "done":
            self.productive += 1

    def should_stop(self, state, step_num, new_entities):
        if self.productive >= self.cap:
            return HookDecision(stop="step_budget_exceeded")
        return False

# ... then in composable_loop(...):
hooks=[StepBudget(max_productive=8)]
```

For real token-cost tracking, wire a [`BackendRouter`](recipes.md)
(it owns the token counters) and read `router.total_input_tokens`
/ `router.total_output_tokens` between steps.

See [Hooks](hooks.md) for the full protocol and a dozen recipes.

## 5. Capture the trajectory

```python
from looplet import ProvenanceSink

sink = ProvenanceSink(dir="traces/run_1", redact=lambda s: s.replace("secret", "[REDACTED]"))
llm  = sink.wrap_llm(OpenAIBackend(base_url="https://api.openai.com/v1",
                                    api_key="sk-...", model="gpt-4o-mini"))

for step in composable_loop(llm=llm, tools=tools, hooks=[sink.trajectory_hook()], ...):
    print(step.pretty())
sink.flush()     # writes trajectory.json + steps/*.json + call_*.txt
```

Inspect it from the shell:

```bash
python -m looplet show traces/run_1/
```

## 6. Turn debugging into an eval

```python title="eval_my_agent.py"
from looplet import eval_mark

@eval_mark("verdict")
def eval_returns_answer(ctx):
    return "answer" in ctx.final_output

@eval_mark("budget")
def eval_stopped_cleanly(ctx):
    return ctx.completed            # stop_reason == "done"
```

Run against any saved trajectory:

```bash
looplet eval traces/ --evals eval_my_agent.py --threshold 0.7 -v
```

## 7. Run or share a bundle

A bundle is a portable skill folder that builds normal looplet
primitives. It is the beginner-friendly way to run or share a complete
capability without hiding the underlying loop.

```bash
python -m looplet run ./skills/coder "Fix the tests" --workspace .
python -m looplet blueprint ./skills/coder --workspace .
python -m looplet export-code ./skills/coder coder_agent.py
```

Advanced users can package an importable looplet factory as a bundle:

```bash
python -m looplet package my_agent:build ./skills/my-agent \
    --name my-agent \
    --description "Run my custom looplet agent."
```

Claude/Agent Skills-style folders can be wrapped when they are
instruction-only, and looplet reports adapter gaps when scripts or
resources need explicit tools:

```bash
python -m looplet wrap-claude-skill ./claude-skills/pdf ./skills/pdf
```

---

## Next steps

- [Tutorial](tutorial.md) — hooks, compaction, crash-resume, approval, in five steps.
- [Recipes](recipes.md) — Ollama, MCP, cost accounting, multi-model routing.
- [Skills](skills.md) — lazy skills, runnable bundles, blueprints, and Claude Skill wrapping.
- [Pitfalls](pitfalls.md) — ten sharp edges worth knowing.
- [Hooks reference](hooks.md) — every extension point, every signature.

??? question "Where are the sub-agents, the planner, the memory manager?"
    There aren't any. A sub-agent is a function that calls
    `composable_loop` and returns a value — then you expose it as a
    `ToolSpec`. A planner is a hook that inspects `session_log` and
    returns an `InjectContext(...)` in `pre_prompt`. A memory manager
    is `StaticMemorySource` plus a `compact_service`. Nothing is
    hidden.
