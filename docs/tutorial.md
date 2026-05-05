# Tutorial — build your first agent in 5 steps

This walks through the essentials. By the end you'll have an agent with
hooks, context management, crash-resume, and approval.

## Step 1 — Install and run the hello world

```bash
pip install "looplet[openai]"
python -m looplet.examples.hello_world
```

Connects to any OpenAI-compatible API — set `OPENAI_BASE_URL` and
`OPENAI_MODEL` to point at your provider (OpenAI, Ollama, Groq,
Together, vLLM, …).

## Step 2 — Understand the loop

The core is one `for` loop. You own iteration — pause, filter, break:

```python
from looplet import (
    composable_loop, LoopConfig, DefaultState, BaseToolRegistry, ToolSpec,
)

tools = BaseToolRegistry()
tools.register(ToolSpec(name="greet", description="Greet someone",
                        parameters={"name": "str"},
                        execute=lambda *, name: {"greeting": f"Hello, {name}!"}))
tools.register(ToolSpec(name="done", description="Finish",
                        parameters={"answer": "str"},
                        execute=lambda *, answer: {"answer": answer}))

for step in composable_loop(
    llm=my_llm,                       # any LLMBackend — OpenAI, Anthropic, local
    tools=tools,
    state=DefaultState(max_steps=5),
    config=LoopConfig(max_steps=5),
    task={"goal": "Greet Alice, then finish."},
):
    print(step.pretty())
```

## Step 3 — Add a hook

Hooks are plain Python classes. Implement only the methods you need:

```python
from looplet import HookDecision, InjectContext

class MyGuardrail:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool == "write" and "test_" not in tool_call.args.get("file_path", ""):
            return InjectContext("You wrote code but no tests. Write tests first.")
        return None

    def check_done(self, state, session_log, context, step_num):
        return HookDecision(block="Not done yet — run tests first.")
```

Pass hooks to the loop:

```python
composable_loop(..., hooks=[MyGuardrail()])
```

See [hooks.md](hooks.md) for the full walkthrough.

## Step 4 — Add context management

For long sessions, add the default compaction service so the agent can
keep working under context pressure:

```python
from looplet import (
    DefaultCompactService,
    ContextBudget, ThresholdCompactHook,
)

config = LoopConfig(
    max_steps=50,
    compact_service=DefaultCompactService(
        keep_recent=2,
        keep_recent_tool_results=5,
    ),
)
hooks = [ThresholdCompactHook(ContextBudget(context_window=128_000))]
```

`DefaultCompactService` prunes old bulky tool results, summarizes older
working context, keeps recent steps verbatim, and reports what changed
through compaction lifecycle events. Use `compact_chain(...)` with
`PruneToolResults`, `SummarizeCompact`, and `TruncateCompact` when you
want a custom policy.

## Step 5 — Add crash-resume and approval

One line for crash-safe checkpoints. Add `ApprovalHook` for human
sign-off on risky actions:

```python
from looplet import ApprovalHook

config = LoopConfig(
    max_steps=50,
    checkpoint_dir="./checkpoints",   # auto-save after every step, auto-resume on restart
)
hooks = [ApprovalHook()]              # stops loop when tool returns needs_approval=True
```

## See it all together

Run the complete coding agent example (bash, read, write, edit, glob,
grep, think — same tools as Claude Code):

```bash
python -m looplet.examples.coding_agent "implement fizzbuzz" --model gpt-4o
python -m looplet.examples.coding_agent --trace ./traces/   # save trajectory
```

## Debug — see what the LLM sees

```python
from looplet import preview_prompt

print(preview_prompt(task={"goal": "fix the bug"}, tools=my_tools, state=my_state))
```

## Testing without a real LLM

`looplet.testing` ships a scripted mock backend so you can unit-test
hooks, tools, and your agent wiring without hitting a provider:

```python
from looplet.testing import MockLLMBackend

llm = MockLLMBackend(responses=[
    '{"tool": "add", "args": {"a": 2, "b": 3}, "reasoning": "sum"}',
    '{"tool": "done", "args": {}, "reasoning": "finished"}',
])
```

## Next

- [hooks.md](hooks.md) — compose hooks for guardrails,
  metrics, caching, approval.
- [docs/evals.md](evals.md) — score your agent as you debug it.
- [provenance.md](provenance.md) — capture the exact
  prompts and trajectory.
- [docs/recipes.md](recipes.md) — Ollama, OTel, MCP, cost accounting.
