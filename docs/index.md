---
title: A Python loop for agents that call tools
description: Run one loop, inspect each tool result, and stop from your code. Add hooks, saved runs, and tests when you need them.
hide:
  - navigation
  - toc
---

<!-- markdownlint-disable MD025 MD033 MD036 -->

<section class="hero" markdown>

<p class="hero-eyebrow">Python 3.11 or newer</p>

# Looplet

<p class="hero-kicker">A Python loop for agents that call tools.</p>

<p class="hero-sub" markdown>
Choose your model, tools, task, and limits.
Looplet runs the tools and yields each result to your code.
Inspect the result or stop before the next call.
Add hooks, saved runs, and tests when you need them.
</p>

<div class="hero-cta">
  <a href="quickstart/" class="md-button md-button--primary">Run your first loop</a>
  <a href="install/" class="md-button">Install and configure</a>
  <a href="https://github.com/hsaghir/looplet" class="md-button">GitHub</a>
</div>

</section>

<div class="proof-strip" markdown>

<div class="proof-item" markdown>
**Run**

Use normal Python functions as tools.
</div>

<div class="proof-item" markdown>
**Inspect**

Read the call, result, and error in each step.
</div>

<div class="proof-item" markdown>
**Stop**

Stop your iteration before the next tool call.
</div>

<div class="proof-item" markdown>
**Extend**

Add only the optional features that you need.
</div>

</div>

## Start with one loop

Install the source version as described in the [Quickstart](quickstart.md).
This example uses fixed responses and does not contact a model service.

```python title="first_loop.py"
from contextlib import closing

from looplet import DefaultState, LoopConfig, MockLLMBackend, composable_loop, tool, tools_from


@tool(description="Find the team that owns a service.")
def lookup_owner(service: str) -> dict:
  owners = {"payments": "fintech-platform", "search": "discovery"}
  return {"service": service, "owner": owners.get(service)}


llm = MockLLMBackend([
  '{"tool":"lookup_owner","args":{"service":"payments"}}',
  '{"tool":"done","args":{"summary":"The payments owner is fintech-platform."}}',
])
tools = tools_from([lookup_owner], include_done=True)
state = DefaultState(max_steps=2)

with closing(composable_loop(
  llm=llm,
  tools=tools,
  state=state,
  config=LoopConfig(max_steps=2, use_native_tools=False),
  task={"goal": "Find the payments owner, then finish."},
)) as steps:
  for step in steps:
    print(step.tool_call.tool, step.tool_result.data)

print(state.termination_reason)
```

Each `Step` contains a tool call and its result.
The tool has already run when you receive the step.
Use `break` to stop before the next call.
`closing` closes the loop and runs its end hooks.

The example selects the JSON-text protocol for its fixed responses.
Real models use native tool calls by default.
The [Quickstart](quickstart.md) explains each field and shows early exit.
See [Install](install.md) to configure a real model.

## Add only what you need

<div class="workflow" markdown>

<div class="workflow-step" markdown>

**Hooks**

Approve calls or change loop decisions at specific points.
</div>

<div class="workflow-step" markdown>

**Saved runs**

Save prompts, responses, steps, and stop reasons as files.
</div>

<div class="workflow-step" markdown>

**Tests**

Read the result that the agent produced and check it independently.
</div>

<div class="workflow-step" markdown>

**Cartridges**

Put tools, prompts, and configuration in a directory for review or distribution.
</div>

</div>

These features use the same loop.
They are not required for the first example.

[Hooks](hooks.md) | [Capture and replay](provenance.md) |
[Tests and graders](evals.md) | [Cartridges](cartridge.md)

## Check a tool change

Run `looplet-proof` after the source install to check the complete regression example.

<div class="proof-terminal" markdown>

<div class="proof-terminal__title">$ looplet-proof</div>

```text
1. CAPTURE v1 with fixed model responses
   collected profit: 200
   required eval: FAIL (0.00)

2. CHANGE one tool line
   - "profit": revenue + cost,
   + "profit": revenue - cost,

3. REPLAY with fresh v2 tool execution
   same model decisions: true
   collected profit: 40
   required eval: PASS (1.00)
```

</div>

<p class="proof-caption" markdown>
The example needs no model key or network.
It reuses fixed model responses with changed tool code.
A collector reads the report and a grader checks it.
[Read the example and its limits.](regression-demo.md)
</p>

## Choose a guide

<div class="home-paths" markdown>

<div class="home-path" markdown>

### Replace a loop

Keep your provider and tools.
Replace only the loop and check its behavior before you add other features.

[Migrate an existing loop](migrate.md) | [Build the first loop](quickstart.md)

</div>

<div class="home-path" markdown>

### Check a failure

Save the run and read the result that the agent produced.
Choose fixed responses or new model runs for the change that you need to check.

[Failure to regression](regression-demo.md) | [Choose an experiment](experiments.md)

</div>

<div class="home-path" markdown>

### Find an interface

Find commands, Python functions, saved files, and runtime controls.

[CLI](cli.md) | [Python API](api.md) | [Saved artifacts](artifacts.md)

</div>

</div>

## Use other processes or hosts

Protocol adapters can connect tools, hooks, shared state, and model access across processes.
Start with the Python loop.
Add a process boundary when your deployment needs one.

[Study portability](portability.md) | [Inspect the cartridge format](cartridge.md)

## Know what each check proves

| Evidence | Use it for | Do not claim |
| --- | --- | --- |
| `Step` objects | Inspect results and stop between calls | Approval before a tool runs or proof that an answer is correct |
| Saved run | Read what the model saw, returned, and called | That saved prompts are safe to publish |
| Replay | Check changed tools or hooks with fixed responses | Better decisions from a prompt or model change |
| Collector and grader | Check a file, record, or service result | Protection from code that can change the grader |
| New model runs | Check prompt, model, schema, or context changes | General performance from one sample |

Completion does not prove that an answer is correct.
Replay fixes model responses, not tool effects.
Keep protected acceptance checks outside the code that a candidate can change.
Run untrusted code in an isolated process or operating system environment.

[Capture and replay](provenance.md) | [Behavioral evals](evals.md) |
[Saved artifact reference](artifacts.md)

## When Looplet fits

<div class="fit-grid" markdown>

<div class="fit-panel fit-panel--yes" markdown>

### Looplet is a good fit when

- One model calls tools until the task ends.
- Your team uses Python, Git, and pytest.
- Your code must control when tools run.
- You need saved runs and tests for agent changes.

</div>

<div class="fit-panel fit-panel--no" markdown>

### Choose another tool when

- Your main task needs a durable workflow graph.
- You need a complete assistant, not a Python library.
- You need a managed dashboard as your main interface.
- A small loop that you wrote already meets your needs.

</div>

</div>

Looplet can run inside a workflow engine and export evidence to other systems.
It does not replace those systems.
The core package has no third-party runtime dependencies.
Provider adapters are optional extras.

[Read the selection guide](why-looplet.md) | [Check the FAQ](faq.md) |
[Operate a production loop](operations.md)

<p class="home-footer" markdown>
[Run your first loop](quickstart.md){ .md-button .md-button--primary }
[Install Looplet](install.md){ .md-button }
[Add tests](tutorial.md){ .md-button }
</p>

<!-- markdownlint-enable MD025 MD033 MD036 -->
