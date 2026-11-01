---
title: A Python loop for agents that call tools
description: Run a tool-calling agent from Python functions. Inspect each step, add policy hooks, and check results with saved responses.
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
Give Looplet a model, a task, and Python functions.
It checks tool calls, runs the functions, and yields each result.
Your code controls the loop.
</p>

<div class="hero-cta">
   <a href="python/" class="md-button md-button--primary">Run your first loop</a>
  <a href="install/" class="md-button">Install and configure</a>
  <a href="https://github.com/hsaghir/looplet" class="md-button">GitHub</a>
</div>

</section>

## Start with one loop

This example uses a Python function and fixed model responses.
It needs no model key or model service.
Install the [source preview from the quickstart](python.md#install) first.
The Python-first API is unreleased and is not in PyPI `0.4.1`.
Development source builds identify as `0.5.0.dev0`.

```python title="first_loop.py"
from looplet import MockLLMBackend, composable_loop


def lookup_owner(service: str) -> str:
    """Find the team that owns a service."""
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return owners.get(service, "unknown")


responses = [
   '{"tool":"lookup_owner","args":{"service":"payments"}}',
   '{"tool":"done","args":{"summary":"fintech-platform"}}',
]
llm = MockLLMBackend(responses, cycle=False)
for step in composable_loop(llm, task="Who owns payments?", tools=[lookup_owner]):
    print(step.pretty())
```

The first tool result is `fintech-platform`.
The second call completes the run.
Your function stays callable as ordinary Python.
Use `run_result()` when you need the final result instead of an iterator.
The [quickstart](python.md) checks the result, adds a permission hook, and replays this same agent.

## Add control as the agent grows

Looplet handles model retries, context limits, and private run evidence.
You can replace these defaults and keep the same tools and loop.

<div class="workflow" markdown>

<div class="workflow-step" markdown>

**Control calls**

Set limits and check arguments before a tool runs.
[Use hooks](hooks.md) for your application policy.
</div>

<div class="workflow-step" markdown>

**Handle long runs**

Shorten old context and retrieve original tool results when needed.
[Set runtime controls](operations.md).
</div>

<div class="workflow-step" markdown>

**Check outputs**

Save model responses and rerun tools after a change.
[Grade the actual result](evals.md).
</div>

<div class="workflow-step" markdown>

**Share an agent**

Keep dependency construction in a Python factory.
Use a [cartridge](cartridge.md) when you need file-based packaging.
</div>

</div>

Runs save private evidence by default.
Compaction can discard details.
You control evidence access and retention.
See [default policies and opt-outs](operations.md#standard-agent-harness).

## Check a tool change

Run `looplet-proof` after installation to check the complete regression example.

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

[Migrate an existing loop](migrate.md) | [Build the first loop](python.md)

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

## Know what each check proves

| Evidence             | Use it for                                        | Do not claim                                                   |
| -------------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| `Step` objects       | Inspect results and stop between calls            | Approval before a tool runs or proof that an answer is correct |
| Saved run            | Read what the model saw, returned, and called     | That saved prompts are safe to publish                         |
| Replay               | Check changed tools or hooks with fixed responses | Better decisions from a prompt or model change                 |
| Collector and grader | Check a file, record, or service result           | Protection from code that can change the grader                |
| New model runs       | Check prompt, model, schema, or context changes   | General performance from one sample                            |

Completion does not prove that an answer is correct.
Replay fixes model responses, not tool effects.
Keep protected acceptance checks outside the code that a candidate can change.
Looplet does not create an operating system sandbox.
Run untrusted tools in a restricted container or VM.
A Git worktree only separates edits.

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
[Run your first loop](python.md){ .md-button .md-button--primary }
[Install Looplet](install.md){ .md-button }
[Add tests](tutorial.md){ .md-button }
</p>

<!-- markdownlint-enable MD025 MD033 MD036 -->
