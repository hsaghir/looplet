---
title: Test-driven harness engineering for Python agents
description: Own, inspect, and regression-test Python agent harnesses with reviewable files, captured evidence, and outcome-based release gates.
hide:
  - navigation
  - toc
---

<!-- markdownlint-disable MD025 MD033 MD036 -->

<section class="hero" markdown>

<p class="hero-eyebrow">Test-driven harness engineering for Python agents</p>

# Looplet

<p class="hero-kicker">Own the loop. Test every change.</p>

<p class="hero-sub" markdown>
Keep prompts, tools, hooks, cases, and graders in code and files your team can
review. Capture a failure, inspect the resulting world, and turn the behavior
into a required pytest or CI contract.
</p>

<div class="hero-cta">
  <a href="regression-demo/" class="md-button md-button--primary">Run the network-free proof</a>
  <a href="install/" class="md-button">Install and configure</a>
  <a href="https://github.com/hsaghir/looplet" class="md-button">GitHub</a>
</div>

</section>

<div class="proof-strip" markdown>

<div class="proof-item" markdown>
**Reviewable**

Harness changes are ordinary Python, YAML, Markdown, and JSON.
</div>

<div class="proof-item" markdown>
**Observable**

Model calls and tool dispatches can become durable evidence.
</div>

<div class="proof-item" markdown>
**Re-executable**

Recorded responses can exercise fresh tool and hook code.
</div>

<div class="proof-item" markdown>
**Gateable**

Host-observed outcomes become required release checks.
</div>

</div>

<div class="proof-terminal" markdown>

<div class="proof-terminal__title">$ uv run python examples/regression_demo/run_demo.py</div>

```text
1. CAPTURE v1 with fixed model responses
   collected profit: 200
   required eval: FAIL (0.00)

2. CHANGE one reviewable harness line
   - "profit": revenue + cost,
   + "profit": revenue - cost,

3. REPLAY with fresh v2 tool execution
   same model decisions: true
   collected profit: 40
   required eval: PASS (1.00)
```

</div>

<p class="proof-caption" markdown>
No API key and no network. The response sequence stays fixed while changed
tool code executes again and an independent collector checks the output.
[Read the proof and its limits.](regression-demo.md)
</p>

## Start with the job in front of you

<div class="home-paths" markdown>

<div class="home-path" markdown>

### I have a private tool loop

Adapt one tool, replace only the control loop, and establish parity before
adding hooks or cartridges.

[Migrate an existing loop](migrate.md) | [Build the first loop](quickstart.md)

</div>

<div class="home-path" markdown>

### I have a failure worth preserving

Capture the run, collect the real outcome, and decide whether replay, a mock,
or fresh model samples answer the question.

[Failure to regression](regression-demo.md) | [Choose an experiment](experiments.md)

</div>

<div class="home-path" markdown>

### I need the exact interface

Find commands, Python entry points, artifact files, and operational controls
without reading the package source.

[CLI](cli.md) | [Python API](api.md) | [Saved artifacts](artifacts.md)

</div>

</div>

## One workflow from prototype to release

<div class="workflow" markdown>

<div class="workflow-step" markdown>
<span class="workflow-step__num">01</span>

**Build**

Own the model, tools, state, and dispatch loop in Python or a cartridge.
</div>

<div class="workflow-step" markdown>
<span class="workflow-step__num">02</span>

**Capture**

Persist prompts, responses, steps, stop reasons, and metadata as readable files.
</div>

<div class="workflow-step" markdown>
<span class="workflow-step__num">03</span>

**Test**

Collect resulting world state and compare it with grader-only expectations.
</div>

<div class="workflow-step" markdown>
<span class="workflow-step__num">04</span>

**Ship**

Make required graders and thresholds fail closed in pytest or CI.
</div>

</div>

## The execution boundary stays visible

```python title="owner_lookup.py"
from looplet import OpenAIBackend, composable_loop, tool, tools_from


@tool(description="Look up one service owner by name.")
def lookup_owner(service: str) -> dict:
    owners = {"payments": "fintech-platform", "search": "discovery"}
    return {"service": service, "owner": owners.get(service)}


for step in composable_loop(
    llm=OpenAIBackend.from_env(),
    tools=tools_from([lookup_owner], include_done=True),
    task={"goal": "Find the owner of payments, then finish."},
    max_steps=5,
):
    print(step.pretty())
```

Every dispatch returns to the caller as a typed `Step`. Hooks can observe or
steer prompt construction, permissions, dispatch, completion, compaction, and
lifecycle events without requiring a graph runtime. Cartridges are optional;
they package the same harness as reviewable files when that helps distribution
or code review.

[Follow the quickstart](quickstart.md) | [Read the hook protocol](hooks.md) |
[Inspect cartridge boundaries](cartridge.md)

## Evidence has different jobs

| Evidence | Use it for | Do not claim |
| --- | --- | --- |
| Yielded `Step` stream | Live routing, approval, display, and instrumentation | Independent product correctness |
| Provenance trace | What the model saw, returned, and dispatched | That recorded prompts are safe to publish |
| Captured-response replay | Tool, hook, permission, state, and dispatch changes under fixed model responses | Better future model decisions |
| Outcome collector and grader | Whether the resulting file, command, record, or service state is correct | Isolation when the candidate owns the runner |
| Fresh sampled cases | Prompt, model, schema, and context changes that affect decisions | Universal performance from one sample |

Looplet calls replay **captured-response replay** because tools, clocks,
networks, randomness, and side effects execute again. Protected promotion
oracles belong in a host-owned runner; arbitrary untrusted code also requires
OS or process isolation.

[Capture and replay](provenance.md) | [Behavioral evals](evals.md) |
[Saved artifact reference](artifacts.md)

## Designed for a specific team and stage

<div class="fit-grid" markdown>

<div class="fit-panel fit-panel--yes" markdown>

### Looplet is a good fit when

- one model calls tools until it is done;
- your team already reviews Python, files, pytest, and CI;
- prompt, tool, model, or hook changes need regression evidence;
- exact interception points and local artifacts matter;
- you want to own execution rather than adopt a hosted control plane.

</div>

<div class="fit-panel fit-panel--no" markdown>

### Choose another layer when

- the workflow is naturally a durable branching graph;
- a managed control plane should be the source of truth;
- you need a finished assistant, sandbox, or annotation product;
- your main need is fleet analytics or a hosted experiment dashboard;
- a small disposable loop is still enough.

</div>

</div>

Looplet can run inside a workflow engine and export to observability systems.
It does not try to replace either one. Core uses only the Python standard
library; provider SDKs are optional extras.

[Read the selection guide](why-looplet.md) | [Check the FAQ](faq.md) |
[Operate a production loop](operations.md)

<p class="home-footer" markdown>
[Run the proof](regression-demo.md){ .md-button .md-button--primary }
[Install Looplet](install.md){ .md-button }
[Browse the tutorial](tutorial.md){ .md-button }
</p>

<!-- markdownlint-enable MD025 MD033 MD036 -->
