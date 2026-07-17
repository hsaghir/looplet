# Social launch variants

Use one primary link per post. Prefer the executable proof over the homepage.
Do not post all variants simultaneously.

## One-line repository description

Test-driven harness engineering for Python agents: own the loop, capture failed
runs, and gate harness changes with outcome evals.

## Short post

Your agent works. Then a prompt, tool, or model changes. What actually broke?

Looplet is test-driven harness engineering for Python agents: an owned loop,
reviewable harness files, readable run evidence, captured-response replay, and
outcome graders for pytest/CI.

Run the proof with no model or network:
https://hsaghir.github.io/looplet/regression-demo/

## Technical thread

**1/** A tool-calling agent can finish successfully and still leave the world
wrong. The final response is not an oracle.

**2/** Looplet's regression workflow separates three jobs:

- provenance explains the run;
- captured-response replay controls model-output variation;
- collectors + graders decide whether the outcome is acceptable.

**3/** The demo fixes two model decisions, changes one tool line, reads the
fresh `report.json`, and moves one required grader from 0.00 → 1.00. No API key
or network.

**4/** Replay caveat: model responses are fixed; tools, clocks, networks, and
side effects execute again. Prompt/model changes require fresh sampled runs.

**5/** The harness can live as ordinary files: prompt, tools, hooks, resources,
cases, collectors, and graders. The loop still yields every `Step` to Python.

**6/** This is for a real single-loop harness after the prototype stage. Use a
graph runtime for a real graph, a hosted eval platform for annotation and
analytics, or keep a raw loop while it is enough.

Proof: https://hsaghir.github.io/looplet/regression-demo/

## LinkedIn / engineering-lead version

Most agent teams can build a tool-calling demo. The release problem comes
later: a prompt fixes one case and breaks another, a model upgrade chooses a
new path, or a tool reports success while producing the wrong artifact.

We are repositioning Looplet around that job: **test-driven harness engineering
for Python agents**.

Looplet makes the harness around the model reviewable and testable:

- your code owns an iterator-first loop;
- prompts, tools, hooks, resources, and evals can live in ordinary files;
- prompts, responses, steps, and stop reasons become readable evidence;
- captured responses can drive fresh harness execution;
- independent collectors grade world state and required checks fail CI.

The launch proof is intentionally small and network-free: fixed model decisions
call a buggy report tool, a collector observes the wrong profit, one harness
line changes, and the same required grader goes red → green.

Captured-response replay is not deterministic simulation; side effects execute
again. That limitation is part of the design, not buried in fine print.

https://hsaghir.github.io/looplet/regression-demo/

## Reddit / forum version

**Title:** OSS Python library for turning agent harness failures into pytest/CI contracts

I maintain Looplet, a zero-third-party-dependency core for a single model→tool
loop. I recently overhauled the project around the post-prototype problem
rather than "yet another agent framework."

The concrete workflow is: capture a run → preserve a case → collect actual
world state → write a required grader → change a reviewable harness file → use
captured-response replay when model responses should stay fixed → gate in CI.

The demo is scripted and network-free. It also calls out the limitation that
replay does not freeze tools or external side effects.

I would value criticism of the experiment boundary and eval trust model,
especially from teams keeping hidden holdouts outside candidate-editable agent
files.

Proof/source: https://hsaghir.github.io/looplet/regression-demo/

## Newsletter blurb

**Looplet reframes the agent loop as a testable harness.** The Python project
now leads with a network-free red-to-green proof: capture fixed model
responses, change one tool implementation, independently collect the fresh
artifact, and enforce the outcome with a required grader. It deliberately
targets single-loop agents rather than graph orchestration or hosted eval
analytics, and keeps zero third-party dependencies in core.
[Run the proof](https://hsaghir.github.io/looplet/regression-demo/).

## Copy to avoid

- "Deterministic replay for agents"
- "Make any agent production-safe"
- "Automatically optimize/evolve your agent"
- "A replacement for LangGraph"
- "The fastest/most accurate agent framework"
- "Evals the agent cannot game" (only a host-owned runner with an explicit
  isolation boundary can support that narrower claim)
- "No dependencies" without saying **core runtime** and acknowledging optional
  provider extras
