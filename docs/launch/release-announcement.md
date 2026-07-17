# Release / repository announcement

<!-- markdownlint-disable MD034 MD036 -->

## Title

**Looplet, relaunched: own the loop, test every harness change**

## Short release summary

Looplet is being relaunched around the job it is best at: **test-driven harness
engineering for Python agents**.

The relaunch will ship as `looplet==0.3.0`; `0.2.0` remains the earlier
published distribution with the old public story until release.

The new entry point is a network-free regression proof. It captures one failed
run, changes one reviewable tool line, replays the same model responses through
fresh harness code, independently collects the resulting artifact, and moves a
required grader from red to green.

## Full announcement

A working agent demo is no longer the hard part. The hard part is changing the
harness around the model without guessing what regressed.

The Looplet relaunch brings its public story in line with that problem:

> **Own the loop. Test every change.**

Looplet is a small Python engine and test bench for a single-loop,
tool-calling harness. It keeps execution observable, harness source reviewable,
and behavioral evidence close to pytest and CI.

### What is now front and center

- **Owned execution:** `composable_loop()` yields every tool dispatch as an
  explicit `Step` to host code.
- **Reviewable harnesses:** cartridges keep prompts, tools, hooks, resources,
  runtime policy, cases, collectors, and graders in ordinary files.
- **Readable evidence:** provenance capture persists prompts, responses,
  trajectories, stop reasons, and metadata.
- **Captured-response replay:** recorded model outputs can drive a fresh
  harness without another model call.
- **Outcome contracts:** collectors inspect world state; graders use
  grader-only expected data; discovered required checks fail closed in pytest
  or CI.

### Run the proof

```bash
git clone https://github.com/hsaghir/looplet
cd looplet
uv sync
uv run python examples/regression_demo/run_demo.py
```

After publication, install it with `pip install --upgrade looplet==0.3.0`.

It requires no API key or network. The generated directory contains both
harness versions, the captured model-call cassette, fresh workspaces,
trajectories, independently collected artifacts, grader-only expected data, and
grader results.

Proof walkthrough: https://hsaghir.com/looplet/regression-demo/

### An important boundary

Replay is not deterministic simulation. It fixes captured model responses;
tools and side effects execute again. Use it to isolate harness-runtime changes.
Use fresh sampled runs to evaluate prompt or model changes.

Looplet also does not try to become a graph runtime, hosted eval platform,
turnkey assistant, or automatic optimizer. Those systems can consume its
steps, artifacts, cartridges, and behavioral contracts without being pulled
into core.

### Existing users

The low-level philosophy is unchanged: iterator-first execution, plain Python
protocols, provider-agnostic backends, and zero third-party dependencies in the
core runtime. The relaunch changes the public center of gravity, not the narrow
core.

Start here:

- Docs: https://hsaghir.com/looplet/
- Selection guide: https://hsaghir.com/looplet/why-looplet/
- Quickstart: https://hsaghir.com/looplet/quickstart/
- Repository: https://github.com/hsaghir/looplet

If you maintain a real harness, open a behavioral-regression issue with a
redacted case and observable outcome. Those failures should drive what Looplet
builds next.

## GitHub discussion prompt

What is the last agent failure that looked successful in the final response
but was wrong in the world? If you can describe the host-observable outcome,
we can help translate it into a collector and regression contract.
