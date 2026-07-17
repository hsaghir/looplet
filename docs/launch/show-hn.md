# Show HN draft

<!-- markdownlint-disable MD034 MD036 -->

## Recommended title

**Show HN: Looplet – turn failed Python agent runs into regression tests**

Alternatives for a second venue, not A/B title churn on HN:

- Looplet: test-driven harness engineering for Python agents
- An owned agent loop with captured-response replay and outcome evals

Avoid "framework," "deterministic replay," "self-improving," "production-safe,"
and superlatives.

## Post

I built Looplet for the part of agent development that starts after the first
demo works.

A prompt or model change fixes one case and breaks another. A tool still
returns "success" but writes the wrong file. The final answer looks plausible,
but the world state is wrong. Most of the hard work is in the harness around
the model: prompt, tools, context, permissions, stop rules, and evidence. It is
not in writing another `while` loop.

Looplet is a small Python library for owning and regression-testing that
harness. The loop is an iterator that yields each tool dispatch as a typed
`Step`. Hooks are duck-typed objects. A cartridge can keep the prompt, tools,
hooks, resources, cases, collectors, and graders in reviewable files.

The smallest proof is network-free:

```text
1. CAPTURE v1 with fixed model responses
   model decisions: publish_report -> done
   collected profit: 200
   required eval: FAIL (0.00)

2. CHANGE one tool line
   - profit = revenue + cost
   + profit = revenue - cost

3. REPLAY captured responses through fresh v2 execution
   same decisions: true
   collected profit: 40
   required eval: PASS (1.00)
```

The collector reads `report.json` after the run; it does not trust the agent's
completion message. Expected data stays grader-only. Required graders fail the
CLI/CI when a discovered required grader is filtered, skipped, errored,
invalid, or below the gate. Detecting a grader deleted before discovery needs a
trusted expected-grader manifest and is not claimed here.

Important caveat: this is **captured-response replay**, not deterministic
simulation. The recorded model responses are held constant, but tools, clocks, networks,
randomness, and other side effects execute again. It is useful when changing a
tool, hook, permission, state, or loop runtime. Prompt/model changes need fresh
sampled runs.

Core Looplet has zero third-party runtime dependencies; OpenAI and Anthropic
SDKs are optional. It is for one model calling tools in a loop. If the system
is naturally a durable branching graph, use a graph runtime. If you want
hosted annotation and trace analytics, use an eval platform. If 20 lines still
solve the problem, keep the 20 lines.

Proof: https://hsaghir.com/looplet/regression-demo/

Repo: https://github.com/hsaghir/looplet

I would especially value feedback from people maintaining a real agent
harness: what change is hardest to regression-test today, and what outcome can
your host independently observe?

## First comment

The code behind the proof is intentionally small:

- one scripted backend with two fixed responses;
- one buggy and one fixed `publish_report` tool;
- one collector that reads the generated artifact;
- one required grader using grader-only expected data;
- persisted cartridges, trajectories, artifacts, and scores.

Source: https://github.com/hsaghir/looplet/tree/master/examples/regression_demo

The main design choice I would like to test publicly is the separation between:

1. a trace that explains a run;
2. replay that controls model-output variation;
3. an outcome eval that decides whether the behavior is acceptable.

Those often get collapsed into one "agent eval" concept, which can create
false confidence.

## Likely questions

### Why not just write the loop?

A disposable loop is a good choice. Looplet is for when that loop has grown
permissions, hooks, evidence, replay, serialization, and CI contracts. The
comparison is with the private harness the loop becomes, not with the first 20
lines.

### Why not LangGraph?

Use it when branching nodes, joins, graph-native checkpoints, and visualization
are real application structure. Looplet targets one repeated model→tool loop.

### Isn't replay misleading?

It would be if called deterministic. The docs explicitly scope it to captured
model responses and fresh harness execution. It controls one variable; it does
not freeze the world.

### Why colocate evals with the cartridge?

They are versioned self-tests. Protected promotion holdouts remain host-owned
outside candidate-editable files.

### Does this optimize prompts?

No. Search, statistics, candidate generation, and promotion algorithms belong
outside core. Looplet supplies artifacts and contracts those systems can use.
