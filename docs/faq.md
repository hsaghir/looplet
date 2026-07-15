# FAQ

## What is Looplet?

Looplet is a small Python library for **test-driven agent harness
engineering**. It gives a single-loop, tool-calling agent an owned execution
loop, protocol-based hooks, reviewable cartridge files, captured run evidence,
captured-response replay, and outcome-grounded evals.

The model is not the harness. The harness is the code and policy around the
model: prompts, tools, state, permissions, context assembly, stop conditions,
evidence, and release gates. Looplet makes that layer ordinary Python and
ordinary files.

## Is Looplet an agent framework?

It is deliberately smaller than most products called agent frameworks. It
does not provide a graph DSL, hosted control plane, visual builder, vector
database, multi-agent protocol, or finished chat application. It provides the
engine and test bench for a harness you own.

That boundary matters after a prototype works. The difficult question is no
longer "can the model call a tool?" It is "can we change this prompt, tool,
hook, permission, or dependency without silently breaking the outcome?"

## Which layer should I use?

| If you need... | Start with... | Why |
| --- | --- | --- |
| A tiny experiment with one or two tool calls | A raw Python loop | It is the least machinery; add Looplet only when evidence, policy, or regression gates become real needs. |
| A single-loop Python agent whose harness must stay inspectable and testable | **Looplet** | Iterator-first execution, plain hooks, cartridges, provenance, replay, and eval contracts are one coherent layer. |
| A branching workflow with joins, durable node checkpoints, or graph visualization | A graph runtime such as LangGraph or Burr | The graph is real domain structure; forcing it into one loop would hide that structure. |
| A broad typed application framework with provider integrations and rich model abstractions | Pydantic AI, Strands, or another full framework | They intentionally own more of the stack and may save integration work. |
| Hosted experiment tracking, annotation queues, dataset management, or production monitoring | An eval/observability platform | Looplet writes portable evidence and eval results; it is not a hosted analytics product. |
| A ready-made coding or research agent | A turnkey harness or agent SDK | Looplet is for teams that want to own and change the harness, not merely configure a finished one. |

These choices can compose. A host application can route between several
Looplet cartridges. A Looplet run can export evidence to an observability
platform. A framework-owned application can use a different eval platform.
Choose based on which layer your team needs to own.

## Why not just write the loop ourselves?

Do that when the loop is genuinely disposable. A basic tool-call loop is not
hard to write.

Looplet becomes useful when the surrounding contracts matter:

- every tool call must be yielded as a typed `Step`;
- hooks must observe or steer behavior without forking the loop;
- prompts, responses, and trajectories must be saved as readable evidence;
- a failed run must become a case and required grader;
- captured model responses must be reusable while harness code changes;
- the same harness must be serializable, diffable, and loadable from files;
- sync and async behavior, cancellation, permissions, and parse failures must
  fail predictably.

The comparison is not "Looplet versus 20 lines." It is Looplet versus the
private harness those 20 lines tend to become.

## Why not LangGraph?

Use LangGraph when the application is truly a graph: multiple nodes own
different state, branches join, humans interrupt at node boundaries, or you
want graph-native checkpointing and visualization. That is a stronger fit
than Looplet.

Use Looplet when one model repeatedly calls tools until completion and you
want that loop—not a graph runtime—to remain the reviewable unit. Hooks handle
cross-cutting behavior; cartridges package the harness; collectors and graders
test its outcomes.

## Why not an eval platform?

Hosted eval products solve important problems Looplet does not try to solve:
team dashboards, annotation operations, large dataset exploration, trace
search, and managed production telemetry.

Looplet's eval machinery lives closer to source control and execution. Cases
are JSON, graders are Python, required checks have exit codes, and collectors
inspect world state after a run. Use it to make behavioral contracts part of
the harness and CI. Export its artifacts elsewhere when you need fleet-level
analysis.

## What is a cartridge?

A cartridge is a reviewable directory containing a harness contract: prompt,
tools, hook declarations, resources, memory, config, and optionally a colocated
self-test bundle. It loads into the same `AgentPreset` and loop used by the
Python API.

Think "deployable harness source," not "agent personality file." External,
host-owned holdouts should remain outside the cartridge so a candidate cannot
edit the oracle that promotes it.

## Is replay deterministic?

No. Looplet provides **captured-response replay**. It fixes the model responses
recorded in one run and executes a fresh harness against them. Tools, hooks,
permissions, state, clocks, networks, randomness, and other side effects run
again.

That makes replay useful for isolating changes to tool or hook behavior. It
does not predict how a prompt or model change would alter future decisions,
and it is not bit-for-bit simulation. Re-sample model calls for those
questions; mock or isolate side effects when needed.

## Does Looplet optimize prompts or evolve agents automatically?

No. Looplet supplies inspectable harness artifacts, evidence, cases, and
behavioral gates that an optimization system could use. Search algorithms,
statistics, candidate generation, and promotion policy belong outside the
core. Keeping them separate avoids turning Looplet into a dumping ground for
domain-specific optimization strategies.

## Does it require a model provider SDK?

No. The core has zero third-party runtime dependencies. Install the optional
`openai` or `anthropic` extra, or implement the small backend protocol over
your existing client. The regression demo and tests use scripted backends and
do not require a network or API key.

## When is Looplet the wrong choice?

Do not choose it merely because you need an agent-shaped demo. It is likely the
wrong layer if:

- your workflow is naturally a graph or durable state machine;
- you need a finished UI, hosted control plane, or managed sandbox;
- your team does not want to own Python harness code;
- you mainly need production analytics or annotation operations;
- a raw loop remains small enough to understand and throw away.

The best first step is the [network-free regression proof](regression-demo.md),
then the [selection guide](why-looplet.md).
