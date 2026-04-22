# FAQ

## Why not LangGraph?

LangGraph is the right tool for a lot of agent work, and for some projects it's
a better fit than `looplet`. Here's the honest comparison so you can pick
deliberately instead of by vibes.

### Use LangGraph when

- **Your agent is a multi-node graph, not a loop.** A triage node hands off
  to a research branch and a coding branch, each branch has its own state,
  and they join again at a review node. That's the shape LangGraph is
  designed around. Trying to express it as a single `for step in loop(...)`
  with hooks is fighting the tool.
- **You want durable checkpointing as a first-class feature.** LangGraph has
  built-in checkpointer backends (SQLite, Postgres, Redis) with interrupt
  and resume semantics tied directly to node boundaries. `looplet` lets you
  build that — `ProvenanceSink` dumps every step and a hook can persist
  state — but LangGraph gives it to you out of the box.
- **You're already in the LangChain ecosystem.** Your prompts are
  `ChatPromptTemplate`s, your retrievers are LangChain retrievers, your
  streaming consumers expect LangChain events. Bridging all of that through
  `looplet` adds friction.
- **You need the graph visualiser.** LangGraph's Studio / `.get_graph()` view
  is genuinely useful for explaining a multi-agent system to someone who
  doesn't want to read Python.

### Use looplet when

- **Your agent really is a loop.** One LLM calling tools until it's done.
  That's ~80% of real agent work, and a graph for it is overkill.
- **You want to see every prompt the LLM saw, in order, without a debugger.**
  `step.pretty()` and `ProvenanceSink` are the whole point.
- **You want to shape behaviour without forking.** A 10-line hook that
  redacts PII, injects docs, or rewrites tool args composes with everything
  else — no class hierarchy, no node refactor.
- **Cold-import time and dependency footprint matter.** Core `looplet` pulls
  in zero third-party packages; see [Benchmarks](benchmarks.md).
- **You want your debug trace and your eval harness to be the same
  artifact.** The pytest-style `eval_*` helpers read `ProvenanceSink`
  output directly.

If you're building a research swarm or a multi-stage ETL with a literal DAG,
use LangGraph. If you're building an agent that reasons and calls tools
until it's done, the loop is the product and `looplet` stays out of your way.
