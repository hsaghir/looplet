# Roadmap: test-driven harness engineering

Looplet is the small Python engine and test bench for an agent harness you
own. The roadmap is organized around one question:

> Can a team change a real tool-calling harness and know, before release, what
> behavior improved, regressed, or remains unproven?

This is a direction document, not a dated promise. Shipped behavior is
specified by the code, tests, and changelog.

## Current status: 0.3.0 (Beta)

The foundation is in place:

- iterator-first sync and async tool-calling loops that yield explicit `Step`
  records;
- protocol/duck-typed hooks for policy, context, permissions, stop conditions,
  capture, and evaluation;
- cartridges that package prompts, tools, config, hooks, resources, memory,
  runtime defaults, and optional self-test eval bundles as reviewable files;
- prompt, response, and trajectory evidence through `ProvenanceSink`;
- captured-response replay through a fresh harness execution;
- case data, outcome collectors, pytest-style graders, required marks, and CI
  exit codes;
- grader-only expected data and explicit guidance for host-owned holdout
  boundaries;
- zero third-party dependencies in the core runtime, with provider SDKs as
  optional extras.

The public proof is the
[network-free failure-to-regression demo](regression-demo.md): one fixed
set of model decisions, one tool-code change, and one required outcome grader
moving from red to green.

## Design constraints

Every roadmap item must preserve these constraints.

1. **Own the loop.** The host can observe, interrupt, and compose every step.
2. **Test outcomes, not historical trajectories.** Tool sequences are useful
   for debugging and harness-plumbing assertions, not as a default quality
   oracle.
3. **Fail closed at integrity boundaries.** Discovered required graders that
  are filtered, skipped, errored, invalid, or failing; empty grader suites;
  collector errors; malformed expected data; and sandbox escapes must not
  produce false greens. Detecting deletion before discovery requires a trusted
  expected-grader manifest.
4. **Keep the core domain-neutral.** New behavior should compile into a hook,
   tool, collector, grader, cartridge, recipe, or downstream package unless
   the execution engine itself must understand it.
5. **Use ordinary Python and files.** No graph DSL, magic globals, mandatory
   inheritance, or hidden control plane.
6. **Name experimental limits.** Captured-response replay holds model responses constant;
   it does not make fresh tools or side effects deterministic.
7. **Keep promotion oracles outside candidate authority.** Colocated cartridge
  evals are versioned self-tests. A promotion runner must keep oracle data and
  capabilities out of candidate inputs, runtime, resources, tools, and files;
  arbitrary candidate code requires OS or process isolation.
8. **Preserve zero-dependency core.** Optional integrations must not impose
   ambient cost on every user.

## Priority 1: make the regression workflow obvious

The first product goal is for a post-prototype Python team to complete this
path without reverse-engineering the library:

1. express its existing loop with `composable_loop()`;
2. capture a failed run as readable evidence;
3. preserve the task as a case;
4. collect the resulting world state independently;
5. write a required grader;
6. change one reviewable harness component;
7. replay captured responses where that experiment is valid;
8. run fresh sampled cases when model decisions are the variable;
9. gate the behavior in normal CI.

Planned work:

- keep one network-free red-to-green proof protected by tests;
- add a trusted expected-grader manifest before claiming deletion detection;
- improve errors for invalid required graders, malformed bundles, and unsafe
  case paths;
- make persisted eval runs easy to inspect and attach to pull requests;
- document clear choices between replay, mocks, and fresh model sampling;
- publish migration recipes for teams replacing a private raw loop without
  rewriting their tools.

## Priority 2: behavioral contract ergonomics

Cases, collectors, and graders should feel as routine as pytest fixtures and
assertions while retaining explicit trust boundaries.

Planned work:

- better CLI summaries for per-case required checks and collector failures;
- stable machine-readable reports for CI annotations and downstream systems;
- recipes for common outcome classes such as file artifacts, command results,
  structured reports, and host-owned test suites;
- explicit helpers for separating agent-visible fixtures from grader-only
  expected data;
- documentation and tests for versioned cartridge self-tests plus host-owned
  holdouts with explicit isolation boundaries;
- small, evidence-backed grader utilities only where they generalize across
  domains.

Not planned for core: leaderboard statistics, experiment dashboards,
annotation operations, generic prompt scoring, or a large catalog of
subjective domain graders. Those fit downstream packages and hosted eval
platforms better.

## Priority 3: evidence portability and review

A run should be understandable without a proprietary viewer and useful beyond
Looplet itself.

Planned work:

- stabilize and document provenance/eval artifact schemas;
- add schema-version and compatibility tests for saved evidence;
- improve human-readable harness and run diffs;
- strengthen redaction guidance and secret-safety tests;
- provide optional export recipes for OpenTelemetry and hosted observability
  systems without coupling core execution to them;
- keep artifact directories composed of readable text and JSON wherever
  practical.

Captured-response replay will remain honestly scoped. Bit-for-bit replay of
arbitrary side-effecting tools is not a core promise; deterministic behavior
requires the host to isolate or mock those effects.

## Priority 4: cartridge lifecycle hardening

Cartridges are the review and distribution unit for a harness, not a second
runtime.

Planned work:

- tighten schema validation and migration diagnostics;
- preserve round-trip guarantees for the supported JSON-able surface;
- test loading, resource cleanup, inheritance, references, and runtime-tier
  wiring across supported Python versions;
- keep `describe`, `diff`, `hash`, and portability reports useful in code
  review;
- clarify registry/signing integration while keeping signatures outside the
  content-addressed cartridge body;
- maintain one canonical format shared by the Python API, CLI, examples, and
  eval runner.

## Ecosystem work that belongs outside core

The following projects are useful, but should consume Looplet rather than
expand its execution engine:

- harness search, prompt optimization, evolutionary strategies, and candidate
  promotion algorithms;
- statistical experiment analysis and benchmark orchestration;
- domain-specific agents, graders, permissions, and tool catalogs;
- hosted trace dashboards, annotation queues, and dataset management;
- registries, signatures, and organizational policy distribution;
- finished coding agents, chat shells, and web applications.

Looplet should make these systems easier to build by exposing stable steps,
artifacts, cartridges, and eval contracts. It should not pick one search or
optimization philosophy for every user.

## Explicitly not on the roadmap

- A graph DSL or branching workflow designer. Use a graph runtime when the
  application is truly a graph.
- A built-in multi-agent handoff protocol.
- A prompt-template language, vector database, or memory store.
- A hosted control plane, web dashboard, annotation product, or sandbox
  service.
- A turnkey coding/research agent as the core product.
- Automatic harness optimization or claims of self-improvement.
- Unrestricted deterministic replay of real-world side effects.
- Domain-specific loop phases, planner modes, to-do systems, or hidden global
  state.

## Path to 1.0

Looplet reaches 1.0 when:

1. the owned-loop, hook, cartridge, provenance, and eval surfaces have a
   documented compatibility contract;
2. at least three independent codebases use those surfaces for real harness
   changes over a sustained period;
3. required evals and protected-oracle boundaries have no known false-green
   integrity defects;
4. sync and async paths remain in behavioral parity;
5. saved artifacts have explicit schema/version compatibility tests;
6. the Understand → Build → Test → Ship documentation path is complete;
7. there are no unresolved issues tagged `api-design` or `breaking`.

A public symbol is not added merely to make the roadmap look complete.
Pre-1.0 additions still require a concrete use case, a generalization argument,
and regression evidence.

## How to influence the roadmap

Open an issue with:

- the harness change you are trying to make;
- the observable outcome or failure;
- why existing hooks, tools, collectors, graders, cartridges, or downstream
  composition are insufficient;
- a minimal reproduction or evidence bundle;
- the smallest API that would solve the generalized problem.

A focused external recipe is often the fastest path to evidence. Promotion
into core comes only after the pattern proves general.
