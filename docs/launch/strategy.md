# Relaunch strategy

## Decision

Position Looplet in the category **agent harness engineering**, with the wedge
**test-driven harness engineering for Python agents**.

The old frame—small/composable agent loop, or agents generated from a
paragraph—described implementation traits in a crowded framework market. It
did not create urgency. The new frame starts at the post-prototype problem:

> The agent works. Now how does a team change its prompt, tools, hooks, model,
> or permissions without guessing what broke?

Loop ownership remains the philosophy. Regression evidence becomes the reason
to adopt.

## Ideal user

**Primary ICP:** a Python team with a real, single-loop tool-calling agent that
has moved beyond its first demo.

Signals:

- prompt, model, or tool changes are reviewed in pull requests;
- failures recur but are not preserved as executable cases;
- the team has a private while-loop or is fighting a larger abstraction;
- final agent claims cannot be trusted without checking files, tests, APIs, or
  database state;
- pytest and CI are already normal workflow;
- local evidence and source ownership matter more than a hosted dashboard.

Not the primary ICP:

- someone seeking the fastest hello-world agent;
- a naturally branching durable workflow;
- a team wanting a finished assistant or hosted control plane;
- an eval organization primarily needing annotation queues and fleet-wide
  analytics.

## Category sentence

> Looplet is a small Python engine and test bench for a tool-calling harness
> you own: capture failed runs, change reviewable harness code, grade observed
> outcomes, and gate the behavior in pytest or CI.

## Message hierarchy

1. **Outcome:** change an agent harness with regression evidence.
2. **Proof:** a network-free failed run becomes a required red-to-green
   contract while captured model decisions remain fixed.
3. **Mechanism:** owned iterator loop + readable provenance + captured-response
   replay + collectors/graders.
4. **Review unit:** cartridges keep prompts, tools, hooks, resources, cases,
   and graders in ordinary files.
5. **Boundary:** one loop, no graph DSL, no hosted control plane, no automatic
   optimization, zero third-party core runtime dependencies.
6. **Caveat:** replay holds model responses constant; fresh tools and side effects still
   execute.

Do not lead with protocol count, import time, generated agents, or feature
breadth. Those can support a decision after the behavioral proof is clear.

## Proof ladder

Use claims in this order:

1. **Executable proof:** `examples/regression_demo/run_demo.py`.
2. **Protected test:** `tests/test_regression_demo.py`.
3. **Inspectable source:** cartridge case, collector, grader, and tool diff.
4. **Integrity suite:** fail-closed required graders, grader-only expected data,
   protected holdouts, path containment, redaction, and online/offline parity.
5. **Same-model study:** optional evidence that a compact owned harness can be
   competitive; never the hero claim.
6. **Package property:** zero third-party core runtime dependencies.

## Primary call to action

**Run the proof.** It is lower commitment and more credible than asking a new
reader to install a provider SDK or generate an agent.

Secondary calls to action:

1. bring an existing loop through the quickstart;
2. preserve one real failure as a case + collector + required grader;
3. open a behavioral-regression issue with redacted evidence.

## Launch sequence

### Phase 0 — gates

- Merge the relaunch only after proof, full checks, strict docs build, links,
  dark/light responsive views, and claims review pass.
- Ensure PyPI metadata and hosted docs show the same category sentence.
- Pre-create issue labels/backlog only after searching for duplicates.

### Phase 1 — repository relaunch

- Merge the documentation/metadata/proof PR.
- Deploy the site and verify canonical URLs.
- Publish the GitHub release/discussion announcement.
- Pin the proof-oriented discussion or issue.
- Make the regression proof the first README action.

### Phase 2 — technical launch

- Post Show HN with the executable mechanism, tradeoffs, and caveat—not a
  generic product pitch.
- Reply with concrete implementation details and acknowledge when a graph,
  hosted eval platform, or raw loop is a better fit.
- Avoid synchronized vote solicitation.

### Phase 3 — audience-specific distribution

- Python/testing audiences: emphasize pytest workflow and ordinary files.
- Agent engineering audiences: emphasize harness changes and oracle integrity.
- Local/OSS audiences: emphasize ownership and zero-dependency core.
- Do not cross-post identical copy; adapt the opening problem to each forum.

### Phase 4 — evidence follow-up

Within 48 hours:

- turn repeated questions into FAQ clarifications;
- open only the highest-signal roadmap issues;
- fix proof or docs failures before continuing promotion;
- publish one artifact walkthrough, not another positioning essay.

Within 30 days:

- collect opt-in third-party use cases;
- report proof runs, qualified issues, and external regression contracts;
- update claims only when evidence changes;
- decide whether the category resonates before expanding features.

## Success measures

Prefer evidence of qualified adoption over vanity metrics.

| Signal | Why it matters |
| --- | --- |
| Proof page visits → quickstart or repository | Readers understood enough to continue |
| Reproduced demo runs / questions about artifacts | The mechanism, not only the headline, landed |
| External cases + collectors + required graders | Users adopted the test-driven workflow |
| Behavioral-regression issues with usable evidence | The new issue path attracts the intended problems |
| Independent projects listed in `THIRD_PARTY_USERS.md` | Sustained use beyond the maintainer |
| Repeat contributors to integrity/docs/recipes | A category community is forming |

Stars, impressions, and launch-day traffic are useful diagnostics, not the
product goal.

## Response posture

- Be specific and non-defensive.
- Say "use LangGraph/Burr" when the workflow is a graph.
- Say "keep your raw loop" when evidence and policy are not yet problems.
- Say "use a hosted eval platform" for annotation operations and dashboards.
- Do not imply Looplet makes a model smarter or an agent safe.
- Ask critics for a concrete harness change and observable outcome; that is the
  level at which Looplet should be evaluated.
