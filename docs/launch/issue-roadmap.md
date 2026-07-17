# Launch issue roadmap

The public backlog is now curated into independent work units. Every open task
has a concrete deliverable, explicit non-goals, and a verification path. The
`launch-ready` label means the scope is accepted and ready to claim; it does not
promise that the work belongs in core.

## Priority 0: release and feedback

- [#18](https://github.com/hsaghir/looplet/issues/18): pinned launch feedback
  focused on one concrete harness failure and an independently observable
  outcome. This is an operational thread, not claimable work.
- Release `0.3.0` only after the repository, docs, distribution, and public
  metadata gates in `launch-checklist.md` pass.

## Priority 1: teach the integrity boundary

- [#100](https://github.com/hsaghir/looplet/issues/100): runnable holdout recipe
  that keeps oracle capabilities out of candidate task, runtime, resources,
  tools, and writable files, with explicit OS-isolation limits.
- [#101](https://github.com/hsaghir/looplet/issues/101): replay experiment guide
  that distinguishes tool, hook, prompt, and model changes.
- [#105](https://github.com/hsaghir/looplet/issues/105): network-free migration
  from a private tool loop to one outcome-grounded regression contract.

These are small documentation/example contributions. They must not add a
sandbox product, deterministic replay claim, framework comparison benchmark,
or generated-agent promise.

Only #101 is currently a first-contributor, `launch-ready` chunk. #100 requires
a reviewed trust-boundary design, and #105 spans example, provenance, eval,
tests, and docs despite remaining useful launch work.

## Priority 2: make evidence portable and reviewable

- [#103](https://github.com/hsaghir/looplet/issues/103): provenance and eval
  artifact compatibility policy before any new schema fields land.
- [#102](https://github.com/hsaghir/looplet/issues/102): versioned JSON eval
  summary whose fail-closed states match the CLI exit code, blocked by #103.
- [#104](https://github.com/hsaghir/looplet/issues/104): redaction-preserving
  diff for persisted eval-run outcomes, separate from cartridge source diff
  and blocked by #103.

Dependency order: #103 establishes compatibility rules before #102 or #104
freezes a public representation. Only #103 carries `launch-ready` today;
#102 and #104 become claimable after that contract lands.

## Optional provider work

- [#4](https://github.com/hsaghir/looplet/issues/4): optional Gemini adapter
  using the current `google-genai` SDK.
- [#5](https://github.com/hsaghir/looplet/issues/5): optional Bedrock Converse
  adapter, pending a reviewed async/dependency design before it becomes
  launch-ready.
- [#6](https://github.com/hsaghir/looplet/issues/6): verified
  `llama-cpp-python` recipe; PR #91 is the active contribution path, so it is
  not advertised as unclaimed work.

Provider work stays in optional extras or documentation. It does not change the
core loop, and compatibility claims must name the model and verification date.

## Closed during launch cleanup

- #10 was partially delivered: the smaller Makefile contributor contract
  ships, while the remaining wrapper targets were explicitly closed as not
  planned.
- #7 duplicated the shipped `planner.cartridge` subagent pattern and pulled the
  product toward a turnkey research agent.
- #8 treated trajectory efficiency metrics as generic quality scores.
- #55 bundled five unrelated generated-agent recipes into one issue.
- #56 called factory quality the product and proposed step count as a release
  gate. The factory is now optional scaffolding, not the product boundary.

## Triage rubric

Score each proposed core change from 0 to 2 on:

1. observed user failure;
2. independently verifiable outcome;
3. cross-domain generality;
4. inability to express it with existing composition points;
5. preservation of fail-closed integrity;
6. zero-dependency core;
7. network-free acceptance test.

Do not schedule a core feature below 11/14. Use a recipe or downstream
experiment to gather missing evidence first. Search, statistics, optimization,
domain policy, dashboards, and turnkey agents remain outside core.
