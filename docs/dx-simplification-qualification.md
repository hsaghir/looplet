# DX Simplification Qualification

## Scope

This stack implements five local simplifications without replacing ordinary Python
functions, dataclasses, protocols, or the direct sync/async loop entry points:

1. Shared configuration resolution and redacted effective-value/source explanations.
2. Declaration-only cartridge and bundle inspection, separate from live instantiation.
3. One terminal-name/schema resolution path, preserving legacy declaration fields.
4. Shared optional-keyword and event-hook dispatch plans, preserving control semantics.
5. One content-redacted diagnostic schema for live results and persisted evidence.

`AgentRuntime` remains optional. Owned presets remain mutable and single-use;
repeated/concurrent executions need fresh factories. Borrowed components do not
silently become owned. Core still declares no third-party runtime dependencies.

The frozen control is `7b694c5c3b62e1a1b96804ffde4242dadb5f35ed`; the combined
runtime candidate is `4debaaf7136f268152c02aa30c34033e5ed21564`. This report is a
later documentation-only addition, not part of the executed runtime snapshot.

## Checks

Each implementation passed its independent full `make check`. The stacked push
gates passed 3,042 / 3,053 / 3,056 / 3,060 / 3,106 tests respectively, each with
two optional skips. The final source and noneditable installed-wheel suites both
passed 3,106 tests with two skips; text style, Ruff, formatting, and Pyright passed.
Imports were verified to resolve to `site-packages`, and wheel metadata confirms
zero unconditional runtime requirements.

All 22 shipped cartridges passed owned-preset contract validation and static
declaration inspection. Existing portability, delegation, lifecycle, quality-gate,
and static-command snippets ran in the suite. All seven installed demo entry
points also passed offline: hello, coding, data, Ollama hello, pretty presentation,
regression, and scripted approval. Pretty is a presentation demo, not live model
execution. The enterprise host example reports diagnostics, redacted configuration
origins, and resolved terminal names.

Fault checks covered direct/preset/runtime and sync/async execution, native and
explicit JSON-script modes, alternate terminal schema rejection/acceptance,
observer versus control-hook errors, weak callable identity, LEP bootstrap,
permissions, cancellation, checkpoint/restore, cleanup, and captured-response
replay. Replay holds responses fixed but reexecutes tools and side effects.

Dogfood found and repaired duplicate LEP bootstrap delivery, telemetry leaking
into tool-context metadata, alternate-terminal output loss, cancelled historical
output leakage, and stale bootstrap call/duration measurements. Scripted-demo
counts now derive from real rows, and approved deletion mutates those rows;
missing/denied approval does not. All failed probes and setup attempts were retained.

## Downstream Qualification

A frozen Analytics comparison used noneditable packages, identical non-Looplet
dependencies, identical source/task/oracle hashes, and nine matching opening
prompts. Baseline Analytics passed 685 tests; candidate passed 695, including ten
new serialization-shape cases. Ruff and lock checks passed. Both had the same
75 existing type diagnostics, compared as complete diagnostic multisets.

The completed live pair on 2026-10-02 used pinned DAB revision
`b24c8f5586121d4d2f8a5a793ebac530858dd1ab`, requested `gpt-5.6-luna` / max
effort, local Responses/native transport, direct profile, nine workers, 200 steps,
16,000 output tokens per call, 120-second analysis and 3,600-second case limits,
8,000,000-character context, and all uncertainty options disabled. Control ran
first; code/settings were not changed during either arm.

| Observed Result | Control | Candidate |
| --- | ---: | ---: |
| Accepted cases | 5/9 | 5/9 |
| Analytical completions | 8/9 | 8/9 |
| Retained case timeouts | 1 | 1 |
| Charged model calls | 214 | 309 |
| Steps | 320 | 442 |
| Captured responses | 213 | 308 |
| Identically regraded completed evals | 8 | 8 |
| Nested execution-failure steps | 22 | 30 |
| Persisted provider-usage records | 0 | 0 |

Both accepted BookReview 2, CRM 3/7/8, and StockIndex 1; no passing case was lost
or gained. AG News 3 timed out in both arms and has no completed eval bundle.
Each arm retains one charged in-flight call without a captured response.
All captured outer calls were native, with no text fallback. Original CLI exit 1
means not every required grader passed; the pair driver completed with exit 0.

Offline audits made no model calls and changed none of the 1,069 control or 1,475
candidate evidence files. Installed package, dependency, context, task, and oracle
hashes still matched their frozen snapshots. Candidate host diagnostic/configuration
reports exist for all eight completed cases; the killed case has no final report.
Shared typed tool errors are distinct from nested failures returned in tool data.

## Interpretation

This supports compatibility and observable host/declaration behavior, not an
accuracy, speed, cost, or leaderboard improvement. Candidate effort was higher.
One sequential matched pair cannot establish causal performance effects or
general non-regression. Official validator acceptance is not independent semantic
truth. Provider-returned model-version and billing evidence were not persisted;
the recorded model/effort are requested settings. Tokens and cost remain unknown.

Configuration explanations are not full mutation audits or secret anonymizers;
structural identifiers remain visible. Static views mark runtime-required bodies
and cannot prove runtime equivalence. Diagnostic logical calls are not billing or
retry counts, and completion is not grading correctness. Full persisted records
remain content-bearing host evidence, unlike the redacted diagnostic projection.

The five PRs form a reviewable dependency stack, not automatic deployment. The
downstream integration keeps prompts, tools, domain rules, and graders unchanged.
Primary dirty checkouts and active environments were not modified.

## Local Evidence

- Frozen inputs and post-run audits: `/tmp/looplet-dx-qualification-20261001/`.
- Original pair exit log: `/tmp/looplet-dx-matched-pair-20261001.log`.
- Installed-wheel gate: `/tmp/looplet-dx-installed-wheel-check-20261001.log`.
- Installed demos: `/tmp/looplet-dx-installed-demos-20261002.log`.
- Original live traces remain in the isolated Analytics worktrees under
  `eval-runs/nine-case-dx-{control,candidate}-9w-20261001-trial-1`.
