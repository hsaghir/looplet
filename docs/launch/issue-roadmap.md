# Contributor issue roadmap

Last reviewed: 2026-10-22 against published `master`.

There are six open contribution tasks and one feedback thread. Each task has
an outcome, explicit non-goals, and a verification path. The `launch-ready`
label means its scope is accepted; follow any proposal or review checkpoint in
the issue before coding. It does not promise that the work belongs in core.

## Start here

- [#101](https://github.com/hsaghir/looplet/issues/101): the focused first
  contribution. Extend the existing [experiment guide](../experiments.md)
  with an explicit fixed/fresh/isolated variable matrix and a network-free,
  isolated hook or permission example. Reuse the tested
  [failure-to-regression proof](../regression-demo.md) for the tool-change
  pattern instead of creating another replay guide.
- [#6](https://github.com/hsaghir/looplet/issues/6): a verified
  `llama-cpp-python` local-model recipe. This is documentation-only and
  available for a fresh contribution: PR #91 closed without the required
  end-to-end verification. A contributor needs suitable local-model hardware
  and must record the exact model, quantization, hardware, and verification
  date. It is not a network-free verification task.

## Substantial engineering contribution

- [#104](https://github.com/hsaghir/looplet/issues/104): a redaction-preserving
  diff of persisted eval-run outcomes, separate from cartridge source diff.
  The former #103 dependency is complete. Current runs retain a frozen grader
  inventory in trajectory metadata, and `EvalReport.from_records()` reads it.
  Propose the command or helper, exit codes, and legacy-record handling before
  implementation; do not guess required marks from grader names or treat
  missing inventory as a passing comparison.

This is no longer blocked on an artifact-format project, but still needs an
API and compatibility proposal. It is not a good first issue.

## Maintainer-reviewed example

- [#100](https://github.com/hsaghir/looplet/issues/100): a runnable holdout
  recipe that keeps oracle capabilities out of candidate task, runtime,
  resources, tools, prompts, and writable files. Existing explanatory docs do
  not replace the requested red/green example and traversal/symlink tests
  through a real candidate tool boundary. Review the data-flow and isolation
  design before coding; separate directories alone are not a sandbox.

This trust-boundary contribution must not add a managed sandbox product,
automatic promotion, or a claim of protection against arbitrary same-user
code without OS or process isolation.

## Optional provider work

- [#4](https://github.com/hsaghir/looplet/issues/4): optional sync and async
  Gemini adapters using `google-genai`. Follow the current public backend
  contracts for request admission, retry ownership, and observed metadata.
  Propose signatures and a representative response fixture before coding.
- [#5](https://github.com/hsaghir/looplet/issues/5): optional Bedrock Converse
  adapters, pending a reviewed async strategy and dependency choice. It is
  not launch-ready until that design is accepted.

These adapters remain missing, but provider expansion is optional rather than
a prerequisite for the documentation and evidence work above. Keep SDKs in
optional extras, preserve lazy imports and sync/async parity, and name the
verified model and date. Do not change the core loop or advertise unsupported
conversation or retry capabilities.

## Feedback, not claimable work

- [#18](https://github.com/hsaghir/looplet/issues/18): the pinned feedback
  thread for one concrete harness failure and an independently observable
  outcome. Keep implementation proposals in their own issues.

## Completed prerequisites

- [#103](https://github.com/hsaghir/looplet/issues/103): the saved-artifact
  compatibility policy is documented in the [artifact reference](../artifacts.md).
- [#102](https://github.com/hsaghir/looplet/issues/102): versioned eval summaries
  and fail-closed assessment semantics are available; they are not pending
  contribution tasks.
- [#105](https://github.com/hsaghir/looplet/issues/105): the network-free
  private-loop migration example is delivered, not an open task.

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
