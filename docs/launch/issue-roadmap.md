# Relaunch issue roadmap

These are issue drafts, not permission to expand core. Search open/closed issues
before posting. Open only items with a maintainer, a narrow acceptance test, and
no duplicate. Suggested order follows user value, not implementation novelty.

## Launch-ready issues

### 1. Add a `looplet regression` recipe that promotes a saved failure to a case skeleton

**Labels:** `enhancement`, `evals`, `help wanted`

**Problem:** The conceptual path from a useful saved run to a hand-authored case
is clear, but users must copy task metadata and artifact fields manually.

**Scope:** A recipe or CLI helper that writes a reviewable **draft** case and
collector TODOs. It must not infer correctness, generate expected values from
the agent's claim, or mutate a cartridge without an explicit target.

**Acceptance:**

- network-free fixture input;
- agent-visible task copied without grader-only data;
- output refuses overwrite by default;
- explicit TODO for host-observed outcome and expected oracle;
- docs state that promotion requires human review.

**Boundary:** Drafting ergonomics only; no automatic grader generation or
self-improvement loop.

---

### 2. Emit a stable machine-readable eval summary for CI annotations

**Labels:** `enhancement`, `evals`

**Problem:** Human CLI output is useful locally, but CI integrations need a
versioned summary of case, grader, required status, score/label, and errors.

**Acceptance:**

- documented schema version;
- required/missing/error states remain distinct;
- collector failures cannot serialize as passing;
- golden compatibility tests;
- no hosted service dependency.

**Boundary:** Portable output only; dashboards and analytics remain downstream.

---

### 3. Publish a host-owned holdout reference recipe

**Labels:** `documentation`, `eval-integrity`, `good first issue`

**Problem:** Colocated self-tests are easy to understand; teams need one exact
example where the candidate can edit its cartridge but cannot read or modify
the promotion suite.

**Acceptance:**

- candidate and holdout in separate roots;
- host runtime binds protected collector/expected data;
- test proves traversal/symlink attempts cannot reach the oracle;
- docs compare visible guidance tests, cartridge self-tests, and release
  holdouts.

**Boundary:** Filesystem recipe and tests; no new sandbox product.

---

### 4. Add PR-friendly cartridge and eval-run diff output

**Labels:** `enhancement`, `provenance`, `help wanted`

**Problem:** `looplet diff` covers harness structure, while reviewers also need
a concise, redaction-aware summary of changed outcomes across persisted runs.

**Acceptance:**

- separate harness diff from outcome/eval diff;
- no claim that step-sequence differences are quality regressions;
- stable text output suitable for PR comments;
- secrets are never newly revealed by the renderer;
- fixtures cover missing/malformed artifacts.

**Boundary:** Comparison/reporting only; no statistical significance engine.

---

### 5. Document replay experiment patterns: tool change, hook change, prompt change

**Labels:** `documentation`, `good first issue`

**Problem:** Users can misuse captured-response replay to evaluate a prompt
change even though responses are already fixed.

**Acceptance:**

- executable tool-change example using replay;
- hook/permission example with mocked side effects;
- prompt/model example using fresh sampled runs instead of replay;
- decision table naming fixed and fresh variables.

**Boundary:** Documentation/examples only unless a concrete API defect appears.

---

### 6. Define provenance and eval artifact schema/version policy

**Labels:** `api-design`, `provenance`, `evals`

**Problem:** Evidence is intentionally portable, but downstream consumers need
to know which fields are stable and how migrations fail.

**Acceptance:**

- inventory current artifact files and schemas;
- classify stable, optional, and internal fields;
- forward/backward compatibility policy;
- fixtures for at least one prior-version load;
- no opaque database or mandatory exporter.

**Boundary:** Stabilize existing artifacts before adding fields.

## Evidence-building issues (open after launch feedback)

### 7. Migration recipe: replace a private while-loop incrementally

Produce before/after code showing tools retained, iterator introduced, then
provenance and one behavioral contract added. Avoid framework comparison
rhetoric. Acceptance is a network-free tested example and a "keep your raw
loop" exit criterion.

### 8. Outcome collector cookbook

Add small tested recipes for file/schema, subprocess/test suite, HTTP probe with
mock transport, and database query with an in-memory fixture. Keep domain
policy in examples. Each recipe must state trust and side-effect boundaries.

### 9. Optional OpenTelemetry evidence export recipe

Map stable Looplet steps/provenance to OTel without adding a core runtime
dependency. Acceptance includes optional installation, no import impact on
core, redaction guidance, and a network-free exporter test.

### 10. Re-run the same-model harness benchmark with repeated trials

Only proceed with a preregistered task set, environment metadata, repeated
runs, uncertainty intervals, raw task-level output, and no universal
performance headline. Keep benchmark dependencies outside runtime.

## Questions to turn into issues only with evidence

- A new lifecycle hook: show why existing hook phases cannot express it.
- A reusable grader: provide at least two unrelated domains and a protected
  outcome source.
- New cartridge syntax: show the repeated plain-Python/file pattern and a
  migration strategy.
- Prompt optimization or harness search: build downstream first; core work is
  limited to missing generic artifact/contract primitives proven by that use.
- Hosted UI or annotation flow: integrate downstream; do not add to Looplet
  core.

## Triage rubric

Score each candidate issue 0–2 on:

1. observed user failure;
2. outcome can be verified independently;
3. generalizes across domains;
4. cannot be expressed with existing composition points;
5. preserves fail-closed integrity;
6. keeps runtime dependency-free;
7. has a network-free acceptance test.

Do not schedule a core feature below 11/14. A recipe or downstream experiment
may proceed earlier to gather the missing evidence.
