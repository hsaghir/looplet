# Bounded Tool Results Dogfood

This report qualifies unreleased result projection and retention contracts.
It is a focused integration check, not an accuracy benchmark or protected promotion oracle.
Earlier failed runs remain separate from the qualified results below.

## Candidate

| Input | Revision or setting |
| --- | --- |
| Core base | `1c90ef2a55d194e4aadd78dabf97185bcb53dbad` |
| Analytics base | `0e469947c58826085fec23950f354b00b8424ad1` |
| Models | `gpt-6.1-sol`, `gpt-6.1-terra` |
| Proxy | `http://127.0.0.1:19823/v1` |
| Python | `3.13.7` |
| OpenAI SDK | `3.6.0` |

Trials used an installed Core wheel, not an editable Core import.
All 304 non-metadata wheel members matched the source and installed files.
The qualified wheel SHA-256 is:

```text
bf5dbe79aa483490b9cb8e269f30ff76908f1444fbdb378f44d76729f908bf84
```

The final controller rechecked those source and installed bytes before and after qualification.
It also bound both changed Analytics files by SHA-256.
Analytics kept its existing dependency pin; a separate environment tested that unchanged pin.
The consumer repair recomputes selected projections from observed original receipts and rejects mismatches.
It does not replace the original receipt with its model preview.

## Examples

Six scripted trials passed without HTTP requests.
Twelve real-endpoint trials passed across both models and Python/cartridge entry paths.
The host observer captured 91 client HTTP sends, matching the reported backend calls per trial.
SDK retries were disabled explicitly.

| Case | Library | Cartridge | Observed outcome |
| --- | --- | --- | --- |
| Invoice reconciliation | 2/2 | 2/2 | Three exact saved paid-total records |
| Policy extraction | 2/2 | 2/2 | Four exact saved section and requirement records |
| Settlement evidence | 2/2 | 2/2 | Correct independently graded settlement file |

Invoice and policy examples intentionally retain large per-tool previews to exercise compaction.
The settlement example uses a source report exceeding 200,000 characters.
Its decisive section is absent from the initial 3,000-character result envelope.
Each settlement trial observed an explicit oversized-page refusal, then a smaller delivered page.
The grader checked the actual saved reference and amount, not the model's completion claim.
An independently reopened `FileToolResultStore` recovered the complete original report.

Deterministic tests cover sync/async, native/JSON-text, and Python/cartridge execution.
They also check escaped nested JSON, aggregate batch limits, stable admitted previews,
fixed-boundary discovery, quota failures, line endings, and resource/policy round trips.

## Analytics

The final randomized large-output check passed over Chat and Responses, with 13 captured sends.
The host generated the expected amount before running the agent and kept it outside the task.
The actual `workspace_exec` output passed through Core before projection.
Both transports delivered a source envelope of at most 6,000 characters without the answer,
preserved its source reference, delivered a retrieval page containing the answer,
and completed with the independently graded integer.
An earlier two-transport check also passed with 13 sends; it is retained separately.

The broader eight-case workflow matrix initially passed five cases with 65 captured sends.
The private observer incorrectly rejected explicit review sampling before transmission.
After that observer-only correction, separate retries passed both policy cases;
the Terra grounded join still refused context. Those three retries used 23 sends.
Successful cases were not replayed for collection repair.

Final workflow coverage is seven of eight cases, not eight of eight.
Saved-response diagnostics reproduced the remaining helper-admission refusal with both
the candidate and the unchanged pinned Core, without any network sends.
This establishes a baseline limitation for that saved sequence, not a passing live workflow.
No analytical approval rule, model budget, or text-fallback policy was weakened.
Outer calls leave sampling unset, but existing review helpers explicitly select `temperature=0.0`.

## Local Gates

| Gate | Final result |
| --- | --- |
| Core `make check` | 6,700 passed, 2 skipped; text style, lint, formatting, and Pyright passed |
| Analytics with candidate Core | 959 passed |
| Analytics with unchanged pinned Core | 955 passed, 4 expected projection-test skips |
| Analytics lock, Ruff, full Pyright | Passed; Pyright checked both Core environments |
| Analytics sdist and wheel | Built successfully |

The earlier complete Core run found two regressions; both were repaired before the final gate.
An earlier Analytics run had 958 passes and one `os.setsid()` failure because pytest itself
was a process-group leader. A child-process launch passed that focused test and both full suites.
Warnings and failed receipts remain preserved. Local checks used Python 3.13;
other supported Python versions remain CI responsibilities.

## Reproduce

Build the candidate Core wheel, install Analytics' locked development environment,
then substitute that wheel without changing the project's dependency pin:

```bash
uv sync --locked --group dev
uv --no-config pip install --python .venv/bin/python --no-deps --reinstall "$CORE_WHEEL"

export OPENAI_BASE_URL=http://127.0.0.1:19823/v1
export OPENAI_API_KEY=local-proxy
RUN_ROOT=$(mktemp -d)
uv run --no-sync python -m looplet.examples.default_harness \
  --scripted --cases invoices policy evidence --output "$RUN_ROOT/scripted"
uv run --no-sync python -m looplet.examples.default_harness \
  --cases invoices policy evidence --output "$RUN_ROOT/live"
uv run --no-sync python "$CORE_CHECKOUT/scripts/dogfood_default_harness_analytics.py" \
  --output "$RUN_ROOT/analytics"
```

Set `CORE_WHEEL` and `CORE_CHECKOUT` to the candidate artifact and checkout.
Every output directory must be new. Do not synchronize after substituting the wheel.
The randomized delivery check and wire observer are host-owned qualification scripts.

## Evidence And Limits

The private local bundle is `looplet-bounded-results-evidence-2v4xYLv3`.
Qualified example receipts are under `qualified-v2`; final consumer and gate receipts
are under `qualified-v2/final`. `artifact-parity.json`, `consumer-source.json`,
`final-gates.json`, and the two Analytics JUnit files bind the final qualification.
Earlier wheels, failures, retries, stdout/stderr, and wire receipts remain distinct.
Raw source data, provider continuity items, and authenticated headers are not committed.

The observer counts client sends, not hidden proxy retries or provider billing.
Character budgets cover result envelopes, not exact provider tokens or whole requests.
Retrieval access is host authority, not redaction or a sandbox.
File retention has a content quota and one active writer per directory, not fsync/WAL durability.
Raw steps, provenance, checkpoint snapshots, and selected JSON values can still use full memory.
See [bounded model views](provenance.md#bounded-model-views) for the operational contract.