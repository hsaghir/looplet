# Bounded Tool Results Dogfood

This report qualifies unreleased result projection and retention contracts.
It is a focused integration check, not an accuracy benchmark or protected promotion oracle.
Earlier failed runs remain separate from the qualified results below.
The initial qualification appears first. See the [follow-up qualification](#follow-up-recovery-qualification) for the later artifacts and retained failures.

## Candidate

| Input           | Revision or setting                        |
| --------------- | ------------------------------------------ |
| Core base       | `1c90ef2a55d194e4aadd78dabf97185bcb53dbad` |
| Downstream base | `0e469947c58826085fec23950f354b00b8424ad1` |
| Models          | `gpt-6.1-sol`, `gpt-6.1-terra`             |
| Proxy           | `http://127.0.0.1:19823/v1`                |
| Python          | `3.13.7`                                   |
| OpenAI SDK      | `3.6.0`                                    |

Trials used an installed Core wheel, not an editable Core import.
All 304 non-metadata wheel members matched the source and installed files.
The qualified wheel SHA-256 is:

```text
bf5dbe79aa483490b9cb8e269f30ff76908f1444fbdb378f44d76729f908bf84
```

The final controller rechecked those source and installed bytes before and after qualification.
It also bound both changed Downstream files by SHA-256.
Downstream kept its existing dependency pin; a separate environment tested that unchanged pin.
The consumer repair recomputes selected projections from observed original receipts and rejects mismatches.
It does not replace the original receipt with its model preview.

## Examples

Six scripted trials passed without HTTP requests.
Twelve real-endpoint trials passed across both models and Python/cartridge entry paths.
The host observer captured 91 client HTTP sends, matching the reported backend calls per trial.
SDK retries were disabled explicitly.

| Case                   | Library | Cartridge | Observed outcome                                 |
| ---------------------- | ------- | --------- | ------------------------------------------------ |
| Invoice reconciliation | 2/2     | 2/2       | Three exact saved paid-total records             |
| Policy extraction      | 2/2     | 2/2       | Four exact saved section and requirement records |
| Settlement evidence    | 2/2     | 2/2       | Correct independently graded settlement file     |

Invoice and policy examples intentionally retain large per-tool previews to exercise compaction.
The settlement example uses a source report exceeding 200,000 characters.
Its decisive section is absent from the initial 3,000-character result envelope.
Each settlement trial observed an explicit oversized-page refusal, then a smaller delivered page.
The grader checked the actual saved reference and amount, not the model's completion claim.
An independently reopened `FileToolResultStore` recovered the complete original report.

Deterministic tests cover sync/async, native/JSON-text, and Python/cartridge execution.
They also check escaped nested JSON, aggregate batch limits, stable admitted previews,
fixed-boundary discovery, quota failures, line endings, and resource/policy round trips.

## Downstream

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

| Gate                                  | Final result                                                              |
| ------------------------------------- | ------------------------------------------------------------------------- |
| Core `make check`                     | 6,700 passed, 2 skipped; text style, lint, formatting, and Pyright passed |
| Downstream with candidate Core        | 959 passed                                                                |
| Downstream with unchanged pinned Core | 955 passed, 4 expected projection-test skips                              |
| Downstream lock, Ruff, full Pyright   | Passed; Pyright checked both Core environments                            |
| Downstream sdist and wheel            | Built successfully                                                        |

The earlier complete Core run found two regressions; both were repaired before the final gate.
An earlier Downstream run had 958 passes and one `os.setsid()` failure because pytest itself
was a process-group leader. A child-process launch passed that focused test and both full suites.
Warnings and failed receipts remain preserved. Local checks used Python 3.13;
other supported Python versions remain CI responsibilities.

## Reproduce

Build the candidate Core wheel and install it in a fresh environment for the public examples:

```bash
uv pip install --python .venv/bin/python "$CORE_WHEEL"

export OPENAI_BASE_URL=http://127.0.0.1:19823/v1
export OPENAI_API_KEY=local-proxy
RUN_ROOT=$(mktemp -d)
uv run --no-sync python -m looplet.examples.default_harness \
  --scripted --cases invoices policy evidence --output "$RUN_ROOT/scripted"
uv run --no-sync python -m looplet.examples.default_harness \
  --cases invoices policy evidence --output "$RUN_ROOT/live"
```

Set `CORE_WHEEL` to the candidate artifact.
Every output directory must be new. Do not synchronize after substituting the wheel.
Downstream integration, randomized delivery, and wire observers are host-owned qualification scripts.

## Evidence And Limits

The private local bundle is `looplet-bounded-results-evidence-2v4xYLv3`.
Qualified example receipts are under `qualified-v2`; final consumer and gate receipts
are under `qualified-v2/final`. `artifact-parity.json`, `consumer-source.json`,
`final-gates.json`, and the two Downstream JUnit files bind the final qualification.
Earlier wheels, failures, retries, stdout/stderr, and wire receipts remain distinct.
Raw source data, provider continuity items, and authenticated headers are not committed.

The observer counts client sends, not hidden proxy retries or provider billing.
Character budgets cover result envelopes, not exact provider tokens or whole requests.
Retrieval access is host authority, not redaction or a sandbox.
File retention has a content quota and one active writer per directory, not fsync/WAL durability.
These historical artifacts predate default disk-backed history and checkpoint catalog references.
They do not qualify the later memory changes. Explicit full exports and selected JSON extraction
can still materialize complete values.
See [bounded model views](provenance.md#bounded-model-views) for the operational contract.

## Follow-Up Recovery Qualification

This separate cohort tests actionable refusals, complete receipt verification, optional request admission, automatic resource binding, and incremental cached retrieval.
It also tests selected-conversation metadata accounting and output headroom during compaction.
The model still chooses its method. No automatic retrieval, tool retry, summary call, or analytical approval bypass was added.

### Artifacts

The final cohort is `qualified-v4` in the private bundle `looplet-result-recovery-evidence-20261027-tdh7bCM7`.
Trials used fresh, non-editable Core and Downstream wheels.
All 304 Core and 85 Downstream non-metadata members matched their source and installed files.
The controller rechecked those source bytes before and after each phase.
The models, proxy, Python version, and OpenAI SDK match the initial qualification settings above.
SDK retries were disabled.

Core wheel SHA-256:

```text
c35ea121b49b57bcc090b22e12bed1ce0dae4aabdd106ac0232e212c799d754c
```

Downstream wheel SHA-256:

```text
30f6418cd47715ee3351ab6bae8ffee08f89aea5fdb772fc3ec118ddb47d24f8
```

Downstream retains its existing public Core dependency pin.
Candidate-wheel checks and unchanged-pin checks use separate environments.
Documentation changes do not replace the tested wheels or rewrite their hashes.

### Live Outcomes

| Final-artifact matrix                                             | Passed | Client HTTP sends |
| ----------------------------------------------------------------- | ------ | ----------------- |
| Public examples: three cases, two models, Python and cartridge    | 12/12  | 111               |
| Randomized blind retrieval: eight scenarios per model             | 16/16  | 94                |
| Downstream: Chat, Responses, SSE, and failed-command recovery     | 8/8    | 66                |
| Downstream: paid totals, grounded join, temporal join, and policy | 8/8    | 55                |
| Total for this cohort only                                        | 44/44  | 326               |

These are independently graded outcomes, not completion claims or an estimated general success rate.
Wire receipts retain actual request bodies and responses privately, without authenticated headers.
Client sends do not measure hidden proxy retries or provider billing.

The public settlement examples delivered a bounded preview without the answer.
Each observed an oversized-page refusal, then a smaller page containing the answer.
The grader checked the saved settlement and reopened retained source evidence.
Invoice and policy examples checked exact saved records while exercising deterministic compaction.

Blind fixtures randomized account IDs, amounts, and confirmation codes outside the model's task.
Scenarios placed evidence early, centrally, or late; nested escaped Unicode; added distractors; omitted the target; exhausted retention quota; or revoked retrieval access.
Each model ran all eight scenarios. Native and JSON-text modes alternated; this was not a full scenario/protocol cross-product.
The grader compared the actual saved JSON with a host-owned oracle.
Missing records required `not_found`; unavailable evidence required `unavailable`, without invented values.
Raw source equality and complete retained values were checked when retention was available.

These blind trials used a 12,000-token configured window, 2,000 output allowance, and 16 tool steps.
The explicit deterministic compact service kept one recent entry and tool result.
The host admission estimator divided serialized UTF-8 request bytes by four, rounded up.
Its 10,000 input ceiling was an approximation, not provider-exact token accounting.
It does not measure provider-held history referenced by Responses continuation IDs.

Six Downstream delivery trials retrieved answers from untrimmed returned stdout exceeding 20,000 characters.
They covered Chat native tools, Responses native tools, and actual Responses SSE on both models.
The SSE trials captured 105 and 120 events.
Two additional trials recovered evidence after an intentional command exit status of seven.
Those checks used full-stream byte counts and SHA-256 hashes, not an assumption that returned stdout was untrimmed.
Every delivery trial matched host admissions to captured physical requests.
The first bounded source envelope omitted the answer; an actual retrieved page supplied it.

Both final grounded joins passed under the unchanged 16-model-call ceiling.
Review approval, exact evidence IDs, source-reading computation, and final submission checks remained enforced.
The previously missing stateless-helper capability now survives backend wrappers.

### Failure History

Earlier receipts remain separate and do not count toward the final cohort.
An initial blind fixture reserved 5,000 output tokens within a 6,000-token window.
Its resulting 1,000-token input allowance correctly refused native requests.
Correcting that fixture exposed a compaction mismatch: selected metadata and output headroom were missing from its trigger estimate.
The repair has sync and async regression tests.
Loading the previous functions in a private counterfactual process makes both tests fail, without changing public source or calling a model.

An earlier artifact's 6,000-token-window stress cohort passed 11/16 trials with 114 client sends.
Five native trials exhausted their tool-step budget despite receiving complete answer-bearing pages.
That artifact predates the final conversation-envelope and callback repairs.
Those results do not qualify the final artifact, and final normal-window success does not establish that the tight-window limitation is fixed.

The separate `qualified-v3` normal-window cohort passed 16/16 blind trials, 12/12 public examples, and eight delivery outcomes.
Its broader Downstream matrix passed 7/8, using 57 sends.
Sol exhausted the shared model-call ceiling after submitting evidence IDs that differed from its reviewed bundle.
The final fresh 8/8 cohort does not relabel that failure or prove that a callback change repaired model reasoning.

The full `qualified-v3` Downstream suite also exposed 31 integration failures.
Core had invoked a legacy host recorder with typed receipt arguments.
The repair uses the explicit `record_tool_receipt(call, result)` capability and leaves legacy recording to the host hook.
All 31 observed failures passed their focused reruns before rebuilding the final wheels.
The existing sync/async wrapper tests now reject accidental calls to the legacy recorder.

One earlier failed-command observer incorrectly required a large in-memory stdout value.
The complete stream was retained, but Downstream had returned a 2,000-character preview.
A separate hash-bound audit verified failed-command recovery with zero new model requests.
The original false observer receipt remains unchanged. Final-artifact trials use the corrected stream-specific check.

A failed replay observer let SDK client cloning bypass its monkeypatch.
It reached the proxy with an invalid model name and produced at least six failed requests.
Its repair used the SDK transport and a process-local socket refusal.
Only the repaired probe is zero-network evidence. Provider serialization warnings remain private, not printed as unfiltered console output.

### Storage And Budget Limits

A separate warm-cache allocation probe measured a bounded read plus search over 2 MiB and 8 MiB sources.
Extra traced Python allocation peaked at 270,962 and 270,922 bytes, respectively.
This is not RSS, a cold extraction measurement, or a whole-agent memory bound.
The first JSON extraction, raw history, provenance, and checkpoint snapshots can still materialize complete values.
Late offsets still require linear scanning. There is no constant-time offset index or cross-process writer protocol.

Delivery refusal is separate from execution failure. It does not advance a page cursor or silently deliver partial page content.
Error, warning, reference, and delivery fields participate in exact projection verification, including after checkpoint restoration.
Conversation round-trips preserve selected warnings and the shared native/text envelope.
An error too large for its control envelope is explicitly marked as shortened; raw evidence remains complete.

Request admission remains optional and synchronous. The host supplies its counter and accounts for output allowance and provider-held state.
Character budgets, approximate compaction, and whole-request token admission remain distinct controls.
Retrieval authority is still host-owned, not a redaction guarantee or sandbox.

### Reproduction And Gates

Use the reproduction commands above with the tested wheel and a new output directory.
Do not synchronize the project environment after substituting the candidate wheel.
Randomized fixture generation, protected oracles, and the HTTP observer remain host-owned qualification code.
Focused public regressions cover delivery hints, escaped envelopes, failed receipts, aggregate reprojection, cached chunk boundaries, resource binding, and request admission.

| Final local gate                      | Result                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Core `make check`                     | 6,796 passed, 2 skipped; text style, Ruff, formatting, and Pyright passed                       |
| Downstream with candidate Core        | 1,011 passed                                                                                    |
| Downstream with unchanged pinned Core | 955 passed, 56 expected feature skips                                                           |
| Downstream Pyright                    | Passed with both Core environments                                                              |
| Downstream Ruff and dependency lock   | Passed; public dependency pin unchanged                                                         |
| Core and Downstream distributions     | Both sdist and wheel builds passed; rebuilt wheels are byte-identical to the live-tested wheels |

Full-suite receipts are saved under `qualified-v4/gates` and require unchanged tracked source during execution.
Earlier full gates remain bound to their original artifacts, not the final wheel.
Raw evidence, failed receipts, controller code, and final gate receipts remain in the private local bundle.