# Standard Harness Dogfood

The original report covers unreleased source changes tested on October 25, 2026.
It is a focused integration check, not an accuracy benchmark or protected promotion oracle.
The final section records a fresh check of the memory and default-recovery changes on October 27, 2026.

## Inputs

| Input                  | Revision or setting                        |
| ---------------------- | ------------------------------------------ |
| Core base              | `a1fae70cbccfe3546c6f4f92cc449d9f8f0cacc1` |
| Downstream base        | `c18ebbecc139c591c3273c709b144a1dc9892bcd` |
| Models                 | `gpt-6.1-sol`, `gpt-6-luna`                |
| Proxy                  | `http://127.0.0.1:19823/v1`                |
| Local Python           | `3.13.7`                                   |
| Fresh-agent OpenAI SDK | `3.6.0`                                    |
| Downstream OpenAI SDK  | `3.6.0`                                    |

The final trials used an installed Core wheel, not an editable Core import.
All 301 non-metadata wheel members matched the source and installed files.
These include 295 Core package files and six fixture files.
The tested wheel SHA-256 is:

```text
6a9007dc6a2fabb2bc6ffa2c63a4dd65381823affe8690bb95f2bb535b095810
```

A separate environment installed only Core, without provider SDKs or other runtime dependencies.
Its package smoke test passed, including 20 scripted executions and four early-close checks.
The four scripted fresh-agent trials also passed in that environment.

## Fresh Agents

`looplet.examples.default_harness` builds invoice and policy agents without hand-assembled harness hooks.
Each trial creates fresh state and outputs.
The cartridge path strictly exports and reloads the same Python agent.
Large source payloads exercise automatic context compaction.

Each row covers both models:

| Agent                  | Entry path | Passed | Recorded calls per trial | Compactions per trial |
| ---------------------- | ---------- | ------ | ------------------------ | --------------------- |
| Invoice reconciliation | Library    | 2/2    | 7                        | 2                     |
| Invoice reconciliation | Cartridge  | 2/2    | 7                        | 2                     |
| Policy controls        | Library    | 2/2    | 9                        | 3                     |
| Policy controls        | Cartridge  | 2/2    | 9                        | 3                     |

All eight final trials reached accepted completion with correct files and no tool errors.
There were 64 recorded backend invocations and 20 observed compactions.
Invoice grading checks three exact JSON records with paid totals of 1136, 1936, and 2736.
Policy grading checks four exact section and requirement records.
Neither grader accepts a model's completion claim as proof of file contents.

These tools intentionally expose computed invoice totals and narrow file-writing operations.
They test harness integration, extraction, context pressure, and delivery, not general numerical reasoning or shell security.

The shipped `hello_world` example also completed with both models on the qualified wheel.
Both greeting and completion evaluators scored 1.00.
Sol recovered from one native-protocol parse error; Luna completed without a parse-error step.
Both runs used an explicit eight-call admission limit.

## Downstream

The downstream script creates source files and computes expected values before running the agents.
Expected answers stay outside the agent task.
Sources include cancellations, negative reversals, date boundaries, changing customer segments, and a competing premium policy.
The self-check validates all 42 native tool schemas without model requests.

Each row covers both models:

| Case                             | Profile    | Transport | Expected | Passed |
| -------------------------------- | ---------- | --------- | -------- | ------ |
| January paid net amount          | `direct`   | Chat      | 405      | 2/2    |
| Current enterprise customer join | `grounded` | Responses | 130      | 2/2    |
| Effective-date enterprise join   | `direct`   | Responses | 150      | 2/2    |
| Standard overdue policy          | `standard` | Chat      | 30       | 2/2    |

All eight final trials completed and passed exact decimal grading.
Each trial retained its result, configuration, source hashes, and provenance trace.
The host set 16 steps, 16 model calls, a 180-second case timeout, and a 90-second backend timeout.
Downstream disabled SDK retries for these trials.

`standard` completion does not imply grounded evidence verification.
The companion Downstream change preserves that distinction.
It also isolates tool-internal Responses reviews from pending parent tool calls.
The Downstream dependency pin remains unchanged; qualification explicitly substituted the candidate Core wheel.

## Friction Found

Failed runs were retained instead of replacing their reports with passing outcomes.

| Observed failure                                                               | Repair or decision                                                                     |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Implicit model admission broke custom state and existing checkpoint contracts. | Keep `max_model_calls` explicit; use a dispatch hook for the automatic tool guard.     |
| Structured payloads did not affect context-pressure estimates.                 | Count structured arguments, results, and errors as well as text.                       |
| Strict export lost stock tool bodies or example module globals.                | Use importable stock functions and canonical domain module references.                 |
| Concurrent free calls overshot a post-dispatch limit.                          | Reserve before dispatch; test actual side-effect counts in both engines.               |
| Editable Core imports failed inside isolated Downstream workers.               | Qualify installed wheels; keep the sandbox restrictions unchanged.                     |
| Optional dictionary type descriptors produced invalid native schemas.          | Normalize type aliases and optional markers without changing authored mappings.        |
| Responses reviews advanced a parent with unanswered native calls.              | Send isolated stateless reviews and preserve the parent checkpoint and queued outputs. |
| Accepted non-grounded submissions were labeled incomplete.                     | Separate accepted completion from evidence verification in Downstream.                 |
| Copying execution config broke caller-owned live configuration.                | Preserve config identity and restore only the automatically installed service.         |
| One policy trial wrote correct files but repeated work until its step limit.   | Return the saved-record inventory from each domain write tool.                         |

The policy failure remains in `fresh-live-final`; its file grade passed but its run did not complete.
After the example-only inventory repair, the unchanged eight-case matrix passed in `fresh-live-qualified`.
This does not establish general fact retention across compaction.

## Release Checks

| Local gate                                                             | Result                  |
| ---------------------------------------------------------------------- | ----------------------- |
| Complete Core suite                                                    | 6,531 passed, 2 skipped |
| Complete Downstream suite with candidate Core wheel                    | 920 passed              |
| Complete Downstream suite with unchanged locked Core pin               | 920 passed              |
| Core Ruff, formatting, Pyright, and text style                         | Passed                  |
| Downstream Ruff, full Pyright with both Core revisions, and lock check | Passed                  |
| Core and Downstream sdist and wheel builds                             | Passed                  |

An earlier full Core run found one unsynchronized site changelog, alongside 6,530 passing tests.
The site copy was synchronized, all 16 release-metadata checks passed, and the complete Core suite then passed.
That failed run and its coverage report remain separate from the final passing receipt.

## Reproduce

Build Core from this branch.
The public examples can be run from a fresh environment with the candidate Core wheel.
Then substitute the exact candidate wheel:

```bash
uv sync --locked --group dev
uv --no-config pip install --python .venv/bin/python --no-deps --reinstall "$CORE_WHEEL"

export OPENAI_BASE_URL=http://127.0.0.1:19823/v1
export OPENAI_API_KEY=local-proxy
RUN_ROOT=$(mktemp -d)

uv run --no-sync python -m looplet.examples.default_harness \
  --scripted --output "$RUN_ROOT/fresh-scripted"
uv run --no-sync python -m looplet.examples.default_harness \
  --output "$RUN_ROOT/fresh-live"
```

Set `CORE_WHEEL` to the candidate artifact path.
Do not synchronize this environment after substituting the wheel.
Private downstream qualification uses host-owned scripts and its unchanged dependency pin.
Every output directory must be new.

## Evidence And Limits

The local evidence bundle is `looplet-default-harness-evidence-20261025.Me6MzL`.
The public-example matrix report is `fresh-live-qualified/summary.json`.
The downstream matrix remains in the host-owned private evidence bundle.
The final Core regression receipt is `core-full-final.xml`.
Downstream regression receipts retain their original private filenames in that bundle.
The source, artifact, and installed-file check is `final-artifact-parity.json`.
Shipped-example logs are `hello-world-qualified-sol.log` and `hello-world-qualified-luna.log`.
Earlier failures, traces, logs, and intermediate artifacts remain in the same bundle.
Raw traces are not committed to the package.

Provider Pydantic serialization warnings appeared in both transports.
Some Downstream calls required native-protocol recovery before accepted completion.
Those logs remain part of the result; this was not a clean-first-attempt claim.
No outer-loop text fallback was enabled.

Fresh-agent call counts record backend invocations; SDK retries were not explicitly disabled there.
Downstream reports host model-call counters.
Neither count independently proves physical HTTP traffic or billing.

Context estimates are approximate and compaction is lossy.
The automatic tool budget is per invocation, not a durable resume quota or execution deadline.
The standard harness adds no hidden filesystem memory, summary-model calls, tracing, approval prompts, or OS sandbox.
Hosts still own credentials, deadlines, security boundaries, and protected release evaluations.
Local checks used Python 3.13; other supported Python versions remain CI responsibilities.

## October 27 Memory And Recovery Qualification

This section qualifies the later disk-backed retention and default-recovery changes.
The earlier results, artifacts, and failures remain historical evidence.
The earlier Release Checks table does not qualify these changes.
Current full regression results are recorded in the pull request.

The fresh trials used `gpt-6.1-sol` and `gpt-6.1-terra` through the same local proxy.
They used Python 3.13.7 and OpenAI SDK 3.6.0.
The Core wheel and installed package matched all 305 non-metadata members against source.
The qualified Core wheel SHA-256 is:

```text
e413f2da29df586714836433f24a40d560b61498482b2d4d8b83e28963c8e850
```

| Cohort | Coverage | Passed | Captured client sends |
| ------ | -------- | ------ | --------------------- |
| Public examples | Invoice, policy, and evidence agents; both models; Python and cartridge entry | 12/12 | 106 |
| Default persistence | Evidence recovery; both models; Python and cartridge entry; sync and async drivers | 8/8 | 52 |
| Downstream workflows | Paid amount, current join, effective-date join, and overdue policy; both models | 8/8 | 56 |
| Total | Independent outcome checks | 28/28 | 214 |

The default-persistence trials supplied no explicit persistence resources.
They set only the run directory to choose an artifact location.
The default harness composed disk-backed history, full result storage, provenance, checkpoints, and branch records.
The checks verified complete source recovery, separate-store checkpoint restore, and independently retained final results.
Captured requests omitted the oversized source answer until the agent retrieved the relevant page.
Native and text protocols alternate across the eight trials.
Not every combination of model, entry path, driver, and protocol was run.

The public examples produced independently graded files after accepted completion.
The downstream workflows passed the same exact decimal checks: 405, 130, 150, and 30.
Their profiles, review rules, 16-call ceiling, and unchanged Core dependency pin remained intact.
The host substituted the candidate wheel only in the isolated qualification environment.

All three cohorts disabled SDK retries and retained request journals.
The 214 captured sends matched their recorded model-call totals.
These are client HTTP requests, not independent proof of provider work or billing.
One readiness inference is separate from these totals.

Review found two defects before the final wheel was built.
Same-store checkpoint restore now rejects missing or corrupt source evidence.
A persistence setup failure now restores the caller's automatic compaction-service setting in both drivers.
Focused regression checks passed after each repair.

Two observer defects occurred before this final qualification.
The first observer expected a different SDK HTTP transport module.
The first default-persistence observer imported the async backend from the wrong module.
Those failed cohorts sent no model requests and remain separate from the passing results.
A fake-server observer check and two scripted default-persistence trials passed before the corrected live cohort.
Provider warnings and complete response payloads remain in private diagnostic files.

The local evidence bundle is `looplet-merge-qualification-20261027.Ur9z7e`.
Its cohort summaries are `live-public/summary.json`, `live-defaults-v2/summary.json`, and `live-companion/summary.json`.
The request journals retain all 214 sends without authenticated headers.
The source and installed-file bindings are recorded in `artifacts.json`.

Disk-backed retention bounds automatically accumulated payloads, not total process RSS.
One large active value, caller-held records, explicit exports, and tool implementations can still use substantial RAM.
Temporary spools are not recovery journals or filesystem durability guarantees.
Checkpoints do not roll back external effects.
Hosts still own sandboxing, deadlines, credentials, storage limits, and cleanup.