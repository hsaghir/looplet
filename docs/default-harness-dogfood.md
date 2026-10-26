# Standard Harness Dogfood

This report covers unreleased source changes tested on October 25, 2026.
It is a focused integration check, not an accuracy benchmark or protected promotion oracle.

## Inputs

| Input | Revision or setting |
| --- | --- |
| Core base | `a1fae70cbccfe3546c6f4f92cc449d9f8f0cacc1` |
| Analytics base | `c18ebbecc139c591c3273c709b144a1dc9892bcd` |
| Models | `gpt-6.1-sol`, `gpt-6-luna` |
| Proxy | `http://127.0.0.1:19823/v1` |
| Local Python | `3.13.7` |
| Fresh-agent OpenAI SDK | `3.6.0` |
| Analytics OpenAI SDK | `3.6.0` |

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

| Agent | Entry path | Passed | Recorded calls per trial | Compactions per trial |
| --- | --- | --- | --- | --- |
| Invoice reconciliation | Library | 2/2 | 7 | 2 |
| Invoice reconciliation | Cartridge | 2/2 | 7 | 2 |
| Policy controls | Library | 2/2 | 9 | 3 |
| Policy controls | Cartridge | 2/2 | 9 | 3 |

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

## Analytics

The downstream script creates source files and computes expected values before running the agents.
Expected answers stay outside the agent task.
Sources include cancellations, negative reversals, date boundaries, changing customer segments, and a competing premium policy.
The self-check validates all 42 native tool schemas without model requests.

Each row covers both models:

| Case | Profile | Transport | Expected | Passed |
| --- | --- | --- | --- | --- |
| January paid net amount | `direct` | Chat | 405 | 2/2 |
| Current enterprise customer join | `grounded` | Responses | 130 | 2/2 |
| Effective-date enterprise join | `direct` | Responses | 150 | 2/2 |
| Standard overdue policy | `standard` | Chat | 30 | 2/2 |

All eight final trials completed and passed exact decimal grading.
Each trial retained its result, configuration, source hashes, and provenance trace.
The host set 16 steps, 16 model calls, a 180-second case timeout, and a 90-second backend timeout.
Analytics disabled SDK retries for these trials.

`standard` completion does not imply grounded evidence verification.
The companion Analytics change preserves that distinction.
It also isolates tool-internal Responses reviews from pending parent tool calls.
The Analytics dependency pin remains unchanged; qualification explicitly substituted the candidate Core wheel.

## Friction Found

Failed runs were retained instead of replacing their reports with passing outcomes.

| Observed failure | Repair or decision |
| --- | --- |
| Implicit model admission broke custom state and existing checkpoint contracts. | Keep `max_model_calls` explicit; use a dispatch hook for the automatic tool guard. |
| Structured payloads did not affect context-pressure estimates. | Count structured arguments, results, and errors as well as text. |
| Strict export lost stock tool bodies or example module globals. | Use importable stock functions and canonical domain module references. |
| Concurrent free calls overshot a post-dispatch limit. | Reserve before dispatch; test actual side-effect counts in both engines. |
| Editable Core imports failed inside isolated Analytics workers. | Qualify installed wheels; keep the sandbox restrictions unchanged. |
| Optional dictionary type descriptors produced invalid native schemas. | Normalize type aliases and optional markers without changing authored mappings. |
| Responses reviews advanced a parent with unanswered native calls. | Send isolated stateless reviews and preserve the parent checkpoint and queued outputs. |
| Accepted non-grounded submissions were labeled incomplete. | Separate accepted completion from evidence verification in Analytics. |
| Copying execution config broke caller-owned live configuration. | Preserve config identity and restore only the automatically installed service. |
| One policy trial wrote correct files but repeated work until its step limit. | Return the saved-record inventory from each domain write tool. |

The policy failure remains in `fresh-live-final`; its file grade passed but its run did not complete.
After the example-only inventory repair, the unchanged eight-case matrix passed in `fresh-live-qualified`.
This does not establish general fact retention across compaction.

## Release Checks

| Local gate | Result |
| --- | --- |
| Complete Core suite | 6,531 passed, 2 skipped |
| Complete Analytics suite with candidate Core wheel | 920 passed |
| Complete Analytics suite with unchanged locked Core pin | 920 passed |
| Core Ruff, formatting, Pyright, and text style | Passed |
| Analytics Ruff, full Pyright with both Core revisions, and lock check | Passed |
| Core and Analytics sdist and wheel builds | Passed |

An earlier full Core run found one unsynchronized site changelog, alongside 6,530 passing tests.
The site copy was synchronized, all 16 release-metadata checks passed, and the complete Core suite then passed.
That failed run and its coverage report remain separate from the final passing receipt.

## Reproduce

Build Core from this branch.
From an Analytics checkout, install its locked development environment.
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
uv run --no-sync python "$CORE_CHECKOUT/scripts/dogfood_default_harness_analytics.py" \
  --self-check --output "$RUN_ROOT/analytics-self-check"
uv run --no-sync python "$CORE_CHECKOUT/scripts/dogfood_default_harness_analytics.py" \
  --output "$RUN_ROOT/analytics-live"
```

Set `CORE_WHEEL` and `CORE_CHECKOUT` to the candidate artifact and checkout paths.
Do not synchronize this environment after substituting the wheel.
Synchronization restores Analytics' unchanged Git dependency pin.
Every output directory must be new.

## Evidence And Limits

The local evidence bundle is `looplet-default-harness-evidence-20261025.Me6MzL`.
Final matrix reports are `fresh-live-qualified/summary.json` and `analytics-live-v3/summary.json`.
The final Core regression receipt is `core-full-final.xml`.
Analytics receipts are `analytics-full-qualified.xml` and `analytics-full-pinned.xml`.
The source, artifact, and installed-file check is `final-artifact-parity.json`.
Shipped-example logs are `hello-world-qualified-sol.log` and `hello-world-qualified-luna.log`.
Earlier failures, traces, logs, and intermediate artifacts remain in the same bundle.
Raw traces are not committed to the package.

Provider Pydantic serialization warnings appeared in both transports.
Some Analytics calls required native-protocol recovery before accepted completion.
Those logs remain part of the result; this was not a clean-first-attempt claim.
No outer-loop text fallback was enabled.

Fresh-agent call counts record backend invocations; SDK retries were not explicitly disabled there.
Analytics reports host model-call counters.
Neither count independently proves physical HTTP traffic or billing.

Context estimates are approximate and compaction is lossy.
The automatic tool budget is per invocation, not a durable resume quota or execution deadline.
The standard harness adds no hidden filesystem memory, summary-model calls, tracing, approval prompts, or OS sandbox.
Hosts still own credentials, deadlines, security boundaries, and protected release evaluations.
Local checks used Python 3.13; other supported Python versions remain CI responsibilities.