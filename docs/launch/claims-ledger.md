# Launch claims ledger

This file is a release gate. Every public claim needs a source, a qualifier,
and language that must not be substituted for it.

| Approved claim | Evidence | Required qualifier | Do not say |
| --- | --- | --- | --- |
| Looplet supports test-driven harness engineering for a single-loop Python agent. | Public API, cartridges, provenance, eval modules, regression demo. | Positioning/category statement, not an industry-standard category claim. | "The standard agent harness platform." |
| Host code owns iteration and receives explicit `Step` records. | `composable_loop()` generator and loop tests. | The loop still owns model→parse→dispatch mechanics; host owns iteration boundaries. | "Full control over every provider-internal action." |
| Core Looplet has zero third-party runtime dependencies. | `pyproject.toml` has `dependencies = []`; dependency test/benchmark. | OpenAI/Anthropic SDKs are optional extras; development dependencies exist. | "Looplet has no dependencies" without scope. |
| A failed harness run can become a required regression contract. | `examples/regression_demo/`, `tests/test_regression_demo.py`. | The demo is scripted and tests a tool-code change, not general model quality. | "Any agent failure becomes a test automatically." |
| Captured model responses can run through fresh harness code without another model call. | `RecordingLLMBackend`, `replay_loop()`, replay tests, demo. | Tools, hooks, state, permissions, and side effects execute again. | "Deterministic replay" or "free A/B testing" without variable scope. |
| Provenance artifacts are human-readable text/JSON. | Prompt/response files, manifests, trajectories, CLI `show`. | Sensitive values may be present; redaction must be configured at the right boundary. | "Safe to commit every trace." |
| Collectors can grade world state independently of the agent's final claim. | `EvalHook` collectors, demo collector, eval tests. | Independence depends on collector design and host ownership. | "Agents cannot game Looplet evals." |
| Top-level case expected data is withheld from the live agent task. | Eval runner/persistence implementation and integrity tests. | Candidate runtime and files are not protected; promotion needs a host-owned runner and OS/process isolation against arbitrary code. | "All cartridge evals are hidden." |
| Required graders and collector/evaluator errors fail the CLI. | Eval CLI implementation and focused integrity suite. | State the exact command/configuration; library users can choose other handling. | "Looplet guarantees no false greens." |
| Cartridges are reviewable harness directories. | Cartridge format, describe/diff/hash commands, round-trip tests. | Round-trip is defined for supported serializable fields; Python escape hatches remain. | "Lossless serialization of arbitrary Python." |
| A compact Looplet coder matched Copilot correctness in recorded small same-model suites. | `benchmarks/coder_vs_agents/*` reports. | Historical, small samples; serving stacks differ; open-ended judging is noisy. | "Looplet beats Copilot" or universal speed/cost claims. |
| Looplet is appropriate for one model calling tools in a loop. | Architecture and API. | Graph runtimes are a better fit for genuine branching workflows. | "Most/all agents are just loops" or unsupported percentages. |

## Claims requiring fresh validation before each launch

- current PyPI version and Python classifiers;
- CI status and exact test result;
- hosted documentation URLs after deployment;
- benchmark environment/package versions;
- CLI spelling and output shown in public copy;
- the generated proof evidence tree;
- repository labels and issue links;
- any third-party user or adoption statement.

## Evidence quality levels

1. **Strong:** network-free executable proof + protected regression test.
2. **Good:** focused unit/integration suite over the exact integrity boundary.
3. **Conditional:** reproducible external benchmark with raw task-level output
   and explicit caveats.
4. **Weak:** screenshot, anecdote, generated demo, or maintainer assertion.

Lead with levels 1–2. Use level 3 only for a narrow secondary claim. Never turn
level 4 into a headline.

## Language rules

- Say **captured-response replay**, not deterministic replay.
- Say **outcome-grounded**, not objective, unless the outcome truly has an
  objective verifier.
- Say **host-owned holdout with an explicit isolation boundary**, not hidden eval, when security matters.
- Say **core runtime**, not whole development environment, for dependency
  claims.
- Say **can fail closed under the CLI contract**, not guarantees safety.
- Distinguish a scripted harness regression from sampled model evaluation.
