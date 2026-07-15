# Good first issues

Curated, well-scoped tasks for first-time contributors. Each is a few hours of work and doesn't require deep familiarity with the loop internals.

## How to claim

1. Browse [open issues labelled `good first issue`](https://github.com/hsaghir/looplet/issues?q=is%3Aopen+label%3A%22good+first+issue%22) on GitHub. Each issue body has the **acceptance criteria, file pointers, and scope** — read it carefully before starting.
2. **Comment on the issue** ("I'd like to work on this — starting with `<module>`") so others don't duplicate work.
3. Open one PR per logical chunk. If the issue has a checklist, tick the box you completed and **don't claim "Closes #N"** unless every box is ticked.
4. Submit. Allow ~1 week per claim before someone else may pick the issue back up.

## Categories

| Theme | Example issues |
| --- | --- |
| **Regression contracts** | Turn one reproduced failure into a case, independent collector, required grader, and network-free test |
| **Outcome recipes** | File artifacts, host-owned test suites, schema checks, API probes, and database observations |
| **Evidence portability** | Small exporters or schema examples for OTel and hosted observability systems, outside core runtime |
| **Provider recipes** | Gemini, Bedrock, or local-model adapters implemented against the small backend protocol |
| **Cartridge examples** | Focused harnesses with reviewable tools and colocated self-tests under `examples/` |
| **Documentation** | Fill in missing docstrings on public symbols (one PR per module) |
| **Integrity ergonomics** | Clearer fail-closed errors for required graders, malformed evidence, or unsafe case paths |
| **Benchmarks** | Reproduce an existing snapshot with environment metadata and explicit caveats; do not add leaderboard claims |

## What makes a good first PR

- **Small, focused.** One module or one recipe per PR; reviewer can hold it in their head.
- **Tested.** Any behaviour change needs at least one test; mocks for LLM backends so CI doesn't need network.
- **Outcome-grounded.** Prefer an independently observed artifact over a required tool sequence.
- **Narrow.** Search, statistics, optimization, domain policy, and dashboards stay in recipes or downstream packages.
- **`make check` clean.** Lint + format + pyright + pytest all green locally.
- **Honest about scope.** If you only finished part of an umbrella issue, say "Towards #N" not "Closes #N".

See [CONTRIBUTING.md](https://github.com/hsaghir/looplet/blob/master/CONTRIBUTING.md) for dev setup and the full PR checklist.
