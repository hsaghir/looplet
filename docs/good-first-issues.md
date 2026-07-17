# Good first issues

Curated, well-scoped tasks for first-time contributors. Each is a few hours of work and doesn't require deep familiarity with the loop internals.

## How to claim

1. Browse [open issues labelled `good first issue` and `launch-ready`](https://github.com/hsaghir/looplet/issues?q=is%3Aopen+label%3A%22good+first+issue%22+label%3Alaunch-ready) on GitHub. Each issue body has the **acceptance criteria, file pointers, and scope**. Read it carefully before starting.
2. **Comment on the issue** ("I'd like to work on this - starting with `<module>`") so others don't duplicate work.
3. Open one PR per logical chunk. If the issue has a checklist, tick the box you completed and **don't claim "Closes #N"** unless every box is ticked.
4. Submit. Allow ~1 week per claim before someone else may pick the issue back up.

## Categories

| Theme | Example issues |
| --- | --- |
| **Replay experiments** | [#101](https://github.com/hsaghir/looplet/issues/101): document fixed and fresh variables for tool, hook, prompt, and model changes |
| **Documentation clarity** | Small, accepted follow-ups carrying both `good first issue` and `launch-ready`; claimed work has invitation labels removed |

## What makes a good first PR

- **Small, focused.** One module or one recipe per PR; reviewer can hold it in their head.
- **Tested.** Any behaviour change needs at least one test; mocks for LLM backends so CI doesn't need network.
- **Outcome-grounded.** Prefer an independently observed artifact over a required tool sequence.
- **Narrow.** Search, statistics, optimization, domain policy, and dashboards stay in recipes or downstream packages.
- **`make check` clean.** Lint + format + pyright + pytest all green locally.
- **Complete.** Launch-ready issues are intentionally small. Open one PR that
	satisfies the whole acceptance list rather than a partial umbrella change.

See [CONTRIBUTING.md](https://github.com/hsaghir/looplet/blob/master/CONTRIBUTING.md) for dev setup and the full PR checklist.
