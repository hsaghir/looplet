# Good first issues

Curated, well-scoped tasks for first-time contributors. Each is a few hours of work and doesn't require deep familiarity with the loop internals.

## How to claim

1. Browse [open issues labelled `good first issue`](https://github.com/hsaghir/looplet/issues?q=is%3Aopen+label%3A%22good+first+issue%22) on GitHub. Each issue body has the **acceptance criteria, file pointers, and scope** — read it carefully before starting.
2. **Comment on the issue** ("I'd like to work on this — starting with `<module>`") so others don't duplicate work.
3. Open one PR per logical chunk. If the issue has a checklist, tick the box you completed and **don't claim "Closes #N"** unless every box is ticked.
4. Submit. Allow ~1 week per claim before someone else may pick the issue back up.

## Categories

| Theme | Example issues |
|---|---|
| **New backends** | Gemini, Bedrock — implement the `LLMBackend` protocol over a new SDK |
| **Recipes** | Local LLM via `llama-cpp-python`, OTel, MCP — runnable end-to-end docs |
| **Eval recipes** | Port common evals into `looplet.evals.recipes` as importable functions |
| **Examples** | New workspaces under `examples/` (e.g. research agent demonstrating `run_sub_loop`) |
| **Documentation** | Fill in missing docstrings on public symbols (one PR per module) |
| **Developer ergonomics** | Makefile targets, error-message improvements |
| **Cookbook & benchmarks** | `looplet new` recipes for new tool surfaces; factory output quality benchmarks |

## What makes a good first PR

- **Small, focused.** One module or one recipe per PR; reviewer can hold it in their head.
- **Tested.** Any behaviour change needs at least one test; mocks for LLM backends so CI doesn't need network.
- **`make check` clean.** Lint + format + pyright + pytest all green locally.
- **Honest about scope.** If you only finished part of an umbrella issue, say "Towards #N" not "Closes #N".

See [CONTRIBUTING.md](https://github.com/hsaghir/looplet/blob/master/CONTRIBUTING.md) for dev setup and the full PR checklist.
