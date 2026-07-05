# looplet coder vs GitHub Copilot CLI — head-to-head benchmark

**TL;DR** — With the **same model** driving both (`claude-sonnet-4.6`), the looplet
`coder.cartridge` and the GitHub Copilot CLI **both solved 9/9 tasks** (5 coding +
4 non-coding). They are at **parity on correctness**. looplet was **~21 % faster
end-to-end** (164.8 s vs 208.3 s) and ran a **much leaner context** (≈0.56 M est.
tokens / 43 LLM calls vs Copilot's 1.49 M input tokens self-reported). Copilot's
extra weight buys a **broader capability envelope** (GitHub/web/browser MCP tools,
subagents, session resume) that these self-contained tasks never exercised.

> This is a **harness comparison, not a model comparison.** Both agents use the
> identical underlying model, so differences come from the *scaffolding*: system
> prompt, tool set, loop control, and startup cost — not raw model intelligence.

---

## Method (fairness controls)

| Control | Setting |
|---|---|
| Model | `claude-sonnet-4.6` for **both** (`--model` for Copilot; `OPENAI_MODEL` for looplet) |
| looplet LLM path | Copilot proxy `http://127.0.0.1:19823/v1` (so both hit the same model family) |
| Isolation | Every (task × tool) runs in a **fresh empty workspace**; seed files written identically |
| Permissions | Copilot `--allow-all` (non-interactive); looplet cartridge default `allow` |
| Budget | looplet `max_steps=20`; Copilot manages its own |
| Verification | A **deterministic verifier** per task (run the code / check the answer file). No human grading. |
| Invocation | Each tool used **as shipped**: `copilot -p …` CLI; looplet drives `composable_loop` exactly as `looplet run-workspace` does. |

Harness lives in this directory — `bench.py` (orchestrator),
`tasks.py` (tasks + verifiers), `looplet_runner.py` (looplet driver), `report.py`
(regenerates `TABLES.md`). Re-run everything: `python bench.py`.

---

## Per-task results

| Task | Kind | looplet | wall | steps¹ | copilot | wall | credits | in/out tok² |
|---|---|:--:|--:|--:|:--:|--:|--:|--:|
| c1 fizzbuzz | coding | ✅ | 15.5s | 4 | ✅ | 15.7s | 9.5 | 79.5k / 275 |
| c2 roman numerals | coding | ✅ | 25.2s | 5 | ✅ | 17.7s | 9.6 | 79.5k / 357 |
| c3 fix bugs in stats.py | coding | ✅ | 21.0s | 5 | ✅ | 26.1s | 15.4 | 240.3k / 864 |
| c4 write pytest suite | coding | ✅ | 28.6s | 9 | ✅ | 46.4s | 23.4 | 448.4k / 1600 |
| c5 word-frequency CLI | coding | ✅ | 20.1s | 4 | ✅ | 24.5s | 12.6 | 160.0k / 622 |
| n1 revenue from JSON | other | ✅ | 17.0s | 6 | ✅ | 19.3s | 11.1 | 119.8k / 419 |
| n2 aggregate a CSV | other | ✅ | 11.1s | 3 | ✅ | 18.5s | 10.8 | 119.2k / 329 |
| n3 reasoning word problem | other | ✅ | 12.6s | 3 | ✅ | 21.3s | 11.3 | 119.8k / 603 |
| n4 extract/dedupe/sort emails | other | ✅ | 13.9s | 4 | ✅ | 19.0s | 11.0 | 119.3k / 426 |

¹ **`steps` are not directly comparable across tools.** looplet `steps` = loop
iterations (includes reads + the `done` tool). Copilot `steps` = the number of
action bullets it surfaced. Treat them as *shape*, not a common unit.

² Copilot self-reports real tokens; its large **input** counts include the
re-sent/cached system prompt + MCP tool schemas each turn.

---

## Aggregate

| Metric | looplet | copilot |
|---|--:|--:|
| **Tasks passed** | **9/9** | **9/9** |
| — coding | 5/5 | 5/5 |
| — non-coding | 4/4 | 4/4 |
| Total wall time | **164.8s** | 208.3s |
| Avg wall / task | **18.3s** | 23.1s |
| Won on wall time | **7 / 9** | 2 / 9 |
| AI credits (Copilot self-report) | — | 114.7 total (~12.7 avg) |
| Input tokens (Copilot) | — | 1,485,800 |
| Output tokens (Copilot) | — | 5,495 |
| Est. total tokens (looplet, chars/4) | ~561,868 | — |
| LLM calls (looplet) | 43 (~4.8/task) | — |

---

## Analysis by dimension

### 1. Correctness — **tie (9/9 each)**

On well-scoped, verifiable tasks with a strong shared model, both harnesses are
reliable. Neither made a logic error that survived to the verifier. This is the
expected result: the *model* does the reasoning, and both scaffolds are competent
enough to get out of its way.

### 2. Speed — **looplet faster (won 7/9)**

looplet averaged 18.3 s vs Copilot's 23.1 s. The main driver is **startup
overhead**: Copilot boots MCP servers (`github-mcp-server`, `devtools`,
`supervisor`, `computer-use`) on every invocation (~5–8 s observed) and carries a
larger system prompt. looplet cold-imports fast and dispatches immediately.
Copilot's two wins (c2, c3) were tasks where looplet spent an extra read/verify
step.

### 3. Cost & context weight — **looplet dramatically leaner**

Copilot pushed **1.49 M input tokens** across 9 short tasks; the single
`c4 write-tests` task alone cost **448 k input tokens / 23.4 credits**. That's the
price of a batteries-included harness: a big system prompt plus full MCP tool
schemas resent (and cached) every turn. looplet's coder carries a compact prompt
and 4–6 tools, estimated at ~0.56 M **total** tokens over the same 9 tasks. Even
allowing for the estimate being rough, the direction is unambiguous: **looplet
does the same work with far less context per call.**

### 4. Code quality — **similar logic, different hygiene defaults**

Same model → near-identical algorithms. The difference is *style nudging*:

- **looplet** emitted type hints, module/function docstrings, `encoding=` on
  `open()`, and shebangs — its cartridge system prompt asks for this and a
  `LinterHook` (ruff) runs after edits.
- **Copilot** produced leaner, minimal code (no type hints/docstrings unless
  asked) — equally correct, less ceremony.

Both `c4` suites had 5 test functions incl. the divide-by-zero `pytest.raises`.

### 5. Capability envelope — **Copilot broader out of the box**

Copilot ships with GitHub MCP, a browser/devtools server, web fetch, subagents,
session resume (`--resume`), autopilot, and image/attachment input. None were
needed here, but on tasks that touch GitHub, the web, or a live browser, Copilot
would pull ahead with **zero extra wiring**. looplet's coder is deliberately
minimal (bash/read/write/edit/grep/glob); you'd add tools yourself.

### 6. Control & observability — **looplet's whole point**

looplet exposes every `Step` (prompt, tool call, result, usage) and every decision
as a `Protocol` hook you own — the benchmark's token instrumentation was a 10-line
backend wrapper, and swapping model/permissions/compaction is a config edit.
Copilot is a polished, comparatively closed product: you configure it, you don't
reach between step 6 and step 8.

---

## Important caveats

1. **Same model** — this measures scaffolding, not intelligence. A different model
   would move both lines together.
2. **Task profile is favourable to both** — short, single-file, deterministically
   checkable. They do **not** stress long-horizon, multi-file, or ambiguous
   agentic work, where harness quality (planning, context management, recovery)
   matters far more and would likely separate the two.
3. **Token figures aren't perfectly commensurable** — Copilot reports real tokens
   (incl. cached); looplet's is a chars/4 estimate because the proxy strips the
   `usage` block. Use them for *order-of-magnitude*, not decimals.
4. **`steps` counted differently per tool** (see note ¹).
5. **Different intents** — Copilot is an end-user assistant; looplet is a library
   whose `coder.cartridge` is a starting point you edit. "Winning" means different
   things for each.

---

## Bottom line

For everyday, well-specified coding and data tasks, your looplet coder example is
**already competitive with a shipping commercial agent** on the axis that matters
most — it got the right answer every time — while being **faster to start and much
lighter on context/cost**. Copilot's advantage isn't accuracy here; it's the
**breadth of built-in tools and integrations** and product polish. Where looplet
wins structurally is **ownership**: same results, a fraction of the context, and
every step open to inspection and modification.
