# Hard / long-horizon benchmark - looplet coder vs GitHub Copilot CLI

Companion to [REPORT.md](REPORT.md) (the short-task suite). Both paths requested
`claude-sonnet-4.6` through separate serving connections, used a fresh isolated
workspace per run, and a **hidden test suite** (written only at verify time,
after the agent finishes) grading each result. looplet ran with `max_steps=40`.

**TL;DR** - On four genuinely hard, multi-file, long-horizon tasks, **both agents
passed 4/4**. Wall-time was **near parity** (looplet 608 s vs Copilot 631 s, ~4 %) -
the 21 % gap from the easy suite mostly closed because model "thinking" now
dominates over Copilot's fixed startup cost. There was **no consistent efficiency
winner**: looplet was much faster on the KV-store and regex engine; Copilot was
faster on the expression evaluator. Both produced correct, idiomatic designs;
looplet's observed outputs were more structured/verbose, while Copilot's were
more compact.

---

## The tasks (why they're hard)

| Task | What makes it long-horizon |
| --- | --- |
| **h1 expression evaluator** | Multi-module package (tokenizer + recursive-descent parser + evaluator); operator precedence & right-associative `**`; unary minus in exponents; `eval`/`ast` banned |
| **h2 KV-store** | Stateful system: TTL expiry against an injectable clock, LRU eviction at capacity, JSON persistence round-trip, and a CLI - many interacting invariants |
| **h3 refactor + extend** | Read a messy monolith, split into layers **without breaking a provided test suite**, then add a transaction-history + `undo_last` feature. The classic real-world job. |
| **h4 regex engine** | Backtracking matcher for `. * + ? ^ $ [..] [a-z] [^..]`, `re` module banned - a hard algorithm with 26 hidden cases |

---

## Per-task results

| Task | looplet | wall | steps¹ | LLM calls | copilot | wall | actions¹ | credits | in/out tok² |
| --- | :---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| h1 expr eval | ✅ | 274.2s | 24 | 25 | ✅ | 178.6s | 7 | 53.5 | 410k / 14.9k |
| h2 KV-store | ✅ | **57.2s** | 7 | 7 | ✅ | 136.7s | 3 | 38.8 | 188k / 10.6k |
| h3 refactor | ✅ | 153.0s | 26 | 28 | ✅ | 150.3s | 12 | 44.7 | 572k / 11.0k |
| h4 regex engine | ✅ | **124.0s** | 16 | 19 | ✅ | 165.4s | 3 | 39.3 | 198k / 14.7k |

¹ Not a common unit: looplet `steps` = loop iterations (write → run tests → fix →
re-run, incl. reads & `done`); Copilot `actions` = surfaced action bullets.
² Copilot self-reported; input includes the re-sent/cached system prompt + MCP schemas.

---

## Aggregate

| Metric | looplet | copilot |
| --- | ---: | ---: |
| **Tasks passed** | **4/4** | **4/4** |
| Total wall time | **608.4s** | 630.9s |
| Avg wall / task | **152.1s** | 157.7s |
| Won on wall time | 2 / 4 (h2, h4) | 2 / 4 (h1, ~h3 tie) |
| AI credits (Copilot) | - | 176.3 (~44 avg) |
| Input tokens (Copilot) | - | 1,367,200 |
| Output tokens (Copilot) | - | 51,200 |
| Est. total tokens (looplet, chars/4) | ~1,150,000 | - |
| LLM calls (looplet) | 79 (~20/task) | - |

---

## Analysis

### Correctness - tie again (4/4 each)

Both cleared every hidden-test gate: right-associative `**` and `2**-1 == 0.5` in
the parser; LRU eviction + TTL expiry against an injected clock; a refactor that
kept the untouched provided suite green **and** added working `history`/`undo_last`;
and all 26 regex cases. Complexity did **not** break either harness at this scale.

### Speed - near parity, task-dependent

The easy-suite pattern (looplet always faster due to lower startup cost) **did not
hold**. On long tasks, minutes of model reasoning dwarf Copilot's ~5–8 s MCP boot.

- looplet crushed **h2** (57 s vs 137 s) and won **h4** (124 s vs 165 s).
- Copilot won **h1** (179 s vs 274 s) - looplet took 24 iterations of
  write/test/fix to converge the multi-module parser; Copilot got there in fewer,
  larger moves.
- **h3** was a dead heat (~152 s).

### Iteration style - looplet loops tighter, Copilot moves in bigger chunks

looplet consistently logged **more, smaller steps** (16–26) with frequent
test-run/fix cycles; Copilot surfaced **few large actions** (3–12) but spent more
tokens *per* action (its output-token and credit counts are much higher than on the
easy suite - it "thinks" more per turn). Two valid strategies; both converged.

### Cost - comparable order of magnitude on hard work

On the easy suite looplet looked dramatically leaner; on hard tasks the gap
narrows (Copilot 1.37 M input / 51 k output tokens; looplet ~1.15 M est. total).
Real reasoning on hard problems costs tokens no matter the harness. Copilot's
per-task credit spend (~44) is the headline number if you pay per credit.

### Code design - matched model family, different observed defaults

- **h3**: both refactored well - a thin `app.py` facade (37–46 lines) delegating to
  two new layered modules (`models` + `operations`/`ledger`), global dict replaced,
  behaviour preserved.
- **h4**: **looplet** wrote a 275-line **OO token-class hierarchy** (`Literal`,
  `AnyChar`, `CharClass`, …) - readable, extensible; **Copilot** wrote a compact
  170-line **tuple-node + recursive backtracker**. Both correct.
- **h2**: both reached for the canonical `OrderedDict` + `move_to_end` LRU (~200
  lines) - idiomatic.

The recurring theme: looplet's cartridge (system prompt + ruff `LinterHook`) pushes
more structured, typed, documented code; Copilot defaults to leaner solutions.

---

## Honest notes

- **A verifier bug was found and fixed mid-run.** The first h1 grading falsely
  failed both agents because a naive substring check flagged the legitimate
  `_eval(node)` recursion as the banned builtin `eval(`. The scanner now uses word
  boundaries + AST-name awareness; both agents had in fact produced correct
  multi-module parsers. (Lesson: grade the *build*, not a substring.)
- **Tokens aren't perfectly commensurable** (Copilot real incl. cache; looplet
  chars/4 estimate - the proxy strips `usage`).
- **Still bounded scope.** These are hard but self-contained and objectively
  checkable. They don't test ambiguous product requirements, huge existing
  codebases, external services, or multi-hour autonomy - where planning, context
  management, and recovery would separate harnesses further.

---

## Bottom line

Scaled up to hard, multi-file, long-horizon work, your looplet coder example
**holds parity with Copilot on correctness (4/4 vs 4/4)** and is **competitive on
speed and cost** - winning outright on two of four tasks. The easy-suite takeaway
("looplet is leaner and faster") becomes "**they're peers on hard problems**,"
because long model calls dominate both end-to-end paths. looplet's structural
edge remains **ownership**: the same observed pass count, tighter test/fix
loops you can inspect and modify, versus Copilot's broader built-in toolbox and
product polish.
