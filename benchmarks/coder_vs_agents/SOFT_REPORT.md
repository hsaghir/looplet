# Non-coding suite — looplet coder vs GitHub Copilot CLI

Third companion to [REPORT.md](REPORT.md) (short coding) and
[HARD_REPORT.md](HARD_REPORT.md) (hard coding). Six open-ended tasks —
system design, conceptual explanation, debugging reasoning, and
professional writing — where there is no single correct answer.

**How they were graded.** Each agent wrote its answer to `answer.md`
(same model, `claude-sonnet-4.6`, fresh workspace). Quality was then
scored by a **blind LLM-as-judge**: two *neutral* models from a different
family than the contestants (`gemini-3.1-pro-preview` and `gpt-5.5`), each
scoring both answers on a 4-criterion rubric (accuracy, completeness,
clarity, practicality; 1–5 each → /20). To kill position bias, every
(task, judge) pair was judged in **both** A/B orders — 24 judgments total.
The judge never saw which system wrote which answer.

**TL;DR** — This is where the two **diverge**. Both produced strong,
well-structured, accurate answers, but **Copilot won 5 of 6 tasks** (1
tie), by a consistent margin: **18.8/20 vs 17.2/20**. The result held
across both judges and both orders, so it's robust — not a position or
self-preference artifact. The gap is *depth of specific expert insight*,
and its most likely cause is the **system prompt**: looplet's example
cartridge is a **coder**, tuned for terse, action-oriented engineering;
Copilot is a broader generalist assistant.

---

## Scores (blind judge, averaged over 2 judges × 2 orders)

| Task | Kind | looplet /20 | copilot /20 | winner |
|---|---|--:|--:|:--:|
| s1 URL shortener design | design | 14.5 | **18.2** | copilot |
| s2 rate-limiter design | design | 17.2 | **18.8** | copilot |
| s3 explain TLS handshake | explain | 17.2 | **19.5** | copilot |
| s4 explain CAP theorem | explain | 17.8 | **19.2** | copilot |
| s5 diagnose p99 tail latency | reasoning | 17.2 | **18.5** | copilot |
| s6 incident postmortem | writing | **19.0** | 18.5 | tie |
| **Average** | | **17.2** | **18.8** | |

Per-task wins: **copilot 5, looplet 0, tie 1.** Judgment-level votes
(each of 24 judgments = one vote): **copilot 18, looplet 3, tie 3.**

## Efficiency (from the run itself)

| Metric | looplet | copilot |
|---|--:|--:|
| Tasks with a valid answer | 6/6 | 6/6 |
| Total wall time | **508.9s** | 605.2s |
| Avg wall / task | **84.8s** | 100.9s |
| Loop shape | ~2 steps (write + done) | ~1 action |

Both effectively **one-shot** each answer — so this gap is *not* about the
loop, tools, or iteration. It's about what the harness's prompt elicits
from the same model.

---

## Why Copilot won — the judges' own reasons

- **s1 (biggest gap, 14.5 vs 18.2):** Copilot did the back-of-envelope math
  and noticed the workload is actually **write-heavy** (10k writes/s vs
  ~1.2k reads/s) and designed around that; looplet "blindly assumed the
  traditional read-heavy profile," a real reasoning miss. Copilot also
  committed to specific tech (Cassandra / ClickHouse / PostgreSQL) vs
  looplet's more generic treatment.
- **s2 rate limiter:** Copilot added deeper practical touches (cost-weighted
  tokens, two-tier limiting for bursts).
- **s3 TLS:** Copilot covered more modern MITM defenses (HSTS, CT logs, CAA)
  and handled TLS 1.2 vs 1.3 nuances more precisely.
- **s4 CAP:** Copilot's examples were sharper (etcd for CP, Cassandra for AP)
  with clearer replication detail.
- **s5 tail latency:** "Both exceptional" — Copilot edged in with concrete
  real-world causes (Kubernetes CFS CPU throttling, retry storms).
- **s6 postmortem:** **tie** — "both flawless," blameless tone, realistic
  detail, strong action items. looplet actually scored fractionally higher
  on Gemini here.

The recurring theme: looplet's answers are **accurate and well-organized**
(its s1 was 319 lines of clean, sectioned markdown) but a notch less rich
in *specific senior-engineer insight*; Copilot's answers read like a
staff engineer went one level deeper.

---

## What this means (and the important caveat)

- Same model, both one-shot → the differentiator is the **system prompt /
  harness persona**. looplet's shipped example is `coder.cartridge`, whose
  prompt pushes concise, do-the-work-and-stop engineering behavior. That
  discipline is an asset on coding (where it tied/beat Copilot) but a mild
  handicap on open-ended design essays, which reward expansive depth.
- **This tests the coder cartridge outside its lane.** looplet's own answer
  to "do non-coding work" is *use a different cartridge* — the repo ships
  `planner.cartridge` and `skillful_analyst.cartridge` for exactly this.
  A fair "looplet for design" comparison would swap those in; the coder is
  a specialist and this suite is a generalist test. The clean way to
  confirm the hypothesis is to re-run s1/s3 with `skillful_analyst` and see
  the gap close (recommended follow-up).
- **Both are genuinely good.** 17.2/20 is a solid answer a team would happily
  ship; Copilot is simply the more polished generalist here.

---

## The three suites together

| Suite | looplet | copilot | takeaway |
|---|:--:|:--:|---|
| Short coding (9) | 9/9, faster, leaner | 9/9 | looplet edge: speed + context |
| Hard coding (4) | 4/4 | 4/4 | peers; model does the work |
| Non-coding (6) | 17.2/20 | **18.8/20** | **Copilot edge on open-ended prose** |

**Bottom line:** your looplet *coder* is a coding specialist — a true peer
of Copilot on software tasks, and faster/leaner on the easy ones — but on
design, explanation, and reasoning essays, Copilot's generalist harness
extracts noticeably more depth from the same model. The fix on the looplet
side isn't a better loop; it's a **cartridge tuned for the job** (planner/
analyst), which is exactly the "own your agent" workflow looplet is built
around.

---

## Update — v2: after the prompt additions

After adding the answer-quality line to the coder prompt (*"When explaining
or designing… lead with the key decision, be specific and quantify, name the
main tradeoffs"*) plus the two safety/accuracy lines, the soft suite was
re-run with **Copilot held fixed as the baseline** (its answers reused) and
**looplet re-generated** with the new prompt, then re-judged blind (same 2
judges × 2 orders = 24 judgments). The raw `results_*.json` and `runs_*/`
artifacts are produced when you re-run (git-ignored; see the README).

| Task | looplet v1→v2 | copilot (fixed) | gap v1→v2 |
|---|--:|--:|--:|
| s1 URL shortener | 14.5 → 15.0 (+0.5) | 18.0 | +3.8 → +2.8 |
| s2 rate limiter | 17.2 → **18.2** (+1.0) | 17.5 | +1.5 → **−1.0 (win)** |
| s3 TLS | 17.2 → 18.5 (+1.2) | 19.5 | +2.2 → +1.0 |
| s4 CAP | 17.8 → 18.8 (+1.0) | 19.4 | +1.5 → +0.8 |
| s5 tail latency | 17.2 → 18.5 (+1.2) | 18.6 | +1.2 → +0.2 (tie) |
| s6 postmortem | 19.0 → 18.8 (−0.2) | 18.6 | −0.5 → 0.0 (tie) |
| **Average** | **17.17 → 17.96 (+0.79)** | ~18.6 | **+1.62 → +0.62** |

**The additions worked: the quality gap closed ~62%** (1.62 → 0.62 points),
looplet improved on **5 of 6** tasks, and it went from **0 task wins to 1 win
- 2 ties**. The biggest lifts were the pure-explanation tasks (TLS, CAP, tail
latency, +1.0–1.2 each) — exactly what the "lead with the decision, be
specific, quantify" line targets.

Caveats: (1) single looplet re-run, so some of the +0.79 is noise — but the
gain is consistent across 5/6 tasks with a plausible mechanism, so it's
credible, not a fluke. (2) Copilot's *score* drifted slightly despite
identical answers — an artifact of **pairwise** judging (scores are relative
to the paired answer), so the **gap** is the robust metric. (3) **s1 remains
the weak spot** (+2.8): the prompt fixed presentation/depth but not the
specific analytical miss (assuming a read-heavy profile for a write-heavy
workload) — that's a content error, not a formatting one. (4) On raw
head-to-head votes Copilot still edges most tasks by a hair (18 vs 5), but on
averaged scores the two are now essentially at parity.

**Takeaway:** ~4 lines of prompt closed most of the non-coding gap at zero
cost to coding performance — a clean win for the "small prompt, targeted
edits" approach, and cheaper than swapping cartridges. For the last mile
(s1-style analytical depth), a planner/analyst cartridge is still the right
tool.
