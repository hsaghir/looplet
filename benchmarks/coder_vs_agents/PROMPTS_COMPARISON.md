# Agent system prompts: looplet coder vs Copilot CLI, Codex, Claude Code

A structural comparison of the four coding-agent system prompts, gathered
2026-07-04. Ties back to the [non-coding benchmark](SOFT_REPORT.md), where
prompt/persona is one plausible explanation for the observed quality gap, not
a causally isolated result.

> **First, a correction on the premise.** These are *not* all open source:
>
> | Agent | "Open source"? | How this prompt was obtained | Authority |
> | --- | --- | --- | --- |
> | **OpenAI Codex CLI** | ✅ Yes (Apache-2.0) | Official repo `openai/codex` → `codex-rs/core/gpt_5_codex_prompt.md` | authoritative, current |
> | **GitHub Copilot CLI** | ❌ No (proprietary) | Extracted from the installed npm bundle `@github/copilot/app.js` | authoritative (the real prompt used in our benchmark) |
> | **Claude Code** | ❌ No (proprietary) | Community extraction (x1xhlol leak repo) | indicative, ~Sonnet-4 / Aug-2025 snapshot - may be partial/dated |
> | **looplet coder** | ✅ Yes (your repo) | `examples/coder.cartridge/prompts/system.md` | authoritative |
>
> So: only Codex is genuinely open source. Copilot's ships (obfuscated) in
> the package you install and is extractable. Claude Code's is neither
> published nor in a readable package - what circulates is reverse-engineered.

---

## 1. Provenance & construction

- **looplet coder** - a single hand-written **32-line / ~2.4 KB Markdown file**
  (`system.md`, ~600 tokens), plus two dynamic memory sources wired in via
  `config.yaml` (`@instructions_memory` = repo AGENTS.md/CLAUDE.md if present,
  `@project_memory` = git branch + project context). Static, human-readable,
  diffable.
- **Copilot CLI** - **not a file; assembled in code.** `app.js` builds an
  XML document from ~10 parametrised sub-templates and renders `.asXML()`:
  `identity` (preamble, `tone_and_style`, `search_and_delegation`,
  `tool_efficiency`, `version_information`, `model_information`,
  `task_instructions`, `environment_context`), `code_change_instructions`
  (`rules_for_code_changes`), `guidelines`, `environment_limitations`
  (sandbox rules), `selectedAgentInstructions`, and a `lastInstructions`
  tail. It even splits static vs dynamic blocks for **prompt caching**
  (`SPLIT_SYSTEM_MESSAGE_CACHE`) - which is exactly why our benchmark saw
  huge *cached* input-token counts.
- **Codex CLI** - one **large Markdown file** (`gpt_5_codex_prompt.md`),
  five top sections, dominated by a very detailed *output formatting* spec.
- **Claude Code** - one **large prompt** (several thousand words) with ~9
  sections, heavy on tone/verbosity rules and TodoWrite task-management.

## 2. Identity line (how each frames itself)

| Agent | Opening |
| --- | --- |
| looplet coder | "You are an expert software engineer. … You never guess - you read first, then act." |
| Copilot CLI | "You are the GitHub Copilot CLI, a terminal assistant built by GitHub. **You are an interactive CLI tool that helps users with software engineering tasks.**" |
| Claude Code | "**You are an interactive CLI tool that helps users with software engineering tasks.**" |
| Codex CLI | "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer." |

> **Notable:** Copilot CLI's identity sentence is **verbatim Claude Code's** -
> Copilot CLI is clearly descended from the Claude Code prompt lineage.

## 3. Feature-by-feature

| Dimension | looplet coder | Copilot CLI | Codex CLI | Claude Code |
| --- | --- | --- | --- | --- |
| Size (static) | ~600 tok | few-K tok, composed | large | large |
| Read-before-edit | ✅ explicit | ✅ | ✅ | ✅ ("understand conventions first") |
| "Check the lib exists first" | ❌ | ~ | ✅ | ✅ (emphatic) |
| Test after changes | ✅ ("after EVERY change") | ✅ | ✅ | ✅ + "run lint/typecheck" |
| Planning / todos | ✅ `todo()` tool | ✅ todo | ✅ Plan tool ("skip easiest 25%") | ✅ TodoWrite ("VERY frequently") |
| Bug-first-write-failing-test | ✅ | ~ | ~ | ~ |
| Conciseness rule | "Minimal changes" | "concise, but thorough" | "very concise; teammate tone" | ✅✅ "**fewer than 4 lines**", 1-word answers |
| Output-formatting spec | ❌ (none) | ✅ `tone_and_style` | ✅✅ **huge** final-answer section | ✅ examples + markdown rules |
| Sandbox / environment awareness | ❌ | ✅✅ detailed sandbox rules | ✅ (dirty worktree, ASCII) | ✅ `<env>` block |
| Don't-revert-user-changes | ❌ | ~ | ✅✅ "STOP IMMEDIATELY" | ~ |
| Git-commit caution | worktree confirm | ✅ | never destructive git | ✅ "NEVER commit unless asked" |
| Sub-agent delegation | ✅ `subagent()` | ✅ `search_and_delegation`, fleet mode | ~ | ✅ Task tool |
| Parallel tool calls | ❌ (config: `concurrent_dispatch:false`) | ✅ | ~ | ✅✅ "batch in a single message" |
| Security refusals | ❌ (in prompt) | ✅ (perms system) | ~ | ✅✅ "defensive security only", no URL-guessing |
| Code comments stance | "docstrings on public fns" | rules | "comments rare" | "**DO NOT ADD ANY COMMENTS** unless asked" |
| Model self-identification | ❌ | ✅ `model_information` | ✅ (in identity) | ✅ ("you are Sonnet 4 …") |

## 4. Philosophy in one line each

- **looplet coder** - *disciplined engineer*: read → plan → edit → test →
  done. Terse, opinionated, one screen.
- **Copilot CLI** - *composable product*: many small tuned modules, sandbox
  & permission aware, cache-optimised, self-identifying.
- **Codex CLI** - *presentation-obsessed teammate*: correctness + a very
  precise spec for how the final message should read.
- **Claude Code** - *minimal-token operator*: extreme concision, aggressive
  TodoWrite planning, strong safety refusals.

---

## 5. So how does looplet's coder prompt stack up?

**Better than its size suggests.** At ~600 tokens it encodes most of the
same *engineering discipline* the big three spend thousands of tokens on:
read-before-edit, test-after-every-change, write-a-failing-test-first,
minimal diffs, type hints/docstrings, `think()` when stuck. Philosophically
it's closest to **Claude Code** (terse, read-first, mimic conventions,
verify) - which is striking for a hand-written example file.

**What it deliberately omits** (and how that may relate to the benchmark):

1. **No output/answer-quality spec.** Codex devotes its largest section to
   *how to present work*; Copilot has `tone_and_style` + "be thorough in
   your work." looplet's prompt says nothing about depth, structure, or
   thoroughness of prose - it's all about *making changes*. That is very
   plausibly why it trailed on the **non-coding design/explanation tasks**
   (17.2 vs 18.8): the model was told to be a terse code-changer, so it
   produced accurate-but-lean essays.
2. **No "verify the library/API exists first"** (both Codex and Claude Code
   stress this - a real hallucination guard).
3. **No environment/sandbox or don't-revert-unrelated-changes rules** that
   Copilot and Codex use to stay safe in messy real repos.
4. **Parallel tool calls off** (`concurrent_dispatch:false`) where Claude
   Code/Copilot lean hard into batching for speed.

## 6. Concrete, cheap borrowings for looplet's coder

- Add ~4 lines of **answer-quality guidance** for non-edit responses ("When
  explaining or designing: lead with the key insight, be specific and
  quantify, cover the important tradeoffs") - directly targets the
  non-coding gap without bloating the prompt.
- Add a one-liner: **"Before using a library/API, confirm it exists in this
  codebase (imports, manifest)."**
- Add Codex's **"if you see unexpected changes you didn't make, stop and
  ask"** and "never `git reset --hard`/`checkout --` without approval."
- Consider enabling **parallel read-only tool calls** for exploration speed.
- The looplet-native alternative for non-coding work remains: **use a
  `planner`/`skillful_analyst` cartridge**, not the coder - different job,
  different prompt, which is the whole "own your agent" point.

**Bottom line:** looplet's coder prompt is a compact,
information-dense distillation of the same craft the commercial agents
encode - it just scopes itself tightly to *writing and testing code* and
skips the presentation/safety scaffolding the others carry. That focus is
consistent with the observed coding and open-ended prose results, but the
benchmark does not isolate prompt effects from serving-path or sample variance.

---

## 7. Pi (earendil-works/pi) - the minimalist datapoint

Pi is the coding agent the looplet README cites as ranking #2 on
TerminalBench with just four tools. Source is open (MIT, TypeScript):
`packages/coding-agent/src/core/system-prompt.ts` (`buildSystemPrompt`).

**Its entire default system prompt is ~15 lines:**

> You are an expert coding assistant operating inside pi, a coding agent
> harness. You help users by reading files, executing commands, editing
> code, and writing new files.
> Available tools: `read` / `bash` / `edit` / `write`
> Guidelines: `<tool-conditional bullets>` · "Be concise in your responses"
> · "Show file paths clearly when working with files"
> Current date / working directory.

Key design points:

- **Smaller than looplet's** (which is already small). No workflow list, no
  code-quality section, no safety rules in the prompt.
- **Guidance lives in the tool descriptions, not the prompt.** Pi's `edit`
  tool *description* carries the discipline: "Every edits[].oldText must
  match a unique, non-overlapping region… merge nearby changes into one
  edit… don't include large unchanged regions." Its `promptGuidelines`
  inject one-liners only for the tools that are actually active.
- **Four default tools** (`read`, `bash`, `edit`, `write`); `grep`/`find`/`ls`
  are opt-in. **No web tool at all** - if Pi needs the network it uses
  `bash` + `curl`.
- Philosophy is identical to looplet's: *the harness is the product; keep
  the prompt tiny and co-locate specifics with tools.*

> **Takeaway for looplet:** Pi is proof that a tiny prompt + a handful of
> tools is enough to top a benchmark. It validates looplet's approach and
> argues *against* bloating the coder prompt - new guidance should go into
> **tool descriptions** (which looplet already supports via `tool.yaml`),
> not into `system.md`.

---

## 8. Web/search tooling - what each uses, and looplet's gap

| Agent | Fetch a URL | Search the web | Backend |
| --- | --- | --- | --- |
| Claude Code | `WebFetch` | **`WebSearch`** | Anthropic server-side |
| Codex CLI | (via tools) | **`web_search`** | OpenAI server-side (config/`--search`) |
| Copilot CLI | `fetch` | via **MCP** (fetch/search servers) + `--allow-all-urls` | pluggable MCP |
| **looplet coder** | ✅ **`web_fetch`** (stdlib, zero-dep) | ❌ **none** | - |
| Pi | ❌ (uses `bash`+curl) | ❌ | - |

So looplet already matches the **fetch** capability (its `web_fetch` is a
dependency-free `urllib` + `HTMLParser` reader with optional LLM
summarization). The only gap versus Claude Code / Codex is a **`web_search`**
tool - *query a search engine, get ranked results* - so the agent can find
the right URL before fetching it.

**What to add without bloat - a zero-dependency `web_search` tool.** It fits
looplet's philosophy exactly: one more `tools/web_search/` folder, no new
runtime deps, guidance in the tool description (not the prompt). Sketch:

- Default backend: **DuckDuckGo** HTML endpoint
  (`https://lite.duckduckgo.com/lite/?q=…` or `https://html.duckduckgo.com/html/`),
  parsed with the same stdlib `HTMLParser` `web_fetch` already uses → returns
  top-N `{title, url, snippet}`. **No API key, no dependency.**
- Optional swappable backend gated on an env var (`BRAVE_API_KEY` /
  `TAVILY_API_KEY` / `SEARXNG_URL`) for higher-quality results - the
  "own your agent" swap, off by default.
- Typical loop: `web_search(query)` → pick a result → `web_fetch(url)`.

No prompt change is required; the tool's own `description` teaches usage,
just like `web_fetch` does today.

---

## 9. Recommendations - add these, skip the rest (no bloat)

Given Pi's evidence, the right move is **minimal**. Two of the three prompt
lines below are safety/accuracy guards; one closes the non-coding gap.
Everything else belongs in tool descriptions.

**A. Three lines for `system.md` (~+4 lines total, targeted):**

1. Answer-quality (closes the 17.2→18.8 non-coding gap): under *Code
   quality* or a new one-line note - *"When explaining or designing instead
   of editing: lead with the key decision, be specific and quantify, and
   name the main tradeoffs."*
2. Hallucination guard (Codex/Claude Code/Pi all have it): in *Tool rules* -
   *"Before using a library or API, confirm it already exists in this repo
   (imports, manifest) - don't assume."*
3. Repo-safety (Codex): in *Tool rules* - *"Never revert changes you didn't
   make; if you see unexpected edits, stop and ask. Never `git reset --hard`
   / `checkout --` without approval."*

**B. One new tool (no prompt bloat):** the zero-dep `web_search` above.

**C. Deliberately NOT adding** (would bloat, low ROI for the coder): Codex's
giant final-answer formatting section; Claude Code's "<4 lines" verbosity
regime; sandbox/environment blocks; parallel-tool-call essay. If any of
these matter, they belong in a *different cartridge* or in `config.yaml`
(e.g., flip `concurrent_dispatch` for parallel reads), not `system.md`.

**Net:** ~4 prompt lines + 1 tool. That keeps the coder's ~600-token prompt
essentially as lean as Pi's while closing the two real gaps the benchmarks
exposed (open-ended depth, and web search).
