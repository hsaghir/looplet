# Changelog

All notable changes to `looplet` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `state.step_context`: per-step ephemeral dict for hook-to-hook communication.
  The loop clears it at step start; hooks write/read within the step.
  See [hooks.md](hooks.md#hook-to-hook-communication-step_context).
- `docs/faq.md`: "Why not LangGraph?" honest comparison (thanks @mvanhorn, #17)

## [0.1.7] - 2026-04-21

First public release of `looplet`.

### Added (launch polish)
- `ROADMAP.md` with a frozen v1.0 API contract and explicit
  out-of-scope list.
- `docs/` site scaffold (tutorial, evals, recipes, hooks, good-first-issues,
  discussions-seed, demo-script) + mkdocs-material config + GitHub
  Pages workflow.
- `THIRD_PARTY_USERS.md` social-proof seed.
- `src/looplet/examples/ollama_hello.py` — zero-API-key onboarding.
- Codecov upload step in CI (non-blocking).
- Leaner README (<170 lines) with the pydantic-ai-harness disambiguation
  moved to the top.

### Added (evals — pytest-style agent evaluation)
- **Eval framework** (`looplet.evals`). Write `eval_*` functions
  that take `EvalContext` and return any of `float`, `bool`, `str`,
  `dict`, or `EvalResult`. The framework normalizes all return types.
- **`eval_discover(path)`** — auto-discovers eval functions in
  `eval_*.py` files (like pytest discovers `test_*`).
- **`eval_run(evals, ctx)`** — runs evaluators, auto-detects
  `llm` parameter for LLM-as-judge, catches errors gracefully.
- **`eval_run_batch(evals, contexts)`** — runs same evals across
  multiple trajectories with per-eval avg/min/max aggregation.
- **`eval_mark(*tags)`** — decorator for categorizing evals.
  `eval_run` and `eval_run_batch` accept `include=`/`exclude=` to
  filter by marks.
- **`eval_cli(args)`** — CLI runner with threshold-based pass/fail
  exit codes for CI integration.
- **`EvalHook`** — LoopHook that builds EvalContext at `on_loop_end`
  and runs all evaluators automatically during development.
- **`EvalContext.from_trajectory_dir()`** — loads context from saved
  trajectories with support for both looplet and benchmark formats.

### Added (MCP + skills)
- **`MCPToolAdapter`** — wraps MCP server tools as `ToolSpec` instances
  via JSON-RPC over stdio. No MCP SDK required.
- **`Skill`** — bundles tools + context + prompt fragment into one
  loadable unit. `skill.register(registry)` adds all tools.

### Added (approval)
- **`ApprovalHook`** — stops the loop when a tool returns
  `needs_approval=True`. Combined with `checkpoint_dir` for
  crash-safe async human-in-the-loop approval.
- Renamed `elicit` → `approval` uniformly: `LoopConfig.approval_handler`,
  `ToolContext.request_approval`, `ToolContext.approve()`.

### Changed (naming cleanup)
- Renamed internal names for clarity: `coerce_text` → `to_text`,
  `DiminishingReturnsTracker` → `StallDetector`,
  `reactive_compact` → `emergency_truncate`,
  `compress_session_log` → `age_session_entries`,
  `enforce_result_budget` → `trim_results`,
  `should_compress_context` → `is_context_oversized`,
  `HEAVY_BLOCK_KINDS` → `LARGE_CONTENT_TYPES`,
  `DefaultSummarizer` → `default_summarizer`.
- Renamed compact services: `DefaultCompactService` → `TruncateCompact`,
  `LLMCompactService` → `SummarizeCompact`.
- Renamed `normalise_hook_return` → `normalize_hook_return`.
- Moved `concurrent_dispatch` and `reactive_recovery` from `FLAGS`
  global singleton to `LoopConfig` fields.
- Trimmed `__all__` from 154 → 54 symbols organized into labeled tiers.

### Changed (developer experience)
- Added `preview_prompt()` — shows what the LLM sees before the first
  call. Invaluable for debugging.
- Added `TrajectoryRecorder.summary()` — one-liner run summary.
- Added `--trace DIR` to coding_agent example for trajectory recording.
- Added step-by-step tutorial to README (5 progressive steps).
- Added `LoopConfig` docstring with "start here" guide listing the
  4 essential fields.
- Added `FileCheckpointStore.load_latest()` + auto-resume wiring in
  `composable_loop` — crash-resume is now one line:
  `LoopConfig(checkpoint_dir="./ckpt")`.

### Removed
- Removed `async_loop.py` (feature-frozen, no consumers).
- Removed 3 mock examples (calculator, code_review, research).
  Replaced with `hello_world.py` (real LLM) + `coding_agent.py`
  (Claude Code-equivalent tools: bash, read, write, edit, glob,
  grep, think, done).
- Removed all back-compat aliases.
- Removed all internal project references (cadence, primal_security).

### Added (compaction strategies)
- **`PruneToolResults`** — new zero-LLM-call compaction service that
  clears old tool-result content while keeping conversation structure
  intact. Configurable `keep_recent` (how many recent tool results
  to preserve) and `compactable_tools` (restrict to specific tools).
  Cheapest possible compaction — use as the first stage in a chain.
- **`compact_chain(*services)`** — combinator that tries compaction
  services in order; first stage that has an effect wins. Replaces
  the need for a separate `ChainedCompactService` class. Usage:
  `compact_chain(PruneToolResults(), SummarizeCompact(), TruncateCompact())`.
- **`CompactOutcome.cleanup`** — optional post-compact callback.
  When set, `run_compact()` invokes it after firing `POST_COMPACT`.
  Use for domain-specific state resets (clear caches, re-inject
  context, reset token baselines) without the loop knowing details.

### Changed (renames — back-compat aliases kept)
- **`DefaultCompactService`** → **`TruncateCompact`** — clearer name
  for "drop old entries, keep N recent, zero LLM calls."
- **`LLMCompactService`** → **`SummarizeCompact`** — clearer name
  for "LLM summarizes middle, keeps N recent."
- Old names (`DefaultCompactService`, `LLMCompactService`) remain
  as aliases and continue to work.

### Added (context management pt. 2)
- **Prompt caching infrastructure** (`looplet.cache`). New
  `CachePolicy` dataclass declares which stable prompt sections
  (system prompt, tool schemas, memory) should carry Anthropic-style
  `cache_control` markers, with per-section TTL (`ephemeral` / `1h`).
  `LoopConfig.cache_policy` threads per-turn `CacheBreakpoint` lists
  (label + SHA-256 hash + TTL) to backends that expose
  `generate_with_cache(..., cache_breakpoints=[...])`. Backends
  without the kwarg keep working unchanged — caching is strictly
  additive. `CacheBreakDetector` ships as a drop-in observer hook
  that records section-hash changes across turns for cache-miss
  telemetry.
- **`LLMCompactService`** — new compaction strategy that spends one
  LLM call to summarise the session. Produces a dense 4-section
  summary (task goal, findings, open questions, recent decisions)
  spliced into the session log as a synthetic entry after
  keep-recent pruning. Falls back to deterministic keep-recent on
  any summariser error. Trade-off vs `DefaultCompactService`: one
  LLM call per compaction for preserved reasoning chains.
- **Threshold-tier context budgeting** (`looplet.budget`). New
  `ContextBudget` dataclass with `warning_at` / `error_at` /
  `compact_buffer` tiers. `ThresholdCompactHook` is a ready-to-register
  `should_compact` implementation that fires proactive compaction
  once estimated tokens cross the configured tier.
  `BudgetTelemetry` observer records per-step tier samples and
  exposes `peak_tier` for production dashboards.

### Added (architecture improvements)
- **Proactive compact hook slot** — `LoopHook.should_compact(state,
  session_log, conversation, step_num) -> bool`. Fires at the top of
  each step, before prompt build. Any hook returning `True` triggers
  the configured `CompactService` preemptively. Complements the
  reactive `prompt_too_long` path — use for message-count or
  token-estimate heuristics. `StreamingHook` gets a no-op stub.
- **Tool-result streaming via `TOOL_PROGRESS`** — new
  `LifecycleEvent.TOOL_PROGRESS`. When hooks are present, the loop
  builds a `ToolContext.on_progress` callback per tool-call that
  emits `TOOL_PROGRESS` (with the originating `tool_call`) whenever
  the tool invokes `ctx.report_progress(stage, data)`. Observers can
  stream intermediate output from long-running tools without
  blocking dispatch.
- **Budget-aware turn continuation** — new
  `LoopConfig.max_turn_continuations: int = 0`. When `> 0` and the
  backend exposes `last_stop_reason`, `llm_call_with_retry` will
  re-prompt up to N times on `stop_reason == "max_tokens"` and
  concatenate outputs so long thoughts aren't truncated mid-message.
  `LLMResult` gains `stop_reason` and `continuations` fields.
- **`build_briefing` / `build_prompt` as hook slots** — both are now
  optional methods on `LoopHook`. First hook returning a non-`None`
  string wins; the loop falls back to `LoopConfig.build_briefing` /
  `config.build_prompt` / the built-in default. Lets domain hooks
  own prompt construction without threading callables through
  `LoopConfig` separately.
- **`DomainAdapter`** — new dataclass bundling the five domain
  callables (`build_briefing`, `extract_entities`, `build_trace`,
  `build_prompt`, `extract_step_metadata`) into a single object.
  `LoopConfig.domain: DomainAdapter | None = None` seeds matching
  flat fields when they are `None`. Flat fields still win over the
  adapter, which wins over built-in defaults — use the adapter to
  package a reusable agent in one handle instead of five kwargs.

### Removed (breaking)
- **`InvestigationLog`** backward-compat alias is gone — use
  `SessionLog` directly.
- **`HARNESS_FLAGS`** backward-compat alias is gone — use `FLAGS`.
- **Legacy `CADENCE_*` environment variables** for feature flags are
  no longer read; use the `LOOPLET_*` prefix.
- **`_clone_tools_excluding`** private alias is gone — use
  `clone_tools_excluding`.
- **`LoopConfig.permissions`** is gone. Register a
  `PermissionHook(PermissionEngine(...))` in `hooks=[...]` instead —
  it flows through the same unified `HookDecision` + event bus as
  every other hook.

### Added
- **Unified hook vocabulary — `HookDecision`** (`looplet.hook_decision`).
  All hook slots now accept a single `HookDecision` return type (legacy
  `None` / `bool` / `str` returns still work via `normalise_hook_return`).
  Helpers `Allow()`, `Deny(reason)`, `Block(reason)`, `Stop(reason)`,
  `Continue()`, `InjectContext(text)` make intent explicit at the call
  site.
- **Lifecycle events — `on_event(payload)`** (`looplet.events`).
  `LoopHook` gained an optional `on_event(EventPayload)` method. The
  loop now fires 10 named events: `SESSION_START`, `PRE_LLM_CALL`,
  `POST_LLM_RESPONSE`, `PRE_TOOL_USE`, `POST_TOOL_USE`,
  `POST_TOOL_FAILURE`, `PRE_COMPACT`, `POST_COMPACT`, `STOP`,
  `SUBAGENT_START`, `SUBAGENT_STOP`. Any hook can subscribe with a
  single method instead of implementing every slot.
- **`PermissionHook`** (`looplet.permissions`) — wraps
  `PermissionEngine` and plugs it into the event bus so policy
  decisions flow through the same `HookDecision` path as custom hooks.
- **`CompactService` + `DefaultCompactService` + `run_compact(...)`**
  (`looplet.compact`) — reactive compaction is now a swappable
  service with `PRE_COMPACT` / `POST_COMPACT` events.
- **`LoopConfig.render_messages_override`** — byte-exact escape hatch.
  Receives `(messages, default_prompt, step_num)` and returns the
  exact prompt string sent to the LLM. Lets advanced callers take full
  control of prompt rendering without forking the loop.
- **First-class subagents** — `run_sub_loop(..., subagent_id=...)`
  now fires `SUBAGENT_START` / `SUBAGENT_STOP` events on the parent's
  hooks and returns `subagent_id` in the result dict for correlation.
- **`replay_loop(trace_dir, tools=...)`** — rerun a captured trace
  through a fresh `composable_loop` without calling the LLM again.
  Useful for golden-trajectory regression tests, hook A/Bs, and
  cost-free loop diffs. Raises `RuntimeError` if the replay loop
  requests more calls than were recorded or diverges in method
  (`generate` vs `generate_with_tools`). Falls back to
  `call_NN_response.txt` files when `manifest.jsonl` is missing.
- **`python -m looplet show <trace-dir>`** — stdlib-only CLI that
  prints a one-page summary of a captured trace (run id, termination,
  per-step tool calls with durations, LLM totals). Exit code 1 when
  the directory is missing or malformed.
- **`looplet.provenance`** — new module for debugging agent runs:
  - `RecordingLLMBackend` / `AsyncRecordingLLMBackend` wrap any backend
    and capture every prompt, system prompt, tool schema, response,
    duration, and error as `LLMCall` records. `generate_with_tools` is
    surfaced only when the wrapped backend supports it, so
    `NativeToolBackend` detection stays honest.
  - `TrajectoryRecorder` hook captures a structured `Trajectory` per
    run (steps, context-before, termination reason, embedded `Tracer`
    spans) and writes `trajectory.json` + `steps/step_NN.json`.
  - `ProvenanceSink` is a 3-line facade: `wrap_llm(...)`,
    `trajectory_hook()`, `flush()`.
  - On-disk layout is diff-friendly: `call_NN_prompt.txt` /
    `call_NN_response.txt` per LLM call plus a `manifest.jsonl`.
  - Both recorders accept `redact=` for secret scrubbing and
    `max_chars_per_call=` for bounded memory.
  - See [Provenance guide](provenance.md) for API reference,
    recipes, and performance notes.
- `Step.pretty()` — human-readable CLI formatter complementing
  `Step.summary()` (which is tuned for LLM context assembly).

## [0.1.6] - 2026-04-17

### Added
- **`looplet.testing`** — public test-utility module exposing
  `MockLLMBackend` and `AsyncMockLLMBackend` (scripted, zero-dependency)
  so downstream packages can unit-test hooks, tools, and backends
  without a real LLM provider.
- **PyPI publish workflow** (`.github/workflows/publish.yml`) that
  builds + publishes on version tags via PyPI trusted publishing.
- **README positioning matrix** comparing `looplet` to LangGraph,
  DSPy, and smolagents; observability/OTel wiring example; stability &
  versioning policy; real `AnthropicBackend` usage in quick-start.

### Fixed
- `resume_loop_state()` now restores the checkpointed `Conversation`
  thread (was silently dropping multi-turn message history on resume).
- `RoutingLLMBackend.generate_with_tools` is now gated dynamically via
  `__getattr__` so `hasattr(llm, "generate_with_tools")` returns a
  truthful answer for the currently-selected backend (consistent with
  `_FallbackLLM` and `CostTracker`).
- Async `__llm_error__` step is now recorded through `_history` to
  match the sync loop (previously caused session-log/conversation
  drift on LLM failure).

### Previously added in this release
- **`ToolError` taxonomy** — structured `ErrorKind` enum
  (`PERMISSION_DENIED`, `TIMEOUT`, `VALIDATION`, `EXECUTION`, `PARSE`,
  `CONTEXT_OVERFLOW`, `RATE_LIMIT`, `NETWORK`, `CANCELLED`) plus a
  `ToolError` dataclass. `ToolResult` now carries both `error: str`
  (for JSON-safe display) and `error_detail: ToolError` (for
  introspection).
- **`PermissionEngine`** — declarative `ALLOW` / `DENY` / `ASK` /
  `DEFAULT` rules with fail-closed `arg_matcher`, plug-in `ask_handler`
  for human-in-the-loop, and an append-only denial audit log.
- **`CancelToken`** — cooperative cancellation is now threaded through
  `LoopConfig` → `llm_call_with_retry` / `async_llm_call_with_retry`
  → `ToolContext.cancel_token`, so both the next LLM call and any
  in-flight tool can stop cleanly.
- **`ToolContext.elicit`** — `LoopConfig.elicit_handler` surfaces a
  generic `elicit(prompt) → str` protocol to tools for interactive
  prompts.
- **Multi-block messages** — `Message.content` supports a `list` of
  `ContentBlock(kind, data)` alongside plain `str`. `HEAVY_BLOCK_KINDS`
  (`image` / `audio` / `video` / `binary`) are stripped before
  summarization.
- **Async `build_trace`** — `async_composable_loop` now stashes the
  built trace on `state.trace` at exit (async generators can't
  `return` a value).
- **`SyncToAsyncAdapter.generate_with_tools`** — router-selected sync
  backends keep native-tools support in the async loop.
- **Preflight context check** — async loop matches sync by skipping a
  doomed LLM call when the prompt is already too long under
  `FLAGS.reactive_recovery`.
- **Checkpoint state counters** — `resume_loop_state` now round-trips
  `state.queries_used` and `state.budget_remaining` so budget
  enforcement continues across resume.

### Changed
- `ToolResult.error` narrowed back to `str | None` (JSON-safe). Use
  `ToolResult.error_detail` for structured introspection.
- `PermissionRule.matches()` now fails closed *per decision type*:
  `DENY` rules match on matcher errors (block), `ALLOW` / `ASK` rules
  do not (don't accidentally grant).
- `PermissionEngine._resolve_default` collapses ambiguous engine
  defaults (`ASK` / `DEFAULT`) to `DENY` so a decision never leaks into
  a `PermissionOutcome` where both `.allowed` and `.denied` are False.
- `ToolSpec._accepts_ctx` is computed eagerly at `register()` time (and
  self-heals in `dispatch()` for specs inserted directly).
- `_backend_accepts_cancel_token` cache keyed by `(type, method_name)`
  instead of `id()` (eliminates id-recycling hazard).
- `_classify_exception` broadened to detect `asyncio.CancelledError`,
  rate-limit, context-overflow, and parse exceptions by class name /
  message content.
- `SyncToAsyncAdapter._adapter_cache` now prefers the backend object
  itself as the dict key, with `id()` as a fallback for unhashable
  backends.
- `SessionLog.to_list()` includes `recall_key` for full round-trip
  through checkpoints.
- `ToolError.context` now round-trips through `Conversation.serialize`
  / `deserialize`.
- Permission-denied results from hooks now populate `error_detail` with
  `ErrorKind.PERMISSION_DENIED` (parity with the `PermissionEngine`
  path) in both sync and async loops.

### Fixed
- `_rebuild_prompt` now renders `memory` and falls back to the
  structured `build_prompt` from `looplet.prompts` instead of a
  bare f-string, restoring parity with the first-pass build.
- `_deserialize_message` now reconstructs `ToolError` from serialized
  `error_kind` / `error_retriable` / `error_context` fields.
- `_NullSessionLog` (async) gained the attributes the async loop
  expects: `entries`, `current_theory`, `to_list()`, `compact()`.

## [0.1.5] - initial public import

- Initial release as a standalone package. See the extraction
  commit history for the pre-extraction development timeline.
