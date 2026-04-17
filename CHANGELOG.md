# Changelog

All notable changes to `openharness` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`replay_loop(trace_dir, tools=...)`** — rerun a captured trace
  through a fresh `composable_loop` without calling the LLM again.
  Useful for golden-trajectory regression tests, hook A/Bs, and
  cost-free loop diffs. Raises `RuntimeError` if the replay loop
  requests more calls than were recorded or diverges in method
  (`generate` vs `generate_with_tools`). Falls back to
  `call_NN_response.txt` files when `manifest.jsonl` is missing.
- **`python -m openharness show <trace-dir>`** — stdlib-only CLI that
  prints a one-page summary of a captured trace (run id, termination,
  per-step tool calls with durations, LLM totals). Exit code 1 when
  the directory is missing or malformed.
- **`openharness.provenance`** — new module for debugging agent runs:
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
  - See [PROVENANCE_GUIDE.md](PROVENANCE_GUIDE.md) for API reference,
    recipes, and performance notes.
- `Step.pretty()` — human-readable CLI formatter complementing
  `Step.summary()` (which is tuned for LLM context assembly).

## [0.1.6] - 2026-04-17

### Added
- **`openharness.testing`** — public test-utility module exposing
  `MockLLMBackend` and `AsyncMockLLMBackend` (scripted, zero-dependency)
  so downstream packages can unit-test hooks, tools, and backends
  without a real LLM provider.
- **PyPI publish workflow** (`.github/workflows/publish.yml`) that
  builds + publishes on version tags via PyPI trusted publishing.
- **README positioning matrix** comparing `openharness` to LangGraph,
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
  structured `build_prompt` from `openharness.prompts` instead of a
  bare f-string, restoring parity with the first-pass build.
- `_deserialize_message` now reconstructs `ToolError` from serialized
  `error_kind` / `error_retriable` / `error_context` fields.
- `_NullSessionLog` (async) gained the attributes the async loop
  expects: `entries`, `current_theory`, `to_list()`, `compact()`.

## [0.1.5] - initial public import

- Extracted from `cadence` as a standalone package. See the extraction
  commit history for the pre-extraction development timeline.
