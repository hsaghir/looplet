# Changelog

All notable changes to `looplet` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-07-16

### Added

- **Cartridge-native behavioral contracts.** Cartridges can ship cases,
  independent outcome collectors, and required graders under `evals/`.
  `run_cartridge_evals()` executes the complete contract in fresh per-case
  workspaces and persists trajectories, artifacts, grader-only expectations,
  and results for offline review.
- **Out-of-process control surfaces.** The RPC protocol now covers lifecycle
  events, cancellation, completion payloads, checkpoint/resume, backend
  selection, capability negotiation, and protocol versioning. LEP hooks, SSP
  state service, and MGP model gateway provide explicit cross-runtime
  boundaries without changing the core loop.
- **Cartridge portability analysis and portable examples.** `looplet
  portability` reports blockers and supported tiers. Portable twins for the
  shipped cartridges demonstrate MCP/LEP boundaries and are protected by
  cross-runtime conformance tests.
- **Runtime interaction primitives.** `SkillRuntime.ask_user()` and
  `batch_ask_user()` support host-supplied structured questions while retaining
  a sequential fallback for adapters without native form rendering.
- **Declarative hook contracts.** `ViewSpec`, thread-rewrite support, and
  related hook surfaces make context views and rewrites inspectable.
- **Provider usage telemetry.** Model token usage is recorded on state and
  lifecycle events so hooks and host applications can account for real calls.
- **Environment-selected backend construction.** `make_backend()` and the RPC
  backend command support Anthropic, OpenAI, OpenAI-compatible proxy, and
  explicit deterministic mock configuration through documented environment
  variables.
- **Public regression proof.** A network-free example captures fixed model
  responses, changes one tool implementation, independently collects the fresh
  artifact, and moves a required grader from red to green.

### Changed

- **Looplet is now positioned as test-driven harness engineering.** The README,
  documentation site, package metadata, roadmap, contribution guidance, and
  issue forms lead with the post-prototype workflow: own the loop, capture
  failures, grade independently observed outcomes, and gate harness changes in
  pytest or CI.
- **Agent-factory output is explicitly a draft.** `looplet new` defaults to
  `./agent.cartridge`, recommends `run-cartridge`, and requires review plus
  behavioral contracts before release. `run-workspace` remains a compatibility
  alias.
- **Replay terminology is precise.** Documentation says captured-response
  replay and states that tools and side effects execute again. Prompt and model
  changes require fresh sampled runs.
- **The shipped coder harness is more capable and cache-friendly.** It adds a
  reviewable `web_search` tool, sharper safety and answer-quality guidance, and
  keeps stable prompt prefixes to improve provider cache reuse.
- **Same-model harness evidence is documented with caveats.** Reproducible
  benchmark reports compare harness scaffolding rather than claiming universal
  model or framework superiority.

### Fixed

- **Eval integrity boundaries fail closed.** Zero scores remain scores;
  invalid returns, evaluator errors, collector failures, explicit failing
  labels, discovered required graders that are filtered, skipped, errored,
  invalid, or below the pass boundary, empty grader suites, malformed
  trajectories/metrics, duplicate eval modules, invalid thresholds, unsafe
  case IDs, seed traversal, symlink escapes, and stale workspaces no longer
  produce false greens. Detecting a grader deleted before discovery still
  requires a trusted expected-grader manifest.
- **Online and offline grading use the same evidence.** Expected data stays
  grader-only, terminal payloads survive persistence, and session logs,
  metadata, artifacts, and stop reasons round-trip.
- **Provider cache policy now reaches real requests.** OpenAI and Anthropic
  adapters translate configured cache breakpoints into provider request fields
  while preserving byte-equivalent behavior when caching is disabled.
- **RPC and MCP edges are hardened.** MCP JSON-string arguments and non-dict
  tool results are normalized safely; MCP response reads and startup cleanup
  honor configured timeouts; RPC checkpoint capabilities, completion reasons,
  and backend construction now reflect actual runtime behavior.
- **Installed `looplet new` includes its factory.** Wheels and sdists now carry
  `agent_factory.cartridge` plus its `coder.cartridge` parent, with a clean-wheel
  CI smoke test covering factory loading and CLI help.

### Developer experience

- Repository and launch documentation now use ASCII punctuation consistently;
  CI enforces the no-em-dash rule across tracked UTF-8 files.
- Release metadata, mirrored documentation, lockfile version, changelog date,
  strict docs build, and publication tag are covered by automated gates.
- The public issue backlog is split into independent evidence, compatibility,
  and provider work with explicit dependencies and non-goals.

## [0.2.0] - 2026-05-14

### Added
- **`mcp_servers:` slot in `config.yaml`** - declarative MCP
  (Model Context Protocol) transport for cross-language tools. Loader
  spawns one stdio subprocess per entry, runs MCP discovery, and
  registers every discovered tool into the agent’s registry alongside
  in-process Python tools. Optional per-server `tools:` allow-list.
  `AgentPreset` is now a context manager (`__exit__` terminates spawned
  MCP subprocesses cleanly).
- **`examples/mcp_demo.cartridge/`** - fully self-contained demo. The
  MCP server is a ~60-line Python stdio process bundled at
  `_server/calc.py` (no Node, no npm, no external deps).
- **`cartridge_root` auto-injected into the runtime dict**, so YAML
  fields can reference `${runtime.cartridge_root}` (most importantly in
  `mcp_servers.<name>.command`).
- **`language:` field** in cartridge metadata + per-tool
  `description.md` files (closes paper gaps).

### Changed
- **Cartridge spec v2 is now the only supported shape.** Loader
  hard-fails on `schema_version != 2`. RUNTIME-tier keys must live in
  `runtime.yaml`, not `config.yaml`.
- Round-2 cleanup: extracted three pure helpers from `_load.py`,
  trimmed re-exports, tightened render→runtime boundary.
- Four v2 cuts: `tags`, single-file constraints, `${py:}` references,
  `done_tools` plural.

### Fixed
- **`MCPToolAdapter` now uses newline-delimited JSON framing** per the
  MCP stdio spec. Previous LSP-style `Content-Length:` framing silently
  failed against every real MCP server. Init-failure errors now include
  exit code and stderr tail.

### Removed
- **All v1 back-compat.** Public surface trimmed: `Workspace`,
  `WorkspaceLayout`, `WorkspaceSerializationError`, `workspace_to_preset`,
  `preset_to_workspace`, `scaffold_workspace`, `ContextManagerHook`.
  Use `Cartridge*` equivalents.
- Magic `prompts/briefing.md` / `prompts/recovery.md` auto-load and
  `setup.py` escape hatch.
- Alias modules: `scaffold.py`, `prompt_files.py`, `hot_reload.py`,
  `spec_slots.py`, `workspace.py`.

### Added
- **Unified workspace reference grammar.** Three forms, one resolver,
  applied uniformly to every string value the cartridge loader
  processes:
  ```yaml
  max_steps:       ${runtime.max_steps:-15}     # per-invocation data
  compact_service: ${ref:compact_service}       # resource registry
  state:           ${py:my.app.state:MyState}   # imported Python symbol
  ```
  `${runtime.x}` supports nested lookup (`${runtime.a.b}`) and
  defaults (`${runtime.x:-15}`). The legacy `"@name"` form continues
  to work as an alias for `${ref:name}` so existing cartridges load
  unchanged. See `docs/workspace.md#reference-grammar` for the full
  spec.
- **`AgentPreset.resources` field.** the cartridge loader now
  populates this dict with every resource it built. Callers that
  need post-load access to live objects (benchmarks, evidence-bundle
  writers, SDK shims) read from `preset.resources` by name - no more
  module-hunting to reach a resource the cartridge constructed.
  Empty for presets built directly in code.
- **Declarative `state:` directive in `config.yaml`.** Cartridges
  can now describe their state class via the same grammar instead
  of relying on the `state_factory` constructor arg of
  `cartridge_to_preset`:
  ```yaml
  state: ${py:my.app.state:MyAgentState}   # → MyAgentState(max_steps=...)
  state: ${ref:my_state}                   # → resource as-is
  ```
  Priority: `state:` directive > `state_factory` arg > `DefaultState`.
  Closes the last gap where a non-trivial workspace had to write
  `setup.py` just to attach a custom state class.

### Fixed
- **YAML parser now skips full-line comments.** `#` lines used to
  raise `CartridgeSerializationError`. Same fix applied to the
  runtime-substitution pre-pass so `${runtime.x}` inside a YAML
  comment doesn't fire the regex.

### Removed (BREAKING)

- **`looplet.flags` module deleted.** All feature flags migrated to
  `LoopConfig` fields in 0.1.6 and the module had been deprecated
  since. The `FLAGS` singleton, `_Flags` class, and `LOOPLET_*`
  environment variables are gone. Use the equivalent `LoopConfig`
  fields directly: `LoopConfig(concurrent_dispatch=True)`,
  `LoopConfig(reactive_recovery=True)`, etc.
- **`looplet.scaffolding.StallDetector` class deleted.** Superseded
  by `StagnationHook` in `looplet.stagnation`. The
  back-compat bridge methods on `StepProgressTracker`
  (`consecutive_empty`, `is_diminishing`, `record_step`,
  `guidance_text`) are also removed - use the native
  `consecutive_unproductive` / `is_stagnating` properties.

### Changed
- **Documentation cleanup.** Stale "v1 / v2 / legacy / cartridge"
  framing trimmed from public docstrings and comments now that the
  migration to the cartridge format is complete. No behavioural
  change. ROADMAP entries that have shipped are dropped.
- **Coding and research presets now use `DefaultCompactService`.** The
  preset path gets the same production compaction policy users can
  import directly: prune old tool payloads, summarize older context,
  keep recent steps verbatim, and report stage-level outcomes.
- **Docs and shipped examples now teach `DefaultCompactService` first.**
  README, tutorial, AGENTS guide, packaged examples, and bundled
  workspace `compact_service.py` resources now point to the coherent
  default service, leaving `compact_chain(...)` as the custom-policy
  escape hatch.

### Added
- **`DefaultCompactService` and `default_compact_service(...)`.** A
  clear production default for context compaction that composes
  `PruneToolResults`, `SummarizeCompact`, and deterministic truncate
  fallback into one inspectable service. `CompactOutcome` now reports
  session-log entry counts, compacted step ranges, summaries, and a
  JSON-able `to_dict()` used in `POST_COMPACT` event payloads.
- **`extends:` workspace composition.** A cartridge's `config.yaml` may
  now declare `extends: <path>`. At load time the parent workspace is
  recursively materialized and overlaid with the child via a tempdir;
  child files override parent files at matching paths. Multi-level
  inheritance works (grandparent → parent → child), cycles raise
  `CartridgeSerializationError`, missing parent paths raise a clear
  error. (#44)
- **`examples/agent_factory.cartridge`.** First product built on
  `extends:`. Inherits all `coder.cartridge` tools and adds a
  `validate_workspace` tool that calls `cartridge_to_preset()` and
  returns structured success/error. ~4500-char system prompt teaching
  the cartridge v2 grammar, robustness rules, and `extends:` usage. (#44, #45)
- **`looplet.scaffold.scaffold_cartridge()`.** Plain Python helper that
  creates a working but stubbed cartridge skeleton in one call:
  `cartridge.json` + `config.yaml` + `prompts/system.md` +
  `tools/<name>/{tool.yaml, execute.py}` stubs (raise
  `NotImplementedError`) + the standard `done` tool. Idempotent  -
  re-running preserves existing files via `_write_if_absent`. (#46)
- **`builtin_tools:` directive in `config.yaml`.** Cartridges can opt
  into looplet-shipped tools without writing a `tools/<name>/` dir:
  ```yaml
  builtin_tools:
    - subagent
    - scaffold_cartridge
  ```
  Resolved at load time via `looplet.builtin_tools.AVAILABLE`. (#46)
- **`subagent` built-in tool.** Invokes another workspace as a
  sub-loop, sharing the parent's LLM and forwarding the parent's
  `workspace_config.path` as the sub-loop's `runtime["workspace"]`.
  Returns the sub-loop's final `done` summary. Recursion-guarded via
  `contextvars.ContextVar` (default `max_depth=5`). Sequential only  -
  parallel fan-out deferred. (#46)
- **`scaffold_cartridge` built-in tool.** Agent-callable wrapper
  around the scaffold helper. The agent factory uses it as the very
  first tool call. (#46)

### Fixed
- **`scaffold_cartridge` wrote invalid JSON** (`cartridge.json`
  emitted single-quoted Python repr instead of double-quoted JSON).
  `Workspace.from_directory()` and any external `json.loads()` failed.
  Now uses `json.dumps(name)` to emit RFC-compliant JSON.
- **`agent_factory` `_extract_json` example** in `prompts/system.md`
  was double-escaped inside an r-string (`\\s` instead of `\s`,
  `\\[` instead of `\[`), so the regex matched nothing. Agents that
  copied the helper verbatim got a broken extractor. Fixed.
- **`subagent` did not actually inherit parent runtime.** Docstring
  promised `workspace_config` propagation; code read
  `ctx.metadata["runtime"]` which the loop never sets. Now reads the
  parent's `workspace_config` resource and forwards
  `runtime["workspace"]` to the sub-loop's resource builders.
- **`subagent` recursion depth via process-global env var
  (`LOOPLET_SUBAGENT_DEPTH`)** - two parallel parent loops in the
  same process raced. Replaced with a `ContextVar` (threadsafe and
  per-async-task). The sub-loop receives a freshly-constructed
  `runtime` (it does NOT share the parent's `resources` /
  `file_cache` instances - only the cartridge path is forwarded).
- **`validate_workspace` was silent on TODO-laden scaffolds.** Now
  scans the system prompt for `<TODO:` markers and tool execute.py
  files for `NotImplementedError("scaffold:` and surfaces both as
  warnings - agents can no longer `done` on an unfilled skeleton. (#48)
- **`subagent` cwd-fallback was silent.** When neither the parent's
  `workspace_config` resource nor `ctx.metadata['runtime']` is
  present, the sub-loop's `runtime['workspace']` falls back to
  `Path.cwd()` AND the response now includes a structured
  `warning` field with explicit recovery hints. (#48)
- **Tool name vs directory name mismatches** (e.g. `tools/foo/`
  whose `tool.yaml` declares `name: WRONG_NAME`) used to silently
  register the wrong name and leave the agent unable to use `foo`.
  The loader now warns in loose mode and raises
  `CartridgeSerializationError` in strict mode. (#48)
- **Documentation cleanups (#48).** `subagent` module docstring no
  longer claims to "share the parent's runtime" (it constructs a
  fresh one). `builtin_tools/__init__.py` now lists both shipped
  built-ins (`subagent`, `scaffold_cartridge`).
- **Internal cleanup (#48).** Removed redundant `extends:` line
  check; rewrote tempdir registry as module-level state with single
  `atexit.register`; inlined a one-line `_is_absolute` helper;
  removed duplicate import; switched `subagent.max_steps` sentinel
  from `0` to `None`.

- **`examples/coder.cartridge` per-tool guidance + safety.** Three
  information-additive improvements modelled on patterns observed in
  production coding agents:
  1. **Read-required-first on `edit_file`.** `FileCache` now tracks
     every path passed to `read_file`; `edit_file` refuses with a
     model-actionable error (`{error, missing: "prior_read",
     recovery: "read_file(...)"}`) when called on a file that hasn't
     been read in the current session. Editing without reading is the
     #1 cause of `old_string` mismatch failures.
  2. **`bash` safety classifiers.** New `classify_bash_command` and
     `classify_sed_command` helpers in `coder_lib_tools.py` flag
     destructive command/flag combinations (`rm -rf`, `git push
     --force`, `git reset --hard`, `shutdown`, `mkfs`, …) and
     `sed -i` in-place edits (which bypass the file_cache and cause
     stale reads). The bash tool refuses both with a structured error
     pointing at a safer alternative (`edit_file` for in-place edits).
     The classifiers are exported so other cartridges can reuse them.
  3. **Rich per-tool descriptions.** Every `tool.yaml` in
     `examples/coder.cartridge` rewritten as a multi-paragraph
     description (Usage / Refusals / Examples / Recovery sections)
     using YAML block scalars (`|-`). The looplet workspace YAML
     loader gained `|`/`|-`/`>` block-scalar support so these
     descriptions round-trip correctly.
- **`ToolError.recovery_hint`** - structured suggestion (dict or str)
  for how the caller could recover. The dispatcher now populates it
  on the four self-correctable errors: unknown-tool ("did you mean?"),
  unexpected-argument (`{unexpected, did_you_mean, expected}`),
  missing-argument (`{missing, provided, expected}`), and empty
  required-string-argument (`{empty_param, expected}`). Information-
  additive: smarter models exploit the structured hint to self-correct
  without re-discovering the catalog from prose; existing models still
  see the same human-readable error message.
- **`looplet.LLMResponsesExhausted`** + **`MockLLMBackend(cycle=False)`**
  - opt-in test ergonomics. The default still cycles for backward
  compatibility; passing `cycle=False` makes the mock raise instead of
  silently re-using `responses[0]` past the last scripted answer
  (which previously made over-running loops look "stuck on step 1").
  Same flag on `AsyncMockLLMBackend`.
- **`run_sub_loop(parent_hooks=...)`** - opt-in event forwarding from
  a sub-loop to the parent's observability stack. When supplied, the
  parent's hooks (e.g. `MetricsHook`, `StreamingHook`,
  `TrajectoryRecorder`) receive every lifecycle event the sub-loop
  emits via their `on_event` method, tagged with `subagent_id` in the
  payload's `extra` dict so consumers can route / nest. Defaults to
  `None` - no forwarding, sub-loop fully isolated.

### Changed
- **`tool.yaml requires:` validated at workspace-load time.** A typo
  in `requires: [my_resoruce]` (missing or mistyped resource name)
  used to silently set `ctx.resources["my_resoruce"] = None` at
  dispatch and crash deep inside the tool body with `AttributeError`.
  The loader now warns in loose mode (default) and raises
  `CartridgeSerializationError` in strict mode, naming the
  unresolvable resource and listing the available ones - surfaces
  the bug at its source.

### Changed
- **Naming consolidation.** Dropped the legacy "cartridge" /
  "Composable Harness Workspace (CHW)" / "workspace v2" terminology
  in favour of the two canonical names already used in code:
  `Workspace` (the round-trippable directory format from
  `looplet.cartridge`) and `SkillBundle` (the runnable folder format
  from `looplet.bundles`). All docstrings, doc pages, README, and
  comments now use these names. The `ClaudeSkillCompatibility.level`
  string `"looplet-cartridge"` is renamed to `"looplet-bundle"`  -
  the only minor breaking change in this consolidation. Renamed
  `tests/test_cartridge_round_trip_smoke.py` to
  `tests/test_skill_bundle_round_trip_smoke.py`. No behavioural
  change; the `_chw_*` synthetic module-name prefixes (used
  internally by the cartridge loader) keep their names.

### Removed (BREAKING)

- **All `setup.py` files removed from shipped example cartridges.**
  Every workspace under `examples/*.workspace/` is now fully
  declarative; the imperative `setup.py` mechanism remains in the
  loader as the documented escape hatch for callers with truly
  imperative load-time wiring needs but no shipped example needs
  one. Migrations:
  * `coder.cartridge`: tools moved from `WORKSPACE_CONFIG` /
    `FILE_CACHE` module-globals to `requires: [...]` in `tool.yaml`
    + `ctx.resources[...]` in `execute.py`. `compact_service` moved
    to `resources/compact_service.py`.
  * `threat_intel.cartridge`, `dep_doctor.cartridge`,
    `git_detective.cartridge`: `compact_service` moved to
    `resources/compact_service.py`. `git_detective` tools moved
    from `REPO_CONFIG` module-globals to `requires: [repo_config]`
    + `ctx.resources["repo_config"]`.
  * `hello.cartridge`: `greet` tool moved from `_GREETING_LOG`
    module-global to `requires: [greeting_log]` +
    `ctx.resources["greeting_log"]`.

### v1 example modules deleted

- The legacy `examples/coder/`,
  `examples/dep_doctor/`, `examples/git_detective/`, and
  `examples/threat_intel/` agent-CLI directories have been removed.
  Their tool functions, hook classes, and helpers now live inside the
  matching `examples/<name>.workspace/` Composable Harness Cartridges
  as co-located helper modules (`<wsname>_lib.py` for the simple
  examples, `coder_lib_{tools,hooks,wiring}.py` for the coder one).
  The v2 cartridges are now the only published agent surface.
- The `examples/coder/skill/` SkillBundle was relocated to
  `tests/fixtures/coder_skill_bundle/` (with vendored sibling modules
  so it loads without any `examples.coder.*` import). All
  `looplet.bundles` / `looplet.blueprints` test coverage continues to
  exercise it via the new fixture path.
- Removed tests that targeted the deleted v1 modules:
  `tests/test_coder_example_smoke.py`,
  `tests/test_coder_reliability_smoke.py`,
  `tests/test_dep_doctor_example_smoke.py`,
  `tests/test_git_detective_example_smoke.py`,
  `tests/test_threat_intel_example_smoke.py`, and
  `test_distributions_include_coder_cartridge_and_dependency` from
  `tests/test_cartridge_round_trip_smoke.py`.

### Changed
- **Workspace loader pushes the cartridge root onto `sys.path`** for
  the duration of `cartridge_to_preset`, so a cartridge's tools /
  hooks / resources / setup.py can `from <wsname>_lib import X`
  without a dedicated import shim. The path is removed on exit.
  Cartridges should pick a unique top-level lib filename
  (`<wsname>_lib.py`, not bare `lib.py`) to avoid sys.modules cache
  collisions when two cartridges are loaded back-to-back in the same
  process.

### Added
- **Cartridge discovery without import.** `discover_skill_bundles(roots)`
  walks one or more roots and returns `BundleCard` records (name,
  description, entrypoint, tags, metadata, ok/errors) without
  importing the entrypoint. Powers the new
  `python -m looplet list-bundles <roots…>` CLI for product UIs and
  agent menus, with `--json` and `--include-invalid` modes.
- **Eval cases as data.** `EvalCase`, `load_cases`, `save_case`,
  `pytest_param_cases`, and the `parametrize_cases(path)` decorator
  let users write hand-edited JSON/JSONL cases that round-trip into
  pytest with their `marks` carried through. `assert_evals_pass(ctx,
  evals)` collapses the run/filter/pretty-print failure idiom into one
  call (with cached discovery for parametrized tests).
- **`looplet eval cases ls|show`** CLI subcommands for browsing case
  corpora directly from the terminal.
- **Outcome-grounded evals.** `EvalContext.artifacts` and
  `EvalHook(collectors=…)` let you grade *what changed in the world*
  (test results, repo diff) instead of grepping the trajectory.
  Trajectory directories may now ship an `artifacts.json` next to
  `trajectory.json`; `EvalContext.from_trajectory_dir` loads it
  automatically. Collectors that raise or return non-dicts are skipped
  silently - observers must never break a run.
- **`AgentPreset.run(llm, …)`** convenience method drives
  `composable_loop` with the preset's wiring in one call.
- **`composable_loop(…, max_steps=N, system_prompt=…)`** keyword
  shorthands for inline agents that don't construct a `LoopConfig`.
- **`OpenAIBackend.from_env()` / `AnthropicBackend.from_env()` /
  `AsyncOpenAIBackend.from_env()`** classmethods that read
  `OPENAI_*` / `ANTHROPIC_*` env vars in one line.
- **`OpenAIBackend(api_key=…)` no longer requires `base_url`** - the
  cloud path now works with just an API key (or env vars).
- **`BaseToolRegistry.tool` decorator** registers a `ToolSpec` in one
  step, mirroring the module-level `@tool` decorator.
- **`save_cases(cases, directory)`** plural form symmetric with
  :func:`load_cases`. Refuses to write when two cases share an `id`
  (which would silently overwrite each other on disk).
- **`metadata` dict on `ToolCall` and `ToolResult`** (PR #24) for
  carrying out-of-band tags through the loop without subclassing.
  Round-trips through `to_dict()`.
- **`metadata` dict on `StepRecord` and `LLMCall`** (PR #19) for
  per-step / per-call annotations on saved trajectories.
- **`LifecycleEvent.HOOK_DECISION`** (PR #20) fires whenever a hook
  returns a non-noop `HookDecision`. Payload carries the slot,
  hook name, and serialized decision - single observation point for
  every gate, redaction, or short-circuit in the run.
- **`LifecycleEvent.DONE_ACCEPTED`** (PR #21) fires after
  `check_done` accepts the `done()` call and the final payload is
  committed. Payload includes the `tool_call` and `tool_result` of
  the accepted termination - observer-only, fired right before STOP.
- **`serialize_harness(...)` + `TrajectoryRecorder(harness_snapshot=…)`**
  (PR #22) record a stable JSON-friendly snapshot of the agent
  config, tool list, hooks, and LLM backend on every saved
  trajectory. Lands in `trajectory.metadata["harness_snapshot"]`.
- **`tool_call` kwarg on `LoopHook.check_done`** (PR #23) so quality
  gates can inspect the agent's pending answer before it terminates.
  Backward-compatible: existing `check_done(self, state, log, ctx,
  step_num)` signatures continue to work via `inspect.signature`
  detection.

### Changed
- **Coder example split into modules.** `examples/coder/agent.py` now
  delegates to `examples/coder/{tools,hooks,wiring}.py` so the library
  entrypoint and the runnable cartridge in `examples/coder/skill/`
  share *exactly* the same composition. Modify behavior in
  `wiring.py` once and both surfaces pick it up. Public symbols
  re-exported for back-compat.
- **`looplet eval` subcommand** now routes to `looplet.evals.eval_cli`
  with full `-h`/`--help` support; the top-level CLI no longer eats
  option-like tokens before they reach the eval parser. The eval
  help text now also documents the `cases ls|show` subcommands.
- **Coder hooks default to "steer, don't restrict".** `TestGuardHook`
  ships in observe-only mode (failures inject a briefing nudge but
  `done()` is never blocked); `StagnationHook` uses
  `result_size_fingerprint` with a lenient threshold so legitimate
  retries don't trip a stall warning. Pass `test_strict=True` to
  recover the legacy hard-block behavior.
- **Coder example ships an outcome-grounded `EvalHook`** that re-runs
  the project's pytest suite after the loop and surfaces
  `tests_passing` via `ctx.artifacts`.

### Fixed
- **`EvalContext.from_trajectory_dir` now preserves `trajectory.metadata`.**
  Previously only four well-known top-level fields (`run_id`,
  `started_at`, `ended_at`, `termination_reason`) were copied into
  `EvalContext.metadata`, silently dropping `harness_snapshot`
  (added by PR #22's `TrajectoryRecorder(harness_snapshot=…)` kwarg)
  and any user-attached metadata. The full `trajectory.metadata`
  dict is now overlaid first, with the four top-level fields
  applied on top.
- **`TrajectoryRecorder` now reflects late `metadata` mutations.**
  When a downstream hook ran `post_dispatch` *after*
  `TrajectoryRecorder` and tagged `tool_call.metadata` /
  `tool_result.metadata` (the documented annotation point added by
  PR #24), the mutations were silently lost because the recorder
  had already snapshotted via `to_dict()`. The recorder now sweeps
  `state.steps` in `on_loop_end` and refreshes the metadata fields
  on every captured `StepRecord` so hook ordering no longer
  affects the saved trajectory.
- **`OpenAIBackend.from_env` / `AnthropicBackend.from_env` /
  `AsyncOpenAIBackend.from_env` raise clean errors upfront.**
  Previously, the OpenAI variants leaked an SDK-level `OpenAIError`
  when neither `OPENAI_API_KEY` nor `OPENAI_BASE_URL` was set, and
  `AnthropicBackend.from_env` raised a `TypeError` from looplet's
  own constructor. Both now raise a single `RuntimeError` with an
  actionable message naming the env var to set. The OpenAI
  variants also now default `api_key` to a sentinel when only
  `OPENAI_BASE_URL` is set, so local-server flows (Ollama / vLLM /
  llama.cpp) work without setting `OPENAI_API_KEY=ollama`.
- **Coder example: clarified the `eval_tests_passed` skip reason.**
  The label now says "no Python project (pyproject.toml/setup.py)
  detected in workspace; collector cannot re-run tests" instead of
  the misleading "no test runner detected", since the collector
  needs a project file to re-run pytest against.
- **`discover_skill_bundles` accepts `on_duplicate=` and `looplet
  list-bundles` no longer crashes on collisions.** Previously, two
  bundles claiming the same `name` field always raised
  `ValueError`, so a single dirty discovery root (e.g. left-over
  pytest fixtures under `/tmp`) made the entire `looplet
  list-bundles` CLI unusable. The function now accepts
  `on_duplicate="raise"` (default, back-compat), `"first_wins"`
  (silent), or `"warn"` (logs each collision to
  `looplet.bundles`). The CLI passes `"warn"` so users see what
  was dropped but still get a list of valid bundles.
- **`Conversation` serialization now round-trips `ToolCall` /
  `ToolResult` `metadata`.** PR #24 added the field but the
  Conversation serializer dropped it silently; `Conversation.deserialize`
  also never restored it. Both sides now plumb the field, so saved
  conversations preserve any out-of-band tags hooks attached.
- **`Message(role="system", …)` no longer breaks serialization.**
  `MessageRole` is a `str, Enum`, so callers naturally pass plain
  strings - but `_serialize_message` did `msg.role.value`, which
  raised `AttributeError` when `role` was a plain `str`.
  `Message.__post_init__` now coerces to `MessageRole` so both call
  styles work identically.
- **`check_done` signature cache no longer poisoned by id reuse.**
  PR #23's backward-compat dispatch cached `_accepts_tool_call_kwarg`
  results keyed on `id(bound_method)`. Bound methods are ephemeral
  in CPython (`obj.method` creates a fresh object each access), so
  they get garbage-collected and their ids get reused for unrelated
  methods on other classes - leaving the cache claiming a
  legacy-signature hook accepts `tool_call`, then raising
  `TypeError: ...check_done() got an unexpected keyword argument
  'tool_call'`. The cache now keys on `id(method.__func__)` (the
  stable underlying function) with a fallback to `id(method)` for
  callables that lack `__func__`.
- **`async_composable_loop` now accepts `max_steps=` and
  `system_prompt=` shorthands.** The sync `composable_loop` got
  these convenience kwargs but the async version was missed  -
  callers had to construct a `LoopConfig` even for one-liner async
  agents. The signatures now match.
- **`generate_kwargs` now reach backends declared as `**kwargs`.**
  `_accepts_kwarg` only matched explicitly-named parameters in the
  backend's `generate(...)` signature, so any backend written as
  `def generate(self, prompt, **kw)` (a common permissive pattern)
  silently dropped every entry of `LoopConfig.generate_kwargs`.
  The helper now also returns True when the function declares a
  `VAR_KEYWORD` parameter, so `top_p`, `response_format`,
  `chat_template_kwargs`, etc. propagate as documented.
- **`save_case(case, "evals/cases/")` no longer creates a file
  literally named `cases`.** The "treat as directory" branch only
  fired when the path already existed, so a non-existent
  trailing-slash path (the obvious "I want a directory" convention
  shown in `docs/evals.md`) wrote the case content into a single
  file at the path. The helper now also treats trailing path
  separators as directory intent and creates the parent directories
  before writing `<dir>/<case.id>.json`.
- **`MetricsCollector.total_llm_calls` is now populated by
  default.** The field was advertised in the report but no built-in
  hook updated it, so it sat at 0 unless callers wired their own
  counter. `MetricsHook.on_event` now increments it on every
  `POST_LLM_RESPONSE` lifecycle event.

## [0.1.8] - 2026-04-24

### Added
- `ctx.llm`: tools receive the loop's LLM backend for internal calls.
  Tracked by `RecordingLLMBackend` with `scope="tool:<name>"` for nested provenance.
- `LLMCall.scope`: provenance field for loop vs tool-internal calls.
- `state.step_context`: per-step ephemeral dict for hook-to-hook communication.
- `LoopConfig.tool_metadata`: static dict merged into every `ToolContext.metadata`.
- `LoopConfig.generate_kwargs`: extra kwargs passed through to every LLM call.
  Can override `temperature`, `max_tokens`, `system_prompt`. Supports
  provider-specific params (`chat_template_kwargs`, `response_format`, `top_p`).
- `async_composable_loop`: async generator for async LLM backends.
- `_SyncBridgeLLM`: sync tools can use `ctx.llm.generate()` even with async backends.
- `OpenAIBackend.tool_choice`: configurable `tool_choice` parameter.
- `PerToolLimitHook.default_limit`: blanket cap for all tools.
- `CompactOutcome.compacted`: property indicating if compaction reduced context.
- `register_done_tool()`: convenience for registering the done tool.
- `EvalResult.passed`: property for pass/fail determination.
- Async tool dispatch in sync loop: `dispatch()` detects coroutine returns.
- 3 example agents: threat intel briefing, git history detective, dependency doctor.

### Changed
- `ToolContext` is now always created (never `None`), with `metadata`
  populated from `state.metadata` (copy, not reference).
- `default_max_tokens` defaults to `None` across all backends - lets
  the provider API decide instead of forcing 2000.
- All docs and examples updated to use convenience `OpenAIBackend(base_url=...)`
  and `register_done_tool()`.

### Fixed
- `Step.to_dict()` key names: `call` → `tool_call`, `result` → `tool_result`.
- Tool validation error now shows what args were provided.
- `Step.summary()` shows dict preview instead of `?`.
- `Trajectory.task` field preserved in trajectory.json for eval round-trip.
- `TrajectoryRecorder(output_dir=...)` auto-saves on loop end.
- `RecoveryRegistry.register` warns on overwrite.
- `clone_tools_excluding` warns on missing names (typo detection).
- Permission audit strips `__…` scaffolding keys.
- Conversation `compact()` marks summary as compaction boundary.

## [0.1.7] - 2026-04-21

First public release of `looplet`.

### Added (launch polish)
- `ROADMAP.md` with a frozen v1.0 API contract and explicit
  out-of-scope list.
- `docs/` site scaffold (tutorial, evals, recipes, hooks, good-first-issues,
  discussions-seed, demo-script) + mkdocs-material config + GitHub
  Pages workflow.
- `THIRD_PARTY_USERS.md` social-proof seed.
- `src/looplet/examples/ollama_hello.py` - zero-API-key onboarding.
- Codecov upload step in CI (non-blocking).
- Leaner README (<170 lines) with the pydantic-ai-harness disambiguation
  moved to the top.

### Added (evals - pytest-style agent evaluation)
- **Eval framework** (`looplet.evals`). Write `eval_*` functions
  that take `EvalContext` and return any of `float`, `bool`, `str`,
  `dict`, or `EvalResult`. The framework normalizes all return types.
- **`eval_discover(path)`** - auto-discovers eval functions in
  `eval_*.py` files (like pytest discovers `test_*`).
- **`eval_run(evals, ctx)`** - runs evaluators, auto-detects
  `llm` parameter for LLM-as-judge, catches errors gracefully.
- **`eval_run_batch(evals, contexts)`** - runs same evals across
  multiple trajectories with per-eval avg/min/max aggregation.
- **`eval_mark(*tags)`** - decorator for categorizing evals.
  `eval_run` and `eval_run_batch` accept `include=`/`exclude=` to
  filter by marks.
- **`eval_cli(args)`** - CLI runner with threshold-based pass/fail
  exit codes for CI integration.
- **`EvalHook`** - LoopHook that builds EvalContext at `on_loop_end`
  and runs all evaluators automatically during development.
- **`EvalContext.from_trajectory_dir()`** - loads context from saved
  trajectories with support for both looplet and benchmark formats.

### Added (MCP + skills)
- **`MCPToolAdapter`** - wraps MCP server tools as `ToolSpec` instances
  via JSON-RPC over stdio. No MCP SDK required.
- **`Skill`** - bundles tools + context + prompt fragment into one
  loadable unit. `skill.register(registry)` adds all tools.

### Added (approval)
- **`ApprovalHook`** - stops the loop when a tool returns
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
- Added `preview_prompt()` - shows what the LLM sees before the first
  call. Invaluable for debugging.
- Added `TrajectoryRecorder.summary()` - one-liner run summary.
- Added `--trace DIR` to coding_agent example for trajectory recording.
- Added step-by-step tutorial to README (5 progressive steps).
- Added `LoopConfig` docstring with "start here" guide listing the
  4 essential fields.
- Added `FileCheckpointStore.load_latest()` + auto-resume wiring in
  `composable_loop` - crash-resume is now one line:
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
- **`PruneToolResults`** - new zero-LLM-call compaction service that
  clears old tool-result content while keeping conversation structure
  intact. Configurable `keep_recent` (how many recent tool results
  to preserve) and `compactable_tools` (restrict to specific tools).
  Cheapest possible compaction - use as the first stage in a chain.
- **`compact_chain(*services)`** - combinator that tries compaction
  services in order; first stage that has an effect wins. Replaces
  the need for a separate `ChainedCompactService` class. Usage:
  `compact_chain(PruneToolResults(), SummarizeCompact(), TruncateCompact())`.
- **`CompactOutcome.cleanup`** - optional post-compact callback.
  When set, `run_compact()` invokes it after firing `POST_COMPACT`.
  Use for domain-specific state resets (clear caches, re-inject
  context, reset token baselines) without the loop knowing details.

### Changed (renames - back-compat aliases kept)
- **`DefaultCompactService`** → **`TruncateCompact`** - clearer name
  for "drop old entries, keep N recent, zero LLM calls."
- **`LLMCompactService`** → **`SummarizeCompact`** - clearer name
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
  without the kwarg keep working unchanged - caching is strictly
  additive. `CacheBreakDetector` ships as a drop-in observer hook
  that records section-hash changes across turns for cache-miss
  telemetry.
- **`LLMCompactService`** - new compaction strategy that spends one
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
- **Proactive compact hook slot** - `LoopHook.should_compact(state,
  session_log, conversation, step_num) -> bool`. Fires at the top of
  each step, before prompt build. Any hook returning `True` triggers
  the configured `CompactService` preemptively. Complements the
  reactive `prompt_too_long` path - use for message-count or
  token-estimate heuristics. `StreamingHook` gets a no-op stub.
- **Tool-result streaming via `TOOL_PROGRESS`** - new
  `LifecycleEvent.TOOL_PROGRESS`. When hooks are present, the loop
  builds a `ToolContext.on_progress` callback per tool-call that
  emits `TOOL_PROGRESS` (with the originating `tool_call`) whenever
  the tool invokes `ctx.report_progress(stage, data)`. Observers can
  stream intermediate output from long-running tools without
  blocking dispatch.
- **Budget-aware turn continuation** - new
  `LoopConfig.max_turn_continuations: int = 0`. When `> 0` and the
  backend exposes `last_stop_reason`, `llm_call_with_retry` will
  re-prompt up to N times on `stop_reason == "max_tokens"` and
  concatenate outputs so long thoughts aren't truncated mid-message.
  `LLMResult` gains `stop_reason` and `continuations` fields.
- **`build_briefing` / `build_prompt` as hook slots** - both are now
  optional methods on `LoopHook`. First hook returning a non-`None`
  string wins; the loop falls back to `LoopConfig.build_briefing` /
  `config.build_prompt` / the built-in default. Lets domain hooks
  own prompt construction without threading callables through
  `LoopConfig` separately.
- **`DomainAdapter`** - new dataclass bundling the five domain
  callables (`build_briefing`, `extract_entities`, `build_trace`,
  `build_prompt`, `extract_step_metadata`) into a single object.
  `LoopConfig.domain: DomainAdapter | None = None` seeds matching
  flat fields when they are `None`. Flat fields still win over the
  adapter, which wins over built-in defaults - use the adapter to
  package a reusable agent in one handle instead of five kwargs.

### Removed (breaking)
- **`InvestigationLog`** backward-compat alias is gone - use
  `SessionLog` directly.
- **`HARNESS_FLAGS`** backward-compat alias is gone - use `FLAGS`.
- **Legacy `CADENCE_*` environment variables** for feature flags are
  no longer read; use the `LOOPLET_*` prefix.
- **`_clone_tools_excluding`** private alias is gone - use
  `clone_tools_excluding`.
- **`LoopConfig.permissions`** is gone. Register a
  `PermissionHook(PermissionEngine(...))` in `hooks=[...]` instead  -
  it flows through the same unified `HookDecision` + event bus as
  every other hook.

### Added
- **Unified hook vocabulary - `HookDecision`** (`looplet.hook_decision`).
  All hook slots now accept a single `HookDecision` return type (legacy
  `None` / `bool` / `str` returns still work via `normalise_hook_return`).
  Helpers `Allow()`, `Deny(reason)`, `Block(reason)`, `Stop(reason)`,
  `Continue()`, `InjectContext(text)` make intent explicit at the call
  site.
- **Lifecycle events - `on_event(payload)`** (`looplet.events`).
  `LoopHook` gained an optional `on_event(EventPayload)` method. The
  loop now fires 10 named events: `SESSION_START`, `PRE_LLM_CALL`,
  `POST_LLM_RESPONSE`, `PRE_TOOL_USE`, `POST_TOOL_USE`,
  `POST_TOOL_FAILURE`, `PRE_COMPACT`, `POST_COMPACT`, `STOP`,
  `SUBAGENT_START`, `SUBAGENT_STOP`. Any hook can subscribe with a
  single method instead of implementing every slot.
- **`PermissionHook`** (`looplet.permissions`) - wraps
  `PermissionEngine` and plugs it into the event bus so policy
  decisions flow through the same `HookDecision` path as custom hooks.
- **`CompactService` + `DefaultCompactService` + `run_compact(...)`**
  (`looplet.compact`) - reactive compaction is now a swappable
  service with `PRE_COMPACT` / `POST_COMPACT` events.
- **`LoopConfig.render_messages_override`** - byte-exact escape hatch.
  Receives `(messages, default_prompt, step_num)` and returns the
  exact prompt string sent to the LLM. Lets advanced callers take full
  control of prompt rendering without forking the loop.
- **First-class subagents** - `run_sub_loop(..., subagent_id=...)`
  now fires `SUBAGENT_START` / `SUBAGENT_STOP` events on the parent's
  hooks and returns `subagent_id` in the result dict for correlation.
- **`replay_loop(trace_dir, tools=...)`** - rerun a captured trace
  through a fresh `composable_loop` without calling the LLM again.
  Useful for golden-trajectory regression tests, hook A/Bs, and
  cost-free loop diffs. Raises `RuntimeError` if the replay loop
  requests more calls than were recorded or diverges in method
  (`generate` vs `generate_with_tools`). Falls back to
  `call_NN_response.txt` files when `manifest.jsonl` is missing.
- **`python -m looplet show <trace-dir>`** - stdlib-only CLI that
  prints a one-page summary of a captured trace (run id, termination,
  per-step tool calls with durations, LLM totals). Exit code 1 when
  the directory is missing or malformed.
- **`looplet.provenance`** - new module for debugging agent runs:
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
  - See [Provenance guide](docs/provenance.md) for API reference,
    recipes, and performance notes.
- `Step.pretty()` - human-readable CLI formatter complementing
  `Step.summary()` (which is tuned for LLM context assembly).

## [0.1.6] - 2026-04-17

### Added
- **`looplet.testing`** - public test-utility module exposing
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
- **`ToolError` taxonomy** - structured `ErrorKind` enum
  (`PERMISSION_DENIED`, `TIMEOUT`, `VALIDATION`, `EXECUTION`, `PARSE`,
  `CONTEXT_OVERFLOW`, `RATE_LIMIT`, `NETWORK`, `CANCELLED`) plus a
  `ToolError` dataclass. `ToolResult` now carries both `error: str`
  (for JSON-safe display) and `error_detail: ToolError` (for
  introspection).
- **`PermissionEngine`** - declarative `ALLOW` / `DENY` / `ASK` /
  `DEFAULT` rules with fail-closed `arg_matcher`, plug-in `ask_handler`
  for human-in-the-loop, and an append-only denial audit log.
- **`CancelToken`** - cooperative cancellation is now threaded through
  `LoopConfig` → `llm_call_with_retry` / `async_llm_call_with_retry`
  → `ToolContext.cancel_token`, so both the next LLM call and any
  in-flight tool can stop cleanly.
- **`ToolContext.elicit`** - `LoopConfig.elicit_handler` surfaces a
  generic `elicit(prompt) → str` protocol to tools for interactive
  prompts.
- **Multi-block messages** - `Message.content` supports a `list` of
  `ContentBlock(kind, data)` alongside plain `str`. `HEAVY_BLOCK_KINDS`
  (`image` / `audio` / `video` / `binary`) are stripped before
  summarization.
- **Async `build_trace`** - `async_composable_loop` now stashes the
  built trace on `state.trace` at exit (async generators can't
  `return` a value).
- **`SyncToAsyncAdapter.generate_with_tools`** - router-selected sync
  backends keep native-tools support in the async loop.
- **Preflight context check** - async loop matches sync by skipping a
  doomed LLM call when the prompt is already too long under
  `FLAGS.reactive_recovery`.
- **Checkpoint state counters** - `resume_loop_state` now round-trips
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
