# looplet — Agent Guide

> This file is optimized for coding agents (Copilot, Claude Code, etc.).
> For human-oriented docs, see [README.md](README.md).

## What is looplet?

A composable tool-calling loop for LLM agents. You own the loop as a
Python iterator (`for step in composable_loop(...)`) and inject behavior
via hook protocols. Zero runtime dependencies. Provider-agnostic.

## Design principles

Keep looplet minimal, simple, powerful, and familiar to Python users.

- **Minimal core:** `composable_loop(...)` stays small and domain-neutral.
    New capabilities should compile into tools, hooks, config, state,
    presets, skills, bundles, CLI helpers, or docs unless the loop itself
    truly needs to know.
- **Simple story:** the LLM proposes a tool call; the registry validates
    and dispatches it; hooks observe or steer; state records the step; the
    loop yields a `Step` to the caller. Use this as the teaching model and
    the design test for new features.
- **Powerful composition:** grow power by combining ordinary looplet
    primitives, not by hiding orchestration inside a closed monolith.
- **Familiar Python:** prefer functions, classes, dataclasses, protocols,
    iterators, and importable factories. Avoid DSLs, magic globals,
    mandatory inheritance, and dependency-heavy plugin systems.
- **Layered cartridges:** skills and cartridges are packaging and
    distribution layers over looplet primitives, not a second runtime.
    Preserve the rule that everything compiles into the core loop story.
- **Honest conversion:** exact wrappers and blueprint comparisons are
    reliable; expanded source generation should depend on explicit recorded
    recipes, not decompiling arbitrary Python.

## Architecture (30-second version)

```
composable_loop(llm, tools, state, config, hooks)
  │
  ├─ 1. pre_prompt hooks → briefing text
  ├─ 2. Build prompt (task + tools + memory + history + briefing)
  ├─ 3. LLM call (with retry, cancellation, continuation)
  ├─ 4. Parse response → ToolCall (native tool_use or JSON text)
  ├─ 5. Permission check → pre_dispatch hooks
  ├─ 6. Tool dispatch → ToolResult (with timing, error classification)
  ├─ 7. post_dispatch hooks → inject follow-up context
  ├─ 8. check_done / should_stop hooks
  └─ 9. yield Step(number, tool_call, tool_result)
       → repeat until done/budget/stop
```

## Key modules

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `loop` | Core loop engine | `composable_loop`, `LoopConfig`, `LoopHook`, `DomainAdapter` |
| `types` | Data types & protocols | `Step`, `ToolCall`, `ToolResult`, `LLMBackend`, `DefaultState`, `ToolContext`, `CancelToken` |
| `tools` | Tool schema inference, registry & dispatch | `@tool`, `tools_from`, `BaseToolRegistry`, `ToolSpec` |
| `backends` | LLM adapters | `OpenAIBackend`, `AnthropicBackend`, `AsyncOpenAIBackend` |
| `permissions` | Declarative permission engine | `PermissionEngine`, `PermissionHook`, `PermissionRule` |
| `compact` | Context management | `compact_chain`, `PruneToolResults`, `SummarizeCompact`, `TruncateCompact` |
| `checkpoint` | Crash-resume | `FileCheckpointStore`, `resume_loop_state` |
| `hooks` | Hook decisions | `HookDecision`, `InjectContext`, `Allow`, `Deny`, `Block`, `Stop` |
| `skills` | Composable bundles | `Skill` |
| `subagent` | Sub-agent spawning | `run_sub_loop`, `clone_tools_excluding` |
| `mcp` | MCP server adapter | `MCPToolAdapter` |
| `evals` | Evaluation system | `EvalHook`, `EvalContext`, `eval_discover`, `eval_run` |
| `provenance` | Trajectory recording | `ProvenanceSink`, `TrajectoryRecorder` |
| `testing` | Mock backends | `MockLLMBackend`, `AsyncMockLLMBackend` |
| `presets` | One-liner agent setup | `coding_agent_preset`, `research_agent_preset`, `minimal_preset` |
| `memory` | Persistent memory | `StaticMemorySource`, `CallableMemorySource` |
| `budget` | Context budgets | `ContextBudget`, `ThresholdCompactHook` |
| `router` | Multi-model routing | `ModelRouter`, `SimpleRouter`, `RoutingLLMBackend` |
| `streaming` | Event emitters | `StreamingHook`, `EventEmitter`, `CallbackEmitter` |
| `validation` | Schema enforcement | `ValidatingToolRegistry`, `OutputSchema` |

## Recipe 1 — Minimal agent (5 lines)

```python
from looplet import composable_loop, LoopConfig, DefaultState, tool, tools_from

@tool
def greet(*, name: str) -> dict:
    """Greet someone."""
    return {"greeting": f"Hello, {name}!"}

tools = tools_from([greet], include_done=True, done_parameters={"answer": "Final answer"})

for step in composable_loop(
    llm=my_llm, tools=tools, state=DefaultState(max_steps=5),
    config=LoopConfig(max_steps=5), task={"goal": "Greet Alice, then finish."},
):
    print(step.pretty())
```

## Recipe 2 — Coding agent with presets

```python
from looplet.presets import coding_agent_preset

preset = coding_agent_preset(workspace="/path/to/project")

for step in composable_loop(
    llm=my_llm,
    tools=preset.tools,
    state=preset.state,
    config=preset.config,
    hooks=preset.hooks,
    task={"description": "Implement a REST API with tests"},
):
    print(step.pretty())
```

## Recipe 3 — Custom coding agent (full control)

```python
from looplet import (
    composable_loop, LoopConfig, DefaultState, tool, tools_from,
    HookDecision, InjectContext, StaticMemorySource,
    compact_chain, PruneToolResults, SummarizeCompact, TruncateCompact,
    ContextBudget, ThresholdCompactHook, EvalHook,
)

# 1. Define tools
@tool(description="Run a shell command.")
def bash(*, command: str) -> dict:
    return my_bash_fn(command=command)

@tool(description="Read a file.")
def read(*, file_path: str) -> dict:
    return my_read_fn(file_path=file_path)

@tool(description="Write a file.")
def write(*, file_path: str, content: str) -> dict:
    return my_write_fn(file_path=file_path, content=content)

tools = tools_from(
    [bash, read, write],
    include_done=True,
    done_parameters={"summary": "Completion summary"},
)

# 2. Write a hook (implement only the methods you need)
class MyHook:
    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool == "write" and "test_" not in tool_call.args.get("file_path", ""):
            return InjectContext("Write tests before calling done().")
        return None

    def check_done(self, state, session_log, context, step_num):
        return None  # or HookDecision(block="reason") to block

    def should_stop(self, state, step_num, new_entities):
        return False

# 3. Configure
config = LoopConfig(
    max_steps=20,
    system_prompt="You are a Python developer. Use bash for tests.",
    compact_service=compact_chain(
        PruneToolResults(keep_recent=5),
        SummarizeCompact(keep_recent=2),
        TruncateCompact(keep_recent=1),
    ),
    memory_sources=[StaticMemorySource("Always write tests first.")],
)

# 4. Run
for step in composable_loop(
    llm=my_llm, tools=tools, state=DefaultState(max_steps=20),
    config=config, hooks=[MyHook(), ThresholdCompactHook(ContextBudget(context_window=128_000))],
    task={"description": "Build a fibonacci module with tests"},
):
    print(step.pretty())
```

## Recipe 4 — Add a tool

```python
from looplet import tool, tools_from

@tool(description="Search for items by query. Returns results list.", concurrent_safe=True)
def my_tool(*, query: str, limit: int = 10) -> dict:
    """Search for items. Returns {"results": [...], "total": int}."""
    results = do_search(query, limit)
    return {"results": results, "total": len(results)}

tools = tools_from([my_tool])
```

## Recipe 5 — Write a hook

Hooks are `@runtime_checkable` Protocols. Implement only the methods you need:

```python
class SecurityGuard:
    """Block dangerous shell commands."""

    def pre_dispatch(self, state, session_log, tool_call, step_num):
        if tool_call.tool == "bash":
            cmd = tool_call.args.get("command", "")
            if any(danger in cmd for danger in ["rm -rf", "sudo", "> /dev/"]):
                from looplet import ToolResult
                return ToolResult(
                    tool="bash", args_summary=cmd[:50],
                    data=None, error="Blocked: dangerous command",
                )
        return None  # allow

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        return None  # no injection

    def check_done(self, state, session_log, context, step_num):
        return None  # allow done

    def should_stop(self, state, step_num, new_entities):
        return False
```

## Recipe 6 — Test without a real LLM

```python
from looplet import composable_loop, LoopConfig, DefaultState, MockLLMBackend, tool, tools_from

def test_my_agent():
    llm = MockLLMBackend(responses=[
        '{"tool": "bash", "args": {"command": "echo hello"}, "reasoning": "test"}',
        '{"tool": "done", "args": {"summary": "done"}, "reasoning": "finished"}',
    ])

    @tool(description="Run command")
    def bash(*, command: str) -> dict:
        return {"stdout": "hello", "exit_code": 0}

    tools = tools_from([bash], include_done=True, done_parameters={"summary": "Summary"})

    steps = list(composable_loop(
        llm=llm, tools=tools, state=DefaultState(max_steps=5),
        config=LoopConfig(max_steps=5), task={"goal": "run echo"},
    ))

    assert len(steps) == 2
    assert steps[0].tool_call.tool == "bash"
    assert steps[1].tool_call.tool == "done"
    assert steps[0].tool_result.error is None
```

## Recipe 7 — MCP server tools

```python
from looplet import BaseToolRegistry
from looplet.mcp import MCPToolAdapter

tools = BaseToolRegistry()

with MCPToolAdapter("npx @modelcontextprotocol/server-filesystem /tmp") as mcp:
    mcp.register_all(tools)
    # Now tools has read_file, write_file, list_directory, etc.
    # Use in composable_loop as normal
```

## Recipe 8 — Sub-agent for focused task

```python
from looplet.subagent import run_sub_loop

result = run_sub_loop(
    llm=my_llm,
    task={"goal": "Review this code for security issues", "code": file_content},
    tools=my_tools,
    max_steps=5,
    system_prompt="You are a security auditor. Find vulnerabilities.",
)
print(result["summary"])   # concise finding
print(result["findings"])  # list of issues found
```

## Recipe 8b — Compose agents as tools

Any agent can be exposed to another agent as a normal tool:

```python
from looplet import tool, tools_from
from looplet.subagent import run_sub_loop

@tool(description="Run a deep-research sub-agent and return a cited summary.")
def deep_research(question: str) -> str:
    """Multi-step research. Returns a cited summary."""
    result = run_sub_loop(
        llm=cheap_llm, tools=research_tools,
        task={"question": question},
        system_prompt="Research deeply. Cite every claim.",
        max_steps=20,
    )
    return result["summary"]

tools = tools_from([deep_research])
# Parent loop now sees `deep_research` as a regular tool.
```

The parent's `ProvenanceSink` will record only the parent trajectory;
wire a separate sink inside the sub-agent for parent-child linked traces.

## Recipe 9 — Permissions

```python
from looplet import PermissionEngine, PermissionHook, PermissionDecision

engine = PermissionEngine(default=PermissionDecision.ALLOW)
engine.allow("read", reason="safe read operation")
engine.allow("glob", reason="safe file listing")
engine.deny("bash",
            arg_matcher=lambda a: "rm " in a.get("command", ""),
            reason="destructive shell command")
engine.ask("write", reason="file modification needs review")

# Use as a hook:
hooks = [PermissionHook(engine)]
```

## Recipe 10 — Crash-resume with checkpoints

```python
config = LoopConfig(
    max_steps=50,
    checkpoint_dir="./checkpoints",  # auto-save every step, auto-resume on restart
)
# If the process crashes, restart the same script — it resumes from last checkpoint.
```

## Recipe 11 — PII redaction + trajectory capture

```python
import re
from looplet import ProvenanceSink, composable_loop

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

def scrub(s: str) -> str:
    return EMAIL_RE.sub("[EMAIL]", s)

sink = ProvenanceSink(dir="traces/run_1", redact=scrub)
llm  = sink.wrap_llm(AnthropicBackend(...))     # PII scrubbed before Anthropic sees it

for step in composable_loop(llm=llm, tools=tools, hooks=[sink.trajectory_hook()], ...):
    print(step.pretty())
sink.flush()
```

**By default `redact` scrubs upstream too:** secrets never reach the
provider or the trace file. Pass `redact_upstream=False` to get the
legacy record-only behaviour.

## Recipe 12 — Budget-capped loop with meaningful stop reason

```python
from looplet import HookDecision, composable_loop

class TokenBudget:
    def __init__(self, cap: int) -> None:
        self.cap, self.total = cap, 0

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        usage = getattr(state.steps[-1], "usage", None)
        if usage is not None:
            self.total = getattr(usage, "total_tokens", self.total)
        return None

    def should_stop(self, state, step_num, new_entities):
        if self.total >= self.cap:
            return HookDecision(stop="budget_exceeded")   # shows up as ctx.stop_reason
        return False
```

Evaluators can then dispatch on the reason:

```python
def eval_finished_cleanly(ctx):
    return ctx.completed                     # shorthand for stop_reason == "done"

def eval_no_hard_timeout(ctx):
    return ctx.stop_reason != "timeout"
```

## Recipe 13 — Discovery-safe eval file

`eval_discover` only collects functions defined IN the eval file.
Imports (decorators, helpers, library functions) are filtered out by
`__module__`, so the following is safe:

```python
# eval_my_agent.py
from looplet import eval_mark              # decorator — not collected
from my_helpers import eval_tool_count     # helper from another module — not collected

@eval_mark("verdict")
def eval_correct_answer(ctx):              # collected
    return ctx.final_output.get("answer") == ctx.task.get("expected")
```

## Common patterns

### Error messages ARE prompts
Tool results with errors should include remediation steps. The LLM reads
these and self-corrects:
```python
return {
    "error": f"File not found: {path}",
    "remediation": "Use glob to find existing files, or write to create.",
}
```

### Just-in-time context injection
Don't frontload all rules in the system prompt. Inject them via
`post_dispatch` when they're actionable:
```python
def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
    if tool_call.tool == "write" and not self._has_tests:
        return InjectContext("Write tests before calling done().")
    return None
```

### Quality gates via check_done
Block premature completion:
```python
def check_done(self, state, session_log, context, step_num):
    if not self._tests_passed:
        return HookDecision(block="Tests must pass before done().")
    return None
```

### Persistent memory (survives compaction)
```python
config = LoopConfig(
    memory_sources=[StaticMemorySource("Always use type hints. Write tests first.")],
    compact_service=compact_chain(...),
)
```

## Pitfalls (read this before generating code)

These are the sharp edges coding agents hit most often. All have
principled fixes in the library; the notes below are the "right way."

1. **`LoopConfig.max_steps` and `DefaultState(max_steps=...)` must
   match.** The loop warns and syncs to the config value, but pass the
   same N to both to silence it:
   ```python
   N = 20
   config = LoopConfig(max_steps=N)
   state  = DefaultState(max_steps=N)
   ```

2. **`redact=` in `ProvenanceSink` / `RecordingLLMBackend` scrubs
   UPSTREAM by default.** Secrets never reach the provider OR the
   trace. Do NOT double-wrap the LLM in a separate redactor — pass the
   callable to the sink:
   ```python
   # ✓ do this
   sink = ProvenanceSink(dir="traces/", redact=scrub_pii)
   llm  = sink.wrap_llm(AnthropicBackend(...))
   # ✗ not this (record-only; PII still hits the provider)
   ```

3. **Use `HookDecision(stop="reason")` in `should_stop`**, not a bare
   `True`. The reason string becomes `EvalContext.stop_reason` and
   lets evaluators tell `"budget_exceeded"` apart from `"timeout"`.

4. **`eval_discover` only collects functions defined in the eval
   file.** Imported decorators and helpers are filtered by
   `__module__`. This is intentional — do not work around it by
   defining pass-through wrappers; just import normally.

5. **`should_stop` fires AFTER the current step**, so the last step in
   the trajectory may not be a `done()` call. Trajectory evaluators
   must handle this via `ctx.stop_reason` / `ctx.completed`, not by
   assuming a terminal `done()` exists.

6. **Tool results with errors should include remediation.** The LLM
   reads `tool_result.error` and `tool_result.data` verbatim. A good
   error includes "what went wrong" and "what to try next":
   ```python
   return {"error": "File not found: x.py",
           "remediation": "Use glob to list existing files."}
   ```

7. **Never swallow exceptions in hooks silently.** Let them propagate
   unless you have a specific recovery. A hook that eats
   `KeyError` can mask a missing `tool_call.args` key that should
   have surfaced as a prompt for the model.

8. **`composable_loop` is a generator.** The `for step in ...` pattern
   is mandatory — the loop does not run if you only call
   `composable_loop(...)`. Consume the iterator, or wrap with
   `list(...)` when you don't care about streaming.

9. **`NativeToolBackend.generate_with_tools`** is surfaced via
   `hasattr` on the wrapped backend. Recording/redacting wrappers
   preserve this automatically; custom wrappers must forward it or
   native tool-calling silently falls back to JSON parsing.

10. **Prefer Protocol-conforming classes over inheritance.** All hooks,
    LLM backends, and states are `@runtime_checkable` Protocols —
    any object with the right methods works. No `LoopHook`
    subclassing, no registration.

## Development commands

```bash
uv sync                       # install deps
uv run pytest                 # full suite (~1062 tests, ~1s)
uv run pytest -m smoke        # smoke tests only
uv run ruff check .           # lint
uv run ruff format --check .  # format check
uv run pyright src/looplet/   # type check
make check                    # all of the above (matches CI)
make install-hooks            # install pre-commit + pre-push git hooks
```

## File structure

```
src/looplet/
  __init__.py          # Public API — all exports here
  loop.py              # Core loop: composable_loop, LoopConfig, LoopHook
  types.py             # Step, ToolCall, ToolResult, LLMBackend, DefaultState
  tools.py             # BaseToolRegistry, ToolSpec
  backends.py          # OpenAI/Anthropic adapters
  permissions.py       # PermissionEngine, rules
  compact.py           # Context compaction strategies
  checkpoint.py        # Crash-resume
  hook_decision.py     # HookDecision, Allow/Deny/Block/Stop/InjectContext
  skills.py            # Skill bundles
  subagent.py          # run_sub_loop — sub-agents as tools
  mcp.py               # MCP server adapter
  evals.py             # Evaluation system + EvalContext.stop_reason
  provenance.py        # Trajectory recording + redact (scrubs upstream by default)
  testing.py           # MockLLMBackend
  presets.py           # One-liner presets (coding, research, minimal)
  memory.py            # Persistent memory sources
  budget.py            # Context budget management
  router.py            # Multi-model routing
  streaming.py         # Event emitters
  scaffolding.py       # LLM retry, truncation, recovery
  validation.py        # Schema enforcement
  prompts.py           # Prompt assembly
  conversation.py      # Message/conversation management
  session.py           # Session log
  recovery.py          # Error recovery registry
  recovery_strategies.py  # Built-in recovery handlers
  context.py           # Context pressure hook
  events.py            # Lifecycle event types
  flags.py             # Feature flags (env vars)
  parse.py             # LLM response parsing
  history.py           # Single-writer turn/step history
  approval.py          # ApprovalHook
  cache.py             # Prompt caching
  telemetry.py         # Tracer / MetricsCollector / MetricsHook / TracingHook
  examples/
    hello_world.py     # Minimal example
    coding_agent.py    # Production reference (bash/read/write/edit/glob/grep)
```

## Type contracts

```python
# The loop yields Step objects:
@dataclass
class Step:
    number: int           # 1-based step index
    tool_call: ToolCall   # what the LLM requested
    tool_result: ToolResult  # what the tool returned

@dataclass
class ToolCall:
    tool: str             # tool name
    args: dict[str, Any]  # keyword arguments
    reasoning: str        # LLM's reasoning
    call_id: str          # unique ID

@dataclass
class ToolResult:
    tool: str             # tool name
    args_summary: str     # compact arg summary
    data: Any             # raw output (dict, list, str, None)
    error: str | None     # error message or None
    error_detail: ToolError | None  # structured error
    duration_ms: float    # execution time

# LLM backends must implement:
class LLMBackend(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = 2000,
                 system_prompt: str = "", temperature: float = 0.2) -> str: ...

# For native tool calling, also implement:
class NativeToolBackend(Protocol):
    def generate_with_tools(self, prompt: str, *, tools: list[dict],
                            max_tokens: int = 2000, system_prompt: str = "",
                            temperature: float = 0.2) -> list[dict]: ...

# Agent state must satisfy:
class AgentState(Protocol):
    steps: list
    queries_used: int
    step_count: int       # property
    budget_remaining: int # property
    def context_summary(self) -> str: ...
    def snapshot(self) -> dict: ...
```

## Error taxonomy

Tools classify errors via `ErrorKind`:
- `PERMISSION_DENIED` — blocked by permission check
- `TIMEOUT` — execution exceeded deadline (retriable)
- `VALIDATION` — bad args or unknown tool
- `EXECUTION` — generic runtime failure
- `PARSE` — LLM response couldn't be parsed
- `CONTEXT_OVERFLOW` — prompt exceeded context window
- `RATE_LIMIT` — provider throttling (retriable)
- `NETWORK` — transport failure (retriable)
- `CANCELLED` — cancelled via CancelToken

## LoopConfig cheat sheet

`LoopConfig` has ~40 fields. Group them mentally as follows — most agents
only touch the first group.

**Essentials (always set these):**
`max_steps`, `system_prompt`, `temperature`, `done_tool`

**Behavior tuning (usually fine as-is):**
`max_tokens`, `recovery_temperature`, `max_turn_continuations`,
`concurrent_dispatch`, `reactive_recovery`, `use_native_tools`,
`context_window`, `max_briefing_tokens`, `acceptance_criteria`

**Domain hooks (bundle into `DomainAdapter` or set individually):**
`build_briefing`, `build_prompt`, `extract_entities`,
`extract_step_metadata`, `build_trace`, `domain`

**Wired-in capabilities (opt-in — each enables one feature):**
`compact_service` (compaction) · `checkpoint_dir` (crash-resume) ·
`cache_policy` (prompt caching) · `router` (multi-model) ·
`tracer` (telemetry) · `recovery_registry` (error recovery) ·
`output_schema` (done() validation) · `memory_sources` (persistent notes) ·
`approval_handler` (human-in-the-loop) · `cancel_token` (cooperative stop) ·
`initial_checkpoint` (resume a specific checkpoint)

**Escape hatches (rare — only when `build_prompt` isn't enough):**
`render_messages_override`

> **Footgun:** `LoopConfig(max_steps=N)` and `DefaultState(max_steps=M)`
> must match. The loop now warns and syncs to the config value, but you
> should still pass the same N to both.

## Canonical hook return values

Every hook method accepts a `HookDecision` (or one of its factory
helpers). Legacy returns (`str`, `bool`, raw `ToolResult`) still work
via `normalize_hook_return`, but new code should use the factory
helpers — they read naturally and compose:

| Intent | Use |
|---|---|
| Allow / no opinion | `return None` |
| Append text to next prompt | `return InjectContext("...")` |
| Block tool call or `done()` | `return Block("reason for the model")` |
| Deny permission | `return Deny("reason")` |
| Stop the loop cleanly | `return Stop("done-ish reason")` |
| Short-circuit with a cached result | `return HookDecision(updated_result=ToolResult(...))` |
| Rewrite the model's tool args | `return HookDecision(updated_args={"path": "..."})` |

Prefer the helpers (`Allow`, `Block`, `Deny`, `Stop`, `InjectContext`)
over bare `HookDecision(...)` for single-intent cases.

## Symbol index (A–Z)

Everything in `from looplet import X` is listed here. Submodule-only
symbols live in `looplet.<module>`.

| Symbol | Module | Purpose |
|---|---|---|
| `Allow` | `hook_decision` | Factory: allow / no opinion |
| `AgentPreset` | `presets` | Dataclass returned by preset fns |
| `AnthropicBackend` | `backends` | Claude LLM adapter |
| `ApprovalHook` | `approval` | Pauses loop for external approval |
| `AsyncMockLLMBackend` | `testing` | Async scripted LLM for tests |
| `BaseToolRegistry` | `tools` | Tool registry + dispatch |
| `Block` | `hook_decision` | Factory: block tool call / done() |
| `CachePolicy` | `cache` | Prompt-caching config |
| `CallableMemorySource` | `memory` | Memory from a callable |
| `CancelToken` | `types` | Cooperative cancellation signal |
| `CompactOutcome` | `compact` | Compaction result record |
| `CompactService` | `compact` | Compaction strategy protocol |
| `ContextBudget` | `budget` | Context window thresholds |
| `Continue` | `hook_decision` | Factory: explicit no-op |
| `Conversation` | `conversation` | Message thread container |
| `DefaultState` | `types` | Built-in `AgentState` impl |
| `Deny` | `hook_decision` | Factory: deny permission |
| `DomainAdapter` | `loop` | Bundle domain callables |
| `ErrorKind` | `types` | Error discriminator enum |
| `EvalContext` | `evals` | Eval run context |
| `EvalHook` | `evals` | Hook for eval scoring |
| `EvalResult` | `evals` | Eval result record |
| `EventPayload` | `events` | Lifecycle event payload |
| `FileCheckpointStore` | `checkpoint` | Disk-backed crash-resume |
| `HookDecision` | `hook_decision` | Unified hook return type |
| `InjectContext` | `hook_decision` | Factory: append text to next prompt |
| `LLMBackend` | `types` | Sync LLM protocol |
| `LifecycleEvent` | `events` | Event name enum |
| `LoopConfig` | `loop` | Loop configuration dataclass |
| `LoopHook` | `loop` | Hook protocol |
| `MCPToolAdapter` | `mcp` | Bridges MCP tools into registry |
| `Message` | `conversation` | Single conversation message |
| `MetricsCollector` | `telemetry` | Metrics backend protocol |
| `MetricsHook` | `telemetry` | Emits metrics during loop |
| `MockLLMBackend` | `testing` | Scripted sync LLM for tests |
| `NativeToolBackend` | `types` | Protocol for native-tool backends |
| `OpenAIBackend` | `backends` | OpenAI LLM adapter |
| `PermissionDecision` | `permissions` | Enum: ALLOW/DENY/ASK/DEFAULT |
| `PermissionEngine` | `permissions` | Declarative permission engine |
| `PermissionHook` | `permissions` | Wraps engine as a hook |
| `PermissionRule` | `permissions` | Single allow/deny/ask rule |
| `ProvenanceSink` | `provenance` | Trajectory recording sink |
| `PruneToolResults` | `compact` | Compaction: drop old results |
| `SessionLog` | `session` | Append-only event log |
| `Skill` | `skills` | Bundle of hooks + tools |
| `StaticMemorySource` | `memory` | Fixed-text memory source |
| `Step` | `types` | Yielded by `composable_loop` |
| `Stop` | `hook_decision` | Factory: stop loop after step |
| `StreamingHook` | `streaming` | Emits lifecycle events |
| `SummarizeCompact` | `compact` | Compaction: summarize via LLM |
| `ThresholdCompactHook` | `budget` | Trigger compact at % window |
| `ToolCall` | `types` | LLM-requested tool call |
| `ToolContext` | `types` | Runtime ctx threaded to tools |
| `ToolError` | `types` | Structured tool error |
| `ToolResult` | `types` | Tool execution result |
| `ToolSpec` | `tools` | Tool definition |
| `Tracer` | `telemetry` | Tracing backend protocol |
| `TracingHook` | `telemetry` | Emits spans during loop |
| `TrajectoryRecorder` | `provenance` | Records trajectories to disk |
| `TruncateCompact` | `compact` | Compaction: drop oldest |
| `coding_agent_preset` | `presets` | Pre-built coding agent |
| `compact_chain` | `compact` | Chain compaction strategies |
| `composable_loop` | `loop` | **The loop generator** |
| `emit_event` | `loop` | Fire a `LifecycleEvent` to hooks |
| `eval_cli` | `evals` | CLI entry for evals |
| `eval_discover` | `evals` | Find eval scenarios |
| `eval_mark` | `evals` | Mark eval scenario outcome |
| `eval_run` | `evals` | Run one eval scenario |
| `eval_run_batch` | `evals` | Run many eval scenarios |
| `minimal_preset` | `presets` | Bare-bones agent preset |
| `preview_prompt` | `prompts` | Debug: render next prompt |
| `replay_loop` | `provenance` | Replay a recorded trajectory |
| `research_agent_preset` | `presets` | Pre-built research agent |
| `run_compact` | `compact` | Invoke a compact service |
| `run_sub_loop` | `subagent` | Spawn a sub-agent loop |

