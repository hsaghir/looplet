# openharness — Agent Guide

> This file is optimized for coding agents (Copilot, Claude Code, etc.).
> For human-oriented docs, see [README.md](README.md).

## What is openharness?

A composable tool-calling loop for LLM agents. You own the loop as a
Python iterator (`for step in composable_loop(...)`) and inject behavior
via hook protocols. One runtime dependency (PyYAML). Provider-agnostic.

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
| `tools` | Tool registry & dispatch | `BaseToolRegistry`, `ToolSpec`, `register_think_tool` |
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
from openharness import composable_loop, LoopConfig, DefaultState, BaseToolRegistry, ToolSpec

tools = BaseToolRegistry()
tools.register(ToolSpec(name="greet", description="Greet someone",
                        parameters={"name": "str"},
                        execute=lambda *, name: {"greeting": f"Hello, {name}!"}))
tools.register(ToolSpec(name="done", description="Finish",
                        parameters={"answer": "str"},
                        execute=lambda *, answer: {"answer": answer}))

for step in composable_loop(
    llm=my_llm, tools=tools, state=DefaultState(max_steps=5),
    config=LoopConfig(max_steps=5), task={"goal": "Greet Alice, then finish."},
):
    print(step.pretty())
```

## Recipe 2 — Coding agent with presets

```python
from openharness.presets import coding_agent_preset

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
from openharness import (
    composable_loop, LoopConfig, DefaultState, BaseToolRegistry, ToolSpec,
    HookDecision, InjectContext, StaticMemorySource,
    compact_chain, PruneToolResults, SummarizeCompact, TruncateCompact,
    ContextBudget, ThresholdCompactHook, EvalHook,
)

# 1. Register tools
tools = BaseToolRegistry()
tools.register(ToolSpec(name="bash", description="Run shell command",
                        parameters={"command": "str"},
                        execute=my_bash_fn))
tools.register(ToolSpec(name="read", description="Read file",
                        parameters={"file_path": "str"},
                        execute=my_read_fn))
tools.register(ToolSpec(name="write", description="Write file",
                        parameters={"file_path": "str", "content": "str"},
                        execute=my_write_fn))
tools.register(ToolSpec(name="done", description="Signal completion",
                        parameters={"summary": "str"},
                        execute=lambda *, summary: {"summary": summary}))

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
from openharness import BaseToolRegistry, ToolSpec

def my_tool(*, query: str, limit: int = 10) -> dict:
    """Search for items. Returns {"results": [...], "total": int}."""
    results = do_search(query, limit)
    return {"results": results, "total": len(results)}

tools = BaseToolRegistry()
tools.register(ToolSpec(
    name="search",
    description="Search for items by query. Returns results list.",
    parameters={"query": "str", "limit": "int (default 10)"},
    execute=my_tool,
    concurrent_safe=True,  # safe to run in parallel with other tools
))
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
                from openharness import ToolResult
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
from openharness import composable_loop, LoopConfig, DefaultState, BaseToolRegistry, ToolSpec
from openharness.testing import MockLLMBackend

def test_my_agent():
    llm = MockLLMBackend(responses=[
        '{"tool": "bash", "args": {"command": "echo hello"}, "reasoning": "test"}',
        '{"tool": "done", "args": {"summary": "done"}, "reasoning": "finished"}',
    ])
    tools = BaseToolRegistry()
    tools.register(ToolSpec(name="bash", description="Run command",
                            parameters={"command": "str"},
                            execute=lambda *, command: {"stdout": "hello", "exit_code": 0}))
    tools.register(ToolSpec(name="done", description="Finish",
                            parameters={"summary": "str"},
                            execute=lambda *, summary: {"summary": summary}))

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
from openharness import BaseToolRegistry
from openharness.mcp import MCPToolAdapter

tools = BaseToolRegistry()

with MCPToolAdapter("npx @modelcontextprotocol/server-filesystem /tmp") as mcp:
    mcp.register_all(tools)
    # Now tools has read_file, write_file, list_directory, etc.
    # Use in composable_loop as normal
```

## Recipe 8 — Sub-agent for focused task

```python
from openharness.subagent import run_sub_loop

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

## Recipe 9 — Permissions

```python
from openharness import PermissionEngine, PermissionHook, PermissionDecision

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

## Development commands

```bash
uv sync                       # install deps
uv run pytest                 # full suite (~1008 tests, ~4s)
uv run pytest -m smoke        # smoke tests only
uv run ruff check .           # lint
uv run ruff format --check .  # format check
uv run mypy src/openharness   # type check
```

## File structure

```
src/openharness/
  __init__.py          # Public API — all exports here
  loop.py              # Core loop: composable_loop, LoopConfig, LoopHook
  types.py             # Step, ToolCall, ToolResult, LLMBackend, DefaultState
  tools.py             # BaseToolRegistry, ToolSpec
  backends.py          # OpenAI/Anthropic adapters
  permissions.py       # PermissionEngine, rules
  compact.py           # Context compaction strategies
  checkpoint.py        # Crash-resume
  hooks.py             # HookDecision, InjectContext
  skills.py            # Skill bundles
  subagent.py          # Sub-agent spawning
  mcp.py               # MCP server adapter
  evals.py             # Evaluation system
  provenance.py        # Trajectory recording
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
  context.py           # Context pressure hook
  events.py            # Lifecycle event types
  flags.py             # Feature flags (env vars)
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
