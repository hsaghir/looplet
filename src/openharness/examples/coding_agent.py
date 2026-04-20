"""Coding agent — a complete example of building with openharness.

Demonstrates how a well-structured agent harness works:

1. **Native tool calling** — uses the provider's native tool protocol
   (OpenAI function calling, Anthropic tool_use) for reliable dispatch.
2. **Just-in-time context** — don't frontload all instructions. Surface
   requirements when they matter: after the agent writes code, remind
   it about tests. After tests fail, inject targeted fix guidance.
3. **Run to completion** — the agent keeps going until the job is done.
   ``should_stop`` returns False; ``check_done`` blocks until tests pass.
4. **Error messages are prompts** — tool results include actionable
   remediation steps, so the agent self-corrects without human input.
5. **Persistent standards** — coding standards are injected via
   ``StaticMemorySource`` and survive all compactions.
6. **Compaction chain** — for long sessions: prune old tool results
   first (free), then LLM-summarize (one call), then truncate (last resort).

This agent:
  - Receives a task ("implement a fibonacci function with tests")
  - Uses the same tools as Claude Code: bash, read, write, edit, glob, grep, think
  - Gets just-in-time review feedback via a ``post_dispatch`` hook
  - Keeps running until tests pass (``check_done`` quality gate)
  - Has persistent memory (coding standards survive compaction)
  - Uses ``compact_chain`` for context management in long sessions

Run::

    python -m openharness.examples.coding_agent
    python -m openharness.examples.coding_agent "implement a linked list" --model claude-sonnet-4
    python -m openharness.examples.coding_agent --base-url https://api.openai.com/v1 --model gpt-4o
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

from openharness import (
    BaseToolRegistry,
    ContextBudget,
    DefaultState,
    DomainAdapter,
    EvalContext,
    EvalHook,
    HookDecision,
    InjectContext,
    LoopConfig,
    PruneToolResults,
    StaticMemorySource,
    SummarizeCompact,
    ThresholdCompactHook,
    TruncateCompact,
    compact_chain,
    composable_loop,
)
from openharness.session import SessionLog
from openharness.tools import ToolSpec
from openharness.types import ToolCall, ToolResult

# ═══════════════════════════════════════════════════════════════════
# 1. TOOLS — same core set as Claude Code / Claude Agent SDK
#
# bash, read, write, edit, glob, grep, think, done
#
# Each tool has rich error messages with remediation steps.
# The model sees these messages and self-corrects — error messages
# ARE prompts.
# ═══════════════════════════════════════════════════════════════════


def _bash(*, command: str, workspace: str = "") -> dict:
    """Execute a bash command in the workspace directory.

    This is the most powerful tool — it can run pytest, install
    packages, search files, compile code, and do anything the
    shell can do. Prefer this over specialized tools when the
    task is naturally expressed as a shell command.
    """
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=60,
            cwd=workspace or None,
        )
        output = result.stdout.strip()
        err = result.stderr.strip()
        # Truncate to keep context lean
        combined = output + ("\n" + err if err else "")
        if len(combined) > 4000:
            combined = combined[:2000] + "\n...(truncated)...\n" + combined[-2000:]
        return {
            "stdout": output[:4000] if output else "",
            "stderr": err[:2000] if err else "",
            "exit_code": result.returncode,
            "next_step": (
                "Command succeeded."
                if result.returncode == 0 else
                f"Command failed (exit {result.returncode}). "
                "Read stderr above and fix the issue."
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "error": "Command timed out after 60 seconds",
            "remediation": "Check for infinite loops or blocking I/O.",
        }


def _read(*, file_path: str, workspace: str = "") -> dict:
    """Read a file and return its content with line numbers."""
    full = os.path.join(workspace, file_path) if workspace else file_path
    try:
        with open(full) as f:
            lines = f.readlines()
        numbered = "".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        if len(numbered) > 8000:
            numbered = numbered[:4000] + "\n...(truncated)...\n" + numbered[-4000:]
        return {"path": file_path, "content": numbered, "line_count": len(lines)}
    except FileNotFoundError:
        return {
            "error": f"File not found: {file_path}",
            "remediation": (
                "The file does not exist yet. Use write to create it, "
                "or glob to find existing files."
            ),
        }


def _write(*, file_path: str, content: str, workspace: str = "") -> dict:
    """Create or overwrite a file with the given content."""
    full = os.path.join(workspace, file_path) if workspace else file_path
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return {"written": file_path, "lines": content.count("\n") + 1}


def _edit(*, file_path: str, old_string: str, new_string: str,
          workspace: str = "") -> dict:
    """Edit a file by replacing an exact string match.

    Use this for targeted fixes — change one function, fix one line.
    For large rewrites, use write instead.
    """
    full = os.path.join(workspace, file_path) if workspace else file_path
    try:
        text = open(full).read()
    except FileNotFoundError:
        return {
            "error": f"File not found: {file_path}",
            "remediation": "Use write to create the file first.",
        }
    count = text.count(old_string)
    if count == 0:
        return {
            "error": "old_string not found in file",
            "remediation": (
                "The exact text was not found. Use read to see the "
                "current file content, then provide the exact string "
                "to replace (including whitespace and indentation)."
            ),
        }
    if count > 1:
        return {
            "error": f"old_string matches {count} locations — ambiguous",
            "remediation": "Include more surrounding context in old_string to make it unique.",
        }
    with open(full, "w") as f:
        f.write(text.replace(old_string, new_string, 1))
    return {"edited": file_path, "replacements": 1}


def _glob(*, pattern: str, workspace: str = "") -> dict:
    """Find files matching a glob pattern (e.g. '**/*.py', 'test_*.py')."""
    import glob as globmod
    base = workspace or "."
    matches = sorted(globmod.glob(os.path.join(base, pattern), recursive=True))
    # Return relative paths
    rel = [os.path.relpath(m, base) for m in matches]
    if len(rel) > 100:
        rel = rel[:100]
        return {"pattern": pattern, "files": rel, "truncated": True, "total": len(matches)}
    return {"pattern": pattern, "files": rel, "total": len(rel)}


def _grep(*, pattern: str, path: str = ".", workspace: str = "") -> dict:
    """Search file contents with a regex pattern (like ripgrep)."""
    full = os.path.join(workspace, path) if workspace else path
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, full],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) > 50:
            lines = lines[:50]
            return {"pattern": pattern, "matches": lines, "truncated": True}
        return {"pattern": pattern, "matches": lines, "total": len(lines)}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "remediation": "Narrow your pattern or path."}


def build_tools(workspace: str) -> BaseToolRegistry:
    """Build the tool registry with Claude Code-equivalent tools.

    Core set: bash, read, write, edit, glob, grep, think, done.
    """
    reg = BaseToolRegistry()

    # Bind workspace into each tool
    reg.register(ToolSpec(
        name="bash",
        description=(
            "Execute a bash command. The most powerful tool — use it for "
            "running tests (python -m pytest), installing packages, "
            "searching, compiling, and any shell operation."
        ),
        parameters={"command": "str"},
        execute=lambda *, command: _bash(command=command, workspace=workspace),
    ))
    reg.register(ToolSpec(
        name="read",
        description="Read a file with line numbers. Use relative paths.",
        parameters={"file_path": "str"},
        execute=lambda *, file_path: _read(file_path=file_path, workspace=workspace),
    ))
    reg.register(ToolSpec(
        name="write",
        description="Create or overwrite a file. Use relative paths.",
        parameters={"file_path": "str", "content": "str"},
        execute=lambda *, file_path, content: _write(
            file_path=file_path, content=content, workspace=workspace),
    ))
    reg.register(ToolSpec(
        name="edit",
        description=(
            "Edit a file by replacing an exact string. For targeted fixes — "
            "provide the exact text to find and the replacement. Use read "
            "first to see the current content."
        ),
        parameters={"file_path": "str", "old_string": "str", "new_string": "str"},
        execute=lambda *, file_path, old_string, new_string: _edit(
            file_path=file_path, old_string=old_string, new_string=new_string,
            workspace=workspace),
    ))
    reg.register(ToolSpec(
        name="glob",
        description="Find files by glob pattern (e.g. '**/*.py', 'test_*.py').",
        parameters={"pattern": "str"},
        execute=lambda *, pattern: _glob(pattern=pattern, workspace=workspace),
    ))
    reg.register(ToolSpec(
        name="grep",
        description="Search file contents with regex (like ripgrep). Returns matching lines with file:line.",
        parameters={"pattern": "str", "path": "str"},
        execute=lambda *, pattern, path=".": _grep(
            pattern=pattern, path=path, workspace=workspace),
    ))
    from openharness.tools import register_think_tool
    register_think_tool(reg)
    reg.register(ToolSpec(
        name="done",
        description="Signal task completion. Only call when ALL tests pass.",
        parameters={"summary": "str"},
        execute=lambda *, summary: {"summary": summary},
    ))
    return reg


# ═══════════════════════════════════════════════════════════════════
# 2. HOOKS — surface instructions at the right time
#
# Don't frontload all requirements. Inject context just-in-time:
# - After each tool call (post_dispatch): review feedback
# - At quality gate (check_done): block done() until tests pass
# - At step start (should_stop): only stop when done, never early
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CodingGuardrailHook:
    """Just-in-time guardrails for a coding agent.

    Demonstrates three patterns:
    1. post_dispatch injects review feedback AFTER the agent writes code
    2. check_done blocks completion until tests actually pass
    3. should_stop keeps the agent running until the job is done
    """

    _tests_passed: bool = False
    _files_written: set = field(default_factory=set)
    _has_tests: bool = False

    def post_dispatch(
        self, state: Any, session_log: SessionLog,
        tool_call: ToolCall, tool_result: ToolResult, step_num: int,
    ) -> HookDecision | None:
        """Just-in-time context injection after each tool call.

        This is the key harness engineering pattern: don't frontload
        requirements. Instead, surface them when they're actionable.
        """
        if tool_call.tool == "write":
            path = tool_call.args.get("file_path", "")
            self._files_written.add(path)
            if path.startswith("test_") or "/test_" in path:
                self._has_tests = True

            # Just-in-time review: after writing code, remind about tests
            if not path.startswith("test_") and not self._has_tests:
                return InjectContext(
                    "You wrote source code but haven't written tests yet. "
                    "Write tests BEFORE running them. Every function needs "
                    "at least one test."
                )

        if tool_call.tool == "bash":
            cmd = tool_call.args.get("command", "")
            data = tool_result.data or {}
            if "pytest" in cmd or "python -m pytest" in cmd:
                self._tests_passed = data.get("exit_code", 1) == 0
                if not self._tests_passed:
                    return InjectContext(
                        "Tests failed. Read the pytest output carefully. "
                        "Fix the EXACT issue described, then run tests again. "
                        "Do not rewrite everything — make targeted fixes."
                    )

        return None

    def check_done(
        self, state: Any, session_log: SessionLog,
        context: Any, step_num: int,
    ) -> HookDecision | None:
        """Quality gate: block done() until tests pass.

        The harness enforces completion criteria so no human needs
        to intervene — the agent runs to completion autonomously.
        """
        if not self._tests_passed:
            return HookDecision(
                block="Cannot complete: tests have not passed yet. "
                "Run run_tests first and fix any failures before calling done()."
            )
        return None

    def should_stop(self, state: Any, step_num: int, new_entities: int) -> bool:
        """Never stop early — let the agent run to completion."""
        return False

    # Protocol stubs
    def pre_loop(self, *a: Any, **k: Any) -> None: return None
    def pre_prompt(self, *a: Any, **k: Any) -> None: return None
    def pre_dispatch(self, *a: Any, **k: Any) -> None: return None
    def check_permission(self, *a: Any, **k: Any) -> None: return None
    def should_compact(self, *a: Any, **k: Any) -> bool: return False
    def build_briefing(self, *a: Any, **k: Any) -> None: return None
    def build_prompt(self, **k: Any) -> None: return None
    def on_loop_end(self, *a: Any, **k: Any) -> int: return 0
    def on_event(self, *a: Any, **k: Any) -> None: return None


# ═══════════════════════════════════════════════════════════════════
# 3. PERSISTENT MEMORY — coding standards
#
# These survive all compactions. The agent sees them on EVERY
# turn, even after context is compressed.
# ═══════════════════════════════════════════════════════════════════

CODING_STANDARDS = StaticMemorySource("""\
## Coding Standards (persistent — survives compaction)

1. **Tests first**: Write test file before or alongside implementation.
2. **Small files**: Each file under 200 lines. Split if larger.
3. **Type hints**: All function signatures must have type annotations.
4. **Docstrings**: Every public function needs a one-line docstring.
5. **No bare except**: Always catch specific exception types.
6. **Naming**: snake_case for functions/variables, PascalCase for classes.
""")


# ═══════════════════════════════════════════════════════════════════
# 4. COMPACTION CHAIN — cheap first, then LLM, then truncate
#
# Long sessions need context management. The chain tries the
# cheapest strategy first:
#   1. PruneToolResults — clear old tool output (free)
#   2. SummarizeCompact — LLM summary of middle (one call)
#   3. TruncateCompact — drop everything except last 2 (free, last resort)
# ═══════════════════════════════════════════════════════════════════

COMPACT_SERVICE = compact_chain(
    PruneToolResults(keep_recent=5),
    SummarizeCompact(keep_recent=2),
    TruncateCompact(keep_recent=1),
)


# ═══════════════════════════════════════════════════════════════════
# 5. DOMAIN ADAPTER — Bundle domain callables in one object
# ═══════════════════════════════════════════════════════════════════

def _build_briefing(state: Any, session_log: SessionLog, context: Any) -> str:
    """Dynamic briefing — surfaces progress summary each turn."""
    steps = getattr(state, "steps", [])
    if not steps:
        return "No work done yet. Start by understanding the task, then write code and tests."

    wrote_files = [s.tool_call.args.get("file_path", "?") for s in steps
                   if s.tool_call and s.tool_call.tool == "write"]
    ran_tests = any(
        s.tool_call.tool == "bash" and "pytest" in s.tool_call.args.get("command", "")
        for s in steps if s.tool_call
    )
    last_test_passed = False
    for s in reversed(steps):
        if s.tool_call and s.tool_call.tool == "bash" and "pytest" in s.tool_call.args.get("command", ""):
            last_test_passed = (s.tool_result.data or {}).get("exit_code", 1) == 0
            break

    lines = [f"Progress: {len(steps)} steps taken."]
    if wrote_files:
        lines.append(f"Files written: {', '.join(wrote_files)}")
    if ran_tests:
        lines.append(f"Last test run: {'PASSED ✓' if last_test_passed else 'FAILED ✗'}")
    if not ran_tests:
        lines.append("WARNING: You haven't run tests yet. Do that before calling done().")
    return "\n".join(lines)


def _build_prompt(*, use_native: bool = False, **kwargs: Any) -> str | None:
    """Custom prompt with JSON format instruction for non-native backends."""
    if use_native:
        return None  # Let the default prompt handle it
    from openharness.prompts import build_prompt as _default  # noqa: PLC0415

    return _default(
        **kwargs,
        action_prompt=(
            'Respond with EXACTLY one JSON object, nothing else:\n'
            '{"tool": "<tool_name>", "args": {<arguments>}, "reasoning": "<why>"}\n'
            'Example: {"tool": "write_file", "args": {"path": "main.py", '
            '"content": "print(1)"}, "reasoning": "create main"}'
        ),
    )


DOMAIN = DomainAdapter(
    build_briefing=_build_briefing,
)


# ═══════════════════════════════════════════════════════════════════
# 6. EVALS — write eval functions as you debug, they run automatically
#
# Each eval_* function takes EvalContext and returns a score.
# These are the same checks you'd do manually when reviewing
# an agent run — formalized as reusable evaluators.
# ═══════════════════════════════════════════════════════════════════


def eval_tests_passed(ctx: EvalContext) -> bool:
    """Did the agent run tests and get them to pass?

    This is the most important eval for a coding agent.
    Found this pattern while debugging: agents that skip testing
    or call done() before tests pass produce bad code.
    """
    for s in reversed(ctx.steps):
        tc = s.tool_call
        tr = s.tool_result
        if getattr(tc, "tool", "") == "bash":
            cmd = getattr(tc, "args", {}).get("command", "")
            if "pytest" in cmd:
                data = getattr(tr, "data", None) or {}
                if data.get("exit_code", 1) == 0:
                    return True
    return False


def eval_wrote_tests(ctx: EvalContext) -> bool:
    """Did the agent write test files (not just implementation)?

    Found while debugging: some agents write code but skip tests,
    then call done() — the guardrail hook catches this at runtime,
    but the eval catches it in batch scoring.
    """
    for s in ctx.steps:
        tc = s.tool_call
        if getattr(tc, "tool", "") == "write":
            path = getattr(tc, "args", {}).get("file_path", "")
            if "test_" in path:
                return True
    return False


def eval_efficiency(ctx: EvalContext) -> float:
    """Score 0-1: how efficiently did the agent complete the task?

    Debugging insight: agents that take >10 steps for a simple task
    are usually stuck in a loop (write→test→fail→rewrite→test→fail).
    Under 5 steps = excellent, 5-10 = good, >10 = poor.
    """
    n = ctx.step_count
    if n <= 3:
        return 1.0
    if n <= 5:
        return 0.9
    if n <= 10:
        return 0.6
    return max(0.2, 1.0 - (n - 10) * 0.1)


def eval_error_recovery(ctx: EvalContext) -> dict:
    """Did the agent recover from errors or get stuck?

    Debugging pattern: count errors and check if the agent
    eventually succeeded after them.
    """
    errors = [s for s in ctx.steps
              if getattr(s.tool_result, "error", None)]
    total = ctx.step_count
    error_rate = len(errors) / max(total, 1)
    recovered = eval_tests_passed(ctx) if errors else True
    return {
        "error_rate": round(error_rate, 2),
        "errors": len(errors),
        "recovered": 1.0 if recovered else 0.0,
        "score": (1.0 - error_rate) * (1.0 if recovered else 0.5),
    }


# All evals for this agent, collected in one list
CODING_EVALS = [eval_tests_passed, eval_wrote_tests, eval_efficiency, eval_error_recovery]


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN — Putting it all together
# ═══════════════════════════════════════════════════════════════════

def run_coding_agent(
    llm: Any,
    task: str = "Implement a fibonacci function in fibonacci.py with tests in test_fibonacci.py",
    *,
    max_steps: int = 20,
    workspace: str | None = None,
    trace_dir: str | None = None,
) -> dict:
    """Run a harnessed coding agent.

    Args:
        llm: Any LLMBackend (OpenAI, Anthropic, mock, etc.)
        task: What to build.
        max_steps: Budget limit.
        workspace: Directory to work in (default: temp dir).
        trace_dir: If set, save a full trajectory recording to this
            directory for replay/debugging.

    Returns:
        The trace dict with all steps, timing, etc.
    """
    ws = workspace or tempfile.mkdtemp(prefix="openharness_example_")

    # Detect native tool support — use it when available for
    # reliability; fall back to JSON-text with format instructions.
    _native = hasattr(llm, "generate_with_tools")
    if _native:
        # Probe: some proxies expose generate_with_tools but silently
        # ignore the tools parameter. Send a trivial tool call and
        # check if the response actually contains a tool_use block.
        try:
            _test = llm.generate_with_tools(
                "Call the test_probe tool now.",
                tools=[{"name": "test_probe", "description": "Probe tool",
                        "input_schema": {"type": "object", "properties": {}}}],
                max_tokens=50, system_prompt="", temperature=0,
            )
            # Native backends return blocks with at least one tool_use
            _has_tool_use = (
                isinstance(_test, list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_use"
                    for b in _test
                )
            )
            if not _has_tool_use:
                _native = False
        except Exception:  # noqa: BLE001
            _native = False

    # When native tools aren't available, add a JSON format instruction
    # via build_prompt so the model knows the expected response shape.
    _domain = DomainAdapter(
        build_briefing=_build_briefing,
        build_prompt=(None if _native else # pyright: ignore[reportArgumentType]
                      lambda **kw: _build_prompt(use_native=False, **kw)),
    )

    print(f"[harness] Tool protocol: {'native' if _native else 'json-text'}")

    config = LoopConfig(
        max_steps=max_steps,
        system_prompt=(
            "You are a software engineer working in a project directory. "
            "Use relative file paths. Use bash to run tests (python -m pytest), "
            "install packages, and execute commands. Use edit for targeted "
            "fixes, write for new files. Run tests before calling done."
        ),
        # Use native tool calling when the backend supports it.
        use_native_tools=_native,
        # Compaction chain: cheap → LLM → truncate
        compact_service=COMPACT_SERVICE,
        # Domain adapter bundles the briefing builder (+ prompt override
        # for JSON-text mode when native tools aren't available)
        domain=_domain,
        # Persistent memory — coding standards survive compaction
        memory_sources=[CODING_STANDARDS],
        # Budget: auto-compact at 80% of context window
        # (ThresholdCompactHook fires should_compact when pressure is high)
    )

    # Budget-aware compaction trigger
    budget = ContextBudget(
        context_window=128_000,
        warning_at=80_000,
        error_at=100_000,
    )

    hooks: list[Any] = [
        CodingGuardrailHook(),
        ThresholdCompactHook(budget),
        EvalHook(evaluators=CODING_EVALS, verbose=True),
    ]

    # Trajectory recording — saves full run for replay/debugging
    _recorder = None
    if trace_dir:
        from openharness.provenance import TrajectoryRecorder  # noqa: PLC0415
        _recorder = TrajectoryRecorder()
        hooks.append(_recorder)

    tools = build_tools(ws)
    state = DefaultState(max_steps=max_steps)

    print(f"[harness] Workspace: {ws}")
    print(f"[harness] Task: {task}")
    print(f"[harness] Max steps: {max_steps}")
    print()

    trace = None
    gen = composable_loop(
        llm=llm,
        tools=tools,
        state=state,
        hooks=hooks,
        config=config,
        task={"description": task, "workspace": ws},
    )

    for step in gen:
        tc = step.tool_call
        tr = step.tool_result
        status = "✓" if not (tr and tr.error) else "✗"
        # Compact display: show tool + key args (skip content/new_string for brevity)
        skip_keys = {"content", "new_string", "old_string"}
        args_str = ", ".join(f"{k}={v!r}" for k, v in tc.args.items() if k not in skip_keys)
        print(f"  Step {step.number}: {tc.tool}({args_str}) {status}")
        if tc.tool == "bash" and tr and tr.data:
            ec = tr.data.get("exit_code", -1)
            cmd = tc.args.get("command", "")
            if "pytest" in cmd:
                print(f"    Tests: {'PASSED ✓' if ec == 0 else 'FAILED ✗'}")

    # Get trace from generator return value
    try:
        gen.send(None)
    except StopIteration as e:
        trace = e.value

    print()
    print(f"[harness] Done. {state.step_count} steps, workspace: {ws}")

    if _recorder and trace_dir:
        _recorder.save(trace_dir)
        print(f"[harness] Trajectory saved to {trace_dir}")

    # Save eval results alongside trajectory
    if trace_dir:
        _eval_hook = next((h for h in hooks if isinstance(h, EvalHook)), None)
        if _eval_hook and _eval_hook.results:
            _eval_hook.save(os.path.join(trace_dir, "eval_results.json"))
            print(f"[harness] Eval results saved to {trace_dir}/eval_results.json")

    return trace or {}


# ═══════════════════════════════════════════════════════════════════
# Run with real LLM
# ═══════════════════════════════════════════════════════════════════


def _get_llm(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Create an OpenAI-compatible backend.

    Reads from arguments, then env vars, then defaults:
      - OPENAI_BASE_URL / base_url  (default: http://127.0.0.1:19823)
      - OPENAI_MODEL / model        (default: gpt-4.1)
      - OPENAI_API_KEY / api_key    (default: "x")

    Works with any OpenAI-compatible API: OpenAI, Azure, local
    proxies, llama.cpp, vLLM, Ollama, LiteLLM, etc.
    """
    from openharness.backends import OpenAIBackend  # noqa: PLC0415

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "This example requires the openai package. "
            "Install it with: pip install openai"
        ) from e

    _url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:19823/v1")
    _model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")
    _key = api_key or os.environ.get("OPENAI_API_KEY", "x")

    client = OpenAI(base_url=_url, api_key=_key)
    return OpenAIBackend(client, model=_model)


def main() -> None:
    """Run the coding agent with a real LLM backend."""
    import argparse

    parser = argparse.ArgumentParser(
        description="openharness coding agent example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requires an OpenAI-compatible API. Set env vars or use flags:\n"
            "  OPENAI_BASE_URL  (default: http://127.0.0.1:19823)\n"
            "  OPENAI_MODEL     (default: gpt-4.1)\n"
            "  OPENAI_API_KEY   (default: x)\n"
        ),
    )
    parser.add_argument("task", nargs="?",
                        default="Implement a fibonacci function in fibonacci.py "
                                "with tests in test_fibonacci.py",
                        help="What to build (default: fibonacci)")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Max agent steps (default: 15)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None,
                        help="Model name")
    parser.add_argument("--trace", default=None, metavar="DIR",
                        help="Save trajectory to DIR for replay/debugging")
    args = parser.parse_args()

    print("=" * 60)
    print("openharness — Coding Agent Example")
    print("=" * 60)
    print()

    llm = _get_llm(base_url=args.base_url, model=args.model)
    print(f"[backend] {llm._model} @ {llm._client.base_url}")
    print()

    run_coding_agent(llm, task=args.task, max_steps=args.max_steps,
                     trace_dir=args.trace)


if __name__ == "__main__":
    main()
