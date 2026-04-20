"""High-level presets for common agent configurations.

Presets reduce the boilerplate needed to set up an agent from ~50 lines
to ~3 lines.  Each preset returns an :class:`AgentPreset` dataclass
containing a pre-configured ``LoopConfig``, hook list, tool registry,
and ``DefaultState`` — everything ``composable_loop`` needs.

Usage::

    from openharness.presets import coding_agent_preset

    preset = coding_agent_preset(workspace="/tmp/my-project")
    for step in composable_loop(
        llm=my_llm,
        tools=preset.tools,
        state=preset.state,
        config=preset.config,
        hooks=preset.hooks,
        task={"description": "Implement fizzbuzz with tests"},
    ):
        print(step.pretty())

Presets are opinionated defaults.  Override any field after creation::

    preset = coding_agent_preset(workspace="/tmp")
    preset.config.max_steps = 50  # override budget
    preset.hooks.append(my_custom_hook)  # add a hook
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

from openharness.budget import ContextBudget, ThresholdCompactHook
from openharness.compact import (
    PruneToolResults,
    SummarizeCompact,
    TruncateCompact,
    compact_chain,
)
from openharness.loop import LoopConfig
from openharness.memory import StaticMemorySource
from openharness.tools import BaseToolRegistry, ToolSpec, register_think_tool
from openharness.types import DefaultState

__all__ = [
    "AgentPreset",
    "coding_agent_preset",
    "research_agent_preset",
    "minimal_preset",
]

# ── Preset container ─────────────────────────────────────────────

@dataclass
class AgentPreset:
    """Complete agent configuration returned by preset functions.

    Contains everything needed for ``composable_loop``.  All fields are
    mutable — override any after construction for customization.
    """

    config: LoopConfig
    """Loop configuration (max_steps, system_prompt, compact chain, etc.)."""

    hooks: list[Any]
    """List of LoopHook instances."""

    tools: BaseToolRegistry
    """Pre-registered tool registry."""

    state: DefaultState
    """Agent state with budget tracking."""

# ── Tool implementations ─────────────────────────────────────────

def _bash(*, command: str, workspace: str = "") -> dict:
    """Execute a bash command.

    .. warning::

       This tool runs arbitrary shell commands with no sandboxing.
       Suitable for local development only.  For production deployments,
       gate with ``PermissionHook`` or replace with a sandboxed executor.
    """
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=120,
            cwd=workspace or None,
        )
        output = result.stdout.strip()
        err = result.stderr.strip()
        combined = output + ("\n" + err if err else "")
        if len(combined) > 8000:
            combined = combined[:4000] + "\n...(truncated)...\n" + combined[-4000:]
        return {
            "stdout": output[:8000] if output else "",
            "stderr": err[:4000] if err else "",
            "exit_code": result.returncode,
            "next_step": (
                "Command succeeded."
                if result.returncode == 0 else
                f"Command failed (exit {result.returncode}). Read stderr and fix."
            ),
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 120s",
                "remediation": "Check for infinite loops or blocking I/O."}

def _read(*, file_path: str, workspace: str = "") -> dict:
    """Read a file with line numbers."""
    full = os.path.join(workspace, file_path) if workspace else file_path
    try:
        with open(full) as f:
            lines = f.readlines()
        numbered = "".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))
        if len(numbered) > 12000:
            numbered = numbered[:6000] + "\n...(truncated)...\n" + numbered[-6000:]
        return {"path": file_path, "content": numbered, "line_count": len(lines)}
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}",
                "remediation": "Use glob to find files, or write to create."}

def _write(*, file_path: str, content: str, workspace: str = "") -> dict:
    """Create or overwrite a file."""
    full = os.path.join(workspace, file_path) if workspace else file_path
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return {"written": file_path, "lines": content.count("\n") + 1}

def _edit(*, file_path: str, old_string: str, new_string: str,
          workspace: str = "") -> dict:
    """Replace exact string match in a file."""
    full = os.path.join(workspace, file_path) if workspace else file_path
    try:
        text = open(full).read()
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}",
                "remediation": "Use write to create the file first."}
    count = text.count(old_string)
    if count == 0:
        return {"error": "old_string not found",
                "remediation": "Use read to see current content. Match exactly."}
    if count > 1:
        return {"error": f"old_string matches {count} locations — ambiguous",
                "remediation": "Include more context to make it unique."}
    with open(full, "w") as f:
        f.write(text.replace(old_string, new_string, 1))
    return {"edited": file_path, "replacements": 1}

def _glob(*, pattern: str, workspace: str = "") -> dict:
    """Find files matching a glob pattern."""
    import glob as globmod
    base = workspace or "."
    matches = sorted(globmod.glob(os.path.join(base, pattern), recursive=True))
    rel = [os.path.relpath(m, base) for m in matches]
    if len(rel) > 100:
        return {"pattern": pattern, "files": rel[:100], "truncated": True, "total": len(matches)}
    return {"pattern": pattern, "files": rel, "total": len(rel)}

def _grep(*, pattern: str, path: str = ".", workspace: str = "") -> dict:
    """Search file contents with regex."""
    full = os.path.join(workspace, path) if workspace else path
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.ts", "--include=*.js",
             "--include=*.md", "--include=*.yaml", "--include=*.yml",
             "--include=*.json", "--include=*.toml", pattern, full],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) > 50:
            return {"pattern": pattern, "matches": lines[:50], "truncated": True}
        return {"pattern": pattern, "matches": lines, "total": len(lines)}
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "remediation": "Narrow your pattern."}

def _build_coding_tools(workspace: str) -> BaseToolRegistry:
    """Build a coding agent tool registry (bash, read, write, edit, glob, grep, think, done)."""
    reg = BaseToolRegistry()

    reg.register(ToolSpec(
        name="bash",
        description=(
            "Execute a bash command. Use for running tests (pytest), installing packages, "
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
        concurrent_safe=True,
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
            "Edit a file by replacing an exact string. Provide the exact text to find "
            "and the replacement. Use read first to see current content."
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
        concurrent_safe=True,
    ))
    reg.register(ToolSpec(
        name="grep",
        description="Search file contents with regex. Returns matching lines with file:line.",
        parameters={"pattern": "str", "path": "str"},
        execute=lambda *, pattern, path=".": _grep(
            pattern=pattern, path=path, workspace=workspace),
        concurrent_safe=True,
    ))
    register_think_tool(reg)
    reg.register(ToolSpec(
        name="done",
        description="Signal task completion. Only call when ALL tests pass.",
        parameters={"summary": "str"},
        execute=lambda *, summary: {"summary": summary},
    ))
    return reg

# ── Coding agent hook ────────────────────────────────────────────

class _CodingGuardrailHook:
    """Just-in-time guardrails: test enforcement and review nudges."""

    def __init__(self) -> None:
        self._tests_passed: bool = False
        self._files_written: set[str] = set()
        self._has_tests: bool = False

    def post_dispatch(
        self, state: Any, session_log: Any,
        tool_call: Any, tool_result: Any, step_num: int,
    ) -> Any:
        from openharness.hook_decision import InjectContext  # noqa: PLC0415

        if tool_call.tool == "write":
            path = tool_call.args.get("file_path", "")
            self._files_written.add(path)
            if path.startswith("test_") or "/test_" in path:
                self._has_tests = True
            elif not self._has_tests:
                return InjectContext(
                    "You wrote source code but no tests yet. Write tests first."
                )

        if tool_call.tool == "bash":
            cmd = tool_call.args.get("command", "")
            data = tool_result.data or {}
            if "pytest" in cmd or "python -m pytest" in cmd:
                self._tests_passed = data.get("exit_code", 1) == 0
                if not self._tests_passed:
                    return InjectContext(
                        "Tests failed. Read output, fix the exact issue, run tests again."
                    )
        return None

    def check_done(self, state: Any, session_log: Any, context: Any, step_num: int) -> Any:
        from openharness.hook_decision import HookDecision  # noqa: PLC0415

        if not self._tests_passed:
            return HookDecision(
                block="Tests have not passed yet. Run tests and fix failures first."
            )
        return None

    def should_stop(self, state: Any, step_num: int, new_entities: int) -> bool:
        return False

# ── Preset functions ─────────────────────────────────────────────

def coding_agent_preset(
    workspace: str = ".",
    *,
    max_steps: int = 20,
    context_window: int = 128_000,
    system_prompt: str = "",
    require_tests: bool = True,
) -> AgentPreset:
    """Pre-configured coding agent with bash/read/write/edit/glob/grep tools.

    Returns an :class:`AgentPreset` ready for ``composable_loop``.
    Includes just-in-time guardrails (test enforcement), compaction chain,
    persistent coding standards, and context budget management.

    Args:
        workspace: Directory the agent operates in.
        max_steps: Maximum tool calls before the loop stops.
        context_window: LLM context window size in tokens.
        system_prompt: Override system prompt. If empty, uses a sensible default.
        require_tests: If True, includes guardrail hook that blocks done()
            until tests pass.

    Example::

        preset = coding_agent_preset(workspace="/tmp/project")
        for step in composable_loop(
            llm=my_llm, tools=preset.tools, state=preset.state,
            config=preset.config, hooks=preset.hooks,
            task={"description": "Build a fibonacci module with tests"},
        ):
            print(step.pretty())
    """
    tools = _build_coding_tools(workspace)

    hooks: list[Any] = []
    if require_tests:
        hooks.append(_CodingGuardrailHook())
    hooks.append(ThresholdCompactHook(ContextBudget(
        context_window=context_window,
        warning_at=int(context_window * 0.6),
        error_at=int(context_window * 0.8),
    )))

    config = LoopConfig(
        max_steps=max_steps,
        system_prompt=system_prompt or (
            "You are a software engineer working in a project directory. "
            "Use relative file paths. Use bash to run tests (python -m pytest), "
            "install packages, and execute commands. Use edit for targeted "
            "fixes, write for new files. Run tests before calling done."
        ),
        compact_service=compact_chain(
            PruneToolResults(keep_recent=5),
            SummarizeCompact(keep_recent=2),
            TruncateCompact(keep_recent=1),
        ),
        memory_sources=[StaticMemorySource(
            "## Coding Standards (persistent)\n"
            "1. Write test file before or alongside implementation.\n"
            "2. Type hints on all function signatures.\n"
            "3. Docstrings on public functions.\n"
            "4. No bare except — catch specific exceptions.\n"
            "5. snake_case for functions, PascalCase for classes.\n"
        )],
        context_window=context_window,
    )

    return AgentPreset(
        config=config,
        hooks=hooks,
        tools=tools,
        state=DefaultState(max_steps=max_steps),
    )

def research_agent_preset(
    workspace: str = ".",
    *,
    max_steps: int = 30,
    context_window: int = 128_000,
    system_prompt: str = "",
) -> AgentPreset:
    """Pre-configured research agent with read/grep/glob/bash tools.

    Optimized for information gathering: larger step budget, no test
    requirement, read-heavy tool set with concurrent dispatch enabled.

    Args:
        workspace: Directory the agent operates in.
        max_steps: Maximum tool calls (default 30 for research).
        context_window: LLM context window size in tokens.
        system_prompt: Override system prompt.

    Example::

        preset = research_agent_preset(workspace="/tmp/codebase")
        for step in composable_loop(
            llm=my_llm, tools=preset.tools, state=preset.state,
            config=preset.config, hooks=preset.hooks,
            task={"description": "Analyze the auth module architecture"},
        ):
            print(step.pretty())
    """
    tools = _build_coding_tools(workspace)

    hooks: list[Any] = [
        ThresholdCompactHook(ContextBudget(
            context_window=context_window,
            warning_at=int(context_window * 0.6),
            error_at=int(context_window * 0.8),
        )),
    ]

    config = LoopConfig(
        max_steps=max_steps,
        system_prompt=system_prompt or (
            "You are a research analyst. Read files, search code, run commands "
            "to understand the codebase. Produce clear, structured findings. "
            "Call done with a comprehensive summary when finished."
        ),
        compact_service=compact_chain(
            PruneToolResults(keep_recent=8),
            SummarizeCompact(keep_recent=3),
            TruncateCompact(keep_recent=2),
        ),
        context_window=context_window,
    )

    return AgentPreset(
        config=config,
        hooks=hooks,
        tools=tools,
        state=DefaultState(max_steps=max_steps),
    )

def minimal_preset(
    *,
    max_steps: int = 10,
    tools: list[ToolSpec] | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> AgentPreset:
    """Minimal agent preset — bare loop with optional custom tools.

    Use as a starting point for custom agents. Provides only the loop
    config and state; bring your own tools or pass them in.

    Args:
        max_steps: Maximum tool calls.
        tools: Optional list of ToolSpec to register. If None,
            registers only a ``done`` tool.
        system_prompt: System prompt for the LLM.

    Example::

        from openharness import ToolSpec
        preset = minimal_preset(tools=[
            ToolSpec(name="search", description="Search",
                     parameters={"q": "str"}, execute=my_search),
        ])
    """
    reg = BaseToolRegistry()
    if tools:
        for spec in tools:
            reg.register(spec)
    # Always include done
    if "done" not in reg.tool_names:
        reg.register(ToolSpec(
            name="done",
            description="Signal task completion.",
            parameters={"summary": "str"},
            execute=lambda *, summary: {"summary": summary},
        ))

    config = LoopConfig(
        max_steps=max_steps,
        system_prompt=system_prompt,
    )

    return AgentPreset(
        config=config,
        hooks=[],
        tools=reg,
        state=DefaultState(max_steps=max_steps),
    )
