#!/usr/bin/env python3
"""looplet coder — a Claude Code-class coding agent built on looplet.

A serious coding agent that reads, writes, edits, tests, and iterates.
Every step is visible. Every decision is auditable. Zero magic.

Usage:
    python examples/coder/agent.py "Add type hints to utils.py"
    python examples/coder/agent.py "Fix the failing test in test_auth.py"
    python examples/coder/agent.py "Create a fibonacci module with tests"

    # In a specific directory:
    python examples/coder/agent.py --workspace /path/to/project "Build feature X"

    # With a local LLM:
    OPENAI_BASE_URL=http://localhost:11434/v1 python examples/coder/agent.py "..."
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from looplet import (
    BaseToolRegistry,
    CallableMemorySource,
    Conversation,
    DefaultState,
    LoopConfig,
    OpenAIBackend,
    StaticMemorySource,
    StreamingHook,
    ToolSpec,
    TrajectoryRecorder,
    composable_loop,
    register_done_tool,
)
from looplet.compact import PruneToolResults, TruncateCompact, compact_chain
from looplet.hook_decision import HookDecision, InjectContext
from looplet.limits import PerToolLimitHook
from looplet.provenance import RecordingLLMBackend
from looplet.resilient import ResilientBackend
from looplet.session import SessionLog
from looplet.stagnation import StagnationHook, tool_call_fingerprint
from looplet.streaming import CallbackEmitter
from looplet.tools import register_think_tool

# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════


def _run(cmd: str, cwd: str, timeout: int = 120) -> dict:
    try:
        r = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        stdout = r.stdout.strip()
        stderr = r.stderr.strip()
        # Truncate very long output
        if len(stdout) > 15000:
            stdout = (
                stdout[:7000]
                + f"\n\n... [{len(stdout) - 14000} chars truncated] ...\n\n"
                + stdout[-7000:]
            )
        if len(stderr) > 5000:
            stderr = (
                stderr[:2000] + f"\n... [{len(stderr) - 4000} chars truncated] ..." + stderr[-2000:]
            )
        return {"stdout": stdout, "stderr": stderr, "exit_code": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


def make_tools(workspace: str) -> BaseToolRegistry:
    """Build the coding agent's tool registry."""
    tools = BaseToolRegistry()

    # ── bash ────────────────────────────────────────────────────
    tools.register(
        ToolSpec(
            name="bash",
            description=(
                "Execute a bash command in the project directory. "
                "Use for: running tests (pytest, npm test), installing packages, "
                "git operations, searching (grep, find, rg), compiling, "
                "and any shell operation. "
                "Commands run with bash -c in the workspace directory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"},
                },
                "required": ["command"],
            },
            execute=lambda *, command: _run(command, workspace),
        )
    )

    # ── read_file ───────────────────────────────────────────────
    def read_file(*, file_path: str, start_line: int = 0, end_line: int = 0) -> dict:
        p = Path(workspace) / file_path
        if not p.exists():
            return {"error": f"File not found: {file_path}"}
        if not p.is_file():
            return {"error": f"Not a file: {file_path}"}
        try:
            lines = p.read_text().splitlines()
            if start_line > 0 and end_line > 0:
                selected = lines[start_line - 1 : end_line]
                numbered = [f"{start_line + i:>4} | {line}" for i, line in enumerate(selected)]
            else:
                numbered = [f"{i + 1:>4} | {line}" for i, line in enumerate(lines)]
            content = "\n".join(numbered)
            if len(content) > 20000:
                content = (
                    content[:10000]
                    + f"\n... [{len(content) - 20000} chars truncated] ...\n"
                    + content[-10000:]
                )
            return {"path": file_path, "content": content, "total_lines": len(lines)}
        except Exception as e:
            return {"error": str(e)}

    tools.register(
        ToolSpec(
            name="read_file",
            description=(
                "Read a file with line numbers. Use relative paths from the project root. "
                "Optionally specify start_line and end_line (1-indexed) to read a range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path to the file"},
                    "start_line": {
                        "type": "integer",
                        "description": "Start line (1-indexed, 0=all)",
                        "default": 0,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line (1-indexed, 0=all)",
                        "default": 0,
                    },
                },
                "required": ["file_path"],
            },
            execute=read_file,
            concurrent_safe=True,
        )
    )

    # ── write_file ──────────────────────────────────────────────
    def write_file(*, file_path: str, content: str) -> dict:
        p = Path(workspace) / file_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lines = content.count("\n") + 1
        return {"written": file_path, "lines": lines}

    tools.register(
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file. Creates parent directories as needed.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to create/overwrite",
                    },
                    "content": {"type": "string", "description": "Complete file content"},
                },
                "required": ["file_path", "content"],
            },
            execute=write_file,
        )
    )

    # ── edit_file (search-and-replace) ──────────────────────────
    def edit_file(*, file_path: str, old_string: str, new_string: str) -> dict:
        p = Path(workspace) / file_path
        if not p.exists():
            return {"error": f"File not found: {file_path}"}
        text = p.read_text()
        count = text.count(old_string)
        if count == 0:
            # Show nearby content for debugging
            lines = text.splitlines()
            # Try to find partial match
            partial = old_string.splitlines()[0] if old_string else ""
            nearby = [
                f"  {i + 1}: {l}" for i, l in enumerate(lines) if partial and partial[:30] in l
            ]
            hint = "\n".join(nearby[:5]) if nearby else "(no partial matches)"
            return {
                "error": f"old_string not found in {file_path} (0 matches). "
                f"Read the file first to see exact content.",
                "hint": hint,
            }
        if count > 1:
            return {
                "error": f"old_string matches {count} locations in {file_path}. "
                f"Include more context lines to make the match unique.",
                "matches": count,
            }
        new_text = text.replace(old_string, new_string, 1)
        p.write_text(new_text)
        return {"edited": file_path, "replacements": 1}

    tools.register(
        ToolSpec(
            name="edit_file",
            description=(
                "Edit a file by replacing an exact string with a new string. "
                "The old_string must match EXACTLY (including whitespace and indentation). "
                "Include 2-3 lines of surrounding context to ensure a unique match. "
                "Use read_file first to see the exact content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative path to edit"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find (must be unique)",
                    },
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            execute=edit_file,
        )
    )

    # ── glob ────────────────────────────────────────────────────
    def glob_files(*, pattern: str) -> dict:
        matches = sorted(
            str(p.relative_to(workspace)) for p in Path(workspace).glob(pattern) if p.is_file()
        )
        if len(matches) > 100:
            matches = matches[:100]
        return {"pattern": pattern, "matches": matches, "count": len(matches)}

    tools.register(
        ToolSpec(
            name="glob",
            description="Find files matching a glob pattern (e.g. '**/*.py', 'tests/test_*.py').",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                },
                "required": ["pattern"],
            },
            execute=glob_files,
            concurrent_safe=True,
        )
    )

    # ── grep ────────────────────────────────────────────────────
    def grep_search(*, pattern: str, path: str = ".", include: str = "") -> dict:
        cmd = "grep -rn --include='*.py' --include='*.js' --include='*.ts' --include='*.md' "
        if include:
            cmd = f"grep -rn --include='{include}' "
        cmd += f"'{pattern}' '{path}' 2>/dev/null | head -50"
        result = _run(cmd, workspace, timeout=10)
        lines = result["stdout"].splitlines() if result["stdout"] else []
        return {"pattern": pattern, "matches": lines, "count": len(lines)}

    tools.register(
        ToolSpec(
            name="grep",
            description="Search file contents with regex. Returns matching lines with file:line:content.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in",
                        "default": ".",
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob filter (e.g. '*.py')",
                        "default": "",
                    },
                },
                "required": ["pattern"],
            },
            execute=grep_search,
            concurrent_safe=True,
        )
    )

    # ── done + think ────────────────────────────────────────────
    register_done_tool(tools)
    register_think_tool(tools)

    return tools


# ═══════════════════════════════════════════════════════════════════
# AUTO-DISCOVERED PROJECT INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════


def _discover_instructions(workspace: str) -> str:
    """Auto-discover project instruction files (like Claude Code's CLAUDE.md)."""
    candidates = [
        "CLAUDE.md",
        ".claude.md",
        "AGENTS.md",
        ".cursorrules",
        "CODING_GUIDELINES.md",
        ".github/copilot-instructions.md",
    ]
    parts = []
    for name in candidates:
        p = Path(workspace) / name
        if p.exists():
            content = p.read_text()[:3000]  # cap at 3000 chars
            parts.append(f"## From {name}\n{content}")
    return "\n\n".join(parts) if parts else ""


def _project_context(workspace: str) -> str:
    """Build a quick project context snapshot."""
    parts = []
    # Git info
    try:
        branch = subprocess.run(
            ["git", "-C", workspace, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if branch:
            parts.append(f"Git branch: {branch}")
    except Exception:
        pass
    # Key files
    for name in ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"]:
        if (Path(workspace) / name).exists():
            parts.append(f"Build: {name}")
    # File count
    try:
        py_count = len(list(Path(workspace).rglob("*.py")))
        if py_count:
            parts.append(f"Python files: {py_count}")
    except Exception:
        pass
    return " | ".join(parts) if parts else "Unknown project"


# ═══════════════════════════════════════════════════════════════════
# HOOKS
# ═══════════════════════════════════════════════════════════════════


class TestGuardHook:
    """Block done() until tests have passed at least once."""

    def __init__(self):
        self._tests_passed = False
        self._files_written: set[str] = set()

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool == "bash":
            cmd = tool_call.args.get("command", "")
            data = tool_result.data or {}
            if "pytest" in cmd or "python -m pytest" in cmd or "npm test" in cmd:
                self._tests_passed = data.get("exit_code", 1) == 0
                if not self._tests_passed:
                    return InjectContext(
                        "Tests FAILED. Read the error output carefully, fix the exact issue, "
                        "and run tests again. Do NOT call done() until tests pass."
                    )
        if tool_call.tool in ("write_file", "edit_file"):
            path = tool_call.args.get("file_path", "")
            self._files_written.add(path)
        return None

    def check_done(self, state, session_log, context, step_num):
        if not self._tests_passed and self._files_written:
            return HookDecision(
                block="Tests have not passed yet. Run tests and fix any failures before calling done()."
            )
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert software engineer. You solve tasks by reading code, \
understanding the codebase structure, making targeted changes, and \
verifying with tests.

## Workflow
1. UNDERSTAND: Read relevant files first. Use grep/glob to find what you need.
2. PLAN: Use think() to plan your approach before making changes.
3. IMPLEMENT: Write or edit files. Make minimal, targeted changes.
4. TEST: Run tests to verify. Fix failures before proceeding.
5. DONE: Call done() with a summary only after tests pass.

## Rules
- Always read a file before editing it.
- Use edit_file for targeted changes (include 2-3 context lines for unique match).
- Use write_file only for new files.
- Run tests after EVERY change. Never skip testing.
- If tests fail, read the error, fix it, re-run. Do not give up.
- Use relative paths from the project root.
- Do not modify files unrelated to the task.
- Prefer small, incremental changes over large rewrites.
"""


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="looplet coder — AI coding agent")
    parser.add_argument("task", help="What to build or fix")
    parser.add_argument("--workspace", "-w", default=os.getcwd(), help="Project directory")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum tool calls")
    parser.add_argument("--no-tests", action="store_true", help="Skip test guard")
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "x")
    model = os.environ.get("OPENAI_MODEL", "llama3.1")

    llm = ResilientBackend(
        OpenAIBackend(base_url=base_url, api_key=api_key, model=model),
        retries=2,
        timeout_s=120,
    )
    recording = RecordingLLMBackend(llm)

    tools = make_tools(workspace)

    # Hooks
    hooks: list = []
    if not args.no_tests:
        hooks.append(TestGuardHook())
    hooks.append(
        StagnationHook(
            fingerprint=tool_call_fingerprint,
            threshold=4,
            nudge="[stagnation] You're repeating yourself. Try a different approach.",
        )
    )
    hooks.append(PerToolLimitHook(default_limit=20, limits={"bash": 15, "read_file": 15}))

    events: list = []
    hooks.append(StreamingHook(CallbackEmitter(events.append)))

    # Memory: project instructions + context
    instructions = _discover_instructions(workspace)
    project_ctx = _project_context(workspace)

    memory_sources = []
    if instructions:
        memory_sources.append(StaticMemorySource(instructions))
    memory_sources.append(
        CallableMemorySource(
            lambda state: (
                f"Project: {project_ctx} | Steps: {getattr(state, 'step_count', 0)}/{args.max_steps}"
            )
        )
    )

    config = LoopConfig(
        max_steps=args.max_steps,
        temperature=0.2,
        system_prompt=SYSTEM_PROMPT,
        compact_service=compact_chain(
            PruneToolResults(keep_recent=8),
            TruncateCompact(keep_recent=4),
        ),
        memory_sources=memory_sources,
    )

    state = DefaultState(max_steps=args.max_steps)
    session_log = SessionLog()
    conv = Conversation()

    # ── Run ──────────────────────────────────────────────────────

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              looplet coder                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Task: {args.task}")
    print(f"  Workspace: {workspace}")
    print(f"  Context: {project_ctx}")
    if instructions:
        print(f"  Instructions: {len(instructions)} chars auto-discovered")
    print(f"  Model: {model}")
    print(f"  Budget: {args.max_steps} steps")
    print()

    with tempfile.TemporaryDirectory() as traj_dir:
        recorder = TrajectoryRecorder(recording_llm=recording, output_dir=traj_dir)
        hooks.append(recorder)

        for step in composable_loop(
            llm=recording,
            task={"description": args.task},
            tools=tools,
            state=state,
            config=config,
            hooks=hooks,
            session_log=session_log,
            conversation=conv,
        ):
            tool = step.tool_call.tool
            err = step.tool_result.error
            data = step.tool_result.data or {}
            _warns = step.tool_result.warnings  # noqa: F841

            if tool == "done":
                print(f"\n  ✓ Done: {data.get('summary', data.get('status', ''))[:120]}")
            elif tool == "think":
                analysis = step.tool_call.args.get("analysis", "")[:100]
                print(f"  💭 #{step.number} think: {analysis}...")
            elif tool == "bash":
                cmd = step.tool_call.args.get("command", "")[:60]
                exit_code = data.get("exit_code", "?")
                mark = "✓" if exit_code == 0 else "✗"
                print(f"  {mark} #{step.number} bash: {cmd}  [exit {exit_code}]")
            elif tool == "read_file":
                path = step.tool_call.args.get("file_path", "?")
                lines = data.get("total_lines", "?")
                print(f"  📖 #{step.number} read: {path} ({lines} lines)")
            elif tool == "write_file":
                path = data.get("written", "?")
                lines = data.get("lines", "?")
                print(f"  ✏️  #{step.number} write: {path} ({lines} lines)")
            elif tool == "edit_file":
                path = step.tool_call.args.get("file_path", "?")
                if err:
                    print(f"  ✗ #{step.number} edit: {path} — {str(err)[:60]}")
                else:
                    print(f"  ✏️  #{step.number} edit: {path}")
            elif tool == "glob":
                count = data.get("count", "?")
                print(f"  🔍 #{step.number} glob: {count} files")
            elif tool == "grep":
                count = data.get("count", "?")
                pattern = step.tool_call.args.get("pattern", "?")[:30]
                print(f"  🔍 #{step.number} grep: '{pattern}' → {count} matches")
            elif err:
                print(f"  ✗ #{step.number} {tool}: {str(err)[:60]}")
            else:
                print(f"  → #{step.number} {tool}")

        # Stats
        scoped = [c for c in recording.calls if c.scope]
        print("\n  ──────────────────────────────────")
        print(f"  Steps: {len(state.steps)}")
        print(f"  LLM calls: {len(recording.calls)} ({len(scoped)} tool-internal)")
        print(f"  Trajectory: {traj_dir}/trajectory.json")
        print()


if __name__ == "__main__":
    main()
