#!/usr/bin/env python3
"""looplet coder — a production-grade coding agent built on looplet.

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
import difflib
import hashlib
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
from looplet.types import ToolContext  # noqa: F401

# ═══════════════════════════════════════════════════════════════════
# EXIT CODE INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

# Maps (command_prefix, exit_code) → human explanation so the model
# doesn't misinterpret non-error exit codes as failures.
_EXIT_CODE_MAP: dict[str, dict[int, str]] = {
    "diff": {1: "files differ (not an error)"},
    "grep": {1: "no match found (not an error)"},
    "test": {1: "expression evaluated to false"},
    "cmp": {1: "files differ (not an error)"},
    "ruff check": {1: "lint issues found"},
    "mypy": {1: "type errors found"},
    "pylint": {1: "lint issues found (fatal=1)", 2: "error (2)", 4: "warning (4)"},
}


def _interpret_exit_code(cmd: str, exit_code: int) -> str | None:
    """Return a human-readable interpretation of a non-zero exit code, or None."""
    if exit_code == 0:
        return None
    for prefix, codes in _EXIT_CODE_MAP.items():
        # Match command prefix (e.g. 'diff' matches 'diff -u a b')
        cmd_stripped = cmd.lstrip()
        if cmd_stripped == prefix or cmd_stripped.startswith(prefix + " "):
            if exit_code in codes:
                return codes[exit_code]
    return None


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════


def _run(cmd: str, cwd: str, timeout: int = 120) -> dict:
    # Sanitize: strip whitespace/newlines, handle None/empty
    if not cmd or not isinstance(cmd, str):
        return {
            "stdout": "",
            "stderr": "Error: empty command. Provide a bash command to execute.",
            "exit_code": 1,
        }
    cmd = cmd.strip()
    if not cmd:
        return {
            "stdout": "",
            "stderr": "Error: empty command after stripping whitespace.",
            "exit_code": 1,
        }
    # Remove common LLM artifacts: leading $, backtick fences
    if cmd.startswith("$ "):
        cmd = cmd[2:]
    if cmd.startswith("```") and cmd.endswith("```"):
        cmd = cmd[3:-3].strip()
        if cmd.startswith("bash\n") or cmd.startswith("sh\n"):
            cmd = cmd.split("\n", 1)[1].strip()
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
        result: dict = {"stdout": stdout, "stderr": stderr, "exit_code": r.returncode}
        # Add semantic exit code interpretation
        interpretation = _interpret_exit_code(cmd, r.returncode)
        if interpretation:
            result["exit_code_note"] = interpretation
        return result
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


def _fuzzy_find(text: str, target: str, threshold: float = 0.6) -> list[tuple[int, float, str]]:
    """Find approximate matches for target's first line in text."""
    target_lines = target.splitlines()
    text_lines = text.splitlines()
    if not target_lines or not text_lines:
        return []
    first_target = target_lines[0].strip()
    matches = []
    for i, line in enumerate(text_lines):
        ratio = difflib.SequenceMatcher(None, line.strip(), first_target).ratio()
        if ratio >= threshold:
            matches.append((i + 1, ratio, line))
    return sorted(matches, key=lambda x: -x[1])[:5]


# ═══════════════════════════════════════════════════════════════════
# FILE CACHE — survives compaction, re-injected into each prompt
# ═══════════════════════════════════════════════════════════════════


class FileCache:
    """Tracks recently read/written files for re-injection after compaction.

    Also tracks file mtimes for stale-file detection: when a bash
    command modifies files that were previously read, the cache can
    report which files have stale content so the model re-reads them.
    """

    def __init__(self, workspace: str, max_files: int = 5, max_chars: int = 8000):
        self._workspace = workspace
        self._max_files = max_files
        self._max_chars = max_chars
        self._recent: dict[str, str] = {}
        self._order: list[str] = []
        self._mtimes: dict[str, float] = {}  # path -> mtime when last read
        self._hashes: dict[str, str] = {}  # path -> content hash when last read

    def record(self, path: str) -> None:
        p = Path(self._workspace) / path
        if not p.exists() or not p.is_file():
            return
        try:
            content = p.read_text()
            self._hashes[path] = hashlib.sha256(content.encode()).hexdigest()[:16]
            self._mtimes[path] = p.stat().st_mtime
            if len(content) > self._max_chars:
                content = content[: self._max_chars] + "\n... [truncated]"
            self._recent[path] = content
            if path in self._order:
                self._order.remove(path)
            self._order.append(path)
            while len(self._order) > self._max_files:
                old = self._order.pop(0)
                self._recent.pop(old, None)
        except Exception:
            pass

    def stale_files(self) -> list[str]:
        """Return paths of files whose mtime changed since last read."""
        stale = []
        for path, cached_mtime in self._mtimes.items():
            p = Path(self._workspace) / path
            try:
                current_mtime = p.stat().st_mtime
                if current_mtime != cached_mtime:
                    stale.append(path)
            except OSError:
                pass
        return stale

    def is_unchanged(self, path: str) -> bool:
        """True if the file content hasn't changed since last read."""
        if path not in self._hashes:
            return False
        p = Path(self._workspace) / path
        try:
            content = p.read_text()
            current_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            return current_hash == self._hashes[path]
        except OSError:
            return False

    def invalidate(self, path: str) -> None:
        """Mark a file as modified so is_unchanged() returns False.

        Call this from write_file/edit_file instead of record(). The
        model hasn't *seen* the new content via read_file, so the
        file_unchanged optimization must not fire on the next read.
        """
        self._hashes.pop(path, None)

    def render(self) -> str:
        if not self._recent:
            return ""
        parts = ["## Recently accessed files (cached)"]
        for path in self._order:
            content = self._recent.get(path, "")
            parts.append(f"\n### {path}\n```\n{content}\n```")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════


def _resolve_safe_path(workspace: str, file_path: str) -> Path | None:
    """Resolve file_path relative to workspace, rejecting traversal.

    Returns the resolved Path if it is inside the workspace, or None
    if the path escapes (via ``..`` or absolute path).
    """
    ws = Path(workspace).resolve()
    target = (ws / file_path).resolve()
    if not str(target).startswith(str(ws) + "/") and target != ws:
        return None
    return target


def make_tools(workspace: str, file_cache: FileCache) -> BaseToolRegistry:
    tools = BaseToolRegistry()

    # ── bash ────────────────────────────────────────────────────
    def bash_execute(*, command: str) -> dict:
        # CWD safety: detect cd outside workspace
        result = _run(command, workspace)
        # Check if command tried to cd outside workspace
        import re  # noqa: PLC0415

        parts = re.split(r"&&|\|\||;|\n", command)
        for part in parts:
            part = part.strip()
            if part.startswith("cd "):
                target = part[3:].strip().strip("'\"")
                resolved = Path(workspace) / target
                try:
                    resolved = resolved.resolve()
                    ws_resolved = Path(workspace).resolve()
                    if not str(resolved).startswith(str(ws_resolved)):
                        result["cwd_warning"] = (
                            f"Warning: 'cd {target}' points outside the project directory. "
                            f"All commands run in the project root. Use relative paths."
                        )
                except Exception:
                    pass
        return result

    tools.register(
        ToolSpec(
            name="bash",
            description="Execute a bash command in the project directory.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command"}},
                "required": ["command"],
            },
            execute=bash_execute,
            timeout_s=600,  # 10 minute cap
        )
    )

    # ── list_dir ────────────────────────────────────────────────
    def list_dir(*, path: str = ".", depth: int = 2) -> dict:
        target = Path(workspace) / path
        if not target.exists():
            return {"error": f"Not found: {path}"}
        if not target.is_dir():
            return {"error": f"Not a directory: {path}"}
        skip = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
        }
        entries: list[str] = []

        def _walk(p: Path, prefix: str, d: int) -> None:
            if d > depth:
                return
            try:
                items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            except PermissionError:
                return
            for item in items:
                if item.name in skip:
                    continue
                if item.is_dir():
                    entries.append(f"{prefix}{item.name}/")
                    _walk(item, prefix + "  ", d + 1)
                elif len(entries) < 200:
                    entries.append(f"{prefix}{item.name}")

        _walk(target, "", 0)
        return {"path": path, "entries": entries, "count": len(entries)}

    tools.register(
        ToolSpec(
            name="list_dir",
            description="List directory contents as a tree. Use at the start to understand project structure.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "depth": {"type": "integer", "default": 2},
                },
                "required": [],
            },
            execute=list_dir,
            concurrent_safe=True,
        )
    )

    # ── read_file ───────────────────────────────────────────────
    def read_file(*, file_path: str, start_line: int = 0, end_line: int = 0) -> dict:
        p = _resolve_safe_path(workspace, file_path)
        if p is None:
            return {"error": f"Path '{file_path}' is outside the project directory."}
        if not p.exists():
            return {"error": f"File not found: {file_path}"}
        # file_unchanged optimization: skip full content if unchanged
        # since last *read* (not edit — edits update the hash but the
        # model hasn't seen the new content via read_file yet).
        if start_line == 0 and end_line == 0 and file_cache.is_unchanged(file_path):
            return {
                "path": file_path,
                "file_unchanged": True,
                "note": "File has not changed since your last read. No need to re-read.",
            }
        try:
            lines = p.read_text().splitlines()
            if start_line > 0 and end_line > 0:
                selected = lines[start_line - 1 : end_line]
                numbered = [f"{start_line + i:>4} | {line}" for i, line in enumerate(selected)]
            elif start_line > 0:
                selected = lines[start_line - 1 :]
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
            file_cache.record(file_path)
            return {"path": file_path, "content": content, "total_lines": len(lines)}
        except Exception as e:
            return {"error": str(e)}

    tools.register(
        ToolSpec(
            name="read_file",
            description="Read a file with line numbers. Optionally specify start_line and/or end_line (1-indexed).",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 0},
                    "end_line": {"type": "integer", "default": 0},
                },
                "required": ["file_path"],
            },
            execute=read_file,
            concurrent_safe=True,
        )
    )

    # ── write_file ──────────────────────────────────────────────
    def write_file(*, file_path: str, content: str) -> dict:
        p = _resolve_safe_path(workspace, file_path)
        if p is None:
            return {"error": f"Path '{file_path}' is outside the project directory."}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        file_cache.invalidate(file_path)
        return {"written": file_path, "lines": content.count("\n") + 1}

    tools.register(
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file. Use for NEW files only. Use edit_file for existing files.",
            parameters={
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"],
            },
            execute=write_file,
        )
    )

    # ── edit_file (search-and-replace with fuzzy fallback) ──────
    def edit_file(*, file_path: str, old_string: str, new_string: str) -> dict:
        p = _resolve_safe_path(workspace, file_path)
        if p is None:
            return {"error": f"Path '{file_path}' is outside the project directory."}
        if not p.exists():
            return {"error": f"File not found: {file_path}"}
        # Guard: no-op edit wastes a step
        if old_string == new_string:
            return {"error": "old_string and new_string are identical. No change needed."}
        text = p.read_text()
        count = text.count(old_string)
        if count == 1:
            new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text)
            file_cache.invalidate(file_path)
            # Structured diff for model verification
            diff = difflib.unified_diff(
                text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=3,
            )
            diff_text = "".join(diff)
            if len(diff_text) > 2000:
                diff_text = diff_text[:2000] + "\n... [diff truncated]"
            return {"edited": file_path, "replacements": 1, "diff": diff_text}
        if count > 1:
            return {
                "error": f"Matches {count} locations. Include more surrounding context for a unique match.",
                "matches": count,
            }
        # Fuzzy fallback
        fuzzy = _fuzzy_find(text, old_string)
        if fuzzy:
            hints = [f"  line {n} ({r:.0%}): {t.strip()[:80]}" for n, r, t in fuzzy[:3]]
            return {
                "error": "Exact match not found. Similar lines:\n"
                + "\n".join(hints)
                + "\n\nRECOVERY: read_file at those lines, then retry with exact text.",
                "similar_lines": [f[0] for f in fuzzy[:3]],
            }
        return {"error": f"Not found in {file_path}. Use read_file to see exact content."}

    tools.register(
        ToolSpec(
            name="edit_file",
            description="Edit a file by replacing an exact string. ALWAYS read_file first. Include 3+ context lines for unique match. If it fails, read the error hints and retry.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            execute=edit_file,
        )
    )

    # ── glob + grep ─────────────────────────────────────────────
    tools.register(
        ToolSpec(
            name="glob",
            description="Find files by glob pattern (e.g. '**/*.py').",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            execute=lambda *, pattern: {
                "pattern": pattern,
                "matches": sorted(
                    str(p.relative_to(workspace))
                    for p in Path(workspace).glob(pattern)
                    if p.is_file()
                )[:100],
            },
            concurrent_safe=True,
        )
    )
    tools.register(
        ToolSpec(
            name="grep",
            description="Search file contents with regex. Returns file:line:content.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "include": {"type": "string", "default": ""},
                },
                "required": ["pattern"],
            },
            execute=lambda *, pattern, path=".", include="": (
                lambda r: {
                    "pattern": pattern,
                    "matches": r["stdout"].splitlines()[:50] if r["stdout"] else [],
                    "count": len(r["stdout"].splitlines()) if r["stdout"] else 0,
                }
            )(
                _run(
                    f"grep -rn {f'--include={chr(39)}{include}{chr(39)} ' if include else ''}{chr(39)}{pattern}{chr(39)} {chr(39)}{path}{chr(39)} 2>/dev/null | head -50",
                    workspace,
                    timeout=10,
                )
            ),
            concurrent_safe=True,
        )
    )

    register_done_tool(tools)
    register_think_tool(tools)
    return tools


# ═══════════════════════════════════════════════════════════════════
# AUTO-DISCOVERED INSTRUCTIONS + CONTEXT
# ═══════════════════════════════════════════════════════════════════


def _discover_instructions(workspace: str) -> str:
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
            parts.append(f"## From {name}\n{p.read_text()[:4000]}")
    return "\n\n".join(parts)


def _project_context(workspace: str) -> str:
    parts = []
    try:
        branch = subprocess.run(
            ["git", "-C", workspace, "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if branch:
            parts.append(f"branch={branch}")
    except Exception:
        pass
    for n in ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"]:
        if (Path(workspace) / n).exists():
            parts.append(n)
    return " | ".join(parts) if parts else "unknown"


# ═══════════════════════════════════════════════════════════════════
# HOOKS
# ═══════════════════════════════════════════════════════════════════


class TestGuardHook:
    def __init__(self):
        self._tests_passed = False
        self._files_written: set[str] = set()

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool == "bash":
            cmd = tool_call.args.get("command", "")
            data = tool_result.data or {}
            if any(
                t in cmd
                for t in ["pytest", "python -m pytest", "npm test", "cargo test", "go test"]
            ):
                self._tests_passed = data.get("exit_code", 1) == 0
                if not self._tests_passed:
                    return InjectContext(
                        "⚠ Tests FAILED. Read the traceback. Find the exact file:line. Read that code. Fix the issue. Re-run tests."
                    )
                return InjectContext("✓ Tests passed.")
        if tool_call.tool in ("write_file", "edit_file"):
            self._files_written.add(tool_call.args.get("file_path", ""))
        return None

    def check_done(self, state, session_log, context, step_num):
        if not self._tests_passed and self._files_written:
            return HookDecision(block="Run tests first. If no tests exist, create them.")
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class StaleFileHook:
    """Detects when bash commands modify files the model previously read.

    Mirrors Claude Code's staleReadFileStateHint: after each bash step,
    checks cached file mtimes and warns the model to re-read before editing.
    """

    def __init__(self, cache: FileCache):
        self._cache = cache

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool != "bash":
            return None
        # Check for stale files even on failed commands — a partial
        # build or interrupted write can still modify files.
        stale = self._cache.stale_files()
        if stale:
            return InjectContext(
                f"⚠ Stale files: {', '.join(stale)} were modified by this command. "
                f"Re-read them with read_file before editing."
            )
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class LinterHook:
    """Runs ruff check after Python file edits, injects diagnostics.

    Mirrors Claude Code's automatic LSP error reporting after file edits.
    Only runs when ruff is available in the workspace.
    """

    def __init__(self, workspace: str):
        self._workspace = workspace
        self._ruff_available: bool | None = None

    def _check_ruff(self) -> bool:
        if self._ruff_available is None:
            try:
                r = subprocess.run(["ruff", "--version"], capture_output=True, timeout=5)
                self._ruff_available = r.returncode == 0
            except Exception:
                self._ruff_available = False
        return self._ruff_available

    def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
        if tool_call.tool not in ("edit_file", "write_file"):
            return None
        file_path = tool_call.args.get("file_path", "")
        if not file_path.endswith(".py"):
            return None
        if tool_result.error:
            return None
        if not self._check_ruff():
            return None
        try:
            r = subprocess.run(
                ["ruff", "check", "--no-fix", file_path],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self._workspace,
            )
            if r.returncode != 0 and r.stdout.strip():
                lines = r.stdout.strip().splitlines()
                if len(lines) > 10:
                    lines = lines[:10] + [f"... and {len(lines) - 10} more issues"]
                return InjectContext(f"⚠ Lint issues in {file_path}:\n" + "\n".join(lines))
        except Exception:
            pass
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


class FileCacheHook:
    def __init__(self, cache: FileCache):
        self._cache = cache

    def pre_prompt(self, state, session_log, context, step_num):
        if step_num > 3:
            return self._cache.render() or None
        return None

    def should_stop(self, state, step_num, new_entities):
        return False


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert software engineer. You solve tasks by understanding \
the codebase, planning carefully, making precise changes, and verifying \
with tests. You never guess — you read first, then act.

## Workflow
1. EXPLORE: list_dir to see structure. glob/grep to find relevant files.
2. READ: read_file on files you need to modify. Understand patterns and conventions.
3. PLAN: think() to plan approach. Break complex tasks into steps.
4. IMPLEMENT: edit_file for existing files, write_file for new files. One file at a time.
5. TEST: bash to run tests after EVERY change. Read failures. Fix and re-run.
6. DONE: done() with summary only after tests pass.

## Tool rules
- ALWAYS read_file before edit_file. Never edit blind.
- edit_file: copy-paste old_string from read_file output exactly. Include 3+ context lines.
- If edit fails "not found": read the hint lines, re-read file at those lines, retry with exact text.
- If edit fails "multiple matches": add more surrounding lines for uniqueness.
- write_file: NEW files only. Never overwrite files you should edit.
- bash: use relative paths. pytest -xvs for tests (stop on first failure).
- For bugs: write a failing test FIRST, then fix the code.

## Code quality
- Follow existing project style and conventions.
- Type hints on function signatures. Docstrings on public functions.
- Minimal changes. Don't refactor unrelated code.
- If stuck after 3 attempts: think() to reconsider approach.
"""


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="looplet coder — AI coding agent")
    parser.add_argument("task", help="What to build or fix")
    parser.add_argument("--workspace", "-w", default=os.getcwd(), help="Project directory")
    parser.add_argument("--max-steps", type=int, default=30, help="Max tool calls")
    parser.add_argument("--no-tests", action="store_true", help="Skip test guard")
    args = parser.parse_args()
    workspace = os.path.abspath(args.workspace)

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "x")
    model = os.environ.get("OPENAI_MODEL", "llama3.1")

    llm = ResilientBackend(
        OpenAIBackend(base_url=base_url, api_key=api_key, model=model), retries=2, timeout_s=120
    )
    recording = RecordingLLMBackend(llm)
    file_cache = FileCache(workspace)
    tools = make_tools(workspace, file_cache)

    hooks: list = []
    if not args.no_tests:
        hooks.append(TestGuardHook())
    hooks.append(FileCacheHook(file_cache))
    hooks.append(StaleFileHook(file_cache))
    hooks.append(LinterHook(workspace))
    hooks.append(
        StagnationHook(
            fingerprint=tool_call_fingerprint,
            threshold=4,
            nudge="[stagnation] Re-read the file, try a different approach, or think().",
        )
    )
    hooks.append(PerToolLimitHook(default_limit=25, limits={"bash": 20, "read_file": 20}))
    events: list = []
    hooks.append(StreamingHook(CallbackEmitter(events.append)))

    instructions = _discover_instructions(workspace)
    project_ctx = _project_context(workspace)
    memory_sources = []
    if instructions:
        memory_sources.append(StaticMemorySource(instructions))
    memory_sources.append(
        CallableMemorySource(
            lambda state: f"[{project_ctx}] step {getattr(state, 'step_count', 0)}/{args.max_steps}"
        )
    )

    config = LoopConfig(
        max_steps=args.max_steps,
        temperature=0.2,
        system_prompt=SYSTEM_PROMPT,
        compact_service=compact_chain(
            PruneToolResults(keep_recent=10), TruncateCompact(keep_recent=5)
        ),
        memory_sources=memory_sources,
    )
    state = DefaultState(max_steps=args.max_steps)
    session_log = SessionLog()
    conv = Conversation()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              looplet coder                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Task: {args.task}")
    print(f"  Workspace: {workspace}")
    print(f"  Context: {project_ctx}")
    if instructions:
        print(f"  Instructions: {len(instructions)} chars")
    print(f"  Model: {model} | Budget: {args.max_steps} steps\n")

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
            if tool == "done":
                print(f"\n  ✓ Done: {data.get('summary', data.get('status', ''))[:120]}")
            elif tool == "think":
                print(f"  💭 #{step.number} {step.tool_call.args.get('analysis', '')[:100]}...")
            elif tool == "bash":
                print(
                    f"  {'✓' if data.get('exit_code') == 0 else '✗'} #{step.number} bash: {step.tool_call.args.get('command', '')[:60]}  [exit {data.get('exit_code', '?')}]"
                )
            elif tool == "read_file":
                print(
                    f"  📖 #{step.number} read: {step.tool_call.args.get('file_path', '?')} ({data.get('total_lines', '?')} lines)"
                )
            elif tool == "write_file":
                print(
                    f"  ✏️  #{step.number} write: {data.get('written', '?')} ({data.get('lines', '?')} lines)"
                )
            elif tool == "edit_file":
                print(
                    f"  {'✏️ ' if not err else '✗ '}#{step.number} edit: {step.tool_call.args.get('file_path', '?')}{' ✓' if not err else ' — ' + str(err)[:50]}"
                )
            elif tool == "list_dir":
                print(f"  📂 #{step.number} list_dir: {data.get('count', '?')} entries")
            elif tool == "glob":
                print(f"  🔍 #{step.number} glob: {len(data.get('matches', []))} files")
            elif tool == "grep":
                print(f"  🔍 #{step.number} grep: {data.get('count', '?')} matches")
            else:
                print(f"  → #{step.number} {tool}")

        scoped = [c for c in recording.calls if c.scope]
        print(
            f"\n  Steps: {len(state.steps)} | LLM calls: {len(recording.calls)} ({len(scoped)} tool-internal)\n"
        )


if __name__ == "__main__":
    main()
