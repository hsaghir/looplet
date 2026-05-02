"""Tool definitions for the coder example.

Pure functions and dataclasses that describe *what the agent can do*:
read and write files, run bash, search the workspace.  No agent
control flow, no hook logic — just the executable surface area.

The module exports a single composition function,
:func:`make_tools`, that wires every ``@tool``-decorated callable
into a :class:`looplet.tools.BaseToolRegistry` ready to plug into
either the library entrypoint or the runnable bundle.

:class:`FileCache` is kept here because every tool in the bundle
either reads from it (``read_file``) or invalidates it
(``write_file``, ``edit_file``); colocating the cache with the
tools that use it keeps the public surface small.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from looplet import tool, tools_from

__all__ = [
    "FileCache",
    "make_tools",
    "_EXIT_CODE_MAP",
    "_interpret_exit_code",
    "_run",
    "_fuzzy_find",
    "_resolve_safe_path",
    "_is_path_inside",
]


# ── Exit code interpretation ───────────────────────────────────────


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


# ── Bash + fuzzy matching helpers ──────────────────────────────────


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


def _resolve_safe_path(workspace: str, file_path: str) -> Path | None:
    """Resolve file_path relative to workspace, rejecting traversal.

    Returns the resolved Path if it is inside the workspace, or None
    if the path escapes (via ``..`` or absolute path).
    """
    ws = Path(workspace).resolve()
    target = (ws / file_path).resolve()
    if not _is_path_inside(target, ws):
        return None
    return target


def _is_path_inside(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


# ── Bash safety helpers ────────────────────────────────────────────


# Commands that destroy data or alter system state. The bash tool
# refuses to run any pipeline whose first token (or, after a
# control-operator split, the first token of any subcommand) appears
# in this set without the user's permission engine having explicitly
# allowed it. The list intentionally covers the common footguns rather
# than trying to be exhaustive — anything novel still flows through
# the regular permission engine, which is the principled gate.
_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset(
    {
        "dd",  # raw block-device writes
        "mkfs",  # filesystem creation
        "shred",  # secure delete
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "kill",  # process termination (kills any pid; use Ctrl-C in tests)
        "killall",
        "pkill",
    }
)

# Argument-pattern flags that turn an otherwise-safe command into a
# destructive one (e.g. ``rm -rf`` vs ``rm`` alone — the latter is
# fine for individual files). Maps command name → tuple of flag
# patterns that elevate it.
_DESTRUCTIVE_FLAGS: dict[str, tuple[str, ...]] = {
    "rm": ("-rf", "-fr", "-Rf", "-fR", "--recursive"),
    "git": (
        "push --force",
        "push -f",
        "reset --hard",
        "clean -fd",
        "branch -D",
        "checkout --force",
    ),
}


def classify_bash_command(command: str) -> dict[str, Any]:
    r"""Inspect ``command`` and return a structured safety classification.

    Returns a dict with:

    * ``destructive`` (bool) — True when the command is in the
      destructive-name set or matches a destructive-flag pattern.
    * ``reasons`` (list[str]) — human-readable reasons; empty when
      the command is considered safe.
    * ``first_token`` (str) — the leading executable name parsed from
      the command (best-effort; complex shell syntax may not parse).

    The bash tool surfaces these as a model-actionable error rather
    than running the command. Callers building their own bash wrapper
    can use ``classify_bash_command`` directly to filter or warn.

    The classification is deliberately conservative: it covers
    *patterns the model commonly tries by mistake* (\`rm -rf\` on
    cwd, ``git push --force`` to a shared remote) and bows out for
    novel commands so the permission engine can decide. It is **not**
    a sandbox.
    """
    reasons: list[str] = []
    # Split on bash control operators so we can inspect each subcommand.
    # ``cmd1 && cmd2 || cmd3 ; cmd4 | cmd5 & cmd6`` → 6 tokens.
    parts = re.split(r"&&|\|\||\||;|&", command)
    first_token = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # The leading word is the command name; strip leading parens
        # / brackets that wrap subshells (best-effort; not a parser).
        head = part.lstrip("()[]{ \t").split()
        if not head:
            continue
        name = head[0]
        if not first_token:
            first_token = name
        if name in _DESTRUCTIVE_COMMANDS:
            reasons.append(f"destructive command {name!r} (in {sorted(_DESTRUCTIVE_COMMANDS)})")
            continue
        bad_flags = _DESTRUCTIVE_FLAGS.get(name, ())
        for flag in bad_flags:
            # Substring match works for both single-token (-rf) and
            # multi-token (push --force) patterns.
            if flag in part:
                reasons.append(f"destructive flag {flag!r} in {name!r} subcommand")
                break
    return {
        "destructive": bool(reasons),
        "reasons": reasons,
        "first_token": first_token,
    }


def classify_sed_command(command: str) -> dict[str, Any]:
    """Detect ``sed -i`` (in-place edit) usage in ``command``.

    ``sed -i`` rewrites files outside the model's read-then-edit
    discipline and bypasses the file_cache invalidation that
    ``edit_file`` does — so a subsequent ``read_file`` may return
    cached pre-edit content. Surface a model-actionable error
    pointing the model at ``edit_file`` instead of refusing
    silently.

    Returns a dict with:

    * ``in_place_edit`` (bool) — True when any subcommand is
      ``sed -i ...`` or ``sed --in-place ...``.
    * ``recommendation`` (str) — short guidance ("use edit_file
      instead") when ``in_place_edit``; empty otherwise.
    """
    parts = re.split(r"&&|\|\||\||;|&", command)
    for part in parts:
        part = part.strip()
        # Token-aware: avoid matching ``sed-i`` package names or
        # ``echo "sed -i"`` content.
        if not part.startswith("sed "):
            continue
        # Look for ``-i`` as a standalone flag or a flag with bundled
        # backup-suffix arg (``-i.bak``); also ``--in-place``.
        tokens = part.split()
        for tok in tokens[1:]:
            if tok == "-i" or tok.startswith("-i.") or tok == "--in-place":
                return {
                    "in_place_edit": True,
                    "recommendation": (
                        "Use the edit_file tool for in-place file edits. "
                        "sed -i bypasses the file cache, so the next "
                        "read_file can return stale content."
                    ),
                }
    return {"in_place_edit": False, "recommendation": ""}


_VIEW_COMMANDS: frozenset[str] = frozenset({"cat", "head", "tail", "less", "more"})


def classify_view_command(command: str) -> dict[str, Any]:
    """Detect ``cat``/``head``/``tail``/``less``/``more`` used to view a single
    source file (which bypasses the read_file file_cache).

    Heuristics:

    * Only flags the *first* subcommand of the pipeline, because using
      ``... | head`` to trim *output* of another command is fine.
    * Refuses only when the command's positional args look like file
      paths (no ``-`` stdin marker, at least one arg without a leading
      ``-`` that points at an existing-looking source file).
    * Lets ``cat``-as-pipe-source through when the file is clearly not
      project source (e.g. ``/proc/...``, ``/dev/...``, ``/tmp/`` outside
      the workspace) — but the conservative default is to refuse so the
      model self-corrects to ``read_file``.

    Returns ``{"viewing_file": bool, "first_token": str, "recommendation": str}``.
    """
    parts = re.split(r"&&|\|\||;", command)
    first = parts[0].strip() if parts else ""
    tokens = first.split()
    if not tokens:
        return {"viewing_file": False, "first_token": "", "recommendation": ""}
    name = tokens[0].rsplit("/", 1)[-1]
    if name not in _VIEW_COMMANDS:
        return {"viewing_file": False, "first_token": name, "recommendation": ""}
    # Look at positional args (skip flags). If none look like files,
    # this is probably reading from stdin / a pipe — let it through.
    positional = [t for t in tokens[1:] if not t.startswith("-")]
    if not positional:
        return {"viewing_file": False, "first_token": name, "recommendation": ""}

    # Allow virtual / device paths (these are never project source).
    def is_project_path(arg: str) -> bool:
        if arg.startswith(("/proc/", "/dev/", "/sys/")):
            return False
        return True

    if not any(is_project_path(p) for p in positional):
        return {"viewing_file": False, "first_token": name, "recommendation": ""}
    return {
        "viewing_file": True,
        "first_token": name,
        "recommendation": (
            f"Use read_file to view source files. `{name}` bypasses "
            "the file_cache, so a subsequent edit_file on the same "
            "path will be refused with 'not been read in current "
            "session'. Pipe-trimming output of another command "
            "(e.g. `grep ... | head`) is still fine."
        ),
    }


# ── File cache ─────────────────────────────────────────────────────


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
        self._read_set: set[str] = set()  # every path ever passed to record()

    def record(self, path: str) -> None:
        p = Path(self._workspace) / path
        if not p.exists() or not p.is_file():
            return
        try:
            content = p.read_text()
            self._hashes[path] = hashlib.sha256(content.encode()).hexdigest()[:16]
            self._mtimes[path] = p.stat().st_mtime
            self._read_set.add(path)
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

    def was_read(self, path: str) -> bool:
        """True iff ``read_file`` has been called for ``path`` in this
        session. Used by ``edit_file`` to enforce read-before-edit:
        editing a file the model hasn't read is almost always a sign
        the model is guessing at content. The error returned by
        ``edit_file`` in that case names the read tool explicitly so
        the model fixes the next call rather than retrying with
        different text.
        """
        return path in self._read_set

    def render(self) -> str:
        if not self._recent:
            return ""
        parts = ["## Recently accessed files (cached)"]
        for path in self._order:
            content = self._recent.get(path, "")
            parts.append(f"\n### {path}\n```\n{content}\n```")
        return "\n".join(parts)


# ── Tool registry ──────────────────────────────────────────────────


def make_tools(workspace: str, file_cache: FileCache):
    @tool(description="Execute a bash command in the project directory.", timeout_s=600)
    def bash(*, command: str) -> dict:
        # CWD safety: detect cd outside workspace
        result = _run(command, workspace)
        # Check if command tried to cd outside workspace
        parts = re.split(r"&&|\|\||;|\n", command)
        for part in parts:
            part = part.strip()
            if part.startswith("cd "):
                target = part[3:].strip().strip("'\"")
                resolved = Path(workspace) / target
                try:
                    resolved = resolved.resolve()
                    ws_resolved = Path(workspace).resolve()
                    if not _is_path_inside(resolved, ws_resolved):
                        result["cwd_warning"] = (
                            f"Warning: 'cd {target}' points outside the project directory. "
                            f"All commands run in the project root. Use relative paths."
                        )
                except Exception:
                    pass
        return result

    @tool(
        description="List directory contents as a tree. Use at the start to understand project structure.",
        concurrent_safe=True,
    )
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

    @tool(
        description="Read a file with line numbers. Optionally specify start_line and/or end_line.",
        concurrent_safe=True,
    )
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

    @tool(description="Create or overwrite a file. Use for new files only.")
    def write_file(*, file_path: str, content: str) -> dict:
        p = _resolve_safe_path(workspace, file_path)
        if p is None:
            return {"error": f"Path '{file_path}' is outside the project directory."}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        file_cache.invalidate(file_path)
        return {"written": file_path, "lines": content.count("\n") + 1}

    @tool(description="Edit a file by replacing an exact string. Always read_file first.")
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

    @tool(description="Find files by glob pattern.", concurrent_safe=True)
    def glob(*, pattern: str) -> dict:
        return {
            "pattern": pattern,
            "matches": sorted(
                str(path.relative_to(workspace))
                for path in Path(workspace).glob(pattern)
                if path.is_file()
            )[:100],
        }

    @tool(
        description="Search file contents with regex. Returns file:line:content.",
        concurrent_safe=True,
    )
    def grep(*, pattern: str, path: str = ".", include: str = "") -> dict:
        target = _resolve_safe_path(workspace, path)
        if target is None:
            return {"error": f"Path '{path}' is outside the project directory."}
        cmd = ["grep", "-rn"]
        if include:
            cmd.append(f"--include={include}")
        cmd.extend(["--", pattern, str(target)])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=workspace,
            )
        except subprocess.TimeoutExpired:
            return {"error": "Search timed out", "pattern": pattern, "matches": [], "count": 0}
        lines = result.stdout.splitlines() if result.stdout else []
        workspace_prefix = str(Path(workspace).resolve()) + os.sep
        relative_lines = [
            line.removeprefix(workspace_prefix) if line.startswith(workspace_prefix) else line
            for line in lines
        ]
        data = {"pattern": pattern, "matches": relative_lines[:50], "count": len(relative_lines)}
        if result.returncode not in (0, 1):
            data["error"] = result.stderr.strip() or f"grep exited {result.returncode}"
        return data

    return tools_from(
        [bash, list_dir, read_file, write_file, edit_file, glob, grep],
        include_think=True,
        include_done=True,
    )
