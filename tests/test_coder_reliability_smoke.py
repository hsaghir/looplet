"""Coder agent reliability features — unit tests.

Tests for the coder-example-level reliability improvements:
- Exit code interpretation
- FileCache with mtime/hash tracking
- file_unchanged optimization in read_file
- old==new guard in edit_file
- Structured diff in edit result
- Stale-file detection after bash
- CWD safety in bash
- LinterHook detection
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

# Import from the coder example
from examples.coder.agent import (
    FileCache,
    LinterHook,
    StaleFileHook,
    _interpret_exit_code,
    _run,
    make_tools,
)
from looplet.types import ToolCall

# ── Exit code interpretation ─────────────────────────────────────


class TestExitCodeInterpretation:
    def test_diff_exit_1(self) -> None:
        assert _interpret_exit_code("diff -u a b", 1) == "files differ (not an error)"

    def test_grep_exit_1(self) -> None:
        assert _interpret_exit_code("grep foo bar", 1) == "no match found (not an error)"

    def test_ruff_check_exit_1(self) -> None:
        assert _interpret_exit_code("ruff check src/", 1) == "lint issues found"

    def test_normal_exit_0_returns_none(self) -> None:
        assert _interpret_exit_code("ls", 0) is None

    def test_unknown_command_exit_1_returns_none(self) -> None:
        assert _interpret_exit_code("unknown_cmd", 1) is None

    def test_exact_command_match(self) -> None:
        assert _interpret_exit_code("diff", 1) == "files differ (not an error)"

    def test_prefix_with_space(self) -> None:
        assert _interpret_exit_code("grep -rn pattern .", 1) is not None

    def test_prefix_without_space_no_match(self) -> None:
        # "differ" should NOT match "diff"
        assert _interpret_exit_code("differ", 1) is None


# ── FileCache mtime/hash tracking ────────────────────────────────


class TestFileCacheMtime:
    def test_record_stores_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello")
            cache = FileCache(tmp)
            cache.record("test.py")
            assert "test.py" in cache._mtimes
            assert "test.py" in cache._hashes

    def test_stale_files_detects_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello")
            cache = FileCache(tmp)
            cache.record("test.py")
            assert cache.stale_files() == []

            # Modify file (ensure mtime changes)
            time.sleep(0.05)
            p.write_text("world")
            os.utime(str(p), (time.time() + 1, time.time() + 1))
            stale = cache.stale_files()
            assert stale == ["test.py"]

    def test_is_unchanged_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello")
            cache = FileCache(tmp)
            cache.record("test.py")
            assert cache.is_unchanged("test.py") is True

    def test_is_unchanged_false_after_modify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello")
            cache = FileCache(tmp)
            cache.record("test.py")
            p.write_text("world")
            assert cache.is_unchanged("test.py") is False

    def test_is_unchanged_unknown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            assert cache.is_unchanged("nonexistent.py") is False


# ── Tool-level features ─────────────────────────────────────────


class TestReadFileUnchanged:
    def test_file_unchanged_returns_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello\nworld\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            # First read — full content
            r1 = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            assert "content" in r1.data

            # Second read (unchanged) — minimal
            r2 = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            assert r2.data.get("file_unchanged") is True
            assert "content" not in r2.data

    def test_file_changed_returns_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            p.write_text("world\n")
            r2 = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            assert "content" in r2.data
            assert "file_unchanged" not in r2.data

    def test_ranged_read_skips_unchanged_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("line1\nline2\nline3\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            # Ranged read always returns content
            r2 = tools.dispatch(
                ToolCall(tool="read_file", args={"file_path": "test.py", "start_line": 2})
            )
            assert "content" in r2.data

    def test_file_unchanged_false_after_edit(self) -> None:
        """After edit_file, read_file must return full content, not file_unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello\nworld\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            tools.dispatch(
                ToolCall(
                    tool="edit_file",
                    args={"file_path": "test.py", "old_string": "hello", "new_string": "goodbye"},
                )
            )
            r = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            assert r.data.get("file_unchanged") is not True
            assert "content" in r.data
            assert "goodbye" in r.data["content"]

    def test_file_unchanged_false_after_write(self) -> None:
        """After write_file, read_file must return full content."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            tools.dispatch(
                ToolCall(
                    tool="write_file",
                    args={"file_path": "test.py", "content": "overwritten\n"},
                )
            )
            r = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "test.py"}))
            assert "content" in r.data
            assert "overwritten" in r.data["content"]


class TestPathTraversal:
    """File tools must reject paths outside the workspace."""

    def test_read_file_rejects_dotdot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)
            r = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "../../etc/passwd"}))
            assert "error" in r.data
            assert "outside" in r.data["error"]

    def test_read_file_rejects_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)
            r = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "/etc/hostname"}))
            assert "error" in r.data
            assert "outside" in r.data["error"]

    def test_write_file_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)
            r = tools.dispatch(
                ToolCall(tool="write_file", args={"file_path": "../escape.txt", "content": "x"})
            )
            assert "error" in r.data
            assert "outside" in r.data["error"]
            # File must NOT exist outside workspace
            assert not (Path(tmp).parent / "escape.txt").exists()

    def test_edit_file_rejects_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)
            r = tools.dispatch(
                ToolCall(
                    tool="edit_file",
                    args={"file_path": "/etc/hostname", "old_string": "x", "new_string": "y"},
                )
            )
            assert "error" in r.data
            assert "outside" in r.data["error"]

    def test_relative_path_inside_workspace_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "sub" / "test.txt").write_text("hello\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)
            r = tools.dispatch(ToolCall(tool="read_file", args={"file_path": "sub/test.txt"}))
            assert "content" in r.data


class TestEditFileGuards:
    def test_old_equals_new_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            r = tools.dispatch(
                ToolCall(
                    tool="edit_file",
                    args={
                        "file_path": "test.py",
                        "old_string": "hello",
                        "new_string": "hello",
                    },
                )
            )
            assert r.data.get("error") is not None
            assert "identical" in r.data["error"]

    def test_successful_edit_returns_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello\nworld\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            r = tools.dispatch(
                ToolCall(
                    tool="edit_file",
                    args={
                        "file_path": "test.py",
                        "old_string": "hello",
                        "new_string": "goodbye",
                    },
                )
            )
            assert r.data["edited"] == "test.py"
            assert "diff" in r.data
            assert "-hello" in r.data["diff"]
            assert "+goodbye" in r.data["diff"]


class TestBashExitCodeInResult:
    def test_diff_exit_1_gets_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Create two different files to diff
            (Path(tmp) / "a.txt").write_text("hello\n")
            (Path(tmp) / "b.txt").write_text("world\n")
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            r = tools.dispatch(ToolCall(tool="bash", args={"command": "diff a.txt b.txt || true"}))
            # Note: diff || true exits 0, so let's test with actual diff
            r2 = tools.dispatch(
                ToolCall(tool="bash", args={"command": "diff a.txt b.txt; echo done"})
            )
            # The exit code comes from echo (0), not diff
            # Let's test _run directly
            result = _run("diff a.txt b.txt", tmp)
            assert result["exit_code"] == 1
            assert result.get("exit_code_note") == "files differ (not an error)"

    def test_normal_command_no_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = _run("echo hello", tmp)
            assert result["exit_code"] == 0
            assert "exit_code_note" not in result


class TestBashCwdSafety:
    def test_cd_outside_workspace_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            r = tools.dispatch(ToolCall(tool="bash", args={"command": "cd /tmp && ls"}))
            data = r.data or {}
            # /tmp is outside workspace, should warn
            if str(Path(tmp).resolve()) != str(Path("/tmp").resolve()):
                assert "cwd_warning" in data

    def test_cd_inside_workspace_no_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subdir = Path(tmp) / "sub"
            subdir.mkdir()
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)

            r = tools.dispatch(ToolCall(tool="bash", args={"command": "cd sub && ls"}))
            data = r.data or {}
            assert "cwd_warning" not in data


# ── Hooks ────────────────────────────────────────────────────────


class TestStaleFileHook:
    def test_detects_stale_after_bash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello")
            cache = FileCache(tmp)
            cache.record("test.py")

            # Simulate bash modifying the file
            time.sleep(0.05)
            p.write_text("modified")
            os.utime(str(p), (time.time() + 1, time.time() + 1))

            hook = StaleFileHook(cache)
            from looplet.types import ToolCall as TC
            from looplet.types import ToolResult as TR

            tc = TC(tool="bash", args={"command": "echo modify"})
            tr = TR(tool="bash", args_summary="", data={"exit_code": 0})
            result = hook.post_dispatch(None, None, tc, tr, 1)
            assert result is not None
            assert "test.py" in str(result)

    def test_no_stale_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.py"
            p.write_text("hello")
            cache = FileCache(tmp)
            cache.record("test.py")

            hook = StaleFileHook(cache)
            from looplet.types import ToolCall as TC
            from looplet.types import ToolResult as TR

            tc = TC(tool="bash", args={"command": "echo noop"})
            tr = TR(tool="bash", args_summary="", data={"exit_code": 0})
            result = hook.post_dispatch(None, None, tc, tr, 1)
            assert result is None


class TestLinterHook:
    def test_skips_non_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hook = LinterHook(tmp)
            from looplet.types import ToolCall as TC
            from looplet.types import ToolResult as TR

            tc = TC(tool="edit_file", args={"file_path": "test.js"})
            tr = TR(tool="edit_file", args_summary="", data={"edited": "test.js"})
            result = hook.post_dispatch(None, None, tc, tr, 1)
            assert result is None

    def test_skips_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hook = LinterHook(tmp)
            from looplet.types import ToolCall as TC
            from looplet.types import ToolResult as TR

            tc = TC(tool="edit_file", args={"file_path": "test.py"})
            tr = TR(tool="edit_file", args_summary="", data=None, error="failed")
            result = hook.post_dispatch(None, None, tc, tr, 1)
            assert result is None

    def test_skips_non_edit_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hook = LinterHook(tmp)
            from looplet.types import ToolCall as TC
            from looplet.types import ToolResult as TR

            tc = TC(tool="bash", args={"command": "echo"})
            tr = TR(tool="bash", args_summary="", data={"exit_code": 0})
            result = hook.post_dispatch(None, None, tc, tr, 1)
            assert result is None


class TestBashTimeout:
    def test_bash_has_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(tmp)
            tools = make_tools(tmp, cache)
            spec = tools._tools["bash"]
            assert spec.timeout_s == 600
