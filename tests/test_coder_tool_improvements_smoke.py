"""Smoke tests for coder.workspace tool improvements.

Covers:
  1. ``edit_file`` refuses without a prior ``read_file`` in the session.
  2. ``classify_bash_command`` flags destructive patterns.
  3. ``classify_sed_command`` flags ``sed -i`` in-place edits.
  4. ``bash`` tool refuses destructive commands and ``sed -i``.
  5. Tool descriptions (loaded from YAML block scalars) include the
     rich multi-paragraph guidance.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Make ``coder_lib_tools`` importable for the classifier-only tests.
_CODER_DIR = Path(__file__).resolve().parents[1] / "examples" / "coder.workspace"
sys.path.insert(0, str(_CODER_DIR))

from coder_lib_tools import (  # noqa: E402
    classify_bash_command,
    classify_sed_command,
    classify_view_command,
)

from looplet import workspace_to_preset  # noqa: E402
from looplet.types import ToolCall  # noqa: E402


@pytest.fixture
def preset():
    with tempfile.TemporaryDirectory() as td:
        Path(td, "foo.py").write_text("hello\n")
        yield workspace_to_preset(str(_CODER_DIR), runtime={"workspace": td})


def test_edit_file_refuses_without_prior_read(preset):
    r = preset.tools.dispatch(
        ToolCall(
            tool="edit_file",
            args={"file_path": "foo.py", "old_string": "hello", "new_string": "bye"},
        )
    )
    assert r.data is not None
    assert "not been read" in r.data.get("error", "")
    assert r.data.get("missing") == "prior_read"
    assert "read_file" in r.data.get("recovery", "")


def test_edit_file_succeeds_after_read(preset):
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "foo.py"}))
    r = preset.tools.dispatch(
        ToolCall(
            tool="edit_file",
            args={"file_path": "foo.py", "old_string": "hello", "new_string": "bye"},
        )
    )
    assert r.data is not None
    assert r.data.get("replacements") == 1


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -fr build",
        "git push --force",
        "git push -f origin main",
        "git reset --hard HEAD~1",
        "echo done && rm -rf node_modules",
        "shutdown -h now",
        "mkfs /dev/sda1",
    ],
)
def test_classify_bash_command_detects_destructive(command):
    res = classify_bash_command(command)
    assert res["destructive"], f"expected destructive: {command}"
    assert res["reasons"]


@pytest.mark.parametrize(
    "command",
    [
        "echo hello",
        "ls -la",
        "git status",
        "rm foo.txt",  # plain rm without -rf is allowed
        "pytest tests/",
    ],
)
def test_classify_bash_command_passes_safe(command):
    res = classify_bash_command(command)
    assert not res["destructive"], f"expected safe: {command}"


@pytest.mark.parametrize(
    "command",
    ["sed -i s/a/b/ foo.py", "sed -i.bak s/a/b/ foo.py", "sed --in-place s/a/b/ foo.py"],
)
def test_classify_sed_command_detects_in_place(command):
    res = classify_sed_command(command)
    assert res["in_place_edit"], f"expected in-place: {command}"
    assert "edit_file" in res["recommendation"]


def test_classify_sed_command_passes_streaming():
    assert not classify_sed_command("sed s/a/b/ foo.py")["in_place_edit"]
    assert not classify_sed_command("cat foo | sed s/a/b/")["in_place_edit"]


@pytest.mark.parametrize(
    "command",
    [
        "cat src/foo.py",
        "cat -n src/foo.py",
        "head -20 src/foo.py",
        "tail -50 logs/err.log",
        "less README.md",
        "cat foo.py bar.py",
    ],
)
def test_classify_view_command_detects_file_view(command):
    res = classify_view_command(command)
    assert res["viewing_file"], f"expected file-view: {command}"
    assert "read_file" in res["recommendation"]


@pytest.mark.parametrize(
    "command",
    [
        "grep TODO src/ | head -20",
        "ls -la | head",
        "cat /proc/cpuinfo",
        "echo hi",
        "pytest",
    ],
)
def test_classify_view_command_passes_pipes_and_virtual(command):
    assert not classify_view_command(command)["viewing_file"], command


def test_bash_tool_refuses_cat_source(preset):
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "cat -n foo.py"}))
    assert r.data is not None
    assert "Refused" in r.data.get("error", "")
    assert "read_file" in r.data.get("error", "")
    assert r.data.get("first_token") == "cat"


def test_bash_tool_refuses_destructive(preset):
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "rm -rf /"}))
    assert r.data is not None
    assert "Refused" in r.data.get("error", "")
    assert r.data.get("first_token") == "rm"


def test_bash_tool_refuses_sed_in_place(preset):
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "sed -i s/a/b/ foo.py"}))
    assert r.data is not None
    assert "sed -i" in r.data.get("error", "")
    assert "edit_file" in r.data.get("error", "")


def test_bash_tool_runs_safe_command(preset):
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo hello"}))
    assert r.data is not None
    assert "hello" in (r.data.get("stdout") or "")


@pytest.mark.parametrize(
    "tool_name,marker",
    [
        ("bash", "Refusals"),
        ("read_file", "edit_file"),
        ("edit_file", "Recovery"),
        ("write_file", "NEW"),
        ("list_dir", "tree"),
        ("glob", "pattern"),
        ("grep", "regex"),
    ],
)
def test_tool_descriptions_are_rich(preset, tool_name, marker):
    desc = preset.tools._tools[tool_name].description
    assert len(desc) > 200, f"{tool_name} description too short ({len(desc)} chars)"
    assert marker.lower() in desc.lower(), f"{tool_name} description missing {marker!r}"


# ── Production-grade hardening ─────────────────────────────────────


def test_read_file_refuses_binary(preset):
    Path(_workspace(preset), "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    r = preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "img.png"}))
    assert r.data.get("binary") is True
    assert "binary" in r.data.get("error", "").lower()


def test_read_file_refuses_directory(preset):
    Path(_workspace(preset), "subdir").mkdir()
    r = preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "subdir"}))
    assert "directory" in r.data.get("error", "")
    assert "list_dir" in r.data.get("recovery", "")


def test_read_file_latin1_fallback(preset):
    Path(_workspace(preset), "latin.txt").write_bytes(b"caf\xe9\n")
    r = preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "latin.txt"}))
    assert r.data.get("encoding") == "latin-1"
    assert "café" in r.data.get("content", "")


def test_read_file_invalid_line_range(preset):
    Path(_workspace(preset), "x.py").write_text("a\nb\nc\n")
    r = preset.tools.dispatch(
        ToolCall(tool="read_file", args={"file_path": "x.py", "start_line": 5, "end_line": 2})
    )
    assert "must be >= start_line" in r.data.get("error", "")


def test_write_file_refuses_existing_without_overwrite(preset):
    Path(_workspace(preset), "x.py").write_text("old\n")
    r = preset.tools.dispatch(
        ToolCall(tool="write_file", args={"file_path": "x.py", "content": "new"})
    )
    assert r.data.get("exists") is True
    assert "edit_file" in r.data.get("recovery", "")


def test_write_file_overwrite_works(preset):
    Path(_workspace(preset), "x.py").write_text("old\n")
    r = preset.tools.dispatch(
        ToolCall(
            tool="write_file",
            args={"file_path": "x.py", "content": "new", "overwrite": True},
        )
    )
    assert r.data.get("written") == "x.py"
    assert Path(_workspace(preset), "x.py").read_text() == "new"


def test_write_file_creates_parent_dirs(preset):
    r = preset.tools.dispatch(
        ToolCall(
            tool="write_file",
            args={"file_path": "deep/nested/dir/x.py", "content": "hi\n"},
        )
    )
    assert r.data.get("written")
    assert Path(_workspace(preset), "deep/nested/dir/x.py").exists()


def test_multi_edit_atomic_success(preset):
    Path(_workspace(preset), "x.py").write_text("def foo():\n    return foo() + 1\n")
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "x.py"}))
    r = preset.tools.dispatch(
        ToolCall(
            tool="multi_edit",
            args={
                "file_path": "x.py",
                "edits": [
                    {"old_string": "def foo()", "new_string": "def bar()"},
                    {"old_string": "foo()", "new_string": "bar()", "replace_all": True},
                ],
            },
        )
    )
    assert r.data.get("edits_applied") == 2
    assert r.data.get("total_replacements") == 2
    assert "def bar()" in Path(_workspace(preset), "x.py").read_text()


def test_multi_edit_atomic_rollback_on_failure(preset):
    original = "def foo():\n    return 1\n"
    Path(_workspace(preset), "x.py").write_text(original)
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "x.py"}))
    r = preset.tools.dispatch(
        ToolCall(
            tool="multi_edit",
            args={
                "file_path": "x.py",
                "edits": [
                    {"old_string": "def foo()", "new_string": "def bar()"},
                    {"old_string": "DOES NOT EXIST", "new_string": "x"},
                ],
            },
        )
    )
    # Second edit failed → file must be unchanged.
    assert r.data.get("failed_edit_index") == 1
    assert Path(_workspace(preset), "x.py").read_text() == original


def test_multi_edit_requires_prior_read(preset):
    Path(_workspace(preset), "x.py").write_text("hi\n")
    r = preset.tools.dispatch(
        ToolCall(
            tool="multi_edit",
            args={
                "file_path": "x.py",
                "edits": [{"old_string": "hi", "new_string": "bye"}],
            },
        )
    )
    assert r.data.get("missing") == "prior_read"


def test_multi_edit_requires_replace_all_for_multiple_matches(preset):
    Path(_workspace(preset), "x.py").write_text("a\na\na\n")
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "x.py"}))
    r = preset.tools.dispatch(
        ToolCall(
            tool="multi_edit",
            args={
                "file_path": "x.py",
                "edits": [{"old_string": "a", "new_string": "b"}],
            },
        )
    )
    assert r.data.get("matches") == 3


def test_bash_spills_long_stdout(preset):
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "yes hello | head -5000"}))
    assert r.data.get("stdout_spill_file", "").startswith(".coder_scratch/")
    assert r.data.get("stdout_full_chars", 0) > 15000
    spill_path = Path(_workspace(preset), r.data["stdout_spill_file"])
    assert spill_path.exists()


def _workspace(preset) -> str:
    """Pull the workspace path out of the loaded preset's resources."""
    return preset.tools._resources["workspace_config"].path


def test_bash_refuses_repeated_command(preset):
    cmd = "echo hi"
    r1 = preset.tools.dispatch(ToolCall(tool="bash", args={"command": cmd}))
    r2 = preset.tools.dispatch(ToolCall(tool="bash", args={"command": cmd}))
    r3 = preset.tools.dispatch(ToolCall(tool="bash", args={"command": cmd}))
    assert "hi" in r1.data.get("stdout", "")
    assert "hi" in r2.data.get("stdout", "")
    assert "no progress" in r3.data.get("error", "")
    assert r3.data.get("repeats") == 3


def test_bash_repeat_detection_normalizes_whitespace(preset):
    preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo  a"}))
    preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo a"}))
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo   a"}))
    assert "no progress" in r.data.get("error", "")


def test_bash_repeat_window_only_holds_recent_4(preset):
    # Window size is 4. After 4 different commands the original
    # repeats are evicted and the same command works again.
    preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo a"}))
    preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo a"}))
    for cmd in ("echo b", "echo c", "echo d", "echo e"):
        preset.tools.dispatch(ToolCall(tool="bash", args={"command": cmd}))
    r = preset.tools.dispatch(ToolCall(tool="bash", args={"command": "echo a"}))
    assert "a" in r.data.get("stdout", "")
