"""Smoke tests for coder.cartridge tool improvements.

Covers:
  1. ``edit_file`` refuses without a prior ``read_file`` in the session.
  2. ``classify_bash_command`` flags destructive patterns.
  3. ``classify_sed_command`` flags ``sed -i`` in-place edits.
  4. ``bash`` tool refuses destructive commands and ``sed -i``.
  5. Tool descriptions (loaded from YAML block scalars) include the
     rich multi-paragraph guidance.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# Make ``coder_lib_tools`` importable for the classifier-only tests.
_CODER_DIR = Path(__file__).resolve().parents[1] / "examples" / "coder.cartridge"
sys.path.insert(0, str(_CODER_DIR))

from coder_lib_tools import (  # noqa: E402
    classify_bash_command,
    classify_sed_command,
    classify_view_command,
)

from looplet import cartridge_to_preset  # noqa: E402
from looplet.types import ToolCall  # noqa: E402


@pytest.fixture
def preset():
    with tempfile.TemporaryDirectory() as td:
        Path(td, "foo.py").write_text("hello\n")
        yield cartridge_to_preset(str(_CODER_DIR), runtime={"workspace": td})


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
        ("todo", "checklist"),
        ("web_fetch", "HTTP"),
        ("subagent", "focused"),
        ("notebook_edit", "Jupyter"),
        ("git_inspect", "read-only"),
        ("worktree", "worktree"),
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
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "x.py"}))
    r = preset.tools.dispatch(
        ToolCall(
            tool="write_file",
            args={"file_path": "x.py", "content": "new", "overwrite": True},
        )
    )
    assert r.data.get("written") == "x.py"
    assert Path(_workspace(preset), "x.py").read_text() == "new"


def test_write_file_overwrite_requires_prior_read(preset):
    Path(_workspace(preset), "x.py").write_text("old\n")
    r = preset.tools.dispatch(
        ToolCall(
            tool="write_file",
            args={"file_path": "x.py", "content": "new", "overwrite": True},
        )
    )
    assert r.data.get("missing") == "prior_read"
    assert "read_file" in r.data.get("recovery", "")


def test_write_file_overwrite_refuses_stale_read(preset):
    target = Path(_workspace(preset), "x.py")
    target.write_text("old\n")
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "x.py"}))
    target.write_text("user changed\n")
    r = preset.tools.dispatch(
        ToolCall(
            tool="write_file",
            args={"file_path": "x.py", "content": "new", "overwrite": True},
        )
    )
    assert r.data.get("stale") is True
    assert target.read_text() == "user changed\n"


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


def test_todo_replace_update_and_persist(preset):
    r = preset.tools.dispatch(
        ToolCall(
            tool="todo",
            args={
                "operation": "replace",
                "todos": [
                    {"title": "Explore", "status": "completed"},
                    {"title": "Implement", "status": "in-progress"},
                ],
            },
        )
    )
    assert r.data.get("count") == 2
    assert r.data["todos"][1]["id"] == 2
    r = preset.tools.dispatch(
        ToolCall(tool="todo", args={"operation": "update", "id": 2, "status": "completed"})
    )
    assert r.data["status_counts"]["completed"] == 2
    list_result = preset.tools.dispatch(ToolCall(tool="todo", args={"operation": "list"}))
    assert [item["title"] for item in list_result.data["todos"]] == ["Explore", "Implement"]


def test_grep_supports_modes_filters_and_pagination(preset):
    workspace = Path(_workspace(preset))
    (workspace / "a.py").write_text("needle one\nneedle two\n")
    (workspace / "b.txt").write_text("needle text\n")
    r = preset.tools.dispatch(
        ToolCall(
            tool="grep",
            args={
                "pattern": "needle",
                "glob": "*.py",
                "output_mode": "content",
                "head_limit": 1,
            },
        )
    )
    assert r.data.get("count") == 2
    assert r.data.get("truncated") is True
    assert r.data.get("matches") == ["a.py:1:needle one"]
    files = preset.tools.dispatch(
        ToolCall(
            tool="grep",
            args={"pattern": "needle", "glob": "*.py", "output_mode": "files_with_matches"},
        )
    )
    assert files.data.get("filenames") == ["a.py"]


def test_web_fetch_extracts_local_html(preset):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><head><title>Docs</title></head><body><h1>Needle Docs</h1><p>Hello from docs.</p></body></html>"
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/docs"
        r = preset.tools.dispatch(ToolCall(tool="web_fetch", args={"url": url}))
    finally:
        server.shutdown()
        server.server_close()
    assert r.data.get("status") == 200
    assert r.data.get("title") == "Docs"
    assert "Hello from docs" in r.data.get("text", "")


def test_notebook_edit_replaces_cell_after_read(preset):
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "abc123",
                "metadata": {},
                "outputs": [],
                "source": ["print('old')\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(_workspace(preset), "analysis.ipynb")
    path.write_text(json.dumps(notebook) + "\n")
    preset.tools.dispatch(ToolCall(tool="read_file", args={"file_path": "analysis.ipynb"}))
    r = preset.tools.dispatch(
        ToolCall(
            tool="notebook_edit",
            args={
                "notebook_path": "analysis.ipynb",
                "cell_id": "abc123",
                "new_source": "print('new')\n",
            },
        )
    )
    assert r.data.get("cell_id") == "abc123"
    updated = json.loads(path.read_text())
    assert updated["cells"][0]["source"] == ["print('new')"]


def test_notebook_edit_requires_prior_read(preset):
    path = Path(_workspace(preset), "analysis.ipynb")
    path.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    r = preset.tools.dispatch(
        ToolCall(tool="notebook_edit", args={"notebook_path": "analysis.ipynb", "cell_id": "x"})
    )
    assert r.data.get("missing") == "prior_read"


def test_git_inspect_status(preset):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    workspace = Path(_workspace(preset))
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True, text=True)
    (workspace / "tracked.txt").write_text("hi\n")
    r = preset.tools.dispatch(ToolCall(tool="git_inspect", args={"operation": "status"}))
    assert r.data.get("exit_code") == 0
    assert "tracked.txt" in r.data.get("stdout", "")


def test_worktree_create_list_remove_managed_path(preset):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    workspace = Path(_workspace(preset))
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=workspace, check=True)
    (workspace / "tracked.txt").write_text("hi\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=workspace, check=True, capture_output=True, text=True
    )
    created = preset.tools.dispatch(
        ToolCall(
            tool="worktree", args={"operation": "create", "name": "experiment", "base_ref": "HEAD"}
        )
    )
    assert created.data.get("exit_code") == 0, created.data
    worktree_path = Path(created.data["path"])
    assert worktree_path.exists()
    listed = preset.tools.dispatch(ToolCall(tool="worktree", args={"operation": "list"}))
    assert str(worktree_path) in listed.data.get("stdout", "")
    refused = preset.tools.dispatch(
        ToolCall(tool="worktree", args={"operation": "remove", "name": "experiment"})
    )
    assert "confirm=true" in refused.data.get("error", "")
    removed = preset.tools.dispatch(
        ToolCall(
            tool="worktree", args={"operation": "remove", "name": "experiment", "confirm": True}
        )
    )
    assert removed.data.get("exit_code") == 0, removed.data
    assert not worktree_path.exists()


def test_subagent_direct_dispatch_explains_missing_llm(preset):
    r = preset.tools.dispatch(ToolCall(tool="subagent", args={"prompt": "Inspect the repo"}))
    assert "ctx.llm" in r.data.get("error", "")


def test_subagent_runs_inside_loop(preset):
    from looplet import composable_loop
    from looplet.testing import MockLLMBackend

    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "subagent", "args": {"prompt": "Think once", "max_steps": 1}}),
            json.dumps({"tool": "think", "args": {"thought": "sub thought"}}),
            json.dumps({"tool": "done", "args": {"summary": "subagent complete"}}),
        ]
    )
    steps = list(
        composable_loop(
            llm=llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task={"q": "run subagent"},
        )
    )
    assert [step.tool_call.tool for step in steps] == ["subagent", "done"]
    assert steps[0].tool_result.data["step_count"] == 1
