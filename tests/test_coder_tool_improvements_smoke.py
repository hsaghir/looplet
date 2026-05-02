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
