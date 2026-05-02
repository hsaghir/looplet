"""Smoke tests for workspace ``extends:`` composition + agent_factory.

Covers:
  1. ``extends:`` inherits parent's tools + lets child override max_steps
  2. ``extends:`` is recursive (grandparent → parent → child)
  3. Cycle detection raises a clean error
  4. agent_factory.workspace loads + extends coder.workspace correctly
  5. validate_workspace tool returns structured success / errors
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from looplet import workspace_to_preset
from looplet.types import ToolCall
from looplet.workspace import WorkspaceSerializationError

PARENT_FILES = {
    "workspace.json": '{"name":"parent","schema_version":1}\n',
    "config.yaml": "max_steps: 10\n",
    "tools/greet/tool.yaml": (
        "name: greet\ndescription: say hello\nparameters:\n  who:\n    type: string\n"
    ),
    "tools/greet/execute.py": ("def execute(*, who):\n    return {'msg': f'hi {who}'}\n"),
    "tools/done/tool.yaml": (
        "name: done\ndescription: finish\nparameters:\n  summary:\n    type: string\n"
    ),
    "tools/done/execute.py": ("def execute(*, summary):\n    return {'summary': summary}\n"),
}


def _make_workspace(root: Path, name: str, files: dict[str, str]) -> Path:
    ws = root / name
    ws.mkdir()
    for rel, content in files.items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return ws


def test_extends_inherits_parent_tools(tmp_path: Path) -> None:
    parent = _make_workspace(tmp_path, "parent.workspace", PARENT_FILES)
    child = _make_workspace(
        tmp_path,
        "child.workspace",
        {
            "workspace.json": '{"name":"child","schema_version":1}\n',
            "config.yaml": f"extends: {parent}\nmax_steps: 25\n",
            "tools/shout/tool.yaml": (
                "name: shout\ndescription: yell\nparameters:\n  msg:\n    type: string\n"
            ),
            "tools/shout/execute.py": ("def execute(*, msg):\n    return {'yell': msg.upper()}\n"),
        },
    )
    p = workspace_to_preset(child)
    assert sorted(p.tools._tools.keys()) == ["done", "greet", "shout"]
    # Child's max_steps wins.
    assert p.config.max_steps == 25


def test_extends_child_can_override_parent_tool(tmp_path: Path) -> None:
    parent = _make_workspace(tmp_path, "parent.workspace", PARENT_FILES)
    child = _make_workspace(
        tmp_path,
        "child.workspace",
        {
            "workspace.json": '{"name":"child","schema_version":1}\n',
            "config.yaml": f"extends: {parent}\n",
            # Override greet with a different implementation.
            "tools/greet/tool.yaml": (
                "name: greet\ndescription: say HELLO\nparameters:\n  who:\n    type: string\n"
            ),
            "tools/greet/execute.py": (
                "def execute(*, who):\n    return {'msg': f'HELLO {who}'}\n"
            ),
        },
    )
    p = workspace_to_preset(child)
    r = p.tools.dispatch(ToolCall(tool="greet", args={"who": "world"}))
    # Child override should win.
    assert r.data.get("msg") == "HELLO world"


def test_extends_cycle_detection(tmp_path: Path) -> None:
    a = _make_workspace(
        tmp_path,
        "a.workspace",
        {
            "workspace.json": '{"name":"a","schema_version":1}\n',
            "config.yaml": "extends: ../b.workspace\n",
        },
    )
    _make_workspace(
        tmp_path,
        "b.workspace",
        {
            "workspace.json": '{"name":"b","schema_version":1}\n',
            "config.yaml": "extends: ../a.workspace\n",
        },
    )
    with pytest.raises(WorkspaceSerializationError, match="circular"):
        workspace_to_preset(a)


def test_extends_recursive_three_levels(tmp_path: Path) -> None:
    grand = _make_workspace(tmp_path, "grand.workspace", PARENT_FILES)
    mid = _make_workspace(
        tmp_path,
        "mid.workspace",
        {
            "workspace.json": '{"name":"mid","schema_version":1}\n',
            "config.yaml": f"extends: {grand}\n",
            "tools/middle/tool.yaml": ("name: middle\ndescription: middle layer\nparameters: {}\n"),
            "tools/middle/execute.py": ("def execute():\n    return {'level': 'mid'}\n"),
        },
    )
    child = _make_workspace(
        tmp_path,
        "child.workspace",
        {
            "workspace.json": '{"name":"child","schema_version":1}\n',
            "config.yaml": f"extends: {mid}\n",
            "tools/child_only/tool.yaml": (
                "name: child_only\ndescription: child layer\nparameters: {}\n"
            ),
            "tools/child_only/execute.py": ("def execute():\n    return {'level': 'child'}\n"),
        },
    )
    p = workspace_to_preset(child)
    # Tools from all three levels.
    assert set(p.tools._tools.keys()) == {"done", "greet", "middle", "child_only"}


def test_extends_missing_parent_raises(tmp_path: Path) -> None:
    child = _make_workspace(
        tmp_path,
        "child.workspace",
        {
            "workspace.json": '{"name":"child","schema_version":1}\n',
            "config.yaml": "extends: ./nonexistent.workspace\n",
        },
    )
    with pytest.raises(WorkspaceSerializationError, match="does not exist"):
        workspace_to_preset(child)


def test_agent_factory_workspace_loads() -> None:
    """examples/agent_factory.workspace must extend coder.workspace
    successfully and add a validate_workspace tool."""
    repo_root = Path(__file__).resolve().parents[1]
    ws = repo_root / "examples" / "agent_factory.workspace"
    p = workspace_to_preset(str(ws), runtime={"workspace": "."})
    tool_names = sorted(p.tools._tools.keys())
    # Inherits all coder tools + adds validate_workspace.
    assert "validate_workspace" in tool_names
    assert "edit_file" in tool_names  # inherited from coder
    assert "multi_edit" in tool_names  # inherited from coder
    assert "done" in tool_names
    # Has its own system prompt (overrides coder's).
    assert p.config.system_prompt and len(p.config.system_prompt) > 1000
    assert "agent factory" in p.config.system_prompt.lower()
    # Child's max_steps wins.
    assert p.config.max_steps == 80


def test_validate_workspace_tool_success(tmp_path: Path) -> None:
    """The validate_workspace tool returns structured success."""
    parent = _make_workspace(tmp_path, "parent.workspace", PARENT_FILES)
    repo_root = Path(__file__).resolve().parents[1]
    factory = repo_root / "examples" / "agent_factory.workspace"
    p = workspace_to_preset(str(factory), runtime={"workspace": str(tmp_path)})
    r = p.tools.dispatch(
        ToolCall(
            tool="validate_workspace",
            args={"workspace_path": "parent.workspace", "strict": True},
        )
    )
    assert r.data.get("valid") is True
    assert sorted(r.data.get("tools", [])) == ["done", "greet"]
    assert r.data.get("system_prompt_chars") == 0
    assert "system_prompt is empty" in str(r.data.get("warnings", []))


def test_validate_workspace_tool_missing_workspace_json(tmp_path: Path) -> None:
    """Validator returns FileNotFoundError when workspace.json is missing."""
    bad = tmp_path / "broken.workspace"
    bad.mkdir()
    (bad / "config.yaml").write_text("max_steps: 10\n")
    repo_root = Path(__file__).resolve().parents[1]
    factory = repo_root / "examples" / "agent_factory.workspace"
    p = workspace_to_preset(str(factory), runtime={"workspace": str(tmp_path)})
    r = p.tools.dispatch(
        ToolCall(
            tool="validate_workspace",
            args={"workspace_path": "broken.workspace"},
        )
    )
    assert "FileNotFoundError" in r.data.get("error", "")
    assert r.data.get("missing") == "workspace.json"
    assert "schema_version" in r.data.get("recovery", "")


def test_validate_workspace_tool_outside_project(tmp_path: Path) -> None:
    """Validator refuses paths outside the project root."""
    repo_root = Path(__file__).resolve().parents[1]
    factory = repo_root / "examples" / "agent_factory.workspace"
    p = workspace_to_preset(str(factory), runtime={"workspace": str(tmp_path)})
    r = p.tools.dispatch(ToolCall(tool="validate_workspace", args={"workspace_path": "/etc"}))
    assert "outside the project" in r.data.get("error", "")
