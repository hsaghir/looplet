"""Smoke tests for ``looplet.scaffold`` + ``builtin_tools: [subagent]``.

Covers:
  * scaffold_workspace creates a loadable skeleton
  * idempotent re-scaffold preserves edits
  * skeleton has done tool by default
  * tool name validation
  * builtin_tools opt-in registers subagent
  * subagent dispatch runs a sub-loop end-to-end
  * subagent recursion guard refuses past max_depth
  * subagent inherits sub-workspace config when max_steps not given
  * subagent override of max_steps applies
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import MockLLMBackend, workspace_to_preset
from looplet.scaffold import scaffold_workspace
from looplet.types import ToolContext
from looplet.workspace import WorkspaceSerializationError

# ── scaffold ────────────────────────────────────────────────────


def test_scaffold_creates_loadable_workspace(tmp_path: Path) -> None:
    p = scaffold_workspace(
        tmp_path / "agent.workspace",
        name="agent",
        tools=["foo", "bar"],
    )
    preset = workspace_to_preset(p)
    assert sorted(preset.tools._tools.keys()) == ["bar", "done", "foo"]
    assert preset.config.max_steps == 20
    assert "agent" in (preset.config.system_prompt or "").lower()
    # Regression: workspace.json must be valid JSON (catch the
    # repr-vs-dumps bug). ``workspace_to_preset`` only checks that the
    # file exists, so we parse explicitly.
    import json as _json

    meta = _json.loads((p / "workspace.json").read_text())
    assert meta == {"name": "agent", "schema_version": 1}


def test_scaffold_workspace_json_handles_special_chars(tmp_path: Path) -> None:
    """Names with quotes / backslashes / non-ASCII must round-trip via JSON."""
    import json as _json

    tricky = 'agent "with quotes" and \\backslashes\\ and 日本語'
    p = scaffold_workspace(tmp_path / "x.workspace", name=tricky, tools=[])
    meta = _json.loads((p / "workspace.json").read_text())
    assert meta["name"] == tricky


def test_scaffold_done_tool_always_added(tmp_path: Path) -> None:
    p = scaffold_workspace(tmp_path / "x.workspace", name="x", tools=[])
    assert (p / "tools" / "done" / "tool.yaml").is_file()
    assert (p / "tools" / "done" / "execute.py").is_file()


def test_scaffold_idempotent_preserves_edits(tmp_path: Path) -> None:
    p = scaffold_workspace(tmp_path / "x.workspace", name="x", tools=["foo"])
    edited = "name: foo\ndescription: edited!\nparameters: {}\n"
    (p / "tools" / "foo" / "tool.yaml").write_text(edited)
    # Re-scaffold should not overwrite.
    scaffold_workspace(p, name="x", tools=["foo", "bar"], overwrite=True)
    assert (p / "tools" / "foo" / "tool.yaml").read_text() == edited
    # New tool should be added.
    assert (p / "tools" / "bar" / "tool.yaml").is_file()


def test_scaffold_refuses_existing_non_empty(tmp_path: Path) -> None:
    p = tmp_path / "x.workspace"
    p.mkdir()
    (p / "stuff.txt").write_text("hi")
    with pytest.raises(FileExistsError, match="non-empty"):
        scaffold_workspace(p, name="x", tools=[])


def test_scaffold_rejects_invalid_tool_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        scaffold_workspace(tmp_path / "x.workspace", name="x", tools=["bad-name"])
    with pytest.raises(ValueError, match="empty"):
        scaffold_workspace(tmp_path / "y.workspace", name="y", tools=[""])


def test_scaffold_workspace_loads_with_factory_setup(tmp_path: Path) -> None:
    """Loading the agent_factory with scaffold runtime kwargs auto-creates the target."""
    repo_root = Path(__file__).resolve().parents[1]
    factory = repo_root / "examples" / "agent_factory.workspace"
    workspace_to_preset(
        str(factory),
        runtime={
            "workspace": str(tmp_path),
            "scaffold_to": "auto.workspace",
            "scaffold_tools": ["alpha", "beta"],
        },
    )
    target = tmp_path / "auto.workspace"
    assert (target / "workspace.json").is_file()
    assert (target / "config.yaml").is_file()
    assert (target / "tools" / "alpha").is_dir()
    assert (target / "tools" / "beta").is_dir()
    assert (target / "tools" / "done").is_dir()


# ── builtin_tools: [subagent] ───────────────────────────────────


def _make_parent_with_subagent(tmp_path: Path) -> Path:
    parent = tmp_path / "parent.workspace"
    scaffold_workspace(parent, name="parent", tools=[])
    cfg = parent / "config.yaml"
    cfg.write_text(cfg.read_text() + "builtin_tools:\n  - subagent\n")
    return parent


def test_builtin_tools_registers_subagent(tmp_path: Path) -> None:
    parent = _make_parent_with_subagent(tmp_path)
    p = workspace_to_preset(parent)
    assert "subagent" in p.tools._tools


def test_builtin_tools_unknown_strict_raises(tmp_path: Path) -> None:
    parent = tmp_path / "p.workspace"
    scaffold_workspace(parent, name="p", tools=[])
    cfg = parent / "config.yaml"
    cfg.write_text(cfg.read_text() + "builtin_tools:\n  - nonexistent_tool\n")
    with pytest.raises(WorkspaceSerializationError, match="unknown builtin tool"):
        workspace_to_preset(parent, strict=True)


def test_subagent_runs_child_loop_to_done(tmp_path: Path) -> None:
    parent = _make_parent_with_subagent(tmp_path)
    child = tmp_path / "child.workspace"
    scaffold_workspace(child, name="child", tools=[])

    p = workspace_to_preset(parent)
    spec = p.tools._tools["subagent"]
    mock = MockLLMBackend(
        responses=[json.dumps({"tool": "done", "args": {"summary": "done-from-child"}})]
    )
    ctx = ToolContext(llm=mock, metadata={})
    result = spec.execute(ctx, workspace=str(child), task="hi", max_steps=5)

    assert result["summary"] == "done-from-child"
    assert result["final_tool"] == "done"
    assert result["depth"] == 1
    assert result["steps_used"] >= 1


def test_subagent_recursion_guard(tmp_path: Path) -> None:
    parent = _make_parent_with_subagent(tmp_path)
    child = tmp_path / "child.workspace"
    scaffold_workspace(child, name="child", tools=[])

    p = workspace_to_preset(parent)
    spec = p.tools._tools["subagent"]
    mock = MockLLMBackend(responses=[])
    ctx = ToolContext(llm=mock, metadata={})

    # Pretend we're already at max depth via the ContextVar.
    from looplet.builtin_tools.subagent import _DEPTH_VAR

    token = _DEPTH_VAR.set(5)
    try:
        result = spec.execute(ctx, workspace=str(child), task="hi", max_depth=5)
        assert "would exceed" in result.get("error", "")
        assert result.get("depth") == 5
    finally:
        _DEPTH_VAR.reset(token)


def test_subagent_missing_workspace_returns_structured_error(tmp_path: Path) -> None:
    parent = _make_parent_with_subagent(tmp_path)
    p = workspace_to_preset(parent)
    spec = p.tools._tools["subagent"]
    ctx = ToolContext(llm=MockLLMBackend(responses=[]), metadata={})
    result = spec.execute(ctx, workspace=str(tmp_path / "nope"), task="hi")
    assert "not found" in result.get("error", "")


# ── scaffold_workspace as a built-in tool ──────────────────────


def test_builtin_tools_registers_scaffold_workspace(tmp_path: Path) -> None:
    parent = tmp_path / "p.workspace"
    scaffold_workspace(parent, name="p", tools=[])
    cfg = parent / "config.yaml"
    cfg.write_text(cfg.read_text() + "builtin_tools:\n  - scaffold_workspace\n")
    p = workspace_to_preset(parent)
    assert "scaffold_workspace" in p.tools._tools


def test_scaffold_workspace_tool_creates_workspace(tmp_path: Path) -> None:
    """Dispatching scaffold_workspace tool builds a loadable child workspace."""
    parent = tmp_path / "factory.workspace"
    scaffold_workspace(parent, name="factory", tools=[])
    cfg = parent / "config.yaml"
    cfg.write_text(cfg.read_text() + "builtin_tools:\n  - scaffold_workspace\n")
    p = workspace_to_preset(parent)
    spec = p.tools._tools["scaffold_workspace"]
    ctx = ToolContext(metadata={})
    result = spec.execute(
        ctx,
        path=str(tmp_path / "child.workspace"),
        name="child",
        tools=["alpha", "beta"],
    )
    assert result.get("scaffolded") is True
    assert "child.workspace" in result.get("path", "")
    # Loadable.
    sub = workspace_to_preset(tmp_path / "child.workspace")
    assert sorted(sub.tools._tools.keys()) == ["alpha", "beta", "done"]


def test_scaffold_workspace_tool_existing_dir_returns_recovery(tmp_path: Path) -> None:
    """File-exists error is returned as structured tool result, not raised."""
    parent = tmp_path / "p.workspace"
    scaffold_workspace(parent, name="p", tools=[])
    cfg = parent / "config.yaml"
    cfg.write_text(cfg.read_text() + "builtin_tools:\n  - scaffold_workspace\n")
    p = workspace_to_preset(parent)
    spec = p.tools._tools["scaffold_workspace"]
    # Pre-create non-empty dir.
    (tmp_path / "blocked").mkdir()
    (tmp_path / "blocked" / "stuff.txt").write_text("hi")
    ctx = ToolContext(metadata={})
    result = spec.execute(ctx, path=str(tmp_path / "blocked"), name="x", tools=["a"])
    assert "FileExistsError" in result.get("error", "")
    assert "overwrite=True" in result.get("recovery", "")


def test_subagent_forwards_workspace_to_subloop_runtime(tmp_path: Path) -> None:
    """Subagent must propagate the parent's workspace path so the
    sub-loop's ``runtime["workspace"]`` is the same project root.

    Regression for the bug where ``runtime`` was read from
    ``ctx.metadata["runtime"]`` (which the loop never sets) instead
    of being constructed from the parent's ``workspace_config``.
    """
    parent = _make_parent_with_subagent(tmp_path)
    child = tmp_path / "child.workspace"
    scaffold_workspace(child, name="child", tools=[])
    # Add a setup.py to the child that records the runtime it received.
    (child / "setup.py").write_text(
        "from pathlib import Path\n"
        "def setup(preset, resources, *, runtime=None, **_):\n"
        "    Path(runtime['workspace'], '_seen.txt').write_text(runtime['workspace'])\n"
        "    return preset\n"
    )

    p = workspace_to_preset(parent, runtime={"workspace": str(tmp_path)})
    spec = p.tools._tools["subagent"]
    mock = MockLLMBackend(responses=[json.dumps({"tool": "done", "args": {"summary": "ok"}})])
    # Build a context that mimics what the dispatcher would produce.
    # The factory normally injects ``workspace_config`` via ``requires``,
    # but a scaffold-only parent has no such resource — so we exercise
    # the documented ``ctx.metadata['runtime']`` fall-through path.
    ctx = ToolContext(
        llm=mock,
        resources={},
        metadata={"runtime": {"workspace": str(tmp_path)}},
    )
    result = spec.execute(ctx, workspace=str(child), task="hi", max_steps=3)
    assert result.get("final_tool") == "done"
    seen = (tmp_path / "_seen.txt").read_text()
    assert seen == str(tmp_path), (
        f"sub-loop saw runtime['workspace']={seen!r}, expected {str(tmp_path)!r}"
    )


def test_validate_workspace_warns_on_unfilled_scaffold(tmp_path: Path) -> None:
    """A freshly scaffolded workspace must surface TODO + NotImplementedError
    warnings so the agent doesn't ``done`` on an empty agent."""
    repo_root = Path(__file__).resolve().parents[1]
    factory = repo_root / "examples" / "agent_factory.workspace"
    # Pre-scaffold a child via the factory's setup.py.
    workspace_to_preset(
        str(factory),
        runtime={
            "workspace": str(tmp_path),
            "scaffold_to": "auto.workspace",
            "scaffold_tools": ["alpha"],
        },
    )
    p = workspace_to_preset(str(factory), runtime={"workspace": str(tmp_path)})
    from looplet.types import ToolCall as _TC

    r = p.tools.dispatch(_TC(tool="validate_workspace", args={"workspace_path": "auto.workspace"}))
    warnings = (r.data or {}).get("warnings", [])
    assert any("TODO" in w for w in warnings), warnings
    assert any("NotImplementedError" in w for w in warnings), warnings


def test_loader_warns_on_tool_name_mismatch(tmp_path: Path, caplog) -> None:
    p = scaffold_workspace(tmp_path / "x.workspace", name="x", tools=["foo"])
    # Edit tool.yaml to use a different name than the directory.
    (p / "tools" / "foo" / "tool.yaml").write_text(
        "name: BADNAME\ndescription: x\nparameters: {}\n"
    )
    import logging

    with caplog.at_level(logging.WARNING):
        workspace_to_preset(p)
    assert any("tool name mismatch" in rec.message for rec in caplog.records), [
        r.message for r in caplog.records
    ]


def test_loader_strict_raises_on_tool_name_mismatch(tmp_path: Path) -> None:
    p = scaffold_workspace(tmp_path / "x.workspace", name="x", tools=["foo"])
    (p / "tools" / "foo" / "tool.yaml").write_text(
        "name: BADNAME\ndescription: x\nparameters: {}\n"
    )
    with pytest.raises(WorkspaceSerializationError, match="tool name mismatch"):
        workspace_to_preset(p, strict=True)


def test_subagent_warns_on_cwd_fallback(tmp_path: Path) -> None:
    parent = _make_parent_with_subagent(tmp_path)
    child = tmp_path / "child.workspace"
    scaffold_workspace(child, name="child", tools=[])
    p = workspace_to_preset(parent)
    spec = p.tools._tools["subagent"]
    mock = MockLLMBackend(responses=[json.dumps({"tool": "done", "args": {"summary": "ok"}})])
    # Empty resources + empty metadata -> cwd fallback fires.
    ctx = ToolContext(llm=mock, resources={}, metadata={})
    result = spec.execute(ctx, workspace=str(child), task="hi", max_steps=3)
    assert "warning" in result, result
    assert "defaulted to cwd" in result["warning"]
