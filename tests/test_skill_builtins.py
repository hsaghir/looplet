"""Tests for declarative skill workspace wiring (no setup.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet import cartridge_to_preset
from looplet.builtin_tools import AVAILABLE
from looplet.skills import (
    Skill,
    SkillManager,
    build_skill_manager_for_workspace,
)
from looplet.types import ToolContext


def test_skill_builtins_registered() -> None:
    assert "search_skills" in AVAILABLE
    assert "activate_skill" in AVAILABLE
    assert AVAILABLE["search_skills"].requires == ["skill_manager"]
    assert AVAILABLE["activate_skill"].requires == ["skill_manager"]


def test_search_skills_returns_helpful_error_without_resource() -> None:
    spec = AVAILABLE["search_skills"]
    ctx = ToolContext(resources={})
    out = spec.execute(ctx, query="anything")  # type: ignore[arg-type]
    assert "error" in out and "skill_manager" in out["error"]


def test_search_skills_with_resource(tmp_path: Path) -> None:
    skill_dir = tmp_path / "csv-stats"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: csv-stats\ndescription: Compute stats for CSV files.\n---\n# body"
    )
    manager = build_skill_manager_for_workspace(tmp_path, skills_subdir=".")
    ctx = ToolContext(resources={"skill_manager": manager})

    out = AVAILABLE["search_skills"].execute(ctx, query="csv")  # type: ignore[arg-type]
    assert "skills" in out and any(s["name"] == "csv-stats" for s in out["skills"])


def test_activate_skill_with_resource(tmp_path: Path) -> None:
    skill_dir = tmp_path / "csv-stats"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: csv-stats\ndescription: Compute stats for CSV files.\n---\n# body"
    )
    manager = build_skill_manager_for_workspace(tmp_path, skills_subdir=".")
    ctx = ToolContext(resources={"skill_manager": manager})

    out = AVAILABLE["activate_skill"].execute(ctx, name="csv-stats")  # type: ignore[arg-type]
    assert out["activated"] == "csv-stats"
    assert "csv-stats" in out["active_skills"]


def test_build_skill_manager_handles_missing_dir(tmp_path: Path) -> None:
    # Empty dir, no SKILL.md anywhere — should still return a manager.
    m = build_skill_manager_for_workspace(tmp_path)
    assert isinstance(m, SkillManager)
    assert m.search("anything") == []


def test_skillful_analyst_workspace_loads_without_setup_py() -> None:
    """The shipped example workspace must be 100% declarative."""
    ws = Path(__file__).resolve().parents[1] / "examples" / "skillful_analyst.cartridge"
    assert not (ws / "setup.py").exists(), "skillful_analyst should not need setup.py"
    assert not (ws / "resources" / "project_root.py").exists(), (
        "skillful_analyst should not need resources/project_root.py — tools read "
        "ctx.resources['runtime'] which the loader auto-injects."
    )

    preset = cartridge_to_preset(ws, strict=True, runtime={"project_root": "/tmp"})
    tool_names = set(preset.tools.tool_names)
    assert {"search_skills", "activate_skill", "done", "read_text", "write_text"} <= tool_names
    hook_classes = {type(h).__name__ for h in preset.hooks}
    assert "SkillActivationHook" in hook_classes
    # Runtime resource is auto-injected
    assert "runtime" in preset.resources
    assert preset.resources["runtime"]["project_root"] == "/tmp"


def test_skill_can_be_activated_via_workspace_loaded_resource(tmp_path: Path) -> None:
    """End-to-end: load the example workspace, search + activate via the loaded tools."""
    ws = Path(__file__).resolve().parents[1] / "examples" / "skillful_analyst.cartridge"
    preset = cartridge_to_preset(ws, strict=True, runtime={"project_root": str(tmp_path)})

    # The loader resolved skill_manager from resources/skill_manager.py.
    # Both built-ins should now find it via ctx.resources.
    search = preset.tools._tools["search_skills"]
    activate = preset.tools._tools["activate_skill"]

    ctx = ToolContext(resources=preset.resources)
    found = search.execute(ctx, query="json")  # type: ignore[arg-type]
    names = {s["name"] for s in found["skills"]}
    assert "json-pretty" in names

    activated = activate.execute(ctx, name="json-pretty")  # type: ignore[arg-type]
    assert activated["activated"] == "json-pretty"
