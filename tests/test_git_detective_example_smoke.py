"""Dogfood tests for the Git Detective example."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _load_git_detective():
    path = Path(__file__).parents[1] / "examples" / "git_detective" / "agent.py"
    spec = importlib.util.spec_from_file_location("git_detective_agent", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Looplet Test")
    _git(repo, "config", "user.email", "looplet@example.test")

    (repo / "README.md").write_text("# demo\n", encoding="utf-8")
    (repo / "src" / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "feat: initial demo")

    (repo / "README.md").write_text("# demo\n\nMore docs.\n", encoding="utf-8")
    (repo / "src" / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fix: update greeting")


class TestGitDetectiveExample:
    def test_build_tools_uses_decorator_schema_and_helpers(self, tmp_path: Path) -> None:
        git_detective = _load_git_detective()
        _init_repo(tmp_path)

        registry = git_detective.make_tools(str(tmp_path))
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "repo_overview",
            "contributor_stats",
            "recent_activity",
            "file_hotspots",
            "coupled_files",
            "commit_patterns",
            "directory_structure",
            "file_age_analysis",
            "think",
            "done",
        ]
        assert info["recent_activity"]["parameters"]["properties"]["days"]["type"] == "string"
        assert info["recent_activity"]["parameters"]["required"] == []
        assert "ctx" not in info["commit_patterns"]["parameters"]["properties"]
        assert info["think"]["free"] is True
        assert "report" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_new_api_end_to_end(self, tmp_path: Path, capsys) -> None:
        git_detective = _load_git_detective()
        _init_repo(tmp_path)

        rc = git_detective.main([str(tmp_path), "--scripted", "--max-steps", "12"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "Probe: backend has no generate_with_tools method" in out
        assert "Report complete" in out
        assert "Used LLM to assess commit quality" in out
        assert "Git History Detective Report" in out
