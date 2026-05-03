"""Smoke tests for ``looplet new`` and ``looplet run-workspace`` CLI.

The CLI's job is straightforward plumbing — load the factory, run a
loop, surface results — so most of these tests exercise UX paths
(missing env vars, bad workspace path, factory location lookup)
without spinning up a real LLM.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from looplet.__main__ import main


def test_new_help() -> None:
    """``looplet new --help`` exits 0 and prints usage."""
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["new", "--help"])
    assert exc.value.code == 0


def test_run_workspace_help() -> None:
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["run-workspace", "--help"])
    assert exc.value.code == 0


def test_new_missing_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When required env vars are unset, ``new`` prints a clear error
    and exits 1."""
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    captured = io.StringIO()
    with patch.object(sys, "stderr", captured):
        rc = main(["new", "a brief", str(tmp_path / "out.workspace")])
    assert rc == 1
    err = captured.getvalue()
    assert "missing required env vars" in err
    # All three should be named.
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        assert var in err
    # The hint surface (the env-var template) must appear.
    assert "OPENAI_MODEL=" in err


def test_new_partial_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Setting some but not all env vars still errors and names the missing ones."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    captured = io.StringIO()
    with patch.object(sys, "stderr", captured):
        rc = main(["new", "a brief", str(tmp_path / "out.workspace")])
    assert rc == 1
    err = captured.getvalue()
    assert "OPENAI_API_KEY" in err
    assert "OPENAI_MODEL" in err
    # The one we DID set should NOT appear in the missing list.
    assert "missing required env vars: OPENAI_BASE_URL" not in err


def test_run_workspace_missing_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    rc = main(["run-workspace", str(tmp_path), "do something"])
    assert rc == 1


def test_run_workspace_path_must_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    captured = io.StringIO()
    nonexistent = tmp_path / "no_such_dir"
    with patch.object(sys, "stderr", captured):
        rc = main(["run-workspace", str(nonexistent), "do x"])
    assert rc == 1
    assert "workspace not found" in captured.getvalue()


def test_factory_workspace_path_resolves_in_repo() -> None:
    """``_factory_workspace_path`` finds the bundled
    examples/agent_factory.workspace when run from the repo."""
    from looplet.cli.factory_commands import _factory_workspace_path

    p = _factory_workspace_path()
    assert p.is_dir()
    assert (p / "workspace.json").is_file()
    assert (p / "config.yaml").is_file()


def test_new_command_registered_on_top_level() -> None:
    """``looplet`` (no subcommand) should mention ``new`` and ``run-workspace``."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["--help"])
    assert exc.value.code == 0
    out = captured.getvalue()
    assert "new" in out
    assert "run-workspace" in out
