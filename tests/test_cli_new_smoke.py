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
    examples/agent_factory.cartridge when run from the repo."""
    from looplet.cli.factory_commands import _factory_workspace_path

    p = _factory_workspace_path()
    assert p.is_dir()
    assert (p / "cartridge.json").is_file()
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


# ── --pretty (stdlib-only renderer) ────────────────────────────


def test_pretty_printer_renders_steps_without_ansi_when_not_tty() -> None:
    """``PrettyPrinter`` should emit plain text (no escape sequences)
    when stdout isn't a TTY — keeps CI logs and ``tee`` clean."""
    import io as _io
    from types import SimpleNamespace as NS

    # Patch _COLOR off and capture stdout.
    from looplet.cli import _pretty as p_mod
    from looplet.cli._pretty import PrettyPrinter

    original = p_mod._COLOR
    p_mod._COLOR = False
    captured = _io.StringIO()
    try:
        with patch.object(sys, "stdout", captured):
            printer = PrettyPrinter(title="testing")
            printer.header(["  task: smoke", "  model: mock"])
            printer.step(
                NS(
                    tool_call=NS(
                        tool="hello",
                        args={"who": "world"},
                        reasoning="say hi",
                    ),
                    tool_result=NS(error=None, data={"ok": True}, duration_ms=4.0),
                )
            )
            printer.step(
                NS(
                    tool_call=NS(
                        tool="bash",
                        args={"command": "ls"},
                        reasoning="list things",
                    ),
                    tool_result=NS(error="permission denied", data=None, duration_ms=1.0),
                )
            )
            printer.finish(summary="finished")
    finally:
        p_mod._COLOR = original

    out = captured.getvalue()
    # No raw ANSI escape sequences leak into the output.
    assert "\033[" not in out, f"ansi leaked: {out!r}"
    # Header content present.
    assert "testing" in out
    assert "task: smoke" in out
    # Per-step content present.
    assert "step  1" in out
    assert "hello" in out
    assert "why" in out
    assert "say hi" in out
    assert "💭" not in out
    assert "step  2" in out
    assert "permission denied" in out
    # Finish line present with stats.
    assert "done" in out
    assert "2 steps" in out
    assert "1 errors" in out
    assert "agent says: finished" in out


def test_new_help_advertises_pretty() -> None:
    """``looplet new --help`` should mention the --pretty flag."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["new", "--help"])
    assert exc.value.code == 0
    assert "--pretty" in captured.getvalue()


def test_run_workspace_help_advertises_pretty() -> None:
    """``looplet run-workspace --help`` should mention the --pretty flag."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["run-workspace", "--help"])
    assert exc.value.code == 0
    assert "--pretty" in captured.getvalue()
