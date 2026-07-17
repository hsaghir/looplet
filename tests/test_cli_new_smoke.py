"""Smoke tests for ``looplet new`` and ``looplet run-cartridge`` CLI.

The CLI's job is straightforward plumbing - load the factory, run a
loop, surface results - so most of these tests exercise UX paths
(missing env vars, bad cartridge path, factory location lookup)
without spinning up a real LLM.

``run-workspace`` remains covered as a compatibility alias.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from looplet.__main__ import main


def test_new_help() -> None:
    """``looplet new --help`` exits 0 and prints usage."""
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["new", "--help"])
    assert exc.value.code == 0


def test_run_cartridge_help() -> None:
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["run-cartridge", "--help"])
    assert exc.value.code == 0


def test_run_workspace_alias_help() -> None:
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["run-workspace", "--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "--help"],
        ["doctor", "--help"],
        ["describe", "--help"],
        ["diff", "--help"],
        ["hash", "--help"],
        ["portability", "--help"],
        ["eval", "--help"],
        ["eval", "run", "--help"],
    ],
    ids=lambda argv: "-".join(argv[:-1]),
)
def test_documented_command_help(argv: list[str]) -> None:
    """Every command copied into launch docs must remain parseable."""
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(argv)
    assert exc.value.code == 0


def test_new_defaults_to_cartridge_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public default is a cartridge, not the legacy workspace suffix."""
    from looplet.cli import factory_commands

    seen: dict[str, Path] = {}

    def fake_cmd_new(args) -> int:
        seen["target"] = args.target
        return 0

    monkeypatch.setattr(factory_commands, "cmd_new", fake_cmd_new)

    assert main(["new", "a brief"]) == 0
    assert seen["target"] == Path("agent.cartridge")


def test_new_missing_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When required env vars are unset, ``new`` prints a clear error
    and exits 1."""
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    captured = io.StringIO()
    with patch.object(sys, "stderr", captured):
        rc = main(["new", "a brief", str(tmp_path / "out.cartridge")])
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
        rc = main(["new", "a brief", str(tmp_path / "out.cartridge")])
    assert rc == 1
    err = captured.getvalue()
    assert "OPENAI_API_KEY" in err
    assert "OPENAI_MODEL" in err
    # The one we DID set should NOT appear in the missing list.
    assert "missing required env vars: OPENAI_BASE_URL" not in err


def test_run_cartridge_missing_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    rc = main(["run-cartridge", str(tmp_path), "do something"])
    assert rc == 1


def test_run_cartridge_path_must_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    captured = io.StringIO()
    nonexistent = tmp_path / "no_such_dir"
    with patch.object(sys, "stderr", captured):
        rc = main(["run-cartridge", str(nonexistent), "do x"])
    assert rc == 1
    assert "cartridge not found" in captured.getvalue()


def test_run_cartridge_load_failure_uses_cartridge_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import looplet
    from looplet.cli import factory_commands

    for var, value in {
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
        "OPENAI_API_KEY": "test",
        "OPENAI_MODEL": "test-model",
    }.items():
        monkeypatch.setenv(var, value)
    cartridge = tmp_path / "broken.cartridge"
    cartridge.mkdir()
    monkeypatch.setattr(factory_commands, "_build_backend", object)

    def fail_load(*_args, **_kwargs):
        raise ValueError("broken manifest")

    monkeypatch.setattr(looplet, "cartridge_to_preset", fail_load)
    captured = io.StringIO()
    with patch.object(sys, "stderr", captured):
        rc = main(["run-cartridge", str(cartridge), "do x"])

    assert rc == 1
    assert "cartridge load failed: broken manifest" in captured.getvalue()


def test_factory_workspace_path_resolves_in_repo() -> None:
    """``_factory_workspace_path`` finds the bundled
    examples/agent_factory.cartridge when run from the repo."""
    from looplet.cli.factory_commands import _factory_workspace_path

    p = _factory_workspace_path()
    assert p.is_dir()
    assert (p / "cartridge.json").is_file()
    assert (p / "config.yaml").is_file()


def test_factory_workspace_path_resolves_installed_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An installed wheel resolves package data without a repo checkout."""
    from looplet import bundled
    from looplet.cli import factory_commands

    package_root = tmp_path / "site-packages" / "looplet"
    fake_module = package_root / "bundled.py"
    factory = package_root / "_bundled" / "agent_factory.cartridge"
    fake_module.parent.mkdir(parents=True)
    factory.mkdir(parents=True)
    (factory / "cartridge.json").write_text('{"schema_version": 2, "name": "factory"}')

    monkeypatch.delenv("LOOPLET_FACTORY_DIR", raising=False)
    monkeypatch.setattr(bundled, "__file__", str(fake_module))

    assert factory_commands._factory_workspace_path() == factory


def test_new_command_registered_on_top_level() -> None:
    """Top-level help advertises the canonical command and compatibility alias."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["--help"])
    assert exc.value.code == 0
    out = captured.getvalue()
    assert "new" in out
    assert "run-cartridge" in out
    assert "run-workspace" in out


def _new_args(target: Path) -> SimpleNamespace:
    return SimpleNamespace(
        description="summarize a URL",
        target=target,
        name=None,
        tool=None,
        max_steps=None,
        quiet=True,
        pretty=False,
    )


def _patch_new_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, object]:
    import looplet
    from looplet.cli import factory_commands

    config = SimpleNamespace(max_steps=3, system_prompt="Reviewable prompt")
    factory_preset = SimpleNamespace(
        config=config,
        tools=SimpleNamespace(_tools={"done": object()}),
        hooks=[],
    )
    produced_preset = SimpleNamespace(
        config=config,
        tools=SimpleNamespace(_tools={"done": object(), "fetch_url": object()}),
    )
    observed: dict[str, object] = {}

    def fake_load(_path: str, runtime=None):
        return factory_preset if runtime is not None else produced_preset

    def fake_loop(**kwargs):
        observed.update(kwargs)
        return iter(())

    monkeypatch.setenv("OPENAI_MODEL", "mock-model")
    monkeypatch.setattr(factory_commands, "_check_env", lambda: 0)
    monkeypatch.setattr(factory_commands, "_build_backend", object)
    monkeypatch.setattr(factory_commands, "_factory_workspace_path", lambda: tmp_path)
    monkeypatch.setattr(looplet, "cartridge_to_preset", fake_load)
    monkeypatch.setattr(looplet, "composable_loop", fake_loop)
    return observed


def test_new_success_labels_generated_code_as_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "url_summary.cartridge"
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 0

    out = capsys.readouterr().out
    assert "draft built" in out
    assert "produced cartridge draft:" in out
    assert f'looplet run-cartridge {target} "<your task>"' in out
    task = observed["task"]
    assert isinstance(task, dict)
    assert "Scaffold a cartridge draft" in task["goal"]


def test_new_missing_output_uses_cartridge_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "missing.cartridge"
    _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 1

    captured = capsys.readouterr()
    assert "draft built" in captured.out
    assert f"cartridge not created at {target}" in captured.err


def test_new_invalid_output_uses_cartridge_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import looplet
    from looplet.cli import factory_commands

    target = tmp_path / "broken.cartridge"
    target.mkdir()
    _patch_new_runtime(monkeypatch, tmp_path)
    load_factory = looplet.cartridge_to_preset

    def fail_produced_load(path: str, runtime=None):
        if runtime is not None:
            return load_factory(path, runtime=runtime)
        raise ValueError("corrupted manifest")

    monkeypatch.setattr(looplet, "cartridge_to_preset", fail_produced_load)

    assert factory_commands.cmd_new(_new_args(target)) == 1

    captured = capsys.readouterr()
    assert "draft built" in captured.out
    assert "produced cartridge failed to load: corrupted manifest" in captured.err


# ── --pretty (stdlib-only renderer) ────────────────────────────


def test_pretty_printer_renders_steps_without_ansi_when_not_tty() -> None:
    """``PrettyPrinter`` should emit plain text (no escape sequences)
    when stdout isn't a TTY - keeps CI logs and ``tee`` clean."""
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


def test_run_cartridge_help_advertises_pretty() -> None:
    """``looplet run-cartridge --help`` should mention the --pretty flag."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["run-cartridge", "--help"])
    assert exc.value.code == 0
    assert "--pretty" in captured.getvalue()
