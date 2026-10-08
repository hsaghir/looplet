"""Smoke tests for ``looplet new`` and ``looplet run-cartridge`` CLI.

The CLI's job is straightforward plumbing - load the factory, run a
loop, surface results - so most of these tests exercise UX paths
(missing env vars, bad cartridge path, factory location lookup)
without spinning up a real LLM.

``run-workspace`` remains covered as a compatibility alias.
"""

from __future__ import annotations

import io
import json
import os
import shlex
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


def test_show_help_advertises_json() -> None:
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["show", "--help"])
    assert exc.value.code == 0
    assert "--json" in captured.getvalue()


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
    captured = io.StringIO()
    with patch.dict(os.environ, {}, clear=True), patch.object(sys, "stderr", captured):
        rc = main(["new", "a brief", str(tmp_path / "out.cartridge")])
    assert rc == 1
    err = captured.getvalue()
    assert "could not resolve a provider" in err
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert var in err


@pytest.mark.parametrize(
    "environment",
    [
        {"OPENAI_BASE_URL": "http://localhost:1234/v1"},
        {"OPENAI_API_KEY": "test"},
        {"ANTHROPIC_API_KEY": "test"},
    ],
)
def test_cli_accepts_provider_defaults(environment: dict[str, str]) -> None:
    from looplet.cli import factory_commands

    with patch.dict(os.environ, environment, clear=True):
        assert factory_commands._check_env() == 0


def test_new_offline_scaffolds_without_backend(tmp_path, monkeypatch, capsys) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "offline.cartridge"
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: pytest.fail("offline scaffolding must not construct a backend"),
    )
    assert (
        main(["new", "Look up service owners", str(target), "--offline", "--tool", "lookup"]) == 0
    )
    assert json.loads((target / "cartridge.json").read_text())["schema_version"] == 2
    assert "Look up service owners" in (target / "prompts" / "system.md").read_text()
    assert (target / "tools" / "lookup" / "execute.py").is_file()
    assert (target / "tools" / "done" / "execute.py").is_file()
    assert "scaffolded cartridge draft" in capsys.readouterr().out


def test_new_offline_preserves_existing_files(tmp_path, capsys) -> None:
    target = tmp_path / "existing.cartridge"
    target.mkdir()
    original = target / "original.txt"
    original.write_text("keep this")

    assert main(["new", "draft", str(target), "--offline"]) == 1
    assert original.read_text() == "keep this"
    assert "error:" in capsys.readouterr().err


def test_inspect_cartridge_does_not_execute_tool_code(tmp_path, capsys) -> None:
    from looplet import scaffold_cartridge

    target = scaffold_cartridge(tmp_path / "safe.cartridge", name="safe", tools=[])
    (target / "tools" / "done" / "execute.py").write_text("raise AssertionError('must not import')")

    assert main(["inspect", str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tools"][0]["name"] == "done"


def test_inspect_bundle_does_not_import_entrypoint(tmp_path, capsys) -> None:
    target = tmp_path / "safe-bundle"
    target.mkdir()
    (target / "SKILL.md").write_text(
        "---\nname: safe\ndescription: Safe metadata.\nentrypoint: looplet.py\n---\n"
    )
    (target / "looplet.py").write_text("raise AssertionError('must not import')")

    assert main(["inspect", str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "safe"
    assert payload["ok"] is True


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
    from looplet import AgentPreset, DefaultState, LoopConfig, MockLLMBackend, tools_from
    from looplet.cli import factory_commands

    closed: list[str] = []
    resource = SimpleNamespace(close=lambda: closed.append("factory"))
    config = LoopConfig(max_steps=3, system_prompt="Reviewable prompt")
    factory_preset = AgentPreset(
        config=config,
        tools=tools_from([], include_done=True),
        hooks=[],
        state=DefaultState(max_steps=3),
        resources={"client": resource},
        owned_resources=[resource],
    )
    produced_preset = SimpleNamespace(
        config=config,
        tools=SimpleNamespace(_tools={"done": object(), "fetch_url": object()}),
        close=lambda: closed.append("produced"),
    )
    observed: dict[str, object] = {"closed": closed}

    def fake_load(_path: str, runtime=None):
        return factory_preset if runtime is not None else produced_preset

    def fake_loop(**kwargs):
        observed.update(kwargs)
        return iter(())

    monkeypatch.setenv("OPENAI_MODEL", "mock-model")
    monkeypatch.setattr(factory_commands, "_check_env", lambda: 0)
    monkeypatch.setattr(factory_commands, "_build_backend", lambda: MockLLMBackend([]))
    monkeypatch.setattr(factory_commands, "_factory_workspace_path", lambda: tmp_path)
    monkeypatch.setattr(looplet, "cartridge_to_preset", fake_load)
    monkeypatch.setattr("looplet.loop.composable_loop", fake_loop)
    return observed


@pytest.mark.parametrize("target_name", ["url_summary.cartridge", "service owners.cartridge"])
def test_new_success_labels_generated_code_as_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target_name: str,
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / target_name
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 0

    out = capsys.readouterr().out
    assert "draft built" in out
    assert "produced cartridge draft:" in out
    command = next(
        line.strip() for line in out.splitlines() if line.strip().startswith("looplet run ")
    )
    assert shlex.split(command) == ["looplet", "run", str(target), "<your task>"]
    task = observed["task"]
    assert isinstance(task, dict)
    assert "Scaffold a cartridge draft" in task["goal"]


@pytest.mark.parametrize("max_steps", [1, 5])
def test_new_max_steps_updates_config_and_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_steps: int,
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "url_summary.cartridge"
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)
    args = _new_args(target)
    args.max_steps = max_steps

    assert factory_commands.cmd_new(args) == 0
    assert observed["config"].max_steps == max_steps
    assert observed["state"].max_steps == max_steps


def test_new_closes_factory_and_produced_presets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "url_summary.cartridge"
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 0
    assert observed["closed"] == ["factory", "produced"]


def test_new_max_steps_limits_actual_factory_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import looplet
    from looplet import MockLLMBackend, tool
    from looplet.cli import factory_commands
    from looplet.loop import composable_loop

    target = tmp_path / "url_summary.cartridge"
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr("looplet.loop.composable_loop", composable_loop)
    preset = looplet.cartridge_to_preset(str(tmp_path), runtime={})
    calls: list[bool] = []

    @tool
    def work() -> dict:
        calls.append(True)
        return {"ok": True}

    preset.tools.register(work)
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: MockLLMBackend(['{"tool":"work","args":{}}'] * 3),
    )
    args = _new_args(target)
    args.max_steps = 1

    assert factory_commands.cmd_new(args) == 0
    assert len(calls) == 1
    assert preset.state.step_count == 1
    assert observed["closed"] == ["factory", "produced"]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_new_closes_factory_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[BaseException],
) -> None:
    from looplet.cli import factory_commands

    observed = _patch_new_runtime(monkeypatch, tmp_path)

    def fail_loop(**kwargs):
        raise failure("factory interrupted")

    monkeypatch.setattr("looplet.loop.composable_loop", fail_loop)

    assert factory_commands.cmd_new(_new_args(tmp_path / "draft.cartridge")) == (
        130 if failure is KeyboardInterrupt else 1
    )
    assert observed["closed"] == ["factory"]


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


def test_run_cartridge_help_advertises_runtime_options() -> None:
    """``looplet run-cartridge --help`` should name its runtime controls."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["run-cartridge", "--help"])
    assert exc.value.code == 0
    out = captured.getvalue()
    assert "--pretty" in out
    assert "--trace-dir" in out
    assert "--parent-trace" in out
    assert "--no-trace" in out
    assert "--json" in out
    assert "read it from stdin" in out
