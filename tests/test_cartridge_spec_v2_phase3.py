"""Cartridge spec v2 - Phase 3 hard removals + migration tool.

* ``schema_version: 2`` hard-fails on:
    - runtime-tier keys in config.yaml
    - magic ``prompts/briefing.md`` / ``prompts/recovery.md`` auto-load
    - ``setup.py`` escape hatch
* ``looplet migrate`` mechanically upgrades v1.x → v2.0 in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from looplet import cartridge_to_preset
from looplet.cartridge._load import CartridgeSerializationError
from looplet.cli.spec_commands import cartridge_migrate, cmd_migrate


def _write_v_cartridge(root: Path, schema_version: int, *, config_text: str) -> None:
    (root / "cartridge.json").write_text(
        json.dumps({"name": "x", "schema_version": schema_version}) + "\n"
    )
    (root / "config.yaml").write_text(config_text)
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("you are a tester")
    (root / "tools" / "done").mkdir(parents=True)
    (root / "tools" / "done" / "tool.yaml").write_text(
        "name: done\ndescription: Finish.\nparameters:\n  summary:\n    type: string\n"
    )
    (root / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx, *, summary: str) -> dict:\n    return {'summary': summary}\n"
    )


# ── v2 hard-errors ──────────────────────────────────────────────────


def test_declaration_inspection_resolves_inheritance_without_writing(tmp_path: Path) -> None:
    from looplet.cartridge import inspect_cartridge

    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    _write_v_cartridge(parent, schema_version=2, config_text="max_steps: 7\ndone_tool: done\n")
    _write_v_cartridge(child, schema_version=2, config_text="extends: ../parent\n")
    (parent / "runtime.yaml").write_text("temperature: 0.1\nmax_tokens: 80\n")
    (child / "runtime.yaml").write_text("temperature: 0.7\n")
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    report = inspect_cartridge(child)
    assert report["config"]["max_steps"] == 7
    assert report["config"]["temperature"] == 0.7
    assert report["config"]["max_tokens"] == 80
    assert report["configuration_sources"]["max_steps"] == str(parent / "config.yaml")
    assert report["configuration_sources"]["temperature"] == str(child / "runtime.yaml")
    assert report["configuration_sources"]["max_tokens"] == str(parent / "runtime.yaml")
    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_declaration_inspection_preserves_unresolved_runtime_and_redacts(tmp_path: Path) -> None:
    from looplet.cartridge import inspect_cartridge

    _write_v_cartridge(tmp_path, schema_version=2, config_text="max_steps: ${runtime.limit}\n")
    (tmp_path / "runtime.yaml").write_text("generate_kwargs:\n  api_key: private-key\n")
    report = inspect_cartridge(tmp_path)
    assert report["config"]["max_steps"] == "${runtime.limit}"
    assert "${runtime.limit}" in report["runtime_required"]
    assert "private-key" not in json.dumps(report)
    assert report["config"]["system_prompt"] == "[redacted]"
    resolved = inspect_cartridge(tmp_path, runtime={"limit": 4}, include_sensitive=True)
    assert resolved["config"]["max_steps"] == 4
    assert resolved["config"]["generate_kwargs"]["api_key"] == "private-key"


def test_declaration_inspection_rejects_extends_cycle(tmp_path: Path) -> None:
    from looplet.cartridge import inspect_cartridge

    _write_v_cartridge(tmp_path, schema_version=2, config_text="extends: .\n")
    with pytest.raises(CartridgeSerializationError, match="circular extends"):
        inspect_cartridge(tmp_path)


def test_declaration_inspection_compiles_model_defaults_and_marks_refs(tmp_path: Path) -> None:
    from looplet.blueprints import blueprint_from_cartridge, compare_blueprints
    from looplet.cartridge import inspect_cartridge

    _write_v_cartridge(tmp_path, schema_version=2, config_text="model:\n  max_tokens: 123\n")
    (tmp_path / "runtime.yaml").write_text("max_tokens: 20\ncompact_service: ${ref:compact}\n")
    report = inspect_cartridge(tmp_path)
    assert report["config"]["max_tokens"] == 123
    assert report["configuration_sources"]["max_tokens"] == str(tmp_path / "config.yaml")
    assert "${ref:compact}" in report["runtime_required"]
    blueprint = blueprint_from_cartridge(tmp_path)
    assert blueprint.name == "x"
    assert [tool.name for tool in blueprint.tools] == ["done"]
    assert not compare_blueprints(blueprint, blueprint).ok


@pytest.mark.parametrize("hooks", ["true", "[3]", "[{}]", "[{a: {}, b: {}}]"])
def test_declaration_inspection_rejects_malformed_builtin_hooks(tmp_path: Path, hooks: str) -> None:
    from looplet.cartridge import inspect_cartridge

    _write_v_cartridge(tmp_path, schema_version=2, config_text=f"builtin_hooks: {hooks}\n")
    with pytest.raises(CartridgeSerializationError, match="builtin_hooks"):
        inspect_cartridge(tmp_path)


def test_declaration_inspection_redacts_protocol_connection_config(tmp_path: Path) -> None:
    from looplet.cartridge import inspect_cartridge

    _write_v_cartridge(tmp_path, schema_version=2, config_text="max_steps: 4\n")
    (tmp_path / "runtime.yaml").write_text(
        "mcp_servers:\n  remote:\n    command: not-a-program\n    env:\n      TOKEN: connection-secret\n"
    )
    report = inspect_cartridge(tmp_path)
    assert "connection-secret" not in json.dumps(report)
    assert (
        inspect_cartridge(tmp_path, include_sensitive=True)["config"]["mcp_servers"]["remote"][
            "env"
        ]["TOKEN"]
        == "connection-secret"
    )


def test_v2_configuration_explanation_reports_file_origins(tmp_path: Path) -> None:
    _write_v_cartridge(tmp_path, schema_version=2, config_text="max_steps: 3\ndone_tool: done\n")
    (tmp_path / "runtime.yaml").write_text("temperature: 0.7\n")
    with cartridge_to_preset(tmp_path) as preset:
        explained = preset.config.explain()
        assert explained["max_steps"]["source"] == "config.yaml"
        assert explained["temperature"]["source"] == "runtime.yaml"
        assert explained["system_prompt"]["source"] == "prompts/system.md"
        assert explained["max_tokens"]["source"] == "default"


def test_v2_rejects_runtime_keys_in_config_yaml(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=2,
        config_text="max_steps: 3\ndone_tool: done\nmax_tokens: 2000\n",
    )
    with pytest.raises(CartridgeSerializationError, match="runtime-tier key"):
        cartridge_to_preset(tmp_path)


def test_v2_rejects_magic_briefing_md(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=2,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "prompts" / "briefing.md").write_text("be brief")
    with pytest.raises(CartridgeSerializationError, match="briefing.md"):
        cartridge_to_preset(tmp_path)


def test_v2_rejects_magic_recovery_md(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=2,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "prompts" / "recovery.md").write_text("retry idea")
    with pytest.raises(CartridgeSerializationError, match="recovery.md"):
        cartridge_to_preset(tmp_path)


def test_v2_rejects_setup_py(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=2,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "setup.py").write_text("def setup(preset, resources): return preset\n")
    with pytest.raises(CartridgeSerializationError, match="setup.py"):
        cartridge_to_preset(tmp_path)


def test_v2_accepts_explicit_builtin_hook_replacement(tmp_path: Path) -> None:
    """A v2 cartridge using ``builtin_hooks:`` (instead of magic files) loads fine."""
    _write_v_cartridge(
        tmp_path,
        schema_version=2,
        config_text=(
            "max_steps: 3\n"
            "done_tool: done\n"
            "builtin_hooks:\n"
            "  - static_briefing:\n"
            "      text: be brief\n"
        ),
    )
    preset = cartridge_to_preset(tmp_path)
    from looplet.cartridge.prompt_files import StaticBriefingHook

    assert any(isinstance(h, StaticBriefingHook) for h in preset.hooks)


# ── looplet migrate ────────────────────────────────────────────────


def test_migrate_splits_runtime_keys_and_bumps_schema(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=1,
        config_text=("max_steps: 3\ndone_tool: done\nmax_tokens: 2000\ntemperature: 0.2\n"),
    )
    report = cartridge_migrate(tmp_path)
    assert report["schema_version_before"] == 1
    assert report["schema_version_after"] == 2
    assert sorted(report["moved_runtime_keys"]) == ["max_tokens", "temperature"]

    # Manifest bumped.
    manifest = json.loads((tmp_path / "cartridge.json").read_text())
    assert manifest["schema_version"] == 2

    # runtime.yaml now carries the keys.
    rt_text = (tmp_path / "runtime.yaml").read_text()
    assert "max_tokens" in rt_text and "temperature" in rt_text

    # config.yaml no longer has them.
    cfg_text = (tmp_path / "config.yaml").read_text()
    assert "max_tokens" not in cfg_text
    assert "temperature" not in cfg_text

    # Migrated cartridge loads cleanly under v2 (no deprecation warnings).
    preset = cartridge_to_preset(tmp_path)
    assert preset.config.max_steps == 3


def test_migrate_converts_magic_briefing_to_builtin_hook(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=1,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "prompts" / "briefing.md").write_text("be brief")
    (tmp_path / "prompts" / "recovery.md").write_text("retry hint")
    report = cartridge_migrate(tmp_path)
    assert sorted(report["added_builtin_hooks"]) == ["recovery_hint", "static_briefing"]

    cfg_text = (tmp_path / "config.yaml").read_text()
    assert "static_briefing" in cfg_text
    assert "recovery_hint" in cfg_text
    # Loads cleanly under v2.
    cartridge_to_preset(tmp_path)


def test_migrate_refuses_with_setup_py(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=1,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "setup.py").write_text("def setup(preset, resources): return preset\n")
    with pytest.raises(RuntimeError, match="setup.py"):
        cartridge_migrate(tmp_path)


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=1,
        config_text="max_steps: 3\ndone_tool: done\nmax_tokens: 2000\n",
    )
    cartridge_migrate(tmp_path)
    # Second run is a no-op (already v2, nothing to move/add).
    report = cartridge_migrate(tmp_path)
    assert report["moved_runtime_keys"] == []
    assert report["added_builtin_hooks"] == []
    assert report["schema_version_after"] == 2
    assert report["changed"] is False
    assert report["wrote_files"] == []


def test_migrate_cli_reports_already_current(tmp_path: Path, capsys) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=2,
        config_text="max_steps: 3\ndone_tool: done\n",
    )

    assert cmd_migrate(SimpleNamespace(cartridge=tmp_path, dry_run=False)) == 0
    assert "already current" in capsys.readouterr().out


def test_migrate_dry_run_does_not_write(tmp_path: Path) -> None:
    _write_v_cartridge(
        tmp_path,
        schema_version=1,
        config_text="max_steps: 3\ndone_tool: done\nmax_tokens: 2000\n",
    )
    cfg_before = (tmp_path / "config.yaml").read_text()
    report = cartridge_migrate(tmp_path, dry_run=True)
    assert report["moved_runtime_keys"] == ["max_tokens"]
    assert (tmp_path / "config.yaml").read_text() == cfg_before
    assert not (tmp_path / "runtime.yaml").exists()
