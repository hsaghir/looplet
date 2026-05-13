"""Cartridge spec v2 — Phase 3 hard removals + migration tool.

* ``schema_version: 2`` hard-fails on:
    - runtime-tier keys in config.yaml
    - magic ``prompts/briefing.md`` / ``prompts/recovery.md`` auto-load
    - ``setup.py`` escape hatch
* ``looplet migrate`` mechanically upgrades v1.x → v2.0 in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import cartridge_to_preset
from looplet.cartridge._load import CartridgeSerializationError
from looplet.cli.spec_commands import cartridge_migrate


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
