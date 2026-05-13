"""Tests for the cartridge spec v2 prep work: sibling ``runtime.yaml``,
field tiering, and the deprecation warning on stray runtime keys in
``config.yaml``.

See ``paper/principled_cartridge_v2.md`` for design rationale.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from looplet import CartridgeLayout, cartridge_to_preset


def _write_minimal_cartridge(
    root: Path, *, config_text: str, runtime_text: str | None = None
) -> None:
    """Build a tiny but loadable cartridge skeleton at ``root``."""
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 1}\n')
    (root / "config.yaml").write_text(config_text)
    if runtime_text is not None:
        (root / "runtime.yaml").write_text(runtime_text)
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("you are a tester")
    (root / "tools" / "done").mkdir(parents=True)
    (root / "tools" / "done" / "tool.yaml").write_text(
        "name: done\ndescription: Finish.\nparameters:\n  summary:\n    type: string\n"
    )
    (root / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx, *, summary: str) -> dict:\n    return {'summary': summary}\n"
    )


def test_field_tiering_carves_loopconfig_into_three_disjoint_buckets() -> None:
    """RUNTIME, HOST, and CONTRACT tiers must be disjoint."""
    runtime = CartridgeLayout.RUNTIME_TIER_FIELDS
    host = CartridgeLayout.HOST_TIER_FIELDS
    contract = CartridgeLayout.contract_tier_fields()
    assert runtime.isdisjoint(host)
    assert runtime.isdisjoint(contract)
    # CONTRACT comes from SERIALIZABLE - RUNTIME, so HOST may not
    # appear there (HOST callables aren't serialisable). Sanity-check
    # the partition covers the documented serialisable surface.
    assert (runtime | contract) >= set(CartridgeLayout.SERIALIZABLE_CONFIG_FIELDS) - {
        "tool_metadata",
        "generate_kwargs",
    }


def test_runtime_yaml_loads_runtime_keys(tmp_path: Path) -> None:
    """Keys in ``runtime.yaml`` populate the same ``LoopConfig`` fields."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="max_tokens: 1234\ntemperature: 0.7\n",
    )
    preset = cartridge_to_preset(str(tmp_path))
    assert preset.config.max_tokens == 1234
    assert preset.config.temperature == 0.7
    assert preset.config.max_steps == 5


def test_runtime_yaml_overrides_config_yaml_for_runtime_keys(tmp_path: Path) -> None:
    """Precedence: runtime.yaml wins over config.yaml for runtime keys."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\nmax_tokens: 100\n",
        runtime_text="max_tokens: 999\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        preset = cartridge_to_preset(str(tmp_path))
    assert preset.config.max_tokens == 999


def test_stray_runtime_key_in_config_yaml_emits_deprecation(tmp_path: Path) -> None:
    """A runtime-tier key in config.yaml fires DeprecationWarning naming the key."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\nmax_tokens: 2000\ntemperature: 0.3\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        cartridge_to_preset(str(tmp_path))
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1
    msg = str(dep[0].message)
    assert "max_tokens" in msg and "temperature" in msg
    assert "runtime.yaml" in msg


def test_runtime_yaml_present_silences_warning(tmp_path: Path) -> None:
    """When the runtime key is in runtime.yaml, no warning fires."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="max_tokens: 2000\ntemperature: 0.3\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        cartridge_to_preset(str(tmp_path))
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep == []


def test_runtime_yaml_rejects_contract_tier_keys(tmp_path: Path) -> None:
    """``runtime.yaml`` may only declare runtime-tier fields; contract
    keys (max_steps, system_prompt, done_tool, model, permissions) are
    rejected with a structured error in strict mode."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 5\ndone_tool: done\n",
        runtime_text="max_steps: 99\n",  # contract key in wrong file
    )
    with pytest.raises(Exception) as exc_info:
        cartridge_to_preset(str(tmp_path), strict=True)
    assert "runtime.yaml" in str(exc_info.value)
    assert "max_steps" in str(exc_info.value)


def test_extends_inherits_parent_runtime_yaml(tmp_path: Path) -> None:
    """A child cartridge that ``extends:`` a parent picks up the
    parent's ``runtime.yaml`` defaults — without this, splitting
    runtime knobs out would silently drop them from descendants."""
    parent = tmp_path / "parent.cartridge"
    parent.mkdir()
    _write_minimal_cartridge(
        parent,
        config_text="max_steps: 10\ndone_tool: done\n",
        runtime_text="max_tokens: 7777\ntemperature: 0.42\n",
    )

    child = tmp_path / "child.cartridge"
    child.mkdir()
    _write_minimal_cartridge(
        child,
        config_text="max_steps: 20\ndone_tool: done\nextends: ../parent.cartridge\n",
    )
    # Remove child's own done tool so the extends picks up parent's.
    # (The minimal scaffold writes one for both; that's fine because
    # extends overlays the directories.)
    preset = cartridge_to_preset(str(child))
    assert preset.config.max_steps == 20  # child overrides parent
    assert preset.config.max_tokens == 7777  # inherited from parent runtime.yaml
    assert preset.config.temperature == 0.42
