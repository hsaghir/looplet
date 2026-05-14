"""Cartridge spec v2 — Phase 2 features.

* ``static_briefing`` / ``recovery_hint`` as builtin_hooks
* ``thread_safe:`` resource declaration
* ``looplet hash`` canonical content hash
* DeprecationWarning on magic prompts/briefing.md and prompts/recovery.md
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from looplet import cartridge_to_preset


def _write_minimal_cartridge(root: Path, *, config_text: str) -> None:
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
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


# ── builtin_hooks: static_briefing + recovery_hint ──────────────────


def test_static_briefing_inline_text_loads_via_builtin_hook(tmp_path: Path) -> None:
    """``builtin_hooks: - static_briefing: { text: ... }`` instantiates the hook."""
    from looplet.cartridge.prompt_files import StaticBriefingHook

    _write_minimal_cartridge(
        tmp_path,
        config_text=(
            "max_steps: 3\n"
            "done_tool: done\n"
            "builtin_hooks:\n"
            "  - static_briefing:\n"
            "      text: |-\n"
            "        Be concise.\n"
        ),
    )
    preset = cartridge_to_preset(tmp_path)
    briefing = [h for h in preset.hooks if isinstance(h, StaticBriefingHook)]
    assert len(briefing) == 1
    assert briefing[0].text == "Be concise."


def test_recovery_hint_path_loads_relative_to_cartridge_root(tmp_path: Path) -> None:
    """``recovery_hint: { path: ... }`` resolves against ``cartridge_root``."""
    from looplet.cartridge.prompt_files import RecoveryHintHook

    _write_minimal_cartridge(
        tmp_path,
        config_text=(
            "max_steps: 3\n"
            "done_tool: done\n"
            "builtin_hooks:\n"
            "  - recovery_hint:\n"
            "      path: hints.md\n"
        ),
    )
    (tmp_path / "hints.md").write_text("If the tool errored, try again with simpler args.")
    preset = cartridge_to_preset(tmp_path)
    recovery = [h for h in preset.hooks if isinstance(h, RecoveryHintHook)]
    assert len(recovery) == 1
    assert "simpler args" in recovery[0].text


def test_static_briefing_rejects_text_and_path_together(tmp_path: Path) -> None:
    """``text:`` and ``path:`` are mutually exclusive."""
    _write_minimal_cartridge(
        tmp_path,
        config_text=(
            "max_steps: 3\n"
            "done_tool: done\n"
            "builtin_hooks:\n"
            "  - static_briefing:\n"
            "      text: hi\n"
            "      path: hi.md\n"
        ),
    )
    # Strict surfaces the ValueError as CartridgeSerializationError.
    from looplet import CartridgeSerializationError

    with pytest.raises(CartridgeSerializationError, match=r"either ``text:``"):
        cartridge_to_preset(tmp_path, strict=True)


def test_thread_safe_resource_declaration_recorded(tmp_path: Path) -> None:
    """``THREAD_SAFE = True`` in a resource module is stashed in the
    ``_resource_thread_safety`` parallel registry."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "fast_db.py").write_text(
        "THREAD_SAFE = True\n\ndef build():\n    return object()\n"
    )
    (tmp_path / "resources" / "shared_state.py").write_text(
        "THREAD_SAFE = False\n\ndef build():\n    return {}\n"
    )
    preset = cartridge_to_preset(tmp_path)
    safety = preset.resources.get("_resource_thread_safety", {})
    assert safety["fast_db"] is True
    assert safety["shared_state"] is False


def test_thread_safe_must_be_bool(tmp_path: Path) -> None:
    """Non-bool ``THREAD_SAFE`` raises at load time."""
    from looplet import CartridgeSerializationError

    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "weird.py").write_text(
        "THREAD_SAFE = 'sometimes'\n\ndef build():\n    return object()\n"
    )
    with pytest.raises(CartridgeSerializationError, match=r"THREAD_SAFE"):
        cartridge_to_preset(tmp_path)


def test_resource_without_thread_safe_omitted_from_registry(tmp_path: Path) -> None:
    """Resources that don't declare THREAD_SAFE are treated as unknown."""
    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "anon.py").write_text("def build():\n    return object()\n")
    preset = cartridge_to_preset(tmp_path)
    safety = preset.resources.get("_resource_thread_safety", {})
    assert "anon" not in safety


# ── looplet hash ─────────────────────────────────────────────────────


def test_cartridge_hash_is_stable_across_runs(tmp_path: Path) -> None:
    """Same content → same digest. Idempotent."""
    from looplet.cli.spec_commands import cartridge_hash

    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    digest1, files1 = cartridge_hash(tmp_path)
    digest2, files2 = cartridge_hash(tmp_path)
    assert digest1 == digest2
    assert files1 == files2
    assert len(digest1) == 64  # sha256 hex


def test_cartridge_hash_changes_on_content_edit(tmp_path: Path) -> None:
    """Edit any content-bearing file → digest changes."""
    from looplet.cli.spec_commands import cartridge_hash

    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    before, _ = cartridge_hash(tmp_path)
    (tmp_path / "prompts" / "system.md").write_text("you are a different tester")
    after, _ = cartridge_hash(tmp_path)
    assert before != after


def test_cartridge_hash_ignores_excluded_dirs(tmp_path: Path) -> None:
    """``__pycache__/``, ``seed/``, ``.git/`` must not affect the hash."""
    from looplet.cli.spec_commands import cartridge_hash

    _write_minimal_cartridge(
        tmp_path,
        config_text="max_steps: 3\ndone_tool: done\n",
    )
    before, _ = cartridge_hash(tmp_path)
    # Add files under excluded directories.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed" / "data.json").write_text("{}")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: x")
    after, _ = cartridge_hash(tmp_path)
    assert before == after, "excluded dirs should not affect hash"
