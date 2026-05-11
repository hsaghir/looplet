"""Cartridge-spec naming aliases.

The Cartridge Spec v1.0 calls the artifact a "cartridge". The
reference implementation has historically used "workspace". Both
terms refer to the same on-disk format.

These tests pin the alias surface:

* ``looplet.cartridge_to_preset`` is ``looplet.workspace_to_preset``.
* ``looplet.scaffold_cartridge`` is ``looplet.scaffold_workspace``.
* ``cartridge.json`` loads identically to ``workspace.json``.
* ``CartridgeSerializationError`` is ``WorkspaceSerializationError``.
* ``Cartridge`` is ``Workspace``.

Removing or weakening any of these aliases is a breaking spec change
and requires a v2 RFC.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import looplet

REPO = Path(__file__).resolve().parents[1]


def test_cartridge_aliases_are_the_same_objects() -> None:
    assert looplet.cartridge_to_preset is looplet.workspace_to_preset
    assert looplet.preset_to_cartridge is looplet.preset_to_workspace
    assert looplet.scaffold_cartridge is looplet.scaffold_workspace
    assert looplet.CartridgeSerializationError is looplet.WorkspaceSerializationError
    assert looplet.Cartridge is looplet.Workspace
    assert looplet.CartridgeLayout is looplet.WorkspaceLayout


def test_cartridge_aliases_in_public_all() -> None:
    for sym in (
        "Cartridge",
        "CartridgeLayout",
        "CartridgeSerializationError",
        "cartridge_to_preset",
        "preset_to_cartridge",
        "scaffold_cartridge",
    ):
        assert sym in looplet.__all__, f"{sym} missing from looplet.__all__"


def test_loader_accepts_cartridge_json_manifest(tmp_path: Path) -> None:
    """A cartridge using cartridge.json instead of workspace.json loads."""
    src = REPO / "examples" / "hello.workspace"
    dst = tmp_path / "hello.cartridge"
    shutil.copytree(src, dst)
    (dst / "workspace.json").rename(dst / "cartridge.json")
    ws = looplet.Workspace.from_directory(dst)
    assert ws.name == "hello"
    preset = looplet.cartridge_to_preset(str(dst), runtime={"workspace": str(dst)})
    assert "done" in preset.tools.tool_names


def test_loader_prefers_workspace_json_when_both_exist(tmp_path: Path) -> None:
    """Back-compat: workspace.json wins if both exist (no surprise rename)."""
    src = REPO / "examples" / "hello.workspace"
    dst = tmp_path / "hello.workspace"
    shutil.copytree(src, dst)
    # Add a divergent cartridge.json that would change the loaded name.
    (dst / "cartridge.json").write_text('{"name": "wrong", "schema_version": 1}\n')
    ws = looplet.Workspace.from_directory(dst)
    assert ws.name == "hello", "workspace.json must win when both manifests exist"


def test_loader_rejects_directory_with_no_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        looplet.cartridge_to_preset(str(empty))
    msg = str(exc_info.value)
    # Error must name a manifest filename so the user knows what's missing.
    assert "workspace.json" in msg
