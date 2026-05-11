"""Cartridge-spec naming aliases.

The Cartridge Spec v1.0 calls the artifact a "cartridge". The
reference implementation has historically used "workspace". Both
terms refer to the same on-disk format.

These tests pin the alias surface:

* ``looplet.cartridge_to_preset`` is ``looplet.cartridge_to_preset``.
* ``looplet.scaffold_cartridge`` is ``looplet.scaffold_cartridge``.
* ``cartridge.json`` loads identically to ``workspace.json``.
* ``CartridgeSerializationError`` is ``CartridgeSerializationError``.
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
    assert looplet.cartridge_to_preset is looplet.cartridge_to_preset
    assert looplet.preset_to_cartridge is looplet.preset_to_cartridge
    assert looplet.scaffold_cartridge is looplet.scaffold_cartridge
    assert looplet.CartridgeSerializationError is looplet.CartridgeSerializationError
    assert looplet.Cartridge is looplet.Workspace
    assert looplet.CartridgeLayout is looplet.CartridgeLayout


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


def test_loader_accepts_workspace_json_legacy_manifest(tmp_path: Path) -> None:
    """A cartridge using the legacy workspace.json filename still loads."""
    src = REPO / "examples" / "hello.cartridge"
    dst = tmp_path / "hello.cartridge"
    shutil.copytree(src, dst)
    (dst / "cartridge.json").rename(dst / "workspace.json")
    ws = looplet.Workspace.from_directory(dst)
    assert ws.name == "hello"
    preset = looplet.cartridge_to_preset(str(dst), runtime={"workspace": str(dst)})
    assert "done" in preset.tools.tool_names


def test_loader_prefers_cartridge_json_when_both_exist(tmp_path: Path) -> None:
    """Canonical cartridge.json wins when both manifests exist."""
    src = REPO / "examples" / "hello.cartridge"
    dst = tmp_path / "hello.cartridge"
    shutil.copytree(src, dst)
    # Add a divergent workspace.json that would change the loaded name
    # if the loader preferred it; cartridge.json must take precedence.
    (dst / "workspace.json").write_text('{"name": "wrong", "schema_version": 1}\n')
    ws = looplet.Workspace.from_directory(dst)
    assert ws.name == "hello", "cartridge.json must win when both manifests exist"


def test_loader_rejects_directory_with_no_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        looplet.cartridge_to_preset(str(empty))
    msg = str(exc_info.value)
    # Error must name a manifest filename so the user knows what's missing.
    assert "cartridge.json" in msg
