from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import Cartridge, CartridgeSerializationError, cartridge_to_preset


def _cartridge(tmp_path: Path, compatibility: dict | None = None) -> Path:
    root = tmp_path / "compat.cartridge"
    root.mkdir()
    manifest = {"name": "compat", "schema_version": 2}
    if compatibility is not None:
        manifest["compatibility"] = compatibility
    (root / "cartridge.json").write_text(json.dumps(manifest))
    (root / "config.yaml").write_text("max_steps: 1\ndone_tool: done\n")
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("compat\n")
    (root / "tools" / "done").mkdir(parents=True)
    (root / "tools" / "done" / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters: {}\n"
    )
    (root / "tools" / "done" / "execute.py").write_text(
        "def execute(ctx):\n    return {'ok': True}\n"
    )
    return root


def test_legacy_manifest_has_empty_compatibility(tmp_path: Path) -> None:
    cartridge = Cartridge.from_directory(_cartridge(tmp_path))
    assert cartridge.compatibility.to_dict() == {}


def test_supported_compatibility_loads(tmp_path: Path) -> None:
    root = _cartridge(
        tmp_path,
        {
            "looplet": ">=0.4,<0.6",
            "languages": ["python"],
            "requires": ["async_tool_dispatch", "run_store"],
            "rpc": ["1.0"],
        },
    )
    preset = cartridge_to_preset(root, strict=True)
    preset.close()


@pytest.mark.parametrize(
    "compatibility, message",
    [
        ({"looplet": ">=9.0"}, "requires looplet"),
        ({"looplet": ""}, "invalid Looplet version constraint"),
        ({"looplet": ">=0.4,"}, "invalid Looplet version constraint"),
        ({"languages": ["rust"]}, "does not support"),
        ({"requires": ["future_feature"]}, "unsupported"),
        ({"rpc": ["9.9"]}, "requires RPC"),
    ],
)
def test_incompatible_cartridge_rejected_before_body_load(
    tmp_path: Path,
    compatibility: dict,
    message: str,
) -> None:
    root = _cartridge(tmp_path, compatibility)
    (root / "tools" / "done" / "execute.py").write_text("raise RuntimeError('body loaded')\n")

    with pytest.raises(CartridgeSerializationError, match=message):
        cartridge_to_preset(root, strict=True)
