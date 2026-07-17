#!/usr/bin/env python3
"""Validate package-only surfaces from an installed Looplet wheel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import looplet
from looplet.bundles import load_skill_bundle
from looplet.cartridge import analyse_cartridge
from looplet.cli.factory_commands import _factory_workspace_path


def main() -> int:
    package_root = Path(looplet.__file__).resolve().parent
    factory = _factory_workspace_path().resolve()

    assert package_root in factory.parents
    assert factory == package_root / "_bundled" / "agent_factory.cartridge"
    assert (factory.parent / "coder.cartridge" / "cartridge.json").is_file()

    portable_coder = looplet.bundled_cartridge_path("coder_portable").resolve()
    assert portable_coder == package_root / "_bundled" / "coder_portable.cartridge"
    portability = analyse_cartridge(portable_coder)
    assert portability.profile == "portable"
    assert portability.blockers == ()

    portable_preset = looplet.cartridge_to_preset(portable_coder, strict=True)
    try:
        assert len(portable_preset.mcp_adapters) == 1
        assert len(portable_preset.state_service_handles) == 1
        assert {
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "grep",
            "subagent",
            "done",
        } <= set(portable_preset.tools.tool_names)
    finally:
        portable_preset.close()

    coder_bundle_path = package_root.parent / "tests" / "fixtures" / "coder_skill_bundle"
    coder_bundle = load_skill_bundle(coder_bundle_path)
    assert coder_bundle.skill.name == "coder"
    assert coder_bundle.card.name == "coder"

    preset = looplet.cartridge_to_preset(factory)
    try:
        tool_names = set(preset.tools.tool_names)
        assert {"done", "read_file", "scaffold_cartridge", "validate_workspace"} <= tool_names
    finally:
        preset.close()

    result = subprocess.run(
        [sys.executable, "-m", "looplet", "new", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Scaffold a reviewable Looplet cartridge draft" in result.stdout
    print(f"Installed wheel smoke passed: looplet {looplet.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
