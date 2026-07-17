#!/usr/bin/env python3
"""Validate package-only surfaces from an installed Looplet wheel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import looplet
from looplet.cli.factory_commands import _factory_workspace_path


def main() -> int:
    package_root = Path(looplet.__file__).resolve().parent
    factory = _factory_workspace_path().resolve()

    assert package_root in factory.parents
    assert factory == package_root / "_bundled" / "agent_factory.cartridge"
    assert (factory.parent / "coder.cartridge" / "cartridge.json").is_file()

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
