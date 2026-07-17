"""Locate reference cartridges shipped with Looplet."""

from __future__ import annotations

from pathlib import Path

__all__ = ["bundled_cartridge_path"]


def bundled_cartridge_path(name: str) -> Path:
    """Return a shipped cartridge from a source checkout or installed package.

    ``name`` may be supplied with or without the ``.cartridge`` suffix. Only a
    single path component is accepted; callers cannot escape the bundled
    cartridge directory.
    """
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("cartridge name must be one non-empty path component")

    directory_name = name if name.endswith(".cartridge") else f"{name}.cartridge"
    here = Path(__file__).resolve()

    installed_candidate = here.parent / "_bundled" / directory_name
    if installed_candidate.is_dir():
        return installed_candidate

    for parent in here.parents:
        source_candidate = parent / "examples" / directory_name
        if source_candidate.is_dir():
            return source_candidate

    raise FileNotFoundError(
        f"Could not locate bundled cartridge {directory_name!r}. Reinstall looplet."
    )
