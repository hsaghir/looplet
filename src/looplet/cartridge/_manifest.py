"""Cartridge dataclass + manifest helpers.

* :class:`Cartridge` — the in-memory representation of a loaded
  cartridge directory. Carries name, version, schema_version,
  metadata, plus the path it was loaded from. Returned by
  :meth:`Cartridge.from_directory`; consumed by
  :func:`looplet.cartridge.preset_to_cartridge` (as the structured
  target it writes into).
* :func:`_manifest_path` and :func:`_manifest_present` — small
  helpers that probe a directory for either ``cartridge.json`` or
  the historical ``workspace.json`` manifest filename. Prefers
  ``cartridge.json`` when both exist.

`Cartridge.to_preset` calls :func:`cartridge_to_preset` lazily to
avoid a circular dep at module-load time (the loader imports this
module to get the dataclass type).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from looplet.presets import AgentPreset

from looplet.cartridge._layout import (
    SCHEMA_VERSION,
    CartridgeLayout,
)

# ── Data class ──────────────────────────────────────────────────


def _manifest_path(root: Path) -> Path | None:
    """Return the path to the cartridge manifest file, or ``None``.

    Accepts both ``cartridge.json`` (Cartridge Spec v1.0 canonical name)
    and ``workspace.json`` (historical alias). Prefers
    ``cartridge.json`` if both exist.
    """
    primary = root / CartridgeLayout.CARTRIDGE_JSON
    if primary.is_file():
        return primary
    legacy = root / CartridgeLayout.WORKSPACE_JSON
    if legacy.is_file():
        return legacy
    return None


def _manifest_present(root: Path) -> bool:
    return _manifest_path(root) is not None


@dataclass
class Cartridge:
    """A loaded Workspace.

    Serves both as the in-memory representation of an on-disk workspace
    and as the structured target of :func:`preset_to_cartridge`.
    """

    path: Path
    name: str = ""
    description: str = ""
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    serialization_warnings: list[str] = field(default_factory=list)

    # ── classmethod builders ───────────────────────────────────

    @classmethod
    def from_directory(cls, path: str | Path) -> "Cartridge":
        """Load workspace metadata from a workspace directory.

        Use :func:`cartridge_to_preset` to materialise the
        :class:`AgentPreset` from the loaded workspace.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"workspace directory not found: {root}")
        meta_path = _manifest_path(root)
        if meta_path is None:
            raise FileNotFoundError(
                f"cartridge metadata not found at "
                f"{root / CartridgeLayout.CARTRIDGE_JSON} "
                f"(or {CartridgeLayout.WORKSPACE_JSON}); is this a Cartridge directory?"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            path=root,
            name=str(meta.get("name", root.name)),
            description=str(meta.get("description", "")),
            schema_version=int(meta.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(meta.get("metadata", {})),
        )

    # ── instance API ───────────────────────────────────────────

    def write_metadata(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / CartridgeLayout.CARTRIDGE_JSON).write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "name": self.name,
                    "description": self.description,
                    "metadata": dict(self.metadata),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def to_preset(self) -> "AgentPreset":
        """Materialise the :class:`AgentPreset` described by this workspace."""
        # Lazy import to break the circular dep: the loader imports
        # this module to know about the Cartridge dataclass type.
        from looplet.cartridge import cartridge_to_preset  # noqa: PLC0415

        return cartridge_to_preset(self.path)
