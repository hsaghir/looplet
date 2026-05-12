"""Hot-reload of workspace presets.

Pi reloads its TypeScript extensions when the underlying files change —
including when the agent edits its own extensions. The pattern is
powerful enough that looplet should support it for workspaces too,
without dragging in ``watchdog`` or any other runtime dependency.

This module ships a tiny mtime-poll watcher: between runs of
:func:`composable_loop`, call :meth:`WorkspaceWatcher.reload_if_changed`
and reuse the new preset if it returns one. No filesystem-watch
threads, no signal handlers, no third-party deps.

Usage::

    from looplet.hot_reload import WorkspaceWatcher

    watcher = WorkspaceWatcher("./my_agent.workspace", runtime={"workspace": "."})
    preset = watcher.preset()         # initial load

    while True:
        for step in composable_loop(
            llm=my_llm,
            tools=preset.tools,
            state=preset.state,
            config=preset.config,
            hooks=preset.hooks,
            task=next_task(),
        ):
            print(step.pretty())

        new = watcher.reload_if_changed()
        if new is not None:
            preset = new            # picked up edits to tools/<x>/execute.py etc.

The watcher keeps the loop body simple and is deliberately *between
runs*, not mid-step. Mid-step hot-reload would require freezing the
event loop, re-binding tools, and reasoning about partial state — a
poor trade for a feature most users only want at the boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from looplet.cartridge import cartridge_to_preset

__all__ = [
    "WorkspaceWatcher",
    "fingerprint_workspace",
]


# Files whose contents can change agent behaviour. Anything outside this
# set is ignored to keep the fingerprint cheap.
_WATCHED_SUFFIXES: tuple[str, ...] = (".py", ".yaml", ".yml", ".md", ".json")


def fingerprint_workspace(root: str | Path) -> dict[str, str]:
    """Return ``{relpath: fingerprint}`` for every watched file under ``root``.

    Each fingerprint is a short string combining ``mtime_ns``, ``size``, and
    a 64-bit BLAKE2b digest of the file contents. The digest is what makes
    the fingerprint correct on filesystems whose ``mtime`` resolution is
    coarse enough to collapse rapid writes (tmpfs, some network mounts).
    Without it, two writes within a few microseconds can yield the same
    ``mtime_ns`` and the watcher silently misses the second edit.

    Returns an empty dict if ``root`` does not exist (so the watcher can
    be constructed before the workspace is generated).
    """
    import hashlib  # noqa: PLC0415

    base = Path(root)
    if not base.is_dir():
        return {}
    fp: dict[str, str] = {}
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _WATCHED_SUFFIXES:
            continue
        # Skip __pycache__/dot dirs to avoid noise.
        rel = p.relative_to(base)
        parts = rel.parts
        if any(part.startswith("__pycache__") or part.startswith(".") for part in parts):
            continue
        try:
            stat = p.stat()
            data = p.read_bytes()
        except OSError:
            continue
        digest = hashlib.blake2b(data, digest_size=8).hexdigest()
        fp[str(rel)] = f"{stat.st_mtime_ns}:{stat.st_size}:{digest}"
    return fp


@dataclass
class WorkspaceWatcher:
    """mtime-poll watcher over a workspace directory.

    Args:
        root: Path to the workspace directory.
        runtime: Forwarded to :func:`cartridge_to_preset` on every
            (re)load. Same shape as the regular workspace runtime dict.
        strict: Forwarded to :func:`cartridge_to_preset`.
    """

    root: str | Path
    runtime: dict[str, Any] | None = None
    strict: bool = False
    _fp: dict[str, str] = field(default_factory=dict, init=False)
    _preset: Any = field(default=None, init=False)

    def preset(self) -> Any:
        """Return the current preset, loading it on first call."""
        if self._preset is None:
            self._preset = cartridge_to_preset(self.root, strict=self.strict, runtime=self.runtime)
            self._fp = fingerprint_workspace(self.root)
        return self._preset

    def changed(self) -> bool:
        """True if the workspace fingerprint has shifted since last load."""
        if self._preset is None:
            return True
        return fingerprint_workspace(self.root) != self._fp

    def reload_if_changed(self) -> Any | None:
        """Return a fresh preset when the workspace changed, else None.

        Side effect: updates the cached preset and fingerprint when a
        reload occurs.
        """
        if not self.changed():
            return None
        self._preset = cartridge_to_preset(self.root, strict=self.strict, runtime=self.runtime)
        self._fp = fingerprint_workspace(self.root)
        return self._preset

    def force_reload(self) -> Any:
        """Reload regardless of fingerprint (for tests / debug)."""
        self._preset = cartridge_to_preset(self.root, strict=self.strict, runtime=self.runtime)
        self._fp = fingerprint_workspace(self.root)
        return self._preset
