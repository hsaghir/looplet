"""Host-owned execution capabilities for tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = ["ExecutionPolicy", "CapabilityDeniedError"]


class CapabilityDeniedError(PermissionError):
    """A tool requested authority that the host did not grant."""


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-owned authority available to one agent run.

    An empty policy preserves legacy behavior for undeclared tools. Once a
    tool declares capabilities, every requested capability must be granted.
    ``workspace_root`` is contextual metadata and does not sandbox arbitrary
    Python; tools should use :meth:`resolve_path` for workspace-aware paths.
    """

    workspace_root: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    environment: frozenset[str] = field(default_factory=frozenset)

    def allows(self, required: Iterable[str]) -> bool:
        """Return whether all declared capabilities are granted."""
        return set(required).issubset(self.capabilities)

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        """Return missing capabilities in stable order."""
        return tuple(sorted(set(required) - self.capabilities))

    def allows_environment(self, name: str) -> bool:
        """Return whether a tool may access the named environment variable."""
        return name in self.environment

    def resolve_path(self, path: str | Path, *, allow_absolute: bool = False) -> Path:
        """Resolve a path against the workspace, rejecting escapes."""
        candidate = Path(path).expanduser()
        absolute_allowed = allow_absolute or "host.absolute_path" in self.capabilities
        if candidate.is_absolute() and not absolute_allowed:
            raise CapabilityDeniedError(
                "absolute paths require the 'host.absolute_path' capability"
            )
        root = Path(self.workspace_root).expanduser().resolve() if self.workspace_root else None
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
            if root
            else candidate.resolve()
        )
        if (
            root is not None
            and not absolute_allowed
            and root not in resolved.parents
            and resolved != root
        ):
            raise CapabilityDeniedError(f"path escapes workspace root: {path!r}")
        return resolved
