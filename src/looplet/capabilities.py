"""Host-owned execution capabilities for tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = ["ExecutionPolicy", "CapabilityDeniedError"]


class CapabilityDeniedError(PermissionError):
    """A tool requested authority that the host did not grant."""

    def __init__(self, message: str, *, capability: str | None = None) -> None:
        self.capability = capability
        super().__init__(message)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-owned authority available to one agent run.

    An empty policy preserves legacy behavior for undeclared tools. Once a
    tool declares capabilities, every requested capability must be granted.
    ``workspace_root`` is contextual metadata and does not sandbox arbitrary
    Python; tools should use :meth:`resolve_path` for workspace-aware paths.
    """

    workspace_root: str | None = None
    scratch_root: str | None = None
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
                "PATH_DENIED: absolute paths are unavailable; use the documented "
                "case-local scratch path instead",
                capability="host.absolute_path",
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
            raise CapabilityDeniedError(
                "PATH_DENIED: path escapes workspace root and leaves the case "
                "workspace; use a relative path or the documented case-local "
                "scratch path",
                capability="workspace.path",
            )
        return resolved

    def scratch_path(self, relative: str | Path = ".") -> Path:
        """Resolve a path inside the run's case-local scratch directory."""
        if not self.scratch_root:
            raise CapabilityDeniedError(
                "SCRATCH_UNAVAILABLE: no case-local scratch path was provisioned",
                capability="workspace.scratch",
            )
        scratch = Path(self.scratch_root).expanduser().resolve()
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CapabilityDeniedError(
                "PATH_DENIED: scratch paths must stay below the case-local scratch directory",
                capability="workspace.scratch",
            )
        resolved = (scratch / candidate).resolve()
        try:
            resolved.relative_to(scratch)
        except ValueError as exc:
            raise CapabilityDeniedError(
                "PATH_DENIED: scratch path leaves the case-local scratch directory",
                capability="workspace.scratch",
            ) from exc
        return resolved
