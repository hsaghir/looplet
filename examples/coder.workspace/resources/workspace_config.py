"""Shared workspace_config — the workspace dir the coder operates in.

The host calls ``workspace_to_preset(path, runtime={"workspace": "/path/to/repo"})``
and this builder reads ``runtime["workspace"]`` to set the path. Tools
and hooks that need the workspace root read it through the @ref
registry as ``"@workspace_config"`` so the same workspace can be
pointed at any repo by changing one runtime kwarg.

Defaults to "." so the workspace can be exercised against the
current working directory in tests / one-off runs.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkspaceConfig:
    path: str = "."

    @property
    def root(self) -> Path:
        return Path(self.path)


def build(runtime=None):
    runtime = runtime or {}
    return WorkspaceConfig(path=str(runtime.get("workspace", ".")))
