"""Shared workspace_config — the workspace dir the coder operates in.

setup.py overwrites the ``path`` attribute at load time based on
the ``runtime`` arg (set by the host CLI). Tools and hooks that
need the workspace path read it through the registry at
``"@workspace_config"``.

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


def build():
    return WorkspaceConfig(path=".")
