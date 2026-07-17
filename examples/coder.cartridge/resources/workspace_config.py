"""Shared workspace_config - the project directory the coder operates in.

The canonical "where am I" value for every coder tool. Resolved by
:func:`looplet.cartridge.runtime_helpers.resolve_project_root`, which
tries (in order): ``runtime["project_root"]``, ``runtime["workspace"]``
(legacy), ``$LOOPLET_PROJECT_ROOT``, ``git rev-parse --show-toplevel``,
and finally ``cwd``. So a host that runs the agent from inside the
target repo doesn't have to pass any runtime kwargs at all.
"""

from dataclasses import dataclass
from pathlib import Path

from looplet.cartridge.runtime_helpers import resolve_project_root


@dataclass
class WorkspaceConfig:
    path: str = "."

    @property
    def root(self) -> Path:
        return Path(self.path)


def build(runtime=None):
    return WorkspaceConfig(path=resolve_project_root(runtime))
