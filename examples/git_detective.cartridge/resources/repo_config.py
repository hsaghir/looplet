"""Shared repo_config - points at the git repository to analyze.

Resolved via :func:`looplet.cartridge.runtime_helpers.resolve_project_root`,
so a host running the agent from inside the target repo doesn't need
to pass any runtime kwargs. Hosts that want to point at a different
repo can pass ``runtime={"project_root": "/path/to/other/repo"}``
(or set ``$LOOPLET_PROJECT_ROOT``).

Legacy ``runtime["workspace"]`` and ``runtime["repo"]`` keys are
still honoured for back-compat.
"""

from dataclasses import dataclass

from looplet.cartridge.runtime_helpers import resolve_project_root


@dataclass
class RepoConfig:
    path: str = "."


def build(runtime=None):
    runtime = runtime or {}
    # ``repo`` is the legacy git_detective-specific key; honour it
    # before falling through to the standard resolution chain.
    if runtime.get("repo"):
        return RepoConfig(path=str(runtime["repo"]))
    return RepoConfig(path=resolve_project_root(runtime))
