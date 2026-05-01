"""Shared repo_config — points at the git repository to analyze.

The host calls ``workspace_to_preset(path, runtime={"repo": "/path/to/repo"})``
and this builder reads ``runtime["repo"]``. Every git_detective tool
reads this via the module global ``REPO_CONFIG`` (set by setup.py
from ``"@repo_config"``) so the same workspace can be pointed at
any repo by changing one runtime kwarg.
"""

from dataclasses import dataclass


@dataclass
class RepoConfig:
    path: str = "."


def build(runtime=None):
    runtime = runtime or {}
    return RepoConfig(path=str(runtime.get("repo", ".")))
