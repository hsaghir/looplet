"""Shared repo_config — points at the git repository to analyze.

The host calls ``workspace_to_preset(path, runtime={"repo": "/path/to/repo"})``
and this builder reads ``runtime["repo"]``. Every git_detective tool
declares ``requires: [repo_config]`` in its ``tool.yaml`` and reads
this via ``ctx.resources["repo_config"]`` so the same workspace can
be pointed at any repo by changing one runtime kwarg.
"""

from dataclasses import dataclass


@dataclass
class RepoConfig:
    path: str = "."


def build(runtime=None):
    runtime = runtime or {}
    return RepoConfig(path=str(runtime.get("repo", ".")))
