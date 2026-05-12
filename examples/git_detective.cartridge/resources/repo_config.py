"""Shared repo_config — points at the git repository to analyze.

The host calls ``cartridge_to_preset(path, runtime={"workspace": "/path/to/repo"})``
and this builder reads ``runtime["workspace"]``. Every git_detective tool
declares ``requires: [repo_config]`` in its ``tool.yaml`` and reads
this via ``ctx.resources["repo_config"]`` so the same workspace can
be pointed at any repo by changing one runtime kwarg.

For back-compat, the legacy ``runtime["repo"]`` key is still honoured.
"""

from dataclasses import dataclass


@dataclass
class RepoConfig:
    path: str = "."


def build(runtime=None):
    runtime = runtime or {}
    # Prefer the standardised ``workspace`` key; fall back to ``repo``
    # for cartridges built before the convention was unified.
    path = runtime.get("workspace") or runtime.get("repo") or "."
    return RepoConfig(path=str(path))
