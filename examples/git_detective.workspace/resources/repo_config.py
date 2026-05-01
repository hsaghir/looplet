"""Shared repo_config — points at the git repository to analyze.

setup.py replaces ``path`` at workspace load time using the host
CLI's ``--repo`` arg. Every git_detective tool reads this via the
module global ``REPO_CONFIG`` (set by setup.py from
``"@repo_config"``) so the same workspace can be pointed at any
repo by changing one line in setup.py.
"""

from dataclasses import dataclass


@dataclass
class RepoConfig:
    path: str = "."


def build():
    return RepoConfig(path=".")
