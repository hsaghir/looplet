"""repo_overview — bridges to the closure-built tool from co-located lib.make_tools(REPO_CONFIG.path)."""

from git_detective_lib import make_tools

REPO_CONFIG = None
_REGISTRY = None


def execute(**kwargs):
    global _REGISTRY
    if _REGISTRY is None:
        repo_path = REPO_CONFIG.path if REPO_CONFIG is not None else "."
        _REGISTRY = make_tools(repo_path)
    return _REGISTRY._tools["repo_overview"].execute(**kwargs)
