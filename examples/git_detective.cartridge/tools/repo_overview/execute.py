"""repo_overview — bridges to the closure-built tool from
co-located ``git_detective_lib.make_tools(repo_config.path)``.

Receives the shared ``repo_config`` resource through
``ctx.resources``; ``tool.yaml`` declares
``requires: [repo_config]``. The closure-built registry is cached
per (module, repo_path) tuple so 8 tools share one registry per
repo without rebuilding on every call.
"""

from git_detective_lib import make_tools

from looplet.types import ToolContext

_REGISTRY_CACHE: dict = {}


def execute(ctx: ToolContext, **kwargs):
    cfg = ctx.resources.get("repo_config")
    repo_path = cfg.path if cfg is not None else "."
    registry = _REGISTRY_CACHE.get(repo_path)
    if registry is None:
        registry = make_tools(repo_path)
        _REGISTRY_CACHE[repo_path] = registry
    return registry._tools["repo_overview"].execute(**kwargs)
