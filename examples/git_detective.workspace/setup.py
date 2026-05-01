"""Wire shared repo_config into every git tool module so the
closure-built tool registry uses the right repository path."""


def setup(preset, resources, tool_modules, hook_modules):
    repo_config = resources.get("repo_config")
    if repo_config is None:
        return preset
    for name, module in tool_modules.items():
        if hasattr(module, "REPO_CONFIG"):
            module.REPO_CONFIG = repo_config
    return preset
