"""Wire shared repo_config into every git tool module so the
closure-built tool registry uses the right repository path. Also
attach compact_service to match the v1 cartridge."""


def setup(preset, resources, tool_modules, hook_modules):
    from looplet.compact import (  # noqa: PLC0415
        PruneToolResults,
        TruncateCompact,
        compact_chain,
    )

    repo_config = resources.get("repo_config")
    if repo_config is not None:
        for name, module in tool_modules.items():
            if hasattr(module, "REPO_CONFIG"):
                module.REPO_CONFIG = repo_config

    preset.config.compact_service = compact_chain(
        PruneToolResults(keep_recent=6),
        TruncateCompact(keep_recent=3),
    )
    return preset
