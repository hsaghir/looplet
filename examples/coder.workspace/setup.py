"""Wire shared resources + non-declarative bits for the coder workspace.

After PR #30 + the harness-search dogfooding the deferred work shrunk
to two jobs that genuinely need real Python at load time:

1. **Inject shared resources into tool module globals** — tools
   accept their kwargs from the LLM, so the @ref registry alone
   can't hand them ``workspace_config`` / ``file_cache``. setup.py
   walks ``tool_modules`` and writes ``WORKSPACE_CONFIG`` /
   ``FILE_CACHE`` into each tool that declares those globals.

2. **Attach the compaction service** — ``compact_service`` is a
   non-JSON-able callable, so it can't go in config.yaml. Same
   chain the v1 cartridge uses.

3. **Append the live-state CallableMemorySource** — the v1
   cartridge's project-context briefing uses a callable that reads
   ``state.step_count`` per step. CallableMemorySource isn't
   round-trippable through YAML.

Every other piece (LinterHook, EvalHook, evaluators, collectors,
PerToolLimitHook, StagnationHook, ThresholdCompactHook, the @ref
shared FileCache) is now declarative under hooks/ and resources/.
"""

from __future__ import annotations


def setup(preset, resources, tool_modules, hook_modules, runtime=None):
    runtime = runtime or {}
    workspace_path = str(runtime.get("workspace", "."))
    file_cache = resources.get("file_cache")
    workspace_config = resources.get("workspace_config")

    # 1. Inject shared resources into tool module globals.
    for module in tool_modules.values():
        if workspace_config is not None and hasattr(module, "WORKSPACE_CONFIG"):
            module.WORKSPACE_CONFIG = workspace_config
        if file_cache is not None and hasattr(module, "FILE_CACHE"):
            module.FILE_CACHE = file_cache

    # 2. Attach the compaction service.
    from looplet.compact import (  # noqa: PLC0415
        PruneToolResults,
        TruncateCompact,
        compact_chain,
    )

    preset.config.compact_service = compact_chain(
        PruneToolResults(keep_recent=10),
        TruncateCompact(keep_recent=5),
    )

    # 3. Append the live-state CallableMemorySource the v1 cartridge
    #    uses for project-context briefing.
    from examples.coder.wiring import build_default_memory_sources  # noqa: PLC0415

    extra_sources = build_default_memory_sources(workspace_path, preset.config.max_steps)
    existing = list(preset.config.memory_sources or [])
    preset.config.memory_sources = existing + extra_sources

    return preset
