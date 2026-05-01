"""Wire shared resources + callable-graph hooks for the coder workspace.

Three jobs run here, all driven by the host-supplied ``runtime`` dict:

1. **Inject shared resources into tool module globals** — tools
   accept their kwargs from the LLM, so the @ref registry alone
   can't hand them ``workspace_config`` / ``file_cache``. setup.py
   walks ``tool_modules`` and writes ``WORKSPACE_CONFIG`` /
   ``FILE_CACHE`` into each tool that declares those globals.

2. **Append callable-graph hooks** — LinterHook (needs runtime
   workspace), EvalHook (needs evaluator + collector callables),
   and an optional StreamingHook (needs an emitter callable) all
   carry callables that don't round-trip through YAML. They get
   appended to ``preset.hooks`` here so the resulting preset
   matches the v1 cartridge feature-for-feature.

3. **Attach the compaction service** — the v1 cartridge wires
   ``compact_service=compact_chain(PruneToolResults(keep_recent=10),
   TruncateCompact(keep_recent=5))``. We re-use the same chain.

Plus a callable memory source for the live project-context briefing
that gets appended to ``config.memory_sources``.
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

    # 2. Append callable-graph hooks. EvalHook needs collector +
    # evaluator callables that don't round-trip through YAML — those
    # stay in setup.py. LinterHook is loaded declaratively from
    # hooks/06_LinterHook/config.yaml using ``${runtime.workspace}``
    # template substitution (no setup.py wiring needed).
    from examples.coder.wiring import build_eval_hook  # noqa: PLC0415

    preset.hooks.append(build_eval_hook(workspace_path))

    # 3. Attach the compaction service.
    from looplet.compact import (  # noqa: PLC0415
        PruneToolResults,
        TruncateCompact,
        compact_chain,
    )

    preset.config.compact_service = compact_chain(
        PruneToolResults(keep_recent=10),
        TruncateCompact(keep_recent=5),
    )

    # 4. Append the live-state callable memory source the v1 cartridge
    #    uses for project-context briefing.
    from examples.coder.wiring import build_default_memory_sources  # noqa: PLC0415

    extra_sources = build_default_memory_sources(workspace_path, preset.config.max_steps)
    existing = list(preset.config.memory_sources or [])
    preset.config.memory_sources = existing + extra_sources

    return preset
