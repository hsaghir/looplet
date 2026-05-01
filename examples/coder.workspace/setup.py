"""Wire shared resources + the one non-declarative bit for coder.workspace.

After the @ref-driven memory_sources support landed, the only jobs
left for setup.py are:

1. **Inject shared resources into tool module globals** — tools
   accept their kwargs from the LLM, so the @ref registry alone
   can't hand them ``workspace_config`` / ``file_cache``. setup.py
   walks ``tool_modules`` and writes ``WORKSPACE_CONFIG`` /
   ``FILE_CACHE`` into each tool that declares those globals.

2. **Attach the compaction service** — ``compact_service`` is a
   non-JSON-able callable; the same @ref machinery as above could
   route it but the v1 cartridge keeps the chain hard-coded so we
   match that.

Project-context memory and the test-runner collector now live in
``resources/project_memory.py`` and ``resources/eval_collectors.py``;
both are wired declaratively through ``config.yaml`` and the
``EvalHook`` config respectively.
"""

from __future__ import annotations


def setup(preset, resources, tool_modules, hook_modules, runtime=None):
    runtime = runtime or {}
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

    return preset
