"""agent_factory.cartridge setup.py — auto-scaffolds the target.

When the factory is loaded with these runtime kwargs:

    runtime={
        "project_root": "/path/to/project",  # OPTIONAL; auto-detected via git toplevel / cwd
        "scaffold_to": "summarizer.cartridge",  # OPTIONAL relative path
        "scaffold_name": "summarizer",          # OPTIONAL, defaults to dir name
        "scaffold_tools": ["a", "b"],            # OPTIONAL, list of tool names
    }

…we scaffold the target cartridge skeleton BEFORE the loop runs, so
the agent starts with the boilerplate already laid out and spends LLM
turns on the interesting work (tool bodies, system prompt, tests).

The agent's system prompt mentions: "If the target cartridge already
has a skeleton, customize it via multi_edit. Otherwise, write the
files yourself." So both paths work — host pre-scaffold OR agent
self-scaffolds.

If ``scaffold_to`` is not provided, this is a no-op and the factory
behaves exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from looplet.cartridge.runtime_helpers import resolve_project_root
from looplet.scaffold import scaffold_cartridge


def setup(preset, resources, *, runtime=None, **_kwargs):
    runtime = runtime or {}
    target = runtime.get("scaffold_to")
    tools = runtime.get("scaffold_tools")
    if not target or not tools:
        return preset

    workspace_root = Path(resolve_project_root(runtime))
    target_path = workspace_root / target if not Path(target).is_absolute() else Path(target)
    name = runtime.get("scaffold_name") or target_path.stem.replace(".cartridge", "").replace(
        ".workspace", ""
    )

    scaffold_cartridge(
        target_path,
        name=name,
        tools=list(tools),
        overwrite=True,  # idempotent — _write_if_absent preserves existing files
    )
    return preset
