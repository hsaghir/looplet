"""StaticMemorySource carrying any project-discovered coding
instructions (CLAUDE.md / AGENTS.md / etc.). Returns ``None`` when
nothing is found so the loader skips the entry.

Resolves the project root via
:func:`looplet.cartridge.runtime_helpers.resolve_project_root`.
"""

from __future__ import annotations

from pathlib import Path

from looplet import StaticMemorySource
from looplet.cartridge.runtime_helpers import resolve_project_root

_INSTRUCTION_FILES = (
    "CLAUDE.md",
    ".claude.md",
    "AGENTS.md",
    ".cursorrules",
    "CODING_GUIDELINES.md",
    ".github/copilot-instructions.md",
)


def _discover(workspace: str) -> str:
    parts: list[str] = []
    for name in _INSTRUCTION_FILES:
        p = Path(workspace) / name
        if p.exists():
            parts.append(f"## From {name}\n{p.read_text()[:4000]}")
    return "\n\n".join(parts)


def build(runtime=None):
    workspace = resolve_project_root(runtime)
    text = _discover(workspace)
    if not text:
        # Returning ``None`` is honoured by the loader's
        # ``render_memory`` path: ``CallableMemorySource``-style
        # ``None`` returns are silently skipped.
        return StaticMemorySource(text="")
    return StaticMemorySource(text=text)
