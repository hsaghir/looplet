"""StaticMemorySource carrying any project-discovered coding
instructions (CLAUDE.md / AGENTS.md / etc.). Returns ``None`` when
nothing is found so the loader skips the entry.

Reads ``runtime['workspace']`` for the project root.
"""

from __future__ import annotations

from pathlib import Path

from looplet import StaticMemorySource

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
    runtime = runtime or {}
    workspace = str(runtime.get("workspace", "."))
    text = _discover(workspace)
    if not text:
        # Returning ``None`` is honoured by the loader's
        # ``render_memory`` path: ``CallableMemorySource``-style
        # ``None`` returns are silently skipped.
        return StaticMemorySource(text="")
    return StaticMemorySource(text=text)
