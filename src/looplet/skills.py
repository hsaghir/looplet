"""Skills — composable bundles of tools + context + instructions.

A skill is the unit of agent capability. Instead of building one
monolithic agent with 40 tools and a massive prompt, build small
skills and compose them::

    python_skill = Skill(
        name="python",
        tools=[ToolSpec(name="bash", ...), ToolSpec(name="read", ...)],
        instructions="You are a Python developer. Run tests before done().",
        memory=StaticMemorySource("PEP 8 rules..."),
    )

    # Load into a loop
    python_skill.register(registry)
    config = LoopConfig(
        system_prompt=python_skill.instructions,
        memory_sources=[python_skill.memory],
    )

Skills are discoverable — the agent can list available skills and
load them on demand via a hook or tool.

This follows Anthropic's "build skills, not agents" pattern from
Code Summit 2025: composable skill modules, not monolithic loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from looplet.tools import BaseToolRegistry, ToolSpec

__all__ = ["Skill"]


@dataclass
class Skill:
    """A composable bundle of tools + context + instructions.

    Args:
        name: Short identifier (e.g. "python", "security", "data").
        tools: List of ToolSpec instances this skill provides.
        instructions: System prompt fragment — appended to the
            agent's system prompt when this skill is active.
        memory: Optional PersistentMemorySource that carries
            domain knowledge surviving all compactions.
        description: One-line description for discovery/listing.

    Usage::

        skill = Skill(
            name="python",
            tools=[bash_spec, read_spec, write_spec],
            instructions="Write tests first. Use type hints.",
        )
        skill.register(my_registry)
    """

    name: str
    tools: list[ToolSpec] = field(default_factory=list)
    instructions: str = ""
    memory: Any = None  # PersistentMemorySource or None
    description: str = ""

    def register(self, registry: BaseToolRegistry) -> int:
        """Register all tools from this skill into a registry.

        Returns the number of tools registered.
        """
        for spec in self.tools:
            registry.register(spec)
        return len(self.tools)

    def tool_names(self) -> list[str]:
        """Return the names of all tools in this skill."""
        return [t.name for t in self.tools]

    def as_catalog_entry(self) -> str:
        """One-line summary for skill discovery listings."""
        tools = ", ".join(self.tool_names()) or "(no tools)"
        desc = self.description or self.instructions[:80]
        return f"[{self.name}] {desc} — tools: {tools}"
