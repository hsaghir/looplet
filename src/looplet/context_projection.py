"""Immutable prompt-context projections shared by sync and async loops."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

__all__ = ["ContextProjection"]


@dataclass(frozen=True)
class ContextProjection:
    """The complete read-only context presented to a prompt renderer.

    The loop owns the live objects; renderers receive this snapshot so a
    projection cannot mutate execution state or depend on private loop
    fields. ``messages`` is a tuple to make the boundary immutable.
    """

    messages: tuple[Any, ...]
    default_prompt: str
    step_num: int
    task: Any = None
    tool_catalog: str = ""
    state_summary: dict[str, Any] | None = None
    context_history: str = ""
    session_log: str = ""
    briefing: str = ""
    memory: str = ""

    def __post_init__(self) -> None:
        """Detach nested prompt inputs from live loop state."""
        object.__setattr__(self, "messages", tuple(copy.deepcopy(self.messages)))
        object.__setattr__(self, "task", copy.deepcopy(self.task))
        object.__setattr__(self, "state_summary", copy.deepcopy(self.state_summary))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic view without live messages."""
        return {
            "message_count": len(self.messages),
            "default_prompt": self.default_prompt,
            "step_num": self.step_num,
            "task": copy.deepcopy(self.task),
            "tool_catalog": self.tool_catalog,
            "state_summary": copy.deepcopy(self.state_summary),
            "context_history": self.context_history,
            "session_log": self.session_log,
            "briefing": self.briefing,
            "memory": self.memory,
        }
