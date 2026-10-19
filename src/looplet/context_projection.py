"""Immutable prompt-context projections shared by sync and async loops."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any

from looplet.context_plan import ContextPlan

__all__ = ["ContextProjection"]


def model_turn_context(state: Any) -> dict[str, Any]:
    from looplet.assistant_turns import TURN_STATE_KEY

    metadata = getattr(state, "metadata", {})
    turns = metadata.get(TURN_STATE_KEY, {}) if isinstance(metadata, dict) else {}
    if not turns:
        return {}
    envelope = getattr(state, "run_envelope", None)
    run_id = envelope.run_id if envelope is not None else "run"
    return {
        "model_turn_id": f"{run_id}:turn-{turns['turn_count']}",
        "model_turn_number": turns["turn_count"],
    }


def invoke_context_callback(callback: Any, kwargs: dict[str, Any], state: Any) -> Any:
    extra = model_turn_context(state)
    if extra:
        parameters = inspect.signature(callback).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs = {
            **kwargs,
            **{
                name: value for name, value in extra.items() if accepts_kwargs or name in parameters
            },
        }
    return callback(**kwargs)


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
    context_plan: ContextPlan | None = None
    scoped_context: str = ""
    scoped_context_plan: ContextPlan | None = None
    model_turn_id: str | None = None
    model_turn_number: int | None = None

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
            **(
                {"model_turn_id": self.model_turn_id, "model_turn_number": self.model_turn_number}
                if self.model_turn_id is not None
                else {}
            ),
            "task": copy.deepcopy(self.task),
            "tool_catalog": self.tool_catalog,
            "state_summary": copy.deepcopy(self.state_summary),
            "context_history": self.context_history,
            "session_log": self.session_log,
            "briefing": self.briefing,
            "memory": self.memory,
            "context_plan": self.context_plan.to_dict() if self.context_plan else None,
            "scoped_context": self.scoped_context,
            "scoped_context_plan": (
                self.scoped_context_plan.to_dict() if self.scoped_context_plan else None
            ),
        }
