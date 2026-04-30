"""Named lifecycle events for the composable loop.

We ship a curated set of events that map to the real integration
points in a general-purpose agent loop. Each event name is a :class:`LifecycleEvent` enum member; hooks
opting into the new event-style API implement :meth:`on_event` and
switch on the name.

The loop still calls the per-method hook API (``pre_prompt``,
``post_dispatch``, …) for existing hooks. Hooks that implement
``on_event`` additionally receive every lifecycle call in one place,
which is the idiomatic shape for user-authored policy hooks that
care about multiple slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "LifecycleEvent",
    "EventPayload",
    "LIFECYCLE_EVENTS",
]


class LifecycleEvent(str, Enum):
    """The lifecycle events the loop emits.

    Ordered roughly by when they fire in a single step:

    * :attr:`SESSION_START` — once, at the top of ``composable_loop``.
    * :attr:`PRE_LLM_CALL` — per step, after prompt/messages are built,
      before the model is invoked.
    * :attr:`POST_LLM_RESPONSE` — per step, after the raw response
      lands, before it is parsed.
    * :attr:`PRE_TOOL_USE` — per tool call, before dispatch. Hooks
      returning ``HookDecision`` here can rewrite args, deny, or
      short-circuit with a cached result.
    * :attr:`TOOL_PROGRESS` — while a tool is executing, whenever it
      calls ``ctx.report_progress(stage, data)``. Observers only.
    * :attr:`POST_TOOL_USE` — per tool call, after a successful
      dispatch. Hooks can rewrite the result before it hits history.
    * :attr:`POST_TOOL_FAILURE` — per tool call, when dispatch raised
      or returned an error. Runs before retry/recovery decisions.
    * :attr:`PRE_COMPACT` — before any conversation compaction runs.
    * :attr:`POST_COMPACT` — after compaction, with a count of
      messages removed / summary length.
    * :attr:`HOOK_DECISION` — fires whenever a hook returns a
      ``HookDecision`` that is not a no-op; payload carries slot,
      hook_name, and the decision dict.
    * :attr:`DONE_ACCEPTED` — fires after ``check_done`` has accepted a
      ``done()`` call and the final payload is committed; payload
      includes ``tool_call`` (the done call) and ``tool_result`` (the
      dispatched done result).
    * :attr:`STOP` — when the loop is about to exit, for any reason.
      The payload includes ``termination_reason``.
    * :attr:`SUBAGENT_START` / :attr:`SUBAGENT_STOP` — when a forked
      sub-agent loop begins and ends. Only fires if subagents are in
      use.
    """

    SESSION_START = "session_start"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_RESPONSE = "post_llm_response"
    PRE_TOOL_USE = "pre_tool_use"
    TOOL_PROGRESS = "tool_progress"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_FAILURE = "post_tool_failure"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    HOOK_DECISION = "hook_decision"
    DONE_ACCEPTED = "done_accepted"
    STOP = "stop"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"


LIFECYCLE_EVENTS = tuple(e.value for e in LifecycleEvent)


@dataclass
class EventPayload:
    """Structured payload passed to :meth:`LoopHook.on_event`.

    The ``event`` field is always present; everything else is slot-
    specific and populated only when meaningful for the current event.
    Hooks should treat unset fields as "not applicable" rather than
    inspecting them.
    """

    event: LifecycleEvent
    step_num: int = 0
    state: Any = None
    session_log: Any = None
    context: Any = None
    # Per-slot optional fields — populated only when the event fires
    # in a context where they make sense. Kept flat to avoid variant
    # juggling at every call site.
    prompt: str | None = None
    raw_response: Any | None = None
    tool_call: Any | None = None
    tool_result: Any | None = None
    termination_reason: str | None = None
    messages_before: int | None = None
    messages_after: int | None = None
    subagent_id: str | None = None
    hook_slot: str | None = None
    hook_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
