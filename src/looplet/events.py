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

from dataclasses import asdict, dataclass, field, fields, is_dataclass
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

    * :attr:`SESSION_START` - once, at the top of ``composable_loop``.
    * :attr:`PRE_LLM_CALL` - per step, after prompt/messages are built,
      before the model is invoked.
    * :attr:`POST_LLM_RESPONSE` - per step, after the raw response
      lands, before it is parsed.
    * :attr:`PRE_TOOL_USE` - per tool call, before dispatch. Hooks
      returning ``HookDecision`` here can rewrite args, deny, or
      short-circuit with a cached result.
    * :attr:`TOOL_PROGRESS` - while a tool is executing, whenever it
      calls ``ctx.report_progress(stage, data)``. Observers only.
    * :attr:`POST_TOOL_USE` - per tool call, after a successful
      dispatch. Hooks can rewrite the result before it hits history.
    * :attr:`POST_TOOL_FAILURE` - per tool call, when dispatch raised
      or returned an error. Runs before retry/recovery decisions.
    * :attr:`PRE_COMPACT` - before any conversation compaction runs.
    * :attr:`POST_COMPACT` - after compaction, with a count of
      messages removed / summary length.
    * :attr:`HOOK_DECISION` - fires whenever a hook returns a
      ``HookDecision`` that is not a no-op; payload carries slot,
      hook_name, and the decision dict.
    * :attr:`DONE_ACCEPTED` - fires after ``check_done`` has accepted a
      ``done()`` call and the final payload is committed; payload
      includes ``tool_call`` (the done call) and ``tool_result`` (the
      dispatched done result).
    * :attr:`STOP` - when the loop is about to exit, for any reason.
      The payload includes ``termination_reason``.
    * :attr:`SUBAGENT_START` / :attr:`SUBAGENT_STOP` - when a forked
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
    # Per-slot optional fields - populated only when the event fires
    # in a context where they make sense. Kept flat to avoid variant
    # juggling at every call site.
    prompt: str | None = None
    raw_response: Any | None = None
    usage: Any | None = None
    tool_call: Any | None = None
    tool_result: Any | None = None
    termination_reason: str | None = None
    messages_before: int | None = None
    messages_after: int | None = None
    subagent_id: str | None = None
    hook_slot: str | None = None
    hook_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        """Best-effort JSON-safe view of this payload's data fields.

        Used by out-of-process observers (e.g. the RPC event stream) that
        forward every lifecycle event over a wire. The serialiser is
        deliberately *small and safe*:

        * The loop-internal object fields ``state``, ``session_log`` and
          ``context`` are always dropped - they are large, often cyclic,
          and never JSON. ``event``/``step_num`` are dropped too because a
          transport surfaces them separately (as ``kind`` / ``step_num``).
        * Every other field is reduced via :func:`_to_jsonable`: scalars
          pass through, enums become their ``value``, objects exposing
          ``to_dict()`` (``ToolCall``/``ToolResult``/``Step``) or plain
          dataclasses are expanded, containers are walked, and anything
          that still cannot be reduced to JSON is **dropped** rather than
          raised on.
        * ``None`` values and the empty ``extra`` dict are omitted to keep
          frames compact.

        This method never raises - a malformed payload must not break the
        loop that emitted it.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            name = f.name
            if name in _NON_SERIALISED_FIELDS:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            if name == "extra" and not value:
                continue
            safe = _to_jsonable(value)
            if safe is not _DROP:
                out[name] = safe
        return out


#: Payload fields never forwarded: ``event``/``step_num`` are surfaced by the
#: transport itself; ``state``/``session_log``/``context`` are large, cyclic,
#: loop-internal objects that are never JSON.
_NON_SERIALISED_FIELDS = frozenset({"event", "step_num", "state", "session_log", "context"})

#: Sentinel returned by :func:`_to_jsonable` for values that cannot be reduced
#: to JSON, so the caller can drop the key entirely (distinct from ``None``,
#: which is a legitimate JSON value a payload field may carry).
_DROP = object()


def _to_jsonable(value: Any) -> Any:
    """Reduce ``value`` to a JSON-serialisable form, or :data:`_DROP`.

    Never raises: any object that cannot be reduced (no ``to_dict``, not a
    dataclass, not a known scalar/container) yields :data:`_DROP` so the
    caller omits it.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            sv = _to_jsonable(v)
            if sv is not _DROP:
                out[str(k)] = sv
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_to_jsonable(v) for v in value]
        return [v for v in items if v is not _DROP]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_jsonable(to_dict())
        except Exception:  # noqa: BLE001 - serialiser must never raise
            return _DROP
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return _to_jsonable(asdict(value))
        except Exception:  # noqa: BLE001 - serialiser must never raise
            return _DROP
    return _DROP
