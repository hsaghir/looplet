"""Structured event emission for real-time observability of agent execution.

Events are dataclasses emitted at key points during a loop run.  Consumers
attach one or more ``EventEmitter`` implementations — callbacks, queues, or
composites — and receive a typed stream of events without coupling to loop
internals.

Typical usage::

    received = []
    hook = StreamingHook(CallbackEmitter(received.append))
    loop_result = composable_loop(state, ..., hooks=[hook])
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from looplet.session import SessionLog
from looplet.types import ToolCall, ToolResult

if TYPE_CHECKING:
    from looplet.types import AgentState, LLMBackend


__all__ = [
    "Event",
    "LoopStartEvent",
    "StepStartEvent",
    "LLMCallStartEvent",
    "LLMCallEndEvent",
    "LLMChunkEvent",
    "ToolDispatchEvent",
    "ToolResultEvent",
    "StepEndEvent",
    "LoopEndEvent",
    "HookEvent",
    "RecoveryEvent",
    "ContextPressureEvent",
    "EventEmitter",
    "CallbackEmitter",
    "CompositeEmitter",
    "QueueEmitter",
    "StreamingHook",
]

# ── Base Event ──────────────────────────────────────────────────


@dataclass
class Event:
    """Base class for all streaming events.

    Subclasses set ``event_type`` automatically via ``__post_init__``.
    Consumers can use ``isinstance`` checks or match on ``event_type`` for
    routing.
    """

    event_type: str = field(default="")
    timestamp: float = field(default_factory=time.time)


# ── Concrete Event Types ────────────────────────────────────────


@dataclass
class LoopStartEvent(Event):
    """Emitted once when the loop begins."""

    task_summary: str = ""
    max_steps: int = 0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class StepStartEvent(Event):
    """Emitted at the start of each step before the LLM prompt is built."""

    step_num: int = 0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class LLMCallStartEvent(Event):
    """Emitted immediately before each LLM call."""

    step_num: int = 0
    prompt_tokens_est: int = 0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class LLMCallEndEvent(Event):
    """Emitted immediately after each LLM call returns."""

    step_num: int = 0
    response_length: int = 0
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class LLMChunkEvent(Event):
    """Emitted for each text chunk during token-level LLM streaming.

    Only emitted when a streaming LLM backend (one with a ``stream()``
    method) is used together with a ``stream`` emitter on the loop.
    """

    step_num: int = 0
    chunk: str = ""
    chunk_index: int = 0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class ToolDispatchEvent(Event):
    """Emitted before a tool is dispatched."""

    step_num: int = 0
    tool_name: str = ""
    args_summary: str = ""

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class ToolResultEvent(Event):
    """Emitted after a tool execution completes."""

    step_num: int = 0
    tool_name: str = ""
    duration_ms: float = 0.0
    has_error: bool = False

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class StepEndEvent(Event):
    """Emitted at the end of each step after post_dispatch hooks run."""

    step_num: int = 0
    classification: str = ""
    new_entities_count: int = 0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class LoopEndEvent(Event):
    """Emitted once after the loop exits for any reason."""

    total_steps: int = 0
    total_llm_calls: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class HookEvent(Event):
    """Emitted by hooks to surface internal messages for observability."""

    hook_name: str = ""
    method: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class RecoveryEvent(Event):
    """Emitted when a recovery strategy is attempted (e.g. parse retry)."""

    strategy: str = ""
    success: bool = False

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class ContextPressureEvent(Event):
    """Emitted when the estimated context usage crosses a tier threshold.

    ``level`` is one of ``"ok"``, ``"warning"``, ``"compact"``,
    ``"blocking"`` — matching ``ContextPressureHook``'s 4-tier budget.

    Consumers typically:

    * ``ok`` — clear any "context almost full" UI indicator.
    * ``warning`` — display a soft notice to the user.
    * ``compact`` — the loop will compact automatically; a debug UI can
      surface the action.
    * ``blocking`` — emergency compaction was forced; expect to see a
      ``compaction_boundary`` message shortly after.
    """

    level: str = "ok"
    estimated_tokens: int = 0
    threshold: int = 0
    context_window: int = 0
    percent_used: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


# ── EventEmitter Protocol ───────────────────────────────────────


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for event consumers.

    Any object with an ``emit(event)`` method satisfies this protocol.
    """

    def emit(self, event: Event) -> None:
        """Deliver an event to this consumer."""
        ...


# ── Emitter Implementations ─────────────────────────────────────


class CallbackEmitter:
    """Calls a user-provided callable with each event.

    Args:
        callback: Callable that receives each emitted ``Event``.
    """

    def __init__(self, callback: Callable[[Event], None]) -> None:
        self._callback = callback

    def emit(self, event: Event) -> None:
        self._callback(event)


class QueueEmitter:
    """Puts events on a ``queue.Queue`` for consumer threads.

    Args:
        q: The queue to put events into.
    """

    def __init__(self, q: "queue.Queue[Event]") -> None:
        self._queue = q

    def emit(self, event: Event) -> None:
        self._queue.put(event)


class CompositeEmitter:
    """Fans out each event to multiple child emitters.

    Args:
        emitters: Sequence of ``EventEmitter`` instances to fan out to.
    """

    def __init__(self, emitters: list[EventEmitter]) -> None:
        self._emitters = list(emitters)

    def emit(self, event: Event) -> None:
        for emitter in self._emitters:
            emitter.emit(event)


# ── StreamingHook ───────────────────────────────────────────────


class StreamingHook:
    """``LoopHook`` implementation that emits structured events.

    Attach a ``StreamingHook`` to any ``composable_loop`` run to receive
    a typed event stream without modifying loop internals.

    Args:
        emitter: The ``EventEmitter`` that will receive all events.
    """

    def __init__(self, emitter: EventEmitter) -> None:
        self._emitter = emitter
        self._total_llm_calls: int = 0
        self._step_llm_calls: int = 0

    def to_config(self) -> dict:
        """Workspace round-trip: emit ``emitter`` as an ``@ref`` so the
        workspace writer auto-generates ``resources/emitter.py``.
        Closure-based emitters (e.g. ``CallbackEmitter(list.append)``)
        will fall through to a None-stub the user must replace; classes
        with a clean ``__init__`` round-trip automatically.
        """
        return {"emitter": "@emitter"}

    @property
    def emitter(self) -> EventEmitter:
        """Public accessor used by the workspace writer to pull the
        live instance for resource auto-emit."""
        return self._emitter

    def pre_loop(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
    ) -> None:
        """Emit ``LoopStartEvent`` once before the loop begins.

        Reads ``max_steps`` from ``state`` (``DefaultState`` carries it).
        When using the loop's ``stream=`` parameter *and* a
        ``StreamingHook`` in ``hooks``, the event will fire twice — prefer
        one mechanism, not both.
        """
        task_summary = getattr(state, "task_summary", "")
        max_steps = getattr(state, "max_steps", 0)
        self._emitter.emit(LoopStartEvent(task_summary=task_summary, max_steps=max_steps))

    def pre_prompt(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Emit ``StepStartEvent`` before each LLM prompt and count the LLM call."""
        self._total_llm_calls += 1
        self._emitter.emit(StepStartEvent(step_num=step_num))
        return None

    def pre_dispatch(
        self,
        state: AgentState,
        session_log: SessionLog,
        tool_call: ToolCall,
        step_num: int,
    ) -> ToolResult | None:
        """Emit ``ToolDispatchEvent``; never intercepts execution."""
        from looplet.tools import _summarize_args_dict  # noqa: PLC0415

        self._emitter.emit(
            ToolDispatchEvent(
                step_num=step_num,
                tool_name=tool_call.tool,
                args_summary=_summarize_args_dict(tool_call.args),
            )
        )
        return None

    def check_permission(self, tool_call: ToolCall, state: AgentState) -> bool:
        """StreamingHook never blocks — always returns True."""
        return True

    def post_dispatch(
        self,
        state: AgentState,
        session_log: SessionLog,
        tool_call: ToolCall,
        tool_result: ToolResult,
        step_num: int,
    ) -> str | None:
        """Emit ``ToolResultEvent`` and ``StepEndEvent`` after each dispatch."""
        self._emitter.emit(
            ToolResultEvent(
                step_num=step_num,
                tool_name=tool_result.tool,
                duration_ms=tool_result.duration_ms,
                has_error=tool_result.error is not None,
            )
        )
        self._emitter.emit(
            StepEndEvent(
                step_num=step_num,
                classification="continue",
                new_entities_count=0,
            )
        )
        return None

    def check_done(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Never blocks done; returns None."""
        return None

    def should_stop(
        self,
        state: AgentState,
        step_num: int,
        new_entities: int,
    ) -> bool:
        """Never stops early; returns False."""
        return False

    def should_compact(
        self,
        state: AgentState,
        session_log: SessionLog,
        conversation: Any,
        step_num: int,
    ) -> bool:
        """Never proactively compacts; returns False."""
        return False

    def build_briefing(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
    ) -> str | None:
        """Pass-through; returns ``None`` to defer to config/default."""
        return None

    def build_prompt(self, **kwargs: Any) -> str | None:
        """Pass-through; returns ``None`` to defer to config/default."""
        return None

    def on_loop_end(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        llm: LLMBackend,
    ) -> int:
        """Emit ``LoopEndEvent`` and return 0 extra LLM calls."""
        total_steps = getattr(state, "step_count", 0)
        reason = getattr(state, "_stop_reason", "completed")
        self._emitter.emit(
            LoopEndEvent(
                total_steps=total_steps,
                total_llm_calls=self._total_llm_calls,
                reason=reason,
            )
        )
        return 0

    def on_event(self, payload: Any) -> None:
        """No-op — :class:`StreamingHook` uses the per-method API."""
        return None
