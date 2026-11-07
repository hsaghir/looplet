"""Structured event emission for real-time observability of agent execution.

Events are dataclasses emitted at key points during a loop run.  Consumers
attach one or more ``EventEmitter`` implementations - callbacks, queues, or
composites - and receive a typed stream of events without coupling to loop
internals.

Typical usage::

    received = []
    hook = StreamingHook(CallbackEmitter(received.append))
    loop_result = composable_loop(state, ..., hooks=[hook])
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from looplet.events import LifecycleEvent
from looplet.session import SessionLog
from looplet.types import CancelToken, ErrorKind, ToolCall, ToolContext, ToolError, ToolResult

if TYPE_CHECKING:
    from looplet.tools import BaseToolRegistry
    from looplet.types import AgentState, LLMBackend


__all__ = [
    "Event",
    "LoopStartEvent",
    "StepStartEvent",
    "LLMCallStartEvent",
    "LLMCallEndEvent",
    "AssistantTurnDecisionEvent",
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
    "StreamingToolExecutor",
    "AsyncStreamingToolExecutor",
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
    model_turn_id: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class LLMCallEndEvent(Event):
    """Emitted immediately after each LLM call returns."""

    step_num: int = 0
    response_length: int = 0
    duration_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    model_turn_id: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__
        usage = self.usage if isinstance(self.usage, dict) else {}
        self.usage = {
            name: value
            for name in ("input", "output", "cache_read", "cache_write")
            if type(value := usage.get(name)) is int and value >= 0
        }


@dataclass
class AssistantTurnDecisionEvent(Event):
    """Metadata-only progress for a host-controlled assistant message."""

    step_num: int = 0
    model_turn_id: str = ""
    action: str = ""
    model_calls: int = 0

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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class ToolResultEvent(Event):
    """Emitted after a tool execution completes."""

    step_num: int = 0
    tool_name: str = ""
    duration_ms: float = 0.0
    has_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = type(self).__name__


@dataclass
class StepEndEvent(Event):
    """Emitted at the end of each step after post_dispatch hooks run."""

    step_num: int = 0
    classification: str = ""
    new_entities_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

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
    ``"blocking"`` - matching ``ContextPressureHook``'s 4-tier budget.

    Consumers typically:

    * ``ok`` - clear any "context almost full" UI indicator.
    * ``warning`` - display a soft notice to the user.
    * ``compact`` - the loop will compact automatically; a debug UI can
      surface the action.
    * ``blocking`` - emergency compaction was forced; expect to see a
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
        self._llm_started_at: dict[str | int, float] = {}

    def to_config(self) -> dict:
        """Cartridge round-trip: emit ``emitter`` as an ``@ref``.

        When ``self._emitter`` was produced by a workspace resource
        builder, the original ref name is preserved (e.g.
        ``resources/event_collector.py`` round-trips as
        ``{"emitter": "@event_collector"}``). Otherwise the writer
        emits the generic ``"@emitter"`` name. Closure-based emitters
        whose origin can't be traced fall through to a None-stub the
        user must replace.
        """
        from looplet.refs import resource_ref_for  # noqa: PLC0415

        ref = resource_ref_for(self._emitter)
        return {"emitter": ref or "@emitter"}

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
        ``StreamingHook`` in ``hooks``, the event will fire twice - prefer
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
        from looplet.context_projection import model_turn_context

        if not model_turn_context(state):
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
        from looplet.context_projection import model_turn_context
        from looplet.tools import _summarize_args_dict  # noqa: PLC0415

        if model_turn_context(state):
            self._emitter.emit(StepStartEvent(step_num=step_num))
        self._emitter.emit(
            ToolDispatchEvent(
                step_num=step_num,
                tool_name=tool_call.tool,
                args_summary=_summarize_args_dict(tool_call.args),
            )
        )
        return None

    def check_permission(self, tool_call: ToolCall, state: AgentState) -> bool:
        """StreamingHook never blocks - always returns True."""
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
                total_llm_calls=getattr(state, "_looplet_run_stats", {}).get(
                    "llm_calls", self._total_llm_calls
                ),
                reason=reason,
            )
        )
        return 0

    def on_event(self, payload: Any) -> None:
        """Emit metadata-only LLM call events from lifecycle callbacks."""
        if payload.event == LifecycleEvent.PRE_LLM_CALL:
            key = payload.model_turn_id or payload.step_num
            self._llm_started_at[key] = time.perf_counter()
            budget = payload.context_budget if isinstance(payload.context_budget, dict) else {}
            estimate = budget.get("estimated_tokens")
            self._emitter.emit(
                LLMCallStartEvent(
                    step_num=payload.step_num,
                    model_turn_id=payload.model_turn_id,
                    prompt_tokens_est=estimate if type(estimate) is int and estimate >= 0 else 0,
                )
            )
        elif payload.event == LifecycleEvent.POST_LLM_RESPONSE:
            key = payload.model_turn_id or payload.step_num
            started_at = self._llm_started_at.pop(key, None)
            self._emitter.emit(
                LLMCallEndEvent(
                    step_num=payload.step_num,
                    model_turn_id=payload.model_turn_id,
                    response_length=len(payload.raw_response)
                    if isinstance(payload.raw_response, str)
                    else 0,
                    duration_ms=(time.perf_counter() - started_at) * 1000
                    if started_at is not None
                    else 0.0,
                    usage=payload.usage if isinstance(payload.usage, dict) else {},
                )
            )
        elif payload.event == LifecycleEvent.ASSISTANT_TURN_DECISION:
            self._emitter.emit(
                AssistantTurnDecisionEvent(
                    step_num=payload.step_num,
                    model_turn_id=payload.model_turn_id or "",
                    action=payload.extra.get("action", ""),
                    model_calls=payload.extra.get("model_calls", 0),
                )
            )
        elif payload.event == LifecycleEvent.DONE_ACCEPTED and payload.model_turn_id is not None:
            self._emitter.emit(StepStartEvent(step_num=payload.step_num))
        return None


# ── Streaming Tool Executors ────────────────────────────────────


@dataclass
class _TrackedToolItem:
    index: int
    call: ToolCall
    ctx: ToolContext | None
    is_concurrent_safe: bool
    status: str = "queued"
    result: ToolResult | None = None
    future: Future[ToolResult] | None = None
    task: asyncio.Task[ToolResult] | None = None


def _cancelled_tool_result(call: ToolCall, message: str) -> ToolResult:
    from looplet.tools import _summarize_args_dict  # noqa: PLC0415

    clean = {k: v for k, v in call.args.items() if not k.startswith("__")}
    err = ToolError(kind=ErrorKind.CANCELLED, message=message, retriable=False)
    return ToolResult(
        tool=call.tool,
        args_summary=_summarize_args_dict(clean),
        data=None,
        error=err.message,
        error_detail=err,
        call_id=call.call_id,
        metadata={"execution_status": "not_executed"},
    )


def _bind_sibling_token(ctx: ToolContext | None, sibling_token: CancelToken) -> ToolContext:
    if ctx is None:
        return ToolContext(cancel_token=sibling_token)
    if ctx.cancel_token is None:
        return replace(ctx, cancel_token=sibling_token)
    return replace(ctx, cancel_token=ctx.cancel_token.create_child(sibling_token))


class StreamingToolExecutor:
    """Execute tool calls as they stream in with concurrency control.

    - Concurrent-safe tools (via ``spec.is_concurrent_safe(call.args)``) can
      execute in parallel with other concurrent-safe tools immediately upon
      submission.
    - Non-concurrent-safe tools wait for active work and execute exclusively.
    - Results are buffered and returned in exact submission order by
      :meth:`results`.
        - :meth:`discard` cancels pending work and signals running tools.
            :meth:`results` waits for started workers to finish.
    """

    def __init__(
        self,
        tools: BaseToolRegistry,
        *,
        ctx: ToolContext | None = None,
        max_workers: int = 10,
        cancel_on_sibling_error: bool = False,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._tools = tools
        self._default_ctx = ctx
        self._max_workers = max_workers
        self._cancel_on_sibling_error = cancel_on_sibling_error
        parent_token = ctx.cancel_token if ctx is not None else None
        self._sibling_token = (
            parent_token.create_child() if parent_token is not None else CancelToken()
        )
        self._items: list[_TrackedToolItem] = []
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._discarded = False
        self._has_errored = False
        self._errored_tool = ""

    def submit(self, call: ToolCall, *, ctx: ToolContext | None = None) -> None:
        """Submit a tool call to the streaming execution queue."""
        with self._lock:
            effective_ctx = ctx if ctx is not None else self._default_ctx
            effective_ctx = _bind_sibling_token(effective_ctx, self._sibling_token)
            spec = self._tools._tools.get(call.tool)
            is_safe = spec.is_concurrent_safe(call.args) if spec is not None else False
            item = _TrackedToolItem(
                index=len(self._items),
                call=call,
                ctx=effective_ctx,
                is_concurrent_safe=is_safe,
            )
            self._items.append(item)
            self._process_queue_locked()

    def discard(self) -> None:
        """Discard all pending and in-progress tools (e.g. on streaming fallback)."""
        with self._lock:
            self._discarded = True
            self._sibling_token.cancel()
            for item in self._items:
                if item.status != "completed":
                    item.result = _cancelled_tool_result(
                        item.call, "Streaming fallback - tool execution discarded"
                    )
                    item.status = "completed"
                if item.future is not None:
                    item.future.cancel()
            self._cond.notify_all()

    def _can_execute_locked(self, is_concurrent_safe: bool) -> bool:
        executing = [t for t in self._items if t.status == "executing"]
        if len(executing) >= self._max_workers:
            return False
        return len(executing) == 0 or (
            is_concurrent_safe and all(t.is_concurrent_safe for t in executing)
        )

    def _process_queue_locked(self) -> None:
        for item in self._items:
            if item.status != "queued":
                if item.status == "executing" and not item.is_concurrent_safe:
                    break
                continue
            if self._discarded:
                item.result = _cancelled_tool_result(
                    item.call, "Streaming fallback - tool execution discarded"
                )
                item.status = "completed"
                continue
            if self._cancel_on_sibling_error and self._has_errored:
                item.result = _cancelled_tool_result(
                    item.call,
                    f"Cancelled: parallel tool call {self._errored_tool!r} errored",
                )
                item.status = "completed"
                continue
            if self._can_execute_locked(item.is_concurrent_safe):
                item.status = "executing"
                item.future = self._pool.submit(self._execute_item, item)
                if not item.is_concurrent_safe:
                    break
            else:
                break

    def _execute_item(self, item: _TrackedToolItem) -> ToolResult:
        try:
            res = self._tools.dispatch(item.call, ctx=item.ctx)
        except Exception as exc:  # pragma: no cover - defensive
            res = _cancelled_tool_result(item.call, str(exc))
        with self._lock:
            if self._discarded:
                item.result = _cancelled_tool_result(
                    item.call, "Streaming fallback - tool execution discarded"
                )
            else:
                item.result = res
                if (
                    res.error is not None
                    and self._cancel_on_sibling_error
                    and not self._has_errored
                ):
                    self._has_errored = True
                    self._errored_tool = item.call.tool
                    self._sibling_token.cancel()
            item.status = "completed"
            self._process_queue_locked()
            self._cond.notify_all()
        return res

    def results(self, timeout: float | None = None) -> list[ToolResult]:
        """Wait for all submitted tools and return results in submission order."""
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._cond:
            while any(item.status != "completed" for item in self._items):
                remaining = (deadline - time.monotonic()) if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("StreamingToolExecutor timed out waiting for tool results")
                self._cond.wait(timeout=remaining)
            out = [item.result for item in self._items if item.result is not None]
        remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else None
        _, pending = wait(
            (item.future for item in self._items if item.future is not None),
            timeout=remaining,
        )
        if pending:
            raise TimeoutError("StreamingToolExecutor timed out waiting for tool results")
        self._pool.shutdown(wait=True)
        return out


class AsyncStreamingToolExecutor:
    """Async counterpart to :class:`StreamingToolExecutor`."""

    def __init__(
        self,
        tools: BaseToolRegistry,
        *,
        ctx: ToolContext | None = None,
        max_concurrent: int = 10,
        cancel_on_sibling_error: bool = False,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._tools = tools
        self._default_ctx = ctx
        self._max_concurrent = max_concurrent
        self._cancel_on_sibling_error = cancel_on_sibling_error
        parent_token = ctx.cancel_token if ctx is not None else None
        self._sibling_token = (
            parent_token.create_child() if parent_token is not None else CancelToken()
        )
        self._items: list[_TrackedToolItem] = []
        self._done_event = asyncio.Event()
        self._done_event.set()
        self._discarded = False
        self._has_errored = False
        self._errored_tool = ""

    def submit(self, call: ToolCall, *, ctx: ToolContext | None = None) -> None:
        """Submit a tool call to the async streaming execution queue."""
        effective_ctx = ctx if ctx is not None else self._default_ctx
        effective_ctx = _bind_sibling_token(effective_ctx, self._sibling_token)
        spec = self._tools._tools.get(call.tool)
        is_safe = spec.is_concurrent_safe(call.args) if spec is not None else False
        item = _TrackedToolItem(
            index=len(self._items),
            call=call,
            ctx=effective_ctx,
            is_concurrent_safe=is_safe,
        )
        self._items.append(item)
        self._done_event.clear()
        self._process_queue()

    def discard(self) -> None:
        """Discard all pending and in-progress tools."""
        self._discarded = True
        self._sibling_token.cancel()
        for item in self._items:
            if item.status != "completed":
                item.result = _cancelled_tool_result(
                    item.call, "Streaming fallback - tool execution discarded"
                )
                item.status = "completed"
            if item.task is not None and not item.task.done():
                item.task.cancel()
        self._done_event.set()

    def _can_execute(self, is_concurrent_safe: bool) -> bool:
        executing = [t for t in self._items if t.status == "executing"]
        if len(executing) >= self._max_concurrent:
            return False
        return len(executing) == 0 or (
            is_concurrent_safe and all(t.is_concurrent_safe for t in executing)
        )

    def _process_queue(self) -> None:
        for item in self._items:
            if item.status != "queued":
                if item.status == "executing" and not item.is_concurrent_safe:
                    break
                continue
            if self._discarded:
                item.result = _cancelled_tool_result(
                    item.call, "Streaming fallback - tool execution discarded"
                )
                item.status = "completed"
                continue
            if self._cancel_on_sibling_error and self._has_errored:
                item.result = _cancelled_tool_result(
                    item.call,
                    f"Cancelled: parallel tool call {self._errored_tool!r} errored",
                )
                item.status = "completed"
                continue
            if self._can_execute(item.is_concurrent_safe):
                item.status = "executing"
                item.task = asyncio.create_task(self._execute_item(item))
                if not item.is_concurrent_safe:
                    break
            else:
                break
        if all(item.status == "completed" for item in self._items):
            self._done_event.set()

    async def _execute_item(self, item: _TrackedToolItem) -> ToolResult:
        try:
            res = await self._tools.async_dispatch(item.call, ctx=item.ctx)
        except asyncio.CancelledError:
            res = _cancelled_tool_result(
                item.call,
                "Streaming fallback - tool execution discarded"
                if self._discarded
                else "Tool execution cancelled",
            )
        except Exception as exc:  # pragma: no cover - defensive
            res = _cancelled_tool_result(item.call, str(exc))
        if self._discarded:
            item.result = _cancelled_tool_result(
                item.call, "Streaming fallback - tool execution discarded"
            )
        else:
            item.result = res
            if res.error is not None and self._cancel_on_sibling_error and not self._has_errored:
                self._has_errored = True
                self._errored_tool = item.call.tool
                self._sibling_token.cancel()
        item.status = "completed"
        self._process_queue()
        return res

    async def results(self) -> list[ToolResult]:
        """Await all submitted tools and return results in submission order."""
        await self._done_event.wait()
        await asyncio.gather(
            *(item.task for item in self._items if item.task is not None),
            return_exceptions=True,
        )
        return [item.result for item in self._items if item.result is not None]
