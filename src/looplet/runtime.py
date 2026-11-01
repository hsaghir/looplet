"""Optional host runtime for lifecycle-safe Looplet runs."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import threading
import time
from collections.abc import Iterator, MutableSequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterable
from uuid import uuid4

from looplet._storage import DiskBackedList
from looplet.checkpoint import Checkpoint, _capture_checkpoint
from looplet.events import EventPayload
from looplet.execution_session import ExecutionSession
from looplet.presets import AgentPreset, ShutdownReport
from looplet.run_records import RunEvent, event_from_payload
from looplet.run_store import RunStore
from looplet.types import CancelToken, RunEnvelope, RunPhase, RunResult, RunStatus

__all__ = ["AgentRuntime", "RunHandle"]


def _failed_runtime_result(
    state: Any,
    envelope: RunEnvelope,
    exc: BaseException,
) -> RunResult:
    state_metadata = getattr(state, "metadata", {})
    metadata = dict(state_metadata) if isinstance(state_metadata, dict) else {}
    stats = getattr(state, "_looplet_run_stats", None)
    if isinstance(stats, dict):
        metadata["looplet_run_stats"] = dict(stats)
    metadata["error"] = f"{type(exc).__name__}: {exc}"
    history = getattr(state, "steps", ())
    return RunResult(
        status=RunStatus.FAILED,
        phase=RunPhase.TERMINAL,
        termination_reason="error",
        output=None,
        steps=history.snapshot() if isinstance(history, DiskBackedList) else tuple(history),
        run_envelope=envelope,
        metadata=metadata,
    )


class _RuntimeEventHook:
    def __init__(
        self,
        envelope: RunEnvelope,
        events: MutableSequence[RunEvent],
        store: RunStore | None,
        persistence_errors: list[str],
    ) -> None:
        self.envelope = envelope
        self.events = events
        self.store = store
        self.persistence_errors = persistence_errors
        self._sequence = 0

    def on_event(self, payload: EventPayload) -> None:
        event = event_from_payload(
            payload,
            envelope=self.envelope,
            sequence=self._sequence,
        )
        self._sequence += 1
        self.events.append(event)
        if self.store is not None:
            try:
                self.store.append_event(event)
            except Exception as exc:  # noqa: BLE001 - persistence is best effort
                self.persistence_errors.append(f"append_event: {type(exc).__name__}: {exc}")


class _RunStoreCheckpointHook:
    checkpoint_owner = True

    def __init__(
        self,
        store: RunStore,
        run_id: str,
        tools: Any,
        llm: Any,
        every: int,
        persistence_errors: list[str],
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.tools = tools
        self.llm = llm
        self.every = every
        self.persistence_errors = persistence_errors
        self._loop_ctx: Any = None
        self._terminal_saved = False
        self.checkpoint_store = self
        self._checkpoint_mirror: Any = None
        self._pending_checkpoint: Checkpoint | None = None
        self._pending_model_call: dict[str, Any] | None = None
        self._latest_checkpoint: Checkpoint | None = None
        self._checkpoint_state: Any = None
        self._async_persistence = False

    def bind_checkpoint_store(self, store: Any, checkpoint_state: Any) -> None:
        self._checkpoint_mirror = store
        self._checkpoint_state = checkpoint_state

    def enable_async_persistence(self) -> None:
        self._async_persistence = True

    def capture_state(self, loop_ctx: Any) -> dict[str, Any]:
        domain_state = dict(self._checkpoint_state(loop_ctx) if self._checkpoint_state else {})
        llm_checkpoint = getattr(self.llm, "checkpoint_state", None)
        if "llm" not in domain_state and callable(llm_checkpoint):
            try:
                domain_state["llm"] = llm_checkpoint()
            except Exception as exc:
                self.persistence_errors.append(f"checkpoint: {type(exc).__name__}: {exc}")
        return copy.deepcopy(domain_state)

    def save(self, checkpoint: Checkpoint, key: str) -> None:
        from looplet.assistant_turns import TURN_STATE_KEY

        pending = self._pending_checkpoint
        acknowledged = key.startswith("step_") and (
            key.endswith("_done") or key == f"step_{checkpoint.step_number}"
        )
        turns = checkpoint.metadata.get(TURN_STATE_KEY, {})
        if turns.get("pending_model_call"):
            self._pending_model_call = copy.deepcopy(turns["pending_model_call"])
        elif (
            key == "model_turn"
            or acknowledged
            or checkpoint.metadata.get("pending_calls")
            or checkpoint.run_status == "cancelled"
        ):
            self._pending_model_call = None
        elif self._pending_model_call is not None:
            checkpoint = replace(
                checkpoint,
                metadata={
                    **checkpoint.metadata,
                    TURN_STATE_KEY: {**turns, "pending_model_call": self._pending_model_call},
                },
            )
        if checkpoint.metadata.get("pending_calls"):
            self._pending_checkpoint = checkpoint
        elif acknowledged:
            self._pending_checkpoint = None
        elif pending is not None:
            checkpoint = replace(
                checkpoint,
                metadata={
                    **checkpoint.metadata,
                    "pending_calls": pending.metadata["pending_calls"],
                },
            )
        self._latest_checkpoint = checkpoint
        if (
            checkpoint.metadata.get("pending_calls")
            or self._pending_model_call is not None
            or (acknowledged and pending is not None)
            or key == "model_turn"
            or checkpoint.is_terminal
            or checkpoint.step_number % self.every == 0
        ):
            try:
                self.store.save_checkpoint(self.run_id, checkpoint)
            except Exception as exc:
                self.persistence_errors.append(f"checkpoint: {type(exc).__name__}: {exc}")
        if self._checkpoint_mirror is not None:
            self._checkpoint_mirror.save(checkpoint, key)

    def load(self, key: str, *, run_id: str | None = None) -> Checkpoint | None:
        if self._checkpoint_mirror is not None:
            return self._checkpoint_mirror.load(key, run_id=run_id)
        if run_id is not None and run_id != self.run_id:
            from looplet.checkpoint import CheckpointIdentityError

            raise CheckpointIdentityError("checkpoint belongs to a different runtime run")
        return self.store.load_checkpoint(self.run_id, key)

    def bind(self, loop_ctx: Any) -> None:
        self._loop_ctx = loop_ctx

    def on_loop_end(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        llm: Any,
    ) -> Any:
        del context, llm
        steps = getattr(state, "steps", ())
        step_num = getattr(steps[-1], "number", 0) if steps else 0
        self._terminal_saved = True
        return self._save(state, session_log, step_num)

    def finalize(self, result: RunResult) -> Any:
        """Persist a terminal snapshot when the loop raised before cleanup."""
        if self._terminal_saved or self._loop_ctx is None:
            return
        state = self._loop_ctx.state
        session_log = self._loop_ctx.session_log
        for name, value in (
            ("run_status", result.status.value),
            ("run_phase", result.phase.value),
            ("termination_reason", result.termination_reason),
        ):
            try:
                setattr(state, name, value)
            except AttributeError:
                pass
        metadata = getattr(state, "metadata", None)
        if isinstance(metadata, dict):
            metadata.update({"run_status": result.status.value, "run_phase": result.phase.value})
            if result.termination_reason is not None:
                metadata["termination_reason"] = result.termination_reason
        self._terminal_saved = True
        return self._save(state, session_log, getattr(self._loop_ctx, "step_num", 0))

    async def finalize_async(self, result: RunResult) -> None:
        pending = self.finalize(result)
        if pending is not None:
            await pending

    def _save(self, state: Any, session_log: Any, step_num: int) -> Any:
        try:
            from looplet.assistant_turns import assistant_checkpoint_metadata

            config_snapshot = None
            if self._latest_checkpoint is not None:
                step_num = max(step_num, self._latest_checkpoint.step_number)
                config_snapshot = {
                    **self._latest_checkpoint.config_snapshot,
                    "queries_used": getattr(state, "queries_used", 0),
                    "budget_remaining": getattr(state, "budget_remaining", 0),
                }
            checkpoint = _capture_checkpoint(
                self._loop_ctx,
                step_number=step_num,
                checkpoint_state=self.capture_state,
                config_snapshot=config_snapshot,
                metadata={
                    **dict(getattr(state, "metadata", {}) or {}),
                    **assistant_checkpoint_metadata(state, checkpoint_store=self),
                },
            )
            checkpoint = replace(
                checkpoint,
                run_status=str(getattr(state, "run_status", "running")),
                run_phase=str(getattr(state, "run_phase", "dispatching")),
                termination_reason=getattr(state, "termination_reason", None),
            )
            if self._async_persistence:
                from looplet.checkpoint import _run_persistence

                return _run_persistence(self.save, checkpoint, key=f"step_{step_num}_terminal")
            self.save(checkpoint, key=f"step_{step_num}_terminal")
        except Exception as exc:  # noqa: BLE001 - persistence is best effort
            self.persistence_errors.append(f"checkpoint: {type(exc).__name__}: {exc}")


class RunHandle:
    """Handle for a background synchronous run."""

    def __init__(
        self,
        *,
        run_id: str,
        cancel_token: CancelToken,
        future: concurrent.futures.Future[RunResult],
        events: MutableSequence[RunEvent],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        self.run_id = run_id
        self._cancel_token = cancel_token
        self._future = future
        self._events = events
        self._executor = executor

    @property
    def status(self) -> RunStatus:
        if not self._future.done():
            return RunStatus.RUNNING
        return self.result().status

    @property
    def phase(self) -> RunPhase:
        if not self._future.done():
            return RunPhase.LLM
        return self.result().phase

    def cancel(self, reason: str = "cancelled") -> None:
        del reason
        self._cancel_token.cancel()

    def result(self, timeout: float | None = None) -> RunResult:
        try:
            return self._future.result(timeout=timeout)
        finally:
            if self._future.done():
                self._executor.shutdown(wait=False, cancel_futures=True)

    async def wait(self) -> RunResult:
        try:
            return await asyncio.wrap_future(self._future)
        finally:
            if self._future.done():
                self._executor.shutdown(wait=False, cancel_futures=True)

    def events(self) -> tuple[RunEvent, ...]:
        return tuple(self._events)


@dataclass
class _RuntimeExecution:
    hooks: list[Any]
    result: RunResult | None = None
    checkpoint_hook: _RunStoreCheckpointHook | None = None


class AgentRuntime:
    """Lifecycle wrapper around an existing :class:`AgentPreset`.

    Direct ``composable_loop`` usage remains available. This wrapper adds
    host-owned run identity, event capture, optional persistence, cancellation,
    and exception-safe preset cleanup without changing the loop kernel.
    """

    def __init__(
        self,
        preset: AgentPreset,
        *,
        store: RunStore | None = None,
        session: ExecutionSession | None = None,
        checkpoint_every_n_steps: int | None = None,
    ) -> None:
        self.preset = preset
        self.store = store
        self.session = session
        if checkpoint_every_n_steps is not None and checkpoint_every_n_steps < 1:
            raise ValueError("checkpoint_every_n_steps must be positive")
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self._closed = False
        self._started = False
        self._lock = threading.RLock()
        self._active_tokens: dict[str, CancelToken] = {}
        self._active_handles: dict[str, RunHandle] = {}

    @contextmanager
    def _execution(
        self,
        llm: Any,
        envelope: RunEnvelope,
        token: CancelToken,
        *,
        events: MutableSequence[RunEvent] | None = None,
        claimed: bool = False,
    ) -> Iterator[_RuntimeExecution]:
        """Own setup, persistence, restoration, and release for either driver."""
        if not claimed:
            self._claim(envelope.run_id, token)
        try:
            with self._lock:
                self._ensure_open()
            persistence_errors: list[str] = []
            observer = _RuntimeEventHook(
                envelope,
                events
                if events is not None
                else DiskBackedList(RunEvent.to_dict, RunEvent.from_dict),
                self.store,
                persistence_errors,
            )
            checkpoint_hook = (
                _RunStoreCheckpointHook(
                    self.store,
                    envelope.run_id,
                    self.preset.tools,
                    llm,
                    self.checkpoint_every_n_steps,
                    persistence_errors,
                )
                if self.store is not None and self.checkpoint_every_n_steps is not None
                else None
            )
            execution = _RuntimeExecution(
                [observer, *([checkpoint_hook] if checkpoint_hook is not None else [])],
                checkpoint_hook=checkpoint_hook,
            )
            old_envelope = self.preset.config.run_envelope
            old_token = self.preset.config.cancel_token
            self.preset.config.run_envelope = envelope
            self.preset.config.cancel_token = token
            started = time.perf_counter()
            try:
                if self.store is not None:
                    try:
                        self.store.create(envelope, metadata={"runtime": "AgentRuntime"})
                    except Exception as exc:
                        persistence_errors.append(f"create: {type(exc).__name__}: {exc}")
                yield execution
            finally:
                try:
                    if checkpoint_hook is not None and execution.result is not None:
                        checkpoint_hook.finalize(execution.result)
                finally:
                    with self._lock:
                        self.preset.config.run_envelope = old_envelope
                        self.preset.config.cancel_token = old_token
                if execution.result is not None:
                    result = execution.result
                    result.metadata.setdefault(
                        "runtime_duration_ms", (time.perf_counter() - started) * 1000
                    )
                    if persistence_errors:
                        result.metadata["persistence_warnings"] = persistence_errors
                    if self.store is not None:
                        try:
                            self.store.complete(envelope.run_id, result)
                        except Exception as exc:
                            result.metadata.setdefault("persistence_warnings", []).append(
                                f"complete: {type(exc).__name__}: {exc}"
                            )
        finally:
            self._release(envelope.run_id)

    def run(
        self,
        llm: Any,
        *,
        task: Any = None,
        envelope: RunEnvelope | None = None,
        extra_hooks: Iterable[Any] = (),
        _cancel_token: CancelToken | None = None,
        _event_buffer: MutableSequence[RunEvent] | None = None,
        _claimed: bool = False,
    ) -> RunResult:
        """Run synchronously and always return a host-facing result."""
        run_envelope = envelope or self._new_envelope()
        cancel_token = _cancel_token or self.preset.config.cancel_token or CancelToken()
        with self._execution(
            llm, run_envelope, cancel_token, events=_event_buffer, claimed=_claimed
        ) as execution:
            try:
                if self.session is not None:
                    self.session.attach(run_envelope.run_id)
                execution.result = self.preset.run_result(
                    llm,
                    task=task,
                    extra_hooks=[*extra_hooks, *execution.hooks],
                )
            except Exception as exc:  # noqa: BLE001 - host boundary returns a failed result
                execution.result = _failed_runtime_result(self.preset.state, run_envelope, exc)
            return execution.result

    async def run_async(
        self,
        llm: Any,
        *,
        task: Any = None,
        envelope: RunEnvelope | None = None,
        extra_hooks: Iterable[Any] = (),
    ) -> RunResult:
        """Run with the async loop while sharing the same lifecycle contract."""
        run_envelope = envelope or self._new_envelope()
        cancel_token = self.preset.config.cancel_token or CancelToken()
        with self._execution(llm, run_envelope, cancel_token) as execution:
            try:
                if self.session is not None:
                    self.session.attach(run_envelope.run_id)
                execution.result = await self.preset.run_result_async(
                    llm,
                    task=task,
                    extra_hooks=[*extra_hooks, *execution.hooks],
                )
            except asyncio.CancelledError:
                execution.result = replace(
                    RunResult.from_state(self.preset.state), run_envelope=run_envelope
                )
                raise
            except Exception as exc:  # noqa: BLE001
                execution.result = _failed_runtime_result(self.preset.state, run_envelope, exc)
            finally:
                if execution.checkpoint_hook is not None and execution.result is not None:
                    await execution.checkpoint_hook.finalize_async(execution.result)
            assert execution.result is not None
            return execution.result

    def start(
        self, llm: Any, *, task: Any = None, envelope: RunEnvelope | None = None
    ) -> RunHandle:
        """Start a background synchronous run."""
        run_envelope = envelope or self._new_envelope()
        token = CancelToken()
        self._claim(run_envelope.run_id, token)
        events: MutableSequence[RunEvent] = DiskBackedList(RunEvent.to_dict, RunEvent.from_dict)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="looplet-run"
        )
        try:
            future = executor.submit(
                self.run,
                llm,
                task=task,
                envelope=run_envelope,
                extra_hooks=(),
                _cancel_token=token,
                _event_buffer=events,
                _claimed=True,
            )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            self._release(run_envelope.run_id)
            raise
        handle = RunHandle(
            run_id=run_envelope.run_id,
            cancel_token=token,
            future=future,
            events=events,
            executor=executor,
        )
        with self._lock:
            self._active_handles[run_envelope.run_id] = handle
        future.add_done_callback(lambda _: self._active_handles.pop(run_envelope.run_id, None))
        return handle

    def close(self, *, timeout: float = 5.0) -> ShutdownReport:
        with self._lock:
            self._closed = True
            tokens = list(self._active_tokens.values())
            handles = list(self._active_handles.values())
        for token in tokens:
            token.cancel()
        deadline = time.monotonic() + max(0.0, timeout)
        for handle in handles:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                handle.result(timeout=remaining)
            except (TimeoutError, concurrent.futures.TimeoutError):
                pass
            except Exception:
                pass
        with self._lock:
            if self._active_tokens:
                self._closed = False
                return ShutdownReport(errors=("active runs did not stop before runtime close",))
        report = self.preset.close()
        if self.session is not None:
            self.session.close()
        return report

    def __enter__(self) -> "AgentRuntime":
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _new_envelope(self) -> RunEnvelope:
        return RunEnvelope(run_id=uuid4().hex[:12])

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentRuntime is closed")

    def _claim(self, run_id: str, token: CancelToken) -> None:
        with self._lock:
            self._ensure_open()
            if self._started:
                raise RuntimeError(
                    "AgentRuntime is single-use; create a new runtime for another run"
                )
            if self._active_tokens:
                raise RuntimeError(
                    "AgentRuntime supports one active run; create a separate runtime for concurrency"
                )
            self._active_tokens[run_id] = token
            self._started = True

    def _release(self, run_id: str) -> None:
        with self._lock:
            self._active_tokens.pop(run_id, None)
