"""Optional host runtime for lifecycle-safe Looplet runs."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any, Iterable
from uuid import uuid4

from looplet.checkpoint import Checkpoint
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
    return RunResult(
        status=RunStatus.FAILED,
        phase=RunPhase.TERMINAL,
        termination_reason="error",
        output=None,
        steps=tuple(getattr(state, "steps", ())),
        run_envelope=envelope,
        metadata={"error": f"{type(exc).__name__}: {exc}"},
    )


class _RuntimeEventHook:
    def __init__(
        self,
        envelope: RunEnvelope,
        events: list[RunEvent],
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

    def bind(self, loop_ctx: Any) -> None:
        self._loop_ctx = loop_ctx

    def post_step(self, state: Any, session_log: Any, step_num: int) -> None:
        if step_num % self.every == 0:
            self._save(state, session_log, step_num)

    def on_loop_end(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        llm: Any,
    ) -> int:
        del context, llm
        steps = getattr(state, "steps", ())
        step_num = getattr(steps[-1], "number", 0) if steps else 0
        self._save(state, session_log, step_num)
        self._terminal_saved = True
        return 0

    def finalize(self, result: RunResult) -> None:
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
        self._save(state, session_log, getattr(self._loop_ctx, "step_num", 0))
        self._terminal_saved = True

    def _save(self, state: Any, session_log: Any, step_num: int) -> None:
        try:
            conversation = getattr(state, "conversation", None)
            run_envelope = getattr(state, "run_envelope", None)
            domain_state: dict[str, Any] = {}
            llm_checkpoint = getattr(self.llm, "checkpoint_state", None)
            if callable(llm_checkpoint):
                domain_state["llm"] = llm_checkpoint()
            checkpoint = Checkpoint(
                step_number=step_num,
                session_log_data={
                    "entries": session_log.to_list() if hasattr(session_log, "to_list") else [],
                    "current_theory": getattr(session_log, "current_theory", ""),
                },
                conversation_data=conversation.serialize() if conversation is not None else None,
                config_snapshot={
                    "max_steps": getattr(state, "max_steps", None),
                    "queries_used": getattr(state, "queries_used", 0),
                    "budget_remaining": getattr(state, "budget_remaining", None),
                },
                tool_results_store=self.tools.snapshot_results(),
                domain_state=domain_state,
                metadata=dict(getattr(state, "metadata", {}) or {}),
                run_status=str(getattr(state, "run_status", "running")),
                run_phase=str(getattr(state, "run_phase", "dispatching")),
                termination_reason=getattr(state, "termination_reason", None),
                run_envelope=run_envelope.to_dict() if run_envelope is not None else None,
            )
            self.store.save_checkpoint(self.run_id, checkpoint)
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
        events: list[RunEvent],
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

    def run(
        self,
        llm: Any,
        *,
        task: Any = None,
        envelope: RunEnvelope | None = None,
        extra_hooks: Iterable[Any] = (),
        _cancel_token: CancelToken | None = None,
        _event_buffer: list[RunEvent] | None = None,
        _claimed: bool = False,
    ) -> RunResult:
        """Run synchronously and always return a host-facing result."""
        run_envelope = envelope or self._new_envelope()
        cancel_token = _cancel_token or self.preset.config.cancel_token or CancelToken()
        if not _claimed:
            self._claim(run_envelope.run_id, cancel_token)
        try:
            with self._lock:
                self._ensure_open()
            events = _event_buffer if _event_buffer is not None else []
            persistence_errors: list[str] = []
            observer = _RuntimeEventHook(run_envelope, events, self.store, persistence_errors)
            checkpoint_hook = (
                _RunStoreCheckpointHook(
                    self.store,
                    run_envelope.run_id,
                    self.preset.tools,
                    llm,
                    self.checkpoint_every_n_steps,
                    persistence_errors,
                )
                if self.store is not None and self.checkpoint_every_n_steps is not None
                else None
            )
            old_envelope = self.preset.config.run_envelope
            old_cancel_token = self.preset.config.cancel_token
            self.preset.config.run_envelope = run_envelope
            self.preset.config.cancel_token = cancel_token
            if self.store is not None:
                try:
                    self.store.create(run_envelope, metadata={"runtime": "AgentRuntime"})
                except Exception as exc:  # noqa: BLE001
                    persistence_errors.append(f"create: {type(exc).__name__}: {exc}")
            started = time.perf_counter()
            result: RunResult | None = None
            try:
                if self.session is not None:
                    self.session.attach(run_envelope.run_id)
                for _ in self.preset.run(
                    llm,
                    task=task,
                    extra_hooks=[
                        *extra_hooks,
                        observer,
                        *([checkpoint_hook] if checkpoint_hook is not None else []),
                    ],
                ):
                    pass
                result = RunResult.from_state(self.preset.state)
            except Exception as exc:  # noqa: BLE001 - host boundary returns a failed result
                result = _failed_runtime_result(self.preset.state, run_envelope, exc)
            finally:
                if checkpoint_hook is not None and result is not None:
                    checkpoint_hook.finalize(result)
                self.preset.config.run_envelope = old_envelope
                self.preset.config.cancel_token = old_cancel_token
            assert result is not None
            result.metadata.setdefault(
                "runtime_duration_ms", (time.perf_counter() - started) * 1000
            )
            if persistence_errors:
                result.metadata["persistence_warnings"] = persistence_errors
            if self.store is not None:
                try:
                    self.store.complete(run_envelope.run_id, result)
                except Exception as exc:  # noqa: BLE001
                    result.metadata.setdefault("persistence_warnings", []).append(
                        f"complete: {type(exc).__name__}: {exc}"
                    )
            return result
        finally:
            self._release(run_envelope.run_id)

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
        self._claim(run_envelope.run_id, cancel_token)
        try:
            with self._lock:
                self._ensure_open()
            events: list[RunEvent] = []
            persistence_errors: list[str] = []
            observer = _RuntimeEventHook(run_envelope, events, self.store, persistence_errors)
            checkpoint_hook = (
                _RunStoreCheckpointHook(
                    self.store,
                    run_envelope.run_id,
                    self.preset.tools,
                    llm,
                    self.checkpoint_every_n_steps,
                    persistence_errors,
                )
                if self.store is not None and self.checkpoint_every_n_steps is not None
                else None
            )
            old_envelope = self.preset.config.run_envelope
            old_cancel_token = self.preset.config.cancel_token
            self.preset.config.run_envelope = run_envelope
            self.preset.config.cancel_token = cancel_token
            if self.store is not None:
                try:
                    self.store.create(run_envelope, metadata={"runtime": "AgentRuntime"})
                except Exception as exc:  # noqa: BLE001
                    persistence_errors.append(f"create: {type(exc).__name__}: {exc}")
            started = time.perf_counter()
            result: RunResult | None = None
            try:
                if self.session is not None:
                    self.session.attach(run_envelope.run_id)
                async for _ in self.preset.run_async(
                    llm,
                    task=task,
                    extra_hooks=[
                        *extra_hooks,
                        observer,
                        *([checkpoint_hook] if checkpoint_hook is not None else []),
                    ],
                ):
                    pass
                result = RunResult.from_state(self.preset.state)
            except Exception as exc:  # noqa: BLE001
                result = _failed_runtime_result(self.preset.state, run_envelope, exc)
            finally:
                if checkpoint_hook is not None and result is not None:
                    checkpoint_hook.finalize(result)
                with self._lock:
                    self.preset.config.run_envelope = old_envelope
                    self.preset.config.cancel_token = old_cancel_token
            assert result is not None
            result.metadata.setdefault(
                "runtime_duration_ms", (time.perf_counter() - started) * 1000
            )
            if persistence_errors:
                result.metadata["persistence_warnings"] = persistence_errors
            if self.store is not None:
                try:
                    self.store.complete(run_envelope.run_id, result)
                except Exception as exc:  # noqa: BLE001
                    result.metadata.setdefault("persistence_warnings", []).append(
                        f"complete: {type(exc).__name__}: {exc}"
                    )
            return result
        finally:
            self._release(run_envelope.run_id)

    def start(
        self, llm: Any, *, task: Any = None, envelope: RunEnvelope | None = None
    ) -> RunHandle:
        """Start a background synchronous run."""
        run_envelope = envelope or self._new_envelope()
        token = CancelToken()
        self._claim(run_envelope.run_id, token)
        events: list[RunEvent] = []
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
