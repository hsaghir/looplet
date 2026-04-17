"""Provenance — capture exactly what the LLM saw and what the agent did.

Two layers:

* **LLM-call provenance** (:class:`RecordingLLMBackend`,
  :class:`AsyncRecordingLLMBackend`) — wrap any :class:`LLMBackend` and
  record every prompt, system prompt, tool schema list, raw response,
  duration, and any error. Dump to a promptflow-style directory of
  ``call_NN_prompt.txt`` / ``call_NN_response.txt`` files plus a
  ``manifest.jsonl`` for machine consumption.

* **Trajectory provenance** (:class:`TrajectoryRecorder`) — a
  :class:`LoopHook` that records a structured :class:`Trajectory` for an
  entire run: per-step timing, tool-call / tool-result dicts, the
  context briefing shown to the LLM, termination reason, and optional
  :class:`Span` tree from an embedded :class:`Tracer`. Dump as a single
  ``trajectory.json`` plus ``steps/step_NN.json`` files.

* :class:`ProvenanceSink` bundles both into a single 3-line drop-in::

        sink = ProvenanceSink(dir="traces/run_1/")
        llm = sink.wrap_llm(AnthropicBackend(...))
        for step in composable_loop(llm=llm, hooks=[sink.trajectory_hook()], ...):
            ...
        sink.flush()

Design notes:

- No third-party dependencies.
- Works for sync or async loops (both ``RecordingLLMBackend`` variants
  implement their matching protocol).
- ``generate_with_tools`` is surfaced only when the wrapped backend
  provides it (so ``hasattr(llm, "generate_with_tools")`` stays honest
  for native-tools detection).
- Recorders keep bounded memory: ``max_chars_per_call`` truncates
  captured strings with an elision marker, and writes to disk flush the
  in-memory list on ``save()``.
- A user-supplied ``redact`` callable can scrub secrets from prompts
  and responses before storage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from openharness.telemetry import Span, Tracer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "LLMCall",
    "RecordingLLMBackend",
    "AsyncRecordingLLMBackend",
    "StepRecord",
    "Trajectory",
    "TrajectoryRecorder",
    "ProvenanceSink",
    "replay_loop",
]


# ── LLM-call provenance ─────────────────────────────────────────────


@dataclass
class LLMCall:
    """One LLM invocation captured by a recording backend.

    Fields mirror the inputs and outputs of the LLM protocol. ``response``
    is the raw return value — a string for ``generate`` or a list of
    content blocks for ``generate_with_tools``. ``duration_ms`` is
    measured around the wrapped call. ``error`` is a short string when
    the wrapped backend raised; it does not include the traceback.
    """

    index: int
    timestamp: float
    duration_ms: float
    method: str  # "generate" | "generate_with_tools"
    prompt: str
    system_prompt: str
    response: Any  # str for generate; list[dict] for generate_with_tools
    temperature: float = 0.2
    max_tokens: int = 2000
    tools: list[dict[str, Any]] | None = None
    step_num: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "method": self.method,
            "prompt_chars": len(self.prompt),
            "response_chars": len(self.response) if isinstance(self.response, str) else None,
            "response_blocks": len(self.response) if isinstance(self.response, list) else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tool_count": len(self.tools) if self.tools else 0,
            "step_num": self.step_num,
            "error": self.error,
        }
        return d


def _truncate(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    keep = max(0, limit - 40)
    return s[:keep] + f"\n... [truncated {len(s) - keep} chars] ..."


class _RecordingBase:
    """Shared state for sync/async recording backends."""

    def __init__(
        self,
        backend: Any,
        *,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self._backend = backend
        self._max_chars = max_chars_per_call
        self._redact = redact
        self.calls: list[LLMCall] = []
        # Set by a TrajectoryRecorder hook so captured calls link back to
        # the step they happened in; optional.
        self.current_step_num: int | None = None

    def _scrub(self, s: str) -> str:
        out = self._redact(s) if self._redact is not None else s
        return _truncate(out, self._max_chars)

    def _record(
        self,
        *,
        method: str,
        prompt: str,
        system_prompt: str,
        response: Any,
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        started_at: float,
        error: str | None = None,
    ) -> LLMCall:
        duration_ms = (time.time() - started_at) * 1000.0
        if isinstance(response, str):
            stored_response: Any = self._scrub(response)
        elif isinstance(response, list):
            stored_response = [dict(b) if isinstance(b, dict) else b for b in response]
        else:
            stored_response = response
        call = LLMCall(
            index=len(self.calls),
            timestamp=started_at,
            duration_ms=duration_ms,
            method=method,
            prompt=self._scrub(prompt),
            system_prompt=self._scrub(system_prompt),
            response=stored_response,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            step_num=self.current_step_num,
            error=error,
        )
        self.calls.append(call)
        return call

    # ── disk IO ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        """Write captured calls as ``call_NN_prompt.txt`` / ``_response.txt``.

        Also writes ``manifest.jsonl`` with one :class:`LLMCall` summary
        per line. Returns the resolved directory path.
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as mf:
            for c in self.calls:
                idx = f"{c.index:02d}"
                (root / f"call_{idx}_prompt.txt").write_text(
                    _format_prompt_block(c), encoding="utf-8"
                )
                (root / f"call_{idx}_response.txt").write_text(
                    _format_response_block(c), encoding="utf-8"
                )
                mf.write(json.dumps(c.to_dict()) + "\n")
        return root

    def reset(self) -> None:
        self.calls.clear()
        self.current_step_num = None


def _format_prompt_block(call: LLMCall) -> str:
    parts = [f"# call {call.index:02d} — method={call.method}"]
    if call.step_num is not None:
        parts.append(f"# step={call.step_num}")
    parts.append(
        f"# temperature={call.temperature} max_tokens={call.max_tokens} "
        f"tools={len(call.tools) if call.tools else 0}"
    )
    if call.system_prompt:
        parts.append("\n## SYSTEM PROMPT\n")
        parts.append(call.system_prompt)
    parts.append("\n## USER PROMPT\n")
    parts.append(call.prompt)
    if call.tools:
        parts.append("\n## TOOLS\n")
        parts.append(json.dumps(call.tools, indent=2))
    return "\n".join(parts)


def _format_response_block(call: LLMCall) -> str:
    parts = [
        f"# call {call.index:02d} — duration={call.duration_ms:.1f}ms "
        f"error={'yes' if call.error else 'no'}",
    ]
    if call.error:
        parts.append(f"\n## ERROR\n{call.error}")
    if isinstance(call.response, str):
        parts.append("\n## RESPONSE\n")
        parts.append(call.response if call.response else "(empty)")
    elif isinstance(call.response, list):
        parts.append("\n## RESPONSE (content blocks)\n")
        parts.append(json.dumps(call.response, indent=2, default=str))
    else:
        parts.append("\n## RESPONSE (repr)\n")
        parts.append(repr(call.response))
    return "\n".join(parts)


class RecordingLLMBackend(_RecordingBase):
    """Wrap any :class:`LLMBackend` and capture every call.

    Example::

        from openharness.provenance import RecordingLLMBackend

        llm = RecordingLLMBackend(AnthropicBackend(api_key=...))
        for step in composable_loop(llm=llm, ...):
            ...
        llm.save("traces/run_1/")
        print(f"{len(llm.calls)} LLM calls captured")

    ``generate_with_tools`` is only surfaced when the wrapped backend
    provides it, so :class:`NativeToolBackend` detection via ``hasattr``
    keeps working unchanged.
    """

    def __init__(
        self,
        backend: Any,
        *,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(
            backend, max_chars_per_call=max_chars_per_call, redact=redact
        )
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl  # type: ignore[attr-defined]

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        started = time.time()
        error: str | None = None
        response: str = ""
        try:
            response = self._backend.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(
                method="generate",
                prompt=prompt,
                system_prompt=system_prompt,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
                started_at=started,
                error=error,
            )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        started = time.time()
        error: str | None = None
        response: list[dict[str, Any]] = []
        try:
            response = self._backend.generate_with_tools(
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(
                method="generate_with_tools",
                prompt=prompt,
                system_prompt=system_prompt,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                started_at=started,
                error=error,
            )


class AsyncRecordingLLMBackend(_RecordingBase):
    """Async counterpart to :class:`RecordingLLMBackend`."""

    def __init__(
        self,
        backend: Any,
        *,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(
            backend, max_chars_per_call=max_chars_per_call, redact=redact
        )
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl  # type: ignore[attr-defined]

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        started = time.time()
        error: str | None = None
        response: str = ""
        try:
            response = await self._backend.generate(
                prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(
                method="generate",
                prompt=prompt,
                system_prompt=system_prompt,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
                started_at=started,
                error=error,
            )

    async def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        started = time.time()
        error: str | None = None
        response: list[dict[str, Any]] = []
        try:
            response = await self._backend.generate_with_tools(
                prompt,
                tools=tools,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(
                method="generate_with_tools",
                prompt=prompt,
                system_prompt=system_prompt,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                started_at=started,
                error=error,
            )


# ── Trajectory provenance ───────────────────────────────────────────


@dataclass
class StepRecord:
    """One loop iteration, structured for trajectory storage."""

    step_num: int
    timestamp: float
    duration_ms: float
    pretty: str
    tool_call: dict[str, Any]
    tool_result: dict[str, Any]
    context_before: str = ""
    llm_call_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_num": self.step_num,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "pretty": self.pretty,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "context_before": self.context_before,
            "llm_call_indices": list(self.llm_call_indices),
        }


@dataclass
class Trajectory:
    """Complete record of one agent loop run."""

    run_id: str
    started_at: float
    ended_at: float | None = None
    steps: list[StepRecord] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    termination_reason: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "termination_reason": self.termination_reason,
            "metadata": self.metadata,
            "step_count": len(self.steps),
            "llm_call_count": len(self.llm_calls),
            "steps": [s.to_dict() for s in self.steps],
            "llm_calls": [c.to_dict() for c in self.llm_calls],
        }


class TrajectoryRecorder:
    """Hook that captures a :class:`Trajectory` across one loop run.

    Install via the ``hooks=`` list on :func:`composable_loop`. Optionally
    pair with a :class:`RecordingLLMBackend` — the recorder will stamp
    ``current_step_num`` on it before each prompt so captured calls link
    back to the step they belonged to.

    Example::

        rec = TrajectoryRecorder()
        for step in composable_loop(llm=llm, hooks=[rec], ...):
            ...
        rec.save("traces/run_1/")
    """

    def __init__(
        self,
        *,
        recording_llm: _RecordingBase | None = None,
        capture_context: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        self.trajectory = Trajectory(
            run_id=uuid4().hex[:12],
            started_at=time.time(),
        )
        self._recording_llm = recording_llm
        self._capture_context = capture_context
        self._tracer: Tracer | None = tracer if tracer is not None else Tracer()
        self._loop_span: Span | None = None
        self._pending_context: str = ""
        self._pending_llm_start: int = 0
        self._step_start_time: float = 0.0

    # ── hook methods ────────────────────────────────────────────

    def pre_loop(self, state: Any, session_log: Any, context: Any) -> None:
        if self._loop_span is None and self._tracer is not None:
            self._loop_span = self._tracer.start_span("loop.run")

    def pre_prompt(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        if self._recording_llm is not None:
            self._recording_llm.current_step_num = step_num
        if self._capture_context and context is not None:
            # ``context`` is whatever the domain's briefing builder returns;
            # stringify defensively so we never raise from a hook.
            try:
                self._pending_context = str(context)
            except Exception:  # pragma: no cover — defensive
                self._pending_context = ""
        self._pending_llm_start = (
            len(self._recording_llm.calls) if self._recording_llm is not None else 0
        )
        self._step_start_time = time.time()
        return None

    def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        tool_result: Any,
        step_num: int,
    ) -> str | None:
        duration_ms = (time.time() - self._step_start_time) * 1000.0
        pretty = ""
        if state is not None and hasattr(state, "steps") and state.steps:
            last = state.steps[-1]
            if hasattr(last, "pretty"):
                pretty = last.pretty()
        llm_indices: list[int] = []
        if self._recording_llm is not None:
            llm_indices = list(
                range(self._pending_llm_start, len(self._recording_llm.calls))
            )
        tc_dict = tool_call.to_dict() if hasattr(tool_call, "to_dict") else {
            "tool": getattr(tool_call, "tool", "?"),
            "args": getattr(tool_call, "args", {}),
        }
        tr_dict = tool_result.to_dict() if hasattr(tool_result, "to_dict") else {
            "tool": getattr(tool_result, "tool", "?"),
            "error": getattr(tool_result, "error", None),
        }
        self.trajectory.steps.append(
            StepRecord(
                step_num=step_num,
                timestamp=self._step_start_time,
                duration_ms=duration_ms,
                pretty=pretty,
                tool_call=tc_dict,
                tool_result=tr_dict,
                context_before=self._pending_context,
                llm_call_indices=llm_indices,
            )
        )
        self._pending_context = ""
        return None

    def on_loop_end(self, state: Any, session_log: Any, context: Any, llm: Any) -> int:
        self.trajectory.ended_at = time.time()
        if self._recording_llm is not None:
            self.trajectory.llm_calls = list(self._recording_llm.calls)
        if self._tracer is not None and self._loop_span is not None:
            self._tracer.end_span(self._loop_span)
            self.trajectory.spans = list(self._tracer.root_spans)
        # Sweep ``state.steps`` for any Step that was yielded but not
        # routed through ``post_dispatch`` — notably the ``done`` step,
        # which the loop handles on its own termination path.
        captured_nums = {s.step_num for s in self.trajectory.steps}
        if state is not None and hasattr(state, "steps"):
            for st in state.steps:
                num = getattr(st, "number", None)
                if num is None or num in captured_nums:
                    continue
                tc = getattr(st, "tool_call", None)
                tr = getattr(st, "tool_result", None)
                tc_dict = tc.to_dict() if tc is not None and hasattr(tc, "to_dict") else {}
                tr_dict = tr.to_dict() if tr is not None and hasattr(tr, "to_dict") else {}
                pretty = st.pretty() if hasattr(st, "pretty") else ""
                self.trajectory.steps.append(
                    StepRecord(
                        step_num=num,
                        timestamp=0.0,
                        duration_ms=0.0,
                        pretty=pretty,
                        tool_call=tc_dict,
                        tool_result=tr_dict,
                        context_before="",
                        llm_call_indices=[],
                    )
                )
        self.trajectory.steps.sort(key=lambda s: s.step_num)
        # Infer termination reason from the last step, if available.
        if self.trajectory.steps:
            last_tool = self.trajectory.steps[-1].tool_call.get("tool", "")
            if last_tool == "done":
                self.trajectory.termination_reason = "done"
            elif self.trajectory.steps[-1].tool_result.get("error"):
                self.trajectory.termination_reason = "error"
            else:
                self.trajectory.termination_reason = "max_steps_or_stop"
        else:
            self.trajectory.termination_reason = "no_steps"
        return 0

    # ── disk IO ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        """Write ``trajectory.json`` + per-step files + (optional) LLM calls.

        Layout::

            <dir>/
              trajectory.json         # full trajectory as one JSON doc
              steps/step_00.json      # per-step records for easy diffing
              steps/step_01.json
              ...
              call_00_prompt.txt      # if recording_llm was attached
              call_00_response.txt
              manifest.jsonl
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "trajectory.json").write_text(
            json.dumps(self.trajectory.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        steps_dir = root / "steps"
        steps_dir.mkdir(exist_ok=True)
        for s in self.trajectory.steps:
            (steps_dir / f"step_{s.step_num:02d}.json").write_text(
                json.dumps(s.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        if self._recording_llm is not None:
            self._recording_llm.save(root)
        return root


# ── Unified sink ────────────────────────────────────────────────────


class ProvenanceSink:
    """3-line facade: wrap the LLM, add the hook, flush on exit.

    Example::

        sink = ProvenanceSink(dir="traces/run_1/")
        llm = sink.wrap_llm(AnthropicBackend(...))
        for step in composable_loop(llm=llm, hooks=[sink.trajectory_hook()], ...):
            ...
        sink.flush()

    The sink is safe to reuse across runs — call :meth:`reset` between
    them, or construct a fresh sink per run (cheaper and clearer).
    """

    def __init__(
        self,
        dir: str | Path,
        *,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
        capture_context: bool = True,
    ) -> None:
        self._dir = Path(dir)
        self._max_chars = max_chars_per_call
        self._redact = redact
        self._capture_context = capture_context
        self._recording_llm: _RecordingBase | None = None
        self._hook: TrajectoryRecorder | None = None

    def wrap_llm(self, backend: Any, *, async_: bool | None = None) -> Any:
        """Wrap ``backend`` in a recording backend and stash a reference.

        Pass ``async_=True`` to force the async variant; otherwise the
        sink inspects ``generate`` — if it is a coroutine function the
        async wrapper is used.
        """
        import inspect

        if async_ is None:
            gen = getattr(backend, "generate", None)
            async_ = bool(gen) and inspect.iscoroutinefunction(gen)
        cls: type[_RecordingBase] = (
            AsyncRecordingLLMBackend if async_ else RecordingLLMBackend
        )
        self._recording_llm = cls(
            backend, max_chars_per_call=self._max_chars, redact=self._redact
        )
        return self._recording_llm

    def trajectory_hook(self) -> TrajectoryRecorder:
        """Create (or return the cached) :class:`TrajectoryRecorder`."""
        if self._hook is None:
            self._hook = TrajectoryRecorder(
                recording_llm=self._recording_llm,
                capture_context=self._capture_context,
            )
        return self._hook

    def flush(self) -> Path:
        """Write everything to disk and return the directory path."""
        if self._hook is not None:
            return self._hook.save(self._dir)
        if self._recording_llm is not None:
            return self._recording_llm.save(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir

    def reset(self) -> None:
        if self._recording_llm is not None:
            self._recording_llm.reset()
        self._hook = None


# ── Replay ──────────────────────────────────────────────────────────


class _ReplayLLMBackend:
    """Internal LLM backend that returns captured responses in order.

    Used by :func:`replay_loop`. Not exported in ``__init__.__all__`` —
    most users should call :func:`replay_loop` instead. Surface it via
    ``from openharness.provenance import _ReplayLLMBackend`` if you need
    low-level control.

    Raises :class:`RuntimeError` if the loop asks for more LLM calls
    than were recorded, or if ``generate_with_tools`` is requested but
    the recorded call used plain ``generate`` (or vice versa).
    """

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self._calls = calls
        self._index = 0
        # Surface ``generate_with_tools`` only if any recorded call used
        # it — keeps ``hasattr`` detection honest for the loop.
        if any(c.get("method") == "generate_with_tools" for c in calls):
            self.generate_with_tools = self._generate_with_tools_impl  # type: ignore[attr-defined]

    def _next(self, expected_method: str) -> dict[str, Any]:
        if self._index >= len(self._calls):
            raise RuntimeError(
                f"replay exhausted: loop asked for call #{self._index + 1} "
                f"but only {len(self._calls)} were recorded. Reduce the "
                f"loop's max_steps or re-record the trace."
            )
        call = self._calls[self._index]
        if call.get("method") != expected_method:
            raise RuntimeError(
                f"replay mismatch at call #{self._index}: recorded "
                f"method={call.get('method')!r} but loop called "
                f"{expected_method!r}. Your tool registry or hooks may "
                f"have diverged from the recorded run."
            )
        self._index += 1
        return call

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        call = self._next("generate")
        response = call.get("response", "")
        if not isinstance(response, str):
            # Defensive: the manifest should always hold a string for
            # generate(), but tolerate callers that loaded raw dicts.
            return json.dumps(response, default=str)
        return response

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict[str, Any]]:
        call = self._next("generate_with_tools")
        response = call.get("response", [])
        if isinstance(response, list):
            return response
        # Defensive: recorded as a JSON string, decode back.
        try:
            decoded = json.loads(response) if isinstance(response, str) else []
            return decoded if isinstance(decoded, list) else []
        except Exception:
            return []


def _load_trace_calls(trace_dir: Path) -> list[dict[str, Any]]:
    """Read the call stream from ``trace_dir``.

    Prefers ``manifest.jsonl`` (which contains structured metadata per
    call) and joins in the body text from ``call_NN_response.txt``. If
    the manifest is missing, falls back to the response files alone
    (every call treated as ``generate`` returning the file contents).
    """
    manifest = trace_dir / "manifest.jsonl"
    calls: list[dict[str, Any]] = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            idx = entry.get("index")
            response_txt = _read_call_body(trace_dir, idx)
            if entry.get("method") == "generate_with_tools":
                # The on-disk .txt file contains a JSON content-block
                # dump after the "## RESPONSE (content blocks)" header.
                response: Any = _extract_content_blocks(response_txt)
            else:
                response = _extract_response_text(response_txt)
            calls.append({
                "index": idx,
                "method": entry.get("method", "generate"),
                "response": response,
            })
    else:
        # Fallback: scan call_NN_response.txt files.
        idx = 0
        while True:
            body = _read_call_body(trace_dir, idx)
            if body is None:
                break
            calls.append({
                "index": idx,
                "method": "generate",
                "response": _extract_response_text(body),
            })
            idx += 1
    if not calls:
        raise FileNotFoundError(
            f"no recorded calls found in {trace_dir} — expected "
            f"manifest.jsonl or call_NN_response.txt files"
        )
    return calls


def _read_call_body(trace_dir: Path, idx: int | None) -> str | None:
    if idx is None:
        return None
    path = trace_dir / f"call_{idx:02d}_response.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _extract_response_text(body: str | None) -> str:
    """Pull the response payload out of a ``call_NN_response.txt`` body.

    The on-disk format begins with ``# call NN ...`` metadata and then
    has a ``## RESPONSE`` header. Strip everything up to that header.
    """
    if not body:
        return ""
    marker = "## RESPONSE\n"
    if marker in body:
        return body.split(marker, 1)[1].lstrip("\n")
    return body


def _extract_content_blocks(body: str | None) -> list[dict[str, Any]]:
    if not body:
        return []
    marker = "## RESPONSE (content blocks)\n"
    if marker in body:
        payload = body.split(marker, 1)[1].strip()
        try:
            decoded = json.loads(payload)
            return decoded if isinstance(decoded, list) else []
        except Exception:
            return []
    return []


def replay_loop(
    trace_dir: str | Path,
    *,
    tools: Any,
    state: Any | None = None,
    hooks: list[Any] | None = None,
    config: Any | None = None,
    task: dict[str, Any] | None = None,
) -> Any:
    """Replay a captured trace through a fresh agent loop.

    Reads the recorded LLM calls from ``trace_dir`` (as written by
    :meth:`RecordingLLMBackend.save` or :meth:`ProvenanceSink.flush`)
    and yields :class:`Step`\\s from :func:`composable_loop` using a
    replay LLM that returns each captured response in order.

    The user's tool registry, hooks, permission engine, and state are
    the ones passed in — this is what makes replay useful: change those
    and diff the step output without spending a dollar on the LLM.

    Args:
        trace_dir: Directory containing ``manifest.jsonl`` +
            ``call_NN_response.txt`` files.
        tools: Tool registry for the replay loop.
        state: Optional :class:`AgentState`. Defaults to a fresh
            :class:`DefaultState` sized to the recorded call count.
        hooks: Optional hooks to install on the replay loop.
        config: Optional :class:`LoopConfig` — if omitted, a default is
            constructed with ``max_steps`` matching the recorded call
            count.
        task: Optional task dict (defaults to ``{}``).

    Yields:
        :class:`Step` objects from the replay.

    Raises:
        FileNotFoundError: ``trace_dir`` has no manifest or response files.
        RuntimeError: The replay loop requests more calls than were
            recorded, or the method (``generate`` vs
            ``generate_with_tools``) diverges from what was recorded.

    Example::

        from openharness.provenance import replay_loop

        for step in replay_loop("traces/run_1/", tools=my_tools):
            print(step.pretty())
    """
    from openharness.loop import LoopConfig, composable_loop  # noqa: PLC0415
    from openharness.types import DefaultState  # noqa: PLC0415

    trace_path = Path(trace_dir)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace directory does not exist: {trace_path}")
    calls = _load_trace_calls(trace_path)
    backend = _ReplayLLMBackend(calls)
    if config is None:
        config = LoopConfig(max_steps=max(len(calls), 1))
    if state is None:
        state = DefaultState(max_steps=max(len(calls), 1))
    yield from composable_loop(
        llm=backend,
        tools=tools,
        task=task or {},
        state=state,
        hooks=hooks or [],
        config=config,
    )
