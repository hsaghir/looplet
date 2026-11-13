"""Provenance - capture exactly what the LLM saw and what the agent did.

Two layers:

* **LLM-call provenance** (:class:`RecordingLLMBackend`,
  :class:`AsyncRecordingLLMBackend`) - wrap any :class:`LLMBackend` and
  record every prompt, system prompt, tool schema list, raw response,
  duration, and any error. Dump to a promptflow-style directory of
  ``call_NN_prompt.txt`` / ``call_NN_response.txt`` files plus a
  ``manifest.jsonl`` for machine consumption.

* **Trajectory provenance** (:class:`TrajectoryRecorder`) - a
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
- Recorders spool complete records to private temporary disk and load
    them lazily. ``max_chars_per_call`` bounds readable projections only.
- A user-supplied ``redact`` callable can scrub secrets from prompts
  and responses before storage.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import MutableSequence, Sequence
from contextlib import aclosing, contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterator
from uuid import uuid4

from looplet._backend_contract import (
    MUTABLE_CAPABILITIES,
    TerminalLLMError,
    backend_attribute,
    invoke_backend,
    json_snapshot,
)
from looplet._storage import DiskBackedList
from looplet.artifact_compat import (
    _artifact_path,
    _atomic_write,
    _read_artifact,
    _validate_model_call_files,
    begin_artifact_write,
    write_artifact_descriptor,
)
from looplet.backend_contract import (
    _capture_model_request,
    _ModelRequestCapture,
    _record_request_capture,
    _record_response_capture,
)
from looplet.telemetry import Span, Tracer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CALL_ARTIFACT = re.compile(r"call_\d+_(?:prompt|response)\.txt")
_CALL_RESPONSE_ARTIFACT = re.compile(r"call_(\d+)_response\.txt")
_STEP_ARTIFACT = re.compile(r"step_\d+\.json")


def _clear_call_artifacts(root: Path) -> None:
    for path in root.iterdir():
        if path.is_file() and _CALL_ARTIFACT.fullmatch(path.name):
            path.unlink()
    (root / "manifest.jsonl").unlink(missing_ok=True)


def _model_attempt_root(directory: Path) -> Path:
    if directory.is_symlink():
        raise ValueError(f"Invalid model attempt directory: {directory}")
    return directory


def _rehydrate_attempt_conversation(root: Path, req: Any, blob_cache: dict[str, Any]) -> None:
    if not isinstance(req, dict):
        return
    conv_in = req.get("conversation_input")
    if not isinstance(conv_in, dict):
        return
    conv_obj = conv_in.get("conversation")
    if not isinstance(conv_obj, dict) or "message_blob_refs" not in conv_obj:
        return
    refs = conv_obj.pop("message_blob_refs")
    if not isinstance(refs, list):
        raise ValueError("Invalid message_blob_refs in model attempt")
    blob_root = _model_attempt_root(root / "blobs")
    restored_msgs: list[Any] = []
    for ref in refs:
        if not isinstance(ref, str) or not re.fullmatch(r"[0-9a-f]{64}", ref):
            raise ValueError(f"Invalid model attempt blob digest: {ref!r}")
        if ref not in blob_cache:
            blob_file = _artifact_path(blob_root, f"{ref}.json")
            raw_bytes = blob_file.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != ref:
                raise ValueError(f"Corrupt model attempt blob: {ref}")
            blob_cache[ref] = json.loads(raw_bytes.decode("utf-8"))
        restored_msgs.append(copy.deepcopy(blob_cache[ref]))
    conv_obj["messages"] = restored_msgs


_ATTEMPT_ID_LOCK = Lock()
_LAST_ATTEMPT_KEY: tuple[int, int] = (0, 0)


def _new_attempt_id() -> str:
    """Return a 32-hex-char attempt ID monotonic in admission order."""
    global _LAST_ATTEMPT_KEY
    now_ms = (time.time_ns() // 1_000_000) & 0xFFFFFFFFFFFF
    with _ATTEMPT_ID_LOCK:
        last_ms, seq = _LAST_ATTEMPT_KEY
        if now_ms > last_ms:
            seq = 0
        else:
            now_ms = last_ms
            seq += 1
            if seq > 0xFFFF:
                now_ms = (now_ms + 1) & 0xFFFFFFFFFFFF
                seq = 0
        _LAST_ATTEMPT_KEY = (now_ms, seq)
    return f"{now_ms:012x}{seq:04x}{os.urandom(8).hex()}"


def load_model_attempts(directory: str | Path) -> list[dict[str, Any]]:
    """Read durable attempts independently of trajectory or manifest export.

    Missing outcomes remain ``None``; an intent or pre-dispatch callback does
    not attest that a provider received a request. Records are sorted by ID
    (monotonic in admission order for SDK-generated attempts).
    """
    grouped: dict[str, dict[str, Any]] = {}
    root = _model_attempt_root(Path(directory) / "model-attempts")
    for path in sorted(root.glob("*.json")):
        match = re.fullmatch(r"([0-9a-f]{32})\.(intent|outcome|dispatch-\d{4})\.json", path.name)
        if match is None:
            raise ValueError(f"Invalid model attempt filename: {path.name}")
        attempt_id, kind = match.groups()
        payload = json.loads(_artifact_path(root, path.name).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or (
            payload.get("schema"),
            payload.get("attempt_id"),
            payload.get("kind"),
        ) != ("looplet.model-attempt.v1", attempt_id, kind):
            raise ValueError(f"Invalid model attempt record: {path.name}")
        grouped.setdefault(attempt_id, {})[kind] = payload
    blob_cache: dict[str, Any] = {}
    attempts = []
    for attempt_id, records in sorted(grouped.items()):
        if "intent" not in records:
            raise ValueError(f"Model attempt has no committed intent: {attempt_id}")
        intent = records["intent"]
        _rehydrate_attempt_conversation(root, intent.get("request"), blob_cache)
        outcome = records.get("outcome")
        if isinstance(outcome, dict) and isinstance(outcome.get("capture"), dict):
            cap = outcome["capture"]
            if "request" not in cap and cap.get("request_ref") == "intent":
                cap.pop("request_ref", None)
                cap["request"] = copy.deepcopy(intent.get("request"))
            else:
                _rehydrate_attempt_conversation(root, cap.get("request"), blob_cache)
        attempts.append(
            {
                "attempt_id": attempt_id,
                "intent": intent,
                "prepared_requests": [
                    records[kind] for kind in sorted(records) if kind.startswith("dispatch-")
                ],
                "outcome": outcome,
            }
        )
    return attempts


__all__ = [
    "LLMCall",
    "load_model_attempts",
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
    is the raw return value - a string for ``generate`` or a list of
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
    temperature: float | None = None
    max_tokens: int = 2000
    tools: list[dict[str, Any]] | None = None
    step_num: int | None = None
    error: str | None = None
    scope: str | None = None
    """When set, identifies the origin of this call within the loop.

    ``None`` (default) means a loop-level call (prompt → LLM → parse).
    ``"tool:<name>"`` means the call was made inside a tool via
    ``ctx.llm.generate()``.  Used by provenance consumers to group
    and filter calls by origin."""
    metadata: dict[str, Any] = field(default_factory=dict)
    capture: dict[str, Any] = field(default_factory=dict)
    """Untruncated, redacted request/response snapshot for structured reconstruction."""

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
            "scope": self.scope,
            "metadata": dict(self.metadata),
            "capture": self.capture,
        }
        return d


def _truncate(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    keep = max(0, limit - 40)
    return s[:keep] + f"\n... [truncated {len(s) - keep} chars] ..."


def _redact_value(
    value: Any, redact: Callable[[str], str] | None, *, preserve_opaque: bool = False
) -> Any:
    if isinstance(value, str):
        return redact(value) if redact is not None else value
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, redact, preserve_opaque=preserve_opaque) for item in value]
    if isinstance(value, dict):
        redacted = {
            _redact_value(key, redact): _redact_value(item, redact, preserve_opaque=preserve_opaque)
            for key, item in value.items()
        }
        kind = value.get("kind", value.get("type"))
        if (
            preserve_opaque
            and redacted != value
            and (
                (isinstance(kind, str) and kind in {"thinking", "redacted_thinking", "reasoning"})
                or "signature" in value
                or "encrypted_content" in value
            )
        ):
            raise TerminalLLMError(
                "Upstream redaction cannot alter opaque provider state",
                termination_reason="unsupported_context",
            )
        return redacted
    if redact is not None and value is not None and not isinstance(value, (bool, int, float)):
        return redact(str(value))
    return value


def _recorded_json(value: Any, redact: Callable[[str], str] | None) -> str:
    if redact is not None:
        value = _redact_value(value, redact)
    return json.dumps(value, indent=2, default=str)


def _recorded_model_json(value: dict[str, Any], redact: Callable[[str], str] | None) -> str:
    if redact is not None:
        value = _redact_model_record(value, redact)
    return json.dumps(value, indent=2, default=str)


def _redact_model_record(
    value: dict[str, Any], redact: Callable[[str], str] | None
) -> dict[str, Any]:
    """Redact model evidence without changing SDK-issued linkage identifiers."""
    if redact is None:
        return value
    redacted = _redact_value(value, redact)
    for source, target in (
        (value, redacted),
        (value.get("capture", {}), redacted.get("capture", {})),
    ):
        if not isinstance(source, dict) or not isinstance(target, dict):
            continue
        for field_name in ("request_id", "attempt_id"):
            if field_name in source:
                target[field_name] = source[field_name]
        replay_source = source.get("replay_source")
        target_replay = target.get("replay_source")
        if isinstance(replay_source, dict) and isinstance(target_replay, dict):
            for field_name in ("request_id", "attempt_id"):
                if field_name in replay_source:
                    target_replay[field_name] = replay_source[field_name]
    return redacted


def _recording_backend(backend: Any) -> _RecordingBase | None:
    for _ in range(8):
        if isinstance(backend, _RecordingBase):
            return backend
        attributes = getattr(backend, "__dict__", {})
        backend = next(
            (
                attributes[name]
                for name in ("backend", "_backend", "_inner")
                if attributes.get(name) is not None
            ),
            None,
        )
        if backend is None:
            break
    return None


def _has_replay_backend(backend: Any) -> bool:
    for _ in range(8):
        if isinstance(backend, _ReplayLLMBackend):
            return True
        attributes = getattr(backend, "__dict__", {})
        backend = next(
            (
                attributes[name]
                for name in ("backend", "_backend", "_inner")
                if attributes.get(name) is not None
            ),
            None,
        )
        if backend is None:
            break
    return False


class _PreparedRequestCallbacks:
    _lock = Lock()

    def __init__(self, previous: Callable[[], None] | None) -> None:
        self.previous = previous
        self.active_scopes = 0
        self.observer: ContextVar[Callable[[], None] | None] = ContextVar(
            "looplet_prepared_request_observer", default=None
        )

    def __call__(self) -> None:
        if self.previous is not None:
            self.previous()
        observer = self.observer.get()
        if observer is not None:
            observer()


class _ModelCallIdentity:
    def __init__(self) -> None:
        self._path: ContextVar[tuple[str, ...]] = ContextVar(
            "looplet_model_call_path", default=("loop",)
        )
        self._ordinals: dict[tuple[str, ...], int] = {}
        self._lock = Lock()

    @contextmanager
    def scope(self, scope: str, step_num: int | None) -> Iterator[None]:
        token = self._path.set((*self._path.get(), f"{scope}:{step_num}"))
        try:
            yield
        finally:
            self._path.reset(token)

    def next(self) -> str:
        path = self._path.get()
        with self._lock:
            ordinal = self._ordinals.get(path, 0)
            self._ordinals[path] = ordinal + 1
        return "/".join((*path, str(ordinal)))

    def reset(self) -> None:
        with self._lock:
            self._ordinals.clear()


class _RecordingBase:
    """Shared state for sync/async recording backends."""

    _unbound_conversation = object()

    def __init__(
        self,
        backend: Any,
        *,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
        redact_upstream: bool = True,
        attempt_directory: str | Path | None = None,
    ) -> None:
        self._backend = backend
        self._attempt_directory = Path(attempt_directory) if attempt_directory is not None else None
        self._max_chars = max_chars_per_call
        self._redact = redact
        # When True (default), ``redact`` is also applied to the prompt
        # and system_prompt BEFORE they are forwarded to the wrapped
        # backend - so secrets never leave the process, not just the
        # trace.  Set to False to record-only.
        self._redact_upstream = redact_upstream
        self.calls: MutableSequence[LLMCall] = DiskBackedList(vars, lambda value: LLMCall(**value))
        self._calls_lock = Lock()
        self._call_identity = _ModelCallIdentity()
        self._call_scope: ContextVar[str | None] = ContextVar("looplet_call_scope", default=None)
        self._scoped_step_num: ContextVar[int | None] = ContextVar(
            "looplet_scoped_step_num", default=None
        )
        self._model_call_context: ContextVar[dict[str, Any] | None] = ContextVar(
            "looplet_model_call_context", default=None
        )
        self._conversation_input: ContextVar[dict[str, Any] | None] = ContextVar(
            "looplet_recorded_conversation_input"
        )
        self._blob_lock = Lock()
        self._last_blob_msgs: list[tuple[Any, str]] = []
        self._written_blobs: set[tuple[Path, str]] = set()
        # Set by a TrajectoryRecorder hook so captured calls link back to
        # the step they happened in; optional.
        self.current_step_num: int | None = None

    def __getattr__(self, name: str) -> Any:
        return backend_attribute(self._backend, name)

    def for_backend(self, backend: Any) -> _RecordingBase:
        """Record a router-selected backend into this run's existing call stream."""
        if backend is self:
            return self
        asynchronous = any(
            inspect.iscoroutinefunction(getattr(backend, method, None))
            for method in ("generate", "generate_with_tools")
        )
        cls = AsyncRecordingLLMBackend if asynchronous else RecordingLLMBackend
        recorder = cls(
            backend,
            max_chars_per_call=self._max_chars,
            redact=self._redact,
            redact_upstream=self._redact_upstream,
            attempt_directory=self._attempt_directory,
        )
        recorder.calls = self.calls
        recorder._calls_lock = self._calls_lock
        recorder._call_identity = self._call_identity
        recorder.current_step_num = self.current_step_num
        recorder._call_scope = self._call_scope
        recorder._scoped_step_num = self._scoped_step_num
        recorder._model_call_context = self._model_call_context
        recorder._conversation_input = self._conversation_input
        recorder._blob_lock = self._blob_lock
        recorder._last_blob_msgs = self._last_blob_msgs
        recorder._written_blobs = self._written_blobs
        return recorder

    def __setattr__(self, name: str, value: Any) -> None:
        if name in MUTABLE_CAPABILITIES and "_backend" in self.__dict__:
            setattr(self._backend, name, value)
        else:
            object.__setattr__(self, name, value)

    @contextmanager
    def _scope_calls(self, scope: str, *, step_num: int | None = None) -> Iterator[None]:
        token = self._call_scope.set(self._call_scope.get() or scope)
        outer_step = self._scoped_step_num.get()
        step_token = self._scoped_step_num.set(outer_step if outer_step is not None else step_num)
        try:
            with self._call_identity.scope(scope, step_num):
                nested_scope: Callable[..., Any] | None = getattr(
                    self._backend, "_scope_calls", None
                )
                if callable(nested_scope):
                    with nested_scope(scope, step_num=step_num):
                        yield
                else:
                    yield
        finally:
            self._scoped_step_num.reset(step_token)
            self._call_scope.reset(token)

    def _upstream(self, s: str) -> str:
        """Scrub a prompt before it is sent to the wrapped backend."""
        if self._redact is not None and self._redact_upstream:
            return self._redact(s)
        return s

    def set_model_turn_context(self, context: dict[str, Any]) -> None:
        self._model_call_context.set(json_snapshot(context))
        method = getattr(self._backend, "set_model_turn_context", None)
        if callable(method):
            method(context)

    def set_conversation_input(self, snapshot: dict[str, Any] | None) -> Any:
        method = getattr(self._backend, "set_conversation_input", None)
        if not callable(method):
            if snapshot is not None:
                raise TerminalLLMError(
                    "Backend lacks conversation input support",
                    termination_reason="unsupported_context",
                )
            self._conversation_input.set(None)
            return None
        detached = json_snapshot(snapshot)
        if self._redact_upstream:
            detached = _redact_value(detached, self._redact, preserve_opaque=True)
        self._conversation_input.set(json_snapshot(detached))
        return method(detached)

    def apply_context_compaction(self, snapshot: dict[str, Any]) -> Any:
        method = getattr(self._backend, "apply_context_compaction", None)
        if not callable(method):
            return False
        detached = json_snapshot(snapshot)
        if self._redact_upstream:
            detached = _redact_value(detached, self._redact, preserve_opaque=True)
        return method(detached)

    def _scrub(self, s: str, *, already_redacted: bool = False) -> str:
        out = self._redact(s) if (self._redact is not None and not already_redacted) else s
        return _truncate(out, self._max_chars)

    def _redact_native_response(self, value: Any) -> Any:
        return _redact_value(value, self._redact)

    def _write_attempt(self, attempt_id: str, kind: str, value: dict[str, Any]) -> None:
        assert self._attempt_directory is not None
        root = _model_attempt_root(self._attempt_directory)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        policy = _artifact_path(root, ".gitignore")
        record = _artifact_path(root, f"{attempt_id}.{kind}.json")
        if not policy.exists():
            _atomic_write(policy, "*\n")
        raw = json_snapshot(
            {
                "schema": "looplet.model-attempt.v1",
                "kind": kind,
                "attempt_id": attempt_id,
                **value,
            }
        )
        payload = raw if kind in {"intent", "outcome"} else _redact_model_record(raw, self._redact)
        if kind == "intent" and isinstance(payload.get("request"), dict):
            conv_in = payload["request"].get("conversation_input")
            if isinstance(conv_in, dict) and isinstance(conv_in.get("conversation"), dict):
                conv_obj = conv_in["conversation"]
                msgs = conv_obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    blob_root = _model_attempt_root(root / "blobs")
                    blob_root.mkdir(mode=0o700, exist_ok=True)
                    refs: list[str] = []
                    with self._blob_lock:
                        prev = list(self._last_blob_msgs)
                        new_memo: list[tuple[Any, str]] = []
                        for pos, msg in enumerate(msgs):
                            if pos < len(prev) and prev[pos][0] == msg:
                                digest = prev[pos][1]
                                if (blob_root, digest) not in self._written_blobs:
                                    blob_text = json.dumps(
                                        msg,
                                        ensure_ascii=True,
                                        allow_nan=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    )
                                    blob_path = _artifact_path(blob_root, f"{digest}.json")
                                    if not blob_path.exists():
                                        _atomic_write(blob_path, blob_text)
                                    self._written_blobs.add((blob_root, digest))
                            else:
                                blob_text = json.dumps(
                                    msg,
                                    ensure_ascii=True,
                                    allow_nan=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                digest = hashlib.sha256(blob_text.encode("ascii")).hexdigest()
                                if (blob_root, digest) not in self._written_blobs:
                                    blob_path = _artifact_path(blob_root, f"{digest}.json")
                                    if not blob_path.exists():
                                        _atomic_write(blob_path, blob_text)
                                    self._written_blobs.add((blob_root, digest))
                            refs.append(digest)
                            new_memo.append((msg, digest))
                        self._last_blob_msgs[:] = new_memo
                    conv_in["conversation"] = {
                        k: v for k, v in conv_obj.items() if k != "messages"
                    } | {"message_blob_refs": refs}
        elif kind == "outcome" and isinstance(payload.get("capture"), dict):
            cap = payload["capture"]
            if "request" in cap:
                payload["capture"] = {k: v for k, v in cap.items() if k != "request"} | {
                    "request_ref": "intent"
                }
        _atomic_write(
            record,
            json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        )

    def _write_dispatch(self, attempt_id: str, number: int, request: Any) -> None:
        self._write_attempt(
            attempt_id,
            f"dispatch-{number:04d}",
            {
                "provider_request": request,
                "scope": "backend_pre_dispatch_callback_not_provider_delivery_attestation",
            },
        )

    def _begin_attempt(
        self,
        *,
        method: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float | None,
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
        request_id: str | None = None,
    ) -> str | None:
        if self._attempt_directory is None:
            return None
        attempt_id = _new_attempt_id()
        if options:
            options = {k: v for k, v in options.items() if k not in {"on_chunk", "on_tool_call"}}
        recorded_prompt = (
            prompt if self._redact_upstream or self._redact is None else self._redact(prompt)
        )
        recorded_system = (
            system_prompt
            if self._redact_upstream or self._redact is None
            else self._redact(system_prompt)
        )
        conv_snapshot = self._conversation_input.get(None)
        recorded_conv = (
            conv_snapshot
            if self._redact_upstream or self._redact is None
            else _redact_value(conv_snapshot, self._redact)
        )
        self._write_attempt(
            attempt_id,
            "intent",
            {
                "started_at": time.time(),
                "request_id": request_id,
                "execution": _redact_value(self._model_call_context.get(), self._redact),
                "scope": "logical_request_intent_not_provider_delivery_attestation",
                "request": {
                    "method": method,
                    "prompt": recorded_prompt,
                    "system_prompt": recorded_system,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": _redact_value(tools, self._redact),
                    "options": _redact_value(options, self._redact),
                    "conversation_input": recorded_conv,
                },
            },
        )
        return attempt_id

    @contextmanager
    def _capture_dispatch(
        self,
        attempt_id: str | None,
        prepared_requests: list[Any],
        native_capture: _ModelRequestCapture | None = None,
        *,
        persist_prepared: bool = True,
    ) -> Iterator[_ModelRequestCapture]:
        conversation_input = self._conversation_input.get(self._unbound_conversation)
        bind_conversation = getattr(self._backend, "set_conversation_input", None)
        if conversation_input is not self._unbound_conversation and callable(bind_conversation):
            bind_conversation(conversation_input)
        with _capture_model_request(native_capture) as captured:
            setter = getattr(self._backend, "set_request_attempt_callback", None)
            if not callable(setter):
                yield captured
                return

            def prepared() -> None:
                request = json_snapshot(getattr(self._backend, "last_request", None))
                prepared_requests.append(request)
                if persist_prepared and attempt_id is not None:
                    self._write_dispatch(attempt_id, len(prepared_requests), request)

            with _PreparedRequestCallbacks._lock:
                previous = getattr(self._backend, "request_attempt_callback", None)
                callbacks = (
                    previous
                    if isinstance(previous, _PreparedRequestCallbacks)
                    else _PreparedRequestCallbacks(previous)
                )
                if callbacks is not previous:
                    setter(callbacks)
                callbacks.active_scopes += 1
            token = callbacks.observer.set(prepared)
            try:
                yield captured
            finally:
                callbacks.observer.reset(token)
                with _PreparedRequestCallbacks._lock:
                    callbacks.active_scopes -= 1
                    if not callbacks.active_scopes:
                        setter(callbacks.previous)

    async def _capture_io(self, operation: Callable[[], Any]) -> Any:
        import asyncio

        if self._attempt_directory is None:
            return operation()
        pending = asyncio.get_running_loop().run_in_executor(
            None,
            copy_context().run,
            operation,
        )
        cancelled: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(pending)
                break
            except asyncio.CancelledError as failure:
                if cancelled is None:
                    cancelled = failure
            except Exception as failure:
                if cancelled is not None:
                    raise cancelled from failure
                raise
        if cancelled is not None:
            raise cancelled
        return result

    def _record(
        self,
        *,
        method: str,
        prompt: str,
        system_prompt: str,
        response: Any,
        temperature: float | None,
        max_tokens: int,
        tools: list[dict[str, Any]] | None,
        started_at: float,
        error: str | None = None,
        termination_reason: str | None = None,
        options: dict[str, Any] | None = None,
        attempt_id: str | None = None,
        failure: BaseException | None = None,
        prepared_requests: list[Any] | None = None,
        native_capture: _ModelRequestCapture | None = None,
        persist: bool = True,
        request_id: str | None = None,
    ) -> LLMCall:
        duration_ms = (time.time() - started_at) * 1000.0
        execution = self._model_call_context.get()
        self._model_call_context.set(None)
        if options:
            options = {k: v for k, v in options.items() if k not in {"on_chunk", "on_tool_call"}}
        if isinstance(response, str):
            stored_response: Any = self._scrub(response)
        elif isinstance(response, list):
            stored_response = self._redact_native_response(response)
        else:
            stored_response = response
        if native_capture is not None and (
            native_capture.provider_request is not None or native_capture.replay_source is not None
        ):
            provider_request = native_capture.provider_request
            provider_response = native_capture.provider_response
            stop_reason = native_capture.stop_reason
            usage = native_capture.usage or {}
        else:
            provider_request = prepared_requests[-1] if prepared_requests else None
            if not prepared_requests and (error is None or attempt_id is None):
                provider_request = getattr(self._backend, "last_request", None)
            provider_response = (
                getattr(self._backend, "last_response", None) if error is None else None
            )
            stop_reason = (
                getattr(self._backend, "last_stop_reason", None) if error is None else None
            )
            usage = getattr(self._backend, "last_usage", {}) if error is None else {}
        scoped_step = self._scoped_step_num.get()
        effective_step_num = scoped_step if scoped_step is not None else self.current_step_num
        redacted_prompt = (
            prompt if self._redact_upstream or self._redact is None else self._redact(prompt)
        )
        redacted_system = (
            system_prompt
            if self._redact_upstream or self._redact is None
            else self._redact(system_prompt)
        )
        conv_snapshot = self._conversation_input.get(None)
        redacted_conv = (
            conv_snapshot
            if self._redact_upstream or self._redact is None
            else _redact_value(conv_snapshot, self._redact)
        )
        redacted_tools = _redact_value(tools, self._redact)
        redacted_options = _redact_value(options or {}, self._redact)
        redacted_response = (
            self._redact(response)
            if isinstance(response, str) and self._redact is not None
            else stored_response
            if isinstance(response, list)
            else _redact_value(response, self._redact)
        )
        redacted_error = (
            self._redact(error) if error is not None and self._redact is not None else error
        )
        redacted_replay_source = None
        if native_capture is not None and native_capture.replay_source is not None:
            raw_rs = json_snapshot(native_capture.replay_source)
            redacted_replay_source = _redact_value(raw_rs, self._redact)
            if isinstance(raw_rs, dict) and isinstance(redacted_replay_source, dict):
                for id_field in ("request_id", "attempt_id"):
                    if id_field in raw_rs:
                        redacted_replay_source[id_field] = raw_rs[id_field]
        call = LLMCall(
            index=len(self.calls),
            timestamp=started_at,
            duration_ms=duration_ms,
            method=method,
            prompt=_truncate(redacted_prompt, self._max_chars),
            system_prompt=_truncate(redacted_system, self._max_chars),
            response=(
                _truncate(redacted_response, self._max_chars)
                if isinstance(response, str)
                else stored_response
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            tools=redacted_tools,
            step_num=effective_step_num,
            error=_truncate(redacted_error, self._max_chars)
            if redacted_error is not None
            else None,
            scope=self._call_scope.get(),
            capture=json_snapshot(
                {
                    "schema": "looplet.model-call.v2" if execution else "looplet.model-call.v1",
                    **({"request_id": request_id} if request_id is not None else {}),
                    **(
                        {"replay_source": redacted_replay_source}
                        if redacted_replay_source is not None
                        else {}
                    ),
                    **({"execution": _redact_value(execution, self._redact)} if execution else {}),
                    "request": {
                        "method": method,
                        "prompt": redacted_prompt,
                        "system_prompt": redacted_system,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tools": redacted_tools,
                        "options": redacted_options,
                        "conversation_input": redacted_conv,
                    },
                    "response": redacted_response,
                    "provider_request": _redact_value(
                        json_snapshot(provider_request), self._redact
                    ),
                    "provider_response": (
                        _redact_value(json_snapshot(provider_response), self._redact)
                        if error is None
                        else None
                    ),
                    "stop_reason": (
                        _redact_value(stop_reason, self._redact) if error is None else None
                    ),
                    "usage": (
                        _redact_value(json_snapshot(usage), self._redact) if error is None else {}
                    ),
                    "error": redacted_error,
                    **(
                        {
                            "terminal_error": {
                                "schema": "looplet.terminal-error.v1",
                                "termination_reason": _redact_value(
                                    termination_reason, self._redact
                                ),
                            }
                        }
                        if termination_reason is not None
                        else {}
                    ),
                }
            ),
        )
        if attempt_id is not None:
            assert call.capture is not None
            call.capture["attempt_id"] = attempt_id
        if persist:
            self._commit_call(call, attempt_id, failure)
        return call

    def _commit_call(
        self,
        call: LLMCall,
        attempt_id: str | None,
        failure: BaseException | None,
        prepared_requests: list[Any] | None = None,
    ) -> None:
        if attempt_id is not None:
            try:
                for number, request in enumerate(prepared_requests or (), 1):
                    self._write_dispatch(attempt_id, number, request)
                self._write_attempt(
                    attempt_id,
                    "outcome",
                    {
                        "capture": call.capture,
                        "scope": "logical_outcome_and_backend_reported_provider_envelopes",
                        "response_complete": call.error is None,
                    },
                )
            except Exception as capture_failure:
                if failure is None:
                    raise
                failure.add_note(
                    f"Attempt outcome persistence failed: {type(capture_failure).__name__}: "
                    f"{capture_failure}"
                )
        with self._calls_lock:
            call.index = len(self.calls)
            self.calls.append(call)

    # ── disk IO ─────────────────────────────────────────────────

    def save(self, directory: str | Path, *, publish_descriptor: bool = True) -> Path:
        """Write captured calls as ``call_NN_prompt.txt`` / ``_response.txt``.

        Also writes ``manifest.jsonl`` with one :class:`LLMCall` summary
        per line. Returns the resolved directory path.
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if publish_descriptor:
            begin_artifact_write(root)
        _clear_call_artifacts(root)
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
                entry = c.to_dict()
                if self._redact is not None and entry["metadata"]:
                    entry["metadata"] = _redact_value(entry["metadata"], self._redact)
                mf.write(json.dumps(entry, default=str) + "\n")
        if publish_descriptor:
            write_artifact_descriptor(root, kind="provenance", components=("model_calls",))
        return root

    def reset(self) -> None:
        self.calls.clear()
        self._call_identity.reset()
        self.current_step_num = None
        self._scoped_step_num.set(None)
        self._conversation_input.set(None)


def _format_prompt_block(call: LLMCall) -> str:
    parts = [f"# call {call.index:02d} - method={call.method}"]
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
        f"# call {call.index:02d} - duration={call.duration_ms:.1f}ms "
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

        from looplet.provenance import RecordingLLMBackend

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
        redact_upstream: bool = True,
        attempt_directory: str | Path | None = None,
    ) -> None:
        super().__init__(
            backend,
            max_chars_per_call=max_chars_per_call,
            redact=redact,
            redact_upstream=redact_upstream,
            attempt_directory=attempt_directory,
        )
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl
        if hasattr(backend, "stream") and getattr(backend, "stream_in_loop", True):
            self.stream = self._stream_impl
            self.stream_in_loop = True

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> str:
        started = time.time()
        error: str | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        response: str = ""
        upstream_prompt = self._upstream(prompt)
        upstream_system = self._upstream(system_prompt)
        request_id = self._call_identity.next()
        attempt_id = self._begin_attempt(
            method="generate",
            prompt=upstream_prompt,
            system_prompt=upstream_system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=None,
            options=options,
            request_id=request_id,
        )
        prepared_requests: list[Any] = []
        native_capture = None
        try:
            with self._capture_dispatch(attempt_id, prepared_requests) as native_capture:
                response = invoke_backend(
                    self._backend.generate,
                    upstream_prompt,
                    {
                        **options,
                        "max_tokens": max_tokens,
                        "system_prompt": upstream_system,
                        "temperature": temperature,
                    },
                )
            return response
        except BaseException as exc:
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TerminalLLMError):
                termination_reason = exc.termination_reason
            raise
        finally:
            self._record(
                method="generate",
                prompt=upstream_prompt,
                system_prompt=upstream_system,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
                started_at=started,
                error=error,
                termination_reason=termination_reason,
                options=options,
                attempt_id=attempt_id,
                failure=failure,
                prepared_requests=prepared_requests,
                native_capture=native_capture,
                request_id=request_id,
            )

    def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> list[dict[str, Any]]:
        started = time.time()
        error: str | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        response: list[dict[str, Any]] = []
        upstream_prompt = self._upstream(prompt)
        upstream_system = self._upstream(system_prompt)
        request_id = self._call_identity.next()
        attempt_id = self._begin_attempt(
            method="generate_with_tools",
            prompt=upstream_prompt,
            system_prompt=upstream_system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            options=options,
            request_id=request_id,
        )
        prepared_requests: list[Any] = []
        native_capture = None
        try:
            with self._capture_dispatch(attempt_id, prepared_requests) as native_capture:
                response = invoke_backend(
                    self._backend.generate_with_tools,
                    upstream_prompt,
                    {
                        **options,
                        "tools": tools,
                        "max_tokens": max_tokens,
                        "system_prompt": upstream_system,
                        "temperature": temperature,
                    },
                )
            return response
        except BaseException as exc:
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TerminalLLMError):
                termination_reason = exc.termination_reason
            raise
        finally:
            self._record(
                method="generate_with_tools",
                prompt=upstream_prompt,
                system_prompt=upstream_system,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                started_at=started,
                error=error,
                termination_reason=termination_reason,
                options=options,
                attempt_id=attempt_id,
                failure=failure,
                prepared_requests=prepared_requests,
                native_capture=native_capture,
                request_id=request_id,
            )

    def _stream_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Iterator[str]:
        started = time.time()
        error: str | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        chunks: list[str] = []
        upstream_prompt = self._upstream(prompt)
        upstream_system = self._upstream(system_prompt)
        request_id = self._call_identity.next()
        attempt_id = self._begin_attempt(
            method="generate",
            prompt=upstream_prompt,
            system_prompt=upstream_system,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=None,
            options=options,
            request_id=request_id,
        )
        prepared_requests: list[Any] = []
        native_capture = _ModelRequestCapture()
        try:
            with self._capture_dispatch(attempt_id, prepared_requests, native_capture):
                stream_iter = invoke_backend(
                    self._backend.stream,
                    upstream_prompt,
                    {
                        **options,
                        "max_tokens": max_tokens,
                        "system_prompt": upstream_system,
                        "temperature": temperature,
                    },
                )
                iterator = iter(stream_iter)
            try:
                while True:
                    with self._capture_dispatch(attempt_id, prepared_requests, native_capture):
                        try:
                            chunk = next(iterator)
                        except StopIteration:
                            break
                    chunks.append(chunk)
                    yield chunk
            finally:
                with self._capture_dispatch(attempt_id, prepared_requests, native_capture):
                    close = getattr(stream_iter, "close", None)
                    if close is not None:
                        close()
        except BaseException as exc:
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TerminalLLMError):
                termination_reason = exc.termination_reason
            raise
        finally:
            self._record(
                method="generate",
                prompt=upstream_prompt,
                system_prompt=upstream_system,
                response="".join(chunks),
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
                started_at=started,
                error=error,
                termination_reason=termination_reason,
                options=options,
                attempt_id=attempt_id,
                failure=failure,
                prepared_requests=prepared_requests,
                native_capture=native_capture,
                request_id=request_id,
            )


class AsyncRecordingLLMBackend(_RecordingBase):
    """Async counterpart to :class:`RecordingLLMBackend`."""

    def __init__(
        self,
        backend: Any,
        *,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
        redact_upstream: bool = True,
        attempt_directory: str | Path | None = None,
    ) -> None:
        super().__init__(
            backend,
            max_chars_per_call=max_chars_per_call,
            redact=redact,
            redact_upstream=redact_upstream,
            attempt_directory=attempt_directory,
        )
        if hasattr(backend, "generate_with_tools"):
            self.generate_with_tools = self._generate_with_tools_impl
        if hasattr(backend, "stream") and getattr(backend, "stream_in_loop", True):
            self.stream = self._stream_impl
            self.stream_in_loop = True

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> str:
        import inspect

        started = time.time()
        error: str | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        response: str = ""
        upstream_prompt = self._upstream(prompt)
        upstream_system = self._upstream(system_prompt)
        request_id = self._call_identity.next()
        attempt_id = await self._capture_io(
            lambda: self._begin_attempt(
                method="generate",
                prompt=upstream_prompt,
                system_prompt=upstream_system,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=None,
                options=options,
                request_id=request_id,
            )
        )
        prepared_requests: list[Any] = []
        native_capture = None
        try:
            with self._capture_dispatch(
                attempt_id, prepared_requests, persist_prepared=False
            ) as native_capture:
                pending = invoke_backend(
                    self._backend.generate,
                    upstream_prompt,
                    {
                        **options,
                        "max_tokens": max_tokens,
                        "system_prompt": upstream_system,
                        "temperature": temperature,
                    },
                )
                response = await pending if inspect.isawaitable(pending) else pending
            return response
        except BaseException as exc:
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TerminalLLMError):
                termination_reason = exc.termination_reason
            raise
        finally:
            try:
                call = self._record(
                    method="generate",
                    prompt=upstream_prompt,
                    system_prompt=upstream_system,
                    response=response,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=None,
                    started_at=started,
                    error=error,
                    termination_reason=termination_reason,
                    options=options,
                    attempt_id=attempt_id,
                    failure=failure,
                    prepared_requests=prepared_requests,
                    native_capture=native_capture,
                    persist=False,
                    request_id=request_id,
                )
                await self._capture_io(
                    lambda: self._commit_call(call, attempt_id, failure, prepared_requests)
                )
            finally:
                self._model_call_context.set(None)

    async def _generate_with_tools_impl(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> list[dict[str, Any]]:
        import inspect

        started = time.time()
        error: str | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        response: list[dict[str, Any]] = []
        upstream_prompt = self._upstream(prompt)
        upstream_system = self._upstream(system_prompt)
        request_id = self._call_identity.next()
        attempt_id = await self._capture_io(
            lambda: self._begin_attempt(
                method="generate_with_tools",
                prompt=upstream_prompt,
                system_prompt=upstream_system,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                options=options,
                request_id=request_id,
            )
        )
        prepared_requests: list[Any] = []
        native_capture = None
        try:
            with self._capture_dispatch(
                attempt_id, prepared_requests, persist_prepared=False
            ) as native_capture:
                pending = invoke_backend(
                    self._backend.generate_with_tools,
                    upstream_prompt,
                    {
                        **options,
                        "tools": tools,
                        "max_tokens": max_tokens,
                        "system_prompt": upstream_system,
                        "temperature": temperature,
                    },
                )
                response = await pending if inspect.isawaitable(pending) else pending
            return response
        except BaseException as exc:
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TerminalLLMError):
                termination_reason = exc.termination_reason
            raise
        finally:
            try:
                call = self._record(
                    method="generate_with_tools",
                    prompt=upstream_prompt,
                    system_prompt=upstream_system,
                    response=response,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    started_at=started,
                    error=error,
                    termination_reason=termination_reason,
                    options=options,
                    attempt_id=attempt_id,
                    failure=failure,
                    prepared_requests=prepared_requests,
                    native_capture=native_capture,
                    persist=False,
                    request_id=request_id,
                )
                await self._capture_io(
                    lambda: self._commit_call(call, attempt_id, failure, prepared_requests)
                )
            finally:
                self._model_call_context.set(None)

    async def _stream_impl(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
        **options: Any,
    ) -> Any:
        import inspect

        started = time.time()
        error: str | None = None
        failure: BaseException | None = None
        termination_reason: str | None = None
        chunks: list[str] = []
        upstream_prompt = self._upstream(prompt)
        upstream_system = self._upstream(system_prompt)
        request_id = self._call_identity.next()
        attempt_id = await self._capture_io(
            lambda: self._begin_attempt(
                method="generate",
                prompt=upstream_prompt,
                system_prompt=upstream_system,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=None,
                options=options,
                request_id=request_id,
            )
        )
        prepared_requests: list[Any] = []
        native_capture = _ModelRequestCapture()
        try:
            with self._capture_dispatch(
                attempt_id, prepared_requests, native_capture, persist_prepared=False
            ):
                stream_out = invoke_backend(
                    self._backend.stream,
                    upstream_prompt,
                    {
                        **options,
                        "max_tokens": max_tokens,
                        "system_prompt": upstream_system,
                        "temperature": temperature,
                    },
                )
                if inspect.isawaitable(stream_out):
                    stream_out = await stream_out
                asynchronous = hasattr(stream_out, "__aiter__")
                iterator = aiter(stream_out) if asynchronous else iter(stream_out)
            try:
                while True:
                    with self._capture_dispatch(
                        attempt_id, prepared_requests, native_capture, persist_prepared=False
                    ):
                        try:
                            chunk = await anext(iterator) if asynchronous else next(iterator)
                        except (StopAsyncIteration, StopIteration):
                            break
                    chunks.append(chunk)
                    yield chunk
            finally:
                with self._capture_dispatch(
                    attempt_id, prepared_requests, native_capture, persist_prepared=False
                ):
                    close = getattr(stream_out, "aclose", None) or getattr(
                        stream_out, "close", None
                    )
                    if close is not None:
                        pending = close()
                        if inspect.isawaitable(pending):
                            await pending
        except BaseException as exc:
            failure = exc
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TerminalLLMError):
                termination_reason = exc.termination_reason
            raise
        finally:
            try:
                call = self._record(
                    method="generate",
                    prompt=upstream_prompt,
                    system_prompt=upstream_system,
                    response="".join(chunks),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=None,
                    started_at=started,
                    error=error,
                    termination_reason=termination_reason,
                    options=options,
                    attempt_id=attempt_id,
                    failure=failure,
                    prepared_requests=prepared_requests,
                    native_capture=native_capture,
                    persist=False,
                    request_id=request_id,
                )
                await self._capture_io(
                    lambda: self._commit_call(call, attempt_id, failure, prepared_requests)
                )
            finally:
                self._model_call_context.set(None)


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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "metadata": dict(self.metadata),
        }


@dataclass
class Trajectory:
    """Complete record of one agent loop run."""

    run_id: str
    started_at: float
    ended_at: float | None = None
    task: dict[str, Any] = field(default_factory=dict)
    steps: MutableSequence[StepRecord] = field(default_factory=list)
    llm_calls: MutableSequence[LLMCall] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    termination_reason: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    session_log_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "task": self.task,
            "termination_reason": self.termination_reason,
            "session_log_text": self.session_log_text,
            "metadata": self.metadata,
            "step_count": len(self.steps),
            "llm_call_count": len(self.llm_calls),
            "spans": [s.to_dict() for s in self.spans],
            "steps": [s.to_dict() for s in self.steps],
            "llm_calls": [c.to_dict() for c in self.llm_calls],
        }


class TrajectoryRecorder:
    """Hook that captures a :class:`Trajectory` across one loop run.

    Install via the ``hooks=`` list on :func:`composable_loop`. Optionally
    pair with a :class:`RecordingLLMBackend` - the recorder will stamp
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
        output_dir: str | Path | None = None,
        harness_snapshot: dict[str, Any] | None = None,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self.trajectory = Trajectory(
            run_id=uuid4().hex[:12],
            started_at=time.time(),
            steps=DiskBackedList(vars, lambda value: StepRecord(**value)),
        )
        self._recording_llm = recording_llm
        self._capture_context = capture_context
        self._tracer: Tracer | None = tracer if tracer is not None else Tracer()
        self._loop_span: Span | None = None
        self._pending_context: str = ""
        self._pending_llm_start: int = 0
        self._step_start_time: float = 0.0
        self._output_dir: Path | None = Path(output_dir) if output_dir is not None else None
        self._harness_snapshot = dict(harness_snapshot) if harness_snapshot is not None else None
        self._loop_ctx: Any = None
        self._harness_epochs: list[dict[str, Any]] = []
        self._last_epoch_key: tuple[Any, ...] | None = None
        self._last_exec_objs: dict[str, Any] = {}
        self._exec_revisions: dict[str, int] = {}
        self._last_tool_sigs: dict[str, tuple[Any, ...]] = {}
        # When set, every JSON document written to disk is redacted
        # through this callable. The recorder still captures the
        # un-redacted in-memory trajectory so callers that want raw
        # data can read ``self.trajectory`` directly; only the
        # on-disk artefact is scrubbed.
        self._redact = redact

    # ── hook methods ────────────────────────────────────────────

    def bind(self, loop_ctx: Any) -> None:
        self._loop_ctx = loop_ctx

    def _record_harness_epoch(self, step_num: int | None) -> None:
        if self._loop_ctx is None:
            return
        config = getattr(self._loop_ctx, "config", None)
        tools = getattr(self._loop_ctx, "tools", None)
        if config is None or tools is None or not hasattr(tools, "tool_view"):
            return
        sys_prompt = str(getattr(config, "system_prompt", "") or "")
        sys_sha = hashlib.sha256(sys_prompt.encode("utf-8")).hexdigest()
        view = tools.tool_view()
        raw_specs = getattr(tools, "tool_specs", {}) or {}
        tool_origins = {
            name: spec.origin
            for name, spec in raw_specs.items()
            if getattr(spec, "origin", None) is not None
        }
        curr_exec_objs: dict[str, Any] = {}
        curr_tool_sigs: dict[str, tuple[Any, ...]] = {}
        for name, spec in raw_specs.items():
            exec_obj = getattr(spec, "execute", spec)
            curr_exec_objs[name] = exec_obj
            if name not in self._exec_revisions:
                self._exec_revisions[name] = 0
            elif self._last_exec_objs.get(name) is not exec_obj:
                self._exec_revisions[name] += 1
            schema_json = json.dumps(
                spec.to_api_schema() if hasattr(spec, "to_api_schema") else {},
                sort_keys=True,
                default=str,
            )
            curr_tool_sigs[name] = (
                getattr(spec, "origin", None),
                spec.spec_text() if hasattr(spec, "spec_text") else "",
                schema_json,
                self._exec_revisions[name],
            )
        for stale_name in list(self._exec_revisions):
            if stale_name not in curr_exec_objs:
                self._exec_revisions.pop(stale_name, None)
        epoch_key = (
            sys_sha,
            view.version,
            tuple(sorted(curr_tool_sigs.items())),
        )
        if epoch_key == self._last_epoch_key:
            self._last_exec_objs = curr_exec_objs
            return
        curr_tools = list(view.names)
        entry: dict[str, Any] = {
            "epoch": len(self._harness_epochs),
            "step_num": step_num if step_num is not None else 1,
            "system_prompt_sha256": sys_sha,
            "tool_view_version": view.version,
            "tools": curr_tools,
        }
        if tool_origins:
            entry["tool_origins"] = tool_origins
        if self._harness_epochs:
            prev_list = self._harness_epochs[-1]["tools"]
            prev_set = set(prev_list)
            curr_set = set(curr_tools)
            entry["added_tools"] = [name for name in curr_tools if name not in prev_set]
            entry["removed_tools"] = [name for name in prev_list if name not in curr_set]
            updated = [
                name
                for name in curr_tools
                if name in prev_set and self._last_tool_sigs.get(name) != curr_tool_sigs.get(name)
            ]
            if updated:
                entry["updated_tools"] = updated
        self._harness_epochs.append(entry)
        self._last_epoch_key = epoch_key
        self._last_exec_objs = curr_exec_objs
        self._last_tool_sigs = curr_tool_sigs

    def pre_loop(self, state: Any, session_log: Any, context: Any) -> None:
        if self._loop_span is None and self._tracer is not None:
            self._loop_span = self._tracer.start_span("loop.run")
        # Capture the task from state so trajectory.json includes it
        # for round-tripping via EvalContext.from_trajectory_dir().
        task = getattr(state, "task", None)
        if task is not None and isinstance(task, dict):
            self.trajectory.task = dict(task)

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
            except Exception:  # pragma: no cover - defensive
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
        from looplet._backend_contract import json_snapshot

        duration_ms = (time.time() - self._step_start_time) * 1000.0
        pretty = ""
        if state is not None and hasattr(state, "steps") and state.steps:
            last = state.steps[-1]
            if hasattr(last, "pretty"):
                pretty = last.pretty()
        llm_indices: list[int] = []
        if self._recording_llm is not None:
            llm_indices = [
                call.index
                for call in self._recording_llm.calls[self._pending_llm_start :]
                if call.scope is None or call.step_num in (None, step_num)
            ]
        tc_dict = (
            tool_call.to_dict()
            if hasattr(tool_call, "to_dict")
            else {
                "tool": getattr(tool_call, "tool", "?"),
                "args": getattr(tool_call, "args", {}),
            }
        )
        tr_dict = (
            tool_result.to_full_dict()
            if hasattr(tool_result, "to_full_dict")
            else tool_result.to_dict()
            if hasattr(tool_result, "to_dict")
            else {
                "tool": getattr(tool_result, "tool", "?"),
                "error": getattr(tool_result, "error", None),
            }
        )
        self.trajectory.steps.append(
            StepRecord(
                step_num=step_num,
                timestamp=self._step_start_time,
                duration_ms=duration_ms,
                pretty=pretty,
                tool_call=json_snapshot(tc_dict),
                tool_result=json_snapshot(tr_dict),
                context_before=self._pending_context,
                llm_call_indices=llm_indices,
            )
        )
        self._pending_context = ""
        return None

    def on_event(self, payload: Any) -> None:
        from looplet.events import LifecycleEvent  # noqa: PLC0415

        event = getattr(payload, "event", None)
        if event == LifecycleEvent.PRE_LLM_CALL:
            extra = getattr(payload, "extra", None)
            if isinstance(extra, dict) and "subagent_id" in extra:
                return
            self._record_harness_epoch(getattr(payload, "step_num", None))
            return
        if event == LifecycleEvent.SUBAGENT_STOP:
            extra = getattr(payload, "extra", None)
            extra_dict = dict(extra) if isinstance(extra, dict) else {}
            subagents = self.trajectory.metadata.setdefault("subagents", [])
            if isinstance(subagents, list):
                subagents.append(
                    {
                        "subagent_id": getattr(payload, "subagent_id", None)
                        or extra_dict.get("subagent_id"),
                        "parent_id": extra_dict.get("parent_id"),
                        "termination_reason": getattr(payload, "termination_reason", None),
                        "step_count": extra_dict.get("step_count", 0),
                        "llm_calls": extra_dict.get("llm_calls", 0),
                        "recovered": extra_dict.get("recovered", False),
                    }
                )

    def on_loop_end(self, state: Any, session_log: Any, context: Any, llm: Any) -> int:
        from looplet._backend_contract import json_snapshot

        self.trajectory.ended_at = time.time()
        if session_log is not None and hasattr(session_log, "render"):
            try:
                self.trajectory.session_log_text = str(session_log.render() or "")
            except Exception:  # noqa: BLE001
                logger.warning("Failed to render session log for trajectory", exc_info=True)
        if self._harness_snapshot is not None:
            self.trajectory.metadata["harness_snapshot"] = dict(self._harness_snapshot)
        if self._recording_llm is not None:
            if isinstance(self._recording_llm.calls, DiskBackedList):
                self.trajectory.llm_calls = self._recording_llm.calls.clone()
            else:
                self.trajectory.llm_calls = DiskBackedList(
                    vars, lambda value: LLMCall(**value), self._recording_llm.calls
                )
        if self._tracer is not None and self._loop_span is not None:
            self._tracer.end_span(self._loop_span)
            self.trajectory.spans = list(self._tracer.root_spans)
        # Refresh tool_call/tool_result ``metadata`` on already-captured
        # StepRecords from the live ``state.steps`` objects. Hooks that
        # run *after* TrajectoryRecorder in the post_dispatch chain can
        # mutate ``tool_call.metadata`` / ``tool_result.metadata`` (which
        # the docstrings advertise as the documented annotation point);
        # without this sweep, those mutations are silently lost when the
        # recorder snapshotted via ``to_dict()`` before the tagger ran.
        if state is not None and hasattr(state, "steps"):
            live_steps = iter(state.steps)
            live = next(live_steps, None)
            for index, sr in enumerate(self.trajectory.steps):
                while live is not None and getattr(live, "number", 0) < sr.step_num:
                    live = next(live_steps, None)
                if live is None or getattr(live, "number", None) != sr.step_num:
                    continue
                live_tc = getattr(live, "tool_call", None)
                live_tr = getattr(live, "tool_result", None)
                live_tc_meta = getattr(live_tc, "metadata", None) if live_tc is not None else None
                live_tr_meta = getattr(live_tr, "metadata", None) if live_tr is not None else None
                live_step_meta = getattr(live, "metadata", None)
                if live_tc_meta:
                    sr.tool_call["metadata"] = json_snapshot(live_tc_meta)
                if live_tr_meta:
                    sr.tool_result["metadata"] = json_snapshot(live_tr_meta)
                if live_step_meta:
                    sr.metadata = json_snapshot(live_step_meta)
                self.trajectory.steps[index] = sr
        # Sweep ``state.steps`` for any Step that was yielded but not
        # routed through ``post_dispatch`` - notably the ``done`` step,
        # which the loop handles on its own termination path.
        captured_nums = {record.step_num for record in self.trajectory.steps}
        if state is not None and hasattr(state, "steps"):
            for st in state.steps:
                num = getattr(st, "number", None)
                if num is None or num in captured_nums:
                    continue
                tc = getattr(st, "tool_call", None)
                tr = getattr(st, "tool_result", None)
                tc_dict = tc.to_dict() if tc is not None and hasattr(tc, "to_dict") else {}
                tr_dict = (
                    tr.to_full_dict()
                    if tr is not None and hasattr(tr, "to_full_dict")
                    else tr.to_dict()
                    if tr is not None and hasattr(tr, "to_dict")
                    else {}
                )
                pretty = st.pretty() if hasattr(st, "pretty") else ""
                llm_indices = []
                if self._recording_llm is not None:
                    call_id = getattr(tc, "call_id", None)
                    for call in self._recording_llm.calls:
                        native_match = (
                            call_id
                            and isinstance(call.response, list)
                            and any(
                                isinstance(block, dict) and block.get("id") == call_id
                                for block in call.response
                            )
                        )
                        if call.step_num == num or native_match:
                            llm_indices.append(call.index)
                    if not llm_indices and st is state.steps[-1]:
                        llm_indices = [
                            call.index
                            for call in self._recording_llm.calls[self._pending_llm_start :]
                            if call.scope is None or call.step_num in (None, num)
                        ]
                self.trajectory.steps.append(
                    StepRecord(
                        step_num=num,
                        timestamp=0.0,
                        duration_ms=0.0,
                        pretty=pretty,
                        tool_call=json_snapshot(tc_dict),
                        tool_result=json_snapshot(tr_dict),
                        context_before="",
                        llm_call_indices=llm_indices,
                        metadata=json_snapshot(getattr(st, "metadata", {}) or {}),
                    )
                )
        if any(
            previous.step_num > current.step_num
            for previous, current in pairwise(self.trajectory.steps)
        ):
            if isinstance(self.trajectory.steps, DiskBackedList):
                self.trajectory.steps.sort(key=lambda step: step.step_num)
            else:
                self.trajectory.steps[:] = sorted(
                    self.trajectory.steps, key=lambda step: step.step_num
                )
        # The loop records the authoritative reason before on_loop_end,
        # including custom primary/secondary terminal sentinels. Keep the
        # old inference only for callers that invoke this hook manually.
        state_stop_reason = getattr(state, "_stop_reason", None) if state is not None else None
        if state_stop_reason:
            self.trajectory.termination_reason = str(state_stop_reason)
        elif self.trajectory.steps:
            last_tool = self.trajectory.steps[-1].tool_call.get("tool", "")
            if last_tool == "done":
                self.trajectory.termination_reason = "done"
            elif self.trajectory.steps[-1].tool_result.get("error"):
                self.trajectory.termination_reason = "error"
            else:
                self.trajectory.termination_reason = "max_steps_or_stop"
        else:
            self.trajectory.termination_reason = "no_steps"

        if state is not None:
            run_stats = getattr(state, "_looplet_run_stats", {})
            if isinstance(run_stats, dict):
                self.trajectory.metadata["run_stats"] = json_snapshot(run_stats)
            for key in ("run_status", "run_phase"):
                value = getattr(state, key, None)
                if value is not None:
                    self.trajectory.metadata[key] = getattr(value, "value", str(value))
            self.trajectory.metadata["termination_reason"] = self.trajectory.termination_reason
            for key in (
                "run_envelope",
                "policy_decisions",
                "looplet_assistant_turns",
                "usage_total",
                "replay",
                "replay_policy_override",
                "harness_reloads",
            ):
                value = getattr(state, key, None)
                if value is None:
                    state_metadata = getattr(state, "metadata", None)
                    if isinstance(state_metadata, dict):
                        value = state_metadata.get(key)
                if value is not None:
                    if hasattr(value, "to_dict"):
                        value = value.to_dict()
                    self.trajectory.metadata[key] = json_snapshot(value)
        if len(self._harness_epochs) > 1:
            self.trajectory.metadata["harness_epochs"] = json_snapshot(self._harness_epochs)

        # Auto-save when output_dir was specified at construction time.
        if self._output_dir is not None:
            self.save(self._output_dir)

        return 0

    # ── summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line human-readable summary of the run.

        Example output::

            5 steps, 2 failed, done, 12.3s

        Call after the loop finishes (or mid-run for progress).
        """
        t = self.trajectory
        n = len(t.steps)
        fails = sum(1 for s in t.steps if s.tool_result.get("error"))
        elapsed = (t.ended_at - t.started_at) if t.ended_at else 0
        parts = [f"{n} steps"]
        if fails:
            parts.append(f"{fails} failed")
        parts.append(f"{t.termination_reason or 'running'}")
        if elapsed > 0:
            parts.append(f"{elapsed:.1f}s")
        return ", ".join(parts)

    # ── disk IO ─────────────────────────────────────────────────

    def save(self, directory: str | Path, *, publish_descriptor: bool = True) -> Path:
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
        if publish_descriptor:
            begin_artifact_write(root)
        _clear_call_artifacts(root)
        trajectory = self.trajectory
        fields = {
            "run_id": trajectory.run_id,
            "started_at": trajectory.started_at,
            "ended_at": trajectory.ended_at,
            "task": trajectory.task,
            "termination_reason": trajectory.termination_reason,
            "session_log_text": trajectory.session_log_text,
            "metadata": trajectory.metadata,
            "step_count": len(trajectory.steps),
            "llm_call_count": len(trajectory.llm_calls),
            "spans": [s.to_dict() for s in trajectory.spans],
        }
        shared_redactor = (
            self._recording_llm is not None and self._recording_llm._redact is self._redact
        )
        with (root / "trajectory.json").open("w", encoding="utf-8") as stream:
            stream.write("{")
            for name, value in fields.items():
                stream.write(json.dumps(name) + ": " + _recorded_json(value, self._redact) + ",\n")
            for name, records in (("steps", trajectory.steps), ("llm_calls", trajectory.llm_calls)):
                stream.write(json.dumps(name) + ": [")
                for index, record in enumerate(records):
                    if index:
                        stream.write(",\n")
                    if name == "llm_calls":
                        entry = record.to_dict()
                        if shared_redactor:
                            if self._redact is not None and entry["metadata"]:
                                entry["metadata"] = _redact_value(entry["metadata"], self._redact)
                            stream.write(json.dumps(entry, default=str))
                        else:
                            redacted_entry = (
                                _redact_model_record(entry, self._redact)
                                if self._redact is not None
                                else entry
                            )
                            stream.write(json.dumps(redacted_entry, default=str))
                    else:
                        stream.write(_recorded_json(record.to_dict(), self._redact))
                stream.write("],\n" if name == "steps" else "]\n")
            stream.write("}\n")
        steps_dir = root / "steps"
        steps_dir.mkdir(exist_ok=True)
        for path in steps_dir.iterdir():
            if path.is_file() and _STEP_ARTIFACT.fullmatch(path.name):
                path.unlink()
        for s in self.trajectory.steps:
            step_text = _recorded_json(s.to_dict(), self._redact)
            (steps_dir / f"step_{s.step_num:02d}.json").write_text(
                step_text,
                encoding="utf-8",
            )
        if self._recording_llm is not None:
            self._recording_llm.save(root, publish_descriptor=False)
        components = ["trajectory"]
        if self._recording_llm is not None:
            components.append("model_calls")
        if publish_descriptor:
            write_artifact_descriptor(root, kind="provenance", components=components)
        return root


# ── Unified sink ────────────────────────────────────────────────────


class ProvenanceSink:
    """3-line facade: wrap the LLM, add the hook, flush on exit.

    Omitting ``dir`` selects a unique path under ``.looplet/traces``.
    Existing evidence requires ``overwrite=True``. Repeated flushes of one
    run remain supported. Automatic paths change after ``reset()``.

    Example::

        sink = ProvenanceSink(dir="traces/run_1/")
        llm = sink.wrap_llm(AnthropicBackend(...))
        for step in composable_loop(llm=llm, hooks=[sink.trajectory_hook()], ...):
            ...
        sink.flush()

    With ``redact``, prompt strings are scrubbed before the provider call
    as well as in saved evidence. Use ``redact_upstream=False`` only when
    the provider may receive the original input and scrubbing is record-only.

    The sink is safe to reuse across runs - call :meth:`reset` between
    them, or construct a fresh sink per run (cheaper and clearer).
    ``metadata`` is copied into the trajectory when its hook is created.
    :meth:`wrap_llm` and :meth:`trajectory_hook` may be called in either
    order before execution.
    """

    def __init__(
        self,
        dir: str | Path | None = None,
        *,
        overwrite: bool = False,
        max_chars_per_call: int = 200_000,
        redact: Callable[[str], str] | None = None,
        redact_upstream: bool = True,
        capture_context: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._automatic_dir = dir is None
        self._dir = Path(dir) if dir is not None else Path(".looplet/traces") / uuid4().hex
        self._overwrite = overwrite
        self._flushed = False
        self._owns_attempts = False
        self._check_directory()
        self._max_chars = max_chars_per_call
        self._redact = redact
        self._redact_upstream = redact_upstream
        self._capture_context = capture_context
        self._metadata = dict(metadata or {})
        self._recording_llm: _RecordingBase | None = None
        self._hook: TrajectoryRecorder | None = None

    def _check_directory(self) -> None:
        if self._overwrite or self._flushed or not self._dir.is_dir():
            return
        if any(
            path.name in {"trajectory.json", "manifest.jsonl"}
            or _CALL_ARTIFACT.fullmatch(path.name)
            or (
                path.name == "model-attempts"
                and not self._owns_attempts
                and any(path.glob("*.intent.json"))
            )
            for path in self._dir.iterdir()
        ):
            raise FileExistsError(
                f"Trace already exists in {self._dir}. Use a new directory, "
                "omit dir for a unique path, or set overwrite=True to replace it."
            )

    def wrap_llm(self, backend: Any, *, async_: bool | None = None) -> Any:
        """Wrap ``backend`` in a recording backend and stash a reference.

        A previously created trajectory hook is bound to this wrapper.
        Pass ``async_=True`` to force the async variant; otherwise the
        sink inspects both generation methods. A coroutine function on
        either method selects the async wrapper, including mixed backends.
        """
        import inspect

        if async_ is None:
            async_ = any(
                inspect.iscoroutinefunction(getattr(backend, method, None))
                for method in ("generate", "generate_with_tools")
            )
        cls: type[_RecordingBase] = AsyncRecordingLLMBackend if async_ else RecordingLLMBackend
        self._owns_attempts = True
        self._recording_llm = cls(
            backend,
            max_chars_per_call=self._max_chars,
            redact=self._redact,
            redact_upstream=self._redact_upstream,
            attempt_directory=self._dir / "model-attempts",
        )
        if self._hook is not None:
            self._hook._recording_llm = self._recording_llm
        return self._recording_llm

    def trajectory_hook(self) -> TrajectoryRecorder:
        """Create (or return the cached) :class:`TrajectoryRecorder`."""
        if self._hook is None:
            self._hook = TrajectoryRecorder(
                recording_llm=self._recording_llm,
                capture_context=self._capture_context,
                redact=self._redact,
            )
            self._hook.trajectory.metadata.update(self._metadata)
        return self._hook

    def flush(self) -> Path:
        """Write everything to disk and return the directory path."""
        self._check_directory()
        if self._hook is not None:
            output = self._hook.save(self._dir)
        elif self._recording_llm is not None:
            output = self._recording_llm.save(self._dir)
        else:
            self._dir.mkdir(parents=True, exist_ok=True)
            output = self._dir
        self._flushed = True
        return output

    def reset(self) -> None:
        if self._recording_llm is not None:
            self._recording_llm.reset()
        self._hook = None
        self._flushed = False
        self._owns_attempts = False
        if self._automatic_dir:
            self._dir = Path(".looplet/traces") / uuid4().hex
            if self._recording_llm is not None:
                self._recording_llm._attempt_directory = self._dir / "model-attempts"
                self._owns_attempts = True


# ── Replay ──────────────────────────────────────────────────────────


class _ReplayLLMBackend:
    """Internal LLM backend that selects captured responses by request identity.

    Used by :func:`replay_loop`. Not exported in ``__init__.__all__`` -
    most users should call :func:`replay_loop` instead. Surface it via
    ``from looplet.provenance import _ReplayLLMBackend`` if you need
    low-level control.

    Exhaustion and protocol mismatches raise :class:`TerminalLLMError`.
    Recorded ordinary failures raise :class:`RuntimeError`.
    Public replay loops record these failures instead of raising them.
    """

    def __init__(self, calls: list[dict[str, Any]], *, source_trace: str | None = None) -> None:
        self._calls = calls
        self._source_trace = source_trace
        self._index = 0
        self._call_identity = _ModelCallIdentity()
        self._calls_by_id = (
            {call["request_id"]: call for call in calls}
            if all("request_id" in call for call in calls)
            else None
        )
        self._lock = Lock()
        # Surface ``generate_with_tools`` only if any recorded call used
        # it - keeps ``hasattr`` detection honest for the loop.
        if any(c.get("method") == "generate_with_tools" for c in calls):
            self.generate_with_tools = self._generate_with_tools_impl

    def _scope_calls(self, scope: str, *, step_num: int | None = None) -> Any:
        return self._call_identity.scope(scope, step_num)

    def _next(self, expected_method: str) -> dict[str, Any]:
        with self._lock:
            return self._next_call(expected_method)

    def _next_call(self, expected_method: str) -> dict[str, Any]:
        if self._index >= len(self._calls):
            raise TerminalLLMError(
                f"replay exhausted: loop asked for call #{self._index + 1} "
                f"but only {len(self._calls)} were recorded. Reduce the "
                f"loop's max_steps or re-record the trace.",
                termination_reason="replay_exhausted",
            )
        if self._calls_by_id is None:
            call = self._calls[self._index]
        else:
            request_id = self._call_identity.next()
            call = self._calls_by_id.pop(request_id, None)
            if call is None:
                raise TerminalLLMError(
                    f"replay has no recorded request for identity {request_id!r}",
                    termination_reason="replay_exhausted",
                )
        if call.get("method") != expected_method:
            raise TerminalLLMError(
                f"replay mismatch at call #{self._index}: recorded "
                f"method={call.get('method')!r} but loop called "
                f"{expected_method!r}. Your tool registry or hooks may "
                f"have diverged from the recorded run.",
                termination_reason="llm_error",
            )
        self._index += 1
        self.last_stop_reason = call.get("stop_reason")
        self.last_usage = call.get("usage") or {}
        self.last_request = call.get("provider_request")
        self.last_response = call.get("provider_response")
        if self.last_request is not None:
            _record_request_capture(self.last_request)
        _record_response_capture(
            self.last_response,
            self.last_stop_reason,
            self.last_usage,
            replay_source={
                "trace": self._source_trace,
                "call_index": call.get("index", self._index - 1),
                "request_id": call.get("request_id"),
                "attempt_id": call.get("attempt_id"),
            },
        )
        return call

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> str:
        call = self._next("generate")
        self._raise_recorded_error(call)
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
        temperature: float | None = None,
    ) -> list[dict[str, Any]]:
        call = self._next("generate_with_tools")
        self._raise_recorded_error(call)
        response = call.get("response", [])
        if isinstance(response, list):
            return response
        # Defensive: recorded as a JSON string, decode back.
        try:
            decoded = json.loads(response) if isinstance(response, str) else []
            return decoded if isinstance(decoded, list) else []
        except Exception:
            return []

    @staticmethod
    def _raise_recorded_error(call: dict[str, Any]) -> None:
        error = call.get("error")
        if error:
            terminal = call.get("terminal_error")
            if terminal is not None:
                raise TerminalLLMError(
                    f"replayed LLM call failed: {error}",
                    termination_reason=terminal["termination_reason"],
                )
            raise RuntimeError(f"replayed LLM call failed: {error}")


def _load_trace_calls(trace_dir: Path) -> list[dict[str, Any]]:
    return _load_trace_evidence(trace_dir)[0]


def _load_trace_evidence(
    trace_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Read the call stream from ``trace_dir``.

    Prefers ``manifest.jsonl`` (which contains structured metadata per
    call) and joins in the body text from ``call_NN_response.txt``. If
    the manifest is missing, falls back to the response files alone
    (every call treated as ``generate`` returning the file contents).
    """
    descriptor, entries = _read_artifact(
        trace_dir,
        expected_kinds=("eval_run", "provenance"),
    )
    if descriptor is not None and "model_calls" not in descriptor["components"]:
        raise ValueError(f"Saved artifact in {trace_dir} does not declare a model_calls component")
    manifest = _artifact_path(trace_dir, "manifest.jsonl")
    calls: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    if manifest.exists():
        if entries is None:
            entries = _validate_model_call_files(trace_dir, manifest, require_projections=False)
        for line_number, entry in enumerate(entries, start=1):
            idx = entry["index"]
            method = entry.get("method", "generate")
            capture = entry.get("capture")
            if entry.get("scope") is not None and not (
                isinstance(capture, dict) and capture.get("request_id") is not None
            ):
                raise ValueError(
                    "Recorded helper calls lack request identities. Re-record the trace."
                )
            if capture is not None and capture != {}:
                if (
                    not isinstance(capture, dict)
                    or capture.get("schema")
                    not in {"looplet.model-call.v1", "looplet.model-call.v2"}
                    or not isinstance(capture.get("request"), dict)
                    or capture["request"].get("method") != method
                    or "response" not in capture
                ):
                    raise ValueError(f"invalid structured capture in {manifest} line {line_number}")
                terminal = capture.get("terminal_error")
                if terminal is not None and (
                    not isinstance(terminal, dict)
                    or set(terminal) != {"schema", "termination_reason"}
                    or terminal.get("schema") != "looplet.terminal-error.v1"
                    or not isinstance(terminal.get("termination_reason"), str)
                    or not terminal["termination_reason"]
                    or not isinstance(capture.get("error"), str)
                    or not capture["error"]
                ):
                    raise ValueError(
                        f"invalid terminal error capture in {manifest} line {line_number}"
                    )
                if capture.get("schema") == "looplet.model-call.v2":
                    from looplet.assistant_turns import TURN_STATE_SCHEMA

                    execution = capture.get("execution")
                    if (
                        not isinstance(execution, dict)
                        or execution.get("required_semantics") != TURN_STATE_SCHEMA
                    ):
                        raise ValueError("unsupported required model-call semantics")
                response = capture["response"]
                valid_response = (
                    isinstance(response, str)
                    if method == "generate"
                    else isinstance(response, list)
                    and all(isinstance(block, dict) for block in response)
                )
                if not valid_response:
                    raise ValueError(
                        f"invalid structured response in {manifest} line {line_number}"
                    )
                request_id = capture.get("request_id")
                if request_id is not None:
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError(
                            f"invalid request identity in {manifest} line {line_number}"
                        )
                    if request_id in request_ids:
                        raise ValueError(
                            f"duplicate request identity in {manifest} line {line_number}"
                        )
                    request_ids.add(request_id)
                calls.append({**capture, "index": idx, "method": method})
                continue
            response_txt = _read_call_body(trace_dir, idx)
            if response_txt is None:
                raise FileNotFoundError(
                    f"missing recorded response for {manifest} line {line_number}: "
                    f"call_{idx:02d}_response.txt"
                )
            if method == "generate_with_tools":
                # The on-disk .txt file contains a JSON content-block
                # dump after the "## RESPONSE (content blocks)" header.
                try:
                    response: Any = _extract_content_blocks(response_txt)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid recorded response for {manifest} line {line_number}: {exc}"
                    ) from exc
            else:
                response = _extract_response_text(response_txt)
            calls.append(
                {
                    "index": idx,
                    "method": method,
                    "response": response,
                    "error": entry.get("error"),
                }
            )
    else:
        # Fallback: scan call_NN_response.txt files.
        response_files: list[tuple[int, Path]] = []
        for candidate in trace_dir.iterdir():
            match = _CALL_RESPONSE_ARTIFACT.fullmatch(candidate.name)
            if match is not None:
                path = _artifact_path(trace_dir, candidate.name)
                if path.is_file():
                    response_files.append((int(match.group(1)), path))
        response_files.sort(key=lambda item: item[0])
        for expected_index, (index, path) in enumerate(response_files):
            if index != expected_index:
                raise ValueError(
                    f"non-contiguous recorded responses in {trace_dir}: "
                    f"expected index {expected_index}, got {index}"
                )
            body = path.read_text(encoding="utf-8")
            calls.append(
                {
                    "index": index,
                    "method": "generate",
                    "response": _extract_response_text(body),
                    "error": _extract_recorded_error(body),
                }
            )
    if not calls:
        raise FileNotFoundError(
            f"no recorded calls found in {trace_dir} - expected "
            f"manifest.jsonl or call_NN_response.txt files"
        )
    if any("request_id" in call for call in calls) and not all(
        "request_id" in call for call in calls
    ):
        raise ValueError("Recorded calls mix identified and positional requests")
    return calls, descriptor


def _read_call_body(trace_dir: Path, idx: int | None) -> str | None:
    if idx is None:
        return None
    path = _artifact_path(trace_dir, f"call_{idx:02d}_response.txt")
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


def _extract_recorded_error(body: str | None) -> str | None:
    if not body or "## ERROR\n" not in body:
        return None
    error = body.split("## ERROR\n", 1)[1]
    if "\n## RESPONSE" in error:
        error = error.split("\n## RESPONSE", 1)[0]
    return error.strip() or None


def _extract_content_blocks(body: str | None) -> list[dict[str, Any]]:
    if not body:
        raise ValueError("content-block response is empty")
    marker = "## RESPONSE (content blocks)\n"
    if marker not in body:
        raise ValueError("content-block response header is missing")
    payload = body.split(marker, 1)[1].strip()
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"content blocks are not valid JSON: {exc}") from exc
    if not isinstance(decoded, list) or any(not isinstance(block, dict) for block in decoded):
        raise ValueError("content blocks must be an array of objects")
    return decoded


class _RecordedToolReplayHook:
    """Intercept selected tool calls during replay and return their recorded ToolResult."""

    def __init__(self, recorded_steps: list[dict[str, Any]], stub_tools: set[str] | bool) -> None:
        self._stub_tools = stub_tools
        self._by_tool_queue: dict[str, deque[tuple[int, dict[str, Any]]]] = {}
        self._by_tool_call_id: dict[tuple[str, str], deque[tuple[int, dict[str, Any]]]] = {}
        self._consumed_ordinals: set[int] = set()
        for ordinal, step in enumerate(recorded_steps):
            tc = step.get("tool_call") or {}
            tr = step.get("tool_result") or {}
            tname = tc.get("tool") or tr.get("tool")
            if isinstance(tname, str) and tname:
                pair = (ordinal, step)
                self._by_tool_queue.setdefault(tname, deque()).append(pair)
                cid = tc.get("call_id") or tr.get("call_id")
                if isinstance(cid, str) and cid:
                    self._by_tool_call_id.setdefault((tname, cid), deque()).append(pair)

    def _should_stub(self, tool_name: str) -> bool:
        if self._stub_tools is True:
            return True
        if isinstance(self._stub_tools, set):
            return tool_name in self._stub_tools
        return False

    def pre_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        step_num: int,
    ) -> Any:
        from looplet.types import ToolResult  # noqa: PLC0415

        if not self._should_stub(tool_call.tool):
            return None
        tr_dict = None
        selected: tuple[int, dict[str, Any]] | None = None
        if tool_call.call_id:
            id_queue = self._by_tool_call_id.get((tool_call.tool, tool_call.call_id))
            while id_queue:
                candidate = id_queue.popleft()
                if candidate[0] not in self._consumed_ordinals:
                    selected = candidate
                    break
        if selected is None:
            queue = self._by_tool_queue.get(tool_call.tool)
            while queue:
                candidate = queue.popleft()
                if candidate[0] not in self._consumed_ordinals:
                    selected = candidate
                    break
        if selected is not None:
            self._consumed_ordinals.add(selected[0])
            tr_dict = selected[1].get("tool_result")
        if tr_dict is None:
            raise ValueError(f"No recorded result for stubbed tool {tool_call.tool!r}")
        return ToolResult(
            tool=tool_call.tool,
            args_summary=str(tr_dict.get("args", "")),
            data=copy.deepcopy(tr_dict.get("data")),
            error=tr_dict.get("error"),
            duration_ms=float(tr_dict.get("duration_ms", 0.0)),
            result_key=tr_dict.get("result_key"),
            call_id=tool_call.call_id,
            warnings=list(tr_dict.get("warnings") or []),
            metadata={**dict(tr_dict.get("metadata") or {}), "replayed_from_trace": True},
        )


def replay_loop(
    trace_dir: str | Path,
    *,
    tools: Any,
    state: Any | None = None,
    hooks: list[Any] | None = None,
    config: Any | None = None,
    task: dict[str, Any] | None = None,
    recording_sink: ProvenanceSink | None = None,
    max_steps: int | None = None,
    assistant_turn_policy: Callable[..., Any] | None = None,
    policy_override_name: str | None = None,
    stub_tools: Sequence[str] | bool = (),
) -> Any:
    """Replay a captured trace through a fresh agent loop.

    Reads the recorded LLM calls from ``trace_dir`` (as written by
    :meth:`RecordingLLMBackend.save` or :meth:`ProvenanceSink.flush`)
    and yields :class:`Step`\\s from :func:`composable_loop` using a
    replay backend that selects recorded responses by invocation address.

    The user's tool registry, hooks, permission engine, and state are
    the ones passed in - this is what makes replay useful: change those
    and diff the step output without spending a dollar on the LLM.

    Args:
        trace_dir: Directory containing ``manifest.jsonl`` +
            ``call_NN_response.txt`` files.
        tools: Tool registry for the replay loop.
        state: Optional :class:`AgentState`. Defaults to a fresh
            :class:`DefaultState` sized to the replay budget.
        hooks: Optional synchronous hooks to install on the replay loop.
            Use :func:`async_replay_loop` for async hooks or tools.
        config: Optional :class:`LoopConfig` - if omitted, a default is
            constructed from the recorded tool-step count and protocol.
            Call-only captures count native tool calls or text responses.
        task: Optional task dict. Defaults to the saved trajectory task,
            or ``{}`` when none was saved. An explicit ``{}`` stays empty.
        recording_sink: Optional sink for fresh replay prompts, responses,
            and steps. Flush it after closing the replay.
        max_steps: Optional positive step limit. Overrides the recorded
            budget or a supplied configuration without mutating that configuration.
        stub_tools: Tool names whose saved results replace live execution,
            or ``True`` for all tools. A selected call with no recorded result
            raises ``ValueError`` rather than executing live.

    Yields:
        :class:`Step` objects from the replay.

    Exhaustion stops the run with status ``failed`` and termination reason
    ``replay_exhausted``. A protocol mismatch uses the normal model-call
    failure path and reason ``llm_error``. Neither raises to the caller.
    Pass ``state`` and inspect ``RunResult.from_state(state)`` after iteration.

    Raises:
        FileNotFoundError: ``trace_dir`` has no manifest or response files.
        ValueError: Saved replay defaults have an invalid shape or a selected
            tool stub has no recorded result.

    Example::

        from looplet.provenance import replay_loop

        for step in replay_loop("traces/run_1/", tools=my_tools):
            print(step.pretty())
    """
    from looplet.loop import composable_loop  # noqa: PLC0415

    evidence = _load_replay_evidence(
        trace_dir,
        load_trajectory=config is None or task is None or bool(stub_tools),
        max_steps=max_steps,
    )
    yield from composable_loop(
        **_prepare_replay(
            evidence,
            tools=tools,
            state=state,
            hooks=hooks,
            config=config,
            task=task,
            recording_sink=recording_sink,
            max_steps=max_steps,
            assistant_turn_policy=assistant_turn_policy,
            policy_override_name=policy_override_name,
            stub_tools=stub_tools,
        )
    )


async def async_replay_loop(
    trace_dir: str | Path,
    *,
    tools: Any,
    state: Any | None = None,
    hooks: list[Any] | None = None,
    config: Any | None = None,
    task: dict[str, Any] | None = None,
    recording_sink: ProvenanceSink | None = None,
    max_steps: int | None = None,
    assistant_turn_policy: Callable[..., Any] | None = None,
    policy_override_name: str | None = None,
    stub_tools: Sequence[str] | bool = (),
) -> Any:
    """Replay captured model responses through the async agent loop.

    Accepts the same arguments, saved defaults, and evidence formats as
    :func:`replay_loop`. Async hooks and tools are awaited by
    :func:`async_composable_loop`; synchronous hooks and tools also work.
    Model responses stay fixed while fresh tool and hook effects execute.
    No live model service is called.

    Yields :class:`Step` objects. Loading and validation errors match
    :func:`replay_loop`. Use ``contextlib.aclosing`` when stopping early
    so the underlying async loop is closed with the replay iterator.
    """
    from looplet.async_loop import async_composable_loop  # noqa: PLC0415
    from looplet.checkpoint import _run_persistence

    evidence = await _run_persistence(
        _load_replay_evidence,
        trace_dir,
        load_trajectory=config is None or task is None or bool(stub_tools),
        max_steps=max_steps,
    )
    arguments = _prepare_replay(
        evidence,
        tools=tools,
        state=state,
        hooks=hooks,
        config=config,
        task=task,
        recording_sink=recording_sink,
        max_steps=max_steps,
        assistant_turn_policy=assistant_turn_policy,
        policy_override_name=policy_override_name,
        stub_tools=stub_tools,
    )
    async with aclosing(async_composable_loop(**arguments)) as iterator:
        async for step in iterator:
            yield step


def _load_replay_evidence(
    trace_dir: str | Path,
    *,
    load_trajectory: bool,
    max_steps: int | None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace directory does not exist: {trace_path}")
    calls, descriptor = _load_trace_evidence(trace_path)
    requires_policy = any(
        call.get("execution", {}).get("assistant_policy_required") is True for call in calls
    )
    trajectory: dict[str, Any] = {}
    if load_trajectory or requires_policy:
        trajectory_path = _artifact_path(trace_path, "trajectory.json")
        if trajectory_path.is_file() and (
            descriptor is None or "trajectory" in descriptor["components"]
        ):
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            if not isinstance(trajectory, dict):
                raise ValueError("trajectory.json must contain an object")
    return trace_path, calls, trajectory


def _prepare_replay(
    evidence: tuple[Path, list[dict[str, Any]], dict[str, Any]],
    *,
    tools: Any,
    state: Any | None,
    hooks: list[Any] | None,
    config: Any | None,
    task: dict[str, Any] | None,
    recording_sink: ProvenanceSink | None = None,
    max_steps: int | None = None,
    assistant_turn_policy: Callable[..., Any] | None = None,
    policy_override_name: str | None = None,
    stub_tools: Sequence[str] | bool = (),
) -> dict[str, Any]:
    from looplet.loop import LoopConfig  # noqa: PLC0415
    from looplet.types import DefaultState  # noqa: PLC0415

    trace_path, calls, trajectory = evidence
    requires_policy = any(
        call.get("execution", {}).get("assistant_policy_required") is True for call in calls
    )
    backend = _ReplayLLMBackend(calls, source_trace=str(trace_path))
    if config is None:
        recorded_steps = trajectory.get("step_count")
        if recorded_steps is not None and (
            isinstance(recorded_steps, bool)
            or not isinstance(recorded_steps, int)
            or recorded_steps < 0
        ):
            raise ValueError("trajectory.step_count must be a non-negative integer")
        fallback_steps = sum(
            max(
                sum(block.get("type") == "tool_use" for block in call["response"])
                if call["method"] == "generate_with_tools"
                else 1,
                1,
            )
            for call in calls
        )
        metadata = trajectory.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("trajectory.metadata must be an object")
        run_stats = metadata.get("run_stats", {})
        if not isinstance(run_stats, dict):
            raise ValueError("trajectory.metadata.run_stats must be an object")
        turn_config = run_stats.get("model_turn_config", {})
        if not isinstance(turn_config, dict) or set(turn_config) - {
            "allow_text_completion",
            "max_turn_continuations",
            "max_tokens",
            "done_tool",
            "done_tools",
            "max_model_calls",
            "max_assistant_turns",
            "requires_assistant_turn_policy",
        }:
            raise ValueError("Invalid recorded model_turn_config")
        if (
            "allow_text_completion" in turn_config
            and type(turn_config["allow_text_completion"]) is not bool
        ):
            raise ValueError("Recorded allow_text_completion must be boolean")
        if (
            "requires_assistant_turn_policy" in turn_config
            and type(turn_config["requires_assistant_turn_policy"]) is not bool
        ):
            raise ValueError("Recorded requires_assistant_turn_policy must be boolean")
        for name, minimum in (
            ("max_tokens", 1),
            ("max_turn_continuations", 0),
            ("max_model_calls", 1),
            ("max_assistant_turns", 1),
        ):
            if name != "max_turn_continuations" and turn_config.get(name) is None:
                continue
            if name in turn_config and (
                type(turn_config[name]) is not int or turn_config[name] < minimum
            ):
                raise ValueError(f"Recorded {name} must be an integer >= {minimum}")
        if "done_tool" in turn_config and (
            not isinstance(turn_config["done_tool"], str) or not turn_config["done_tool"]
        ):
            raise ValueError("Recorded done_tool must be a nonempty string")
        done_tools = turn_config.get("done_tools", [])
        if not isinstance(done_tools, list) or any(
            not isinstance(name, str) or not name for name in done_tools
        ):
            raise ValueError("Recorded done_tools must be a list of nonempty strings")
        config = LoopConfig(
            max_steps=recorded_steps or fallback_steps,
            use_native_tools=calls[0]["method"] == "generate_with_tools",
            **turn_config,
        )
    replay_config: LoopConfig = config
    if max_steps is not None or replay_config.router is not None:
        replay_config = replace(
            replay_config,
            max_steps=replay_config.max_steps if max_steps is None else max_steps,
            router=None,
        )
    if task is None:
        recorded_task = trajectory.get("task")
        if recorded_task is not None and not isinstance(recorded_task, dict):
            raise ValueError("trajectory.task must be an object")
        task = recorded_task if recorded_task is not None else {}
    if state is None:
        state = DefaultState(max_steps=replay_config.max_steps)
    state.metadata["replay"] = {
        "schema": "looplet.replay.v1",
        "source_trace": str(trace_path),
        "usage_origin": "captured",
    }
    binding = assistant_turn_policy or replay_config.assistant_turn_policy
    if requires_policy or replay_config.requires_assistant_turn_policy:
        if binding is None:
            raise ValueError("assistant-turn replay requires an explicit host policy binding")
        from looplet.assistant_turns import TURN_STATE_KEY

        metadata = trajectory.get("metadata", {})
        turns = metadata.get(TURN_STATE_KEY, {}) if isinstance(metadata, dict) else {}
        recorded_decisions = turns.get("decisions") if isinstance(turns, dict) else None
        if not isinstance(recorded_decisions, list):
            raise ValueError("assistant-turn replay requires recorded decisions")
        binding = _checked_replay_policy(binding, recorded_decisions, policy_override_name)
    elif policy_override_name is not None:
        if binding is None:
            raise ValueError("a named replay override requires a host policy binding")
        binding = _checked_replay_policy(binding, [], policy_override_name)
    if binding is not None:
        replay_config = replace(replay_config, assistant_turn_policy=binding)
    if policy_override_name is not None:
        state.metadata["replay_policy_override"] = policy_override_name
    effective_hooks = list(hooks or ())
    if stub_tools:
        recorded_steps = trajectory.get("steps")
        if not isinstance(recorded_steps, list):
            raise ValueError("stub_tools replay requires recorded steps in trajectory.json")
        stub_set: set[str] | bool = (
            True if stub_tools is True else {str(name) for name in stub_tools}
        )
        effective_hooks.insert(0, _RecordedToolReplayHook(recorded_steps, stub_set))
    if recording_sink is not None:
        backend = recording_sink.wrap_llm(backend)
        recorder = recording_sink.trajectory_hook()
        if not any(hook is recorder for hook in effective_hooks):
            effective_hooks.append(recorder)
    return {
        "llm": backend,
        "tools": tools,
        "task": task,
        "state": state,
        "hooks": effective_hooks,
        "config": replay_config,
    }


def _checked_replay_policy(
    binding: Callable[..., Any], decisions: list[Any], override_name: str | None
) -> Callable[..., Any]:
    from looplet.assistant_turns import AssistantTurnDecision

    if override_name is not None and (
        not isinstance(override_name, str) or not override_name.strip() or len(override_name) > 128
    ):
        raise ValueError("policy_override_name must contain at most 128 nonempty characters")
    cursor = 0

    def check(decision: Any) -> AssistantTurnDecision:
        nonlocal cursor
        if not isinstance(decision, AssistantTurnDecision):
            raise TypeError("replay policy must return AssistantTurnDecision")
        if override_name is None:
            if cursor >= len(decisions):
                raise ValueError("recorded assistant decisions exhausted")
            expected = decisions[cursor]
            digest = hashlib.sha256(decision.feedback.encode("utf-8")).hexdigest()
            if not isinstance(expected, dict) or (
                expected.get("action"),
                expected.get("feedback_hash"),
            ) != (decision.action, digest):
                raise ValueError(
                    "assistant replay decision differs; supply an explicit named policy override"
                )
        cursor += 1
        return decision

    def policy(context: Any) -> Any:
        decision = binding(context)
        if inspect.isawaitable(decision):

            async def resolve() -> AssistantTurnDecision:
                return check(await decision)

            return resolve()
        return check(decision)

    return policy
