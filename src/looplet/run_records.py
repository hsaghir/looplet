"""Host-facing run and artifact records.

These records adapt existing lifecycle payloads and saved artifacts into one
small, JSON-safe envelope without changing the loop's hook contracts.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from looplet.events import EventPayload
from looplet.types import ErrorKind, RunEnvelope, RunPhase, RunResult, RunStatus, Step

__all__ = ["ArtifactRef", "RunEvent", "RunRecord", "event_from_payload", "run_diagnostics"]


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A durable artifact associated with a run."""

    artifact_id: str
    kind: str
    uri: str
    schema: str | None = None
    version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "uri": self.uri,
            "schema": self.schema,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(data["artifact_id"]),
            kind=str(data["kind"]),
            uri=str(data["uri"]),
            schema=str(data["schema"]) if data.get("schema") is not None else None,
            version=int(data["version"]) if data.get("version") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Immutable, JSON-safe event envelope for host observers and stores."""

    envelope: RunEnvelope | None
    sequence: int
    timestamp: float
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope": self.envelope.to_dict() if self.envelope is not None else None,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": dict(self.payload),
            "artifact_refs": [ref.to_dict() for ref in self.artifact_refs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunEvent":
        raw_envelope = data.get("envelope")
        return cls(
            envelope=RunEnvelope.from_dict(raw_envelope)
            if isinstance(raw_envelope, dict)
            else None,
            sequence=int(data.get("sequence", 0)),
            timestamp=float(data.get("timestamp", 0.0)),
            kind=str(data.get("kind", "unknown")),
            payload=dict(data.get("payload") or {}),
            artifact_refs=tuple(
                ArtifactRef.from_dict(value)
                for value in data.get("artifact_refs", [])
                if isinstance(value, dict)
            ),
        )


def event_from_payload(
    payload: EventPayload,
    *,
    envelope: RunEnvelope | None,
    sequence: int,
    timestamp: float | None = None,
) -> RunEvent:
    """Adapt an existing lifecycle payload without changing its schema."""
    kind = getattr(payload.event, "value", str(payload.event))
    return RunEvent(
        envelope=envelope,
        sequence=sequence,
        timestamp=time.time() if timestamp is None else timestamp,
        kind=kind,
        payload=payload.to_jsonable(),
    )


def run_diagnostics(
    result: RunResult | Mapping[str, Any],
    *,
    events: tuple[RunEvent, ...] = (),
) -> dict[str, Any]:
    """Summarize execution evidence without traversing content or grading data.

    Unknown measurements stay ``None``. Event counts are observations, not
    provider retry counts or proof of tool side effects. Completion is not
    correctness, and this view is not an anonymizer for structural labels.
    """
    if isinstance(result, RunResult):
        status, phase, reason = result.status, result.phase, result.termination_reason
        steps, metadata, envelope = result.steps, result.metadata, result.run_envelope
    else:
        status, phase = result.get("status"), result.get("phase")
        reason = result.get("termination_reason")
        steps = result.get("steps", ())
        metadata = result.get("metadata", {})
        envelope = result.get("run_envelope")
    if type(metadata) is not dict:
        metadata = {}
    if not isinstance(steps, (list, tuple)):
        steps = ()

    def number(value: Any) -> int | float | None:
        if type(value) is int and value >= 0:
            return value
        if type(value) is float and value >= 0 and math.isfinite(value):
            return value
        return None

    raw_usage = metadata.get("usage_total")
    stats = metadata.get("looplet_run_stats")
    if type(stats) is not dict:
        stats = {}
    usage = {
        name: number(raw_usage.get(name))
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_usd",
        )
        if type(raw_usage) is dict and number(raw_usage.get(name)) is not None
    }
    tool_errors: Counter[str] = Counter()
    for step in steps:
        if type(step) is Step:
            error = step.tool_result.error
            detail = step.tool_result.error_detail
            kind = detail.kind if detail is not None else None
        elif type(step) is dict and type(step.get("tool_result")) is dict:
            tool_result = step["tool_result"]
            error = tool_result.get("error")
            detail = tool_result.get("error_detail")
            kind = detail.get("kind") if type(detail) is dict else None
        else:
            continue
        if type(error) is str and error:
            kind_name = kind.value if isinstance(kind, ErrorKind) else kind
            tool_errors[
                kind_name
                if type(kind_name) is str and kind_name in ErrorKind._value2member_map_
                else "unknown"
            ] += 1
    status_name = status.value if isinstance(status, RunStatus) else status
    phase_name = phase.value if isinstance(phase, RunPhase) else phase
    run_id = (
        envelope.run_id
        if isinstance(envelope, RunEnvelope)
        else envelope.get("run_id")
        if type(envelope) is dict
        else metadata.get("run_id")
    )
    safe_reasons = {
        "done",
        "budget",
        "budget_exhausted",
        "max_steps",
        "cancelled",
        "deadline",
        "deadline_exceeded",
        "error",
        "llm_error",
        "hook_stop",
        "hook_requested_stop",
    }
    return {
        "schema": "looplet.run-diagnostics.v1",
        "run_id": run_id if type(run_id) is str else None,
        "status": status_name
        if type(status_name) is str and status_name in RunStatus._value2member_map_
        else "unknown",
        "phase": phase_name
        if type(phase_name) is str and phase_name in RunPhase._value2member_map_
        else "unknown",
        "termination_reason": reason
        if type(reason) is str and reason in safe_reasons
        else "custom"
        if reason is not None
        else None,
        "completed": status_name == "completed" if type(status_name) is str else False,
        "step_count": len(steps),
        "tool_error_count": sum(tool_errors.values()),
        "tool_error_kinds": dict(tool_errors),
        "llm_calls": number(stats.get("llm_calls", metadata.get("llm_calls"))),
        "duration_ms": number(metadata.get("runtime_duration_ms", stats.get("duration_ms"))),
        "usage": usage or None,
        "usage_known": bool(usage),
        "cost_usd": usage.get("cost_usd"),
        "run_error": (type(status_name) is str and status_name == "failed")
        or metadata.get("error") is not None
        or metadata.get("eval_run_error") is not None,
        "persistence_warning_count": len(metadata["persistence_warnings"])
        if type(metadata.get("persistence_warnings")) is list
        else 0,
        "event_counts": dict(Counter(event.kind for event in events)),
    }


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Durable index record for one logical run."""

    run_id: str
    envelope: RunEnvelope | None
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    result: Mapping[str, Any] | None = None
    checkpoint_keys: tuple[str, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()
    events: tuple[RunEvent, ...] = ()

    def diagnostics(self) -> dict[str, Any]:
        """Return the same redacted view for terminal and in-flight records."""
        payload = (
            dict(self.result)
            if self.result is not None
            else {
                "status": self.status,
                "metadata": dict(self.metadata),
            }
        )
        if payload.get("run_envelope") is None:
            payload["run_envelope"] = {"run_id": self.run_id}
        if self.events and not payload.get("phase"):
            payload["phase"] = self.events[-1].payload.get("run_phase")
        return run_diagnostics(payload, events=self.events)

    @classmethod
    def create(
        cls,
        envelope: RunEnvelope,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RunRecord":
        now = time.time()
        return cls(
            run_id=envelope.run_id,
            envelope=envelope,
            status="created",
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )

    def to_dict(self, *, include_events: bool = False) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "envelope": self.envelope.to_dict() if self.envelope is not None else None,
            "status": self.status,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": dict(self.result) if self.result is not None else None,
            "checkpoint_keys": list(self.checkpoint_keys),
            "artifact_refs": [ref.to_dict() for ref in self.artifact_refs],
        }
        if include_events:
            data["events"] = [event.to_dict() for event in self.events]
        return data

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, events: tuple[RunEvent, ...] = ()
    ) -> "RunRecord":
        raw_envelope = data.get("envelope")
        return cls(
            run_id=str(data["run_id"]),
            envelope=RunEnvelope.from_dict(raw_envelope)
            if isinstance(raw_envelope, dict)
            else None,
            status=str(data.get("status", "created")),
            metadata=dict(data.get("metadata") or {}),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            result=dict(data["result"]) if isinstance(data.get("result"), dict) else None,
            checkpoint_keys=tuple(str(value) for value in data.get("checkpoint_keys", [])),
            artifact_refs=tuple(
                ArtifactRef.from_dict(value)
                for value in data.get("artifact_refs", [])
                if isinstance(value, dict)
            ),
            events=events,
        )

    def with_updates(self, **changes: Any) -> "RunRecord":
        values = self.to_dict()
        values.update(changes)
        return RunRecord.from_dict(values, events=self.events)

    @classmethod
    def from_result(
        cls,
        result: RunResult,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RunRecord":
        envelope = result.run_envelope
        run_id = envelope.run_id if envelope is not None else "unknown"
        now = time.time()
        return cls(
            run_id=run_id,
            envelope=envelope,
            status=result.status.value,
            metadata=dict(metadata or result.metadata),
            created_at=now,
            updated_at=now,
            result=result.to_dict(),
        )
