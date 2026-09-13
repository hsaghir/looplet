"""Host-facing run and artifact records.

These records adapt existing lifecycle payloads and saved artifacts into one
small, JSON-safe envelope without changing the loop's hook contracts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from looplet.events import EventPayload
from looplet.types import RunEnvelope, RunResult

__all__ = ["ArtifactRef", "RunEvent", "RunRecord", "event_from_payload"]


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
