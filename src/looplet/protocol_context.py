"""Shared correlation context for Looplet protocol boundaries."""

from __future__ import annotations

import time
from typing import Any

from looplet.types import RunEnvelope


def envelope_params(envelope: RunEnvelope | None) -> dict[str, Any]:
    """Return the optional JSON-safe correlation block for a wire request."""
    if envelope is None:
        return {}
    return {"run_envelope": envelope.to_dict()}


def merge_envelope_params(
    params: dict[str, Any] | None,
    envelope: RunEnvelope | None,
) -> dict[str, Any]:
    """Add correlation context without mutating caller-owned params."""
    merged = dict(params or {})
    if envelope is not None:
        merged.setdefault("run_envelope", envelope.to_dict())
    return merged


def remaining_deadline(envelope: RunEnvelope | None) -> float | None:
    """Return seconds remaining, or ``None`` when no deadline is active."""
    if envelope is None or envelope.deadline_at is None:
        return None
    return max(0.0, envelope.deadline_at - time.time())
