"""Optional grouping for related Looplet runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

__all__ = ["ExecutionSession"]


@dataclass(slots=True)
class ExecutionSession:
    """Group run IDs and optional host state without changing loop semantics."""

    session_id: str = field(default_factory=lambda: uuid4().hex[:12])
    parent_run_id: str | None = None
    run_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    closed: bool = False

    def attach(self, run: Any) -> str:
        if self.closed:
            raise RuntimeError("execution session is closed")
        run_id = str(getattr(run, "run_id", run))
        if run_id not in self.run_ids:
            self.run_ids.append(run_id)
        return run_id

    def fork(self, *, parent_run_id: str | None = None) -> "ExecutionSession":
        if self.closed:
            raise RuntimeError("execution session is closed")
        return ExecutionSession(
            parent_run_id=parent_run_id or (self.run_ids[-1] if self.run_ids else None),
            metadata=dict(self.metadata),
        )

    def close(self) -> None:
        self.closed = True
