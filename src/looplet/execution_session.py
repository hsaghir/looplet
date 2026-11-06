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

    def branch_checkpoint(
        self,
        source: Any,
        *,
        key: str | None = None,
        step_number: int | None = None,
        source_run_id: str | None = None,
        new_run_id: str | None = None,
        inject_message: str | None = None,
    ) -> Any:
        """Branch a checkpoint into a new run attached to this session."""
        if self.closed:
            raise RuntimeError("execution session is closed")
        from looplet.checkpoint import branch_from_checkpoint  # noqa: PLC0415

        effective_source_run = source_run_id or (self.run_ids[-1] if self.run_ids else None)
        branched = branch_from_checkpoint(
            source,
            key=key,
            step_number=step_number,
            source_run_id=effective_source_run,
            new_run_id=new_run_id,
            inject_message=inject_message,
        )
        if branched.run_id is not None:
            self.attach(branched.run_id)
        return branched

    def close(self) -> None:
        self.closed = True
