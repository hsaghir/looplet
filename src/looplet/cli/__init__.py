"""looplet CLI subcommand modules.

Each module here exposes an ``add_subparsers(sub)`` function that
registers one or more subcommands on the top-level argparse parser
in :mod:`looplet.__main__`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from looplet.types import RunResult


def completion_payload(
    result: RunResult,
    *,
    steps: int,
    duration_ms: float,
    trace_dir: Path | None,
) -> dict[str, Any]:
    """The shared CLI completion view for cartridge and bundle runs."""
    return {
        "completed": result.completed,
        "termination_reason": result.termination_reason or "unknown",
        "steps": steps,
        "duration_ms": round(duration_ms, 2),
        "result": result.output,
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
    }
