"""looplet CLI subcommand modules.

Each module here exposes an ``add_subparsers(sub)`` function that
registers one or more subcommands on the top-level argparse parser
in :mod:`looplet.__main__`.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from contextlib import AbstractContextManager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

from looplet.types import RunResult


def add_run_options(parser: ArgumentParser) -> None:
    """Register the shared execution, trace, and cartridge presentation options."""
    parser.add_argument("--trace-dir", type=Path, help="Write provenance trace output here")
    parser.add_argument("--parent-trace", type=Path, help="Link a cartridge run to a parent trace")
    parser.add_argument(
        "--no-trace", action="store_true", help="Disable default provenance capture"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress cartridge per-step output")
    parser.add_argument(
        "--scripted-response",
        action="append",
        default=[],
        help="Mock LLM response; repeat for a network-free run",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Render a live human-friendly cartridge trace"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit one machine-readable completion object"
    )


def execution_output(json_output: bool) -> AbstractContextManager[Any]:
    """Keep authored diagnostics separate from machine-readable completion."""
    return redirect_stdout(sys.stderr) if json_output else nullcontext()


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
