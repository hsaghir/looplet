"""looplet CLI subcommand modules.

Each module here exposes an ``add_subparsers(sub)`` function that
registers one or more subcommands on the top-level argparse parser
in :mod:`looplet.__main__`.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

from looplet.types import RunResult


def add_backend_options(parser: ArgumentParser) -> None:
    """Register provider overrides shared by run and eval commands."""
    parser.add_argument("--model", default=None, help="Override the selected provider's model.")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL (else $OPENAI_BASE_URL)."
    )


def build_cli_backend(*, model: str | None = None, base_url: str | None = None) -> Any:
    """Apply explicit overrides without changing the caller's environment."""
    from looplet.backends import OpenAIBackend, make_backend  # noqa: PLC0415

    if base_url:
        return OpenAIBackend(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or "x",
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )
    return make_backend(model=model) if model is not None else make_backend()


def add_run_options(parser: ArgumentParser) -> None:
    """Register the shared execution, trace, and cartridge presentation options."""
    add_backend_options(parser)
    cartridge = parser.add_argument_group("Python factory and cartridge options")
    parser.add_argument("--trace-dir", type=Path, help="Write provenance trace output here")
    cartridge.add_argument("--parent-trace", type=Path, help="Link a run to a parent trace")
    parser.add_argument(
        "--no-trace", action="store_true", help="Disable default provenance capture"
    )
    cartridge.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    parser.add_argument(
        "--scripted-response",
        action="append",
        default=[],
        help="Mock LLM response; repeat for a network-free run",
    )
    cartridge.add_argument(
        "--pretty", action="store_true", help="Render a live human-friendly agent trace"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit one machine-readable completion object"
    )
    add_completion_options(parser)


def add_completion_options(parser: ArgumentParser) -> None:
    """Require accepted completion unless the caller explicitly permits a stop."""
    completion = parser.add_mutually_exclusive_group()
    completion.add_argument(
        "--require-completed",
        action="store_true",
        help="Return nonzero unless completion is accepted (the default)",
    )
    completion.add_argument(
        "--allow-incomplete",
        dest="require_completed",
        action="store_false",
        help="Allow exit zero after a budget or hook stop without accepted completion",
    )
    parser.set_defaults(require_completed=True)


def execution_output(json_output: bool) -> AbstractContextManager[Any]:
    """Keep authored diagnostics separate from machine-readable completion."""
    return redirect_stdout(sys.stderr) if json_output else nullcontext()


def completion_payload(
    result: RunResult,
    *,
    steps: int,
    duration_ms: float,
    trace_dir: Path | None,
    hooks: Iterable[Any] = (),
) -> dict[str, Any]:
    """The shared CLI completion view for cartridge and bundle runs."""
    from looplet.evals import EvalHook

    assessments = [
        hook.assessment() for hook in hooks if isinstance(hook, EvalHook) and hook.results
    ]
    grading_status = "not_graded"
    if assessments:
        grading_status = (
            "passed" if all(assessment.passed for assessment in assessments) else "failed"
        )
    error = None
    if not result.completed:
        for step in reversed(result.steps):
            message = step.tool_result.error
            if message is None and isinstance(step.tool_result.data, dict):
                message = step.tool_result.data.get("error")
            if message:
                error = {
                    "step": step.number,
                    "tool": step.tool_result.tool,
                    "message": str(message)[:2000],
                }
                break
    return {
        "completed": result.completed,
        "grading_status": grading_status,
        "termination_reason": result.termination_reason or "unknown",
        "steps": steps,
        "duration_ms": round(duration_ms, 2),
        "result": result.output,
        "error": error,
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
    }


def completion_exit_code(result: RunResult, *, require_completed: bool = True) -> int:
    """Fail incomplete CLI runs unless the caller explicitly permits a stop."""
    return int(result.failed or (require_completed and not result.completed))
