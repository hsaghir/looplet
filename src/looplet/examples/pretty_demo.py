"""Deterministic pretty-trace demo for README GIF recording.

This module is a recording utility, not a usage example. It exercises
the same ``PrettyPrinter`` used by ``looplet new --pretty`` and
``looplet run-workspace --pretty``, but feeds it scripted ``Step``
objects so the resulting asciinema/GIF artifact is stable, fast, and
does not require API credentials.

Run::

    python -m looplet.examples.pretty_demo

Use ``--fast`` or ``LOOPLET_PRETTY_DEMO_FAST=1`` to remove pacing
delays during tests.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable
from typing import Any

from looplet import Step, ToolCall, ToolResult
from looplet.cli._pretty import PrettyPrinter


def _pause(seconds: float, *, fast: bool) -> None:
    if not fast:
        time.sleep(seconds)


def _command(text: str, *, fast: bool) -> None:
    print(f"$ {text}", flush=True)


def _step(
    number: int,
    tool: str,
    *,
    args: dict[str, Any],
    reasoning: str,
    data: Any,
    duration_ms: float,
    args_summary: str | None = None,
) -> Step:
    call = ToolCall(tool=tool, args=args, reasoning=reasoning, call_id=f"demo-{number}")
    result = ToolResult(
        tool=tool,
        args_summary=args_summary or ", ".join(f"{key}={value!r}" for key, value in args.items()),
        data=data,
        duration_ms=duration_ms,
        call_id=call.call_id,
    )
    return Step(number=number, tool_call=call, tool_result=result)


def _render(printer: PrettyPrinter, steps: Iterable[Step], *, fast: bool) -> None:
    for step in steps:
        printer.step(step)
        _pause(0.45, fast=fast)


def _build_steps() -> list[Step]:
    return [
        _step(
            1,
            "scaffold_cartridge",
            args={"path": "./url_summarizer.workspace", "tools": ["fetch_url", "summarize"]},
            reasoning="Start from a normal looplet workspace: config, prompt, tools, and done.",
            data={
                "scaffolded": True,
                "tools_created": ["fetch_url", "summarize", "done"],
            },
            duration_ms=12,
        ),
        _step(
            2,
            "write_file",
            args={"file_path": "prompts/system.md", "content": "You summarize URLs."},
            reasoning="Give the agent a tight mission and tell it not to invent page content.",
            data={"written": True, "lines": 9},
            duration_ms=8,
        ),
        _step(
            3,
            "write_file",
            args={"file_path": "tools/fetch_url/execute.py", "content": "stdlib urllib tool"},
            reasoning="Use a tiny Python tool; no framework glue or runtime dependency needed.",
            data={"written": True, "lines": 18},
            duration_ms=10,
        ),
        _step(
            4,
            "validate_workspace",
            args={"path": "./url_summarizer.workspace", "strict": True},
            reasoning="Load the workspace before handing it back so missing files fail early.",
            data={"valid": True, "n_tools": 3},
            duration_ms=26,
        ),
        _step(
            5,
            "done",
            args={"summary": "url_summarizer.workspace is ready"},
            reasoning="The agent can now be shipped, edited, or run locally.",
            data={"summary": "url_summarizer.workspace is ready"},
            duration_ms=3,
        ),
    ]


def _run_steps() -> list[Step]:
    return [
        _step(
            1,
            "fetch_url",
            args={"url": "https://example.com"},
            reasoning="Fetch the source page first; every later step should cite tool data.",
            data={"status": 200, "bytes": 1256},
            duration_ms=184,
        ),
        _step(
            2,
            "extract_title",
            args={"html": "<html><title>Example Domain</title>..."},
            reasoning="Extract the exact browser title before summarizing.",
            data={"title": "Example Domain"},
            duration_ms=11,
        ),
        _step(
            3,
            "summarize_text",
            args={"text": "Example Domain is a placeholder page for documentation examples."},
            reasoning="Compress the page text into the requested two-sentence answer.",
            data={
                "summary": (
                    "Example Domain is a small placeholder page used in documentation. "
                    "It is safe to reference in examples because it has no production content."
                )
            },
            duration_ms=231,
        ),
        _step(
            4,
            "done",
            args={"answer": "Example Domain: placeholder documentation page."},
            reasoning="Return the title and concise summary to the caller.",
            data={"answer": "Example Domain: placeholder documentation page."},
            duration_ms=4,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    fast = os.environ.get("LOOPLET_PRETTY_DEMO_FAST") == "1" or "--fast" in argv
    unknown = [arg for arg in argv if arg != "--fast"]
    if unknown:
        raise SystemExit(f"unknown arguments: {' '.join(unknown)}")

    _command(
        'looplet new "URL summarizer" ./url_summarizer.workspace --pretty',
        fast=fast,
    )
    build_printer = PrettyPrinter(
        title="looplet new \u00b7 building url_summarizer",
        max_steps=len(_build_steps()),
    )
    build_printer.header(
        [
            "  brief:  URL summarizer with fetch_url and summarize tools",
            "  target: ./url_summarizer.workspace",
            "  model:  any OpenAI-compatible endpoint",
        ]
    )
    _render(build_printer, _build_steps(), fast=fast)
    build_printer.finish(summary="url_summarizer.workspace is ready")

    print()
    _command(
        'looplet run-workspace ./url_summarizer.workspace "Summarize example.com" --pretty',
        fast=fast,
    )
    run_printer = PrettyPrinter(
        title="looplet run \u00b7 url_summarizer.workspace",
        max_steps=len(_run_steps()),
    )
    run_printer.header(
        [
            "  task:  Summarize example.com",
            "  model: any OpenAI-compatible endpoint",
        ]
    )
    _render(run_printer, _run_steps(), fast=fast)
    run_printer.finish(summary="Example Domain: placeholder documentation page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
