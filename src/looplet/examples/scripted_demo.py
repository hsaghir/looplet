"""Scripted demo — deterministic, no API key needed.

Used for recording the README GIF and for smoke-testing the loop in
environments without LLM credentials. Every tool call is real; only
the LLM is replaced with a scripted ``MockLLMBackend``.

Run::

    python -m looplet.examples.scripted_demo

Output is identical on every run — that's the point. If you change
this file, re-record the GIF (see ``docs/demo-script.md``).
"""

from __future__ import annotations

import json
import time

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec


def _slow_print(s: str, delay: float = 0.45) -> None:
    """Print with a small pause so the GIF is readable."""
    print(s, flush=True)
    time.sleep(delay)


def main() -> None:
    # ── 1. Define tools — these really run ────────────────────────
    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="search",
            description="Search a log table for a pattern.",
            parameters={"table": "str", "pattern": "str"},
            execute=lambda *, table, pattern: {
                "hits": 12 if pattern == "admin" else 3,
                "table": table,
            },
        )
    )
    tools.register(
        ToolSpec(
            name="rank",
            description="Rank rows in a table by a column.",
            parameters={"table": "str", "by": "str"},
            execute=lambda *, table, by: {
                "top": [
                    {"user": "jsmith", "count": 847},
                    {"user": "admin", "count": 12},
                ]
            },
        )
    )
    tools.register(
        ToolSpec(
            name="done",
            description="Finish with a verdict.",
            parameters={"verdict": "str"},
            execute=lambda *, verdict: {"verdict": verdict},
        )
    )

    # ── 2. Scripted LLM — one JSON per turn ──────────────────────
    llm = MockLLMBackend(
        responses=[
            json.dumps(
                {"tool": "search", "args": {"table": "auth", "pattern": "admin"}}
            ),
            json.dumps({"tool": "rank", "args": {"table": "auth", "by": "user"}}),
            json.dumps(
                {
                    "tool": "done",
                    "args": {"verdict": "jsmith had 847 failed logins — brute force"},
                }
            ),
        ]
    )

    # ── 3. Run the loop — user-owned for-loop ────────────────────
    _slow_print("$ python -m looplet.examples.scripted_demo", delay=0.6)
    _slow_print("# investigating: 'why did the auth alert fire?'", delay=0.7)
    print()

    for step in composable_loop(
        llm=llm,
        tools=tools,
        state=DefaultState(max_steps=5),
        config=LoopConfig(max_steps=5),
        task={"goal": "Investigate the auth alert and render a verdict."},
    ):
        _slow_print(step.pretty())

    print()
    _slow_print("✓ done — 3 steps, 0 errors, deterministic.", delay=0.0)


if __name__ == "__main__":
    main()
