"""Scripted demo — **GIF-recording utility, not a usage example.**

This file exists only to produce the deterministic terminal GIF at the
top of the README. The LLM is replaced with a scripted
``MockLLMBackend`` so every recording is byte-identical.

What the GIF shows:

1. An 8-line ``DebugHook`` (just a ``Protocol`` impl) prints a one-liner
   for every LLM call and every tool dispatch — that's the entire
   "debugging story" of this library.
2. A destructive tool (``delete_rows``) trips ``ApprovalHook``; the
   loop pauses on an ``APPROVAL NEEDED`` prompt. A scripted ``"yes"``
   lets it resume — the same flow your ops engineer, Slack bot, or
   HITL pipeline would use.
3. The loop returns cleanly. All 5 steps are visible. No magic.

For a real usage example, see instead:

* ``hello_world.py`` — the 20-line "first agent" (real LLM).
* ``coding_agent.py`` — a realistic tool-using agent (real LLM).
* ``data_agent.py`` — approval + compact + checkpoints wired together
  (real LLM by default; pass ``--mock`` for CI).

Run::

    python -m looplet.examples.scripted_demo

Output is identical on every run — that's the point. If you change
this file, re-record the GIF (see ``docs/demo-script.md``).
"""

from __future__ import annotations

import json
import time
from typing import Any

from looplet import (
    ApprovalHook,
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec


def _slow_print(s: str, delay: float = 0.35) -> None:
    """Print with a small pause so the GIF is readable."""
    print(s, flush=True)
    time.sleep(delay)


# ── The whole "debugging story" in 8 lines: a Protocol impl ─────


class DebugHook:
    """Prints a one-liner per phase. This is the whole thing."""

    def pre_dispatch(self, state: Any, session_log: Any, tc: Any, step: Any) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in (tc.args or {}).items())
        _slow_print(f"  ↳ dispatch: {tc.tool}({args})", delay=0.25)

    def post_dispatch(
        self, state: Any, session_log: Any, tc: Any, result: Any, step_num: int
    ) -> None:
        preview = json.dumps(result.data or {})[:48]
        _slow_print(f"  ↳ result:   {preview}", delay=0.25)


def main() -> None:
    # ── 1. Tools — cheap real ones + one "dangerous" one ──────────
    rows = [
        {"id": 1, "user": "alice", "status": "paid"},
        {"id": 2, "user": "bob", "status": "paid"},
        {"id": 3, "user": "alice", "status": "cancelled"},
        {"id": 4, "user": "carol", "status": "cancelled"},
    ]

    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="head",
            description="Preview rows.",
            parameters={"n": "int"},
            execute=lambda *, n: {"rows": rows[:n]},
        )
    )
    tools.register(
        ToolSpec(
            name="count_by_status",
            description="Count rows grouped by status.",
            parameters={},
            execute=lambda: {
                "counts": {"paid": 2, "cancelled": 2},
            },
        )
    )

    def delete_rows(*, where_status: str, ctx: Any = None) -> dict:
        # Approval-gated tool. With a handler installed, it blocks
        # until the operator says "yes"; without one, it sets
        # needs_approval and ApprovalHook stops the loop.
        reply = (
            ctx.approve(
                prompt=f"delete all rows where status={where_status!r}?",
                options=["yes", "no"],
            )
            if ctx is not None
            else None
        )
        if reply is None:
            return {
                "needs_approval": True,
                "approval_description": (f"delete_rows(where_status={where_status!r})"),
            }
        if reply != "yes":
            return {"deleted": 0, "reason": f"denied: {reply!r}"}
        survivors = [r for r in rows if r["status"] != where_status]
        return {"deleted": len(rows) - len(survivors), "remaining": len(survivors)}

    tools.register(
        ToolSpec(
            name="delete_rows",
            description="⚠ destructive — requires approval.",
            parameters={"where_status": "str"},
            execute=delete_rows,
        )
    )
    tools.register(
        ToolSpec(
            name="done",
            description="Finish with a summary.",
            parameters={"summary": "str"},
            execute=lambda *, summary: {"summary": summary},
        )
    )

    # ── 2. Scripted LLM — one JSON per turn ──────────────────────
    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "head", "args": {"n": 4}}),
            json.dumps({"tool": "count_by_status", "args": {}}),
            json.dumps({"tool": "delete_rows", "args": {"where_status": "cancelled"}}),
            json.dumps(
                {
                    "tool": "done",
                    "args": {"summary": "cleaned 2 cancelled rows; 2 remain"},
                }
            ),
        ]
    )

    # ── 3. Sync approval handler — scripted "yes" for determinism
    approval_calls: list[str] = []

    def handler(prompt: str, options: list[str] | None) -> str:
        approval_calls.append(prompt)
        _slow_print("", delay=0.0)
        _slow_print(f"  ⚠  APPROVAL NEEDED: {prompt}", delay=0.6)
        _slow_print(f"     [{'/'.join(options or ['y', 'n'])}] > yes", delay=0.7)
        return "yes"

    config = LoopConfig(
        max_steps=6,
        approval_handler=handler,
        system_prompt=(
            "You are a data cleanup agent. Head the table, count by "
            "status, then remove cancelled rows. Destructive actions "
            "need approval."
        ),
    )

    # ── 4. Run — user-owned for-loop ─────────────────────────────
    _slow_print("$ python -m looplet.examples.scripted_demo", delay=0.5)
    _slow_print(
        "# task: head the orders table, count by status, clean cancelled rows",
        delay=0.7,
    )
    print()

    for step in composable_loop(
        llm=llm,
        tools=tools,
        state=DefaultState(max_steps=6),
        config=config,
        task={"goal": "Clean up cancelled rows in orders."},
        hooks=[DebugHook(), ApprovalHook()],
    ):
        _slow_print(step.pretty(), delay=0.45)

    print()
    _slow_print(
        f"✓ done — {len(approval_calls)} approval prompt, 4 tools, deterministic replay.",
        delay=0.0,
    )


if __name__ == "__main__":
    main()
