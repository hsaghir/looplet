"""Data-agent example — approval + compact + checkpoints, all wired together.

The three capabilities that distinguish ``looplet`` from most
frameworks, shown as one agent that actually needs them:

* **Approval** — the agent can describe, head, and group-by a CSV
  freely, but calling ``delete_rows`` must get human sign-off. The
  sync handler blocks on ``input()`` until you approve / deny.

* **Compact** — ``compact_chain(PruneToolResults, TruncateCompact)``
  fires whenever the session grows past a tiny 4 000-token budget,
  so you'll actually watch it happen on a normal-sized run.

* **Checkpoints** — every step is serialised to
  ``./checkpoints/data_agent/``. Kill the script (Ctrl-C) mid-run
  and re-run it — it resumes from the last step with session log
  intact.

Run::

    # Real LLM (default) — reads OPENAI_BASE_URL / OPENAI_API_KEY /
    # OPENAI_MODEL. Works with OpenAI, Ollama, Together, Groq, vLLM, …
    python -m looplet.examples.data_agent
    python -m looplet.examples.data_agent --resume   # resume latest ckpt
    python -m looplet.examples.data_agent --clean    # wipe checkpoints

    # Scripted MockLLMBackend (for CI / offline testing) — the LLM is
    # fixed so the tool sequence is identical every run.
    python -m looplet.examples.data_agent --mock --auto-approve

Keep the example self-contained: the CSV is generated in a temp dir
so you don't need any data files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from looplet import (
    ApprovalHook,
    BaseToolRegistry,
    ContextBudget,
    DefaultState,
    LoopConfig,
    PruneToolResults,
    ThresholdCompactHook,
    TruncateCompact,
    compact_chain,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

CHECKPOINT_DIR = Path("./checkpoints/data_agent")


# ── 1. Tools — read-only + one dangerous one ─────────────────────


def _make_sample_csv() -> Path:
    """Create a small fake orders CSV so the example is self-contained."""
    path = Path(tempfile.gettempdir()) / "looplet_orders.csv"
    rows = [
        {"order_id": 1, "customer": "alice", "amount": 42.0, "status": "paid"},
        {"order_id": 2, "customer": "bob", "amount": 100.0, "status": "paid"},
        {"order_id": 3, "customer": "alice", "amount": 17.5, "status": "refunded"},
        {"order_id": 4, "customer": "carol", "amount": 99.9, "status": "paid"},
        {"order_id": 5, "customer": "bob", "amount": 8.0, "status": "cancelled"},
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def describe_csv(*, path: str) -> dict:
    """Return row and column counts — cheap and safe."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
    }


def head_csv(*, path: str, n: int = 3) -> dict:
    """Return the first N rows — cheap and safe."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"head": rows[:n]}


def groupby_count(*, path: str, column: str) -> dict:
    """Count rows grouped by the given column — cheap and safe."""
    counts: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = row.get(column, "")
            counts[key] = counts.get(key, 0) + 1
    return {"counts": counts}


def delete_rows(*, path: str, where_status: str, ctx=None) -> dict:
    """Delete rows matching a status — **requires approval**.

    Two paths:

    * Sync handler installed → we block on it. A ``"yes"`` proceeds,
      anything else raises.
    * No handler → we set ``needs_approval=True`` and
      :class:`ApprovalHook` halts the loop so an external system can
      approve and resume later (webhook / Slack / ticket).
    """
    reply = ctx.approve(  # returns None in async / headless mode
        prompt=f"Delete all rows in {path} where status={where_status!r}?",
        options=["yes", "no"],
    ) if ctx is not None else None

    if reply is None:
        # Async path — tell ApprovalHook to stop the loop.
        return {
            "needs_approval": True,
            "approval_description": (
                f"delete_rows(path={path!r}, where_status={where_status!r})"
            ),
        }
    if reply.strip().lower() != "yes":
        return {"deleted": 0, "reason": f"user declined: {reply!r}"}

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if r.get("status") != where_status]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    return {"deleted": len(rows) - len(keep), "remaining": len(keep)}


def done(*, summary: str) -> dict:
    return {"summary": summary}


def build_tools() -> BaseToolRegistry:
    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="describe_csv",
            description="Return row count and columns for a CSV path.",
            parameters={"path": "str"},
            execute=describe_csv,
        )
    )
    tools.register(
        ToolSpec(
            name="head_csv",
            description="Return the first N rows of a CSV.",
            parameters={"path": "str", "n": "int"},
            execute=head_csv,
        )
    )
    tools.register(
        ToolSpec(
            name="groupby_count",
            description="Count rows grouped by a column.",
            parameters={"path": "str", "column": "str"},
            execute=groupby_count,
        )
    )
    tools.register(
        ToolSpec(
            name="delete_rows",
            description="Delete rows matching a status. REQUIRES APPROVAL.",
            parameters={"path": "str", "where_status": "str"},
            execute=delete_rows,  # auto-receives ctx via signature inspection
        )
    )
    tools.register(
        ToolSpec(
            name="done",
            description="Finish with a summary.",
            parameters={"summary": "str"},
            execute=done,
        )
    )
    return tools


# ── 2. Scripted LLM script — drives the same sequence every run ──


def scripted_llm(csv_path: str) -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            json.dumps({"tool": "describe_csv", "args": {"path": csv_path}}),
            json.dumps({"tool": "head_csv", "args": {"path": csv_path, "n": 3}}),
            json.dumps(
                {"tool": "groupby_count", "args": {"path": csv_path, "column": "status"}}
            ),
            json.dumps(
                {
                    "tool": "delete_rows",
                    "args": {"path": csv_path, "where_status": "cancelled"},
                }
            ),
            json.dumps(
                {
                    "tool": "done",
                    "args": {"summary": "inspected orders.csv and removed cancellations"},
                }
            ),
        ]
    )


# ── 3. Sync approval handler — blocks on stdin ──────────────────


def cli_approval_handler(prompt: str, options: list[str] | None) -> str | None:
    opt_str = "/".join(options) if options else "(free text)"
    try:
        return input(f"\n  ⚠  APPROVAL NEEDED: {prompt}  [{opt_str}] > ").strip()
    except EOFError:
        # Non-interactive — defer to async path.
        return None


# ── 4. Main ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--mock",
        action="store_true",
        help="use scripted MockLLMBackend (no API key needed; CI / offline)",
    )
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    ap.add_argument("--clean", action="store_true", help="wipe checkpoints first")
    ap.add_argument(
        "--auto-approve",
        action="store_true",
        help="auto-answer 'yes' to every approval prompt (CI)",
    )
    args = ap.parse_args(argv)

    if args.clean and CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"# wiped {CHECKPOINT_DIR}")

    csv_path = _make_sample_csv()
    print(f"# sample csv: {csv_path}")

    tools = build_tools()

    # ── Compact: two-stage chain, tiny budget so it fires on a short run.
    compact_service = compact_chain(
        PruneToolResults(keep_recent=3),
        TruncateCompact(keep_recent=4),
    )
    budget = ContextBudget(
        context_window=4_000,
        warning_at=2_000,
        error_at=3_000,
        compact_buffer=500,
    )

    # ── Checkpoint: setting LoopConfig.checkpoint_dir auto-saves after
    # every step and auto-resumes on the next run if checkpoints exist.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        from looplet.checkpoint import FileCheckpointStore
        store = FileCheckpointStore(str(CHECKPOINT_DIR))
        latest = store.load_latest()
        if latest is not None:
            print(f"# resuming from step {latest.step_number}")

    # ── Approval: sync stdin handler (or auto-yes for CI).
    approval = (
        (lambda _prompt, _opts: "yes")
        if args.auto_approve
        else cli_approval_handler
    )

    # ── LLM.
    if args.mock:
        llm = scripted_llm(str(csv_path))
    else:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover
            raise SystemExit(
                "openai not installed — run `pip install 'looplet[openai]'`, "
                "or re-run with --mock for a scripted demo."
            ) from None

        from looplet.backends import OpenAIBackend

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Set it (or use Ollama with "
                "OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama), "
                "or re-run with --mock for a scripted demo."
            )
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print(f"# llm: {model} via {base_url}")
        llm = OpenAIBackend(
            OpenAI(base_url=base_url, api_key=api_key), model=model
        )

    config = LoopConfig(
        max_steps=10,
        system_prompt=(
            f"You are a careful data analyst. The CSV to analyse is at the "
            f"absolute path: {csv_path}\n"
            "Use that exact path in every tool call. Describe the CSV, "
            "inspect a sample with head_csv, group rows by status with "
            "groupby_count, then call delete_rows to clean up cancelled "
            "orders. Any destructive action must be explicitly approved."
        ),
        compact_service=compact_service,
        checkpoint_dir=str(CHECKPOINT_DIR),
        approval_handler=approval,
        context_window=4_000,
    )
    hooks = [
        ApprovalHook(),
        ThresholdCompactHook(budget, fire_tier="warning"),
    ]

    threshold_hook: ThresholdCompactHook = hooks[1]  # type: ignore[assignment]
    for step in composable_loop(
        llm=llm,
        tools=tools,
        state=DefaultState(max_steps=10),
        config=config,
        task={
            "goal": (
                f"inspect the orders CSV at {csv_path} and clean up "
                "cancelled orders"
            )
        },
        hooks=hooks,
    ):
        print(step.pretty())

    print()
    print(f"# checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"# compact fired at steps: {threshold_hook.fired_at or '—'}")
    print(
        "# run again with --resume to continue from the last saved step, "
        "or --clean to start over."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
