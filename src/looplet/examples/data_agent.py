"""Data-agent example - approval + compact + checkpoints, all wired together.

The three capabilities that distinguish ``looplet`` from most
frameworks, shown as one agent that actually needs them:

* **Approval** - the agent can describe, head, and group-by a CSV
  freely, but calling ``delete_rows`` must get human sign-off. The
  sync handler blocks on ``input()`` until you approve / deny.

* **Compact** - ``DefaultCompactService`` fires whenever the session
    grows past a tiny 4 000-token budget, so you'll actually watch it
    happen on a normal-sized run. This example disables the summary LLM
    call so scripted/offline runs stay deterministic.

* **Checkpoints** - every step is serialised to
  ``./checkpoints/data_agent/``. Kill the script (Ctrl-C) mid-run
  and re-run it - it resumes from the last step with session log
  intact.

Run::

    # Real LLM (default) - reads OPENAI_BASE_URL / OPENAI_API_KEY /
    # OPENAI_MODEL. Works with OpenAI, Ollama, Together, Groq, vLLM, …
    python -m looplet.examples.data_agent
    python -m looplet.examples.data_agent --resume   # resume latest ckpt
    python -m looplet.examples.data_agent --clean    # wipe checkpoints

    # Scripted MockLLMBackend (for CI / offline testing) - the LLM is
    # fixed so the tool sequence is identical every run.
    python -m looplet.examples.data_agent --scripted --auto-approve

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
    ContextBudget,
    DefaultCompactService,
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    ThresholdCompactHook,
    composable_loop,
    probe_native_tool_support,
    tool,
    tools_from,
)
from looplet.types import ToolContext

CHECKPOINT_DIR = Path("./checkpoints/data_agent")


# ── 1. Tools - read-only + one dangerous one ─────────────────────


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


@tool(description="Return row count and columns for a CSV path.", concurrent_safe=True)
def describe_csv(*, path: str) -> dict:
    """Return row and column counts - cheap and safe."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
    }


@tool(description="Return the first N rows of a CSV.", concurrent_safe=True)
def head_csv(*, path: str, n: int = 3) -> dict:
    """Return the first N rows - cheap and safe."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"head": rows[:n]}


@tool(description="Count rows grouped by a column.", concurrent_safe=True)
def groupby_count(*, path: str, column: str) -> dict:
    """Count rows grouped by the given column - cheap and safe."""
    counts: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = row.get(column, "")
            counts[key] = counts.get(key, 0) + 1
    return {"counts": counts}


@tool(description="Delete rows matching a status. Requires approval.")
def delete_rows(*, path: str, where_status: str, ctx: ToolContext | None = None) -> dict:
    """Delete rows matching a status - **requires approval**.

    Two paths:

    * Sync handler installed → we block on it. A ``"yes"`` proceeds,
      anything else raises.
    * No handler → we set ``needs_approval=True`` and
      :class:`ApprovalHook` halts the loop so an external system can
      approve and resume later (webhook / Slack / ticket).
    """
    reply = (
        ctx.approve(  # returns None in async / headless mode
            prompt=f"Delete all rows in {path} where status={where_status!r}?",
            options=["yes", "no"],
        )
        if ctx is not None
        else None
    )

    if reply is None:
        # Async path - tell ApprovalHook to stop the loop.
        return {
            "needs_approval": True,
            "approval_description": (f"delete_rows(path={path!r}, where_status={where_status!r})"),
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


def build_tools():
    return tools_from(
        [describe_csv, head_csv, groupby_count, delete_rows],
        include_think=True,
        include_done=True,
    )


# ── 2. Scripted LLM script - drives the same sequence every run ──


def _tool_call(tool_name: str, args: dict, reasoning: str) -> str:
    return json.dumps({"tool": tool_name, "args": args, "reasoning": reasoning})


def scripted_llm(csv_path: str) -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            _tool_call("describe_csv", {"path": csv_path}, "inspect columns and row count"),
            _tool_call("head_csv", {"path": csv_path, "n": 3}, "sample the data"),
            _tool_call(
                "groupby_count",
                {"path": csv_path, "column": "status"},
                "find status distribution",
            ),
            _tool_call(
                "delete_rows",
                {"path": csv_path, "where_status": "cancelled"},
                "clean cancelled orders after approval",
            ),
            _tool_call(
                "done",
                {"summary": "inspected orders.csv and removed cancellations"},
                "finish after cleanup",
            ),
        ]
    )


# ── 3. Sync approval handler - blocks on stdin ──────────────────


def cli_approval_handler(prompt: str, options: list[str] | None) -> str | None:
    opt_str = "/".join(options) if options else "(free text)"
    try:
        return input(f"\n  ⚠  APPROVAL NEEDED: {prompt}  [{opt_str}] > ").strip()
    except EOFError:
        # Non-interactive - defer to async path.
        return None


# ── 4. Main ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    summary = (__doc__ or "").split("\n\n", 1)[0]
    ap = argparse.ArgumentParser(description=summary)
    ap.add_argument(
        "--mock",
        action="store_true",
        help="legacy alias for --scripted",
    )
    ap.add_argument(
        "--scripted",
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

    # ── Compact: default service, tiny budget so it fires on a short run.
    compact_service = DefaultCompactService(
        keep_recent=4,
        keep_recent_tool_results=3,
        use_llm_summary=False,
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
    approval = (lambda _prompt, _opts: "yes") if args.auto_approve else cli_approval_handler

    # ── LLM.
    use_scripted = args.scripted or args.mock
    if use_scripted:
        llm = scripted_llm(str(csv_path))
        model_label = "scripted MockLLMBackend"
    else:
        try:
            from looplet.backends import OpenAIBackend
        except ImportError:  # pragma: no cover
            raise SystemExit(
                "openai not installed - run `pip install 'looplet[openai]'`, "
                "or re-run with --scripted for a scripted demo."
            ) from None

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Set it (or use Ollama with "
                "OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama), "
                "or re-run with --scripted for a scripted demo."
            )
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print(f"# llm: {model} via {base_url}")
        llm = OpenAIBackend(base_url=base_url, api_key=api_key, model=model)
        model_label = model
    protocol_probe = probe_native_tool_support(llm)
    print(f"# model: {model_label}")
    print(f"# tool protocol: {'native' if protocol_probe.supported else 'json-text'}")
    print(f"# probe: {protocol_probe.reason}")

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
        use_native_tools=protocol_probe.supported,
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
        task={"goal": (f"inspect the orders CSV at {csv_path} and clean up cancelled orders")},
        hooks=hooks,
    ):
        print(step.pretty())

    print()
    print(f"# checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"# compact fired at steps: {threshold_hook.fired_at or ' - '}")
    print(
        "# run again with --resume to continue from the last saved step, or --clean to start over."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
