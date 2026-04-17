"""``python -m openharness`` — CLI entry point.

Subcommands:
    show <trace-dir>    One-page summary of a captured trace directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _fmt_ms(ms: float | int | None) -> str:
    if ms is None:
        return "   -  "
    return f"{int(ms):>5}ms"


def _render_show(trace_dir: Path) -> int:
    if not trace_dir.exists():
        print(f"error: {trace_dir} does not exist", file=sys.stderr)
        return 1

    # ── trajectory.json (optional; short-circuit if missing) ─────
    traj_path = trace_dir / "trajectory.json"
    manifest_path = trace_dir / "manifest.jsonl"
    traj: dict[str, Any] = {}
    if traj_path.exists():
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"error: could not parse {traj_path}: {exc}", file=sys.stderr)
            return 1

    # ── manifest.jsonl (optional) ────────────────────────────────
    calls: list[dict[str, Any]] = []
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except Exception:
                continue

    if not traj and not calls:
        print(f"error: {trace_dir} contains no trajectory.json or "
              "manifest.jsonl — not a trace directory", file=sys.stderr)
        return 1

    # ── Header ──────────────────────────────────────────────────
    run_id = traj.get("run_id") or trace_dir.name
    term = traj.get("termination_reason", "?")
    term_glyph = {"done": "✓", "error": "✗"}.get(term, "·")
    step_count = traj.get("step_count", len(traj.get("steps", [])))
    llm_count = traj.get("llm_call_count", len(calls))
    # Total duration: sum step durations if available, else call durations.
    total_ms = sum(s.get("duration_ms", 0) for s in traj.get("steps", []))
    if total_ms == 0 and calls:
        total_ms = sum(c.get("duration_ms", 0) for c in calls)

    print(
        f"{run_id}  {term_glyph} {term}  "
        f"{step_count} steps  {llm_count} LLM calls  {int(total_ms)}ms"
    )
    print()

    # ── Steps ───────────────────────────────────────────────────
    for s in traj.get("steps", []):
        num = s.get("step_num", "?")
        tc = s.get("tool_call", {}) or {}
        tr = s.get("tool_result", {}) or {}
        tool = tc.get("tool") or tc.get("action") or "?"
        args = s.get("args_summary") or tr.get("args") or ""
        err = tr.get("error") or s.get("error")
        ok = "✗" if err else "✓"
        data = tr.get("data")
        if err:
            tail = f"ERROR: {str(err)[:40]}"
        elif isinstance(data, list):
            tail = f"{len(data)} items"
        elif isinstance(data, dict):
            tail = f"{tr.get('total_items') or len(data)} keys"
        elif data is None:
            tail = ""
        else:
            snippet = str(data)
            tail = snippet if len(snippet) <= 30 else snippet[:27] + "..."
        dur = _fmt_ms(s.get("duration_ms"))
        linked = s.get("llm_call_indices") or []
        link_str = f"call {linked[0]}" if linked else ""
        print(f"#{num}  {ok} {tool}({str(args)[:30]:<30}) "
              f"→ {tail:<20} [{dur}] {link_str}")

    # ── LLM summary ─────────────────────────────────────────────
    if calls:
        total_prompt = sum(c.get("prompt_chars") or 0 for c in calls)
        total_resp = sum(c.get("response_chars") or 0 for c in calls)
        errors = sum(1 for c in calls if c.get("error"))
        print()
        print(
            f"LLM: {len(calls)} calls, "
            f"{total_prompt:,} in / {total_resp:,} out chars, "
            f"{errors} errors"
        )

    # ── Failure modes (if present) ──────────────────────────────
    modes = traj.get("failure_modes") or []
    if modes:
        print()
        for fm in modes:
            print(f"!! {fm}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m openharness",
        description="openharness — inspect captured trace directories",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser(
        "show",
        help="Show a one-page summary of a captured trace directory",
    )
    show.add_argument("trace_dir", type=Path, help="Path to a trace directory")

    args = parser.parse_args(argv)

    if args.command == "show":
        return _render_show(args.trace_dir)
    # Unreachable — argparse rejects unknown commands.
    return 2


if __name__ == "__main__":
    sys.exit(main())
