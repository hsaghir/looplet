#!/usr/bin/env python
"""Render results.json into a detailed Markdown comparison report."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fmt(v, suffix="", dash="—"):
    return f"{v}{suffix}" if v is not None else dash


def main() -> int:
    import sys

    results_name = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    data = json.loads((HERE / results_name).read_text())
    stem = Path(results_name).stem
    meta = data["meta"]
    rows = data["results"]

    # index by (task, tool)
    by = {(r["task"], r["tool"]): r for r in rows}
    tasks, kinds = [], {}
    for r in rows:
        if r["task"] not in kinds:
            tasks.append(r["task"])
            kinds[r["task"]] = r["kind"]

    out = []
    out.append("# looplet coder vs GitHub Copilot CLI — benchmark\n")
    out.append(
        f"- Model (both): `{meta['model']}`  ·  looplet via proxy `{meta['proxy']}`, "
        f"Copilot via its own connection\n"
        f"- Tasks: {meta['n_tasks']}  ·  looplet max_steps: {meta['max_steps']}\n"
        f"- Each task run in a fresh isolated workspace; every result checked by a "
        f"deterministic verifier.\n"
    )

    # per-task table
    out.append("\n## Per-task results\n")
    out.append("| Task | Kind | looplet | wall | steps | copilot | wall | credits | in/out tok |")
    out.append("|---|---|:--:|--:|--:|:--:|--:|--:|--:|")
    for t in tasks:
        lp = by.get((t, "looplet"), {})
        cp = by.get((t, "copilot"), {})
        lp_flag = "✅" if lp.get("passed") else "❌"
        cp_flag = "✅" if cp.get("passed") else "❌"
        out.append(
            f"| {t} | {kinds[t]} "
            f"| {lp_flag} | {_fmt(lp.get('wall_s'), 's')} | {_fmt(lp.get('steps'))} "
            f"| {cp_flag} | {_fmt(cp.get('wall_s'), 's')} | {_fmt(cp.get('credits'))} "
            f"| {_fmt(cp.get('in_tokens'))}/{_fmt(cp.get('out_tokens'))} |"
        )

    # aggregates
    def agg(tool, kind=None):
        recs = [r for r in rows if r["tool"] == tool and (kind is None or r["kind"] == kind)]
        n = len(recs)
        p = sum(r["passed"] for r in recs)
        wall = sum(r["wall_s"] for r in recs)
        return n, p, wall, recs

    out.append("\n## Aggregate\n")
    out.append("| Metric | looplet | copilot |")
    out.append("|---|--:|--:|")
    ln, lp, lwall, lrecs = agg("looplet")
    cn, cp, cwall, crecs = agg("copilot")
    out.append(f"| Tasks passed (overall) | {lp}/{ln} | {cp}/{cn} |")
    for kind in ("coding", "other"):
        _, lpk, _, _ = agg("looplet", kind)
        _, cpk, _, _ = agg("copilot", kind)
        nk = sum(1 for t in tasks if kinds[t] == kind)
        out.append(f"| — {kind} | {lpk}/{nk} | {cpk}/{nk} |")
    out.append(f"| Total wall time | {lwall:.1f}s | {cwall:.1f}s |")
    out.append(f"| Avg wall / task | {lwall / max(ln, 1):.1f}s | {cwall / max(cn, 1):.1f}s |")
    ccred = sum(r["credits"] or 0 for r in crecs)
    out.append(f"| Total AI credits (copilot self-report) | — | {ccred:.1f} |")
    cin = sum(r["in_tokens"] or 0 for r in crecs)
    cout = sum(r["out_tokens"] or 0 for r in crecs)
    out.append(f"| Total input tokens (copilot) | — | {cin:,} |")
    out.append(f"| Total output tokens (copilot) | — | {cout:,} |")
    lest = sum(r["est_total_tokens"] or 0 for r in lrecs)
    out.append(f"| Est. total tokens (looplet, chars/4) | ~{lest:,} | — |")
    lcalls = sum(r["llm_calls"] or 0 for r in lrecs)
    out.append(f"| Total LLM calls (looplet) | {lcalls} | — |")

    # failures detail
    fails = [r for r in rows if not r["passed"]]
    if fails:
        out.append("\n## Failures\n")
        for r in fails:
            out.append(
                f"- **{r['tool']} / {r['task']}**: {r['detail']}"
                + (" (timeout)" if r["timed_out"] else "")
            )
    else:
        out.append("\n_All tasks passed for both tools._\n")

    text = "\n".join(out) + "\n"
    (HERE / f"TABLES_{stem}.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
