#!/usr/bin/env python
"""Benchmark: looplet coder.cartridge vs GitHub Copilot CLI.

Both agents run the SAME task, in a fresh isolated workspace, using the
SAME underlying model (Copilot's Claude via the local copilot proxy for
looplet; Copilot's own connection for the CLI). Each result is checked
by a deterministic verifier.

Usage:
    python bench.py [task_id ...]          # run all, or a subset
Env:
    BENCH_MODEL   (default: claude-sonnet-4.6)
    BENCH_TOOLS   (default: "looplet,copilot")
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TASKS_MOD = os.environ.get("BENCH_TASKS", "tasks")
TASKS = importlib.import_module(TASKS_MOD).TASKS
OUT_FILE = HERE / os.environ.get("BENCH_OUT", "results.json")
RUNS_ROOT = HERE / os.environ.get("BENCH_RUNS", "runs")
# Repo root: this file is benchmarks/coder_vs_agents/bench.py -> up two levels.
LOOPLET_DIR = Path(__file__).resolve().parents[2]
LOOPLET_PY = str(LOOPLET_DIR / ".venv" / "bin" / "python")
CARTRIDGE = str(LOOPLET_DIR / "examples" / "coder.cartridge")
RUNNER = str(HERE / "looplet_runner.py")
RUNS = RUNS_ROOT
PROXY_URL = os.environ.get("BENCH_PROXY", "http://127.0.0.1:19823/v1")
MODEL = os.environ.get("BENCH_MODEL", "claude-sonnet-4.6")
MAX_STEPS = int(os.environ.get("BENCH_MAX_STEPS", "20"))
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "360"))
TOOLS = os.environ.get("BENCH_TOOLS", "looplet,copilot").split(",")


def _seed(ws: Path, task: dict) -> None:
    for rel, content in task.get("seed", {}).items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _fresh_ws(task_id: str, tool: str) -> Path:
    ws = RUNS / task_id / tool
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    return ws


def _num_k(numstr: str, suffix: str) -> int:
    x = float(numstr)
    s = suffix.lower()
    if s == "k":
        return int(x * 1_000)
    if s == "m":
        return int(x * 1_000_000)
    return int(x)


# ── runners ────────────────────────────────────────────────────────
def run_looplet(task: dict, ws: Path) -> dict:
    env = dict(os.environ)
    env["OPENAI_BASE_URL"] = PROXY_URL
    env["OPENAI_API_KEY"] = "x"
    env["OPENAI_MODEL"] = MODEL
    env["LOOPLET_PROJECT_ROOT"] = str(ws)
    cmd = [LOOPLET_PY, RUNNER, CARTRIDGE, str(ws), str(MAX_STEPS), task["prompt"]]
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=TIMEOUT)
        raw = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
    wall = round(time.time() - t0, 2)

    metrics = {}
    for line in raw.splitlines():
        if line.startswith("LOOPLET_RESULT "):
            try:
                metrics = json.loads(line[len("LOOPLET_RESULT ") :])
            except Exception:  # noqa: BLE001
                pass
    return {
        "tool": "looplet",
        "wall_s": wall,
        "timed_out": timed_out,
        "steps": metrics.get("steps"),
        "llm_calls": metrics.get("llm_calls"),
        "est_in_tokens": metrics.get("est_in_tokens"),
        "est_out_tokens": metrics.get("est_out_tokens"),
        "est_total_tokens": metrics.get("est_total_tokens"),
        "credits": None,
        "in_tokens": None,
        "out_tokens": None,
        "summary": metrics.get("summary", ""),
        "tools_used": metrics.get("tools_used", {}),
        "error": metrics.get("error"),
        "raw_tail": raw.strip()[-1200:],
    }


def run_copilot(task: dict, ws: Path) -> dict:
    env = dict(os.environ)
    # Don't let the outer Copilot-agent session env leak into the child CLI.
    for k in ("AI_AGENT", "COPILOT_AGENT", "COPILOT_DEBUG_NONCE"):
        env.pop(k, None)
    cmd = [
        "copilot",
        "-p",
        task["prompt"],
        "--allow-all",
        "--model",
        MODEL,
        "--log-level",
        "none",
        "--no-color",
    ]
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=TIMEOUT)
        raw = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        raw = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
    wall = round(time.time() - t0, 2)

    credits = None
    m = re.search(r"AI Credits\s+([\d.]+)", raw)
    if m:
        credits = float(m.group(1))
    in_tok = out_tok = None
    m = re.search(r"↑\s*([\d.]+)\s*([kKmM]?)", raw)
    if m:
        in_tok = _num_k(m.group(1), m.group(2))
    m = re.search(r"↓\s*([\d.]+)\s*([kKmM]?)", raw)
    if m:
        out_tok = _num_k(m.group(1), m.group(2))
    actions = len(re.findall(r"(?m)^\s*●", raw))

    return {
        "tool": "copilot",
        "wall_s": wall,
        "timed_out": timed_out,
        "steps": actions or None,
        "llm_calls": None,
        "est_in_tokens": None,
        "est_out_tokens": None,
        "est_total_tokens": None,
        "credits": credits,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "summary": "",
        "tools_used": {},
        "error": None,
        "raw_tail": raw.strip()[-1200:],
    }


RUNNERS = {"looplet": run_looplet, "copilot": run_copilot}


def main(argv: list[str]) -> int:
    selected = argv or [t["id"] for t in TASKS]
    tasks = [t for t in TASKS if t["id"] in selected]
    RUNS.mkdir(parents=True, exist_ok=True)
    results = []
    started = datetime.now(timezone.utc).isoformat()

    for task in tasks:
        print(f"\n{'=' * 70}\n[{task['id']}] {task['title']}  ({task['kind']})")
        for tool in TOOLS:
            ws = _fresh_ws(task["id"], tool)
            _seed(ws, task)
            print(f"  → {tool:8s} running…", end="", flush=True)
            rec = RUNNERS[tool](task, ws)
            try:
                passed, detail = task["verify"](ws, LOOPLET_PY)
            except Exception as exc:  # noqa: BLE001
                passed, detail = False, f"verifier crashed: {type(exc).__name__}: {exc}"
            rec.update(
                {"task": task["id"], "kind": task["kind"], "passed": bool(passed), "detail": detail}
            )
            results.append(rec)
            flag = "PASS" if passed else "FAIL"
            extra = f"{rec['steps']} steps" if rec.get("steps") else ""
            to = " TIMEOUT" if rec["timed_out"] else ""
            print(
                f"\r  → {tool:8s} {flag}  {rec['wall_s']:6.1f}s  {extra:10s}{to}  — {detail[:80]}"
            )

    # Merge with any existing results so incremental runs accumulate
    # (a re-run of a (task, tool) pair replaces the old record).
    prior = []
    if OUT_FILE.exists():
        try:
            prior = json.loads(OUT_FILE.read_text()).get("results", [])
        except Exception:  # noqa: BLE001
            prior = []
    ran = {(r["task"], r["tool"]) for r in results}
    merged = [r for r in prior if (r["task"], r["tool"]) not in ran] + results

    out = {
        "meta": {
            "started": started,
            "model": MODEL,
            "proxy": PROXY_URL,
            "max_steps": MAX_STEPS,
            "cartridge": CARTRIDGE,
            "n_tasks": len({r["task"] for r in merged}),
            "tools": TOOLS,
        },
        "results": merged,
    }
    (OUT_FILE).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {len(results)} new (+{len(merged) - len(results)} kept) → {OUT_FILE}")
    _summary(merged)
    return 0


def _summary(results: list[dict]) -> None:
    by_tool: dict[str, list[dict]] = {}
    for r in results:
        by_tool.setdefault(r["tool"], []).append(r)
    print(f"\n{'=' * 70}\nSUMMARY")
    for tool, recs in by_tool.items():
        n = len(recs)
        passed = sum(r["passed"] for r in recs)
        wall = sum(r["wall_s"] for r in recs)
        print(
            f"  {tool:8s}  {passed}/{n} passed   total wall {wall:6.1f}s   "
            f"avg {wall / max(n, 1):5.1f}s"
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
