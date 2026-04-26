"""``python -m looplet`` — CLI entry point.

Subcommands:
    show <trace-dir>    One-page summary of a captured trace directory.
    doctor              Check local looplet/backend configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from looplet import __version__


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
        print(
            f"error: {trace_dir} contains no trajectory.json or "
            "manifest.jsonl — not a trace directory",
            file=sys.stderr,
        )
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
        print(f"#{num}  {ok} {tool}({str(args)[:30]:<30}) → {tail:<20} [{dur}] {link_str}")

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


def _status_line(status: str, name: str, detail: str) -> str:
    marks = {"ok": "OK", "warn": "WARN", "error": "ERROR"}
    return f"{marks.get(status, '?')} {name}: {detail}"


def _doctor_checks(*, probe_backend: bool) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    py_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "name": "python",
            "status": "ok" if py_ok else "error",
            "detail": platform.python_version() + (" (>=3.11)" if py_ok else " (<3.11)"),
        }
    )
    checks.append({"name": "looplet", "status": "ok", "detail": f"version {__version__}"})

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("OPENAI_MODEL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        checks.append(
            {
                "name": "OPENAI_BASE_URL",
                "status": "warn",
                "detail": "not set; set it to probe an OpenAI-compatible backend",
            }
        )
    else:
        checks.append({"name": "OPENAI_BASE_URL", "status": "ok", "detail": base_url})
    checks.append(
        {
            "name": "OPENAI_MODEL",
            "status": "ok" if model else "warn",
            "detail": model or "not set",
        }
    )
    checks.append(
        {
            "name": "OPENAI_API_KEY",
            "status": "ok" if api_key else "warn",
            "detail": "set" if api_key else "not set (local endpoints often accept 'x')",
        }
    )

    if not probe_backend:
        checks.append(
            {"name": "backend_probe", "status": "ok", "detail": "skipped by --no-backend"}
        )
        return checks
    if not base_url or not model:
        checks.append(
            {
                "name": "backend_probe",
                "status": "warn",
                "detail": "skipped; OPENAI_BASE_URL and OPENAI_MODEL are required",
            }
        )
        return checks

    try:
        from looplet.backends import OpenAIBackend  # noqa: PLC0415
        from looplet.native_tools import probe_native_tool_support  # noqa: PLC0415

        llm = OpenAIBackend(base_url=base_url, api_key=api_key or "x", model=model)
        probe = probe_native_tool_support(llm)
        checks.append(
            {
                "name": "native_tools",
                "status": "ok" if probe.supported else "warn",
                "detail": probe.reason,
            }
        )
        if not probe.supported:
            checks.append(
                {
                    "name": "tool_protocol",
                    "status": "ok",
                    "detail": "use LoopConfig(use_native_tools=False) or probe before enabling native tools",
                }
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            {
                "name": "backend_probe",
                "status": "warn",
                "detail": f"could not probe backend: {type(exc).__name__}: {exc}",
            }
        )
    return checks


def _render_doctor(*, probe_backend: bool, json_output: bool, strict: bool) -> int:
    checks = _doctor_checks(probe_backend=probe_backend)
    if json_output:
        print(json.dumps({"checks": checks}, indent=2))
    else:
        print("looplet doctor")
        print()
        for check in checks:
            print(_status_line(check["status"], check["name"], check["detail"]))
    bad = [
        check
        for check in checks
        if check["status"] == "error" or (strict and check["status"] == "warn")
    ]
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m looplet",
        description="looplet — inspect captured trace directories",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser(
        "show",
        help="Show a one-page summary of a captured trace directory",
    )
    show.add_argument("trace_dir", type=Path, help="Path to a trace directory")

    doctor = sub.add_parser(
        "doctor",
        help="Check local looplet configuration and optional backend tool protocol",
    )
    doctor.add_argument(
        "--no-backend",
        action="store_true",
        help="Skip network/backend probing and only check local configuration",
    )
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero for warnings as well as errors",
    )

    args = parser.parse_args(argv)

    if args.command == "show":
        return _render_show(args.trace_dir)
    if args.command == "doctor":
        return _render_doctor(
            probe_backend=not args.no_backend,
            json_output=args.json,
            strict=args.strict,
        )
    # Unreachable — argparse rejects unknown commands.
    return 2


if __name__ == "__main__":
    sys.exit(main())
