"""Read the report from disk instead of trusting the agent's claim."""

from __future__ import annotations

import json
from pathlib import Path


def collect_report(state, runtime) -> dict:
    root = Path((runtime or {}).get("project_root", "."))
    path = root / "report.json"
    if not path.is_file():
        return {"report_exists": False}
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"report_exists": True, "report_error": str(exc)}
    return {
        "report_exists": True,
        "report": report,
        "observed_profit": report.get("profit"),
    }
