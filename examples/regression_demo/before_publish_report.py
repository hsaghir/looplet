"""Report tool implementation used by the regression demo."""

from __future__ import annotations

import json
from pathlib import Path

from looplet.types import ToolContext


def execute(ctx: ToolContext, *, revenue: int, cost: int) -> dict:
    """Write a report from the supplied inputs."""
    root = Path(ctx.resources.get("project_dir") or ".")
    report = {
        "revenue": revenue,
        "cost": cost,
        "profit": revenue + cost,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"written": "report.json", "profit": report["profit"]}
