"""Tool implementations a private on-call harness already owned.

Nothing in this module imports the loop. The migration keeps these
callables unchanged and replaces only the loop around them, so the raw
loop and `composable_loop()` dispatch the same code.

`select_open_incidents_with_bug` is the one implementation the recipe
replaces in its final stage. Every earlier stage uses it.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

INCIDENTS_FILE = "incidents.json"
HANDOFF_FILE = "handoff.json"

Selector = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
ToolCallable = Callable[..., dict[str, Any]]


def select_open_incidents_with_bug(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Original implementation: it never drops the resolved incidents."""
    return list(incidents)


def select_open_incidents(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed implementation: only the work the next owner still carries."""
    return [incident for incident in incidents if incident["status"] != "resolved"]


def read_incidents(workspace: str | Path) -> list[dict[str, Any]]:
    """Read the shift log the harness was pointed at."""
    raw = (Path(workspace) / INCIDENTS_FILE).read_text(encoding="utf-8")
    return list(json.loads(raw)["incidents"])


@dataclass(frozen=True)
class ToolSuite:
    """The private tool callables, bound to one workspace.

    `invocations` records the name of every private callable that ran.
    The recipe uses it to show that the loop swap dispatched this suite
    rather than a rewritten copy of it.
    """

    workspace: Path
    select_open: Selector
    callables: dict[str, ToolCallable]
    invocations: list[str]


def build_tool_suite(
    workspace: str | Path,
    *,
    select_open: Selector = select_open_incidents_with_bug,
) -> ToolSuite:
    """Bind the private tool callables to `workspace`.

    The callables close over the workspace and the selector instead of
    taking them as model-visible arguments, which is what keeps the tool
    schemas identical before and after the migration.
    """
    root = Path(workspace)
    invocations: list[str] = []

    def list_incidents() -> dict[str, Any]:
        """List every incident recorded during the finished shift."""
        invocations.append("list_incidents")
        return {"incidents": read_incidents(root)}

    def publish_handoff(owner: str) -> dict[str, Any]:
        """Write the handoff file for the next on-call owner."""
        invocations.append("publish_handoff")
        open_incidents = select_open(read_incidents(root))
        handoff = {
            "owner": owner,
            "open_count": len(open_incidents),
            "open_ids": [incident["id"] for incident in open_incidents],
        }
        (root / HANDOFF_FILE).write_text(
            json.dumps(handoff, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"written": HANDOFF_FILE, "open_count": handoff["open_count"]}

    return ToolSuite(
        workspace=root,
        select_open=select_open,
        callables={"list_incidents": list_incidents, "publish_handoff": publish_handoff},
        invocations=invocations,
    )


def changed_implementation_lines() -> tuple[str, str]:
    """Quote the line that differs between the two selectors.

    Reading the lines from source keeps the printed change honest: edit
    either selector and this output follows.
    """
    return (
        _return_line(select_open_incidents_with_bug),
        _return_line(select_open_incidents),
    )


def _return_line(fn: Selector) -> str:
    for line in inspect.getsource(fn).splitlines():
        stripped = line.strip()
        if stripped.startswith("return "):
            return stripped
    raise RuntimeError(f"{fn.__name__} has no return statement to quote")
