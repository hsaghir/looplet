"""End-to-end dogfood of the fully-portable ``planner_portable`` cartridge.

Cross-runtime twin of ``planner``. Demonstrates that the "plan mode is a
composition, not a loop feature" story stays fully portable: both the
parent's and the child's ``done`` tools are served over MCP, the
``subagent`` builtin is host-provided, and the ``SubagentTaskGuard`` is a
``kind: lep`` permission hook. No Python tool body is pinned to the host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_CARTRIDGE = _EXAMPLES / "planner_portable.cartridge"
_CHILD = _CARTRIDGE / "planner_child.cartridge"


def test_planner_portable_static_profile_is_portable() -> None:
    for root in (_CARTRIDGE, _CHILD):
        report = analyse_cartridge(root)
        assert report.profile == "portable", root.name
        assert report.blockers == (), root.name


@pytest.mark.timeout(60)
def test_planner_portable_cross_process_composition() -> None:
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        # done() comes from the MCP server; subagent is a host builtin.
        assert len(preset.mcp_adapters) == 1
        assert preset.state_service_handles == []
        assert "done" in preset.tools.tool_names
        assert "subagent" in preset.tools.tool_names

        guard = next(
            h for h in preset.hooks if "SubagentTaskGuard" in getattr(h, "_cartridge_id", "")
        )
        for h in preset.hooks:
            if hasattr(h, "pre_loop"):
                h.pre_loop(None, None, None)

        try:
            # (1) LEP guard (separate process) blocks an empty-task subagent
            #     call and allows a real one.
            assert (
                guard.check_permission(ToolCall(tool="subagent", args={"task": "  "}), None)
                is False
            )
            assert (
                guard.check_permission(
                    ToolCall(tool="subagent", args={"task": "ship the feature"}), None
                )
                is True
            )

            # (2) MCP done tool (separate process) completes the run.
            r = preset.tools.dispatch(ToolCall(tool="done", args={"summary": "1. do X\n2. do Y"}))
            assert r.error is None
            assert r.data["status"] == "completed"
            assert r.data["summary"].startswith("1.")
        finally:
            for h in preset.hooks:
                if hasattr(h, "close"):
                    h.close()
    finally:
        preset.close()
        assert preset.mcp_adapters == []


@pytest.mark.timeout(60)
def test_planner_portable_child_done_over_mcp() -> None:
    preset = cartridge_to_preset(_CHILD)
    try:
        assert len(preset.mcp_adapters) == 1
        assert set(preset.tools.tool_names) == {"done"}
        r = preset.tools.dispatch(ToolCall(tool="done", args={"summary": "1. plan\n2. build"}))
        assert r.error is None
        assert r.data["status"] == "completed"
    finally:
        preset.close()
        assert preset.mcp_adapters == []
