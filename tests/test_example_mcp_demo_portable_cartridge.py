"""End-to-end dogfood of the fully-portable ``mcp_demo_portable`` cartridge.

Cross-runtime twin of ``mcp_demo``. Unlike the original (which keeps the
``done`` completion sentinel as an in-process ``tools/done/execute.py``),
this twin serves BOTH ``add`` and ``done`` from the bundled MCP stdio
server and guards them with a ``kind: lep`` permission hook. Nothing is
pinned to a Python host.

The test loads the cartridge to a live preset (spawning the MCP server
and the LEP hook server - two separate processes) and witnesses that the
MCP transport tools and the LEP permission policy compose: the guard
vets ``add`` operands before dispatch, the MCP ``add`` computes the sum
out of process, and the MCP ``done`` completes the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = Path(__file__).resolve().parent.parent / "examples" / "mcp_demo_portable.cartridge"


def test_mcp_demo_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()
    details = {c.detail for c in report.components}
    assert "mcp" in details  # add + done served over MCP
    assert "lep" in details  # CalcGuard permission policy


@pytest.mark.timeout(60)
def test_mcp_demo_portable_cross_process_composition() -> None:
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        # One MCP server, no state services, no in-process tool bodies.
        assert len(preset.mcp_adapters) == 1
        assert preset.state_service_handles == []

        reg = preset.tools
        assert set(reg.tool_names) == {"add", "done"}

        guard = next(h for h in preset.hooks if "CalcGuard" in getattr(h, "_cartridge_id", ""))
        for h in preset.hooks:
            if hasattr(h, "pre_loop"):
                h.pre_loop(None, None, None)

        try:
            # (1) LEP CalcGuard (separate process) denies non-numeric
            #     operands and allows numeric ones.
            assert (
                guard.check_permission(ToolCall(tool="add", args={"a": "x", "b": 2}), None) is False
            )
            assert guard.check_permission(ToolCall(tool="add", args={"a": 2, "b": 3}), None) is True

            # (2) MCP add tool (separate process) computes the sum.
            r1 = reg.dispatch(ToolCall(tool="add", args={"a": 2, "b": 3}))
            assert r1.error is None
            assert r1.data == {"sum": 5}

            # (3) MCP done tool completes the run.
            r2 = reg.dispatch(ToolCall(tool="done", args={"total": 5}))
            assert r2.error is None
            assert r2.data["status"] == "completed"
            assert r2.data["total"] == 5
        finally:
            for h in preset.hooks:
                if hasattr(h, "close"):
                    h.close()
    finally:
        preset.close()
        assert preset.mcp_adapters == []
