"""Smoke test: the bundled examples/mcp_demo.cartridge actually loads,
spawns its bundled MCP server, dispatches the discovered tool, and
finishes via done(). Pins the canonical end-to-end MCP-transport
example so a regression in the loader, the adapter, the framing, or
the cartridge files themselves shows up here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import cartridge_to_preset
from looplet.testing import MockLLMBackend

CARTRIDGE = Path(__file__).resolve().parent.parent / "examples" / "mcp_demo.cartridge"

pytestmark = pytest.mark.smoke


def test_mcp_demo_cartridge_runs_end_to_end() -> None:
    with cartridge_to_preset(str(CARTRIDGE), strict=True) as preset:
        assert "add" in preset.tools.tool_names, (
            "MCP discovery failed: 'add' tool not registered. The "
            "_server/calc.py either failed to start or the framing "
            "regressed. Re-run scratch/dogfood_mcp_cartridge.py."
        )
        assert len(preset.mcp_adapters) == 1

        llm = MockLLMBackend(
            responses=[
                json.dumps(
                    {
                        "tool": "add",
                        "args": {"a": 7, "b": 5},
                        "reasoning": "compute the sum",
                        "call_id": "1",
                    }
                ),
                json.dumps(
                    {
                        "tool": "done",
                        "args": {"total": 12},
                        "reasoning": "report",
                        "call_id": "2",
                    }
                ),
            ]
        )
        steps = list(preset.run(llm, task={"goal": "add 7 and 5"}))

    assert len(steps) == 2
    assert steps[0].tool_call.tool == "add"
    assert steps[0].tool_result.error is None
    assert "12" in str(steps[0].tool_result.data)
    assert steps[1].tool_call.tool == "done"
    assert steps[1].tool_result.data == {"total": 12, "status": "completed"}
