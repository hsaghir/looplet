"""End-to-end dogfood of the fully-portable ``hello_portable`` cartridge.

This is the cross-runtime twin of ``hello``: every component lives out
of process behind a protocol -

* greet + done ........ MCP stdio server   (``_mcp/tools_server.py``)
* shared greeting log .. State Service      (``_state/greeting_log.py``)
* PolitenessGate ....... ``kind: lep`` hook (reads the state service)
* NameGuard ............ ``kind: lep`` hook

The test loads the cartridge to a live preset (spawning the MCP server,
the state service, and the two LEP hook servers - four separate
processes) and witnesses that they compose: the MCP greet tool's writes
land in the state service, and the LEP PolitenessGate hook - a DIFFERENT
process - reads that same state across the boundary to gate ``done()``.
This reproduces the original in-process ``@ref`` sharing with zero shared
Python objects, proving the State Service primitive closes the
portability gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = Path(__file__).resolve().parent.parent / "examples" / "hello_portable.cartridge"


def test_hello_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()


@pytest.mark.timeout(60)
def test_hello_portable_cross_process_composition() -> None:
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        # The loader spawned exactly one state service + one MCP server.
        assert len(preset.state_service_handles) == 1
        assert len(preset.mcp_adapters) == 1

        reg = preset.tools
        assert set(reg.tool_names) == {"greet", "done"}

        # The injected client proxies the out-of-process greeting log.
        log = preset.resources["greeting_log"]
        assert set(log.methods) == {"record", "names", "entries", "count"}
        assert log.count() == 0

        # Identify the two LEP hooks and open their sessions.
        polite = next(h for h in preset.hooks if "Politeness" in getattr(h, "_cartridge_id", ""))
        nameg = next(h for h in preset.hooks if h is not polite)
        for h in preset.hooks:
            if hasattr(h, "pre_loop"):
                h.pre_loop(None, None, None)

        try:
            # (1) PolitenessGate (separate process) reads the state service
            #     and blocks done() while no greeting has been recorded.
            blocked = polite.check_done(None, None, None, 0)
            assert blocked and "greet at least one person" in blocked

            # (2) NameGuard denies an empty name, allows a real one.
            assert (
                nameg.check_permission(ToolCall(tool="greet", args={"name": "  "}), None) is False
            )
            assert (
                nameg.check_permission(ToolCall(tool="greet", args={"name": "Ada"}), None) is True
            )

            # (3) MCP greet tool (separate process) records into the SAME
            #     state service the hook reads from.
            r1 = reg.dispatch(ToolCall(tool="greet", args={"name": "Ada"}))
            assert r1.error is None
            assert r1.data == {"greeting": "Hello, Ada!"}
            reg.dispatch(ToolCall(tool="greet", args={"name": "Bob"}))

            # (4) The write is visible across the process boundary.
            assert log.count() == 2
            assert log.names() == ["Ada", "Bob"]

            # (5) PolitenessGate now lets done() through (sees count == 2).
            assert polite.check_done(None, None, None, 1) is None

            # (6) MCP done tool completes.
            r3 = reg.dispatch(ToolCall(tool="done", args={"summary": "Greeted both"}))
            assert r3.error is None
            assert r3.data["status"] == "completed"
        finally:
            for h in preset.hooks:
                if hasattr(h, "close"):
                    h.close()
    finally:
        preset.close()
        # Teardown drains the spawned subprocesses.
        assert preset.state_service_handles == []
        assert preset.mcp_adapters == []
