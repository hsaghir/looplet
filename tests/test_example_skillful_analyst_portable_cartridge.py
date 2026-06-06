"""End-to-end dogfood of the fully-portable ``skillful_analyst_portable``.

Cross-runtime twin of ``skillful_analyst``. The done/read_text/write_text
tools are served by a bundled MCP stdio server, the WriteScopeGuard is a
``kind: lep`` permission hook, and the skill system is host-provided
(``search_skills``/``activate_skill`` builtins + ``skill_activation`` hook
+ the RUNTIME-tier ``skill_manager`` resource). Nothing pins it to Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = (
    Path(__file__).resolve().parent.parent / "examples" / "skillful_analyst_portable.cartridge"
)


def test_skillful_analyst_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()
    details = {c.detail for c in report.components}
    assert "mcp" in details  # done/read_text/write_text
    assert "lep" in details  # WriteScopeGuard
    assert "host-service" in details  # skill_manager (RUNTIME tier)


@pytest.mark.timeout(60)
def test_skillful_analyst_portable_cross_process_composition(tmp_path: Path) -> None:
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        assert len(preset.mcp_adapters) == 1
        assert preset.state_service_handles == []
        names = set(preset.tools.tool_names)
        assert {"done", "read_text", "write_text"} <= names
        # Skill builtins are host-provided.
        assert {"search_skills", "activate_skill"} <= names

        guard = next(
            h for h in preset.hooks if "WriteScopeGuard" in getattr(h, "_cartridge_id", "")
        )
        for h in preset.hooks:
            if hasattr(h, "pre_loop"):
                h.pre_loop(None, None, None)

        try:
            # (1) LEP WriteScopeGuard (separate process) blocks empty/escaping
            #     paths, allows an in-scope one.
            assert (
                guard.check_permission(
                    ToolCall(tool="write_text", args={"path": "  ", "content": "x"}),
                    None,
                )
                is False
            )
            assert (
                guard.check_permission(
                    ToolCall(
                        tool="write_text",
                        args={"path": "../escape.txt", "content": "x"},
                    ),
                    None,
                )
                is False
            )
            target = tmp_path / "report.txt"
            assert (
                guard.check_permission(
                    ToolCall(
                        tool="write_text",
                        args={"path": str(target), "content": "x"},
                    ),
                    None,
                )
                is True
            )

            # (2) MCP write_text then read_text (separate process) round-trip.
            #     NB: looplet's dispatch strips trailing whitespace from string
            #     args (same for the original in-process tool), so the content
            #     here has no trailing newline to begin with.
            w = preset.tools.dispatch(
                ToolCall(
                    tool="write_text",
                    args={"path": str(target), "content": "hello\nworld"},
                )
            )
            assert w.error is None
            assert w.data["path"] == str(target)
            assert target.read_text() == "hello\nworld"

            r = preset.tools.dispatch(ToolCall(tool="read_text", args={"path": str(target)}))
            assert r.error is None
            assert r.data["content"] == "hello\nworld"
            assert r.data["lines"] == 2

            # (3) MCP done completes.
            d = preset.tools.dispatch(ToolCall(tool="done", args={"summary": "analysed"}))
            assert d.error is None
            assert d.data["summary"] == "analysed"
        finally:
            for h in preset.hooks:
                if hasattr(h, "close"):
                    h.close()
    finally:
        preset.close()
        assert preset.mcp_adapters == []
