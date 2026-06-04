"""End-to-end dogfood of the fully-portable ``dep_doctor_portable``.

Cross-runtime twin of ``dep_doctor``. All 7 audit tools are served by a
bundled MCP stdio server, the RegistryGuard is a ``kind: lep`` permission
hook, and compaction is the host-provided RUNTIME-tier ``compact_service``
resource. Nothing pins it to Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = Path(__file__).resolve().parent.parent / "examples" / "dep_doctor_portable.cartridge"


def test_dep_doctor_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()
    details = {c.detail for c in report.components}
    assert "mcp" in details  # all 7 audit tools
    assert "lep" in details  # RegistryGuard
    assert "host-service" in details  # compact_service (RUNTIME tier)


@pytest.mark.timeout(60)
def test_dep_doctor_portable_cross_process_composition(tmp_path: Path) -> None:
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        assert len(preset.mcp_adapters) == 1
        assert preset.state_service_handles == []
        names = set(preset.tools.tool_names)
        assert {
            "detect_dep_files",
            "parse_deps",
            "check_package",
            "check_license_compat",
            "find_alternatives",
            "think",
            "done",
        } <= names

        guard = next(h for h in preset.hooks if "RegistryGuard" in getattr(h, "_cartridge_id", ""))
        for h in preset.hooks:
            if hasattr(h, "pre_loop"):
                h.pre_loop(None, None, None)

        try:
            # (1) LEP RegistryGuard (separate process) refuses an empty
            #     package_name, allows a real one.
            assert (
                guard.check_permission(
                    ToolCall(tool="check_package", args={"package_name": "  "}),
                    None,
                )
                is False
            )
            assert (
                guard.check_permission(
                    ToolCall(tool="check_package", args={"package_name": "django"}),
                    None,
                )
                is True
            )

            # (2) MCP detect_dep_files over a real on-disk project.
            (tmp_path / "requirements.txt").write_text(
                "django>=5.0\nrequests==2.32.3\nabandoned-lib\n"
            )
            (tmp_path / "package.json").write_text('{"dependencies": {"express": "^4.21.0"}}')
            d = preset.tools.dispatch(
                ToolCall(tool="detect_dep_files", args={"project_dir": str(tmp_path)})
            )
            assert d.error is None
            assert d.data["count"] == 2
            files = {f["file"] for f in d.data["dep_files"]}
            assert files == {"requirements.txt", "package.json"}

            # (3) MCP parse_deps on the requirements file.
            pd = preset.tools.dispatch(
                ToolCall(
                    tool="parse_deps",
                    args={"file_path": str(tmp_path / "requirements.txt")},
                )
            )
            assert pd.error is None
            assert pd.data["count"] == 3
            parsed = {x["name"] for x in pd.data["dependencies"]}
            assert {"django", "requests", "abandoned-lib"} == parsed

            # (4) MCP check_package surfaces the CVE + abandoned status.
            cp = preset.tools.dispatch(
                ToolCall(tool="check_package", args={"package_name": "abandoned-lib"})
            )
            assert cp.error is None
            assert cp.data["status"] == "abandoned"
            assert cp.data["cves"][0]["severity"] == "CRITICAL"
            assert cp.data["stale"] is True

            # (5) MCP check_license_compat flags a GPL clash.
            lc = preset.tools.dispatch(
                ToolCall(
                    tool="check_license_compat",
                    args={"project_license": "MIT", "dep_license": "GPL-3.0"},
                )
            )
            assert lc.error is None
            assert lc.data["compatible"] is False
            assert lc.data["risk"] == "HIGH"

            # (6) find_alternatives degrades to empty (no host LLM in subprocess).
            fa = preset.tools.dispatch(
                ToolCall(tool="find_alternatives", args={"package_name": "abandoned-lib"})
            )
            assert fa.error is None
            assert fa.data["alternatives"] == []

            # (7) think + done round-trip.
            th = preset.tools.dispatch(
                ToolCall(tool="think", args={"thought": "abandoned-lib is a risk"})
            )
            assert th.error is None
            assert th.data["noted"] is True

            done = preset.tools.dispatch(ToolCall(tool="done", args={"summary": "audit complete"}))
            assert done.error is None
            assert done.data["status"] == "completed"
            assert done.data["summary"] == "audit complete"
        finally:
            for h in preset.hooks:
                if hasattr(h, "close"):
                    h.close()
    finally:
        preset.close()
