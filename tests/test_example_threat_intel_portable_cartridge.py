"""End-to-end dogfood of the fully-portable ``threat_intel_portable``.

Cross-runtime twin of ``threat_intel``. All 7 tools are served by a
bundled MCP stdio server, the FeedAllowlistGuard egress policy is a
``kind: lep`` hook, and compaction is the host-provided RUNTIME-tier
``compact_service`` resource. Nothing pins it to Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.cartridge import analyse_cartridge, cartridge_to_preset
from looplet.types import ToolCall

_CARTRIDGE = Path(__file__).resolve().parent.parent / "examples" / "threat_intel_portable.cartridge"


def test_threat_intel_portable_static_profile_is_portable() -> None:
    report = analyse_cartridge(_CARTRIDGE)
    assert report.profile == "portable"
    assert report.blockers == ()
    details = {c.detail for c in report.components}
    assert "mcp" in details  # all 7 tools
    assert "lep" in details  # FeedAllowlistGuard
    assert "host-service" in details  # compact_service (RUNTIME tier)


@pytest.mark.timeout(60)
def test_threat_intel_portable_cross_process_composition() -> None:
    preset = cartridge_to_preset(_CARTRIDGE)
    try:
        assert len(preset.mcp_adapters) == 1
        assert preset.state_service_handles == []
        names = set(preset.tools.tool_names)
        assert {
            "fetch_feed",
            "search_cve",
            "extract_iocs",
            "map_mitre",
            "assess_risk",
            "think",
            "done",
        } <= names

        guard = next(
            h for h in preset.hooks if "FeedAllowlistGuard" in getattr(h, "_cartridge_id", "")
        )
        for h in preset.hooks:
            if hasattr(h, "pre_loop"):
                h.pre_loop(None, None, None)

        try:
            # (1) LEP FeedAllowlistGuard (separate process) refuses an
            #     off-allowlist feed, allows a trusted one.
            assert (
                guard.check_permission(
                    ToolCall(tool="fetch_feed", args={"feed_name": "evil_feed"}),
                    None,
                )
                is False
            )
            assert (
                guard.check_permission(
                    ToolCall(tool="fetch_feed", args={"feed_name": "cisa_alerts"}),
                    None,
                )
                is True
            )

            # (2) MCP fetch_feed returns the CISA alerts.
            ff = preset.tools.dispatch(
                ToolCall(tool="fetch_feed", args={"feed_name": "cisa_alerts"})
            )
            assert ff.error is None
            assert ff.data["item_count"] == 2
            assert ff.data["items"][0]["severity"] == "CRITICAL"

            # (3) MCP search_cve finds the Bitwarden CVE.
            sc = preset.tools.dispatch(
                ToolCall(tool="search_cve", args={"cve_id": "CVE-2026-3891"})
            )
            assert sc.error is None
            assert sc.data["cvss_v3"] == 9.8
            assert sc.data["vendor"] == "Bitwarden"

            # (4) MCP extract_iocs (deterministic regex extraction).
            ei = preset.tools.dispatch(
                ToolCall(
                    tool="extract_iocs",
                    args={
                        "text": "C2 at update.telecom-infra[.]net and CVE-2026-3891, "
                        "hash SHA256:deadbeef1234",
                    },
                )
            )
            assert ei.error is None
            assert "CVE-2026-3891" in ei.data["cves"]
            assert ei.data["total_iocs"] >= 2
            assert "llm_severity" not in ei.data  # degraded in portable twin

            # (5) MCP map_mitre resolves known + unknown techniques.
            mm = preset.tools.dispatch(
                ToolCall(tool="map_mitre", args={"technique_ids": "T1557,T9999"})
            )
            assert mm.error is None
            assert mm.data["techniques"]["T1557"]["name"] == "Adversary-in-the-Middle"
            assert mm.data["techniques"]["T9999"]["name"] == "Unknown"

            # (6) MCP assess_risk degrades to severity-only (no host LLM).
            ar = preset.tools.dispatch(
                ToolCall(
                    tool="assess_risk",
                    args={
                        "title": "Bitwarden supply chain",
                        "severity": "CRITICAL",
                        "affected_products": "Bitwarden CLI",
                    },
                )
            )
            assert ar.error is None
            assert "no LLM" in ar.data["assessment"]

            # (7) think + done round-trip.
            th = preset.tools.dispatch(
                ToolCall(tool="think", args={"analysis": "supply chain risk is high"})
            )
            assert th.error is None
            assert th.data["noted"] is True

            done = preset.tools.dispatch(
                ToolCall(tool="done", args={"summary": "briefing complete"})
            )
            assert done.error is None
            assert done.data["status"] == "completed"
        finally:
            for h in preset.hooks:
                if hasattr(h, "close"):
                    h.close()
    finally:
        preset.close()
