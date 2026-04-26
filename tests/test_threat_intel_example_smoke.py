"""Dogfood tests for the Threat Intel example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _load_threat_intel():
    path = Path(__file__).parents[1] / "examples" / "threat_intel" / "agent.py"
    spec = importlib.util.spec_from_file_location("threat_intel_agent", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestThreatIntelExample:
    def test_build_tools_uses_decorator_schema_and_helpers(self) -> None:
        threat_intel = _load_threat_intel()

        registry = threat_intel.build_tools()
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "fetch_feed",
            "search_cve",
            "extract_iocs",
            "map_mitre",
            "assess_risk",
            "think",
            "done",
        ]
        assert info["fetch_feed"]["parameters"]["required"] == ["feed_name"]
        assert "ctx" not in info["extract_iocs"]["parameters"]["properties"]
        assert "ctx" not in info["assess_risk"]["parameters"]["properties"]
        assert info["think"]["free"] is True
        assert "briefing" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_new_api_end_to_end(self, capsys) -> None:
        threat_intel = _load_threat_intel()

        rc = threat_intel.main(["--scripted", "--max-steps", "15"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "Probe: backend has no generate_with_tools method" in out
        assert "Briefing complete" in out
        assert "Used LLM to classify severity: CRITICAL" in out
        assert "Used LLM for risk assessment" in out
        assert "Daily Threat Intelligence Briefing" in out
