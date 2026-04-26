"""Dogfood tests for the Dependency Doctor example."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def _load_dep_doctor():
    path = Path(__file__).parents[1] / "examples" / "dep_doctor" / "agent.py"
    spec = importlib.util.spec_from_file_location("dep_doctor_agent", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDependencyDoctorExample:
    def test_build_tools_uses_decorator_schema_and_helpers(self) -> None:
        dep_doctor = _load_dep_doctor()

        registry = dep_doctor.build_tools()
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "detect_files",
            "parse_deps",
            "check_package",
            "check_license",
            "find_alternatives",
            "think",
            "done",
        ]
        assert info["detect_files"]["parameters"]["properties"]["project_dir"]["type"] == "string"
        assert info["check_license"]["parameters"]["required"] == [
            "project_license",
            "dep_license",
        ]
        assert "ctx" not in info["find_alternatives"]["parameters"]["properties"]
        assert info["think"]["free"] is True
        assert "report" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_new_api_end_to_end(self, tmp_path: Path, capsys) -> None:
        dep_doctor = _load_dep_doctor()
        (tmp_path / "requirements.txt").write_text(
            "requests>=2.31\npyyaml==6.0.1\nabandoned-lib==0.3.1\n",
            encoding="utf-8",
        )

        rc = dep_doctor.main([str(tmp_path), "--scripted", "--max-steps", "12"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "Probe: backend has no generate_with_tools method" in out
        assert "Report complete" in out
        assert "Used LLM to find alternatives for abandoned-lib" in out
        assert "Dependency Doctor Report" in out
