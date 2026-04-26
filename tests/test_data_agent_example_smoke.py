"""Dogfood tests for the packaged Data Agent example."""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.examples import data_agent

pytestmark = pytest.mark.smoke


class TestDataAgentExample:
    def test_build_tools_uses_decorator_schema_and_helpers(self) -> None:
        registry = data_agent.build_tools()
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "describe_csv",
            "head_csv",
            "groupby_count",
            "delete_rows",
            "think",
            "done",
        ]
        assert info["describe_csv"]["parameters"]["required"] == ["path"]
        assert info["head_csv"]["parameters"]["properties"]["n"]["type"] == "integer"
        assert info["head_csv"]["parameters"]["required"] == ["path"]
        assert info["groupby_count"]["parameters"]["required"] == ["path", "column"]
        assert "ctx" not in info["delete_rows"]["parameters"]["properties"]
        assert info["think"]["free"] is True
        assert "summary" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_new_api_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", tmp_path / "checkpoints")

        rc = data_agent.main(["--scripted", "--auto-approve", "--clean"])

        assert rc == 0
        assert any((tmp_path / "checkpoints").glob("*.json"))
        out = capsys.readouterr().out
        assert "# tool protocol: json-text" in out
        assert "# probe: backend has no generate_with_tools method" in out
        assert "describe_csv" in out
        assert "delete_rows" in out
        assert "done(summary=inspected orders.csv and removed cancellations)" in out

    def test_mock_alias_remains_compatible(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", tmp_path / "checkpoints")

        rc = data_agent.main(["--mock", "--auto-approve", "--clean"])

        assert rc == 0
        assert any((tmp_path / "checkpoints").glob("*.json"))
