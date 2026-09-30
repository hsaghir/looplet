"""Dogfood tests for the packaged Coding Agent example."""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.examples import coding_agent

pytestmark = pytest.mark.smoke


class TestPackagedCodingAgent:
    def test_run_returns_trace_and_captures_replayable_model_calls(self, tmp_path: Path) -> None:
        from looplet import MockLLMBackend
        from looplet.provenance import _load_trace_calls

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        evidence = tmp_path / "trace"
        result = coding_agent.run_coding_agent(
            MockLLMBackend(coding_agent.scripted_responses(), cycle=False),
            "Implement an add function with tests",
            workspace=str(workspace),
            trace_dir=str(evidence),
            max_steps=8,
            use_native_tools=False,
        )

        assert result["llm_calls"] == 4
        assert len(result["steps"]) == 4
        assert len(_load_trace_calls(evidence)) == 4

    def test_build_tools_uses_decorator_schema_and_helpers(self, tmp_path: Path) -> None:
        registry = coding_agent.build_tools(str(tmp_path))
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "bash",
            "read",
            "write",
            "edit",
            "glob",
            "grep",
            "think",
            "done",
        ]
        assert info["bash"]["parameters"]["properties"]["command"]["type"] == "string"
        assert info["bash"]["parameters"]["required"] == ["command"]
        assert info["read"]["parameters"]["required"] == ["file_path"]
        assert info["write"]["parameters"]["required"] == ["file_path", "content"]
        assert info["grep"]["parameters"]["properties"]["path"]["default"] == "."
        assert info["think"]["free"] is True
        assert "summary" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_harness_end_to_end(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = coding_agent.main(
            [
                "Implement an add function with tests",
                "--scripted",
                "--workspace",
                str(tmp_path),
                "--max-steps",
                "8",
            ]
        )

        assert rc == 0
        assert (tmp_path / "math_utils.py").exists()
        assert (tmp_path / "test_math_utils.py").exists()
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "Tests: PASSED" in out
        assert "Step 4: done" in out
        assert "Eval results" not in out
