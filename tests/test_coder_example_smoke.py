"""Dogfood tests for the Coder example."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.coder import agent as coder
from looplet import NativeToolProbeResult

pytestmark = pytest.mark.smoke


class TestCoderExample:
    def test_make_tools_uses_decorator_schema_and_helpers(self, tmp_path: Path) -> None:
        cache = coder.FileCache(str(tmp_path))

        registry = coder.make_tools(str(tmp_path), cache)
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "bash",
            "list_dir",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "think",
            "done",
        ]
        assert info["bash"]["parameters"]["required"] == ["command"]
        assert info["list_dir"]["parameters"]["required"] == []
        assert info["list_dir"]["parameters"]["properties"]["depth"]["type"] == "integer"
        assert info["read_file"]["parameters"]["required"] == ["file_path"]
        assert info["think"]["free"] is True
        assert "summary" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_new_api_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = coder.main(
            [
                "Create a tiny add function with tests",
                "--workspace",
                str(tmp_path),
                "--scripted",
                "--max-steps",
                "8",
            ]
        )

        assert rc == 0
        assert (tmp_path / "math_utils.py").exists()
        assert (tmp_path / "test_math_utils.py").exists()
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "Probe: backend has no generate_with_tools method" in out
        assert "bash: python -m pytest -q" in out
        assert "Done: Created math_utils.add with tests." in out

    def test_main_probes_recording_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        probed: list[object] = []

        def fake_probe(llm) -> NativeToolProbeResult:
            probed.append(llm)
            return NativeToolProbeResult(False, "fake probe")

        monkeypatch.setattr(coder, "probe_native_tool_support", fake_probe)

        rc = coder.main(
            [
                "Create a tiny add function with tests",
                "--workspace",
                str(tmp_path),
                "--scripted",
                "--max-steps",
                "8",
            ]
        )

        assert rc == 0
        assert isinstance(probed[0], coder.RecordingLLMBackend)
        assert "Probe: fake probe" in capsys.readouterr().out
