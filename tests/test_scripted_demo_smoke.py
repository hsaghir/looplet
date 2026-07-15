"""Dogfood tests for the README GIF scripted demo."""

from __future__ import annotations

import pytest

from looplet.examples import scripted_demo

pytestmark = pytest.mark.smoke


class TestScriptedDemo:
    def test_build_tools_uses_decorator_schema_and_done(self) -> None:
        rows = [
            {"id": 1, "user": "alice", "status": "paid"},
            {"id": 2, "user": "bob", "status": "cancelled"},
        ]

        registry = scripted_demo.build_tools(rows)
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == ["head", "count_by_status", "delete_rows", "done"]
        assert info["head"]["parameters"]["properties"]["n"]["type"] == "integer"
        assert info["head"]["parameters"]["required"] == ["n"]
        assert info["count_by_status"]["parameters"]["required"] == []
        assert info["delete_rows"]["parameters"]["required"] == ["where_status"]
        assert "ctx" not in info["delete_rows"]["parameters"]["properties"]
        assert "summary" in info["done"]["parameters"]["properties"]

    def test_main_accepts_argv_and_runs_without_delays(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(scripted_demo.time, "sleep", lambda _seconds: None)

        rc = scripted_demo.main([])

        assert rc == 0
        out = capsys.readouterr().out
        assert "$ python -m looplet.examples.scripted_demo" in out
        assert "APPROVAL NEEDED" in out
        assert "delete_rows(where_status='cancelled')" in out
        assert "scripted run" in out
