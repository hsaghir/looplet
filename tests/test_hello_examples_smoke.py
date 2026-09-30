"""Dogfood tests for first-run hello examples."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from looplet.examples import hello_world, ollama_hello

pytestmark = pytest.mark.smoke


class TestHelloWorldExample:
    def test_completion_grader_requires_accepted_completion(self) -> None:
        rejected = SimpleNamespace(tool_sequence=["done"], completed=False)
        accepted = SimpleNamespace(tool_sequence=["done"], completed=True)

        assert hello_world.eval_completed(rejected) is False
        assert hello_world.eval_completed(accepted) is True

    def test_build_tools_uses_decorator_schema_and_done(self) -> None:
        registry = hello_world.build_tools()
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == ["greet", "done"]
        assert info["greet"]["parameters"]["properties"]["name"]["type"] == "string"
        assert info["greet"]["parameters"]["required"] == ["name"]
        assert "answer" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_first_run_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = hello_world.main(["--scripted"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "greet(name=Alice)" in out
        assert "greet(name=Bob)" in out
        assert "done(answer=" in out


class TestOllamaHelloExample:
    def test_build_tools_uses_decorator_schema_and_done(self) -> None:
        registry = ollama_hello.build_tools()
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == ["greet", "done"]
        assert info["greet"]["parameters"]["properties"]["name"]["type"] == "string"
        assert info["greet"]["parameters"]["required"] == ["name"]
        assert "answer" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_ollama_first_run_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = ollama_hello.main(["--scripted"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Tool protocol: json-text" in out
        assert "greet(name=Alice)" in out
        assert "greet(name=Bob)" in out
        assert "done(answer=" in out
