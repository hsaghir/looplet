"""Dogfood tests for first-run hello examples."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from looplet.examples import hello_world, ollama_hello

pytestmark = pytest.mark.smoke


class TestHelloWorldExample:
    def test_live_run_uses_shared_provider_resolution(self, monkeypatch, capsys) -> None:
        from looplet import backends

        backend = SimpleNamespace(_model="selected-model")
        selected = {}
        run = {}

        def resolve(**kwargs):
            selected.update(kwargs)
            return backend

        def execute(**kwargs):
            run.update(kwargs)
            return iter(())

        monkeypatch.setattr(backends, "make_backend", resolve)
        monkeypatch.setattr(hello_world, "composable_loop", execute)
        assert hello_world.main(["--model", "selected-model"]) == 0
        assert selected == {"model": "selected-model"}
        assert run["llm"] is backend
        assert run["config"].use_native_tools is True
        assert "Model: selected-model" in capsys.readouterr().out

    def test_explicit_url_retains_openai_override(self, monkeypatch) -> None:
        from looplet import backends

        selected = {}

        def construct(**kwargs):
            selected.update(kwargs)
            return SimpleNamespace(_model=kwargs["model"])

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.setattr(backends, "OpenAIBackend", construct)
        monkeypatch.setattr(
            backends, "make_backend", lambda **kwargs: pytest.fail("explicit URL must win")
        )
        monkeypatch.setattr(hello_world, "composable_loop", lambda **kwargs: iter(()))
        assert hello_world.main(["--base-url", "http://localhost:12345/v1"]) == 0
        assert selected == {
            "base_url": "http://localhost:12345/v1",
            "api_key": "x",
            "model": "gpt-4o",
        }

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


def test_coding_example_uses_shared_provider_defaults(monkeypatch) -> None:
    from looplet import backends
    from looplet.examples import coding_agent

    backend = object()
    selected = {}

    def resolve(**kwargs):
        selected.update(kwargs)
        return backend

    monkeypatch.setattr(backends, "make_backend", resolve)
    assert coding_agent._get_llm(model="chosen-model") is backend
    assert selected == {"model": "chosen-model"}


def test_coding_example_retains_explicit_connection_overrides(monkeypatch) -> None:
    from looplet import backends
    from looplet.examples import coding_agent

    backend = object()
    selected = {}

    def construct(**kwargs):
        selected.update(kwargs)
        return backend

    monkeypatch.setattr(backends, "OpenAIBackend", construct)
    assert (
        coding_agent._get_llm(
            base_url="http://localhost:12345/v1", api_key="example-key", model="chosen-model"
        )
        is backend
    )
    assert selected == {
        "base_url": "http://localhost:12345/v1",
        "api_key": "example-key",
        "model": "chosen-model",
    }


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
