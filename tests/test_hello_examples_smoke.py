"""Dogfood tests for first-run hello examples."""

from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from looplet import MockLLMBackend, RunResult, composable_loop
from looplet.examples import hello_world, ollama_hello

pytestmark = pytest.mark.smoke


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"^```python[^\n]*\n(.*?)^```[ \t]*$", path.read_text(), re.M | re.S)


def test_documented_shell_replay_works_without_provider(tmp_path):
    guide = Path(__file__).resolve().parents[1] / "docs" / "cli.md"
    section = guide.read_text().split("### Replay captured responses\n", 1)[1]
    block = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    commands = block.replace("\\\n", "").strip().splitlines()
    original = None
    for command in commands:
        arguments = shlex.split(command)
        assert arguments[0] == "looplet"
        if arguments[1] == "replay":
            original = (tmp_path / "original-trace" / "trajectory.json").read_bytes()
        result = subprocess.run(
            [sys.executable, "-m", "looplet", *arguments[1:]],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    replayed = json.loads(result.stdout)
    assert replayed["completed"]
    assert Path(replayed["trace_dir"]).is_relative_to(tmp_path / "replayed-workspace")
    assert original == (tmp_path / "original-trace" / "trajectory.json").read_bytes()


def test_documented_summary_omits_private_evidence(tmp_path, monkeypatch):
    from looplet import EvalContext, save_eval_run

    guide = Path(__file__).resolve().parents[1] / "docs" / "provenance.md"
    block = next(block for block in _python_blocks(guide) if "reviewed-summary.json" in block)
    save_eval_run(
        tmp_path / "eval-runs" / "run-001" / "hello",
        context=EvalContext(
            steps=[], final_output="PRIVATE_RESPONSE", artifacts={"raw": "PRIVATE_ARTIFACT"}
        ),
    )
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(guide), "exec"), {"__name__": "__main__"})
    exported = (tmp_path / "reviewed-summary.json").read_text()
    assert json.loads(exported) == {"completed": False, "step_count": 0}
    assert "PRIVATE" not in exported
    assert exported.endswith("\n")
    with pytest.raises(FileExistsError):
        exec(compile(block, str(guide), "exec"), {"__name__": "__main__"})


@pytest.mark.parametrize("document", ["README.md", "docs/index.md"])
def test_documented_first_loop(document: str, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / document
    blocks = _python_blocks(path)
    assert blocks, f"No Python example in {document}"
    observations = []

    def observed_loop(**kwargs):
        backend = kwargs["llm"]
        registry = kwargs["tools"]
        generate = Mock(wraps=backend.generate)
        dispatch = Mock(wraps=registry.dispatch)
        monkeypatch.setattr(backend, "generate", generate)
        monkeypatch.setattr(registry, "dispatch", dispatch)
        observations.append((generate, dispatch))
        return composable_loop(**kwargs)

    monkeypatch.setattr("looplet.composable_loop", observed_loop)
    namespace = {"__name__": "__main__"}
    exec(compile(blocks[0], str(path), "exec"), namespace)
    state = namespace["state"]
    assert [step.tool_call.tool for step in state.steps] == ["lookup_owner", "done"]
    assert state.steps[0].tool_result.data == {
        "service": "payments",
        "owner": "fintech-platform",
    }
    assert state.termination_reason == "done"
    assert RunResult.from_state(state).completed
    assert observations[0][0].call_count == 2
    assert observations[0][1].call_count == 2

    first_loop = _python_blocks(root / "README.md")[0]
    assert ast.dump(ast.parse(blocks[0])) == ast.dump(ast.parse(first_loop))


def test_quickstart_runs_grades_captures_and_replays(tmp_path: Path, monkeypatch) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    blocks = _python_blocks(guide)
    assert blocks
    from_env = Mock()
    monkeypatch.setattr("looplet.OpenAIBackend.from_env", from_env)
    module = ModuleType("first_loop")
    monkeypatch.setitem(sys.modules, "first_loop", module)
    monkeypatch.chdir(tmp_path)
    for block in blocks:
        exec(compile(block, str(guide), "exec"), module.__dict__)

    assert not hasattr(module, "result")
    assert not (tmp_path / "traces").exists()
    from_env.assert_not_called()
    module.test_owner()
    module.test_repeated_runs_are_independent()
    native_responses = module.scripted_backend()
    native_backend = SimpleNamespace(
        generate=Mock(side_effect=AssertionError("live example must use native tools")),
        generate_with_tools=Mock(wraps=native_responses.generate_with_tools),
    )
    from_env.return_value = native_backend
    namespace = {"__name__": "__main__"}
    for block in blocks:
        exec(compile(block, str(guide), "exec"), namespace)
    assert namespace["report"].passed
    assert namespace["captured_result"].output == namespace["replayed_result"].output
    assert namespace["captured_backend"].calls == 2
    assert namespace["state"].termination_reason == "stream_closed"
    from_env.assert_called_once_with()
    native_backend.generate.assert_not_called()
    assert native_backend.generate_with_tools.call_count == 2
    assert [step.tool_call.tool for step in namespace["result"].steps] == ["lookup_owner", "done"]
    assert [step.tool_call.tool for step in namespace["state"].steps] == ["lookup_owner"]
    assert not RunResult.from_state(namespace["state"]).completed
    trace_dir = namespace["trace_dir"]
    assert (trace_dir / "trajectory.json").is_file()
    assert (trace_dir / "manifest.jsonl").is_file()
    assert namespace["saved_report"].passed
    assert namespace["saved_run"].context.expected == namespace["EXPECTED_OWNER"]
    trajectory = json.loads((trace_dir / "trajectory.json").read_text())
    assert "expected" not in trajectory["task"]
    assert json.loads((trace_dir / "expected.json").read_text()) == namespace["EXPECTED_OWNER"]

    from looplet.evals import eval_cli

    (tmp_path / "first_loop.py").write_text("\n\n".join(blocks))
    assert eval_cli([str(trace_dir), "--evals", str(tmp_path / "first_loop.py"), "--json"]) == 0
    original = (trace_dir / "trajectory.json").read_bytes()
    capture_block = next(block for block in blocks if "save_eval_run(trace_dir" in block)
    exec(compile(capture_block, str(guide), "exec"), namespace)
    assert namespace["trace_dir"] != trace_dir
    assert (trace_dir / "trajectory.json").read_bytes() == original
    namespace["result"].output["owner"] = "wrong-owner"
    assert not module.assess_result(namespace["result"]).passed


@pytest.mark.parametrize(
    ("owner", "summary", "passed"),
    [
        ("fintech-platform", "The payments owner is fintech-platform.", True),
        ("fintech-platform", "fintech-platform owns payments.", True),
        ("fintech-platform", "", True),
        ("wrong-owner", "The payments owner is fintech-platform.", False),
    ],
)
def test_quickstart_grades_structured_facts_not_prose(owner, summary, passed):
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    namespace = {"__name__": "first_loop"}
    for block in _python_blocks(guide)[:2]:
        exec(compile(block, str(guide), "exec"), namespace)
    backend = MockLLMBackend(
        [
            json.dumps({"tool": "lookup_owner", "args": {"service": "payments"}}),
            json.dumps(
                {
                    "tool": "done",
                    "args": {"service": "payments", "owner": owner, "summary": summary},
                }
            ),
        ]
    )

    result = namespace["run_agent"](backend)

    assert result.completed
    assert namespace["assess_result"](result).passed is passed


@pytest.mark.parametrize(("max_steps", "completed"), [(None, False), (3, True)])
def test_quickstart_run_budget_is_configurable(max_steps, completed):
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    namespace = {"__name__": "first_loop"}
    for block in _python_blocks(guide)[:2]:
        exec(compile(block, str(guide), "exec"), namespace)
    lookup = json.dumps({"tool": "lookup_owner", "args": {"service": "payments"}})
    backend = MockLLMBackend(
        [
            lookup,
            lookup,
            json.dumps(
                {"tool": "done", "args": {"service": "payments", "owner": "fintech-platform"}}
            ),
        ]
    )
    options = {} if max_steps is None else {"max_steps": max_steps}

    result = namespace["run_agent"](backend, **options)

    assert result.completed is completed
    assert namespace["assess_result"](result).passed is completed
    assert result.termination_reason == ("done" if completed else "budget_exhausted")


def test_quickstart_python_test_and_definition_commands(tmp_path: Path) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    blocks = _python_blocks(guide)
    (tmp_path / "first_loop.py").write_text("\n\n".join(blocks[:2]))
    (tmp_path / "test_first_loop.py").write_text(blocks[2])
    tested = subprocess.run(
        [sys.executable, "-m", "pytest", "test_first_loop.py", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr
    assert "2 passed" in tested.stdout
    checked = subprocess.run(
        [sys.executable, "-m", "looplet", "eval", "check", "first_loop.py", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    report = json.loads(checked.stdout)
    assert report["valid"]
    assert [grader["name"] for grader in report["graders"]] == ["eval_owner"]


def test_eval_introduction_is_self_contained() -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "evals.md"
    namespace = {}
    exec(compile(_python_blocks(guide)[0], str(guide), "exec"), namespace)
    assert namespace["report"].passed


def test_documented_report_contract(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/tutorial.md"
    blocks = re.findall(r'^```\w+ title="([^"]+)"\n(.*?)^```[ \t]*$', path.read_text(), re.M | re.S)
    cartridge = tmp_path / "report_agent.cartridge"
    shutil.copytree(root / "examples/regression_demo/report_agent.cartridge", cartridge)
    for name, code in blocks:
        if name.startswith("tests/") or name.startswith(".github/"):
            continue
        target = cartridge / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code)

    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    test_code = dict(blocks)["tests/test_report_agent.py"]
    exec(compile(test_code, str(path), "exec"), namespace)
    namespace["test_report_contract"](tmp_path)


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
            yield from ()

        monkeypatch.setattr(backends, "make_backend", resolve)
        monkeypatch.setattr(hello_world, "composable_loop", execute)
        assert hello_world.main(["--model", "selected-model"]) == 1
        assert selected == {"model": "selected-model"}
        assert run["llm"] is backend
        assert run["config"].use_native_tools is True
        assert run["task"] == {"description": "Greet Alice and Bob, then finish."}
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

        def execute(**kwargs):
            yield from ()

        monkeypatch.setattr(hello_world, "composable_loop", execute)
        assert hello_world.main(["--base-url", "http://localhost:12345/v1"]) == 1
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
        assert "Result: completed (done)" in out

    def test_incomplete_scripted_run_is_not_reported_as_completed(self, capsys) -> None:
        assert hello_world.main(["--scripted", "--max-steps", "1"]) == 1
        output = capsys.readouterr().out
        assert "Result: stopped (" in output


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
