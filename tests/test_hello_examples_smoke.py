"""Dogfood tests for first-run hello examples."""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from looplet import MockLLMBackend, RunResult, async_composable_loop, composable_loop
from looplet.examples import hello_world, ollama_hello

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("loop_function", [composable_loop, async_composable_loop])
def test_public_loop_help_documents_inputs_and_cleanup(loop_function):
    documentation = inspect.getdoc(loop_function)
    assert documentation is not None
    for parameter in inspect.signature(loop_function).parameters:
        assert f"{parameter}:" in documentation
    assert "dict" in documentation
    assert "task string" in documentation
    assert "Python functions" in documentation
    assert "Explicit registries stay unchanged" in documentation
    assert "override" in documentation
    assert "close()" in documentation


@pytest.mark.parametrize("guide_name", ["README.md", "docs/index.md", "docs/install.md"])
def test_frontdoor_install_matches_its_documented_release_stage(guide_name):
    guide = Path(__file__).resolve().parents[1] / guide_name
    content = guide.read_text()
    command = re.findall(r"^```bash[^\n]*\n(.*?)^```[ \t]*$", content, re.M | re.S)[0]
    arguments = shlex.split(command.replace("\\\n", " "))
    if guide_name == "docs/install.md":
        assert "looplet==0.4.1" in arguments
    else:
        assert "unreleased" in content[: content.index(command)].lower()
        assert any(
            re.fullmatch(
                r"looplet\[openai\] @ https://github.com/hsaghir/looplet/archive/[a-f0-9]{40}\.zip",
                argument,
            )
            for argument in arguments
        )
    assert "-e" not in arguments


@pytest.mark.parametrize("guide_name", ["README.md", "docs/index.md", "docs/python.md"])
def test_python_first_intro_executes_the_same_loop(guide_name, tmp_path, monkeypatch):
    from looplet import OpenAIBackend

    root = Path(__file__).resolve().parents[1]
    content = (root / guide_name).read_text()
    code = re.findall(r"^```python[^\n]*\n(.*?)^```[ \t]*$", content, re.M | re.S)[0]
    canonical = re.findall(
        r"^```python[^\n]*\n(.*?)^```[ \t]*$", (root / "docs/python.md").read_text(), re.M | re.S
    )[0]
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(canonical))
    backend = MockLLMBackend(
        [
            '{"tool":"lookup_owner","args":{"service":"payments"}}',
            '{"tool":"done","args":{"summary":"fintech-platform"}}',
        ],
        cycle=False,
    )
    monkeypatch.setattr(OpenAIBackend, "from_env", staticmethod(lambda: backend))
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(code, guide_name, "exec"), namespace)
    assert namespace["lookup_owner"]("payments") == "fintech-platform"
    assert namespace["step"].tool_result.tool == "done"
    assert backend.calls == 2
    assert not {"LoopConfig", "DefaultState", "AgentPreset", "tools_from"}.intersection(namespace)


def test_python_first_guide_all_python_fences_execute(tmp_path, monkeypatch):
    from looplet import OpenAIBackend
    from looplet.testing import AsyncMockLLMBackend

    guide = Path(__file__).resolve().parents[1] / "docs/python.md"
    blocks = re.findall(r"^```python[^\n]*\n(.*?)^```[ \t]*$", guide.read_text(), re.M | re.S)
    responses = [
        '{"tool":"lookup_owner","args":{"service":"payments"}}',
        '{"tool":"done","args":{"summary":"fintech-platform"}}',
    ]
    monkeypatch.setattr(
        OpenAIBackend, "from_env", staticmethod(lambda: MockLLMBackend(responses, cycle=False))
    )
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for index, code in enumerate(blocks):
        namespace["llm"] = MockLLMBackend(responses, cycle=False)
        exec(compile(code, f"python.md fence {index + 1}", "exec"), namespace)
    assert namespace["result"].completed

    import asyncio

    backend = AsyncMockLLMBackend(responses, cycle=False)
    asyncio.run(namespace["inspect_async"](backend))
    assert backend.calls == 1


def test_quickstart_starts_with_the_release_and_defers_preview_setup():
    docs = Path(__file__).resolve().parents[1] / "docs"
    guide = (docs / "quickstart.md").read_text()
    install = (docs / "install.md").read_text()
    pin = re.search(
        r"looplet @ (https://github.com/hsaghir/looplet/archive/[a-f0-9]{40}\.zip)", install
    )
    assert pin is not None
    commands = re.findall(r"^```bash[^\n]*\n(.*?)^```[ \t]*$", guide, re.M | re.S)
    first = shlex.split(commands[0])
    assert "looplet==0.4.1" in first
    assert ".venv" in first
    optional = guide.index("## Optional experiment starter\n")
    for name in ("first_loop.py", "test_first_loop.py", "first_loop_replay.py"):
        assert guide.index(f'title="{name}"') < optional
    assert guide.index(pin.group(1)) > optional
    preview = shlex.split(next(command for command in commands if pin.group(1) in command))
    assert f"looplet @ {pin.group(1)}" in preview
    assert ".venv-looplet-preview" in preview
    assert "Keep this preview environment active" in guide[optional:]
    installs = re.findall(r'python -m pip install "([^"]+)"', guide)
    assert set(installs) == {
        "looplet==0.4.1",
        "looplet[openai]==0.4.1",
        f"looplet @ {pin.group(1)}",
        f"looplet[openai] @ {pin.group(1)}",
    }


def test_quickstart_provider_extra_keeps_the_released_version():
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    section = guide.read_text().split("## Use the same agent with a real model", 1)[1]
    command = re.findall(r"^```bash[^\n]*\n(.*?)^```[ \t]*$", section, re.M | re.S)[0]
    arguments = shlex.split(command.replace("\\\n", " "))
    assert "looplet[openai]==0.4.1" in arguments
    assert "-e" not in arguments
    assert not any("github.com" in argument for argument in arguments)


def test_factory_shell_workflow_has_a_pinned_preview_install():
    docs = Path(__file__).resolve().parents[1] / "docs"
    install = (docs / "install.md").read_text()
    section = install.split("## Try unreleased factory commands\n", 1)[1].split("\n## ", 1)[0]
    command = re.findall(r"^```bash[^\n]*\n(.*?)^```[ \t]*$", section, re.M | re.S)[0]
    arguments = shlex.split(command.replace("\\\n", " "))
    assert ".venv-looplet-preview" in arguments
    assert "-e" not in arguments
    assert any(
        re.fullmatch(
            r"looplet @ https://github.com/hsaghir/looplet/archive/[a-f0-9]{40}\.zip", item
        )
        for item in arguments
    )
    assert "source fingerprint" in section
    assert "Both builds report version `0.4.1`" in section
    assert "(install.md#try-unreleased-factory-commands)" in (docs / "quickstart.md").read_text()


def test_python_eval_guidance_distinguishes_released_and_preview_runners():
    docs = Path(__file__).resolve().parents[1] / "docs"
    guide = (docs / "evals.md").read_text()
    assert "Published `0.4.1` requires a cartridge for CLI case execution." in guide
    assert "looplet eval run --factory first_loop:build --evals first_loop.py" in guide
    for name in ("evals.md", "cli.md"):
        assert "(install.md#try-unreleased-factory-commands)" in (docs / name).read_text()


def test_factory_run_help_uses_agent_target_name(capsys):
    from looplet.__main__ import main

    with pytest.raises(SystemExit) as exit_status:
        main(["run", "--help"])
    assert exit_status.value.code == 0
    output = capsys.readouterr().out
    assert "AGENT task" in output
    assert "bundle task" not in output
    assert "Python module:factory reference" in output


@pytest.mark.parametrize("completed", [False, True])
def test_primary_cli_run_example_requires_completion(tmp_path, completed):
    guide = Path(__file__).resolve().parents[1] / "docs" / "cli.md"
    section = guide.read_text().split("### `looplet run <path> <task>`", 1)[1]
    command = re.findall(r"^```bash[^\n]*\n(.*?)^```[ \t]*$", section, re.M | re.S)[0]
    arguments = shlex.split(command.replace("\\\n", " "))
    assert arguments[:2] == ["looplet", "run"]
    assert "--require-completed" in arguments
    arguments[arguments.index("--max-steps") + 1] = "1"

    scaffold = subprocess.run(
        [
            sys.executable,
            "-m",
            "looplet",
            "new",
            "Finish with a summary",
            "./agent.cartridge",
            "--offline",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert scaffold.returncode == 0, scaffold.stderr
    response = {
        "tool": "done" if completed else "never_registered",
        "args": {"summary": "Reviewed"} if completed else {},
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "looplet",
            *arguments[1:],
            "--scripted-response",
            json.dumps(response),
            "--json",
            "--no-trace",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == (0 if completed else 1), result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["completed"] is completed
    assert outcome["termination_reason"] == ("done" if completed else "budget_exhausted")


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"^```python[^\n]*\n(.*?)^```[ \t]*$", path.read_text(), re.M | re.S)


def _named_python_blocks(path: Path) -> dict[str, str]:
    return dict(
        re.findall(r'^```python title="([^"]+)"\n(.*?)^```[ \t]*$', path.read_text(), re.M | re.S)
    )


def _quickstart_agent_source(guide: Path) -> str:
    first_loop = _named_python_blocks(guide)["first_loop.py"]
    main = ast.parse(first_loop).body[-1]
    assert isinstance(main, ast.If)
    declarations = "".join(first_loop.splitlines(keepends=True)[: main.lineno - 1])
    grading = next(block for block in _python_blocks(guide) if "def assess_result(" in block)
    factory = next(block for block in _python_blocks(guide) if "def build(runtime):" in block)
    return declarations + "\n\n" + factory + "\n\n" + grading


def _run_documented_shell(guide: Path, name: str, project: Path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("The documented recipe uses Bash")
    match = re.search(
        rf'^```bash title="{re.escape(name)}"\n(.*?)^```[ \t]*$',
        guide.read_text(),
        re.M | re.S,
    )
    assert match is not None
    script = project / name
    script.write_text(match.group(1))
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    return subprocess.run(
        [bash, "-e", "-u", "-o", "pipefail", str(script)],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.parametrize("owner", ["fintech-platform", "data-platform"])
def test_documented_custom_agent_starter_is_complete_and_repeatable(tmp_path, owner):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    project = tmp_path / "custom agent with spaces"
    project.mkdir()
    exported = _run_documented_shell(guide, "copy_agent_starter.sh", project)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    agent = project / "my_agent.py"
    assert agent.read_bytes() == Path(quickstart.__file__).read_bytes()
    assert list(project.glob("*.py")) == [agent]
    agent.write_text(agent.read_text().replace("fintech-platform", owner))
    authored_source = agent.read_bytes()

    first = _run_documented_shell(guide, "custom_agent_workflow.sh", project)
    assert first.returncode == 0, first.stdout + first.stderr
    assert owner in first.stdout
    experiments = project / ".looplet" / "experiments"
    before = {path: path.read_bytes() for path in experiments.rglob("*") if path.is_file()}
    assert before

    second = _run_documented_shell(guide, "custom_agent_workflow.sh", project)
    assert second.returncode == 0, second.stdout + second.stderr
    assert owner in second.stdout
    assert all(path.read_bytes() == content for path, content in before.items())
    directories = list(experiments.iterdir())
    assert len(directories) == 2
    for directory in directories:
        report = json.loads((directory / "experiment.json").read_text())
        assert report["passed"]
        assert report["capture_dir"] == "capture"
        replayed = load_eval_run(directory / report["replay_dir"])
        assert replayed.context.completed and replayed.context.final_output["owner"] == owner
        assert replayed.results and all(result.passed for result in replayed.results)
        saved = load_eval_run(directory / "capture")
        assert saved.context.completed
        assert saved.context.expected["owner"] == owner
        assert saved.context.final_output["owner"] == owner
        assert (
            "expected"
            not in json.loads((directory / "capture" / "trajectory.json").read_text())["task"]
        )

    duplicate = _run_documented_shell(guide, "copy_agent_starter.sh", project)
    assert duplicate.returncode != 0
    assert "FileExistsError" in duplicate.stderr
    assert agent.read_bytes() == authored_source


def test_documented_one_file_case_uses_the_exported_agent(tmp_path):
    from looplet import load_eval_run

    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    project = tmp_path / "case project with spaces"
    project.mkdir()
    exported = _run_documented_shell(guide, "copy_agent_starter.sh", project)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    block = re.search(
        r'^```json title="owner_case\.json"\n(.*?)^```[ \t]*$', guide.read_text(), re.M | re.S
    )
    assert block is not None
    case = json.loads(block.group(1))
    (project / "owner_case.json").write_text(block.group(1))
    result = _run_documented_shell(guide, "one_file_case.sh", project)
    assert result.returncode == 0, result.stdout + result.stderr
    experiment = next((project / ".looplet/experiments").iterdir())
    assert json.loads((experiment / "experiment.json").read_text())["passed"]
    for name in ("capture", "replay"):
        saved = load_eval_run(experiment / name)
        assert saved.case is not None
        assert saved.case.id == case["id"]
        assert saved.case.task == case["task"]
        assert saved.context.expected == case["expected"]
        assert saved.results and all(result.passed for result in saved.results)


def test_documented_one_file_factory_inspection(tmp_path):
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    _run_documented_shell(guide, "copy_agent_starter.sh", tmp_path)
    result = _run_documented_shell(guide, "inspect_agent.sh", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["factory"] == "my_agent:build"
    assert report["mode"] == "runtime"
    assert report["runtime_validated"] is True
    assert {tool["name"] for tool in report["tools"]} == {"lookup_owner", "done"}


def test_documented_one_file_provider_handoff_uses_the_exported_agent(tmp_path, monkeypatch):
    from looplet.examples import quickstart

    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    script = re.search(
        r'^```bash title="one_file_live\.sh"\n(.*?)^```[ \t]*$', guide.read_text(), re.M | re.S
    )
    assert script is not None
    install = (guide.parent / "install.md").read_text()
    pin = re.search(
        r"looplet @ (https://github.com/hsaghir/looplet/archive/[a-f0-9]{40}\.zip)", install
    )
    assert pin is not None and f"looplet[openai] @ {pin.group(1)}" in script.group(1)
    command = next(
        line for line in script.group(1).splitlines() if line.startswith("python my_agent.py ")
    )
    arguments = shlex.split(command)
    assert arguments[:2] == ["python", "my_agent.py"]
    module = ModuleType("my_agent")
    monkeypatch.setitem(sys.modules, "my_agent", module)
    monkeypatch.chdir(tmp_path)
    exec(compile(Path(quickstart.__file__).read_text(), "my_agent.py", "exec"), module.__dict__)
    backend = quickstart.scripted_backend()
    factory = Mock(return_value=backend)
    monkeypatch.setattr("looplet.OpenAIBackend.from_env", factory)

    assert module.main(arguments[2:]) == 0

    factory.assert_called_once_with()
    assert backend.calls == 2
    assert not (tmp_path / "first_loop.py").exists()
    assert len(list((tmp_path / ".looplet/experiments").iterdir())) == 1


def test_copied_quickstart_exports_cli_factory(tmp_path):
    from looplet.examples import quickstart

    agent = tmp_path / "my_agent.py"
    agent.write_bytes(Path(quickstart.__file__).read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "looplet",
            "run",
            "my_agent:build",
            "Find the payments owner, then finish.",
            "--scripted-response",
            '{"tool":"lookup_owner","args":{"service":"payments"}}',
            "--scripted-response",
            '{"tool":"done","args":{"service":"payments","owner":"fintech-platform"}}',
            "--max-steps",
            "2",
            "--json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["completed"]
    assert outcome["grading_status"] == "not_graded"
    assert outcome["result"]["owner"] == "fintech-platform"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("saved_case", [False, True])
@pytest.mark.parametrize("with_evals", [False, True])
def test_copied_quickstart_factory_replays_captured_protocol(
    tmp_path, native, saved_case, with_evals
):
    from looplet import EvalCase, EvalHook, ProvenanceSink, load_eval_run, save_eval_run
    from looplet.examples import quickstart
    from looplet.provenance import _load_trace_calls

    (tmp_path / "my_agent.py").write_bytes(Path(quickstart.__file__).read_bytes())
    source = tmp_path / "captured run"
    sink = ProvenanceSink(dir=source)
    grading = EvalHook(evaluators=[quickstart.eval_owner], expected=quickstart.EXPECTED_OWNER)
    quickstart.run_agent(
        sink.wrap_llm(quickstart.scripted_backend()),
        hooks=[sink.trajectory_hook(), grading],
        use_native_tools=native,
    )
    sink.flush()
    case = (
        EvalCase(
            id="owner",
            task={"goal": "Find the payments owner, then finish."},
            expected=quickstart.EXPECTED_OWNER,
        )
        if saved_case
        else None
    )
    save_eval_run(source, recorder=sink.trajectory_hook(), eval_hook=grading, case=case)
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    arguments = [
        sys.executable,
        "-m",
        "looplet",
        "replay",
        str(source),
        "--factory",
        "my_agent:build",
        "--workspace",
        str(tmp_path / "replay workspace"),
        "--json",
    ]
    if with_evals:
        arguments.extend(["--evals", "my_agent.py"])
    if not saved_case and with_evals:
        seed = tmp_path / "legacy seed inputs"
        seed.mkdir()
        (seed / "notes.txt").write_text("fresh replay input")
        arguments.extend(["--seed-from", str(seed)])
    result = subprocess.run(
        arguments,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["completed"]
    assert outcome["grading_status"] == "not_graded"
    assert outcome["result"]["owner"] == "fintech-platform"
    if not saved_case and with_evals:
        assert (tmp_path / "replay workspace" / "notes.txt").read_text() == "fresh replay input"
    replayed = load_eval_run(outcome["trace_dir"])
    assert replayed.case is not None
    assert replayed.context.expected == quickstart.EXPECTED_OWNER
    assert replayed.case.task == {"goal": "Find the payments owner, then finish."}
    assert (
        "expected"
        not in json.loads((Path(outcome["trace_dir"]) / "trajectory.json").read_text())["task"]
    )
    assert [call["method"] for call in _load_trace_calls(Path(outcome["trace_dir"]))] == [
        call["method"] for call in _load_trace_calls(source)
    ]
    assessed = subprocess.run(
        [
            sys.executable,
            "-m",
            "looplet",
            "eval",
            outcome["trace_dir"],
            "--evals",
            "my_agent.py",
            "--json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert assessed.returncode == 0, assessed.stdout + assessed.stderr
    assert json.loads(assessed.stdout)["passed"]
    assert {path: path.read_bytes() for path in original} == original


def test_packaged_quickstart_runs_without_authored_files(tmp_path, monkeypatch, capsys):
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    first = quickstart.run_agent(quickstart.scripted_backend())
    second = quickstart.run_agent(quickstart.scripted_backend())
    assert first.completed and second.completed
    assert first.output == second.output
    assert first.steps[0] is not second.steps[0]
    assert quickstart.assess_result(first).passed
    assert quickstart.main([]) == 0
    output = capsys.readouterr().out
    assert "Completed: True (done)" in output
    assert "Grade: passed" in output
    assert list(tmp_path.iterdir()) == []


def test_quickstart_failed_grade_explains_expected_and_observed(tmp_path, monkeypatch, capsys):
    from looplet import load_eval_run
    from looplet.__main__ import main
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    case = quickstart.owner_case()
    case.expected["owner"] = "security-team"
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps({"id": case.id, "task": case.task, "expected": case.expected}))
    trace = tmp_path / "failed capture"
    mismatch = "owner: expected 'security-team', observed 'fintech-platform'."
    assert quickstart.main(["--case", str(case_file), "--capture", str(trace)]) == 1
    assert mismatch in capsys.readouterr().err
    saved = load_eval_run(trace)
    assert saved.context.completed
    assert saved.results[0].explanation == mismatch
    assert "security-team" not in (trace / "manifest.jsonl").read_text()
    assert quickstart.main(["--rescore", str(trace)]) == 1
    assert mismatch in capsys.readouterr().err
    assert main(["inspect", str(trace)]) == 0
    assert mismatch in capsys.readouterr().out
    assert main(["eval", str(trace), "--evals", quickstart.__file__, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["cases"][0]["results"][0]["explanation"] == mismatch


def test_packaged_quickstart_capture_rescore_and_replay(tmp_path, monkeypatch):
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    scripted = Mock(wraps=quickstart.scripted_backend)
    monkeypatch.setattr(quickstart, "scripted_backend", scripted)
    trace = tmp_path / "saved-run"
    assert quickstart.main(["--capture", str(trace)]) == 0
    assert scripted.call_count == 1
    before = {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    }
    assert json.loads((trace / "expected.json").read_text()) == quickstart.EXPECTED_OWNER
    assert "expected" not in json.loads((trace / "trajectory.json").read_text())["task"]
    assert quickstart.main(["--rescore", str(trace)]) == 0
    assert quickstart.main(["--replay", str(trace)]) == 0
    assert scripted.call_count == 1
    assert {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    } == before


def test_packaged_quickstart_replay_saves_evidence_and_honors_limits(
    tmp_path, monkeypatch, capsys, recwarn
):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    responses = [
        '{"tool":"lookup_owner","args":{"service":"search"}}',
        '{"tool":"lookup_owner","args":{"service":"payments"}}',
        '{"tool":"done","args":{"service":"payments","owner":"fintech-platform"}}',
    ]
    monkeypatch.setattr(
        quickstart, "scripted_backend", lambda: MockLLMBackend(responses, cycle=False)
    )
    source = tmp_path / "original capture"
    assert quickstart.main(["--capture", str(source), "--max-steps", "3"]) == 0
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    replay = tmp_path / "replayed record"
    assert quickstart.main(["--replay", str(source), "--out", str(replay)]) == 0
    assert load_eval_run(replay).context.completed
    assert len((replay / "manifest.jsonl").read_text().split("\n")) == 4
    assert (
        json.loads((replay / "trajectory.json").read_text())["metadata"]["parent_run_id"]
        == json.loads((source / "trajectory.json").read_text())["run_id"]
    )
    limited = tmp_path / "limited replay"
    assert (
        quickstart.main(["--replay", str(source), "--out", str(limited), "--max-steps", "1"]) == 1
    )
    record = load_eval_run(limited)
    assert record.context.step_count == 1
    assert record.context.stop_reason == "budget_exhausted"
    assert not record.context.completed
    assert "Stop reason: budget_exhausted" in capsys.readouterr().err
    assert not any("differs from LoopConfig" in str(warning.message) for warning in recwarn)
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == original


def test_packaged_quickstart_experiments_are_repeat_safe(tmp_path, monkeypatch):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    assert quickstart.main(["--experiment"]) == 0
    experiments = tmp_path / ".looplet/experiments"
    original = {path: path.read_bytes() for path in experiments.rglob("*") if path.is_file()}
    assert quickstart.main(["--experiment"]) == 0
    assert {path: path.read_bytes() for path in original} == original
    directories = list(experiments.iterdir())
    assert len(directories) == 2
    for directory in directories:
        report = json.loads((directory / "experiment.json").read_text())
        assert report["passed"]
        assert report["stage"] == "complete"
        assert report["stages"] == dict.fromkeys(("capture", "rescore", "replay"), "passed")
        assert report["error"] is None
        assert (directory / ".gitignore").read_text() == "*\n"
        for key in ("capture_dir", "replay_dir"):
            assert not Path(report[key]).is_absolute()
            record = load_eval_run(directory / report[key])
            assert record.context.completed
            assert record.results and all(result.passed for result in record.results)


@pytest.mark.parametrize("failure", ["correctness", "budget"])
@pytest.mark.parametrize("report_file", [False, True])
def test_quickstart_failed_experiment_is_inspectable(
    tmp_path, monkeypatch, capsys, failure, report_file
):
    from looplet import load_eval_run
    from looplet.__main__ import main
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "failed experiment"
    arguments = ["--experiment", "--out", str(directory)]
    if failure == "budget":
        arguments.extend(["--max-steps", "1"])
    else:
        monkeypatch.setattr(
            quickstart, "EXPECTED_OWNER", {"service": "payments", "owner": "security-team"}
        )
    assert quickstart.main(arguments) == 1
    capsys.readouterr()
    report_path = directory / "experiment.json"
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert report["stage"] == "capture"
    assert report["stages"] == {"capture": "failed", "rescore": "not_run", "replay": "not_run"}
    assert report["capture_dir"] == "capture"
    assert report["replay_dir"] is None
    assert not (directory / "replay").exists()
    assert (
        "Stop reason: " + ("budget_exhausted" if failure == "budget" else "done") in report["error"]
    )
    assert (
        "security-team" in report["error"]
        if failure == "correctness"
        else "did not complete" in report["error"]
    )
    assert load_eval_run(directory / "capture").context.completed is (failure == "correctness")
    original = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    target = report_path if report_file else directory
    assert main(["inspect", str(target)]) == 0
    human = capsys.readouterr().out
    assert "capture: failed" in human
    assert "rescore: not_run" in human and "replay: not_run" in human
    assert "looplet inspect " + shlex.quote(str(directory / "capture")) in human
    assert report["error"] in human
    assert main(["inspect", str(target), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("stage", ["rescore", "replay"])
def test_quickstart_experiment_records_later_stage_failure(tmp_path, monkeypatch, stage):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    if stage == "rescore":

        def fail_rescore(*arguments, **options):
            raise AssertionError("controlled rescore failure")

        monkeypatch.setattr(quickstart, "eval_assess", fail_rescore)
    else:
        original = quickstart.replay_agent

        def limited_replay(*arguments, **options):
            return original(*arguments, max_steps=1, **options)

        monkeypatch.setattr(quickstart, "replay_agent", limited_replay)
    directory = tmp_path / "later failure"
    assert quickstart.main(["--experiment", "--out", str(directory)]) == 1
    report = json.loads((directory / "experiment.json").read_text())
    assert report["passed"] is False
    assert report["stage"] == stage and report["stages"][stage] == "failed"
    assert report["stages"]["capture"] == "passed"
    assert load_eval_run(directory / report["capture_dir"]).context.completed
    if stage == "rescore":
        assert report["stages"]["replay"] == "not_run"
        assert report["replay_dir"] is None
        assert "controlled rescore failure" in report["error"]
    else:
        assert report["stages"]["rescore"] == "passed"
        assert (
            load_eval_run(directory / report["replay_dir"]).context.stop_reason
            == "budget_exhausted"
        )
        assert "Stop reason: budget_exhausted" in report["error"]


def test_quickstart_customized_service_keeps_task_and_expectations_together(tmp_path, monkeypatch):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    expected = {"service": "search", "owner": "discovery"}
    monkeypatch.setattr(quickstart, "EXPECTED_OWNER", expected)
    monkeypatch.setattr(
        quickstart,
        "scripted_backend",
        lambda: MockLLMBackend(
            [
                '{"tool":"lookup_owner","args":{"service":"search"}}',
                '{"tool":"done","args":{"service":"search","owner":"discovery"}}',
            ],
            cycle=False,
        ),
    )
    trace = tmp_path / "search capture"
    assert quickstart.main(["--capture", str(trace)]) == 0
    saved = load_eval_run(trace)
    assert saved.case is not None
    assert saved.case.task == {"goal": "Find the search owner, then finish."}
    assert saved.case.expected == saved.context.expected == expected
    assert saved.context.final_output["service"] == "search"
    assert all(result.passed for result in saved.results)
    assert json.loads((trace / "trajectory.json").read_text())["task"] == saved.case.task


@pytest.mark.parametrize("action", ["--rescore", "--replay"])
@pytest.mark.parametrize("artifact", ["evals.json", "case.json", "trajectory.json"])
def test_exported_quickstart_malformed_evidence_is_actionable(tmp_path, action, artifact):
    from looplet.examples import quickstart

    script = tmp_path / "my_agent.py"
    script.write_bytes(Path(quickstart.__file__).read_bytes())
    original = tmp_path / "intact capture"
    quickstart.capture_agent(quickstart.scripted_backend(), original)
    damaged = tmp_path / "damaged capture"
    shutil.copytree(original, damaged)
    (damaged / artifact).write_text("{\n")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(
        [sys.executable, str(script), action, str(damaged)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "error: Could not read saved evidence" in result.stderr
    assert str(damaged) in result.stderr
    assert "Preserve these files" in result.stderr
    assert "--capture NEW_TRACE_DIR" in result.stderr
    assert "Traceback" not in result.stderr
    assert {path: path.read_bytes() for path in before} == before
    assert quickstart.main(["--rescore", str(original)]) == 0


@pytest.mark.parametrize("action", ["--rescore", "--replay"])
def test_quickstart_saved_input_handling_does_not_hide_runtime_errors(
    tmp_path, monkeypatch, action
):
    from looplet.examples import quickstart

    original = tmp_path / "capture"
    quickstart.capture_agent(quickstart.scripted_backend(), original)
    symbol = "eval_assess" if action == "--rescore" else "build"
    monkeypatch.setattr(quickstart, symbol, Mock(side_effect=ValueError("runtime defect")))
    with pytest.raises(ValueError, match="runtime defect"):
        quickstart.main([action, str(original)])


def test_quickstart_explicit_case_survives_python_and_cli_replay(tmp_path, monkeypatch):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    case = {
        "id": "search-case",
        "task": {"goal": "Identify who owns the search service."},
        "expected": {"service": "search", "owner": "discovery"},
    }
    case_path = tmp_path / "search case.json"
    case_path.write_text(json.dumps(case))
    monkeypatch.setattr(
        quickstart,
        "scripted_backend",
        lambda: MockLLMBackend(
            [
                '{"tool":"lookup_owner","args":{"service":"search"}}',
                '{"tool":"done","args":{"service":"search","owner":"discovery"}}',
            ],
            cycle=False,
        ),
    )
    experiment = tmp_path / "search experiment"
    assert (
        quickstart.main(["--case", str(case_path), "--experiment", "--out", str(experiment)]) == 0
    )
    for directory in (experiment / "capture", experiment / "replay"):
        saved = load_eval_run(directory)
        assert saved.case is not None
        assert saved.case.id == case["id"]
        assert saved.case.task == case["task"]
        assert saved.case.expected == saved.context.expected == case["expected"]
        assert saved.results and all(result.passed for result in saved.results)
        assert json.loads((directory / "trajectory.json").read_text())["task"] == case["task"]
    (tmp_path / "my_agent.py").write_bytes(Path(quickstart.__file__).read_bytes())
    replayed = subprocess.run(
        [
            sys.executable,
            "-m",
            "looplet",
            "replay",
            str(experiment / "capture"),
            "--factory",
            "my_agent:build",
            "--evals",
            "my_agent.py",
            "--workspace",
            str(tmp_path / "CLI replay"),
            "--json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    trace = json.loads(replayed.stdout)["trace_dir"]
    assert load_eval_run(trace).context.expected == case["expected"]
    assessed = subprocess.run(
        [sys.executable, "-m", "looplet", "eval", trace, "--evals", "my_agent.py", "--json"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert assessed.returncode == 0, assessed.stdout + assessed.stderr
    assert json.loads(assessed.stdout)["passed"]


@pytest.mark.parametrize("seeded", [False, True])
def test_quickstart_case_rejects_ambiguous_or_seeded_input(tmp_path, seeded):
    from looplet.examples import quickstart

    case = {"id": "owner", "task": {}, "expected": quickstart.EXPECTED_OWNER}
    if seeded:
        case["task"] = {"files": {"input.txt": "private seed"}}
        cases = case
    else:
        cases = [case, {**case, "id": "other"}]
    case_path = tmp_path / "cases.json"
    case_path.write_text(json.dumps(cases))
    with pytest.raises(SystemExit) as error:
        quickstart.main(["--case", str(case_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == [case_path]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("json_text", [False, True])
def test_packaged_quickstart_live_capture_uses_the_same_agent(
    tmp_path, monkeypatch, provider, json_text
):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    backend = quickstart.scripted_backend()
    factory = Mock(return_value=backend)
    adapter = "OpenAIBackend" if provider == "openai" else "AnthropicBackend"
    monkeypatch.setattr(f"looplet.{adapter}.from_env", factory)
    monkeypatch.setattr(
        quickstart, "scripted_backend", Mock(side_effect=AssertionError("scripted"))
    )
    arguments = ["--capture", "live", "--backend", provider]
    if json_text:
        arguments.append("--json-text")
    assert quickstart.main(arguments) == 0
    factory.assert_called_once_with()
    assert backend.calls == 2
    assert load_eval_run(tmp_path / "live").context.completed


@pytest.mark.parametrize(
    "arguments",
    [
        ["--replay", "missing", "--backend", "openai"],
        ["--rescore", "missing", "--max-steps", "1"],
        ["--out", "unused"],
        ["--replay", "missing", "--case", "unused"],
        ["--rescore", "missing", "--case", "unused"],
        ["--json"],
        ["--experiment", "--json"],
        ["--backend", "openai", "--json"],
    ],
)
def test_packaged_quickstart_rejects_ignored_options(arguments, monkeypatch):
    from looplet.examples import quickstart

    scripted = Mock(side_effect=AssertionError("unexpected scripted backend"))
    provider = Mock(side_effect=AssertionError("unexpected provider setup"))
    monkeypatch.setattr(quickstart, "scripted_backend", scripted)
    monkeypatch.setattr(quickstart.OpenAIBackend, "from_env", provider)
    with pytest.raises(SystemExit) as error:
        quickstart.main(arguments)
    assert error.value.code == 2
    scripted.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("missing provider settings"),
        ImportError("missing provider extra"),
        ValueError("invalid provider settings"),
    ],
)
def test_packaged_quickstart_provider_setup_errors_are_actionable(
    tmp_path, monkeypatch, capsys, provider, failure
):
    from looplet.examples import quickstart

    adapter = quickstart.OpenAIBackend if provider == "openai" else quickstart.AnthropicBackend
    setup = Mock(side_effect=failure)
    monkeypatch.setattr(adapter, "from_env", setup)
    output = tmp_path / "experiment"
    assert quickstart.main(["--backend", provider, "--experiment", "--out", str(output)]) == 1
    setup.assert_called_once_with()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"error: {provider} setup failed: {failure}" in captured.err
    assert f"https://hsaghir.com/looplet/install/#configure-{provider}" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_packaged_quickstart_stop_is_explicit(tmp_path, monkeypatch, capsys):
    from looplet.examples import quickstart

    monkeypatch.chdir(tmp_path)
    backend = quickstart.scripted_backend()
    stopped = quickstart.stop_after_lookup(backend)
    assert backend.calls == 1
    assert stopped.termination_reason == "stream_closed"
    assert not stopped.completed
    assert [step.tool_call.tool for step in stopped.steps] == ["lookup_owner"]
    assert quickstart.main(["--stop-after-lookup"]) == 0
    output = capsys.readouterr().out
    assert "Completed: False (stream_closed)" in output
    assert "Grade: passed" not in output


def test_packaged_quickstart_launcher_propagates_failure(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "looplet.examples.quickstart", "--max-steps", "1"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1, result.stderr
    assert "budget_exhausted" in result.stdout


@pytest.mark.parametrize("max_steps, stop_reason", [(1, "budget_exhausted"), (2, "done")])
def test_packaged_quickstart_failed_capture_reports_saved_evidence(
    tmp_path, monkeypatch, capsys, max_steps, stop_reason
):
    import shlex

    from looplet import load_eval_run
    from looplet.__main__ import main
    from looplet.examples import quickstart

    if max_steps == 2:
        monkeypatch.setattr(quickstart, "EXPECTED_OWNER", {"service": "payments", "owner": "wrong"})
    trace = tmp_path / "saved run"

    assert quickstart.main(["--capture", str(trace), "--max-steps", str(max_steps)]) == 1
    output = capsys.readouterr()
    assert "eval_owner:" in output.err
    explanation = (
        "Run did not complete; stop reason: budget_exhausted."
        if max_steps == 1
        else "owner: expected 'wrong', observed 'fintech-platform'."
    )
    assert explanation in output.err
    assert f"Stop reason: {stop_reason}" in output.err
    command = next(
        line.removeprefix("Inspect with: ")
        for line in output.err.splitlines()
        if line.startswith("Inspect with: ")
    )
    assert shlex.split(command) == ["looplet", "inspect", str(trace)]
    assert load_eval_run(trace).context.stop_reason == stop_reason
    before = {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    }
    assert main(["inspect", str(trace)]) == 0
    assert stop_reason in capsys.readouterr().out
    assert {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    } == before


def test_packaged_quickstart_capture_launcher_reports_recovery(tmp_path):
    trace = tmp_path / "saved run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "looplet.examples.quickstart",
            "--capture",
            str(trace),
            "--max-steps",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Stop reason: budget_exhausted" in result.stderr
    assert "Inspect with: looplet inspect" in result.stderr
    assert str(trace) in result.stderr
    assert (trace / "trajectory.json").is_file()


def test_packaged_quickstart_preserves_existing_capture(tmp_path):
    from looplet.examples import quickstart

    trace = tmp_path / "saved-run"
    assert quickstart.main(["--capture", str(trace)]) == 0
    before = {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    }
    assert quickstart.main(["--capture", str(trace)]) == 1
    assert {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    } == before


def test_packaged_quickstart_failed_rescore_returns_nonzero(tmp_path, monkeypatch):
    from looplet.examples import quickstart

    trace = tmp_path / "saved-run"
    assert quickstart.main(["--capture", str(trace)]) == 0
    monkeypatch.setattr(quickstart, "eval_owner", lambda context: False)
    assert quickstart.main(["--rescore", str(trace)]) == 1


@pytest.mark.parametrize("passed", [False, True])
def test_quickstart_rescore_saves_new_verdict_without_replacing_capture(
    tmp_path, monkeypatch, capsys, passed
):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    trace = tmp_path / "saved run"
    assert quickstart.main(["--capture", str(trace)]) == 0
    capsys.readouterr()
    before = {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()}
    monkeypatch.setattr(quickstart, "eval_owner", lambda context: passed)
    monkeypatch.setattr(
        quickstart, "scripted_backend", Mock(side_effect=AssertionError("no agent"))
    )
    target = tmp_path / "new assessment.json"
    assert quickstart.main(["--rescore", str(trace), "--out", str(target)]) == int(not passed)
    captured = capsys.readouterr()
    assert "New assessment using current graders" in captured.out
    assert str(target) in captured.out
    assessment = json.loads(target.read_text())
    assert assessment["assessment"]["passed"] is passed
    assert load_eval_run(trace).results[0].score == 1.0
    report_before = target.read_bytes()
    assert quickstart.main(["--rescore", str(trace), "--out", str(target)]) == 1
    assert "could not save assessment" in capsys.readouterr().err
    assert target.read_bytes() == report_before
    assert quickstart.main(["--rescore", str(trace), "--out", str(trace / "new.json")]) == 1
    assert "outside saved evidence" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()} == before


def test_quickstart_offers_companion_before_file_assembly():
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    text = guide.read_text()
    companion = text.index("python -m looplet.examples.quickstart")
    assert companion < text.index('title="first_loop.py"')
    for mode in ("--capture", "--rescore", "--replay", "--stop-after-lookup"):
        assert f"python -m looplet.examples.quickstart {mode}" in text
    assert "Keep running this agent with Python and testing it with pytest." in text
    assert "You do not need a skill bundle for that transition." in text
    cli = (guide.parent / "cli.md").read_text()
    primary = cli.split("## Choose an agent format\n", 1)[1].split(
        "## Diagnose and inspect runs\n", 1
    )[0]
    assert (
        "| Workflow | Python, including unreleased factory commands | Optional cartridge |"
        in primary
    )
    assert "replay --factory module:factory" in primary
    assert "| Workflow | Python agent | Cartridge | Skill bundle |" not in primary


def _interrupt_after_calls(backend, calls, async_driver, monkeypatch):
    generate = backend.generate
    if async_driver:

        async def interrupt(*arguments, **keywords):
            if backend.calls == calls:
                raise KeyboardInterrupt("Synthetic interruption")
            return await generate(*arguments, **keywords)

    else:

        def interrupt(*arguments, **keywords):
            if backend.calls == calls:
                raise KeyboardInterrupt("Synthetic interruption")
            return generate(*arguments, **keywords)

    monkeypatch.setattr(backend, "generate", interrupt)


def test_documented_migration_adapter_matches_loop_protocol():
    guide = Path(__file__).resolve().parents[1] / "docs" / "migrate.md"
    blocks = _python_blocks(guide)
    owner = {"service": "payments", "owner": "fintech-platform"}
    ownership_service = SimpleNamespace(lookup=Mock(return_value=owner))
    client = Mock()
    client.complete.side_effect = [
        '{"tool":"lookup_owner","args":{"service":"payments"}}',
        '{"tool":"done","args":{"summary":"The payments owner is fintech-platform."}}',
    ]
    logger = Mock()
    namespace = {
        "__name__": "documented_migration",
        "ownership_service": ownership_service,
        "existing_logger": logger,
        "user_request": "Find the payments owner, then finish.",
    }
    tool_block = next(block for block in blocks if "include_done=True" in block)
    adapter_block = next(block for block in blocks if "class ExistingBackend:" in block)
    loop_block = next(block for block in blocks if "config = LoopConfig(" in block)
    for block in (tool_block, adapter_block):
        exec(compile(block, str(guide), "exec"), namespace)
    namespace["current_backend"] = namespace["ExistingBackend"](client)
    exec(compile(loop_block, str(guide), "exec"), namespace)

    ownership_service.lookup.assert_called_once_with("payments")
    assert client.complete.call_count == 2
    assert logger.info.call_count == 2
    final_step = namespace["step"]
    assert final_step.tool_call.tool == "done"
    assert final_step.tool_result.error is None
    assert final_step.tool_result.data["summary"] == "The payments owner is fintech-platform."


@pytest.mark.parametrize("async_driver", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_documented_application_metadata_survives_resume(
    tmp_path, monkeypatch, async_driver, native
):
    from looplet import (
        AsyncMockLLMBackend,
        DefaultState,
        async_composable_loop,
        tools_from,
    )
    from looplet.checkpoint import FileCheckpointStore

    guide = Path(__file__).resolve().parents[1] / "docs" / "operations.md"
    block = next(
        block for block in _python_blocks(guide) if "def checkpoint_application_state(" in block
    )
    namespace = {"__name__": "documented_metadata_recovery"}
    exec(compile(block, str(guide), "exec"), namespace)
    config = namespace["config"]
    config.checkpoint_dir = str(tmp_path)
    config.max_steps = 2
    config.use_native_tools = native
    application = {"entities": ["customer-42"], "limits": {"allowed": [1, 2]}}
    original = DefaultState(max_steps=2, metadata={"app": application})

    def lookup() -> dict:
        """Read one synthetic record."""
        return {"observed": True}

    tools = tools_from([lookup], include_done=True, validate=True)
    backend_class = AsyncMockLLMBackend if async_driver else MockLLMBackend
    backend = backend_class(['{"tool":"lookup","args":{}}'], cycle=False)
    _interrupt_after_calls(backend, 1, async_driver, monkeypatch)

    async def run(llm, state, hooks=()):
        arguments = {"llm": llm, "tools": tools, "state": state, "config": config, "hooks": hooks}
        if async_driver:
            return [step async for step in async_composable_loop(**arguments)]
        return list(composable_loop(**arguments))

    with pytest.raises(KeyboardInterrupt, match="Synthetic interruption"):
        await run(backend, original)
    checkpoint = FileCheckpointStore(tmp_path).load_latest_for_resume()
    assert checkpoint is not None and checkpoint.step_number == 1
    assert checkpoint.domain_state == {"app": application}
    recorded = json.loads(json.dumps(checkpoint.to_dict()))
    observations = []

    class ObserveRestore:
        def pre_prompt(self, state, session_log, context, step_num):
            observations.append(
                (step_num, len(state.steps), state.budget_remaining, state.metadata["app"])
            )

    restored = DefaultState(max_steps=2, metadata={"host_marker": "retained"})
    resumed_backend = backend_class(['{"tool":"done","args":{"summary":"Finished."}}'], cycle=False)
    steps = await run(resumed_backend, restored, [ObserveRestore()])
    assert RunResult.from_state(restored).completed
    assert observations == [(2, 0, 1, application)]
    assert [step.tool_call.tool for step in steps] == ["done"]
    assert restored.metadata["host_marker"] == "retained"
    assert restored.metadata["app"] == application
    assert restored.budget_remaining == 0
    assert restored.queries_used == 1
    assert backend.calls == resumed_backend.calls == 1
    assert checkpoint.to_dict() == recorded


def test_documented_metadata_callbacks_copy_values_and_accept_legacy_snapshots():
    from looplet import DefaultState

    guide = Path(__file__).resolve().parents[1] / "docs" / "operations.md"
    block = next(
        block for block in _python_blocks(guide) if "def checkpoint_application_state(" in block
    )
    namespace = {"__name__": "documented_metadata_callbacks"}
    exec(compile(block, str(guide), "exec"), namespace)
    state = DefaultState(metadata={"app": {"entities": ["customer-42"]}, "run_status": "running"})
    snapshot = namespace["checkpoint_application_state"](SimpleNamespace(state=state))
    state.metadata["app"]["entities"].append("later")
    assert snapshot == {"app": {"entities": ["customer-42"]}}
    restored = DefaultState(metadata={"app": {"existing": True}, "host_marker": "retained"})
    namespace["restore_application_state"](SimpleNamespace(state=restored), {})
    assert restored.metadata["app"] == {"existing": True}
    namespace["restore_application_state"](SimpleNamespace(state=restored), snapshot)
    restored.metadata["app"]["entities"].append("new")
    assert snapshot == {"app": {"entities": ["customer-42"]}}
    assert restored.metadata["host_marker"] == "retained"


@pytest.mark.parametrize("async_driver", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_documented_hooks_share_entities_across_turns(async_driver, native, monkeypatch):
    from looplet import (
        AsyncMockLLMBackend,
        DefaultState,
        LoopConfig,
        async_composable_loop,
        tools_from,
    )

    guide = Path(__file__).resolve().parents[1] / "docs" / "hooks.md"
    block = next(block for block in _python_blocks(guide) if "class EntityExtractor:" in block)
    extractions = []
    briefings = []

    def extract_from(data):
        entities = data.get("entities", [])
        extractions.append(entities)
        return entities

    namespace = {"__name__": "documented_hook_sharing", "extract_from": extract_from}
    exec(compile(block, str(guide), "exec"), namespace)
    hooks = namespace["hooks"]
    pre_prompt = hooks[1].pre_prompt

    def observe_briefing(state, session_log, context, step_num):
        value = pre_prompt(state, session_log, context, step_num)
        if value:
            briefings.append(value)
        return value

    monkeypatch.setattr(hooks[1], "pre_prompt", observe_briefing)

    def lookup() -> dict:
        """Find one synthetic entity."""
        return {"entities": ["customer-42"]}

    responses = [
        '{"tool":"lookup","args":{}}',
        '{"tool":"done","args":{"summary":"Finished."}}',
    ]
    state = DefaultState(max_steps=2)
    arguments = {
        "llm": (AsyncMockLLMBackend if async_driver else MockLLMBackend)(responses),
        "tools": tools_from([lookup], include_done=True, validate=True),
        "state": state,
        "config": LoopConfig(max_steps=2, use_native_tools=native),
        "hooks": hooks,
    }
    if async_driver:
        steps = [step async for step in async_composable_loop(**arguments)]
    else:
        steps = list(composable_loop(**arguments))
    assert [step.tool_call.tool for step in steps] == ["lookup", "done"]
    assert ["customer-42"] in extractions
    assert briefings == ["Entities found so far: customer-42"]
    assert RunResult.from_state(state).completed


@pytest.mark.parametrize("async_driver", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "scenario",
    ["successful-work", "failed-work", "rejected-done", "custom-completion", "rejected-then-work"],
)
async def test_documented_quality_gate_requires_successful_work(async_driver, native, scenario):
    from looplet import (
        AsyncMockLLMBackend,
        DefaultState,
        LoopConfig,
        async_composable_loop,
        tools_from,
    )

    guide = Path(__file__).resolve().parents[1] / "docs" / "hooks.md"
    block = next(block for block in _python_blocks(guide) if "class MinimumEvidenceGate:" in block)
    namespace = {"__name__": "documented_quality_gate"}
    exec(compile(block, str(guide), "exec"), namespace)
    executed = []

    def lookup_owner(service: str) -> dict:
        """Find the team that owns a service."""
        if scenario == "failed-work":
            raise ValueError("Synthetic lookup failure")
        executed.append(service)
        return {"service": service, "owner": "fintech-platform"}

    def complete(summary: str) -> dict:
        """Finish the task."""
        return {"summary": summary}

    terminal = "finish" if scenario == "custom-completion" else "done"
    complete.__name__ = terminal
    lookup = json.dumps({"tool": "lookup_owner", "args": {"service": "payments"}})
    done = json.dumps({"tool": terminal, "args": {"summary": "Finished."}})
    responses = [lookup] * 3 + [done]
    if scenario in ("rejected-done", "custom-completion"):
        responses = [done] * 4
    elif scenario == "rejected-then-work":
        responses = [done] + responses
    gate_class = namespace["MinimumEvidenceGate"]
    gate = gate_class(done_tools=(terminal,)) if terminal != "done" else gate_class()
    state = DefaultState(max_steps=len(responses))
    arguments = {
        "llm": (AsyncMockLLMBackend if async_driver else MockLLMBackend)(responses),
        "tools": tools_from([lookup_owner, complete], validate=True),
        "state": state,
        "config": LoopConfig(max_steps=len(responses), done_tool=terminal, use_native_tools=native),
        "hooks": [gate],
    }
    if async_driver:
        steps = [step async for step in async_composable_loop(**arguments)]
    else:
        steps = list(composable_loop(**arguments))

    expected_completion = scenario in ("successful-work", "rejected-then-work")
    assert RunResult.from_state(state).completed is expected_completion
    assert len(executed) == (3 if expected_completion else 0)
    assert len(steps) == len(responses)
    completion_steps = [step for step in steps if step.tool_call.tool == terminal]
    if expected_completion:
        assert completion_steps[-1].tool_result.error is None
    else:
        assert all(
            "successful non-completion" in step.tool_result.error for step in completion_steps
        )


@pytest.mark.parametrize(
    "test_name", ["test_quality_gate_rejects_early", "test_quality_gate_allows_enough"]
)
def test_documented_quality_gate_unit_examples(test_name):
    guide = Path(__file__).resolve().parents[1] / "docs" / "hooks.md"
    blocks = _python_blocks(guide)
    namespace = {"__name__": "documented_gate_tests"}
    for marker in ("class MinimumEvidenceGate:", "def test_quality_gate_rejects_early("):
        block = next(block for block in blocks if marker in block)
        exec(compile(block, str(guide), "exec"), namespace)
    namespace[test_name]()


@pytest.mark.parametrize("async_driver", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("terminal", ["done", "finish"])
@pytest.mark.parametrize("persist_metadata", [False, True])
async def test_documented_quality_gate_restores_checkpointed_progress(
    tmp_path, monkeypatch, async_driver, native, terminal, persist_metadata
):
    from copy import deepcopy

    from looplet import (
        AsyncMockLLMBackend,
        DefaultState,
        async_composable_loop,
        tools_from,
    )
    from looplet.checkpoint import FileCheckpointStore

    root = Path(__file__).resolve().parents[1]
    namespace = {"__name__": "documented_gate_recovery"}
    for guide_name, marker in (
        ("operations.md", "def checkpoint_application_state("),
        ("hooks.md", "class MinimumEvidenceGate:"),
    ):
        guide = root / "docs" / guide_name
        block = next(block for block in _python_blocks(guide) if marker in block)
        exec(compile(block, str(guide), "exec"), namespace)
    config = namespace["config"]
    config.checkpoint_dir = str(tmp_path)
    config.use_native_tools = native
    config.done_tool = terminal
    if not persist_metadata:
        config.checkpoint_state = None
        config.restore_checkpoint_state = None
    executed = []

    def lookup() -> dict:
        """Read one synthetic record."""
        executed.append("lookup")
        return {"observed": True}

    tools = tools_from([lookup], include_done=True, done_name=terminal, validate=True)
    backend_class = AsyncMockLLMBackend if async_driver else MockLLMBackend
    backend = backend_class(['{"tool":"lookup","args":{}}'] * 3, cycle=False)
    _interrupt_after_calls(backend, 3, async_driver, monkeypatch)
    gate_class = namespace["MinimumEvidenceGate"]

    async def run(llm, state, observers=()):
        arguments = {
            "llm": llm,
            "tools": tools,
            "state": state,
            "config": config,
            "hooks": [gate_class(done_tools=(terminal,)), *observers],
        }
        if async_driver:
            return [step async for step in async_composable_loop(**arguments)]
        return list(composable_loop(**arguments))

    original = DefaultState(max_steps=4, metadata={"app": {"entities": ["customer-42"]}})
    with pytest.raises(KeyboardInterrupt, match="Synthetic interruption"):
        await run(backend, original)
    checkpoint = FileCheckpointStore(tmp_path).load_latest_for_resume()
    assert checkpoint is not None and checkpoint.step_number == 3
    recorded = json.loads(json.dumps(checkpoint.to_dict()))
    observations = []

    class ObserveRestore:
        def pre_prompt(self, state, session_log, context, step_num):
            observations.append(
                {
                    "step_num": step_num,
                    "steps": len(state.steps),
                    "budget": state.budget_remaining,
                    "session_entries": len(session_log.entries),
                    "app": deepcopy(state.metadata.get("app", {})),
                }
            )

    restored = DefaultState(max_steps=4)
    resumed_backend = backend_class(
        [json.dumps({"tool": terminal, "args": {"summary": "Finished."}})], cycle=False
    )
    steps = await run(resumed_backend, restored, [ObserveRestore()])
    assert RunResult.from_state(restored).completed is persist_metadata
    assert observations[0] == {
        "step_num": 4,
        "steps": 0,
        "budget": 1,
        "session_entries": 3,
        "app": ({"entities": ["customer-42"], "successful_calls": 3} if persist_metadata else {}),
    }
    assert [step.number for step in steps] == [4]
    assert [step.tool_call.tool for step in steps] == [terminal]
    assert executed == ["lookup"] * 3
    assert backend.calls == 3 and resumed_backend.calls == 1
    assert restored.budget_remaining == 0 and restored.queries_used == 3
    assert checkpoint.to_dict() == recorded
    if persist_metadata:
        assert steps[0].tool_result.error is None
        assert restored.metadata["app"]["successful_calls"] == 3
    else:
        assert "successful non-completion" in steps[0].tool_result.error


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


def test_documented_bundle_replay_works_without_provider(tmp_path, monkeypatch, capsys):
    from looplet.__main__ import main

    root = Path(__file__).resolve().parents[1]
    guide = root / "docs" / "cli.md"
    block = next(block for block in _python_blocks(guide) if "load_skill_bundle" in block)
    shutil.copytree(
        root / "tests" / "fixtures" / "coder_skill_bundle", tmp_path / "skills" / "coder"
    )
    original_workspace = tmp_path / "original-workspace"
    original_workspace.mkdir()
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: pytest.fail("no provider")
    )
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "run",
                "skills/coder",
                "Review this repository",
                "--project-root",
                str(original_workspace),
                "--trace-dir",
                "bundle-trace",
                "--scripted-response",
                '{"tool":"done","args":{"summary":"Reviewed"}}',
                "--no-tests",
                "--require-completed",
                "--json",
            ]
        )
        == 0
    )
    original = {
        path.relative_to(tmp_path / "bundle-trace"): path.read_bytes()
        for path in (tmp_path / "bundle-trace").rglob("*")
        if path.is_file()
    }
    capsys.readouterr()
    namespace = {"__name__": "__main__"}
    exec(compile(block, str(guide), "exec"), namespace)
    assert namespace["result"].completed
    assert namespace["result"].output["summary"] == "Reviewed"
    assert namespace["runtime"].workspace == Path("replay-workspaces/bundle-001")
    assert {
        path.relative_to(tmp_path / "bundle-trace"): path.read_bytes()
        for path in (tmp_path / "bundle-trace").rglob("*")
        if path.is_file()
    } == original
    with pytest.raises(FileExistsError):
        exec(compile(block, str(guide), "exec"), {"__name__": "__main__"})


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
def test_documented_first_loop(document: str, tmp_path, monkeypatch) -> None:
    from looplet import OpenAIBackend
    from looplet.testing import LLMResponsesExhausted

    root = Path(__file__).resolve().parents[1]
    path = root / document
    blocks = _python_blocks(path)
    assert blocks, f"No Python example in {document}"
    observations = []
    backend = MockLLMBackend(
        [
            '{"tool":"lookup_owner","args":{"service":"payments"}}',
            '{"tool":"done","args":{"summary":"fintech-platform"}}',
        ],
        cycle=False,
    )
    generate = Mock(wraps=backend.generate)
    monkeypatch.setattr(backend, "generate", generate)
    monkeypatch.setattr(OpenAIBackend, "from_env", staticmethod(lambda: backend))

    class ObserveRun:
        def bind(self, context):
            registry = context.tools
            dispatch = Mock(wraps=registry.dispatch)
            monkeypatch.setattr(registry, "dispatch", dispatch)
            observations.append((context.state, dispatch))

    def observed_loop(*args, **kwargs):
        return composable_loop(*args, hooks=[ObserveRun()], **kwargs)

    monkeypatch.setattr("looplet.composable_loop", observed_loop)
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    exec(compile(blocks[0], str(path), "exec"), namespace)
    state, dispatch = observations[0]
    assert [step.tool_call.tool for step in state.steps] == ["lookup_owner", "done"]
    assert state.steps[0].tool_result.data == "fintech-platform"
    assert state.termination_reason == "done"
    result = RunResult.from_state(state)
    assert result.completed
    assert result.output == {"status": "completed", "summary": "fintech-platform"}
    assert generate.call_count == 2
    assert dispatch.call_count == 2
    with pytest.raises(LLMResponsesExhausted):
        namespace["llm"].generate("Unexpected extra model call")

    first_loop = _python_blocks(root / "README.md")[0]
    assert ast.dump(ast.parse(blocks[0])) == ast.dump(ast.parse(first_loop))
    script = tmp_path / "first_loop.py"
    script.write_text(blocks[0])
    typed = subprocess.run(
        [sys.executable, "-m", "pyright", "--pythonpath", sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert typed.returncode == 0, typed.stdout + typed.stderr


async def test_provenance_async_replay_documented_example(tmp_path):
    from looplet import DefaultState, ProvenanceSink, tools_from

    guide = Path(__file__).resolve().parents[1] / "docs" / "provenance.md"
    block = _named_python_blocks(guide)["replay_async.py"]
    namespace = {"__name__": "replay_async"}
    exec(compile(block, str(guide), "exec"), namespace)
    effects = []
    checked = []

    def record_probe(value: str) -> dict:
        effects.append(value)
        return {"value": value}

    class Gate:
        async def check_permission(self, tool_call, state):
            checked.append(tool_call.tool)
            return tool_call.tool != "record_probe"

    tools = tools_from([record_probe], include_done=True, validate=True)
    sink = ProvenanceSink(dir=tmp_path / "capture")
    state = DefaultState(max_steps=2)
    captured = [
        step
        async for step in async_composable_loop(
            llm=sink.wrap_llm(
                MockLLMBackend(
                    [
                        '{"tool":"record_probe","args":{"value":"synthetic"}}',
                        '{"tool":"done","args":{"summary":"Finished."}}',
                    ],
                    cycle=False,
                )
            ),
            tools=tools,
            state=state,
            hooks=[Gate(), sink.trajectory_hook()],
            max_steps=2,
        )
    ]
    trace = sink.flush()
    assert "Permission denied" in captured[0].tool_result.error
    checked.clear()

    replayed = await namespace["replay_agent"](trace, tools, hooks=[Gate()])

    assert effects == [] and "record_probe" in checked
    assert "Permission denied" in replayed[0].tool_result.error
    assert replayed[-1].tool_call.tool == "done"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("record_only", [False, True])
async def test_provenance_redaction_modes_are_explicit_upfront(
    tmp_path, monkeypatch, asynchronous, record_only
):
    guide = Path(__file__).resolve().parents[1] / "docs" / "provenance.md"
    text = guide.read_text()
    assert text.index("redact_upstream=False") < text.index("## Quick start:")
    block = _named_python_blocks(guide)["capture_redaction.py"]
    namespace = {"__name__": "capture_redaction"}
    monkeypatch.chdir(tmp_path)
    exec(compile(block, str(guide), "exec"), namespace)
    seen = []

    class Backend:
        def generate(self, prompt, **kwargs):
            seen.append(prompt)
            return "Observed acct-example-42"

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    sink = namespace["record_only" if record_only else "provider_and_record"]
    recorder = sink.wrap_llm(AsyncBackend() if asynchronous else Backend())
    prompt = "Look up acct-example-42"
    result = await recorder.generate(prompt) if asynchronous else recorder.generate(prompt)
    trace = sink.flush()

    assert seen == [prompt if record_only else "Look up [REDACTED]"]
    assert result == "Observed acct-example-42"
    assert recorder.calls[0].prompt == "Look up [REDACTED]"
    assert all("acct-example-42" not in path.read_text() for path in trace.glob("*.txt"))


def test_provenance_golden_fixture_is_reviewable_and_git_addable(tmp_path, monkeypatch):
    from looplet import DefaultState, ProvenanceSink, replay_loop, tools_from

    repository = Path(__file__).resolve().parents[1]
    guide = repository / "docs" / "provenance.md"
    block = _named_python_blocks(guide)["reviewed_replay_fixture.py"]
    environment = os.environ.copy()
    names = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "--local-env-vars"], text=True
    ).splitlines()
    for name in names:
        environment.pop(name, None)
    subprocess.run(
        ["git", "-c", "init.defaultBranch=master", "init", "--quiet", str(tmp_path)],
        env=environment,
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    raw = ProvenanceSink(dir=tmp_path / "raw-capture")
    raw.wrap_llm(MockLLMBackend(["private-marker"], cycle=False)).generate("private-marker")
    raw_path = raw.flush()
    before = {path: path.read_bytes() for path in raw_path.rglob("*") if path.is_file()}
    namespace = {"__name__": "reviewed_replay_fixture"}

    exec(compile(block, str(guide), "exec"), namespace)

    fixture = namespace["fixture"]
    assert {path: path.read_bytes() for path in raw_path.rglob("*") if path.is_file()} == before
    assert all(
        "private-marker" not in path.read_text() for path in fixture.iterdir() if path.is_file()
    )
    (fixture / "not-reviewed.txt").write_text("private-marker")
    subprocess.run(["git", "add", "--", str(fixture)], env=environment, check=True)
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only"], env=environment, text=True
    ).splitlines()
    expected = {
        ".gitignore",
        "artifact.json",
        "manifest.jsonl",
        "call_00_prompt.txt",
        "call_00_response.txt",
    }
    assert set(staged) == {str(fixture / name) for name in expected}
    state = DefaultState(max_steps=1)
    steps = list(replay_loop(fixture, tools=tools_from([], include_done=True), state=state))
    assert len(steps) == 1 and steps[0].tool_result.error is None
    assert RunResult.from_state(state).completed


@pytest.mark.parametrize(("observed", "passed"), [("hello\n", True), ("wrong\n", False)])
async def test_documented_async_host_collection_grades_observed_artifacts(
    tmp_path, observed, passed
):
    guide = Path(__file__).resolve().parents[1] / "docs" / "evals.md"
    block = next(block for block in _python_blocks(guide) if "async def assess_report" in block)
    namespace = {"__name__": "async_host_example"}
    exec(compile(block, str(guide), "exec"), namespace)
    report_path = tmp_path / "result.txt"
    report_path.write_text(observed, encoding="utf-8")

    report = await namespace["assess_report"](report_path, "hello\n")

    assert report.valid
    assert report.passed is passed


def test_quickstart_real_model_handoff_precedes_advanced_workflows(tmp_path, monkeypatch):
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    text = guide.read_text()
    first_run = text.index("## 2. Run the agent and get its answer")
    live = text.index("## Use the same agent with a real model")
    grading = text.index("## 3. Check the answer")
    capture = text.index("## 4. Capture the same agent")
    assert first_run < live < grading < capture
    assert "This step is optional." in text[live:grading]
    examples = _named_python_blocks(guide)
    module = ModuleType("first_loop")
    monkeypatch.setitem(sys.modules, "first_loop", module)
    monkeypatch.chdir(tmp_path)
    exec(compile(examples["first_loop.py"], str(guide), "exec"), module.__dict__)
    assert not hasattr(module, "assess_result")
    backend = module.scripted_backend()
    from_env = Mock(return_value=backend)
    monkeypatch.setattr("looplet.OpenAIBackend.from_env", from_env)
    namespace = {"__name__": "__main__"}

    exec(compile(examples["first_loop_live.py"], str(guide), "exec"), namespace)

    from_env.assert_called_once_with()
    assert namespace["result"].completed
    assert backend.calls == 2


def test_copied_quickstart_factory_cli_workflow(tmp_path):
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    source = guide.read_text()
    project = tmp_path / "factory project with spaces"
    project.mkdir()
    (project / "first_loop.py").write_text(_quickstart_agent_source(guide))
    case = re.search(r'^```json title="cases/owner\.json"\n(.*?)^```[ \t]*$', source, re.M | re.S)
    assert case is not None
    (project / "cases").mkdir()
    (project / "cases" / "owner.json").write_text(case.group(1))
    first = _run_documented_shell(guide, "python_factory_cli.sh", project)
    assert first.returncode == 0, first.stdout + first.stderr
    experiments = project / ".looplet" / "experiments"
    before = {path: path.read_bytes() for path in experiments.rglob("*") if path.is_file()}
    assert before

    second = _run_documented_shell(guide, "python_factory_cli.sh", project)
    assert second.returncode == 0, second.stdout + second.stderr
    assert all(path.read_bytes() == content for path, content in before.items())
    directories = list(experiments.iterdir())
    assert len(directories) == 2
    for directory in directories:
        run = json.loads((directory / "run.json").read_text())
        assessment = json.loads((directory / "eval.json").read_text())
        replay = json.loads((directory / "replay.json").read_text())
        grading = json.loads((directory / "grade.json").read_text())
        assert run["completed"] and run["grading_status"] == "not_graded"
        assert assessment["passed"] and assessment["valid"]
        assert replay["completed"] and replay["grading_status"] == "not_graded"
        assert grading["passed"] and grading["valid"]
        saved_case = project / assessment["output_dir"] / assessment["cases"][0]["id"]
        assert (saved_case / "case.json").is_file()
        assert (directory / "replay").is_dir()
        assert Path(replay["trace_dir"]).is_dir()
    assert not list(project.rglob("cartridge.json"))
    assert not list(project.rglob("SKILL.md"))


def test_quickstart_scripted_backend_rejects_extra_model_calls() -> None:
    from looplet.testing import LLMResponsesExhausted

    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    namespace = {"__name__": "first_loop"}
    exec(compile(_quickstart_agent_source(guide), str(guide), "exec"), namespace)
    backend = namespace["scripted_backend"]()
    backend.generate("First scripted call")
    backend.generate("Second scripted call")
    with pytest.raises(LLMResponsesExhausted):
        backend.generate("Unexpected extra model call")


@pytest.mark.parametrize("max_steps, stop_reason", [(1, "budget_exhausted"), (2, "done")])
def test_copied_quickstart_failed_capture_reports_saved_evidence(
    tmp_path, monkeypatch, max_steps, stop_reason
):
    from looplet import ProvenanceSink, load_eval_run

    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    module = ModuleType("first_loop")
    monkeypatch.setitem(sys.modules, "first_loop", module)
    exec(compile(_quickstart_agent_source(guide), str(guide), "exec"), module.__dict__)
    original = module.run_agent
    monkeypatch.setattr(
        module, "run_agent", lambda llm, **kwargs: original(llm, max_steps=max_steps, **kwargs)
    )
    if max_steps == 2:
        monkeypatch.setattr(module, "EXPECTED_OWNER", {"service": "payments", "owner": "wrong"})
    trace = tmp_path / "saved run"
    monkeypatch.setattr("looplet.ProvenanceSink", lambda: ProvenanceSink(dir=trace))
    namespace = {"__name__": "first_loop_capture"}
    exec(
        compile(_named_python_blocks(guide)["first_loop_capture.py"], str(guide), "exec"),
        namespace,
    )
    assert not trace.exists()

    with pytest.raises(AssertionError, match="eval_owner: fail") as failure:
        namespace["capture_agent"](module.scripted_backend())

    message = str(failure.value)
    assert f"Stop reason: {stop_reason}" in message
    command = next(
        line.removeprefix("Inspect with: ")
        for line in message.splitlines()
        if line.startswith("Inspect with: ")
    )
    assert shlex.split(command) == ["looplet", "inspect", str(trace)]
    assert load_eval_run(trace).context.stop_reason == stop_reason
    from looplet.__main__ import main

    original = {
        path.relative_to(trace): path.read_bytes() for path in trace.rglob("*") if path.is_file()
    }
    assert main(shlex.split(command)[1:]) == 0
    assert {
        path.relative_to(trace): path.read_bytes() for path in trace.rglob("*") if path.is_file()
    } == original


def test_quickstart_runs_grades_captures_and_replays(tmp_path: Path, monkeypatch) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    examples = _named_python_blocks(guide)
    assert examples
    from_env = Mock()
    monkeypatch.setattr("looplet.OpenAIBackend.from_env", from_env)
    module = ModuleType("first_loop")
    monkeypatch.setitem(sys.modules, "first_loop", module)
    monkeypatch.chdir(tmp_path)
    exec(compile(_quickstart_agent_source(guide), str(guide), "exec"), module.__dict__)
    tests = {}
    exec(compile(examples["test_first_loop.py"], str(guide), "exec"), tests)

    assert not hasattr(module, "result")
    assert not (tmp_path / "traces").exists()
    from_env.assert_not_called()
    tests["test_tools_are_regular_functions"]()
    tests["test_owner"]()
    tests["test_repeated_runs_are_independent"]()
    native_responses = module.scripted_backend()
    native_backend = SimpleNamespace(
        generate=Mock(side_effect=AssertionError("live example must use native tools")),
        generate_with_tools=Mock(wraps=native_responses.generate_with_tools),
    )
    from_env.return_value = native_backend
    observed_agent = Mock(wraps=module.run_agent)
    monkeypatch.setattr(module, "run_agent", observed_agent)
    capture_block = examples["first_loop_capture.py"]
    imported_capture = {"__name__": "first_loop_capture"}
    exec(compile(capture_block, str(guide), "exec"), imported_capture)
    observed_agent.assert_not_called()
    assert not (tmp_path / ".looplet").exists()

    captured = {"__name__": "__main__"}
    exec(compile(capture_block, str(guide), "exec"), captured)
    assert observed_agent.call_count == 1
    assert captured["captured_backend"].calls == 2
    trace_dir = captured["trace_dir"]
    assert len(list(tmp_path.rglob("trajectory.json"))) == 1
    monkeypatch.setattr(sys, "argv", ["first_loop_rescore.py", str(trace_dir)])
    rescored = {"__name__": "__main__"}
    with monkeypatch.context() as released_api:
        released_api.delattr("looplet.EvalReport.save")
        exec(compile(examples["first_loop_rescore.py"], str(guide), "exec"), rescored)
    assert rescored["saved_report"].passed
    replayed = {"__name__": "__main__"}
    monkeypatch.setattr(sys, "argv", ["first_loop_replay.py", str(trace_dir)])
    exec(compile(examples["first_loop_replay.py"], str(guide), "exec"), replayed)
    assert captured["captured_result"].output == replayed["replayed_result"].output
    assert observed_agent.call_count == 1

    stopped = {"__name__": "__main__"}
    exec(compile(examples["first_loop_stop.py"], str(guide), "exec"), stopped)
    assert stopped["state"].termination_reason == "stream_closed"
    assert stopped["backend"].calls == 1
    assert [step.tool_call.tool for step in stopped["state"].steps] == ["lookup_owner"]
    assert not RunResult.from_state(stopped["state"]).completed
    live = {"__name__": "__main__"}
    exec(compile(examples["first_loop_live.py"], str(guide), "exec"), live)
    assert observed_agent.call_count == 2
    from_env.assert_called_once_with()
    native_backend.generate.assert_not_called()
    assert native_backend.generate_with_tools.call_count == 2
    assert [step.tool_call.tool for step in live["result"].steps] == ["lookup_owner", "done"]
    assert (trace_dir / "trajectory.json").is_file()
    assert (trace_dir / "manifest.jsonl").is_file()
    assert rescored["saved_run"].context.expected == module.EXPECTED_OWNER
    trajectory = json.loads((trace_dir / "trajectory.json").read_text())
    assert "expected" not in trajectory["task"]
    assert json.loads((trace_dir / "expected.json").read_text()) == module.EXPECTED_OWNER

    from looplet.evals import eval_cli

    (tmp_path / "first_loop.py").write_text(_quickstart_agent_source(guide))
    assert eval_cli([str(trace_dir), "--evals", str(tmp_path / "first_loop.py"), "--json"]) == 0
    original = (trace_dir / "trajectory.json").read_bytes()
    exec(compile(capture_block, str(guide), "exec"), captured)
    assert observed_agent.call_count == 3
    assert captured["trace_dir"] != trace_dir
    assert (trace_dir / "trajectory.json").read_bytes() == original
    captured["captured_result"].output["owner"] = "wrong-owner"
    assert not module.assess_result(captured["captured_result"]).passed


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
    exec(compile(_quickstart_agent_source(guide), str(guide), "exec"), namespace)
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
    exec(compile(_quickstart_agent_source(guide), str(guide), "exec"), namespace)
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
    examples = _named_python_blocks(guide)
    (tmp_path / "first_loop.py").write_text(_quickstart_agent_source(guide))
    (tmp_path / "test_first_loop.py").write_text(examples["test_first_loop.py"])
    tested = subprocess.run(
        [sys.executable, "-m", "pytest", "test_first_loop.py", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr
    assert "3 passed" in tested.stdout
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

    examples["first_loop.py"] = _quickstart_agent_source(guide)
    examples["first_loop.py"] += (
        "\nreused_tools = [lookup_owner, done]\n"
        "with minimal_preset(tools=reused_tools) as typed_run:\n"
        "    pass\n"
    )
    for name, code in examples.items():
        (tmp_path / name).write_text(code)
    typed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--pythonpath",
            sys.executable,
            "--outputjson",
            *examples,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert typed.returncode == 0, typed.stdout + typed.stderr


@pytest.mark.parametrize(
    ("remaining", "expected"), [(2, "Only 2 steps"), (4, "4 steps remaining"), (8, None)]
)
def test_documented_budget_warning_uses_remaining_count(remaining, expected):
    from looplet import DefaultState

    guide = Path(__file__).resolve().parents[1] / "docs" / "hooks.md"
    block = next(block for block in _python_blocks(guide) if "class BudgetWarningHook:" in block)
    namespace = {}
    exec(compile(block, str(guide), "exec"), namespace)

    message = namespace["BudgetWarningHook"]().pre_prompt(
        DefaultState(max_steps=remaining), None, None, 1
    )

    if expected is None:
        assert message is None
    else:
        assert expected in message
        assert "{remaining}" not in message


def test_eval_introduction_is_self_contained() -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "evals.md"
    namespace = {}
    exec(compile(_python_blocks(guide)[0], str(guide), "exec"), namespace)
    assert namespace["report"].passed


@pytest.mark.parametrize("record_state", ["absent", "committed", "conflicting", "missing"])
def test_documented_checkpoint_reconciliation(tmp_path: Path, record_state: str) -> None:
    from contextlib import closing

    from looplet import DefaultState, LoopConfig, RunEnvelope, tools_from
    from looplet.checkpoint import FileCheckpointStore, UncertainToolEffect

    guide = Path(__file__).resolve().parents[1] / "docs" / "host-integration.md"
    block = next(block for block in _python_blocks(guide) if "def reconcile_create_record" in block)
    namespace = {}
    exec(compile(block, str(guide), "exec"), namespace)
    database = tmp_path / "records.sqlite3"
    if record_state != "missing":
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "CREATE TABLE records (record_id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )
            connection.commit()
    interrupted = True
    calls = []

    def create_record(record_id: str, body: str) -> dict:
        calls.append(record_id)
        if interrupted and record_state in {"absent", "missing"}:
            raise KeyboardInterrupt
        with closing(sqlite3.connect(database)) as connection:
            stored_body = "conflicting body" if record_state == "conflicting" else body
            connection.execute("INSERT INTO records VALUES (?, ?)", (record_id, stored_body))
            connection.commit()
        if interrupted:
            raise KeyboardInterrupt
        return {"record_id": record_id, "body": body}

    tools = tools_from([create_record], include_done=True, validate=True)
    config = LoopConfig(
        max_steps=3,
        use_native_tools=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        run_envelope=RunEnvelope(run_id="record-run"),
    )
    response = json.dumps(
        {"tool": "create_record", "args": {"record_id": "record-42", "body": "verified"}}
    )
    with pytest.raises(KeyboardInterrupt):
        list(composable_loop(MockLLMBackend([response]), tools=tools, config=config))
    store = FileCheckpointStore(config.checkpoint_dir)
    pending = store.load_latest_for_resume(run_id="record-run")
    assert pending is not None
    original = {path.name: path.read_bytes() for path in Path(config.checkpoint_dir).glob("*.json")}
    retry_backend = MockLLMBackend([response])
    with pytest.raises(UncertainToolEffect, match="reconcile-an-interrupted-tool"):
        list(composable_loop(retry_backend, tools=tools, config=config))
    assert retry_backend.calls == 0

    reconcile = namespace["reconcile_create_record"]
    if record_state in {"conflicting", "missing"}:
        expected = ValueError if record_state == "conflicting" else sqlite3.OperationalError
        with pytest.raises(expected):
            reconcile(store, database, run_id="record-run")
        assert original == {
            path.name: path.read_bytes() for path in Path(config.checkpoint_dir).glob("*.json")
        }
        assert store.load_latest_for_resume(run_id="record-run").metadata["pending_calls"]
        assert calls == ["record-42"]
        return

    decision = reconcile(store, database, run_id="record-run")
    resolved = store.load(f"step_{pending.step_number}_pending", run_id="record-run")
    assert resolved is not None
    assert resolved.config_snapshot == pending.config_snapshot
    assert resolved.domain_state == pending.domain_state
    assert resolved.conversation_data == pending.conversation_data
    assert not resolved.metadata["pending_calls"]
    assert resolved.metadata["reconciliation"]["call"] == pending.metadata["pending_calls"][0]
    if record_state == "committed":
        assert decision == "stopped"
        assert resolved.is_terminal and resolved.run_status == "failed"
        assert resolved.termination_reason == "reconciled_tool_effect"
        assert store.load_latest_for_resume(run_id="record-run") is None
        assert calls == ["record-42"]
    else:
        assert decision == "retry" and not resolved.is_terminal
        interrupted = False
        state = DefaultState(max_steps=3)
        steps = list(
            composable_loop(
                MockLLMBackend([response, '{"tool":"done","args":{"summary":"created"}}']),
                tools=tools,
                config=config,
                state=state,
            )
        )
        assert [step.tool_call.tool for step in steps] == ["create_record", "done"]
        assert RunResult.from_state(state).completed
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT * FROM records").fetchall() == [("record-42", "verified")]


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "incomplete",
        "required-failed",
        "required-skipped",
        "required-error",
        "missing-required",
    ],
)
def test_documented_report_contract(tmp_path: Path, monkeypatch, fault: str) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/tutorial.md"
    blocks = re.findall(r'^```\w+ title="([^"]+)"\n(.*?)^```[ \t]*$', path.read_text(), re.M | re.S)
    cartridge = tmp_path / "report_agent.cartridge"
    section = path.read_text().split("## 1. Use the example cartridge\n", 1)[1]
    commands = re.search(r"```bash\n(.*?)```", section, re.S).group(1).strip().splitlines()
    for command in commands:
        arguments = shlex.split(command)
        if arguments[0] == "looplet-proof":
            arguments = [sys.executable, "-m", "looplet.examples.regression_demo", *arguments[1:]]
        result = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (cartridge / "cartridge.json").is_file()
    for name, code in blocks:
        if name.startswith("tests/") or name.startswith(".github/"):
            continue
        target = cartridge / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code)

    faulty_returns = {
        "required-failed": "False",
        "required-skipped": "None",
        "required-error": "1 / 0",
    }
    if fault in faulty_returns:
        grader_path = next((cartridge / "evals").glob("eval_*.py"))
        module = ast.parse(grader_path.read_text())
        grader = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "eval_completed"
        )
        grader.body = [ast.Return(value=ast.parse(faulty_returns[fault], mode="eval").body)]
        grader_path.write_text(ast.unparse(ast.fix_missing_locations(module)) + "\n")

    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    test_code = dict(blocks)["tests/test_report_agent.py"]
    exec(compile(test_code, str(path), "exec"), namespace)
    run_evals = namespace["run_cartridge_evals"]
    observed = []

    def observed_run(*args, **kwargs):
        if fault == "incomplete":
            kwargs["llm"] = MockLLMBackend(
                [
                    json.dumps({"tool": "publish_report", "args": {"revenue": 120, "cost": 80}}),
                    json.dumps({"tool": "missing_completion_tool", "args": {}}),
                ],
                cycle=True,
            )
        records = run_evals(*args, **kwargs)
        if fault == "missing-required":
            records[0].results[:] = [
                result for result in records[0].results if result.name != "eval_completed"
            ]
        observed.extend(records)
        return records

    namespace["run_cartridge_evals"] = observed_run
    if fault == "none":
        namespace["test_report_contract"](tmp_path)
    else:
        with pytest.raises(AssertionError, match="eval_completed"):
            namespace["test_report_contract"](tmp_path)
    assert observed[0].context.artifacts["observed_profit"] == 40
    assert (
        next(
            result.score
            for result in observed[0].results
            if result.name == "eval_profit_is_correct"
        )
        == 1.0
    )
    assert observed[0].context.completed is (fault != "incomplete")


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

    @pytest.mark.parametrize("limit, completed", [(1, False), (5, True)])
    def test_scripted_assistant_turns_are_opt_in_and_bounded(
        self, monkeypatch, capsys, limit, completed
    ) -> None:
        observed = []

        def execute(**kwargs):
            observed.append(kwargs["state"])
            yield from composable_loop(**kwargs)

        monkeypatch.setattr(hello_world, "composable_loop", execute)
        assert hello_world.main(
            ["--scripted", "--assistant-turns", "--max-model-calls", str(limit)]
        ) == (0 if completed else 1)
        state = observed[0]
        assert RunResult.from_state(state).completed is completed
        assert state._looplet_run_stats["llm_calls"] == limit
        assert len(state.steps) == (3 if completed else 0)
        assert state.metadata["looplet_assistant_turns"]["assistant_turns"] == (
            2 if completed else 1
        )
        assert "First identify" not in capsys.readouterr().out

    def test_assistant_example_requires_explicit_positive_model_limit(self) -> None:
        with pytest.raises(SystemExit) as status:
            hello_world.main(["--scripted", "--assistant-turns"])
        assert status.value.code == 2


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
