"""Dogfood tests for first-run hello examples."""

from __future__ import annotations

import ast
import json
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

from looplet import MockLLMBackend, RunResult, composable_loop
from looplet.examples import hello_world, ollama_hello

pytestmark = pytest.mark.smoke


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"^```python[^\n]*\n(.*?)^```[ \t]*$", path.read_text(), re.M | re.S)


def _quickstart_agent_source(guide: Path) -> str:
    blocks = _python_blocks(guide)
    main = ast.parse(blocks[0]).body[-1]
    assert isinstance(main, ast.If)
    declarations = "".join(blocks[0].splitlines(keepends=True)[: main.lineno - 1])
    return declarations + "\n\n" + blocks[1]


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


def test_quickstart_runs_grades_captures_and_replays(tmp_path: Path, monkeypatch) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "quickstart.md"
    blocks = _python_blocks(guide)
    assert blocks
    from_env = Mock()
    monkeypatch.setattr("looplet.OpenAIBackend.from_env", from_env)
    module = ModuleType("first_loop")
    monkeypatch.setitem(sys.modules, "first_loop", module)
    monkeypatch.chdir(tmp_path)
    exec(compile(_quickstart_agent_source(guide), str(guide), "exec"), module.__dict__)
    tests = {}
    exec(compile(blocks[2], str(guide), "exec"), tests)

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
    capture_block = next(block for block in blocks if "def capture_agent" in block)
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
    exec(compile(blocks[4], str(guide), "exec"), rescored)
    assert rescored["saved_report"].passed
    replayed = {"__name__": "__main__"}
    monkeypatch.setattr(sys, "argv", ["first_loop_replay.py", str(trace_dir)])
    exec(compile(blocks[5], str(guide), "exec"), replayed)
    assert captured["captured_result"].output == replayed["replayed_result"].output
    assert observed_agent.call_count == 1

    stopped = {"__name__": "__main__"}
    exec(compile(blocks[6], str(guide), "exec"), stopped)
    assert stopped["state"].termination_reason == "stream_closed"
    assert stopped["backend"].calls == 1
    assert [step.tool_call.tool for step in stopped["state"].steps] == ["lookup_owner"]
    assert not RunResult.from_state(stopped["state"]).completed
    live = {"__name__": "__main__"}
    exec(compile(blocks[7], str(guide), "exec"), live)
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
    blocks = _python_blocks(guide)
    (tmp_path / "first_loop.py").write_text(_quickstart_agent_source(guide))
    (tmp_path / "test_first_loop.py").write_text(blocks[2])
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

    examples = dict(
        re.findall(r'^```python title="([^"]+)"\n(.*?)^```[ \t]*$', guide.read_text(), re.M | re.S)
    )
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
