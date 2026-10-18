"""Regressions for installed-wheel runner frictions found by dogfooding."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pytest

from looplet.__main__ import main
from looplet.cli import factory_commands
from looplet.testing import MockLLMBackend


def _args(
    cartridge: Path,
    *,
    project_root: Path | None = None,
    max_steps: int = 1,
    json_output: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        workspace=cartridge,
        task="finish",
        max_steps=max_steps,
        project_root=project_root,
        trace_dir=None,
        parent_trace=None,
        no_trace=True,
        quiet=False,
        pretty=False,
        json=json_output,
        require_completed=False,
    )


def _patch_backend(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "mock-model")
    monkeypatch.setattr(factory_commands, "_check_env", lambda: 0)
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: MockLLMBackend(responses=responses, cycle=False),
    )


def _custom_terminal_cartridge(root: Path, *, name: str, return_expression: str) -> Path:
    cartridge = root / f"{name}.cartridge"
    (cartridge / "prompts").mkdir(parents=True)
    (cartridge / "tools" / name).mkdir(parents=True)
    (cartridge / "cartridge.json").write_text(json.dumps({"name": name, "schema_version": 2}))
    (cartridge / "config.yaml").write_text(f"max_steps: 1\ndone_tool: {name}\n")
    (cartridge / "runtime.yaml").write_text("use_native_tools: false\n")
    (cartridge / "prompts" / "system.md").write_text(f"Call {name}.\n")
    (cartridge / "tools" / name / "tool.yaml").write_text(
        f"name: {name}\n"
        "description: Finish with an answer.\n"
        "parameters:\n"
        "  answer:\n"
        "    type: string\n"
        "    description: Final answer.\n"
    )
    (cartridge / "tools" / name / "execute.py").write_text(
        f"def execute(*, answer: str):\n    return {return_expression}\n"
    )
    return cartridge


def test_missing_project_root_fails_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    missing = tmp_path / "missing"
    _patch_backend(monkeypatch, [json.dumps({"tool": "done", "args": {"answer": "ok"}})])
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: pytest.fail("missing project root must fail before backend construction"),
    )

    rc = factory_commands.cmd_run_workspace(_args(cartridge, project_root=missing))

    assert rc == 1
    assert not missing.exists()
    assert "project root is not a directory" in capsys.readouterr().err


def test_python_script_has_an_actionable_command(tmp_path, monkeypatch, capsys):
    script = tmp_path / "my agent.py"
    script.write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setattr(
        "looplet.bundles.load_skill_bundle", lambda *_args: pytest.fail("must not load a bundle")
    )
    assert main(["run", str(script), "finish"]) == 1
    message = capsys.readouterr().err
    assert "python " in message and "my agent.py'" in message
    assert "SKILL.md" not in message


@pytest.mark.parametrize(
    ("command", "package_kind"),
    [
        ("run", "cartridge"),
        ("run-cartridge", "cartridge"),
        ("run-workspace", "cartridge"),
        ("run", "bundle"),
        ("run-bundle", "bundle"),
    ],
)
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("chosen-model", None),
        (None, "http://localhost:11434/v1"),
        ("chosen-model", "http://localhost:11434/v1"),
    ],
)
def test_run_provider_options_reach_backend_without_environment_changes(
    tmp_path, monkeypatch, capsys, command, package_kind, json_output, model, base_url
):
    if package_kind == "cartridge":
        target = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
        terminal_args = {"answer": "ok"}
        options = []
    else:
        target = tmp_path / "bundle"
        shutil.copytree(Path(__file__).parent / "fixtures" / "coder_skill_bundle", target)
        terminal_args = {"summary": "finished"}
        options = ["--no-tests"]
    monkeypatch.setenv("LOOPLET_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    environment = dict(os.environ)
    received = []

    def build_backend(**kwargs):
        received.append(kwargs)
        return MockLLMBackend([json.dumps({"tool": "done", "args": terminal_args})])

    def openai_backend(**kwargs):
        return build_backend(provider="openai", **kwargs)

    monkeypatch.setattr("looplet.backends.make_backend", build_backend)
    monkeypatch.setattr("looplet.backends.OpenAIBackend", openai_backend)
    arguments = [
        command,
        str(target),
        "finish",
        "--project-root",
        str(tmp_path),
        "--no-trace",
        *options,
    ]
    if model is not None:
        arguments.extend(["--model", model])
    if base_url is not None:
        arguments.extend(["--base-url", base_url])
    if json_output:
        arguments.append("--json")
    assert main(arguments) == 0
    assert len(received) == 1
    assert received[0].get("model") == (model or "environment-model" if base_url else model)
    assert received[0].get("base_url") == base_url
    assert received[0].get("provider") == ("openai" if base_url else None)
    assert dict(os.environ) == environment
    output = capsys.readouterr()
    if json_output:
        assert json.loads(output.out)["completed"] is True
    assert not output.err


@pytest.mark.parametrize("package_kind", ["cartridge", "bundle"])
def test_python_suffix_directory_keeps_packaged_agent_routing(tmp_path, capsys, package_kind):
    target = tmp_path / "agent.py"
    if package_kind == "cartridge":
        cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
        cartridge.rename(target)
        terminal_args = {"answer": "ok"}
        options = []
    else:
        shutil.copytree(Path(__file__).parent / "fixtures" / "coder_skill_bundle", target)
        terminal_args = {"summary": "finished"}
        options = ["--no-tests"]

    assert (
        main(
            [
                "run",
                str(target),
                "finish",
                "--project-root",
                str(tmp_path),
                "--max-steps",
                "1",
                "--require-completed",
                "--no-trace",
                "--json",
                "--scripted-response",
                json.dumps({"tool": "done", "args": terminal_args}),
                *options,
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["completed"] is True
    assert not output.err


@pytest.mark.parametrize("require_completed", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_incomplete_run_requires_completion_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, require_completed, json_output
) -> None:
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    work = cartridge / "tools" / "work"
    work.mkdir()
    (work / "tool.yaml").write_text("name: work\ndescription: Work.\nparameters: {}\n")
    (work / "execute.py").write_text("def execute():\n    return {'ok': True}\n")
    _patch_backend(
        monkeypatch,
        [json.dumps({"tool": "work", "args": {}, "reasoning": "work"})],
    )

    arguments = _args(cartridge, project_root=tmp_path, json_output=json_output)
    arguments.require_completed = require_completed
    rc = factory_commands.cmd_run_workspace(arguments)

    assert rc == int(require_completed)
    out = capsys.readouterr().out
    if json_output:
        assert json.loads(out)["completed"] is False
    else:
        assert "stopped (budget_exhausted)" in out
        assert "done in" not in out


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("require_completed", [False, True])
@pytest.mark.parametrize("tool_name", ["done", "missing_tool"])
def test_failed_calls_report_their_error_without_false_completion(
    tmp_path, monkeypatch, capsys, json_output, require_completed, tool_name
):
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    _patch_backend(monkeypatch, [json.dumps({"tool": tool_name, "args": {}})])
    arguments = _args(cartridge, project_root=tmp_path, json_output=json_output)
    arguments.require_completed = require_completed

    assert factory_commands.cmd_run_workspace(arguments) == int(require_completed)

    captured = capsys.readouterr()
    expected = "missing required argument" if tool_name == "done" else "Unknown tool"
    if json_output:
        payload = json.loads(captured.out)
        assert payload["completed"] is False
        assert payload["termination_reason"] == "budget_exhausted"
        assert payload["result"] is None
        assert payload["error"]["step"] == 1
        assert payload["error"]["tool"] == tool_name
        assert expected in payload["error"]["message"]
        assert not captured.err
    else:
        assert expected in captured.out
        assert "stopped (budget_exhausted)" in captured.out
        assert "done in" not in captured.out


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("require_completed", [False, True])
@pytest.mark.parametrize("tool_name", ["done", "missing_tool"])
def test_generic_bundle_failure_presentation_uses_shared_error(
    tmp_path, capsys, json_output, require_completed, tool_name
):
    target = tmp_path / "bundle"
    shutil.copytree(Path(__file__).parent / "fixtures" / "coder_skill_bundle", target)
    arguments = [
        "run",
        str(target),
        "finish",
        "--project-root",
        str(tmp_path),
        "--no-trace",
        "--no-tests",
        "--max-steps",
        "1",
        "--model",
        "scripted-model",
        "--scripted-response",
        json.dumps({"tool": tool_name, "args": {"invalid": "wrong field"}}),
    ]
    if json_output:
        arguments.append("--json")
    arguments.append("--require-completed" if require_completed else "--allow-incomplete")

    assert main(arguments) == int(require_completed)

    output = capsys.readouterr()
    expected = "unexpected argument" if tool_name == "done" else "Unknown tool"
    if json_output:
        payload = json.loads(output.out)
        assert payload["completed"] is False
        assert payload["result"] is None
        assert payload["error"]["tool"] == tool_name
        assert expected in payload["error"]["message"]
    else:
        assert "stopped (budget_exhausted)" in output.out
        assert expected in output.out


@pytest.mark.parametrize("command", ["run", "run-cartridge", "run-workspace"])
@pytest.mark.parametrize("policy", [None, "--require-completed", "--allow-incomplete"])
@pytest.mark.parametrize("json_output", [False, True])
def test_cli_requires_completion_by_default(tmp_path, capsys, command, policy, json_output):
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    arguments = [
        command,
        str(cartridge),
        "finish",
        "--project-root",
        str(tmp_path),
        "--max-steps",
        "1",
        "--no-trace",
        "--scripted-response",
        '{"tool":"done","args":{}}',
    ]
    if policy is not None:
        arguments.append(policy)
    if json_output:
        arguments.append("--json")

    assert main(arguments) == int(policy != "--allow-incomplete")
    output = capsys.readouterr()
    if json_output:
        payload = json.loads(output.out)
        assert payload["completed"] is False
        assert payload["termination_reason"] == "budget_exhausted"
        assert "missing required argument" in payload["error"]["message"]
    else:
        assert "stopped (budget_exhausted)" in output.out
        assert "missing required argument" in output.out


@pytest.mark.parametrize("policy", [None, "--require-completed", "--allow-incomplete"])
@pytest.mark.parametrize("json_output", [False, True])
def test_replay_requires_completion_by_default(tmp_path, capsys, policy, json_output):
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    trace = tmp_path / "original-trace"
    assert (
        main(
            [
                "run",
                str(cartridge),
                "finish",
                "--project-root",
                str(tmp_path),
                "--trace-dir",
                str(trace),
                "--allow-incomplete",
                "--max-steps",
                "1",
                "--scripted-response",
                '{"tool":"done","args":{}}',
            ]
        )
        == 0
    )
    capsys.readouterr()
    original = {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    }
    arguments = [
        "replay",
        str(trace),
        "--cartridge",
        str(cartridge),
        "--workspace",
        str(tmp_path / "replay-workspace"),
    ]
    if policy is not None:
        arguments.append(policy)
    if json_output:
        arguments.append("--json")

    assert main(arguments) == int(policy != "--allow-incomplete")
    output = capsys.readouterr()
    if json_output:
        payload = json.loads(output.out)
        assert payload["completed"] is False
        assert payload["termination_reason"] == "budget_exhausted"
        assert payload["error"]["tool"] == "done"
    else:
        assert "stopped (budget_exhausted)" in output.out
    assert {
        str(path.relative_to(trace)): path.read_bytes()
        for path in trace.rglob("*")
        if path.is_file()
    } == original


@pytest.mark.parametrize(
    "command", ["run", "run-cartridge", "run-workspace", "run-bundle", "replay"]
)
def test_completion_flags_are_mutually_exclusive(tmp_path, capsys, command):
    arguments = [command, str(tmp_path)]
    if command == "replay":
        arguments.extend(["--cartridge", str(tmp_path), "--workspace", str(tmp_path / "fresh")])
    else:
        arguments.append("finish")
    arguments.extend(["--require-completed", "--allow-incomplete"])

    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_recovered_completion_does_not_report_an_old_error(tmp_path, monkeypatch, capsys):
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    _patch_backend(
        monkeypatch,
        [
            json.dumps({"tool": "done", "args": {}}),
            json.dumps({"tool": "done", "args": {"answer": "accepted"}}),
        ],
    )
    arguments = _args(cartridge, project_root=tmp_path, max_steps=2)
    arguments.require_completed = True
    assert factory_commands.cmd_run_workspace(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is True
    assert payload["error"] is None


def test_completion_error_is_bounded_and_excludes_tool_payloads():
    from looplet import RunPhase, RunResult, RunStatus, Step, ToolCall, ToolResult
    from looplet.cli import completion_payload

    step = Step(
        number=1,
        tool_call=ToolCall(tool="done", args={"private": "not diagnostic output"}),
        tool_result=ToolResult(
            tool="done",
            args_summary="private",
            data={"private": "not diagnostic output"},
            error="x" * 3000,
        ),
    )
    result = RunResult(RunStatus.STOPPED, RunPhase.TERMINAL, "budget_exhausted", None, (step,))
    payload = completion_payload(result, steps=1, duration_ms=0, trace_dir=None)
    assert payload["error"] == {"step": 1, "tool": "done", "message": "x" * 2000}
    assert "private" not in json.dumps(payload)


@pytest.mark.parametrize("command", ["run", "run-cartridge", "run-workspace"])
@pytest.mark.parametrize("completed", [False, True])
def test_completion_requirement_is_available_on_all_cartridge_commands(
    tmp_path, capsys, command, completed
):
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    tool_name = "done" if completed else "missing_tool"
    assert main(
        [
            command,
            str(cartridge),
            "finish",
            "--project-root",
            str(tmp_path),
            "--require-completed",
            "--no-trace",
            "--json",
            "--max-steps",
            "1",
            "--scripted-response",
            json.dumps({"tool": tool_name, "args": {"answer": "ok"}}),
        ]
    ) == (0 if completed else 1)
    assert json.loads(capsys.readouterr().out)["completed"] is completed


@pytest.mark.parametrize(
    ("name", "return_expression", "expected"),
    [
        ("report", "{'answer': answer}", {"answer": "ok"}),
        ("finish", "answer", "ok"),
    ],
)
def test_custom_terminal_result_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    name: str,
    return_expression: str,
    expected: object,
) -> None:
    cartridge = _custom_terminal_cartridge(
        tmp_path,
        name=name,
        return_expression=return_expression,
    )
    _patch_backend(
        monkeypatch,
        [json.dumps({"tool": name, "args": {"answer": "ok"}, "reasoning": "finish"})],
    )

    rc = factory_commands.cmd_run_workspace(_args(cartridge, project_root=tmp_path))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is True
    assert payload["result"] == expected


def test_runner_closes_real_mcp_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import looplet

    cartridge = Path(__file__).parents[1] / "examples" / "mcp_demo.cartridge"
    captured: dict[str, object] = {}
    original_loader = looplet.cartridge_to_preset

    def capture_loader(*args, **kwargs):
        preset = original_loader(*args, **kwargs)
        captured["preset"] = preset
        captured["adapter"] = preset.mcp_adapters[0]
        original_set_backend = preset.model_gateway.set_backend

        def capture_backend(backend):
            captured["backend"] = backend
            original_set_backend(backend)

        monkeypatch.setattr(preset.model_gateway, "set_backend", capture_backend)
        return preset

    monkeypatch.setattr(looplet, "cartridge_to_preset", capture_loader)
    _patch_backend(
        monkeypatch,
        [
            json.dumps({"tool": "add", "args": {"a": 1, "b": 2}, "reasoning": "add"}),
            json.dumps({"tool": "done", "args": {"total": 3}, "reasoning": "done"}),
        ],
    )

    rc = factory_commands.cmd_run_workspace(_args(cartridge, project_root=tmp_path, max_steps=2))

    assert rc == 0
    json.loads(capsys.readouterr().out)
    preset = captured["preset"]
    adapter = captured["adapter"]
    assert isinstance(preset, looplet.AgentPreset)
    assert isinstance(adapter, looplet.MCPToolAdapter)
    assert isinstance(captured["backend"], MockLLMBackend)
    assert preset.run_claimed
    assert preset.mcp_adapters == []
    assert preset.model_gateway is None
    assert adapter._proc is None


def test_runner_returns_nonzero_on_fatal_provider_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    _patch_backend(monkeypatch, [])

    class FailingBackend:
        def generate(self, prompt: str, **kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(factory_commands, "_build_backend", FailingBackend)

    rc = factory_commands.cmd_run_workspace(_args(cartridge, project_root=tmp_path))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is False
    assert payload["termination_reason"] == "llm_error"
    assert payload["result"] is None
    assert "provider unavailable" in payload["error"]["message"]


def test_runner_reports_trace_failure_and_closes_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import looplet

    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    _patch_backend(monkeypatch, [json.dumps({"tool": "done", "args": {"answer": "ok"}})])
    trace_path = tmp_path / "not-a-directory"
    trace_path.write_text("existing file")
    captured = {}
    original_loader = looplet.cartridge_to_preset

    def capture_loader(*args, **kwargs):
        preset = original_loader(*args, **kwargs)
        captured["preset"] = preset
        return preset

    monkeypatch.setattr(looplet, "cartridge_to_preset", capture_loader)
    arguments = _args(cartridge, project_root=tmp_path)
    arguments.no_trace = False
    arguments.trace_dir = trace_path

    try:
        assert factory_commands.cmd_run_workspace(arguments) == 1
        assert captured["preset"]._closed
        output = capsys.readouterr()
        assert "could not save trace" in output.err
        assert not output.out
    finally:
        if "preset" in captured:
            captured["preset"].close()


@pytest.mark.parametrize("json_output", [False, True])
def test_runner_routes_authored_output_without_changing_human_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, json_output: bool
) -> None:
    cartridge = _custom_terminal_cartridge(tmp_path, name="done", return_expression="{}")
    (cartridge / "tools" / "done" / "execute.py").write_text(
        "print('loading diagnostic')\n"
        "def execute(*, answer: str):\n"
        "    print('execution diagnostic')\n"
        "    return {'answer': answer}\n"
    )
    _patch_backend(monkeypatch, [json.dumps({"tool": "done", "args": {"answer": "ok"}})])

    assert (
        factory_commands.cmd_run_workspace(
            _args(cartridge, project_root=tmp_path, json_output=json_output)
        )
        == 0
    )

    output = capsys.readouterr()
    if json_output:
        payload = json.loads(output.out)
        assert payload["completed"] is True
        assert payload["result"] == {"answer": "ok"}
        assert "loading diagnostic" in output.err
        assert "execution diagnostic" in output.err
    else:
        assert "loading diagnostic" in output.out
        assert "execution diagnostic" in output.out
        assert not output.err


def test_run_dispatches_cartridge_and_preserves_scalar_result(tmp_path: Path, capsys) -> None:
    cartridge = _custom_terminal_cartridge(tmp_path, name="finish", return_expression="answer")

    assert (
        main(
            [
                "run",
                str(cartridge),
                "finish",
                "--workspace",
                str(tmp_path),
                "--json",
                "--no-trace",
                "--scripted-response",
                '{"tool":"finish","args":{"answer":"ok"}}',
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] is True
    assert payload["result"] == "ok"


@pytest.mark.parametrize("command", ["run", "run-cartridge", "run-workspace"])
@pytest.mark.parametrize("flag", ["--quiet", "--pretty", "--parent-trace"])
def test_run_commands_share_cartridge_options(tmp_path, monkeypatch, command, flag):
    cartridge = _custom_terminal_cartridge(tmp_path, name="finish", return_expression="answer")
    captured = {}

    def capture_run(arguments):
        captured.update(vars(arguments))
        return 0

    monkeypatch.setattr(factory_commands, "cmd_run_workspace", capture_run)
    arguments = [command, str(cartridge), "finish", "--project-root", str(tmp_path), flag]
    if flag == "--parent-trace":
        arguments.append(str(tmp_path / "parent"))

    assert main(arguments) == 0
    assert captured["workspace"] == cartridge
    assert captured["project_root"] == tmp_path
    if flag == "--parent-trace":
        assert captured["parent_trace"] == tmp_path / "parent"
    else:
        assert captured[flag.removeprefix("--")] is True


def test_run_rejects_ambiguous_package_before_backend(tmp_path, monkeypatch, capsys) -> None:
    cartridge = _custom_terminal_cartridge(tmp_path, name="finish", return_expression="answer")
    (cartridge / "SKILL.md").write_text("---\nname: also-a-bundle\n---\n")
    monkeypatch.setattr(
        factory_commands, "_build_backend", lambda: pytest.fail("ambiguous format must fail early")
    )
    assert main(["run", str(cartridge), "finish", "--project-root", str(tmp_path)]) == 1
    assert "use run-cartridge or run-bundle explicitly" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag, message",
    [
        ("--scripted", "use --scripted-response for cartridges"),
        ("--no-tests", "--no-tests is only for skill bundles"),
    ],
)
def test_run_rejects_bundle_only_flags_for_cartridges(tmp_path, monkeypatch, capsys, flag, message):
    cartridge = _custom_terminal_cartridge(tmp_path, name="finish", return_expression="answer")
    monkeypatch.setattr(
        factory_commands, "_build_backend", lambda: pytest.fail("unsupported flags must fail early")
    )
    assert main(["run", str(cartridge), "finish", "--project-root", str(tmp_path), flag]) == 1
    assert message in capsys.readouterr().err
