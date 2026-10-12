#!/usr/bin/env python3
"""Validate package-only surfaces from an installed Looplet wheel."""

from __future__ import annotations

import ast
import asyncio
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import looplet
from looplet.bundles import load_skill_bundle
from looplet.cartridge import analyse_cartridge
from looplet.cli.factory_commands import _factory_workspace_path


def _raw_result_client(factory, llm):
    with factory() as preset:
        iterator = preset.run(llm, task={"description": "Greet Ada and finish"})
        try:
            for _step in iterator:
                pass
        finally:
            iterator.close()
        return looplet.RunResult.from_state(preset.state)


def _result_client(factory, llm):
    return looplet.run(factory, "Greet Ada and finish", llm=llm)


def _raw_stream_client(factory, llm):
    with factory() as preset:
        iterator = preset.run(llm, task={"description": "Greet Ada and finish"})
        try:
            for step in iterator:
                assert step.tool_call.tool in {"greet", "done"}
        finally:
            iterator.close()
        return looplet.RunResult.from_state(preset.state)


def _stream_client(factory, llm):
    with looplet.stream(factory, "Greet Ada and finish", llm=llm) as execution:
        for step in execution:
            assert step.tool_call.tool in {"greet", "done"}
    return execution.result


async def _raw_async_result_client(factory, llm):
    with factory() as preset:
        iterator = preset.run_async(llm, task={"description": "Greet Ada and finish"})
        try:
            async for _step in iterator:
                pass
        finally:
            await iterator.aclose()
        return looplet.RunResult.from_state(preset.state)


async def _async_result_client(factory, llm):
    return await looplet.run_async(factory, "Greet Ada and finish", llm=llm)


async def _raw_async_stream_client(factory, llm):
    with factory() as preset:
        iterator = preset.run_async(llm, task={"description": "Greet Ada and finish"})
        try:
            async for step in iterator:
                assert step.tool_call.tool in {"greet", "done"}
        finally:
            await iterator.aclose()
        return looplet.RunResult.from_state(preset.state)


async def _async_stream_client(factory, llm):
    async with looplet.stream_async(factory, "Greet Ada and finish", llm=llm) as execution:
        async for step in execution:
            assert step.tool_call.tool in {"greet", "done"}
    return execution.result


def _qualify_python_clients() -> None:
    built = []
    closed = []
    observed = []

    class Resource:
        def close(self):
            closed.append(self)

    def factory():
        greetings = []

        def greet(name: str) -> dict:
            greetings.append(name)
            return {"greeting": f"Hello, {name}!"}

        preset = looplet.AgentPreset(
            tools=looplet.tools_from([greet], include_done=True),
            state=looplet.DefaultState(max_steps=2),
            config=looplet.LoopConfig(max_steps=2, use_native_tools=False),
            hooks=[],
            owned_resources=[Resource()],
        )
        built.append(preset)
        observed.append(greetings)
        return preset

    pairs = (
        (_raw_result_client, _result_client),
        (_raw_stream_client, _stream_client),
        (_raw_async_result_client, _async_result_client),
        (_raw_async_stream_client, _async_stream_client),
    )
    measured = []
    for raw_client, client in pairs:
        for completed in (True, False):
            responses = [json.dumps({"tool": "greet", "args": {"name": "Ada"}})]
            if completed:
                responses.append(json.dumps({"tool": "done", "args": {"summary": "qualified"}}))
            outcomes = []
            for program in (raw_client, client):
                async_run = inspect.iscoroutinefunction(program)
                backend_type = looplet.AsyncMockLLMBackend if async_run else looplet.MockLLMBackend
                execution = program(factory, backend_type(responses))
                result = asyncio.run(execution) if async_run else execution
                assert result.completed is completed
                assert not result.failed
                assert observed[-1] == (["Ada"] if completed else ["Ada", "Ada"])
                assert len(closed) == len(built)
                outcomes.append((result.status, result.termination_reason, result.output))
            assert outcomes[0] == outcomes[1]

        def statement_count(program):
            syntax = ast.parse(textwrap.dedent(inspect.getsource(program)))
            return sum(
                isinstance(node, ast.stmt)
                and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(syntax)
            )

        measured.append(
            {
                "client": client.__name__,
                "raw_wrapper_statements": statement_count(raw_client),
                "wrapper_statements": statement_count(client),
            }
        )
    assert len(built) == 16
    assert len({id(preset.state) for preset in built}) == 16
    assert all(preset._closed for preset in built)
    print("Scripted client qualification: " + json.dumps({"executions": 16, "wrappers": measured}))


def _smoke_cli_workflow(coder_bundle_path: Path) -> None:
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "LOOPLET_PROVIDER",
        "LOOPLET_ALLOW_MOCK",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "COPILOT_PROXY_URL",
        "COPILOT_PROXY_KEY",
        "LOOPLET_TAX_LLM_BASE_URL",
        "LOOPLET_TAX_LLM_API_KEY",
    ):
        environment.pop(name, None)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def run_cli(*arguments: str, task_input: str | None = None, expected_exit: int = 0):
            result = subprocess.run(
                [sys.executable, "-m", "looplet", *arguments],
                cwd=root,
                env=environment,
                input=task_input,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == expected_exit, result.stderr or result.stdout
            return result

        cartridge = root / "offline.cartridge"
        run_cli("new", "Report completion", str(cartridge), "--offline")
        terminal_code = cartridge / "tools" / "done" / "execute.py"
        original_code = terminal_code.read_text(encoding="utf-8")
        terminal_code.write_text(
            "raise AssertionError('inspection imported tool code')\n", encoding="utf-8"
        )
        inspection = json.loads(run_cli("inspect", str(cartridge), "--json").stdout)
        assert inspection["runtime_validated"] is False
        terminal_code.write_text(original_code, encoding="utf-8")
        runtime_inspection = json.loads(
            run_cli("inspect", str(cartridge), "--mode", "runtime", "--json").stdout
        )
        assert runtime_inspection["mode"] == "runtime"
        assert runtime_inspection["runtime_validated"] is True
        run_cli("new", "Do not overwrite", str(cartridge), "--offline", expected_exit=1)
        assert terminal_code.read_text(encoding="utf-8") == original_code

        response = json.dumps({"tool": "done", "args": {"summary": "completed"}})
        task = "exact task\nsecond line\n"
        completion = json.loads(
            run_cli(
                "run",
                str(cartridge),
                "-",
                "--project-root",
                str(root),
                "--json",
                "--scripted-response",
                response,
                task_input=task,
            ).stdout
        )
        assert completion["completed"] is True
        assert completion["result"]["summary"] == "completed"
        trace = str(completion["trace_dir"])
        inspected_trace = json.loads(run_cli("inspect", trace, "--json").stdout)
        assert inspected_trace == json.loads(run_cli("show", trace, "--json").stdout)
        assert task in inspected_trace["trajectory"]["task"].values()

        follow_up = json.loads(
            run_cli(
                "run",
                str(cartridge),
                "finish follow-up",
                "--project-root",
                str(root),
                "--parent-trace",
                trace,
                "--quiet",
                "--json",
                "--scripted-response",
                response,
            ).stdout
        )
        assert follow_up["completed"] is True
        follow_up_trace = json.loads(run_cli("inspect", follow_up["trace_dir"], "--json").stdout)
        assert (
            follow_up_trace["trajectory"]["metadata"]["parent_run_id"]
            == inspected_trace["trajectory"]["run_id"]
        )
        pretty = run_cli(
            "run",
            str(cartridge),
            "finish",
            "--project-root",
            str(root),
            "--pretty",
            "--no-trace",
            "--scripted-response",
            response,
        )
        assert "completed" in pretty.stdout

        legacy = json.loads(
            run_cli(
                "run-workspace",
                str(cartridge),
                "finish",
                "--project-root",
                str(root),
                "--json",
                "--no-trace",
                "--scripted-response",
                response,
            ).stdout
        )
        assert legacy["completed"] is True
        assert legacy["trace_dir"] is None

        bundle = json.loads(
            run_cli(
                "run-bundle",
                str(coder_bundle_path),
                "finish",
                "--workspace",
                str(root),
                "--max-steps",
                "1",
                "--json",
                "--no-trace",
                "--no-tests",
                "--scripted-response",
                response,
            ).stdout
        )
        assert bundle["completed"] is True
        assert bundle["steps"] == 1
        assert bundle["result"]["summary"] == "completed"

        declaration_bundle = root / "declaration_bundle"
        declaration_bundle.mkdir()
        (declaration_bundle / "SKILL.md").write_text(
            "---\nname: declaration-only\ndescription: Installed blueprint probe.\n---\n",
            encoding="utf-8",
        )
        (declaration_bundle / "looplet.py").write_text(
            "raise AssertionError('declaration inspection imported code')\n", encoding="utf-8"
        )
        nested_bundle = declaration_bundle / "nested"
        nested_bundle.mkdir()
        (nested_bundle / "SKILL.md").write_text(
            "---\nname: aaa-child\ndescription: Nested bundle.\n---\n", encoding="utf-8"
        )
        (nested_bundle / "looplet.py").write_text(
            "raise AssertionError('inspection imported nested code')\n", encoding="utf-8"
        )
        inspected_bundle = json.loads(run_cli("inspect", str(declaration_bundle), "--json").stdout)
        assert inspected_bundle["name"] == "declaration-only"
        assert inspected_bundle["runtime_validated"] is False
        run_cli("inspect", str(declaration_bundle), "--mode", "runtime", expected_exit=1)
        discovered = json.loads(run_cli("list-bundles", str(declaration_bundle), "--json").stdout)
        assert {card["name"] for card in discovered} == {"aaa-child", "declaration-only"}
        declaration = json.loads(
            run_cli("blueprint", str(declaration_bundle), "--declaration-only").stdout
        )
        assert declaration["metadata"]["runtime_validated"] is False
        run_cli("blueprint", str(declaration_bundle), "--instantiate", expected_exit=1)
        run_cli(
            "run-bundle",
            str(coder_bundle_path),
            "finish",
            "--workspace",
            str(root),
            "--pretty",
            expected_exit=1,
        )

        missing = root / "missing"
        run_cli(
            "run",
            str(coder_bundle_path),
            "finish",
            "--workspace",
            str(missing),
            "--scripted-response",
            response,
            expected_exit=1,
        )
        assert not missing.exists()


def main() -> int:
    package_root = Path(looplet.__file__).resolve().parent
    factory = _factory_workspace_path().resolve()
    _qualify_python_clients()

    assert package_root in factory.parents
    assert factory == package_root / "_bundled" / "agent_factory.cartridge"
    assert (factory.parent / "coder.cartridge" / "cartridge.json").is_file()

    portable_coder = looplet.bundled_cartridge_path("coder_portable").resolve()
    assert portable_coder == package_root / "_bundled" / "coder_portable.cartridge"
    portability = analyse_cartridge(portable_coder)
    assert portability.profile == "portable"
    assert portability.blockers == ()

    portable_preset = looplet.cartridge_to_preset(portable_coder, strict=True)
    try:
        assert len(portable_preset.mcp_adapters) == 1
        assert len(portable_preset.state_service_handles) == 1
        assert {
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "grep",
            "subagent",
            "done",
        } <= set(portable_preset.tools.tool_names)
    finally:
        portable_preset.close()

    coder_bundle_path = package_root.parent / "tests" / "fixtures" / "coder_skill_bundle"
    coder_bundle = load_skill_bundle(coder_bundle_path)
    assert coder_bundle.skill.name == "coder"
    assert coder_bundle.card.name == "coder"
    _smoke_cli_workflow(coder_bundle_path)

    preset = looplet.cartridge_to_preset(factory)
    try:
        tool_names = set(preset.tools.tool_names)
        assert {"done", "read_file", "scaffold_cartridge", "validate_workspace"} <= tool_names
    finally:
        preset.close()

    result = subprocess.run(
        [sys.executable, "-m", "looplet", "new", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Scaffold a reviewable Looplet cartridge draft" in result.stdout

    conform = subprocess.run(
        [sys.executable, "-m", "looplet", "conform"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert conform.returncode == 0, conform.stderr or conform.stdout
    assert "Cartridge Spec v2.0 conformance" in conform.stdout

    proof_entry_points = importlib.metadata.entry_points(
        group="console_scripts", name="looplet-proof"
    )
    assert len(proof_entry_points) == 1
    proof_command = shutil.which("looplet-proof", path=str(Path(sys.executable).parent))
    assert proof_command is not None
    with tempfile.TemporaryDirectory() as temp_dir:
        proof = subprocess.run(
            [
                proof_command,
                "--out",
                str(Path(temp_dir) / "proof"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    assert proof.returncode == 0, proof.stderr
    assert "required eval: FAIL (0.00)" in proof.stdout
    assert "required eval: PASS (1.00)" in proof.stdout
    print(f"Installed wheel smoke passed: looplet {looplet.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
