#!/usr/bin/env python3
"""Validate package-only surfaces from an installed Looplet wheel."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import looplet
from looplet.bundles import load_skill_bundle
from looplet.cartridge import analyse_cartridge
from looplet.cli.factory_commands import _factory_workspace_path


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
