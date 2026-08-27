"""Regressions for installed-wheel runner frictions found by dogfooding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

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


def test_incomplete_human_run_does_not_say_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
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

    rc = factory_commands.cmd_run_workspace(
        _args(cartridge, project_root=tmp_path, json_output=False)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "stopped (budget_exhausted)" in out
    assert "done in" not in out


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
    assert preset.mcp_adapters == []
    assert preset.model_gateway is None
    assert adapter._proc is None
