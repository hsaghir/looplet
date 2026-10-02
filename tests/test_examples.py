"""Tests for looplet example agents - verify they run without errors."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CARTRIDGES = sorted(path.parent for path in EXAMPLES.rglob("cartridge.json"))


@pytest.mark.parametrize("cartridge", CARTRIDGES, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_every_example_cartridge_has_a_valid_owned_preset(cartridge, tmp_path):
    from looplet import cartridge_to_preset, validate_preset_contract

    with cartridge_to_preset(
        cartridge, runtime={"project_root": str(tmp_path)}, strict=True
    ) as preset:
        validation = validate_preset_contract(preset)
        assert validation.ok, validation.errors
        for spec in preset.tools.tool_specs.values():
            assert spec.to_api_schema()["input_schema"] == spec.to_json_schema()


@pytest.mark.parametrize("cartridge", CARTRIDGES, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_every_example_cartridge_has_an_offline_declaration_view(cartridge, tmp_path):
    from looplet.cartridge import inspect_cartridge

    report = inspect_cartridge(cartridge, runtime={"project_root": str(tmp_path)})
    assert report["mode"] == "declaration-only"
    assert report["runtime_validated"] is False
    assert isinstance(report["tools"], list)
    assert "config" in report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("approval", [None, "no", "yes"])
def test_scripted_demo_changes_real_rows_only_after_approval(approval):
    from looplet.examples.scripted_demo import build_tools
    from looplet.types import ToolCall, ToolContext

    rows = [
        {"id": 1, "status": "paid"},
        {"id": 2, "status": "cancelled"},
        {"id": 3, "status": "cancelled"},
    ]
    original = list(rows)
    tools = build_tools(rows)
    before = tools.dispatch(ToolCall(tool="count_by_status", args={}))
    assert before.data == {"counts": {"paid": 1, "cancelled": 2}}
    context = ToolContext(request_approval=lambda prompt, options: approval)
    result = tools.dispatch(
        ToolCall(tool="delete_rows", args={"where_status": "cancelled"}), ctx=context
    )
    assert result.error is None
    if approval == "yes":
        assert rows == [original[0]]
        assert result.data == {"deleted": 2, "remaining": 1}
        assert tools.dispatch(ToolCall(tool="count_by_status", args={})).data == {
            "counts": {"paid": 1}
        }
    else:
        assert rows == original


def test_scripted_demo_executes_four_steps_without_waiting(monkeypatch, capsys):
    from looplet.examples import scripted_demo

    monkeypatch.setattr(scripted_demo.time, "sleep", lambda delay: None)
    assert scripted_demo.main([]) == 0
    assert "1 approval prompt, 4 tools" in capsys.readouterr().out


def _snippet(relative):
    path = EXAMPLES / "snippets" / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portability_snippet_executes_all_three_paths():
    snippet = _snippet("08_portability/run_three_ways.py")
    assert snippet.via_local_loop() == 2
    assert snippet.via_subagent() == 2
    assert snippet.via_scripted_rerun() == 2


def test_delegation_snippet_keeps_child_resources_and_hooks(monkeypatch):
    from looplet import MockLLMBackend

    snippet = _snippet("04_subagent/demo.py")
    backend = MockLLMBackend(
        [
            '{"tool":"ask_helper","args":{"question":"Greet Alice"}}',
            '{"tool":"greet","args":{"name":"Alice"}}',
            '{"tool":"done","args":{"summary":"greeted"}}',
            '{"tool":"done","args":{"summary":"helper finished"}}',
        ],
        cycle=False,
    )
    monkeypatch.setattr(snippet, "_backend", lambda: backend)
    assert snippet.main(["Delegate greeting Alice"]) == 0
    assert backend.calls == 4


def test_lifecycle_snippet_runs_cartridge_wiring(monkeypatch, capsys):
    import sys

    snippet = _snippet("09_lifecycle/tag_trajectory.py")
    monkeypatch.setattr(sys, "argv", ["tag_trajectory.py", str(EXAMPLES / "hello.cartridge")])
    snippet.main()
    record = json.loads(capsys.readouterr().out)
    assert [step["tool"] for step in record["trajectory"]] == ["greet", "done"]
    assert all(step["ok"] for step in record["trajectory"])


@pytest.mark.parametrize(
    ("relative", "arguments"),
    [
        ("02_ablation/ablate.py", []),
        ("03_diff/diff_workspaces.py", ["hello.cartridge", "hello.cartridge"]),
        ("06_registry/registry.py", ["list", "."]),
        ("07_admission/admit.py", ["hello.cartridge"]),
        ("10_evolution/evolve.py", ["hello.cartridge", "1"]),
    ],
)
def test_static_snippet_commands_execute(relative, arguments):
    import subprocess
    import sys

    command = [
        sys.executable,
        str(EXAMPLES / "snippets" / relative),
        *[
            str(EXAMPLES / argument)
            if argument.endswith(".cartridge") or argument == "."
            else argument
            for argument in arguments
        ],
    ]
    result = subprocess.run(
        command, cwd=EXAMPLES.parent, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_quality_gate_snippet_preserves_completion_rejection():
    import shlex
    import sys

    snippet = _snippet("11_quality_gate/quality_gate.py")
    interpreter = shlex.quote(sys.executable)
    passing = snippet.QualityGate(cmd=f"{interpreter} -c 'raise SystemExit(0)'")
    failing = snippet.QualityGate(cmd=f"{interpreter} -c 'raise SystemExit(1)'")
    assert passing.check_done(None, None, None, 1) is None
    decision = failing.check_done(None, None, None, 1)
    assert decision.is_block()
    assert "exit 1" in decision.block


# ── hello_world ──────────────────────────────────────────────────


def test_hello_world_importable():
    import looplet.examples.hello_world  # noqa: F401


def test_hello_world_has_main():
    import looplet.examples.hello_world as m

    assert hasattr(m, "main") and callable(m.main)


# ── coding_agent ─────────────────────────────────────────────────


def test_coding_agent_importable():
    import looplet.examples.coding_agent  # noqa: F401


def test_coding_agent_has_run_function():
    import looplet.examples.coding_agent as m

    assert hasattr(m, "run_coding_agent") and callable(m.run_coding_agent)


def test_coding_agent_has_build_tools():
    import looplet.examples.coding_agent as m

    assert hasattr(m, "build_tools") and callable(m.build_tools)


def test_coding_agent_has_guardrail_hook():
    import looplet.examples.coding_agent as m

    assert hasattr(m, "CodingGuardrailHook")
