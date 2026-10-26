"""Tests for looplet example agents - verify they run without errors."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CARTRIDGES = sorted(path.parent for path in EXAMPLES.rglob("cartridge.json"))


def test_default_harness_example_grades_both_agents_and_entry_paths(tmp_path):
    from looplet.examples.default_harness import grade_deliverable, main

    assert main(["--scripted", "--output", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["trials"]) == 4
    assert all(trial["passed"] for trial in summary["trials"])
    assert all(trial["compactions"] > 0 for trial in summary["trials"])
    assert not any(trial["tool_errors"] for trial in summary["trials"])
    deliverable = tmp_path / "scripted-invoices-cartridge" / "deliverable"
    (deliverable / "batch-0.json").write_text('{"batch":0,"paid_total":-1}')
    assert not grade_deliverable("invoices", deliverable)["passed"]


@pytest.mark.parametrize(
    ("tool_name", "first_args", "second_args", "expected"),
    [
        (
            "record_invoice_total",
            {"batch": 0, "paid_total": 1136},
            {"batch": 1, "paid_total": 1936},
            ["batch-0.json", "batch-1.json"],
        ),
        (
            "record_control",
            {"section": "accounts", "requirement": "saved accounts requirement"},
            {"section": "privileges", "requirement": "saved privileges requirement"},
            ["accounts.json", "privileges.json"],
        ),
    ],
)
def test_default_harness_writes_report_durable_progress(
    tool_name, first_args, second_args, expected, tmp_path
):
    from looplet import ToolContext
    from looplet.examples import default_harness

    write = getattr(default_harness, tool_name)
    context = ToolContext(metadata={"deliverable_dir": str(tmp_path)})
    assert write(**first_args, ctx=context)["completed_records"] == expected[:1]
    assert write(**second_args, ctx=context)["completed_records"] == expected
    assert write(**first_args, ctx=context)["completed_records"] == expected
    assert all(
        "completed_records" not in json.loads(path.read_text()) for path in tmp_path.glob("*.json")
    )


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


@pytest.mark.parametrize("mode", ["sync", "async", "parallel"])
@pytest.mark.parametrize(
    "name",
    [
        "hello",
        "hello_portable",
        "skillful_analyst",
        "skillful_analyst_portable",
        "dep_doctor",
        "dep_doctor_portable",
        "git_detective",
        "git_detective_portable",
    ],
)
@pytest.mark.timeout(60)
async def test_shipped_agents_as_children_use_actual_tools(name, mode, tmp_path, monkeypatch):
    from contextlib import ExitStack

    from looplet import cartridge_to_preset, run_sub_loop, run_sub_loop_async, run_sub_loops_async
    from looplet.testing import AsyncMockLLMBackend, MockLLMBackend

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("LOOPLET_PROJECT_ROOT", str(project))
    family = name.removesuffix("_portable")
    if family == "dep_doctor":
        (project / "requirements.txt").write_text("requests==2.32.3\n")
    elif family == "git_detective":
        from runpy import run_path

        run_path(str(Path(__file__).with_name("test_example_git_detective_portable_cartridge.py")))[
            "_make_repo"
        ](project)

    count = 2 if mode == "parallel" else 1
    requests = []
    presets = []
    with ExitStack() as owned:
        for child_index in range(count):
            target = project / f"report-{child_index}.txt"
            if family == "hello":
                calls = [{"tool": "greet", "args": {"name": f"Person {child_index}"}}]
            elif family == "skillful_analyst":
                calls = [
                    {
                        "tool": "write_text",
                        "args": {"path": str(target), "content": f"report {child_index}\nverified"},
                    },
                    {"tool": "read_text", "args": {"path": str(target)}},
                ]
            elif family == "dep_doctor":
                calls = [
                    {"tool": "parse_deps", "args": {"file_path": str(project / "requirements.txt")}}
                ]
            else:
                calls = [{"tool": "repo_overview", "args": {}}]
            calls.append({"tool": "done", "args": {"summary": f"checked {child_index}"}})
            backend_type = MockLLMBackend if mode == "sync" else AsyncMockLLMBackend
            preset = owned.enter_context(
                cartridge_to_preset(
                    EXAMPLES / f"{name}.cartridge",
                    runtime={"project_root": str(project)},
                )
            )
            presets.append(preset)
            requests.append(
                {
                    "llm": backend_type([json.dumps(call) for call in calls], cycle=False),
                    "sub_tools": preset.tools,
                    "hooks": preset.hooks,
                    "state": preset.state,
                    "config": preset.config,
                    "task": {"goal": f"Check {family} tools"},
                }
            )
        if mode == "parallel":
            results = await run_sub_loops_async(requests, max_concurrency=2)
        elif mode == "async":
            results = [await run_sub_loop_async(**requests[0])]
        else:
            results = [run_sub_loop(**requests[0])]
        for child_index, result in enumerate(results):
            assert result["completed"], result
            assert result["output"]["summary"] == f"checked {child_index}"
            assert all(not step["tool_result"].get("error") for step in result["steps"]), result
            observed = result["steps"][0]["tool_result"]["data"]
            if family == "hello":
                assert observed["greeting"] == f"Hello, Person {child_index}!"
            elif family == "skillful_analyst":
                expected = f"report {child_index}\nverified"
                assert (project / f"report-{child_index}.txt").read_text() == expected
                assert result["steps"][1]["tool_result"]["data"]["content"] == expected
            elif family == "dep_doctor":
                assert observed["count"] == 1
                assert observed["dependencies"][0]["name"] == "requests"
            else:
                assert observed["total_commits"] == 2
            assert requests[child_index]["llm"].calls == len(result["steps"])
    assert all(not preset.mcp_adapters and not preset.state_service_handles for preset in presets)


def test_scripted_demo_executes_four_steps_without_waiting(monkeypatch, capsys):
    from looplet.examples import scripted_demo

    monkeypatch.setattr(scripted_demo.time, "sleep", lambda delay: None)
    assert scripted_demo.main([]) == 0
    assert "1 approval prompt, 4 tools" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "completed", "exit_code"),
    [([], True, 0), (["--stop-after-lookup"], False, 0), (["--max-steps", "1"], False, 1)],
)
def test_quickstart_labels_results_without_implying_an_unchecked_pass(
    arguments, completed, exit_code, monkeypatch, capsys
):
    from looplet.examples import quickstart

    def unexpected_provider(*args, **kwargs):
        raise AssertionError("The default Quickstart must not construct a provider backend")

    monkeypatch.setattr(quickstart.OpenAIBackend, "from_env", unexpected_provider)
    monkeypatch.setattr(quickstart.AnthropicBackend, "from_env", unexpected_provider)
    assert quickstart.main(arguments) == exit_code
    output = capsys.readouterr().out
    assert "Output: " in output
    assert f"Completed: {completed} (" in output
    assert ("Grade: passed" in output) is completed


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


async def test_async_delegation_snippet_runs_real_greetings(capsys):
    snippet = _snippet("04_subagent/async_demo.py")
    results = await snippet.main(["Alice", "Bob", "Charlie"])
    assert [result["steps"][0]["tool_result"]["data"]["greeting"] for result in results] == [
        "Hello, Alice!",
        "Hello, Bob!",
        "Hello, Charlie!",
    ]
    assert all(result["completed"] and result["llm_calls"] == 2 for result in results)
    assert capsys.readouterr().out.splitlines() == [
        "Greeted Alice",
        "Greeted Bob",
        "Greeted Charlie",
    ]


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
