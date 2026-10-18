"""Smoke tests for ``looplet new`` and ``looplet run-cartridge`` CLI.

The CLI's job is straightforward plumbing - load the factory, run a
loop, surface results - so most of these tests exercise UX paths
(missing env vars, bad cartridge path, factory location lookup)
without spinning up a real LLM.

``run-workspace`` remains covered as a compatibility alias.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from looplet.__main__ import main


def test_new_help() -> None:
    """``looplet new --help`` exits 0 and prints usage."""
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["new", "--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        (None, None),
        ("chosen-model", None),
        (None, "http://localhost:11434/v1"),
        ("chosen-model", "http://localhost:11434/v1"),
    ],
)
def test_new_provider_options_share_resolver_without_environment_changes(
    tmp_path, monkeypatch, capsys, model, base_url
):
    from looplet import MockLLMBackend

    monkeypatch.setenv("LOOPLET_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    environment = dict(os.environ)
    received = []

    def make_backend(**kwargs):
        received.append(kwargs)
        return MockLLMBackend([])

    def stop_at_factory():
        raise ValueError("controlled factory fixture boundary")

    monkeypatch.setattr("looplet.backends.make_backend", make_backend)
    monkeypatch.setattr(
        "looplet.backends.OpenAIBackend", lambda **kwargs: make_backend(provider="openai", **kwargs)
    )
    monkeypatch.setattr("looplet.cli.factory_commands._factory_workspace_path", stop_at_factory)
    arguments = ["new", "Read the local input", str(tmp_path / "local.cartridge")]
    if model is not None:
        arguments.extend(["--model", model])
    if base_url is not None:
        arguments.extend(["--base-url", base_url])
    assert main(arguments) == 1
    assert "controlled factory fixture boundary" in capsys.readouterr().err
    assert len(received) == 1
    assert received[0].get("model") == (model or "environment-model" if base_url else model)
    assert received[0].get("base_url") == base_url
    assert received[0].get("provider") == ("openai" if base_url else None)
    assert dict(os.environ) == environment


def test_root_help_discloses_workflow_before_advanced_commands(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    simple = capsys.readouterr().out
    for command in ("new", "run", "inspect", "eval"):
        assert f"  {command:10s}" in simple
    assert "run-workspace" not in simple
    assert "--help-all" in simple
    assert "python your_agent.py" in simple
    assert "Start here: python -m looplet.examples.quickstart" in simple
    assert "Shell: run a Python module:factory" in simple
    assert "eval and replay also accept --factory" in simple
    assert "Skill bundles are advanced packaging" in simple
    with pytest.raises(SystemExit) as stopped:
        main(["--help-all"])
    assert stopped.value.code == 0
    detailed = capsys.readouterr().out
    assert "run-workspace" in detailed
    assert "describe" in detailed
    assert "blueprint" in detailed


def test_run_help_groups_format_specific_options(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["run", "--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "Python factory and cartridge options:" in output
    assert "Skill-bundle-only options:" in output
    assert "--require-completed" in output
    assert "--allow-incomplete" in output
    assert "the default" in " ".join(output.split())


def test_run_cartridge_help() -> None:
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["run-cartridge", "--help"])
    assert exc.value.code == 0


def test_run_workspace_alias_help() -> None:
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(["run-workspace", "--help"])
    assert exc.value.code == 0


def test_show_help_advertises_json() -> None:
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["show", "--help"])
    assert exc.value.code == 0
    assert "--json" in captured.getvalue()


def test_explicit_cartridge_inspection_modes_share_runtime_report(tmp_path, capsys):
    from looplet.cartridge.scaffold import scaffold_cartridge

    root = scaffold_cartridge(tmp_path / "inspect.cartridge", name="inspect", tools=[])
    reports = []
    for command in ("inspect", "describe"):
        assert main([command, str(root), "--mode", "runtime", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["mode"] == "runtime"
        assert report["runtime_validated"] is True
        reports.append(report)
    assert reports[0] == reports[1]
    assert main(["inspect", str(root / "cartridge.json"), "--mode", "declaration", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "declaration-only"
    assert report["runtime_validated"] is False


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_python_factory_declaration_never_imports_code(
    tmp_path, monkeypatch, capsys, explicit_mode
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "safe_inspection.py").write_text("raise AssertionError('must not import')\n")
    arguments = ["inspect", "safe_inspection:build", "--json"]
    if explicit_mode:
        arguments.extend(["--mode", "declaration"])
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["factory"] == "safe_inspection:build"
    assert report["mode"] == "declaration-only"
    assert report["runtime_validated"] is False
    assert "safe_inspection" not in sys.modules


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_python_factory_runtime_inspection_validates_once_and_closes(
    tmp_path, monkeypatch, capsys, valid, json_output
):
    from looplet import bundles

    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "live_inspection", raising=False)
    sentinel = tmp_path / "lifecycle.txt"
    (tmp_path / "live_inspection.py").write_text(
        "from pathlib import Path\n"
        "from looplet import minimal_preset\n"
        "print('authored import output')\n"
        f"sentinel = Path({str(sentinel)!r})\n"
        "class Resource:\n"
        "    def close(self):\n"
        "        print('authored shutdown output')\n"
        "        with sentinel.open('a') as output: output.write('close\\n')\n"
        "def build(runtime):\n"
        "    print('authored build output')\n"
        "    assert runtime.max_steps == 7\n"
        f"    assert runtime.workspace == Path({str(tmp_path)!r})\n"
        "    with sentinel.open('a') as output: output.write('build\\n')\n"
        "    preset = minimal_preset(max_steps=runtime.max_steps)\n"
        "    preset.owned_resources.append(Resource())\n"
        + ("" if valid else "    preset.config.done_tool = 'missing'\n")
        + "    return preset\n"
    )
    validation_calls = []
    original = bundles.validate_preset_contract

    def validate(*arguments, **options):
        validation_calls.append(True)
        return original(*arguments, **options)

    monkeypatch.setattr(bundles, "validate_preset_contract", validate)
    options = ["--json"] if json_output else []
    assert main(
        [
            "inspect",
            "live_inspection:build",
            "--mode",
            "runtime",
            "--workspace",
            str(tmp_path),
            "--max-steps",
            "7",
            *options,
        ]
    ) == (0 if valid else 1)
    captured = capsys.readouterr()
    assert validation_calls == [True]
    assert sentinel.read_text() == "build\nclose\n"
    assert "Traceback" not in captured.err
    if json_output:
        assert "authored import output" in captured.err
        assert "authored build output" in captured.err
        assert "authored shutdown output" in captured.err
    if valid:
        if json_output:
            report = json.loads(captured.out)
            assert report["mode"] == "runtime"
            assert report["runtime_validated"] is True
            assert report["factory"] == "live_inspection:build"
            assert report["source"]["kind"] == "python-factory"
            assert report["config"]["max_steps"] == 7
        else:
            assert "runtime wiring validated" in captured.out
            assert "tools: done" in captured.out
            assert "completion tool: done" in captured.out
            assert "step limit: 7" in captured.out
            assert "tool protocol: native" in captured.out
            assert "hooks: 0" in captured.out
            command = next(
                line.split("full blueprint: ", 1)[1]
                for line in captured.out.splitlines()
                if "full blueprint: " in line
            )
            arguments = shlex.split(command)
            assert arguments[:2] == ["looplet", "inspect"]
            assert arguments[arguments.index("--max-steps") + 1] == "7"
            assert arguments[arguments.index("--workspace") + 1] == str(tmp_path)
            assert "--json" in arguments
    else:
        assert "missing" in captured.err


def test_python_factory_inspection_closes_after_blueprint_failure(tmp_path, monkeypatch, capsys):
    from looplet import minimal_preset

    closed = []

    class Resource:
        def close(self):
            closed.append(True)

    preset = minimal_preset()
    preset.owned_resources.append(Resource())
    monkeypatch.setattr("looplet.cli.factory_commands._build_factory_preset", lambda *args: preset)

    def fail(*arguments, **options):
        raise ValueError("controlled blueprint failure")

    monkeypatch.setattr("looplet.blueprints.blueprint_from_preset", fail)
    assert main(["inspect", "broken:build", "--mode", "runtime", "--json"]) == 1
    assert closed == [True]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "controlled blueprint failure" in captured.err


def test_python_suffixed_trace_directory_remains_inspectable(tmp_path, capsys):
    from looplet.examples import quickstart

    trace = tmp_path / "capture.py"
    assert quickstart.main(["--capture", str(trace)]) == 0
    capsys.readouterr()
    (trace / "experiment.json").write_text("[]\n")
    assert main(["inspect", str(trace), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trajectory"]["termination_reason"] == "done"


@pytest.mark.parametrize("legacy_paths", [False, True])
@pytest.mark.parametrize("move_original", [False, True])
def test_copied_experiment_inspects_its_own_records(tmp_path, capsys, legacy_paths, move_original):
    from looplet import load_eval_run
    from looplet.examples import quickstart

    original = tmp_path / "original experiment"
    assert quickstart.main(["--experiment", "--out", str(original)]) == 0
    capsys.readouterr()
    report_path = original / "experiment.json"
    report = json.loads(report_path.read_text())
    if legacy_paths:
        report["capture_dir"] = str(original / "capture")
        report["replay_dir"] = str(original / "replay")
        report_path.write_text(json.dumps(report))
    copied = tmp_path / "shared experiment"
    shutil.copytree(original, copied)
    if move_original:
        original.rename(tmp_path / "archived original")
    before = {path: path.read_bytes() for path in copied.rglob("*") if path.is_file()}
    for target in (copied, copied / "experiment.json"):
        assert main(["inspect", str(target)]) == 0
        human = capsys.readouterr().out
        assert "missing trace" not in human and str(original) not in human
        for stage in ("capture", "replay"):
            assert "looplet inspect " + shlex.quote(str(copied / stage)) in human
            assert load_eval_run(copied / stage).context.completed
        assert main(["inspect", str(target), "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == report
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("malformed", [False, True])
def test_saved_eval_suite_inspection_preserves_case_grades(
    tmp_path, capsys, monkeypatch, malformed
):
    from looplet import EvalCase
    from looplet.examples import quickstart

    suite = tmp_path / "saved suite"
    for case_id, owner in (("owner", "fintech-platform"), ("wrong", "other-team")):
        case = EvalCase(id=case_id, task={}, expected={"service": "payments", "owner": owner})
        try:
            quickstart.capture_agent(quickstart.scripted_backend(), suite / case_id, case=case)
        except AssertionError:
            assert case_id == "wrong"
    if malformed:
        (suite / "wrong" / "evals.json").write_text("{\n")
    (suite / "unrelated directory").mkdir()
    before = {path: path.read_bytes() for path in suite.rglob("*") if path.is_file()}

    def unexpected_grader(context):
        pytest.fail("Inspection must not execute current graders")

    monkeypatch.setattr(quickstart, "eval_owner", unexpected_grader)
    assert main(["inspect", str(suite), "--json"]) == int(malformed)
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "eval_suite" and report["grading_source"] == "saved"
    assert [case["id"] for case in report["cases"]] == ["owner", "wrong"]
    assert report["cases"][0]["results"][0]["score"] == 1.0
    assert report["cases"][0]["trace_dir"] == "owner"
    if malformed:
        assert "Invalid evals.json" in report["cases"][1]["error"]
    else:
        assert report["cases"][1]["completed"] is True
        assert report["cases"][1]["results"][0]["score"] == 0.0
    assert main(["inspect", str(suite)]) == int(malformed)
    human = capsys.readouterr().out
    assert "Capture-time grades" in human
    for name in ("owner", "wrong"):
        assert "looplet inspect " + shlex.quote(str(suite / name)) in human
    assert {path: path.read_bytes() for path in suite.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("passed", [False, True])
@pytest.mark.parametrize("report_file", [False, True])
def test_legacy_experiment_inspection_is_read_only(tmp_path, capsys, passed, report_file):
    capture = tmp_path / "capture with spaces"
    capture.mkdir()
    report = {
        "capture_dir": "capture with spaces",
        "replay_dir": "missing replay",
        "passed": passed,
    }
    report_path = tmp_path / "experiment.json"
    report_path.write_text(json.dumps(report))
    target = report_path if report_file else tmp_path
    assert main(["inspect", str(target)]) == 0
    human = capsys.readouterr().out
    assert f"capture: {'passed' if passed else 'unknown'}" in human
    assert "looplet inspect " + shlex.quote(str(capture)) in human
    assert f"missing trace: {tmp_path / 'missing replay'}" in human
    assert main(["inspect", str(target), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert json.loads(report_path.read_text()) == report
    assert list(capture.iterdir()) == []
    assert not (tmp_path / "missing replay").exists()


@pytest.mark.parametrize("passed", [False, True])
@pytest.mark.parametrize("name", ["strict assessment.json", "experiment.json"])
def test_saved_assessment_inspection_is_portable_and_read_only(tmp_path, capsys, passed, name):
    from looplet import EvalContext, eval_assess

    def eval_current(context):
        return passed

    original = tmp_path / "original"
    capture = original / "capture"
    capture.mkdir(parents=True)
    report = eval_assess([eval_current], EvalContext(steps=[]))
    report.save(original / name, evidence={"original run": capture})
    copied = tmp_path / "shared assessment"
    shutil.copytree(original, copied)
    original.rename(tmp_path / "archived original")
    target = copied / name
    before = {path: path.read_bytes() for path in copied.rglob("*") if path.is_file()}
    assert main(["inspect", str(target)]) == 0
    human = capsys.readouterr().out
    assert "Saved assessment:" in human
    assert "passed" in human if passed else "failed" in human
    assert "Separate from capture-time grades" in human
    assert "looplet inspect " + shlex.quote(str(copied / "capture")) in human
    assert "missing trace" not in human
    assert main(["inspect", str(target), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(target.read_text())
    (copied / "capture").rmdir()
    assert main(["inspect", str(target)]) == 0
    assert "missing trace" in capsys.readouterr().out
    assert {path: path.read_bytes() for path in copied.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    ("score", "label"),
    [(0.6, None), (0.0, "measured"), (0.7, "accepted"), (1.0, "pass"), (None, "skipped")],
)
def test_saved_assessment_inspection_shows_scores_and_labels(
    tmp_path, capsys, version, score, label
):
    from looplet import EvalContext, EvalResult, eval_assess

    def eval_current(context):
        return EvalResult(
            name="eval_current", score=score, label=label, explanation="grader detail"
        )

    report = eval_assess([eval_current], EvalContext(steps=[]), threshold=0.7, version=version)
    target = tmp_path / "numeric assessment.json"
    report.save(target)
    before = target.read_bytes()
    saved = json.loads(before)
    result = saved["assessment"]["cases"][0]["results"][0]
    assert main(["inspect", str(target)]) == 0
    human = capsys.readouterr().out
    assert "threshold: 0.7" in human
    assert f"eval_current: {result['state']}" in human
    assert "grader detail" in human
    if score is None:
        assert "score=" not in human
    else:
        assert f"score={score}" in human
    if label:
        assert f"label={label}" in human
    assert main(["inspect", str(target), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == saved
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    ["json", "version", "summary", "threshold", "cases", "results", "evidence"],
)
def test_saved_assessment_inspection_rejects_malformed_reports(tmp_path, capsys, damage):
    from looplet import EvalContext, eval_assess

    report_path = tmp_path / "assessment.json"
    eval_assess([lambda context: True], EvalContext(steps=[])).save(report_path)
    report = json.loads(report_path.read_text())
    if damage == "version":
        report["version"] = True
    elif damage == "summary":
        report["assessment"] = []
    elif damage == "threshold":
        report["assessment"]["threshold"] = float("nan")
    elif damage == "cases":
        report["assessment"]["cases"] = [None]
    elif damage == "results":
        report["assessment"]["cases"][0]["results"] = [None]
    elif damage == "evidence":
        report["evidence"] = {"run": []}
    report_path.write_text("{\n" if damage == "json" else json.dumps(report))
    before = report_path.read_bytes()
    assert main(["inspect", str(report_path), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "invalid saved assessment" in captured.err
    assert "Traceback" not in captured.err
    assert report_path.read_bytes() == before


@pytest.mark.parametrize(
    "report",
    [
        [],
        {},
        {"passed": "false", "capture_dir": None, "replay_dir": None},
        {"passed": False, "capture_dir": [], "replay_dir": None},
        {"passed": False, "capture_dir": None},
        {"passed": False, "capture_dir": None, "replay_dir": None, "stages": []},
        {
            "passed": False,
            "capture_dir": None,
            "replay_dir": None,
            "stages": {"capture": "wrong", "rescore": "not_run", "replay": "not_run"},
        },
    ],
)
def test_malformed_experiment_inspection_has_concise_error(tmp_path, capsys, report):
    report_path = tmp_path / "experiment.json"
    report_path.write_text(json.dumps(report))
    assert main(["inspect", str(report_path), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid experiment report" in captured.err
    assert "Traceback" not in captured.err


def test_python_file_inspection_explains_factory_reference(tmp_path, capsys):
    agent = tmp_path / "my_agent.py"
    agent.write_text("raise AssertionError('must not import')\n")
    assert main(["inspect", str(agent), "--mode", "runtime"]) == 1
    assert "looplet inspect my_agent:build --mode runtime" in capsys.readouterr().err


def test_explicit_bundle_declaration_modes_never_import_entrypoint(tmp_path, capsys):
    (tmp_path / "SKILL.md").write_text("---\nname: safe\ndescription: Safe fixture.\n---\n")
    (tmp_path / "looplet.py").write_text("raise AssertionError('must not import')\n")
    for command in ("inspect", "blueprint"):
        arguments = [command, str(tmp_path), "--mode", "declaration"]
        if command == "inspect":
            arguments.append("--json")
        assert main(arguments) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["mode"] == "declaration-only"
        assert report["runtime_validated"] is False


@pytest.mark.parametrize("mode", ["declaration", "runtime"])
def test_inspection_distinguishes_unfinished_tools_from_valid_wiring(tmp_path, capsys, mode):
    from looplet.cartridge.scaffold import scaffold_cartridge

    root = scaffold_cartridge(tmp_path / "draft.cartridge", name="draft", tools=["lookup"])
    implementation = root / "tools" / "lookup" / "execute.py"
    assert main(["inspect", str(root), "--mode", mode, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["runtime_validated"] is (mode == "runtime")
    assert report["readiness"] == "draft"
    assert report["unfinished_tools"] == [str(implementation)]
    assert any("Unimplemented scaffold tool" in warning for warning in report["warnings"])

    assert main(["inspect", str(root), "--mode", mode]) == 0
    assert str(implementation) in capsys.readouterr().err

    implementation.write_text(
        '"""Historical scaffold: implement lookup marker, not an unfinished body."""\n'
        'def execute(ctx, **kwargs):\n    return {"owner": "platform"}\n'
    )
    assert main(["inspect", str(root), "--mode", mode, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["readiness"] == "not_assessed"
    assert report["unfinished_tools"] == []
    assert report["warnings"] == []


@pytest.mark.parametrize("command", ["run", "run-cartridge", "run-workspace"])
@pytest.mark.parametrize("json_output", [False, True])
def test_existing_trace_fails_cleanly_before_loading_agent(
    tmp_path, monkeypatch, capsys, command, json_output
):
    from looplet.cartridge.scaffold import scaffold_cartridge

    cartridge = scaffold_cartridge(tmp_path / "agent.cartridge", name="agent", tools=[])
    trace = tmp_path / "trace"
    trace.mkdir()
    original = b'{"run_id":"previous-run"}\n'
    (trace / "trajectory.json").write_bytes(original)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("must reject existing trace before loading an agent or provider")

    monkeypatch.setattr("looplet.cartridge.cartridge_to_preset", unexpected_load)
    monkeypatch.setattr("looplet.cli.factory_commands._build_backend", unexpected_load)
    options = ["--json"] if json_output else []
    assert (
        main(
            [
                command,
                str(cartridge),
                "Finish",
                "--project-root",
                str(tmp_path),
                "--trace-dir",
                str(trace),
                *options,
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert not captured.out
    assert "Choose a new --trace-dir or omit it" in captured.err
    assert "Traceback" not in captured.err
    assert "overwrite=True" not in captured.err
    assert (trace / "trajectory.json").read_bytes() == original


def test_runtime_cartridge_json_redirects_authored_shutdown_output(tmp_path, capsys, monkeypatch):
    from looplet import minimal_preset
    from looplet.cartridge.scaffold import scaffold_cartridge

    root = scaffold_cartridge(tmp_path / "runtime.cartridge", name="runtime", tools=[])

    class Resource:
        def close(self):
            print("authored shutdown output")

    def load(*args, **kwargs):
        print("authored import output")
        preset = minimal_preset()
        preset.owned_resources.append(Resource())
        return preset

    monkeypatch.setattr("looplet.cartridge.cartridge_to_preset", load)
    assert main(["inspect", str(root), "--mode", "runtime", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["runtime_validated"] is True
    assert "authored import output" in captured.err
    assert "authored shutdown output" in captured.err


def test_conflicting_inspection_modes_are_rejected(tmp_path):
    for command in ("describe", "blueprint"):
        with pytest.raises(SystemExit) as stopped:
            main([command, str(tmp_path), "--mode", "declaration", "--instantiate"])
        assert stopped.value.code == 2


@pytest.mark.parametrize("valid", [False, True])
def test_runtime_bundle_inspection_validates_once_and_closes(tmp_path, capsys, valid):
    sentinel = tmp_path / "lifecycle.txt"
    (tmp_path / "SKILL.md").write_text("---\nname: live\ndescription: Live fixture.\n---\n")
    (tmp_path / "looplet.py").write_text(
        "from pathlib import Path\n"
        "from looplet import minimal_preset\n"
        f"sentinel = Path({str(sentinel)!r})\n"
        "class Resource:\n"
        "    def close(self):\n"
        "        with sentinel.open('a') as output: output.write('close\\n')\n"
        "def build(runtime):\n"
        "    with sentinel.open('a') as output: output.write('build\\n')\n"
        "    preset = minimal_preset()\n"
        "    preset.owned_resources.append(Resource())\n"
        + ("" if valid else "    preset.config.done_tool = 'missing'\n")
        + "    return preset\n"
    )
    assert main(["inspect", str(tmp_path), "--mode", "runtime", "--json"]) == (0 if valid else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "runtime"
    assert report["runtime_validated"] is valid
    assert bool(report["metadata"]["errors"]) is not valid
    assert sentinel.read_text() == "build\nclose\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "--help"],
        ["doctor", "--help"],
        ["describe", "--help"],
        ["diff", "--help"],
        ["hash", "--help"],
        ["portability", "--help"],
        ["eval", "--help"],
        ["eval", "run", "--help"],
    ],
    ids=lambda argv: "-".join(argv[:-1]),
)
def test_documented_command_help(argv: list[str]) -> None:
    """Every command copied into launch docs must remain parseable."""
    with pytest.raises(SystemExit) as exc, patch("sys.stdout", new=io.StringIO()):
        main(argv)
    assert exc.value.code == 0


def test_new_defaults_to_cartridge_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public default is a cartridge, not the legacy workspace suffix."""
    from looplet.cli import factory_commands

    seen: dict[str, Path] = {}

    def fake_cmd_new(args) -> int:
        seen["target"] = args.target
        return 0

    monkeypatch.setattr(factory_commands, "cmd_new", fake_cmd_new)

    assert main(["new", "a brief"]) == 0
    assert seen["target"] == Path("agent.cartridge")


def test_new_missing_env_vars(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When required env vars are unset, ``new`` prints a clear error
    and exits 1."""
    captured = io.StringIO()
    with patch.dict(os.environ, {}, clear=True), patch.object(sys, "stderr", captured):
        rc = main(["new", "a brief", str(tmp_path / "out.cartridge")])
    assert rc == 1
    err = captured.getvalue()
    assert "could not resolve a provider" in err
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert var in err


@pytest.mark.parametrize(
    "environment",
    [
        {"OPENAI_BASE_URL": "http://localhost:1234/v1"},
        {"OPENAI_API_KEY": "test"},
        {"ANTHROPIC_API_KEY": "test"},
    ],
)
def test_cli_accepts_provider_defaults(environment: dict[str, str]) -> None:
    from looplet.cli import factory_commands

    with patch.dict(os.environ, environment, clear=True):
        assert factory_commands._check_env() == 0


def test_new_offline_scaffolds_without_backend(tmp_path, monkeypatch, capsys) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "offline.cartridge"
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: pytest.fail("offline scaffolding must not construct a backend"),
    )
    assert (
        main(["new", "Look up service owners", str(target), "--offline", "--tool", "lookup"]) == 0
    )
    assert json.loads((target / "cartridge.json").read_text())["schema_version"] == 2
    assert "Look up service owners" in (target / "prompts" / "system.md").read_text()
    assert (target / "tools" / "lookup" / "execute.py").is_file()
    assert (target / "tools" / "done" / "execute.py").is_file()
    output = capsys.readouterr().out
    assert "scaffolded cartridge draft" in output
    assert str(target / "prompts" / "system.md") in output
    assert str(target / "tools" / "lookup") in output
    assert "--mode runtime" in output
    assert "do not prove behavior" in output


def test_new_offline_preserves_existing_files(tmp_path, capsys) -> None:
    target = tmp_path / "existing.cartridge"
    target.mkdir()
    original = target / "original.txt"
    original.write_text("keep this")

    assert main(["new", "draft", str(target), "--offline"]) == 1
    assert original.read_text() == "keep this"
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("budget", ["0", "1", "-1"])
def test_new_offline_rejects_ignored_factory_budget(tmp_path, capsys, budget):
    target = tmp_path / "agent.cartridge"
    assert main(["new", "draft", str(target), "--offline", "--max-steps", budget]) == 1
    assert not target.exists()
    message = capsys.readouterr().err
    assert "--max-steps limits the live factory" in message
    assert "config.yaml" in message


def test_inspect_cartridge_does_not_execute_tool_code(tmp_path, capsys) -> None:
    from looplet import scaffold_cartridge

    target = scaffold_cartridge(tmp_path / "safe.cartridge", name="safe", tools=[])
    (target / "tools" / "done" / "execute.py").write_text("raise AssertionError('must not import')")

    assert main(["inspect", str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tools"][0]["name"] == "done"


def test_inspect_bundle_does_not_import_entrypoint(tmp_path, capsys) -> None:
    target = tmp_path / "safe-bundle"
    target.mkdir()
    (target / "SKILL.md").write_text(
        "---\nname: safe\ndescription: Safe metadata.\nentrypoint: looplet.py\n---\n"
    )
    (target / "looplet.py").write_text("raise AssertionError('must not import')")

    assert main(["inspect", str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "safe"
    assert payload["ok"] is True


@pytest.mark.parametrize("explicit_file", [False, True])
def test_inspect_parent_bundle_does_not_substitute_nested_child(
    tmp_path, capsys, explicit_file
) -> None:
    from looplet import discover_skill_bundles

    parent = tmp_path / "parent"
    child = parent / "child"
    for root, name in [(parent, "zzz-parent"), (child, "aaa-child")]:
        root.mkdir(parents=True)
        (root / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name}.\nentrypoint: looplet.py\n---\n"
        )
        (root / "looplet.py").write_text("raise AssertionError('must not import')")

    path = parent / "SKILL.md" if explicit_file else parent
    assert main(["inspect", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "zzz-parent"
    assert payload["path"] == str(parent.resolve())
    assert [card.name for card in discover_skill_bundles(parent)] == [
        "aaa-child",
        "zzz-parent",
    ]


def test_run_cartridge_missing_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    rc = main(["run-cartridge", str(tmp_path), "do something"])
    assert rc == 1


def test_run_cartridge_path_must_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    captured = io.StringIO()
    nonexistent = tmp_path / "no_such_dir"
    with patch.object(sys, "stderr", captured):
        rc = main(["run-cartridge", str(nonexistent), "do x"])
    assert rc == 1
    assert "cartridge not found" in captured.getvalue()


def test_run_cartridge_load_failure_uses_cartridge_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import looplet
    from looplet.cli import factory_commands

    for var, value in {
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
        "OPENAI_API_KEY": "test",
        "OPENAI_MODEL": "test-model",
    }.items():
        monkeypatch.setenv(var, value)
    cartridge = tmp_path / "broken.cartridge"
    cartridge.mkdir()
    monkeypatch.setattr(factory_commands, "_build_backend", object)

    def fail_load(*_args, **_kwargs):
        raise ValueError("broken manifest")

    monkeypatch.setattr(looplet, "cartridge_to_preset", fail_load)
    captured = io.StringIO()
    with patch.object(sys, "stderr", captured):
        rc = main(["run-cartridge", str(cartridge), "do x"])

    assert rc == 1
    assert "cartridge load failed: broken manifest" in captured.getvalue()


def test_factory_workspace_path_resolves_in_repo() -> None:
    """``_factory_workspace_path`` finds the bundled
    examples/agent_factory.cartridge when run from the repo."""
    from looplet.cli.factory_commands import _factory_workspace_path

    p = _factory_workspace_path()
    assert p.is_dir()
    assert (p / "cartridge.json").is_file()
    assert (p / "config.yaml").is_file()


def test_factory_workspace_path_resolves_installed_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An installed wheel resolves package data without a repo checkout."""
    from looplet import bundled
    from looplet.cli import factory_commands

    package_root = tmp_path / "site-packages" / "looplet"
    fake_module = package_root / "bundled.py"
    factory = package_root / "_bundled" / "agent_factory.cartridge"
    fake_module.parent.mkdir(parents=True)
    factory.mkdir(parents=True)
    (factory / "cartridge.json").write_text('{"schema_version": 2, "name": "factory"}')

    monkeypatch.delenv("LOOPLET_FACTORY_DIR", raising=False)
    monkeypatch.setattr(bundled, "__file__", str(fake_module))

    assert factory_commands._factory_workspace_path() == factory


def test_new_command_registered_on_top_level() -> None:
    """Detailed help advertises the canonical command and compatibility alias."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["--help-all"])
    assert exc.value.code == 0
    out = captured.getvalue()
    assert "new" in out
    assert "run-cartridge" in out
    assert "run-workspace" in out


def _new_args(target: Path) -> SimpleNamespace:
    return SimpleNamespace(
        description="summarize a URL",
        target=target,
        name=None,
        tool=None,
        max_steps=None,
        quiet=True,
        pretty=False,
    )


def _patch_new_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, object]:
    import looplet
    from looplet import AgentPreset, DefaultState, LoopConfig, MockLLMBackend, tools_from
    from looplet.cli import factory_commands
    from looplet.loop import composable_loop

    closed: list[str] = []
    resource = SimpleNamespace(close=lambda: closed.append("factory"))
    config = LoopConfig(max_steps=3, system_prompt="Reviewable prompt")
    factory_preset = AgentPreset(
        config=config,
        tools=tools_from([], include_done=True),
        hooks=[],
        state=DefaultState(max_steps=3),
        resources={"client": resource},
        owned_resources=[resource],
    )
    produced_preset = SimpleNamespace(
        config=config,
        tools=SimpleNamespace(_tools={"done": object(), "fetch_url": object()}),
        close=lambda: closed.append("produced"),
    )
    observed: dict[str, object] = {"closed": closed}

    def fake_load(_path: str, runtime=None):
        return factory_preset if runtime is not None else produced_preset

    def fake_loop(**kwargs):
        observed.update(kwargs)
        return composable_loop(
            **{**kwargs, "llm": MockLLMBackend(['{"tool":"done","args":{"summary":"draft"}}'])}
        )

    monkeypatch.setenv("OPENAI_MODEL", "mock-model")
    monkeypatch.setattr(factory_commands, "_check_env", lambda: 0)
    monkeypatch.setattr(factory_commands, "_build_backend", lambda: MockLLMBackend([]))
    monkeypatch.setattr(factory_commands, "_factory_workspace_path", lambda: tmp_path)
    monkeypatch.setattr(looplet, "cartridge_to_preset", fake_load)
    monkeypatch.setattr("looplet.loop.composable_loop", fake_loop)
    return observed


@pytest.mark.parametrize("target_name", ["url_summary.cartridge", "service owners.cartridge"])
def test_new_success_labels_generated_code_as_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target_name: str,
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / target_name
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 0

    out = capsys.readouterr().out
    assert "draft built" in out
    assert "produced cartridge draft:" in out
    command = next(
        line.strip() for line in out.splitlines() if line.strip().startswith("looplet run ")
    )
    assert shlex.split(command) == ["looplet", "run", str(target), "<your task>"]
    task = observed["task"]
    assert isinstance(task, dict)
    assert "Scaffold a cartridge draft" in task["goal"]


@pytest.mark.parametrize("max_steps", [1, 5])
def test_new_max_steps_updates_config_and_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_steps: int,
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "url_summary.cartridge"
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)
    args = _new_args(target)
    args.max_steps = max_steps

    assert factory_commands.cmd_new(args) == 0
    assert observed["config"].max_steps == max_steps
    assert observed["state"].max_steps == max_steps


def test_new_closes_factory_and_produced_presets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "url_summary.cartridge"
    target.mkdir()
    observed = _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 0
    assert observed["closed"] == ["factory", "produced"]


@pytest.mark.parametrize("fresh_target", [False, True])
def test_new_max_steps_limits_actual_factory_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fresh_target: bool,
) -> None:
    import looplet
    from looplet import MockLLMBackend, tool
    from looplet.cli import factory_commands
    from looplet.loop import composable_loop

    target = tmp_path / "url_summary.cartridge"
    if not fresh_target:
        target.mkdir()
        (target / "keep.txt").write_text("existing draft")
    observed = _patch_new_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr("looplet.loop.composable_loop", composable_loop)
    preset = looplet.cartridge_to_preset(str(tmp_path), runtime={})
    calls: list[bool] = []

    @tool
    def work() -> dict:
        calls.append(True)
        return {"ok": True}

    preset.tools.register(work)
    monkeypatch.setattr(
        factory_commands,
        "_build_backend",
        lambda: MockLLMBackend(['{"tool":"work","args":{}}'] * 3),
    )
    args = _new_args(target)
    args.max_steps = 1
    if fresh_target:
        args.tool = ["work"]

    assert factory_commands.cmd_new(args) == 1
    assert len(calls) == 1
    assert preset.state.step_count == 1
    assert observed["closed"] == ["factory"]
    captured = capsys.readouterr()
    assert "factory stopped (budget_exhausted)" in captured.err
    assert f"partial cartridge draft retained at {target}" in captured.err
    assert "draft built" not in captured.out
    if fresh_target:
        assert (target / "tools" / "work" / "execute.py").is_file()
    else:
        assert (target / "keep.txt").read_text() == "existing draft"


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_new_closes_factory_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[BaseException],
) -> None:
    from looplet.cli import factory_commands

    observed = _patch_new_runtime(monkeypatch, tmp_path)

    def fail_loop(**kwargs):
        raise failure("factory interrupted")

    monkeypatch.setattr("looplet.loop.composable_loop", fail_loop)

    assert factory_commands.cmd_new(_new_args(tmp_path / "draft.cartridge")) == (
        130 if failure is KeyboardInterrupt else 1
    )
    assert observed["closed"] == ["factory"]


def test_new_missing_output_uses_cartridge_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from looplet.cli import factory_commands

    target = tmp_path / "missing.cartridge"
    _patch_new_runtime(monkeypatch, tmp_path)

    assert factory_commands.cmd_new(_new_args(target)) == 1

    captured = capsys.readouterr()
    assert "draft built" not in captured.out
    assert f"cartridge not created at {target}" in captured.err


def test_new_invalid_output_uses_cartridge_language(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import looplet
    from looplet.cli import factory_commands

    target = tmp_path / "broken.cartridge"
    target.mkdir()
    _patch_new_runtime(monkeypatch, tmp_path)
    load_factory = looplet.cartridge_to_preset

    def fail_produced_load(path: str, runtime=None):
        if runtime is not None:
            return load_factory(path, runtime=runtime)
        raise ValueError("corrupted manifest")

    monkeypatch.setattr(looplet, "cartridge_to_preset", fail_produced_load)

    assert factory_commands.cmd_new(_new_args(target)) == 1

    captured = capsys.readouterr()
    assert "draft built" not in captured.out
    assert "produced cartridge failed to load: corrupted manifest" in captured.err


# ── --pretty (stdlib-only renderer) ────────────────────────────


def test_pretty_printer_renders_steps_without_ansi_when_not_tty() -> None:
    """``PrettyPrinter`` should emit plain text (no escape sequences)
    when stdout isn't a TTY - keeps CI logs and ``tee`` clean."""
    import io as _io
    from types import SimpleNamespace as NS

    # Patch _COLOR off and capture stdout.
    from looplet.cli import _pretty as p_mod
    from looplet.cli._pretty import PrettyPrinter

    original = p_mod._COLOR
    p_mod._COLOR = False
    captured = _io.StringIO()
    try:
        with patch.object(sys, "stdout", captured):
            printer = PrettyPrinter(title="testing")
            printer.header(["  task: smoke", "  model: mock"])
            printer.step(
                NS(
                    tool_call=NS(
                        tool="hello",
                        args={"who": "world"},
                        reasoning="say hi",
                    ),
                    tool_result=NS(error=None, data={"ok": True}, duration_ms=4.0),
                )
            )
            printer.step(
                NS(
                    tool_call=NS(
                        tool="bash",
                        args={"command": "ls"},
                        reasoning="list things",
                    ),
                    tool_result=NS(error="permission denied", data=None, duration_ms=1.0),
                )
            )
            printer.finish(summary="finished")
    finally:
        p_mod._COLOR = original

    out = captured.getvalue()
    # No raw ANSI escape sequences leak into the output.
    assert "\033[" not in out, f"ansi leaked: {out!r}"
    # Header content present.
    assert "testing" in out
    assert "task: smoke" in out
    # Per-step content present.
    assert "step  1" in out
    assert "hello" in out
    assert "why" in out
    assert "say hi" in out
    assert "💭" not in out
    assert "step  2" in out
    assert "permission denied" in out
    # Finish line present with stats.
    assert "done" in out
    assert "2 steps" in out
    assert "1 errors" in out
    assert "agent says: finished" in out


def test_new_help_advertises_pretty() -> None:
    """``looplet new --help`` should mention the --pretty flag."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["new", "--help"])
    assert exc.value.code == 0
    assert "--pretty" in captured.getvalue()


def test_run_cartridge_help_advertises_runtime_options() -> None:
    """``looplet run-cartridge --help`` should name its runtime controls."""
    captured = io.StringIO()
    with pytest.raises(SystemExit) as exc, patch.object(sys, "stdout", captured):
        main(["run-cartridge", "--help"])
    assert exc.value.code == 0
    out = captured.getvalue()
    assert "--pretty" in out
    assert "--trace-dir" in out
    assert "--parent-trace" in out
    assert "--no-trace" in out
    assert "--json" in out
    assert "read it from stdin" in out
