"""End-to-end: run a cartridge against its shipped evals (the CLI core).

``run_cartridge_evals`` ties the whole "evals ship with the agent
version" story together - seed each case's sandbox, run the cartridge as
a live agent with an online :class:`EvalHook`, grade, and optionally
persist. Driven here with a :class:`MockLLMBackend` and a minimal
in-test cartridge so it is fully deterministic (no network). Also covers
the ``looplet eval run`` CLI preflight/error paths.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from looplet import (
    EvalReport,
    EvalRunRecord,
    assert_evals_pass,
    eval_assess,
    eval_cli,
    eval_discover,
    eval_run,
    load_eval_run,
    load_evals,
    run_cartridge_evals,
)
from looplet.testing import MockLLMBackend


@pytest.fixture(autouse=True)
def _isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ── A minimal but real cartridge built on the fly ────────────────

_SYSTEM = "Implement greeting.py so test_greeting.py passes, then call done."

_DONE_TOOL_YAML = """\
name: done
description: Signal completion with a one-line summary.
parameters:
  summary:
    type: string
    description: One-line summary.
"""
_DONE_EXEC = "def execute(*, summary: str) -> dict:\n    return {'status': 'completed', 'summary': summary}\n"

_WRITE_TOOL_YAML = """\
name: write_file
description: Write text to a file in the project root.
parameters:
  path:
    type: string
    description: Relative path.
  content:
    type: string
    description: File contents.
requires:
  - project_dir
"""
_WRITE_EXEC = """\
from looplet.types import ToolContext


def execute(ctx: ToolContext, *, path: str, content: str) -> dict:
    from pathlib import Path

    root = Path(ctx.resources.get("project_dir") or ".")
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"ok": True, "bytes": len(content)}
"""

_PROJECT_DIR_RES = """\
def build(runtime=None):
    return (runtime or {}).get("project_root", ".")
"""

_GRADERS = """\
from looplet import eval_mark


@eval_mark("smoke")
def eval_completed(ctx):
    return ctx.completed


def eval_wrote_file(ctx):
    return "write_file" in ctx.tool_sequence


def eval_expected_available(ctx):
    return ctx.task.get("expected") == {"file_written": True}


def eval_judge_quality(ctx, llm):
    # LLM-as-judge grader: only runs when a judge backend is supplied.
    raw = llm.generate("rate 0..1").strip()
    import re
    m = re.search(r"[01](?:\\.\\d+)?", raw)
    return float(m.group(0)) if m else 0.0
"""

_COLLECTOR = """\
def collect_marker(state, runtime):
    from pathlib import Path

    root = Path((runtime or {}).get("project_root", "."))
    return {
        "file_written": (root / "greeting.py").exists(),
        "expected_leaked": "expected" in state.task,
    }
"""

_CASE = {
    "id": "make_greeting",
    "task": {
        "goal": "Write greeting.py",
        "files": {
            "test_greeting.py": "from greeting import hi\n\n\ndef test_hi():\n    assert hi() == 'hi'\n"
        },
    },
    "expected": {"file_written": True},
    "marks": ["regression"],
}


def _make_cartridge(root: Path, *, with_evals: bool = True) -> Path:
    cart = root / "greeter.cartridge"
    (cart).mkdir(parents=True, exist_ok=True)
    (cart / "cartridge.json").write_text(
        json.dumps({"name": "greeter", "schema_version": 2, "description": "test"})
    )
    (cart / "config.yaml").write_text("max_steps: 6\ndone_tool: done\n")
    (cart / "runtime.yaml").write_text("use_native_tools: false\n")
    (cart / "prompts").mkdir(exist_ok=True)
    (cart / "prompts" / "system.md").write_text(_SYSTEM)
    # tools
    for name, ty, ex in (
        ("done", _DONE_TOOL_YAML, _DONE_EXEC),
        ("write_file", _WRITE_TOOL_YAML, _WRITE_EXEC),
    ):
        d = cart / "tools" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "tool.yaml").write_text(ty)
        (d / "execute.py").write_text(ex)
    # resources
    (cart / "resources").mkdir(exist_ok=True)
    (cart / "resources" / "project_dir.py").write_text(_PROJECT_DIR_RES)
    if not with_evals:
        return cart
    # evals slot
    (cart / "evals" / "cases").mkdir(parents=True, exist_ok=True)
    (cart / "evals" / "cases" / "make_greeting.json").write_text(json.dumps(_CASE))
    (cart / "evals" / "eval_correctness.py").write_text(_GRADERS)
    (cart / "evals" / "collect_outcome.py").write_text(_COLLECTOR)
    return cart


@pytest.mark.parametrize("path_kind", ["directory", "file"])
def test_standalone_python_eval_has_actionable_run_guidance(
    tmp_path, monkeypatch, capsys, path_kind
):
    definitions = tmp_path / "evals"
    assert eval_cli(["init", str(definitions)]) == 0
    assert eval_cli(["check", str(definitions), "--json"]) == 0
    capsys.readouterr()
    grader = definitions / "eval_outcome.py"
    grader.write_text("raise AssertionError('must not import')\n")
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: pytest.fail("must not build a provider")
    )
    path = definitions if path_kind == "directory" else grader
    assert eval_cli(["run", str(path)]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "standalone Python eval directory" in output.err
    assert "pytest" in output.err and "load_evals()" in output.err
    assert "eval check" in output.err
    assert "evals/evals/cases" not in output.err


def test_generated_definitions_work_with_existing_cartridge_runner(tmp_path):
    cartridge = _make_cartridge(tmp_path, with_evals=False)
    assert eval_cli(["init", str(cartridge / "evals")]) == 0
    assert eval_cli(["check", str(cartridge / "evals")]) == 0
    definitions = load_evals(cartridge / "evals")
    records = run_cartridge_evals(
        cartridge,
        llm=MockLLMBackend(
            [
                json.dumps(
                    {"tool": "write_file", "args": {"path": "result.txt", "content": "hello"}}
                ),
                json.dumps({"tool": "done", "args": {"summary": "written"}}),
            ]
        ),
        output_dir=tmp_path / "runs",
        capture=False,
    )
    assert records[0].case == definitions.cases[0]
    assert records[0].context.expected == definitions.cases[0].expected
    assert (
        "expected" not in json.loads((records[0].directory / "trajectory.json").read_text())["task"]
    )
    EvalReport.from_records(records).assert_passed()


def _scripted() -> list[str]:
    return [
        json.dumps(
            {
                "tool": "write_file",
                "args": {"path": "greeting.py", "content": "def hi():\n    return 'hi'\n"},
                "reasoning": "create module",
            }
        ),
        json.dumps({"tool": "done", "args": {"summary": "wrote greeting.py"}, "reasoning": "done"}),
    ]


def _add_embedded_eval_hook(cartridge: Path) -> None:
    hook_dir = cartridge / "hooks" / "09_EmbeddedEvaluation"
    hook_dir.mkdir(parents=True)
    (hook_dir / "config.yaml").write_text(
        "class_name: EmbeddedEvaluation\nkwargs:\n  project_root: '@project_dir'\n"
    )
    (hook_dir / "hook.py").write_text(
        "from pathlib import Path\n"
        "from looplet import EvalHook\n\n"
        "class EmbeddedEvaluation(EvalHook):\n"
        "    def __init__(self, project_root):\n"
        "        def collect_existing(state):\n"
        "            marker = Path(project_root) / 'embedded-collector-count.txt'\n"
        "            count = int(marker.read_text()) if marker.exists() else 0\n"
        "            marker.write_text(str(count + 1))\n"
        "            return {'embedded_count': count + 1}\n"
        "        super().__init__([lambda ctx: True], collectors=[collect_existing])\n"
    )


# ── run_cartridge_evals (the CLI core) ───────────────────────────


def test_run_cartridge_evals_end_to_end(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    out = tmp_path / "runs"

    records = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        output_dir=out,
    )

    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, EvalRunRecord)
    assert rec.case.id == "make_greeting"
    # online graders scored the live run
    scores = {r.name: r.score for r in rec.results}
    assert scores["eval_completed"] == 1.0
    assert scores["eval_wrote_file"] == 1.0
    assert scores["eval_expected_available"] == 1.0
    # collector populated artifacts from the seeded+written sandbox
    assert rec.context.artifacts.get("file_written") is True
    assert rec.context.artifacts.get("expected_leaked") is False

    # persisted as an offline fixture, reloadable + re-gradable
    persisted = out / "make_greeting"
    assert (persisted / "trajectory.json").is_file()
    assert (persisted / "artifacts.json").is_file()
    assert (persisted / "manifest.jsonl").is_file()
    model_calls = [
        json.loads(line) for line in (persisted / "manifest.jsonl").read_text().split("\n") if line
    ]
    assert len(model_calls) == 2
    trajectory = json.loads((persisted / "trajectory.json").read_text())
    assert "expected" not in trajectory["task"]
    assert json.loads((persisted / "expected.json").read_text()) == {"file_written": True}
    reloaded = load_eval_run(persisted)
    assert reloaded.context.tool_sequence == ["write_file", "done"]
    assert reloaded.context.task["expected"] == {"file_written": True}


@pytest.mark.parametrize("capture", [False, True])
def test_repeated_eval_preserves_evidence_before_model_calls(tmp_path, capture):
    cart = _make_cartridge(tmp_path)
    output = tmp_path / "runs"
    run_cartridge_evals(cart, llm=MockLLMBackend(_scripted()), output_dir=output, capture=capture)
    original = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    (cart / "evals" / "cases" / "a_fresh_case.json").write_text(
        json.dumps({**_CASE, "id": "a_fresh_case"})
    )
    backend = MockLLMBackend(_scripted())

    with pytest.raises(FileExistsError, match="Choose a new output_dir"):
        run_cartridge_evals(cart, llm=backend, output_dir=output, capture=capture)

    assert backend.calls == 0
    assert not (output / "a_fresh_case").exists()
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == original


def test_repeated_eval_cli_reports_existing_output_without_a_traceback(tmp_path, capsys):
    cart = _make_cartridge(tmp_path)
    output = tmp_path / "runs"
    run_cartridge_evals(cart, llm=MockLLMBackend(_scripted()), output_dir=output)
    backend = MockLLMBackend(_scripted())

    with patch("looplet.backends.make_backend", return_value=backend):
        assert eval_cli(["run", str(cart), "--out", str(output), "--json"]) == 1

    captured = capsys.readouterr()
    assert not captured.out
    assert "Choose a new output_dir (CLI: --out)" in captured.err
    assert "Traceback" not in captured.err
    assert backend.calls == 0


def test_run_cartridge_evals_no_output_uses_tempdir(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    records = run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()))
    assert len(records) == 1
    # sandbox dir exists and was seeded.
    assert (records[0].directory / "test_greeting.py").is_file()


def test_persisted_eval_capture_can_be_disabled_explicitly(tmp_path):
    from looplet.artifact_compat import read_artifact_descriptor

    cart = _make_cartridge(tmp_path)
    record = run_cartridge_evals(
        cart, llm=MockLLMBackend(_scripted()), output_dir=tmp_path / "runs", capture=False
    )[0]
    assert record.context.completed
    assert not (record.directory / "manifest.jsonl").exists()
    assert "model_calls" not in read_artifact_descriptor(record.directory)["components"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory permissions")
def test_captured_eval_directory_is_private_before_model_call(tmp_path):
    cart = _make_cartridge(tmp_path)
    out = tmp_path / "runs"
    run_dir = out / "make_greeting"
    calls = []

    class Backend(MockLLMBackend):
        def generate(self, prompt, **kwargs):
            assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
            calls.append(prompt)
            return super().generate(prompt, **kwargs)

    previous_umask = os.umask(0)
    try:
        record = run_cartridge_evals(cart, llm=Backend(_scripted()), output_dir=out)[0]
    finally:
        os.umask(previous_umask)

    assert len(calls) == 2 and record.context.completed
    assert stat.S_IMODE(record.directory.stat().st_mode) == 0o700
    assert (record.directory / "manifest.jsonl").is_file()
    assert load_eval_run(record.directory).context.completed


@pytest.mark.parametrize("native", [False, True])
def test_saved_eval_replays_captured_responses_with_fresh_tool_effects(tmp_path, native):
    from looplet import cartridge_to_preset
    from looplet.provenance import replay_loop

    cart = _make_cartridge(tmp_path)
    if native:
        (cart / "runtime.yaml").write_text("use_native_tools: true\n")

        class Backend:
            def __init__(self):
                self.responses = iter(_scripted())

            def generate(self, prompt, **kwargs):
                raise AssertionError("native protocol expected")

            def generate_with_tools(self, prompt, **kwargs):
                response = json.loads(next(self.responses))
                return [
                    {
                        "type": "tool_use",
                        "id": response["tool"],
                        "name": response["tool"],
                        "input": response["args"],
                    }
                ]

        backend = Backend()
    else:
        backend = MockLLMBackend(_scripted())
    record = run_cartridge_evals(cart, llm=backend, output_dir=tmp_path / "runs")[0]
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with cartridge_to_preset(cart, runtime={"project_root": str(fresh)}) as preset:
        steps = list(replay_loop(record.directory, tools=preset.tools, config=preset.config))
    assert [step.tool_call.tool for step in steps] == ["write_file", "done"]
    assert (fresh / "greeting.py").read_text() == "def hi():\n    return 'hi'\n"
    assert (record.directory / "workspace" / "greeting.py").is_file()


@pytest.mark.parametrize("capture", [False, True])
def test_eval_evidence_redaction_scrubs_upstream_and_persisted_records(tmp_path, capture):
    cart = _make_cartridge(tmp_path)
    marker = "PRIVATE_MARKER"
    case_file = cart / "evals" / "cases" / "make_greeting.json"
    case = json.loads(case_file.read_text())
    case["task"]["goal"] += marker
    case["expected"]["private"] = marker
    case_file.write_text(json.dumps(case))
    seen = []

    class Backend(MockLLMBackend):
        def generate(self, prompt, **kwargs):
            seen.append(prompt)
            return super().generate(prompt, **kwargs)

    record = run_cartridge_evals(
        cart,
        llm=Backend(_scripted()),
        output_dir=tmp_path / "runs",
        capture=capture,
        redact=lambda text: text.replace(marker, "[REDACTED]"),
    )[0]
    assert seen and all(marker not in prompt for prompt in seen)
    for path in record.directory.rglob("*"):
        if path.is_file() and "workspace" not in path.relative_to(record.directory).parts:
            assert marker not in path.read_text()
    assert (record.directory / "manifest.jsonl").exists() is capture


def test_interrupted_eval_persists_partial_capture_without_claiming_completion(tmp_path):
    cart = _make_cartridge(tmp_path)
    hook_dir = cart / "hooks" / "01_Interrupt"
    hook_dir.mkdir(parents=True)
    (hook_dir / "config.yaml").write_text("class_name: Interrupt\n")
    (hook_dir / "hook.py").write_text(
        "class Interrupt:\n    def pre_prompt(self, state, session_log, context, step_num):\n        if state.steps:\n            raise RuntimeError('interrupted run')\n"
    )
    output = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="interrupted run"):
        run_cartridge_evals(cart, llm=MockLLMBackend(_scripted()), output_dir=output)
    record = load_eval_run(output / "make_greeting")
    assert not record.context.completed
    assert record.context.stop_reason == "error"
    assert record.context.tool_sequence == ["write_file"]
    assert len((record.directory / "manifest.jsonl").read_text().strip().split("\n")) == 1


@pytest.mark.parametrize("capture", [False, True])
def test_cli_capture_policy_matches_api(tmp_path, monkeypatch, capture):
    cart = _make_cartridge(tmp_path)
    monkeypatch.setattr(
        "looplet.backends.OpenAIBackend", lambda **kwargs: MockLLMBackend(_scripted())
    )
    out = tmp_path / "runs"
    args = ["run", str(cart), "--out", str(out), "--base-url", "http://unused.invalid/v1", "--json"]
    if not capture:
        args.append("--no-capture")
    assert eval_cli(args) == 0
    assert (out / "make_greeting" / "manifest.jsonl").exists() is capture


def test_failed_no_output_run_cleans_owned_sandbox(tmp_path, monkeypatch):
    import tempfile

    cart = _make_cartridge(tmp_path)
    hook_dir = cart / "hooks" / "01_Interrupt"
    hook_dir.mkdir(parents=True)
    (hook_dir / "config.yaml").write_text("class_name: Interrupt\n")
    (hook_dir / "hook.py").write_text(
        "class Interrupt:\n    def pre_prompt(self, *args, **kwargs):\n        raise RuntimeError('interrupted run')\n"
    )
    created = []
    original = tempfile.mkdtemp

    def make(*args, **kwargs):
        path = Path(original(*args, **kwargs))
        if kwargs.get("prefix", "").startswith("evalcase_"):
            created.append(path)
        return str(path)

    monkeypatch.setattr("looplet.evals.tempfile.mkdtemp", make)
    with pytest.raises(RuntimeError, match="interrupted run"):
        run_cartridge_evals(cart, llm=MockLLMBackend(_scripted()))
    assert created and all(not path.exists() for path in created)


def test_self_test_runner_collects_once_instead_of_using_embedded_eval_hook(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    _add_embedded_eval_hook(cart)
    collector_file = cart / "evals" / "collect_outcome.py"
    collector_file.write_text(
        _COLLECTOR + "\ndef collect_count(state, runtime):\n"
        "    from pathlib import Path\n"
        "    marker = Path(runtime['project_root']) / 'self-test-collector-count.txt'\n"
        "    count = int(marker.read_text()) if marker.exists() else 0\n"
        "    marker.write_text(str(count + 1))\n"
        "    return {'collection_count': count + 1}\n"
    )

    record = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        output_dir=tmp_path / "runs",
    )[0]

    workspace = record.directory / "workspace"
    assert record.context.artifacts["collection_count"] == 1
    assert record.context.artifacts["expected_leaked"] is False
    assert (workspace / "self-test-collector-count.txt").read_text() == "1"
    assert not (workspace / "embedded-collector-count.txt").exists()
    assert "<lambda>" not in {result.name for result in record.results}


def test_host_can_still_explicitly_load_an_embedded_eval_hook(tmp_path: Path) -> None:
    from looplet import cartridge_to_preset

    cart = _make_cartridge(tmp_path)
    _add_embedded_eval_hook(cart)
    workspace = tmp_path / "host-workspace"
    workspace.mkdir()
    with cartridge_to_preset(cart, runtime={"project_root": str(workspace)}) as preset:
        list(preset.run(MockLLMBackend(responses=_scripted()), task={"goal": "Write greeting.py"}))

    assert (workspace / "embedded-collector-count.txt").read_text() == "1"


@pytest.mark.parametrize("version", [None, 1, 2])
@pytest.mark.parametrize(
    ("body", "required", "judge", "collector_body", "passed"),
    [
        pytest.param("return True", False, False, None, True, id="boolean-pass"),
        pytest.param("return False", False, False, None, False, id="boolean-fail"),
        pytest.param("return 0.5", False, False, None, True, id="score-boundary"),
        pytest.param("return 0.4999", False, False, None, False, id="score-below-boundary"),
        pytest.param("return 0.0", False, False, None, False, id="zero-score"),
        pytest.param("return 'correct'", False, False, None, True, id="legacy-pass-label"),
        pytest.param("return 'partial'", False, False, None, False, id="unknown-label"),
        pytest.param("return {'score': 0.0}", True, False, None, False, id="explicit-zero"),
        pytest.param("return {'steps': 20}", False, False, None, (True, False), id="metric-only"),
        pytest.param("return {'steps': 20}", True, False, None, False, id="required-metric"),
        pytest.param(
            "return EvalResult(metrics={'f1': 0.1})",
            False,
            False,
            None,
            (True, False),
            id="passive-f1",
        ),
        pytest.param("return {'f1': 0.1}", False, False, None, False, id="legacy-f1-gate"),
        pytest.param("return True", False, True, None, (True, False), id="optional-judge-skipped"),
        pytest.param("return True", True, True, None, False, id="required-judge-skipped"),
        pytest.param(
            "return EvalResult(score=1.0, label='wrong')",
            False,
            False,
            None,
            False,
            id="contradictory-fail",
        ),
        pytest.param(
            "return EvalResult(score=0.0, label='pass')",
            False,
            False,
            None,
            False,
            id="contradictory-pass",
        ),
        pytest.param("return float('nan')", False, False, None, False, id="invalid-score"),
        pytest.param(
            "return {'latency': float('inf')}", False, False, None, False, id="invalid-metric"
        ),
        pytest.param(
            "raise RuntimeError('grader exploded')", False, False, None, False, id="grader-error"
        ),
        pytest.param("return EvalResult()", False, False, None, False, id="empty-result"),
        pytest.param("return None", False, False, None, False, id="missing-return"),
        pytest.param(
            "return EvalResult(name='not_eval_gate', score=1.0)",
            False,
            False,
            None,
            False,
            id="spoofed-grader-name",
        ),
        pytest.param(
            "return True",
            False,
            False,
            "raise RuntimeError('collector exploded')",
            False,
            id="collector-error",
        ),
        pytest.param(
            "return True", False, False, "return 'not a dict'", False, id="collector-shape-error"
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Inferring an eval gate score:DeprecationWarning")
def test_verdict_agrees_live_saved_pytest_and_cli(
    tmp_path: Path,
    monkeypatch,
    capsys,
    body: str,
    required: bool,
    judge: bool,
    collector_body: str | None,
    passed: bool | tuple[bool, bool],
    version: int | None,
) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_correctness.py").write_text(
        "from looplet import EvalResult, eval_mark\n\n"
        + ("@eval_mark('required')\n" if required else "")
        + f"def eval_gate(ctx{', llm' if judge else ''}):\n    {body}\n"
    )
    if collector_body is not None:
        (cart / "evals" / "collect_outcome.py").write_text(
            _COLLECTOR + f"\ndef collect_fault(state):\n    {collector_body}\n"
        )

    record = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        output_dir=tmp_path / "live",
    )[0]
    reloaded = load_eval_run(record.directory)
    graders = eval_discover(cart / "evals", strict=True)
    offline = eval_run(graders, reloaded.context)
    signature = lambda results: [
        (result.name, result.score, result.label, result.metrics, result.explanation)
        for result in results
    ]
    assert signature(offline) == signature(record.results)
    assert record.context.artifacts["expected_leaked"] is False
    effective_version = 2 if version is None else version
    assessment_args = {} if version is None else {"version": version}
    if isinstance(passed, tuple):
        passed = passed[effective_version - 1]
    captured = EvalReport.from_records([reloaded], **assessment_args)
    assert captured.passed is passed
    rescored = eval_assess(graders, reloaded.context, case=reloaded.case, **assessment_args)
    assert rescored.passed is passed
    assert captured.grader_manifest == rescored.grader_manifest
    if passed:
        assert_evals_pass(reloaded.context, graders, **assessment_args)
    else:
        with pytest.raises(AssertionError):
            assert_evals_pass(reloaded.context, graders, **assessment_args)

    expected_exit = 0 if passed else 1
    policy_args = (
        []
        if version is None
        else ["--summary-version" if version == 1 else "--assessment-version", str(version)]
    )
    assert (
        eval_cli([str(tmp_path / "live"), "--evals", str(cart / "evals"), *policy_args])
        == expected_exit
    )
    capsys.readouterr()
    assert (
        eval_cli([str(tmp_path / "live"), "--evals", str(cart / "evals"), "--json", *policy_args])
        == expected_exit
    )
    saved_report = json.loads(capsys.readouterr().out)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://looplet.invalid/v1")
    monkeypatch.setattr(
        "looplet.backends.make_backend",
        lambda **kwargs: MockLLMBackend(responses=_scripted()),
    )
    assert (
        eval_cli(["run", str(cart), "--out", str(tmp_path / "cli"), *policy_args]) == expected_exit
    )
    capsys.readouterr()
    assert eval_cli(["run", str(cart), "--no-save", "--json", *policy_args]) == expected_exit
    report = json.loads(capsys.readouterr().out)
    for assessment in (report, saved_report):
        for case in assessment["cases"]:
            for result in case["results"]:
                result.pop("duration_ms", None)
    assert report == saved_report
    assert report["version"] == effective_version
    assert report["passed"] is passed
    assert report["state"] == ("pass" if passed else "fail")


def test_no_output_eval_record_cleans_up_owned_tempdir(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    record = run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()))[0]
    sandbox = record.directory

    record.cleanup()

    assert not sandbox.exists()


def test_persisted_eval_record_cleanup_preserves_evidence(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    output = tmp_path / "runs"
    record = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        output_dir=output,
    )[0]
    persisted = record.directory

    record.cleanup()

    assert (persisted / "trajectory.json").is_file()


def test_run_cartridge_evals_case_filter(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    records = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        cases=["make_greeting"],
    )
    assert [record.case.id for record in records] == ["make_greeting"]


def test_run_cartridge_evals_rejects_unknown_case_filter(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    with pytest.raises(ValueError, match="Unknown eval case"):
        run_cartridge_evals(
            cart,
            llm=MockLLMBackend(responses=_scripted()),
            cases=["typo"],
        )


def test_run_cartridge_evals_rejects_empty_cases_and_graders(tmp_path):
    cart = _make_cartridge(tmp_path)
    with pytest.raises(ValueError, match="No eval cases selected"):
        run_cartridge_evals(cart, llm=MockLLMBackend(_scripted()), cases=[])
    (cart / "evals" / "eval_correctness.py").unlink()
    with pytest.raises(ValueError, match="No graders selected"):
        run_cartridge_evals(cart, llm=MockLLMBackend(_scripted()))


def test_run_cartridge_evals_rejects_expected_inside_agent_task(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    case_path = cart / "evals" / "cases" / "make_greeting.json"
    case_data = json.loads(case_path.read_text())
    case_data["task"]["expected"] = {"file_written": True}
    case_path.write_text(json.dumps(case_data))
    with pytest.raises(ValueError, match="reserved grader data"):
        run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()))


def test_new_eval_output_has_fresh_workspace_and_preserves_previous_run(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    out = tmp_path / "runs"
    stale = out / "make_greeting" / "workspace" / "stale.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("previous run\n")

    records = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        output_dir=tmp_path / "next-runs",
    )
    assert not (records[0].directory / "workspace" / "stale.py").exists()
    assert stale.read_text() == "previous run\n"


def test_case_sandbox_overrides_base_runtime_project_root(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    records = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        runtime={"project_root": str(outside)},
    )
    assert not (outside / "greeting.py").exists()
    assert (records[0].directory / "greeting.py").is_file()


# ── LLM-as-judge wiring (deterministic via MockLLMBackend) ───────


def test_judge_grader_runs_when_judge_llm_supplied(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    records = run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        judge_llm=MockLLMBackend(responses=["0.8"]),
    )
    by = {r.name: r for r in records[0].results}
    assert by["eval_judge_quality"].score == 0.8


def test_judge_grader_skipped_without_judge_llm(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    records = run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()))
    by = {r.name: r for r in records[0].results}
    # eval_run marks llm-requiring graders "skipped" when no judge is supplied.
    assert by["eval_judge_quality"].label == "skipped"


def test_cli_run_with_judge_flag(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    # --judge (no --judge-model) reuses the agent backend, so one FIFO queue:
    # agent consumes _scripted() (2), then the judge grader consumes "0.8".
    shared = MockLLMBackend(responses=_scripted() + ["0.8"])
    import looplet.backends as _backends

    monkeypatch.setattr(_backends, "make_backend", lambda **kw: shared)
    out = tmp_path / "runs"
    rc = eval_cli(["run", str(cart), "--judge", "--out", str(out)])
    assert rc == 0
    reloaded = load_eval_run(out / "make_greeting")
    scores = {r.name: r.score for r in reloaded.results}
    assert scores.get("eval_judge_quality") == 0.8


def test_cli_run_without_judge_skips_judge_grader(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    shared = MockLLMBackend(responses=_scripted())  # only agent calls, no judge
    import looplet.backends as _backends

    monkeypatch.setattr(_backends, "make_backend", lambda **kw: shared)
    out = tmp_path / "runs"
    rc = eval_cli(["run", str(cart), "--out", str(out)])
    assert rc == 0
    reloaded = load_eval_run(out / "make_greeting")
    by = {r.name: r for r in reloaded.results}
    assert by["eval_judge_quality"].label == "skipped"


def test_cli_run_json_emits_one_eval_report(tmp_path: Path, monkeypatch, capsys) -> None:
    cart = _make_cartridge(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=_scripted()),
    )
    out = tmp_path / "runs"

    rc = eval_cli(["run", str(cart), "--out", str(out), "--threshold", "1.0", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "looplet.eval-summary"
    assert payload["version"] == 2
    assert payload["state"] == "pass"
    assert payload["passed"] is True
    assert payload["threshold"] == 1.0
    assert payload["output_dir"] == str(out)
    assert payload["integrity_failures"] == []
    assert payload["cases"][0]["id"] == "make_greeting"
    assert payload["cases"][0]["marks"] == ["regression"]
    assert payload["cases"][0]["completed"] is True
    by_name = {result["name"]: result for result in payload["cases"][0]["results"]}
    assert set(by_name) >= {
        "eval_completed",
        "eval_wrote_file",
    }
    assert by_name["eval_completed"]["marks"] == ["smoke"]
    assert by_name["eval_completed"]["state"] == "pass"
    assert by_name["eval_completed"]["required_status"] == "not_required"
    manifest = {item["name"]: item for item in payload["grader_manifest"]}
    assert manifest["eval_completed"] == {
        "name": "eval_completed",
        "marks": ["smoke"],
        "required": False,
    }


@pytest.mark.parametrize(
    ("score", "label", "threshold", "required", "state"),
    [
        (0.25, None, 0.5, False, "threshold_fail"),
        (0.25, None, 0.0, True, "explicit_fail"),
        (None, "skipped", 0.5, True, "skipped"),
        (None, "skipped", 0.5, False, None),
        (None, "error", 0.5, False, "grader_error"),
        (0.5, None, 0.5, False, None),
        (1.0, "fail", 0.5, False, "explicit_fail"),
        (1.0, "pass", 0.5, False, None),
    ],
)
def test_cli_eval_human_failure_details_follow_policy(
    tmp_path, monkeypatch, capsys, score, label, threshold, required, state
):
    import looplet.evals as evaluation
    from looplet import EvalResult, eval_mark
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    load_original = evaluation.load_cartridge_evals

    @eval_mark(*(["required"] if required else []))
    def eval_policy(ctx):
        return EvalResult(score=score, label=label, explanation="Policy diagnostic marker")

    def eval_success(ctx):
        return True

    def load_definitions(*arguments, **keywords):
        return load_original(*arguments, **keywords)._replace(graders=[eval_policy, eval_success])

    monkeypatch.setattr(evaluation, "load_cartridge_evals", load_definitions)
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: MockLLMBackend(_scripted())
    )
    arguments = ["eval", "run", str(cart), "--no-save", "--threshold", str(threshold)]
    assert main(arguments) == (1 if state is not None else 0)
    output = capsys.readouterr().out
    assert ("Policy diagnostic marker" in output) is (state is not None)
    if state is not None:
        assert f"make_greeting / eval_policy: {state}" in output


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_run_routes_authored_output_without_changing_human_mode(
    tmp_path: Path, monkeypatch, capsys, json_output: bool
) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_correctness.py").write_text(
        "print('grader import diagnostic')\n" + _GRADERS + "\ndef eval_noisy(ctx):\n"
        "    print('grader execution diagnostic')\n"
        "    return True\n"
    )
    (cart / "tools" / "done" / "execute.py").write_text(
        "print('tool import diagnostic')\n"
        "def execute(*, summary: str) -> dict:\n"
        "    print('tool execution diagnostic')\n"
        "    return {'status': 'completed', 'summary': summary}\n"
    )
    (cart / "resources" / "project_dir.py").write_text(
        "def build(runtime=None):\n"
        "    print('resource build diagnostic')\n"
        "    return (runtime or {}).get('project_root', '.')\n"
    )
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: MockLLMBackend(responses=_scripted())
    )
    arguments = ["run", str(cart)]
    if json_output:
        arguments.append("--json")

    assert eval_cli(arguments) == 0

    output = capsys.readouterr()
    if json_output:
        assert json.loads(output.out)["passed"] is True
    diagnostics = output.err if json_output else output.out
    for phase in (
        "grader import",
        "grader execution",
        "tool import",
        "tool execution",
        "resource build",
    ):
        assert f"{phase} diagnostic" in diagnostics


def test_cli_run_json_preserves_threshold_failure_exit(tmp_path: Path, monkeypatch, capsys) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_zero.py").write_text("def eval_zero(ctx):\n    return 0.0\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=_scripted()),
    )

    rc = eval_cli(["run", str(cart), "--threshold", "1.0", "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "fail"
    assert payload["passed"] is False
    zero = next(
        result for result in payload["cases"][0]["results"] if result["name"] == "eval_zero"
    )
    assert zero["score"] == 0.0
    assert zero["state"] == "threshold_fail"


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("passed", [False, True])
def test_cli_default_output_preserves_each_run(
    tmp_path, monkeypatch, capsys, json_output, capture, passed
):
    cart = _make_cartridge(tmp_path)
    responses = (
        _scripted() if passed else [json.dumps({"tool": "done", "args": {"summary": "done"}})]
    )
    monkeypatch.setattr("looplet.backends.make_backend", lambda **kwargs: MockLLMBackend(responses))
    arguments = ["run", str(cart)]
    if json_output:
        arguments.append("--json")
    if not capture:
        arguments.append("--no-capture")
    saved = []
    for _ in range(2):
        assert eval_cli(arguments) == (0 if passed else 1)
        output = capsys.readouterr()
        if json_output:
            directory = Path(json.loads(output.out)["output_dir"])
        else:
            directory = next(
                directory
                for directory in (tmp_path / ".looplet" / "evals").iterdir()
                if directory not in saved
            )
            assert str(directory) in output.out
        assert directory.parent == tmp_path / ".looplet" / "evals"
        assert directory not in saved
        record = load_eval_run(directory / "make_greeting")
        assert record.context.completed
        assert (record.directory / "manifest.jsonl").exists() is capture
        assert (record.directory / ".gitignore").read_text() == "*\n"
        if not json_output:
            import shlex

            from looplet.__main__ import main

            command = next(
                line.removeprefix("Inspect: ")
                for line in output.out.splitlines()
                if line.startswith("Inspect: ")
            )
            assert main(shlex.split(command)[1:]) == 0
            capsys.readouterr()
        saved.append(directory)
    assert len(list((tmp_path / ".looplet" / "evals").iterdir())) == 2
    assert all((directory / "make_greeting" / "trajectory.json").is_file() for directory in saved)


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("max_steps", [None, 1])
@pytest.mark.parametrize("native_tools", [False, True])
def test_cli_replay_uses_fresh_workspace_without_provider(
    tmp_path, monkeypatch, capsys, json_output, max_steps, native_tools
):
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    (cart / "runtime.yaml").write_text(f"use_native_tools: {str(native_tools).lower()}\n")
    (cart / "tools" / "write_file" / "execute.py").write_text(
        _WRITE_EXEC.replace(
            "    root = Path", "    print('authored replay diagnostic')\n    root = Path"
        )
    )
    case_file = cart / "evals" / "cases" / "make_greeting.json"
    case_data = json.loads(case_file.read_text())
    case_data["expected"]["private_marker"] = "ORACLE_ONLY_FOR_GRADING"
    case_file.write_text(json.dumps(case_data))
    records = run_cartridge_evals(
        cart, llm=MockLLMBackend(_scripted()), output_dir=tmp_path / "original"
    )
    original = records[0].directory
    snapshot = {
        path.relative_to(original): path.read_bytes()
        for path in original.rglob("*")
        if path.is_file()
    }
    capsys.readouterr()

    def unexpected_provider(**kwargs):
        pytest.fail("replay must not construct a provider")

    monkeypatch.setattr("looplet.backends.make_backend", unexpected_provider)
    workspace = tmp_path / "replay workspace"
    arguments = [
        "replay",
        str(original),
        "--cartridge",
        str(cart),
        "--workspace",
        str(workspace),
        "--require-completed",
    ]
    if json_output:
        arguments.append("--json")
    if max_steps is not None:
        arguments.extend(["--max-steps", str(max_steps)])
    assert main(arguments) == (1 if max_steps == 1 else 0)
    output = capsys.readouterr()
    assert (workspace / "greeting.py").read_bytes() == (
        original / "workspace" / "greeting.py"
    ).read_bytes()
    assert (workspace / "test_greeting.py").is_file()
    if json_output:
        completion = json.loads(output.out)
        assert completion["completed"] is (max_steps is None)
        assert "authored replay diagnostic" in output.err
        replay_trace = Path(completion["trace_dir"])
    else:
        assert "captured responses" in output.out
        replay_trace = next((workspace / ".looplet" / "traces").iterdir())
    assert replay_trace.is_relative_to(workspace)
    source_id = json.loads((original / "trajectory.json").read_text())["run_id"]
    replayed = json.loads((replay_trace / "trajectory.json").read_text())
    assert replayed["metadata"]["parent_run_id"] == source_id
    assert replayed["metadata"]["source_case_id"] == "make_greeting"
    saved = load_eval_run(replay_trace)
    assert saved.case.id == "make_greeting"
    assert saved.context.expected == case_data["expected"]
    assert saved.context.artifacts == {"file_written": True, "expected_leaked": False}
    assert saved.results == []
    assert "ORACLE_ONLY_FOR_GRADING" not in (replay_trace / "call_00_prompt.txt").read_text()
    assert (replay_trace / ".gitignore").read_text() == "*\n"
    assert snapshot == {
        path.relative_to(original): path.read_bytes()
        for path in original.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("outcome", ["unchanged", "changed_tool", "collector_error"])
def test_cli_replay_collects_fresh_results_for_rescoring(tmp_path, monkeypatch, capsys, outcome):
    import shlex

    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    content = json.loads(_scripted()[0])["args"]["content"]
    (cart / "evals" / "collect_outcome.py").write_text(
        "def collect_output(state, runtime):\n"
        "    from pathlib import Path\n"
        '    root = Path(runtime["project_root"])\n'
        '    return {"file_contents": (root / "greeting.py").read_text()}\n'
    )
    (cart / "evals" / "eval_correctness.py").write_text(
        f'def eval_output(ctx):\n    return ctx.artifacts["file_contents"] == {content!r}\n'
    )
    original = run_cartridge_evals(
        cart, llm=MockLLMBackend(_scripted()), output_dir=tmp_path / "original"
    )[0]
    EvalReport.from_records([original]).assert_passed()
    if outcome == "changed_tool":
        implementation = cart / "tools" / "write_file" / "execute.py"
        implementation.write_text(
            _WRITE_EXEC.replace("target.write_text(content)", 'target.write_text("changed result")')
        )
    elif outcome == "collector_error":
        (cart / "evals" / "collect_outcome.py").write_text(
            'def collect_output(state, runtime):\n    raise RuntimeError("fresh collection failed")\n'
        )
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: pytest.fail("no provider")
    )
    capsys.readouterr()
    workspace = tmp_path / "replay workspace"
    assert (
        main(
            [
                "replay",
                str(original.directory),
                "--cartridge",
                str(cart),
                "--workspace",
                str(workspace),
                "--require-completed",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    command = next(
        line.strip().removeprefix("Grade: ")
        for line in output.out.splitlines()
        if line.strip().startswith("Grade: ")
    )
    replayed = load_eval_run(next((workspace / ".looplet" / "traces").iterdir()))
    assert replayed.context.expected == original.context.expected
    if outcome == "collector_error":
        assert "file_contents" not in replayed.context.artifacts
        assert any(
            result.name == "collector:collect_output" and result.label == "error"
            for result in replayed.results
        )
    else:
        assert replayed.context.artifacts["file_contents"] == (
            "changed result" if outcome == "changed_tool" else content
        )
        assert replayed.results == []
    assert main(shlex.split(command)[1:]) == (0 if outcome == "unchanged" else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is (outcome == "unchanged")
    assert report["valid"] is (outcome != "collector_error")
    assert report["cases"][0]["id"] == "make_greeting"


@pytest.mark.parametrize("skill_file", [False, True])
def test_cli_replay_explains_skill_bundle_format(tmp_path, monkeypatch, capsys, skill_file):
    from looplet.__main__ import main

    bundle = Path(__file__).parent / "fixtures" / "coder_skill_bundle"
    monkeypatch.setattr("looplet.bundles.load_skill_bundle", lambda *args: pytest.fail("no import"))
    workspace = tmp_path / "replayed"
    assert (
        main(
            [
                "replay",
                str(tmp_path / "capture"),
                "--cartridge",
                str(bundle / "SKILL.md" if skill_file else bundle),
                "--workspace",
                str(workspace),
                "--task",
                "Review this repository",
                "--json",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "not skill bundles" in output.err
    assert "load_skill_bundle() with replay_loop()" in output.err
    assert "#replay-skill-bundles" in output.err
    assert "cartridge not found" not in output.err
    assert not workspace.exists()


def test_cli_replay_executes_changed_tools(tmp_path, monkeypatch, capsys):
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    original = run_cartridge_evals(
        cart, llm=MockLLMBackend(_scripted()), output_dir=tmp_path / "original"
    )[0].directory
    original_contents = (original / "workspace" / "greeting.py").read_text()
    implementation = cart / "tools" / "write_file" / "execute.py"
    implementation.write_text(
        _WRITE_EXEC.replace(
            "target.write_text(content)", 'target.write_text(content + "\\nreplayed = True\\n")'
        )
    )
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: pytest.fail("no provider")
    )
    capsys.readouterr()
    workspace = tmp_path / "changed-tool-replay"
    assert (
        main(
            [
                "replay",
                str(original),
                "--cartridge",
                str(cart),
                "--workspace",
                str(workspace),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["completed"]
    assert (workspace / "greeting.py").read_text() == original_contents + "\nreplayed = True\n"
    assert (original / "workspace" / "greeting.py").read_text() == original_contents


@pytest.mark.parametrize("python_test", [False, True])
def test_cli_eval_init_prints_runnable_next_steps(tmp_path, capsys, python_test):
    import subprocess
    import sys

    from looplet.__main__ import main

    definitions = tmp_path / "project with spaces" / "evals"
    arguments = ["eval", "init", str(definitions)]
    if python_test:
        arguments.append("--python")
    assert main(arguments) == 0
    output = capsys.readouterr()
    assert "Check: looplet eval check" in output.out
    starter = definitions / "test_agent.py"
    assert starter.exists() is python_test
    assert main(["eval", "check", str(definitions), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    if not python_test:
        return
    assert "Test: python -m pytest" in output.out
    command = [sys.executable, "-m", "pytest", str(starter), "-q"]
    passing = subprocess.run(
        command, cwd=definitions.parent, capture_output=True, text=True, timeout=20
    )
    assert passing.returncode == 0, passing.stdout + passing.stderr
    case_path = definitions / "cases" / "hello.json"
    case = json.loads(case_path.read_text())
    case["expected"]["content"] = "The grader must reject the observed hello output."
    case_path.write_text(json.dumps(case))
    failing = subprocess.run(
        command, cwd=definitions.parent, capture_output=True, text=True, timeout=20
    )
    assert failing.returncode == 1, failing.stdout + failing.stderr
    assert "1 failed" in failing.stdout


@pytest.mark.parametrize("relative", ["cases/hello.json", "eval_outcome.py", "test_agent.py"])
def test_cli_eval_init_python_preserves_existing_files(tmp_path, capsys, relative):
    from looplet.__main__ import main

    definitions = tmp_path / "evals"
    existing = definitions / relative
    existing.parent.mkdir(parents=True)
    existing.write_text("user-owned content\n")
    assert main(["eval", "init", str(definitions), "--python"]) == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert existing.read_text() == "user-owned content\n"
    assert [path for path in definitions.rglob("*") if path.is_file()] == [existing]


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("save_evidence", [False, True])
def test_cli_eval_exposes_failed_grader_explanations(
    tmp_path, monkeypatch, capsys, json_output, save_evidence
):
    import looplet.evals as evaluation
    from looplet import EvalResult, eval_mark
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    load_original = evaluation.load_cartridge_evals

    @eval_mark("required")
    def eval_validation(ctx):
        return EvalResult(
            score=0.0,
            label="fail",
            explanation="No validation result was collected",
            details=["The host did not supply validation evidence."],
        )

    def eval_success(ctx):
        return EvalResult(score=1.0, explanation="Passing grader detail")

    def load_definitions(*arguments, **keywords):
        return load_original(*arguments, **keywords)._replace(
            graders=[eval_validation, eval_success]
        )

    monkeypatch.setattr(evaluation, "load_cartridge_evals", load_definitions)
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: MockLLMBackend(_scripted())
    )
    arguments = ["eval", "run", str(cart)]
    if save_evidence:
        arguments.extend(["--out", str(tmp_path / "records")])
    else:
        arguments.append("--no-save")
    if json_output:
        arguments.append("--json")
    assert main(arguments) == 1
    output = capsys.readouterr()
    assert not output.err
    assert "No validation result was collected" in output.out
    assert "The host did not supply validation evidence." in output.out
    if json_output:
        report = json.loads(output.out)
        assert report["passed"] is False
        assert "Passing grader detail" in output.out
    else:
        assert "make_greeting" in output.out
        assert "eval_validation" in output.out
        assert "Passing grader detail" not in output.out


@pytest.mark.parametrize("saved_case", [False, True])
@pytest.mark.parametrize("original_budget", [1, 2])
@pytest.mark.parametrize(
    ("native_tools", "batched_turn"), [(False, False), (True, False), (True, True)]
)
def test_cli_replay_restores_task_and_bounds_default_budget(
    tmp_path, monkeypatch, capsys, saved_case, original_budget, native_tools, batched_turn
):
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    (cart / "runtime.yaml").write_text(f"use_native_tools: {str(native_tools).lower()}\n")
    responses = _scripted()
    if batched_turn:
        responses = [json.dumps({"tools": [json.loads(response) for response in responses]})]
    if saved_case:
        original = run_cartridge_evals(
            cart,
            llm=MockLLMBackend(responses),
            output_dir=tmp_path / "original",
            max_steps=original_budget,
        )[0].directory
    else:
        project = tmp_path / "project"
        project.mkdir()
        original = tmp_path / "original"
        arguments = [
            "run",
            str(cart),
            "Write greeting.py",
            "--project-root",
            str(project),
            "--trace-dir",
            str(original),
            "--max-steps",
            str(original_budget),
            "--json",
        ]
        for response in responses:
            arguments.extend(["--scripted-response", response])
        assert main(arguments) == 0
    snapshot = {
        path.relative_to(original): path.read_bytes()
        for path in original.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: pytest.fail("no provider")
    )
    capsys.readouterr()
    workspace = tmp_path / "replayed"
    assert main(
        [
            "replay",
            str(original),
            "--cartridge",
            str(cart),
            "--workspace",
            str(workspace),
            "--require-completed",
            "--json",
        ]
    ) == (1 if original_budget == 1 else 0)
    output = capsys.readouterr()
    completion = json.loads(output.out)
    assert completion["completed"] is (original_budget == 2)
    assert completion["termination_reason"] == (
        "budget_exhausted" if original_budget == 1 else "done"
    )
    assert completion["steps"] == original_budget
    assert "replay exhausted" not in output.err
    replayed = json.loads((Path(completion["trace_dir"]) / "trajectory.json").read_text())
    assert replayed["task"]["goal"] == "Write greeting.py"
    assert (workspace / "greeting.py").is_file()
    assert snapshot == {
        path.relative_to(original): path.read_bytes()
        for path in original.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("explicit_task", [False, True])
@pytest.mark.parametrize(
    "recorded_task",
    [
        None,
        {
            "goal": "Recorded original task",
            "ticket": {"id": 7},
            "files": {"private.txt": "not agent task data"},
            "expected": {"private_marker": "ORACLE_ONLY_FOR_GRADING"},
        },
        "invalid task",
        {"files": {}, "expected": {"private_marker": "ORACLE_ONLY_FOR_GRADING"}},
    ],
)
def test_cli_raw_replay_uses_saved_task_or_requires_override(
    tmp_path, monkeypatch, capsys, explicit_task, recorded_task
):
    from looplet import ProvenanceSink
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    sink = ProvenanceSink(tmp_path / "raw")
    backend = sink.wrap_llm(MockLLMBackend(_scripted()))
    backend.generate("original task")
    backend.generate("original task after tool")
    trace = sink.flush()
    if recorded_task is not None:
        (trace / "trajectory.json").write_text(
            json.dumps({"run_id": "raw-capture", "task": recorded_task})
        )
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: pytest.fail("no provider")
    )
    workspace = tmp_path / "raw replay"
    arguments = [
        "replay",
        str(trace),
        "--cartridge",
        str(cart),
        "--workspace",
        str(workspace),
        "--json",
    ]
    if explicit_task:
        arguments.extend(["--task", "Write greeting.py"])
    has_saved_task = isinstance(recorded_task, dict) and "goal" in recorded_task
    assert main(arguments) == (0 if explicit_task or has_saved_task else 1)
    output = capsys.readouterr()
    if explicit_task or has_saved_task:
        completion = json.loads(output.out)
        assert completion["completed"]
        replay_trace = Path(completion["trace_dir"])
        replayed = json.loads((replay_trace / "trajectory.json").read_text())
        assert replayed["task"] == (
            {"goal": "Write greeting.py"}
            if explicit_task
            else {"goal": "Recorded original task", "ticket": {"id": 7}}
        )
        assert "ORACLE_ONLY_FOR_GRADING" not in (replay_trace / "call_00_prompt.txt").read_text()
    else:
        assert "raw traces need --task" in output.err
        assert not workspace.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "existing_workspace",
        "symlink_workspace",
        "nested_workspace",
        "missing_calls",
        "invalid_budget",
    ],
)
def test_cli_replay_preflights_before_authored_load(tmp_path, monkeypatch, capsys, failure):
    from looplet import ProvenanceSink
    from looplet.__main__ import main

    cart = _make_cartridge(tmp_path)
    sink = ProvenanceSink(tmp_path / "trace")
    sink.wrap_llm(MockLLMBackend(_scripted())).generate("original task")
    trace = sink.flush()
    workspace = tmp_path / "fresh"
    if failure == "existing_workspace":
        workspace.mkdir()
        (workspace / "keep.txt").write_text("untouched")
    elif failure == "symlink_workspace":
        workspace.symlink_to(tmp_path / "symlink-target", target_is_directory=True)
    elif failure == "nested_workspace":
        workspace = trace / "nested"
    elif failure == "missing_calls":
        (trace / "call_00_response.txt").unlink()
    arguments = [
        "replay",
        str(trace),
        "--cartridge",
        str(cart),
        "--workspace",
        str(workspace),
        "--task",
        "Write greeting.py",
        "--json",
    ]
    if failure == "invalid_budget":
        arguments.extend(["--max-steps", "0"])

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid replay must fail before authored cartridge load")

    monkeypatch.setattr("looplet.cartridge_to_preset", unexpected_load)
    assert main(arguments) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "cannot prepare replay" in output.err
    assert "Traceback" not in output.err
    if failure == "existing_workspace":
        assert (workspace / "keep.txt").read_text() == "untouched"
    elif failure == "symlink_workspace":
        assert workspace.is_symlink()
        assert not (tmp_path / "symlink-target").exists()
    else:
        assert not workspace.exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_no_save_explicitly_disables_evidence(tmp_path, monkeypatch, capsys, json_output):
    cart = _make_cartridge(tmp_path)
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: MockLLMBackend(_scripted())
    )
    arguments = ["run", str(cart), "--no-save"]
    if json_output:
        arguments.append("--json")
    assert eval_cli(arguments) == 0
    output = capsys.readouterr()
    if json_output:
        assert json.loads(output.out)["output_dir"] is None
    else:
        assert "evidence not saved (--no-save)" in output.out
    assert not (tmp_path / ".looplet").exists()


def test_cli_no_save_rejects_explicit_output_before_backend(tmp_path, monkeypatch):
    cart = _make_cartridge(tmp_path)

    def unexpected_backend(**kwargs):
        pytest.fail("must reject conflicting output options before backend setup")

    monkeypatch.setattr("looplet.backends.make_backend", unexpected_backend)
    with pytest.raises(SystemExit) as stopped:
        eval_cli(["run", str(cart), "--no-save", "--out", str(tmp_path / "runs")])
    assert stopped.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_unwritable_default_output_fails_before_model_call(
    tmp_path, monkeypatch, capsys, json_output
):
    cart = _make_cartridge(tmp_path)
    backend = MockLLMBackend(_scripted())
    original_mkdir = Path.mkdir

    def fail_evidence_directory(directory, *args, **kwargs):
        if directory.is_relative_to(tmp_path / ".looplet"):
            raise PermissionError("read-only evidence directory")
        return original_mkdir(directory, *args, **kwargs)

    monkeypatch.setattr("looplet.backends.make_backend", lambda **kwargs: backend)
    monkeypatch.setattr(Path, "mkdir", fail_evidence_directory)
    arguments = ["run", str(cart)] + (["--json"] if json_output else [])
    assert eval_cli(arguments) == 1
    output = capsys.readouterr()
    if json_output:
        assert not output.out
    assert "read-only evidence directory" in output.err
    assert "writable --out" in output.err and "--no-save" in output.err
    assert "Traceback" not in output.err
    assert backend.calls == 0


def test_cli_human_table_aligns_long_names_and_metrics(tmp_path, monkeypatch, capsys):
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_metrics.py").write_text(
        "def eval_extremely_long_metric_grader_name(ctx):\n"
        "    return {'observed_artifact_count_exceeding_column_width': 1234567890.0}\n"
    )
    monkeypatch.setattr(
        "looplet.backends.make_backend", lambda **kwargs: MockLLMBackend(_scripted())
    )
    assert eval_cli(["run", str(cart), "--no-save"]) == 0
    output = capsys.readouterr().out.splitlines()
    header = next(line for line in output if line.startswith("case "))
    row = next(line for line in output if line.startswith("make_greeting "))
    assert [cell.strip() for cell in header.split("|")] == [
        "case",
        "eval_completed",
        "eval_expected_available",
        "eval_extremely_long_metric_grader_name",
        "eval_judge_quality",
        "eval_wrote_file",
    ]
    assert "observed_artifact_count_exceeding_column_width=1.23457e+09" in row
    assert [index for index, character in enumerate(header) if character == "|"] == [
        index for index, character in enumerate(row) if character == "|"
    ]
    assert len(header) == len(row)


def test_cli_run_unknown_case_returns_failure(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=[]),
    )
    assert eval_cli(["run", str(cart), "--case", "typo"]) == 1


def test_cli_run_rejects_invalid_threshold_before_backend_setup(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    assert eval_cli(["run", str(cart), "--threshold", "nan"]) == 1


def test_cli_run_fails_on_evaluator_error(tmp_path: Path, monkeypatch, capsys) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_broken.py").write_text(
        "def eval_broken(ctx):\n    raise RuntimeError('grader broke')\n"
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=_scripted()),
    )
    assert eval_cli(["run", str(cart), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    broken = next(
        result for result in payload["cases"][0]["results"] if result["name"] == "eval_broken"
    )
    assert broken["state"] == "grader_error"
    assert broken["error"] == "grader broke"
    assert payload["passed"] is False


def test_cli_run_fails_on_failing_verdict_label(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_verdict.py").write_text("def eval_verdict(ctx):\n    return 'wrong'\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=_scripted()),
    )
    assert eval_cli(["run", str(cart)]) == 1


def test_cli_run_fails_when_required_judge_is_skipped(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_required.py").write_text(
        "from looplet import eval_mark\n\n"
        "@eval_mark('required')\n"
        "def eval_required_judge(ctx, llm):\n"
        "    raise AssertionError('must not run without a judge')\n"
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=_scripted()),
    )
    out = tmp_path / "runs"
    assert eval_cli(["run", str(cart), "--out", str(out)]) == 1
    by_name = {result.name: result for result in load_eval_run(out / "make_greeting").results}
    assert by_name["eval_required_judge"].label == "skipped"
    assert "must not run" not in by_name["eval_required_judge"].explanation


def test_cli_run_fails_on_collector_error(tmp_path: Path, monkeypatch, capsys) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "collect_outcome.py").write_text(
        "def collect_broken(state, runtime):\n    raise RuntimeError('collector broke')\n"
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    import looplet.backends as _backends

    monkeypatch.setattr(
        _backends,
        "make_backend",
        lambda **kw: MockLLMBackend(responses=_scripted()),
    )
    assert eval_cli(["run", str(cart), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    broken = next(
        result
        for result in payload["cases"][0]["results"]
        if result["name"] == "collector:collect_broken"
    )
    assert broken["state"] == "collector_error"
    assert broken["error"] == "RuntimeError: collector broke"
    assert payload["passed"] is False


# ── CLI preflight / error paths (no live model needed) ───────────


def test_cli_run_help_advertises_json(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        eval_cli(["run", "--help"])
    assert exc.value.code == 0
    assert "--json" in capsys.readouterr().out


def test_cli_run_missing_cartridge() -> None:
    assert eval_cli(["run", "/no/such/cartridge"]) == 1


def test_cli_run_no_evals_dir(tmp_path: Path, monkeypatch) -> None:
    # A cartridge directory with no evals/ → error.
    cart = tmp_path / "empty.cartridge"
    cart.mkdir()
    (cart / "cartridge.json").write_text('{"name": "x", "schema_version": 2}')
    assert eval_cli(["run", str(cart)]) == 1


def test_cli_run_broken_eval_module_fails_preflight(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    (cart / "evals" / "eval_broken.py").write_text("def broken(:\n")
    assert eval_cli(["run", str(cart)]) == 1


def test_cli_run_no_backend_configured(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        assert eval_cli(["run", str(cart)]) == 1


@pytest.mark.parametrize(
    ("environment", "provider"),
    [
        ({"OPENAI_API_KEY": "fixture-key"}, "OpenAIBackend"),
        ({"OPENAI_BASE_URL": "http://localhost:12345/v1"}, "OpenAIBackend"),
        ({"ANTHROPIC_API_KEY": "fixture-key"}, "AnthropicBackend"),
    ],
)
def test_cli_uses_shared_provider_defaults(tmp_path, monkeypatch, environment, provider) -> None:
    from looplet import backends

    cart = _make_cartridge(tmp_path)
    selected = []

    def resolve(**kwargs):
        selected.append(kwargs)
        return MockLLMBackend(_scripted())

    monkeypatch.setattr(getattr(backends, provider), "from_env", resolve)
    with patch.dict(os.environ, environment, clear=True):
        assert eval_cli(["run", str(cart), "--json"]) == 0
    assert selected == [{"model": None}]


def test_cli_judge_model_uses_selected_provider(tmp_path, monkeypatch, capsys) -> None:
    cart = _make_cartridge(tmp_path)
    selected = []

    def resolve(*, model):
        selected.append(model)
        return MockLLMBackend(["0.8"] if model == "judge-model" else _scripted())

    monkeypatch.setattr("looplet.backends.make_backend", resolve)
    assert (
        eval_cli(
            ["run", str(cart), "--model", "agent-model", "--judge-model", "judge-model", "--json"]
        )
        == 0
    )
    assert selected == ["agent-model", "judge-model"]
    report = json.loads(capsys.readouterr().out)
    by_name = {item["name"]: item for item in report["cases"][0]["results"]}
    assert by_name["eval_judge_quality"]["score"] == 0.8


def test_cli_explicit_endpoint_applies_to_agent_and_judge(tmp_path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    selected = []

    def construct(**kwargs):
        selected.append(kwargs)
        return MockLLMBackend(["0.8"] if kwargs["model"] == "judge-model" else _scripted())

    monkeypatch.setattr("looplet.backends.OpenAIBackend", construct)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture-key"}, clear=True):
        assert (
            eval_cli(
                [
                    "run",
                    str(cart),
                    "--base-url",
                    "http://localhost:12345/v1",
                    "--model",
                    "agent-model",
                    "--judge-model",
                    "judge-model",
                    "--json",
                ]
            )
            == 0
        )
    assert selected == [
        {"base_url": "http://localhost:12345/v1", "api_key": "x", "model": "agent-model"},
        {"base_url": "http://localhost:12345/v1", "api_key": "x", "model": "judge-model"},
    ]
