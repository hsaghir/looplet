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
    EvalRunRecord,
    assert_evals_pass,
    eval_cli,
    eval_discover,
    eval_run,
    load_eval_run,
    run_cartridge_evals,
)
from looplet.testing import MockLLMBackend

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


def _make_cartridge(root: Path) -> Path:
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
    # evals slot
    (cart / "evals" / "cases").mkdir(parents=True, exist_ok=True)
    (cart / "evals" / "cases" / "make_greeting.json").write_text(json.dumps(_CASE))
    (cart / "evals" / "eval_correctness.py").write_text(_GRADERS)
    (cart / "evals" / "collect_outcome.py").write_text(_COLLECTOR)
    return cart


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
        pytest.param("return {'steps': 20}", False, False, None, True, id="metric-only"),
        pytest.param("return {'steps': 20}", True, False, None, False, id="required-metric"),
        pytest.param(
            "return EvalResult(metrics={'f1': 0.1})", False, False, None, True, id="passive-f1"
        ),
        pytest.param("return {'f1': 0.1}", False, False, None, False, id="legacy-f1-gate"),
        pytest.param("return True", False, True, None, True, id="optional-judge-skipped"),
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
    passed: bool,
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
    if passed:
        assert_evals_pass(reloaded.context, graders)
    else:
        with pytest.raises(AssertionError):
            assert_evals_pass(reloaded.context, graders)

    expected_exit = 0 if passed else 1
    assert eval_cli([str(tmp_path / "live"), "--evals", str(cart / "evals")]) == expected_exit
    capsys.readouterr()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://looplet.invalid/v1")
    monkeypatch.setattr(
        "looplet.backends.make_backend",
        lambda **kwargs: MockLLMBackend(responses=_scripted()),
    )
    assert eval_cli(["run", str(cart), "--out", str(tmp_path / "cli")]) == expected_exit
    capsys.readouterr()
    assert eval_cli(["run", str(cart), "--json"]) == expected_exit
    report = json.loads(capsys.readouterr().out)
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


def test_run_cartridge_evals_rejects_expected_inside_agent_task(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    case_path = cart / "evals" / "cases" / "make_greeting.json"
    case_data = json.loads(case_path.read_text())
    case_data["task"]["expected"] = {"file_written": True}
    case_path.write_text(json.dumps(case_data))
    with pytest.raises(ValueError, match="reserved grader data"):
        run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()))


def test_run_cartridge_evals_resets_persisted_workspace(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    out = tmp_path / "runs"
    stale = out / "make_greeting" / "workspace" / "stale.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("must disappear\n")

    run_cartridge_evals(
        cart,
        llm=MockLLMBackend(responses=_scripted()),
        output_dir=out,
    )
    assert not stale.exists()


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
    assert payload["version"] == 1
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
