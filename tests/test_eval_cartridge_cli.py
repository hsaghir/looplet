"""End-to-end: run a cartridge against its shipped evals (the CLI core).

``run_cartridge_evals`` ties the whole "evals ship with the agent
version" story together — seed each case's sandbox, run the cartridge as
a live agent with an online :class:`EvalHook`, grade, and optionally
persist. Driven here with a :class:`MockLLMBackend` and a minimal
in-test cartridge so it is fully deterministic (no network). Also covers
the ``looplet eval run`` CLI preflight/error paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from looplet import (
    EvalRunRecord,
    eval_cli,
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
def eval_completed(ctx):
    return ctx.completed


def eval_wrote_file(ctx):
    return "write_file" in ctx.tool_sequence


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
    return {"file_written": (root / "greeting.py").exists()}
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
    # collector populated artifacts from the seeded+written sandbox
    assert rec.context.artifacts.get("file_written") is True

    # persisted as an offline fixture, reloadable + re-gradable
    persisted = out / "make_greeting"
    assert (persisted / "trajectory.json").is_file()
    assert (persisted / "artifacts.json").is_file()
    reloaded = load_eval_run(persisted)
    assert reloaded.context.tool_sequence == ["write_file", "done"]


def test_run_cartridge_evals_no_output_uses_tempdir(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    records = run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()))
    assert len(records) == 1
    # sandbox dir exists and was seeded.
    assert (records[0].directory / "test_greeting.py").is_file()


def test_run_cartridge_evals_case_filter(tmp_path: Path) -> None:
    cart = _make_cartridge(tmp_path)
    # filter to a non-existent id → zero runs
    records = run_cartridge_evals(cart, llm=MockLLMBackend(responses=_scripted()), cases=["nope"])
    assert records == []


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

    monkeypatch.setattr(_backends, "OpenAIBackend", lambda **kw: shared)
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

    monkeypatch.setattr(_backends, "OpenAIBackend", lambda **kw: shared)
    out = tmp_path / "runs"
    rc = eval_cli(["run", str(cart), "--out", str(out)])
    assert rc == 0
    reloaded = load_eval_run(out / "make_greeting")
    by = {r.name: r for r in reloaded.results}
    assert by["eval_judge_quality"].label == "skipped"


# ── CLI preflight / error paths (no live model needed) ───────────


def test_cli_run_missing_cartridge() -> None:
    assert eval_cli(["run", "/no/such/cartridge"]) == 1


def test_cli_run_no_evals_dir(tmp_path: Path, monkeypatch) -> None:
    # A cartridge directory with no evals/ → error.
    cart = tmp_path / "empty.cartridge"
    cart.mkdir()
    (cart / "cartridge.json").write_text('{"name": "x", "schema_version": 2}')
    assert eval_cli(["run", str(cart)]) == 1


def test_cli_run_no_backend_configured(tmp_path: Path, monkeypatch) -> None:
    cart = _make_cartridge(tmp_path)
    for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    # cartridge is valid, but no model is configured → preflight error.
    assert eval_cli(["run", str(cart)]) == 1
