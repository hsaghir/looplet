"""Online-eval inspection, promotion to offline, and case seeding.

Closes the three follow-up gaps after per-case persistence:

* ``EvalHook.context`` — inspect an ONLINE run's trajectory / tool calls
  with the same :class:`EvalContext` surface used offline.
* ``promote_to_offline`` — turn a live online run into a durable offline
  fixture (no separate TrajectoryRecorder needed).
* ``seed_case_workspace`` — materialise a case's seed files, so a runner
  no longer hand-rolls sandbox seeding.

Driven with a MockLLMBackend + a real (tiny) tool registry so the loop
actually runs — no network, deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import (
    DefaultState,
    EvalCase,
    EvalContext,
    EvalHook,
    LoopConfig,
    composable_loop,
    eval_run,
    load_eval_run,
    promote_to_offline,
    seed_case_workspace,
    tool,
    tools_from,
)
from looplet.testing import MockLLMBackend


@tool(description="Echo a message back.")
def echo(*, msg: str) -> dict:
    return {"echoed": msg}


def _build_tools():
    return tools_from([echo], include_done=True, done_parameters={"answer": "final answer"})


def _scripted():
    return [
        json.dumps({"tool": "echo", "args": {"msg": "hello"}, "reasoning": "say hi"}),
        json.dumps({"tool": "done", "args": {"answer": "did it"}, "reasoning": "finish"}),
    ]


def _grader_completed(ctx):
    return ctx.completed


def _grader_used_echo(ctx):
    return "echo" in ctx.tool_sequence


def _run_online(collectors=None, metadata=None) -> EvalHook:
    """Drive a real loop with an EvalHook attached; return the hook."""
    hook = EvalHook(
        evaluators=[_grader_completed, _grader_used_echo],
        collectors=collectors or [],
    )
    for _ in composable_loop(
        llm=MockLLMBackend(responses=_scripted()),
        tools=_build_tools(),
        state=DefaultState(max_steps=5, metadata=dict(metadata or {})),
        config=LoopConfig(max_steps=5, use_native_tools=False),
        task={"goal": "echo hello then finish"},
        hooks=[hook],
    ):
        pass
    return hook


# ── online inspection ────────────────────────────────────────────


def test_evalhook_context_exposes_trajectory_online() -> None:
    hook = _run_online()
    ctx = hook.context
    assert isinstance(ctx, EvalContext)
    # Same inspection surface as offline: tool calls are visible.
    assert ctx.tool_sequence == ["echo", "done"]
    assert ctx.completed is True
    # And the same graders score it.
    online = {r.name: r.score for r in hook.results}
    assert online == {"_grader_completed": 1.0, "_grader_used_echo": 1.0}


def test_evalhook_context_none_before_run() -> None:
    hook = EvalHook(evaluators=[_grader_completed])
    assert hook.context is None


def test_evalhook_extracts_secondary_terminal_output() -> None:
    tools = tools_from(
        [],
        include_done=True,
        done_name="escalate",
        done_parameters={"blocked_on": "reason for escalation"},
    )
    hook = EvalHook(evaluators=[_grader_completed])
    list(
        composable_loop(
            llm=MockLLMBackend(
                responses=[
                    json.dumps(
                        {
                            "tool": "escalate",
                            "args": {"blocked_on": "approval"},
                            "reasoning": "blocked",
                        }
                    )
                ]
            ),
            tools=tools,
            state=DefaultState(max_steps=2),
            config=LoopConfig(
                max_steps=2,
                done_tool="resolve",
                done_tools=["escalate"],
                use_native_tools=False,
            ),
            task={"goal": "resolve or escalate"},
            hooks=[hook],
        )
    )
    assert hook.context is not None
    assert hook.context.completed is True
    assert hook.context.final_output == {"blocked_on": "approval"}


# ── promote online → offline ─────────────────────────────────────


def test_promote_to_offline_roundtrips(tmp_path: Path) -> None:
    def collect(state) -> dict:
        return {"tests_passing": True}

    hook = _run_online(collectors=[collect], metadata={"trial": "candidate-a"})
    case = EvalCase(id="echo_case", task={"goal": "echo hello then finish"})

    out = promote_to_offline(tmp_path / "promoted", eval_hook=hook, case=case)

    # The promoted run is indistinguishable from an offline-captured one.
    assert (out / "trajectory.json").is_file()
    assert (out / "artifacts.json").is_file()
    assert (out / "evals.json").is_file()
    assert (out / "case.json").is_file()

    rec = load_eval_run(out)
    # Trajectory + tool calls survived the promotion.
    assert rec.context.tool_sequence == ["echo", "done"]
    assert rec.context.completed is True
    # Outcome artifacts survived.
    assert rec.context.artifacts == {"tests_passing": True}
    # Judge-visible evidence and run metadata survive online → offline.
    assert hook.context is not None
    assert hook.context.session_log_text
    assert rec.context.session_log_text == hook.context.session_log_text
    assert rec.context.metadata == hook.context.metadata
    # Case survived.
    assert rec.case.id == "echo_case"
    # Online scores survived.
    assert {r.name: r.score for r in rec.results} == {
        "_grader_completed": 1.0,
        "_grader_used_echo": 1.0,
    }


def test_promoted_run_regrades_identically_offline(tmp_path: Path) -> None:
    hook = _run_online()
    out = promote_to_offline(tmp_path / "p", eval_hook=hook)
    rec = load_eval_run(out)
    # The SAME graders score the promoted (offline) run identically.
    offline = eval_run([_grader_completed, _grader_used_echo], rec.context)
    online = {r.name: r.score for r in hook.results}
    assert {r.name: r.score for r in offline} == online


def test_promote_without_run_raises() -> None:
    hook = EvalHook(evaluators=[_grader_completed])
    with pytest.raises(ValueError):
        promote_to_offline("unused", eval_hook=hook)


def test_save_eval_run_from_context_directly(tmp_path: Path) -> None:
    from looplet import save_eval_run

    hook = _run_online(
        collectors=[lambda state: {"from_context": True}],
        metadata={"source": "live"},
    )
    # context= source path (no recorder, no eval_hook).
    out = save_eval_run(tmp_path / "c", context=hook.context)
    rec = load_eval_run(out)
    assert rec.context.tool_sequence == ["echo", "done"]
    assert rec.context.artifacts == {"from_context": True}
    assert rec.context.metadata["source"] == "live"
    assert rec.context.session_log_text == hook.context.session_log_text


def test_save_eval_run_no_source_raises(tmp_path: Path) -> None:
    from looplet import save_eval_run

    with pytest.raises(ValueError):
        save_eval_run(tmp_path / "x")


# ── seed_case_workspace ──────────────────────────────────────────


def test_seed_case_workspace_writes_files(tmp_path: Path) -> None:
    case = EvalCase(
        id="seeded",
        task={
            "goal": "fix it",
            "files": {
                "mod.py": "x = 1\n",
                "pkg/sub.py": "y = 2\n",
                "test_mod.py": "from mod import x\n",
            },
        },
    )
    out = seed_case_workspace(case, tmp_path / "sb")
    assert (out / "mod.py").read_text() == "x = 1\n"
    assert (out / "pkg" / "sub.py").read_text() == "y = 2\n"  # nested path created
    assert (out / "test_mod.py").is_file()


def test_seed_case_workspace_no_files_is_noop(tmp_path: Path) -> None:
    case = EvalCase(id="nofiles", task={"goal": "operate on existing repo"})
    out = seed_case_workspace(case, tmp_path / "sb")
    assert out.is_dir()
    assert list(out.iterdir()) == []


def test_seed_case_workspace_bad_files_type_raises(tmp_path: Path) -> None:
    case = EvalCase(id="bad", task={"files": ["not", "a", "dict"]})
    with pytest.raises(ValueError):
        seed_case_workspace(case, tmp_path / "sb")


@pytest.mark.parametrize(
    "case_id",
    ["", ".", "..", "../escape", "nested/case", "a\\b", "line\nbreak", "nul\x00byte"],
)
def test_eval_case_rejects_unsafe_id(case_id: str) -> None:
    with pytest.raises(ValueError, match="EvalCase id"):
        EvalCase(id=case_id)


@pytest.mark.parametrize("seed_path", ["../escape.py", "/tmp/escape.py", "a\\b.py"])
def test_seed_case_workspace_rejects_escape(tmp_path: Path, seed_path: str) -> None:
    case = EvalCase(id="unsafe_seed", task={"files": {seed_path: "bad\n"}})
    with pytest.raises(ValueError, match="seed path"):
        seed_case_workspace(case, tmp_path / "sb")


def test_seed_case_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    (sandbox / "linked").symlink_to(outside, target_is_directory=True)
    case = EvalCase(id="symlink_seed", task={"files": {"linked/escape.py": "bad\n"}})
    with pytest.raises(ValueError, match="seed path"):
        seed_case_workspace(case, sandbox)
