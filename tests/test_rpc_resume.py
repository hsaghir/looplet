"""Checkpoint + resume over the RPC wire (RPC §1.5).

The RPC server makes the loop's existing crash-resume machinery
(:mod:`looplet.checkpoint`) drivable out-of-process:

* When a ``run``/``resume`` is given a ``checkpoint_dir``, the loop saves a
  full :class:`~looplet.checkpoint.Checkpoint` after every step (session log
  + conversation + config snapshot, serialised verbatim by
  :class:`~looplet.checkpoint.FileCheckpointStore`). The server surfaces each
  write as a ``{"event":"checkpoint","id":"step_N","step_num":N,"path":...}``
  frame so an orchestrator can record restart points.
* ``{"cmd":"resume","checkpoint":"<id|path>","task":{...}}`` loads a saved
  checkpoint and starts a run with it as the loop's ``initial_checkpoint`` —
  the continuation picks up at step ``N+1`` with the prior session log
  restored, even in a brand-new server process.

These tests drive a multi-step scripted-backend run with checkpointing, capture
a ``checkpoint`` frame, spawn a FRESH :class:`~looplet.rpc.RPCServer`, resume
from that checkpoint, and assert the continuation resumes at ``N+1`` with
preserved session state (reusing ``checkpoint.py`` across the boundary).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from looplet.cartridge.scaffold import scaffold_cartridge
from looplet.checkpoint import FileCheckpointStore
from looplet.rpc import RPCServer
from looplet.testing import MockLLMBackend

# ── module-level backend factories (resolvable by dotted path) ───────


def make_greet_only_backend() -> MockLLMBackend:
    # Never calls done(); a single scripted response cycles greet forever so
    # the loop runs to its max_steps budget (deterministic step count).
    return MockLLMBackend(
        responses=['{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}']
    )


def make_greet_then_done_backend() -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            '{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}',
            '{"tool": "done", "args": {"summary": "resumed and finished"}, "reasoning": "done"}',
        ]
    )


# ── harness ──────────────────────────────────────────────────────────


def _scaffold(tmp_path: Path) -> Path:
    ws = scaffold_cartridge(tmp_path / "w.workspace", name="w", tools=["greet"])
    # A real parameters block so the dispatcher accepts greet (an empty
    # ``parameters: {}`` rejects every call with a VALIDATION error).
    (ws / "tools" / "greet" / "tool.yaml").write_text(
        "name: greet\n"
        "description: |-\n"
        "  Greet someone by name.\n"
        "parameters:\n"
        "  name:\n"
        "    type: string\n"
        "    description: The name to greet.\n"
    )
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name):\n    return {'greeting': f'Hi {name}'}\n"
    )
    return ws


def _server(ws: Path, tmp_path: Path, *, backend: Any) -> tuple[RPCServer, io.StringIO]:
    """A freshly loaded RPCServer — models a distinct process per call."""
    out = io.StringIO()
    server = RPCServer(in_stream=io.StringIO(""), out_stream=out)
    server.cmd_load_workspace({"path": str(ws), "runtime": {"workspace": str(tmp_path)}})
    server.backend = backend
    return server, out


def _events(out: io.StringIO) -> list[dict]:
    out.seek(0)
    return [json.loads(line) for line in out.read().splitlines() if line.strip()]


def _frames(out: io.StringIO, event: str) -> list[dict]:
    return [e for e in _events(out) if e.get("event") == event]


# ── AC-1: checkpoint frames emitted on checkpoint writes ─────────────


def test_run_emits_checkpoint_frames(tmp_path: Path) -> None:
    ck = tmp_path / "ckpts"
    ws = _scaffold(tmp_path)
    server, out = _server(ws, tmp_path, backend=make_greet_only_backend())
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 3, "checkpoint_dir": str(ck)})

    ckpts = _frames(out, "checkpoint")
    assert ckpts, "expected at least one checkpoint frame when checkpoint_dir is set"
    # Each frame carries an id, a step_num and a resolvable path.
    step_nums = sorted(f["step_num"] for f in ckpts)
    assert step_nums == [1, 2, 3], f"expected checkpoints for steps 1..3, got {step_nums}"
    for f in ckpts:
        assert f["id"] == f"step_{f['step_num']}"
        assert Path(f["path"]).is_file(), f"checkpoint path should exist: {f['path']}"

    # The terminal done frame still arrives exactly once, after the checkpoints.
    done = _frames(out, "done")
    assert len(done) == 1
    order = [e["event"] for e in _events(out)]
    assert order.index("done") == len(order) - 1, "done must be the final frame"


def test_no_checkpoint_frames_without_checkpoint_dir(tmp_path: Path) -> None:
    # Back-compat: a plain run (no checkpoint_dir) emits NO checkpoint frames.
    ws = _scaffold(tmp_path)
    server, out = _server(ws, tmp_path, backend=make_greet_then_done_backend())
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 5})
    assert _frames(out, "checkpoint") == []
    assert len(_frames(out, "done")) == 1


# ── AC-1/AC-2: resume in a fresh process continues at N+1, state preserved ──


def test_resume_from_checkpoint_continues_at_next_step(tmp_path: Path) -> None:
    ck = tmp_path / "ckpts"
    ws = _scaffold(tmp_path)

    # --- original run (process #1): two greet steps, hits max_steps=2 ---
    server1, out1 = _server(ws, tmp_path, backend=make_greet_only_backend())
    server1.cmd_run({"task": {"goal": "greet"}, "max_steps": 2, "checkpoint_dir": str(ck)})
    ckpts1 = _frames(out1, "checkpoint")
    assert {f["step_num"] for f in ckpts1} == {1, 2}
    last = max(ckpts1, key=lambda f: f["step_num"])
    assert last["step_num"] == 2
    checkpoint_ref = last["path"]  # cross-process: resume by the on-disk path

    # --- fresh process (process #2): resume from the step-2 checkpoint ---
    server2, out2 = _server(ws, tmp_path, backend=make_greet_then_done_backend())
    server2.cmd_resume(
        {
            "checkpoint": checkpoint_ref,
            "task": {"goal": "greet"},
            "max_steps": 4,
            "checkpoint_dir": str(ck),
        }
    )

    steps2 = _frames(out2, "step")
    assert steps2, "resume produced no steps"
    assert steps2[0]["step_num"] == 3, (
        f"resume must pick up at step N+1 (=3), got {steps2[0]['step_num']}"
    )
    done2 = _frames(out2, "done")
    assert len(done2) == 1
    assert done2[0]["stop_reason"] == "done"

    # session/log/step state preserved across the process boundary: a
    # checkpoint written by the resumed run includes the ORIGINAL entries
    # (steps 1 & 2) plus the continuation — proving checkpoint.py round-tripped.
    latest = FileCheckpointStore(str(ck)).load_latest()
    assert latest is not None
    entries = latest.session_log_data.get("entries", [])
    steps_in_log = [e["step"] for e in entries]
    assert steps_in_log[:2] == [1, 2], f"original session entries lost: {steps_in_log}"
    assert max(steps_in_log) >= 3, "resumed continuation not recorded in the session log"


def test_resume_over_the_wire_via_serve_forever(tmp_path: Path) -> None:
    # Same as above but driven entirely through the JSONL command stream, so
    # `resume` is exercised as a real inbound wire command end-to-end.
    ck = tmp_path / "ckpts"
    ws = _scaffold(tmp_path)

    server1, out1 = _server(ws, tmp_path, backend=make_greet_only_backend())
    server1.cmd_run({"task": {"goal": "greet"}, "max_steps": 2, "checkpoint_dir": str(ck)})
    last = max(_frames(out1, "checkpoint"), key=lambda f: f["step_num"])
    checkpoint_ref = last["path"]

    commands = [
        {"cmd": "load_workspace", "path": str(ws), "runtime": {"workspace": str(tmp_path)}},
        {"cmd": "set_backend", "factory": "tests.test_rpc_resume:make_greet_then_done_backend"},
        {
            "cmd": "resume",
            "checkpoint": checkpoint_ref,
            "task": {"goal": "greet"},
            "max_steps": 4,
            "checkpoint_dir": str(ck),
        },
        {"cmd": "quit"},
    ]
    in_buf = io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n")
    out = io.StringIO()
    RPCServer(in_stream=in_buf, out_stream=out).serve_forever()

    steps = _frames(out, "step")
    assert steps and steps[0]["step_num"] == 3
    done = _frames(out, "done")
    assert len(done) == 1 and done[0]["stop_reason"] == "done"


def test_resume_by_id_against_checkpoint_dir(tmp_path: Path) -> None:
    # Resume accepts a bare checkpoint id (resolved against checkpoint_dir via
    # FileCheckpointStore), not only an absolute path.
    ck = tmp_path / "ckpts"
    ws = _scaffold(tmp_path)
    server1, out1 = _server(ws, tmp_path, backend=make_greet_only_backend())
    server1.cmd_run({"task": {"goal": "greet"}, "max_steps": 2, "checkpoint_dir": str(ck)})

    server2, out2 = _server(ws, tmp_path, backend=make_greet_then_done_backend())
    server2.cmd_resume(
        {
            "checkpoint": "step_2",  # bare id
            "task": {"goal": "greet"},
            "max_steps": 4,
            "checkpoint_dir": str(ck),
        }
    )
    steps2 = _frames(out2, "step")
    assert steps2 and steps2[0]["step_num"] == 3


def test_resume_missing_checkpoint_raises(tmp_path: Path) -> None:
    ws = _scaffold(tmp_path)
    server, _out = _server(ws, tmp_path, backend=make_greet_then_done_backend())
    try:
        server.cmd_resume(
            {"checkpoint": "does_not_exist", "task": {}, "checkpoint_dir": str(tmp_path / "nope")}
        )
    except ValueError:
        pass
    else:
        raise AssertionError("resume from a missing checkpoint must raise ValueError")
