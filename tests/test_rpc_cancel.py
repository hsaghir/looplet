"""Mid-run cancel over the RPC wire (RPC §1.4).

A ``{"cmd": "cancel"}`` frame arriving on stdin *while a ``run`` is
streaming* must trip the loop's :class:`~looplet.types.CancelToken`
cooperatively: the in-flight run stops between turns and emits the
terminal ``done`` frame with ``stop_reason == "cancelled"`` (the §1.3
completion contract). No ``step``/``event`` frames may follow ``done``,
and the server's stdin reader thread must wind down cleanly (no leak,
no hang).

The crux these tests exercise: ``cmd_run`` runs ``composable_loop``
*synchronously* on the calling thread, so the cancel cannot be observed
by re-reading stdin on that same thread. The server reads stdin on a
small background reader thread during the run and flips the token the
loop already holds — ``composable_loop`` itself is never modified.

Determinism: a scripted backend blocks inside ``generate()`` at a chosen
turn so a known number of steps have streamed before the cancel is
injected. The test only releases the backend *after* it has confirmed
the token was tripped over the wire, so there is no race between the
cancel landing and the loop's between-turns cancel check.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from looplet.cartridge.scaffold import scaffold_cartridge
from looplet.rpc import RPCServer
from looplet.types import CancelToken

# ── scripted backend that pauses mid-run ─────────────────────────────


class _BlockingBackend:
    """Emits a ``greet`` tool call every turn (never ``done``) so the loop
    would otherwise run to ``max_steps``. On the ``block_call``-th
    ``generate()`` it signals ``reached`` and waits on ``release`` before
    returning — pinning the run mid-flight so a cancel can be injected at a
    known step boundary.
    """

    def __init__(self, block_call: int) -> None:
        self._block_call = block_call
        self.calls = 0
        self.reached = threading.Event()
        self.release = threading.Event()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        if self.calls == self._block_call:
            self.reached.set()
            # Wait for the test to confirm the cancel was processed.
            self.release.wait(timeout=5.0)
        return '{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}'


# ── harness ──────────────────────────────────────────────────────────


def _scaffold(tmp_path: Path) -> Path:
    ws = scaffold_cartridge(tmp_path / "w.workspace", name="w", tools=["greet"])
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


def _make_pipe_stream() -> tuple[Any, Any]:
    """Return ``(read_stream, write_stream)`` text wrappers over an OS pipe so
    the test thread can feed the server's reader thread incrementally."""
    r_fd, w_fd = os.pipe()
    read_stream = os.fdopen(r_fd, "r", encoding="utf-8")
    write_stream = os.fdopen(w_fd, "w", encoding="utf-8")
    return read_stream, write_stream


def _events(out: io.StringIO) -> list[dict]:
    out.seek(0)
    return [json.loads(line) for line in out.read().splitlines() if line.strip()]


# ── AC-1/AC-2: wire cancel trips the token and ends with cancelled ───


def test_cancel_command_mid_run_emits_cancelled(tmp_path: Path) -> None:
    """A pre-installed (orchestrator-supplied) token is tripped by the wire
    cancel; the run ends with ``stop_reason='cancelled'``."""
    ws = _scaffold(tmp_path)
    read_stream, write_stream = _make_pipe_stream()
    out = io.StringIO()
    server = RPCServer(in_stream=read_stream, out_stream=out)
    server.cmd_load_workspace({"path": str(ws), "runtime": {"workspace": str(tmp_path)}})

    block_call = 4  # pause while producing turn 4 → 3 steps already streamed
    backend = _BlockingBackend(block_call=block_call)
    server.backend = backend

    # Orchestrator pre-installs the cancel token; the server must reuse it.
    token = CancelToken()
    server.preset.config.cancel_token = token

    max_steps = 50
    runner = threading.Thread(
        target=server.cmd_run,
        args=({"task": {"goal": "greet"}, "max_steps": max_steps},),
        daemon=True,
    )
    runner.start()

    # Wait until the run is pinned mid-flight (3 steps streamed).
    assert backend.reached.wait(timeout=5.0), "backend never reached the block point"

    # Inject the cancel over the wire and wait for the server's reader thread
    # to observe it and trip the token — no direct token.cancel() here.
    write_stream.write(json.dumps({"cmd": "cancel"}) + "\n")
    write_stream.flush()

    deadline = time.time() + 5.0
    while not token.is_cancelled and time.time() < deadline:
        time.sleep(0.01)
    assert token.is_cancelled, "wire cancel did not trip the CancelToken"

    # Release the backend; the loop emits the in-flight step then stops at the
    # next between-turns cancel check.
    backend.release.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive(), "run did not terminate promptly after cancel"

    # Close the write end → reader thread sees EOF and exits cleanly.
    write_stream.close()
    reader = server._reader_thread  # noqa: SLF001 - white-box liveness check
    assert reader is not None
    reader.join(timeout=5.0)
    assert not reader.is_alive(), "stdin reader thread leaked / hung"
    read_stream.close()

    frames = _events(out)
    done = [f for f in frames if f["event"] == "done"]
    assert len(done) == 1, f"expected exactly one done frame, got {len(done)}"
    assert done[0]["stop_reason"] == "cancelled"

    # The run stopped well before exhausting the step budget.
    assert done[0]["steps"] < max_steps
    assert done[0]["steps"] == block_call  # 3 before block + 1 in-flight

    # No step/event frames after the terminal done.
    done_idx = frames.index(done[0])
    assert not any(f["event"] in ("step", "event") for f in frames[done_idx + 1 :]), (
        "frames emitted after the terminal done"
    )


def test_cancel_command_creates_token_when_none_preinstalled(tmp_path: Path) -> None:
    """When the preset has no cancel token, the server installs its own and the
    wire cancel still stops the run."""
    ws = _scaffold(tmp_path)
    read_stream, write_stream = _make_pipe_stream()
    out = io.StringIO()
    server = RPCServer(in_stream=read_stream, out_stream=out)
    server.cmd_load_workspace({"path": str(ws), "runtime": {"workspace": str(tmp_path)}})
    assert server.preset.config.cancel_token is None  # fresh cartridge

    backend = _BlockingBackend(block_call=3)
    server.backend = backend

    runner = threading.Thread(
        target=server.cmd_run,
        args=({"task": {"goal": "greet"}, "max_steps": 50},),
        daemon=True,
    )
    runner.start()
    assert backend.reached.wait(timeout=5.0)

    # The server-created token is exposed for the duration of the run.
    deadline = time.time() + 5.0
    while server._active_cancel_token is None and time.time() < deadline:  # noqa: SLF001
        time.sleep(0.01)
    token = server._active_cancel_token  # noqa: SLF001
    assert token is not None, "server did not install a cancel token"

    write_stream.write(json.dumps({"cmd": "cancel"}) + "\n")
    write_stream.flush()
    deadline = time.time() + 5.0
    while not token.is_cancelled and time.time() < deadline:
        time.sleep(0.01)
    assert token.is_cancelled

    backend.release.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive()

    write_stream.close()
    reader = server._reader_thread  # noqa: SLF001
    assert reader is not None
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    read_stream.close()

    done = [f for f in _events(out) if f["event"] == "done"]
    assert len(done) == 1
    assert done[0]["stop_reason"] == "cancelled"


def test_uncancelled_run_still_completes_and_reader_is_clean(tmp_path: Path) -> None:
    """The reader-thread machinery must not change the normal (no-cancel) path:
    a run that is never cancelled streams to ``done`` and the reader winds down
    cleanly on EOF."""
    ws = _scaffold(tmp_path)
    read_stream, write_stream = _make_pipe_stream()
    out = io.StringIO()
    server = RPCServer(in_stream=read_stream, out_stream=out)
    server.cmd_load_workspace({"path": str(ws), "runtime": {"workspace": str(tmp_path)}})

    # Backend that calls done() on the 2nd turn → clean DONE termination.
    from looplet.testing import MockLLMBackend

    server.backend = MockLLMBackend(
        responses=[
            '{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}',
            '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "finish"}',
        ]
    )
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 5})

    write_stream.close()
    reader = server._reader_thread  # noqa: SLF001
    assert reader is not None
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    read_stream.close()

    done = [f for f in _events(out) if f["event"] == "done"]
    assert len(done) == 1
    assert done[0]["stop_reason"] == "done"


def test_cancel_outside_run_is_harmless_noop() -> None:
    """A stray ``cancel`` with no active run must not crash the dispatch loop
    (it is consumed by the run's watcher when a run is active; outside a run it
    is a no-op)."""
    commands = [{"cmd": "cancel"}, {"cmd": "quit"}]
    in_buf = io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n")
    out_buf = io.StringIO()
    rc = RPCServer(in_stream=in_buf, out_stream=out_buf).serve_forever()
    assert rc == 0
    out_buf.seek(0)
    events = [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]
    # No error frame for the stray cancel; server reaches a clean quit.
    assert not any(e["event"] == "error" for e in events)
    assert events[-1]["event"] == "ready" and events[-1].get("quit") is True
