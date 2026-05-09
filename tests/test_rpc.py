"""Tests for looplet.rpc — stdio JSONL RPC mode.

Uses MockLLMBackend via a factory function so the loop is deterministic.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from looplet.rpc import RPCServer
from looplet.scaffold import scaffold_workspace
from looplet.testing import MockLLMBackend


# Module-level factory so RPC's _import_factory can find it via dotted path.
def make_mock_backend() -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            '{"tool": "greet", "args": {"name": "world"}, "reasoning": "test greet"}',
            '{"tool": "done", "args": {"answer": "hi"}, "reasoning": "finish"}',
        ]
    )


def _drive(commands: list[dict]) -> list[dict]:
    """Run RPCServer over an in-memory pipe; return parsed events."""
    in_buf = io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n")
    out_buf = io.StringIO()
    server = RPCServer(in_stream=in_buf, out_stream=out_buf)
    server.serve_forever()
    out_buf.seek(0)
    return [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]


def test_unknown_command_emits_error_and_continues() -> None:
    events = _drive(
        [
            {"cmd": "totally-bogus"},
            {"cmd": "quit"},
        ]
    )
    assert events[0]["event"] == "error"
    assert "unknown" in events[0]["message"]
    assert events[-1]["event"] == "ready" and events[-1].get("quit") is True


def test_bad_json_does_not_kill_server(tmp_path: Path) -> None:
    in_buf = io.StringIO("not-json\n" + json.dumps({"cmd": "quit"}) + "\n")
    out_buf = io.StringIO()
    RPCServer(in_stream=in_buf, out_stream=out_buf).serve_forever()
    out_buf.seek(0)
    events = [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]
    assert events[0]["event"] == "error" and "JSON" in events[0]["message"]
    assert events[-1]["event"] == "ready"


def test_run_without_workspace_errors() -> None:
    events = _drive(
        [
            {"cmd": "run", "task": {"goal": "x"}},
            {"cmd": "quit"},
        ]
    )
    assert any(e["event"] == "error" and "load_workspace" in e["message"] for e in events)


def test_full_run_streams_steps_and_done(tmp_path: Path) -> None:
    ws = scaffold_workspace(tmp_path / "w.workspace", name="w", tools=["greet"])
    # Replace the NotImplementedError stub with a real implementation
    (ws / "tools" / "greet" / "execute.py").write_text(
        "def execute(ctx, *, name):\n    return {'greeting': f'Hello, {name}!'}\n"
    )

    events = _drive(
        [
            {"cmd": "load_workspace", "path": str(ws), "runtime": {"workspace": str(tmp_path)}},
            {"cmd": "set_backend", "factory": "tests.test_rpc:make_mock_backend"},
            {"cmd": "run", "task": {"goal": "greet world"}, "max_steps": 5},
            {"cmd": "quit"},
        ]
    )

    by_event: dict[str, list[dict]] = {}
    for e in events:
        by_event.setdefault(e["event"], []).append(e)

    assert "ready" in by_event
    assert "step" in by_event
    assert any(s["step"]["tool_call"]["tool"] == "greet" for s in by_event["step"])
    assert any(s["step"]["tool_call"]["tool"] == "done" for s in by_event["step"])

    done = by_event["done"]
    assert len(done) == 1 and done[0]["steps"] >= 2


def test_set_backend_bad_spec_emits_error() -> None:
    events = _drive(
        [
            {"cmd": "set_backend", "factory": "no-colon-here"},
            {"cmd": "quit"},
        ]
    )
    assert any(e["event"] == "error" for e in events)
