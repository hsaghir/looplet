"""Tests for the full LifecycleEvent stream over the RPC wire (RPC §1.2).

During ``run`` the server forwards every in-process
:class:`~looplet.events.LifecycleEvent` to stdout as an ``event`` frame::

    {"event": "event", "kind": <LifecycleEvent.value>, "step_num": N,
     "payload": {...}}

in addition to the existing ``step`` frames (which stay byte-for-byte
unchanged). The forwarder subscribes through the existing ``on_event``
hook bus — a pure observer hook appended to the run's hook list — so
``composable_loop``'s core is untouched. A small, safe serialiser
(:meth:`EventPayload.to_jsonable`) drops non-JSON objects and never
raises into the loop.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from looplet.cartridge import cartridge_to_preset
from looplet.cartridge.scaffold import scaffold_cartridge
from looplet.events import EventPayload, LifecycleEvent
from looplet.rpc import RPCServer
from looplet.testing import MockLLMBackend


# Module-level factory so RPC's _import_factory can resolve it by dotted path.
def make_mock_backend() -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            '{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}',
            '{"tool": "done", "args": {"summary": "hi"}, "reasoning": "finish"}',
        ]
    )


def _drive(commands: list[dict]) -> list[dict]:
    """Run RPCServer over an in-memory pipe; return parsed events."""
    in_buf = io.StringIO("\n".join(json.dumps(c) for c in commands) + "\n")
    out_buf = io.StringIO()
    RPCServer(in_stream=in_buf, out_stream=out_buf).serve_forever()
    out_buf.seek(0)
    return [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]


def _scaffold(tmp_path: Path) -> Path:
    ws = scaffold_cartridge(tmp_path / "w.workspace", name="w", tools=["greet"])
    # Declare the `name` parameter so the dispatcher accepts the call (an
    # empty ``parameters: {}`` would reject every greet with a VALIDATION
    # error, firing POST_TOOL_FAILURE instead of POST_TOOL_USE).
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


def _run_events(tmp_path: Path) -> list[dict]:
    ws = _scaffold(tmp_path)
    return _drive(
        [
            {"cmd": "load_workspace", "path": str(ws), "runtime": {"workspace": str(tmp_path)}},
            {"cmd": "set_backend", "factory": "tests.test_rpc_event_stream:make_mock_backend"},
            {"cmd": "run", "task": {"goal": "greet world"}, "max_steps": 5},
            {"cmd": "quit"},
        ]
    )


# ── AC-1: every LifecycleEvent arrives as an `event` frame ───────────


def test_lifecycle_events_stream_as_event_frames(tmp_path: Path) -> None:
    events = _run_events(tmp_path)
    event_frames = [e for e in events if e["event"] == "event"]
    assert event_frames, "no `event` frames were streamed during run"

    kinds = {e["kind"] for e in event_frames}
    # The scripted greet→done run exercises a tool dispatch and a done()
    # acceptance, so at minimum these lifecycle events must be forwarded.
    for required in (
        LifecycleEvent.SESSION_START.value,
        LifecycleEvent.PRE_TOOL_USE.value,
        LifecycleEvent.POST_TOOL_USE.value,
        LifecycleEvent.DONE_ACCEPTED.value,
        LifecycleEvent.STOP.value,
    ):
        assert required in kinds, f"missing lifecycle event {required!r}; got {sorted(kinds)}"


def test_every_event_frame_has_kind_step_num_payload(tmp_path: Path) -> None:
    events = _run_events(tmp_path)
    event_frames = [e for e in events if e["event"] == "event"]
    assert event_frames
    valid_kinds = set(LifecycleEvent.__members__.values())
    valid_kind_values = {k.value for k in valid_kinds}
    for frame in event_frames:
        assert frame["event"] == "event"
        assert frame["kind"] in valid_kind_values
        assert isinstance(frame["step_num"], int)
        assert isinstance(frame["payload"], dict)
        # The whole frame must round-trip through JSON (it already did to
        # be parsed, but assert explicitly for the contract).
        json.dumps(frame)


# ── AC-2: existing `step` frames unchanged; serialisation safe ───────


def test_step_frames_unchanged_alongside_events(tmp_path: Path) -> None:
    events = _run_events(tmp_path)
    step_frames = [e for e in events if e["event"] == "step"]
    assert step_frames, "step frames must still be emitted"
    # Exact legacy shape: {event, step_num, step:{...with tool_call...}}.
    for s in step_frames:
        assert set(s) == {"event", "step_num", "step"}
        assert isinstance(s["step_num"], int)
        assert "tool_call" in s["step"]
    tools = {s["step"]["tool_call"]["tool"] for s in step_frames}
    assert {"greet", "done"} <= tools

    done = [e for e in events if e["event"] == "done"]
    assert len(done) == 1 and done[0]["steps"] >= 2


def test_hook_decision_event_streamed(tmp_path: Path) -> None:
    """A non-noop HookDecision surfaces as a ``hook_decision`` event frame."""
    from looplet import InjectContext

    class _InjectOnce:
        def __init__(self) -> None:
            self._fired = False

        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            if not self._fired and tool_call.tool == "greet":
                self._fired = True
                return InjectContext("remember to finish")
            return None

    ws = _scaffold(tmp_path)
    out_buf = io.StringIO()
    server = RPCServer(in_stream=io.StringIO(), out_stream=out_buf)
    server.cmd_load_workspace({"path": str(ws), "runtime": {"workspace": str(tmp_path)}})
    server.preset.hooks.append(_InjectOnce())
    server.backend = make_mock_backend()
    server.cmd_run({"task": {"goal": "greet world"}, "max_steps": 5})

    out_buf.seek(0)
    frames = [json.loads(line) for line in out_buf.read().splitlines() if line.strip()]
    hook_decisions = [
        f
        for f in frames
        if f["event"] == "event" and f["kind"] == LifecycleEvent.HOOK_DECISION.value
    ]
    assert hook_decisions, "a non-noop HookDecision must stream as a hook_decision event"
    # The decision detail rides in the payload's extra dict.
    assert any("decision" in hd["payload"].get("extra", {}) for hd in hook_decisions)


# ── EventPayload.to_jsonable safe-serialiser unit tests ──────────────


def test_event_payload_to_jsonable_never_raises_and_drops_non_json() -> None:
    class _Unserialisable:
        pass

    payload = EventPayload(
        event=LifecycleEvent.PRE_TOOL_USE,
        step_num=3,
        prompt="hello",
        raw_response=_Unserialisable(),  # non-JSON, no to_dict → dropped
        extra={"ok": 1, "bad": _Unserialisable(), "nested": {"deep": _Unserialisable()}},
    )
    data = payload.to_jsonable()
    assert isinstance(data, dict)
    # Whole thing must be JSON serialisable (the point of the serialiser).
    json.dumps(data)
    assert data["prompt"] == "hello"
    assert "raw_response" not in data  # dropped — not JSON, no to_dict
    assert data["extra"]["ok"] == 1
    assert "bad" not in data["extra"]
    assert "deep" not in data["extra"]["nested"]


def test_event_payload_to_jsonable_excludes_framework_objects() -> None:
    from looplet.types import ToolCall

    payload = EventPayload(
        event=LifecycleEvent.PRE_TOOL_USE,
        step_num=1,
        state=object(),
        session_log=object(),
        context=object(),
        tool_call=ToolCall(tool="greet", args={"name": "x"}, reasoning="r", call_id="c1"),
    )
    data = payload.to_jsonable()
    # Loop-internal object fields are never serialised.
    assert "state" not in data
    assert "session_log" not in data
    assert "context" not in data
    # Objects exposing to_dict are serialised through it.
    assert data["tool_call"]["tool"] == "greet"
    json.dumps(data)
