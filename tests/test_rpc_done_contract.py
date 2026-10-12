"""Completion contract over the RPC wire (RPC §1.3).

The ``done`` event is the terminal frame of every ``run``. Per the
orchestration contract it carries::

    {"event": "done", "stop_reason": <StopReason>, "steps": N,
     "output": <obj|null>}

* ``stop_reason`` is a value from the frozen :class:`~looplet.rpc.StopReason`
  enum (``done|max_steps|budget|stagnated|cancelled|error``) - the same set
  advertised in the capability handshake's ``stop_reasons``.
* ``steps`` is retained for back-compat (number of step frames emitted).
* ``output`` is the structured payload of the accepted ``done()`` sentinel
  (``done_steps.done_output``), or ``null`` when the loop stopped without an
  accepted ``done()``.

These tests drive each termination path (done / max_steps / budget /
stagnated / cancelled / error) and assert it maps to the correct
``stop_reason``, plus that a ``done()`` (optionally gated by an
``output_schema``) surfaces its output on the wire.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from looplet.cartridge.scaffold import scaffold_cartridge
from looplet.done_steps import done_output, last_accepted_done
from looplet.hook_decision import HookDecision
from looplet.rpc import STOP_REASONS, RPCServer, StopReason, map_stop_reason
from looplet.testing import MockLLMBackend
from looplet.types import CancelToken, DefaultState, Step, ToolCall, ToolResult
from looplet.validation import FieldSpec, OutputSchema

# ── module-level backend factories (resolvable by dotted path) ───────


def make_done_backend() -> MockLLMBackend:
    return MockLLMBackend(
        responses=[
            '{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}',
            '{"tool": "done", "args": {"summary": "all set"}, "reasoning": "finish"}',
        ]
    )


def make_greet_only_backend() -> MockLLMBackend:
    # Never calls done(); cycles greet forever so the loop runs to max_steps.
    return MockLLMBackend(
        responses=['{"tool": "greet", "args": {"name": "world"}, "reasoning": "greet"}']
    )


def make_boom_backend() -> "_BoomBackend":
    return _BoomBackend()


class _BoomBackend:
    """Backend whose generate() raises - drives the error termination path."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        raise RuntimeError("backend boom")


# ── test hooks ───────────────────────────────────────────────────────


class _BudgetStopHook:
    """should_stop fires a budget-class stop after the first step."""

    def should_stop(self, state: Any, step_num: int, new_entities: int) -> HookDecision:
        return HookDecision(stop="budget_exceeded")


class _StagnationStopHook:
    """should_stop fires a stagnation-class stop after the first step."""

    def should_stop(self, state: Any, step_num: int, new_entities: int) -> HookDecision:
        return HookDecision(stop="stagnated")


class _CancelAfterFirstStep:
    """Cooperatively cancels the token in post_dispatch (mid-run cancel)."""

    def __init__(self, token: CancelToken) -> None:
        self._token = token

    def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        tool_result: Any,
        step_num: int,
    ) -> None:
        self._token.cancel()
        return None


# ── harness ──────────────────────────────────────────────────────────


def _scaffold(tmp_path: Path) -> Path:
    ws = scaffold_cartridge(tmp_path / "w.workspace", name="w", tools=["greet"])
    # Declare the `name` parameter so the dispatcher accepts greet (an empty
    # ``parameters: {}`` would reject every call with a VALIDATION error).
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


def _load(tmp_path: Path, *, backend: Any) -> tuple[RPCServer, io.StringIO]:
    ws = _scaffold(tmp_path)
    out = io.StringIO()
    server = RPCServer(in_stream=io.StringIO(""), out_stream=out)
    server.cmd_load_workspace({"path": str(ws), "runtime": {"workspace": str(tmp_path)}})
    server.backend = backend
    return server, out


def _events(out: io.StringIO) -> list[dict]:
    out.seek(0)
    return [json.loads(line) for line in out.read().splitlines() if line.strip()]


def _done_frame(out: io.StringIO) -> dict:
    frames = [e for e in _events(out) if e["event"] == "done"]
    assert len(frames) == 1, f"expected exactly one done frame, got {len(frames)}"
    return frames[0]


# ── AC-2: each termination path maps to the right stop_reason ────────


def test_done_sentinel_maps_to_done_and_surfaces_output(tmp_path: Path) -> None:
    server, out = _load(tmp_path, backend=make_done_backend())
    server.cmd_run({"task": {"goal": "greet world"}, "max_steps": 5})
    done = _done_frame(out)
    assert done["stop_reason"] == "done"
    # output is the accepted done() payload (scaffolded done returns this).
    assert done["output"] == {"summary": "all set", "done": True}
    # steps retained (back-compat).
    assert "steps" in done and done["steps"] >= 2


def test_max_steps_exhaustion_maps_to_max_steps(tmp_path: Path) -> None:
    server, out = _load(tmp_path, backend=make_greet_only_backend())
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 3})
    done = _done_frame(out)
    assert done["stop_reason"] == "max_steps"
    assert done["output"] is None
    assert done["steps"] == 3


def test_budget_hook_maps_to_budget(tmp_path: Path) -> None:
    server, out = _load(tmp_path, backend=make_greet_only_backend())
    server.preset.hooks.append(_BudgetStopHook())
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 5})
    done = _done_frame(out)
    assert done["stop_reason"] == "budget"
    assert done["steps"] >= 1


def test_stagnation_hook_maps_to_stagnated(tmp_path: Path) -> None:
    server, out = _load(tmp_path, backend=make_greet_only_backend())
    server.preset.hooks.append(_StagnationStopHook())
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 5})
    done = _done_frame(out)
    assert done["stop_reason"] == "stagnated"


def test_cancel_token_maps_to_cancelled(tmp_path: Path) -> None:
    token = CancelToken()
    server, out = _load(tmp_path, backend=make_greet_only_backend())
    server.preset.config.cancel_token = token
    server.preset.hooks.append(_CancelAfterFirstStep(token))
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 5})
    done = _done_frame(out)
    assert done["stop_reason"] == "cancelled"


def test_backend_error_maps_to_error_and_still_emits_done(tmp_path: Path) -> None:
    server, out = _load(tmp_path, backend=make_boom_backend())
    server.cmd_run({"task": {"goal": "greet"}, "max_steps": 5})
    events = _events(out)
    # The error frame stays (diagnostics), AND a terminal done frame is emitted.
    assert any(e["event"] == "error" for e in events)
    done = _done_frame(out)
    assert done["stop_reason"] == "error"
    assert done["output"] is None


# ── AC-1: stop_reason is always in the frozen enum ───────────────────


def test_every_done_stop_reason_is_in_frozen_enum(tmp_path: Path) -> None:
    server, out = _load(tmp_path, backend=make_done_backend())
    server.cmd_run({"task": {"goal": "greet world"}, "max_steps": 5})
    done = _done_frame(out)
    assert done["stop_reason"] in STOP_REASONS


def test_done_with_output_schema_surfaces_output(tmp_path: Path) -> None:
    # A done() gated by an output_schema: when the args validate, the loop
    # accepts the sentinel and the structured output reaches the wire.
    server, out = _load(tmp_path, backend=make_done_backend())
    server.preset.config.output_schema = OutputSchema(
        fields={"summary": FieldSpec(name="summary", field_type="str", required=True)}
    )
    server.cmd_run({"task": {"goal": "greet world"}, "max_steps": 5})
    done = _done_frame(out)
    assert done["stop_reason"] == "done"
    assert done["output"] == {"summary": "all set", "done": True}


def test_alternate_terminal_surfaces_the_same_accepted_output_as_python() -> None:
    from looplet import AgentPreset, LoopConfig, RunResult, tools_from

    def submit(answer: str) -> dict:
        return {"answer": answer}

    def build() -> AgentPreset:
        return AgentPreset(
            config=LoopConfig(max_steps=1, use_native_tools=False, done_tools=["submit"]),
            state=DefaultState(max_steps=1),
            tools=tools_from([submit]),
            hooks=[],
        )

    response = '{"tool":"submit","args":{"answer":"accepted"}}'
    with build() as direct:
        list(direct.run(MockLLMBackend([response]), task={"goal": "finish"}))
        expected = RunResult.from_state(direct.state)
        assert expected.completed

    with build() as preset:
        output = io.StringIO()
        server = RPCServer(
            in_stream=io.StringIO(""),
            out_stream=output,
            preset=preset,
            backend=MockLLMBackend([response]),
        )
        server.cmd_run({"task": {"goal": "finish"}, "max_steps": 1})

        terminal = _done_frame(output)
        assert terminal["stop_reason"] == "done"
        assert terminal["output"] == expected.output == {"answer": "accepted"}
        assert set(terminal) == {"event", "stop_reason", "steps", "output"}


def test_repeated_loaded_rpc_runs_use_fresh_state_and_preserve_wire(tmp_path):
    server, output = _load(tmp_path, backend=make_done_backend())
    original = server.preset
    try:
        server.cmd_run({"task": {"goal": "first"}})
        server.cmd_run({"task": {"goal": "second"}})
        assert server.preset is not original
        assert server.preset.state is not original.state
        assert original._closed
        terminals = [frame for frame in _events(output) if frame["event"] == "done"]
        assert len(terminals) == 2
        assert all(frame["stop_reason"] == "done" for frame in terminals)
        assert all(set(frame) == {"event", "stop_reason", "steps", "output"} for frame in terminals)
    finally:
        server._close_preset()


@pytest.mark.parametrize("wrong_type", [False, True])
def test_failed_preset_rebuild_emits_terminal_and_allows_retry(tmp_path, monkeypatch, wrong_type):
    server, output = _load(tmp_path, backend=make_done_backend())

    def rebuild(*args, **kwargs):
        if wrong_type:
            return object()
        raise RuntimeError("preset rebuild failed")

    try:
        server.cmd_run({"task": {"goal": "first"}})
        previous = server.preset
        with monkeypatch.context() as patch:
            patch.setattr("looplet.rpc.cartridge_to_preset", rebuild)
            server.cmd_run({"task": {"goal": "second"}})
        terminals = [frame for frame in _events(output) if frame["event"] == "done"]
        assert terminals == [
            {
                "event": "done",
                "stop_reason": "done",
                "steps": 2,
                "output": {"summary": "all set", "done": True},
            },
            {"event": "done", "stop_reason": "error", "steps": 0, "output": None},
        ]
        assert any(frame["event"] == "error" for frame in _events(output))
        assert server.preset is previous
        assert server._active_cancel_token is None
        server.cmd_run({"task": {"goal": "retry"}})
        terminals = [frame for frame in _events(output) if frame["event"] == "done"]
        assert len(terminals) == 3
        assert terminals[-1]["stop_reason"] == "done"
        assert previous._closed
    finally:
        server._close_preset()


def test_driver_close_error_still_stops_watchers_and_emits_terminal(monkeypatch):
    import threading

    from looplet import minimal_preset

    threads = []
    create_thread = threading.Thread

    def tracked_thread(*args, **kwargs):
        thread = create_thread(*args, **kwargs)
        threads.append(thread)
        return thread

    class BrokenClose:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise RuntimeError("driver close failed")

    monkeypatch.setattr("looplet.rpc.threading.Thread", tracked_thread)
    with minimal_preset() as preset:
        original_config = preset.config
        monkeypatch.setattr(preset, "run", lambda *args, **kwargs: BrokenClose())
        output = io.StringIO()
        server = RPCServer(
            in_stream=io.StringIO(""),
            out_stream=output,
            preset=preset,
            backend=MockLLMBackend(),
        )
        server.cmd_run({"task": {}, "max_steps": 2})
        assert _done_frame(output)["stop_reason"] == "error"
        assert any("driver close failed" in event.get("message", "") for event in _events(output))
        assert preset.config is original_config
        assert server._active_cancel_token is None
        assert all(not thread.is_alive() for thread in threads)


def test_repeated_rpc_factory_runs_have_fresh_state_resources_and_binding():
    from looplet import RunEnvelope, minimal_preset

    built = []
    closed = []
    bound = []

    class Resource:
        def close(self):
            closed.append(self)

    class Observer:
        def set_run_envelope(self, envelope):
            bound.append(envelope.run_id)

        def pre_loop(self, *args):
            pass

    def factory():
        preset = minimal_preset(max_steps=1)
        preset.config.use_native_tools = False
        preset.config.run_envelope = RunEnvelope(run_id=f"rpc-{len(built)}")
        preset.hooks.append(Observer())
        preset.owned_resources.append(Resource())
        built.append(preset)
        return preset

    output = io.StringIO()
    server = RPCServer(
        in_stream=io.StringIO(""),
        out_stream=output,
        preset=factory(),
        preset_factory=factory,
        backend=MockLLMBackend(['{"tool":"done","args":{"summary":"ok"}}']),
    )
    try:
        server.cmd_run({"task": {}})
        server.cmd_run({"task": {}})
        frames = [json.loads(line) for line in output.getvalue().splitlines()]
        terminals = [frame for frame in frames if frame["event"] == "done"]
        assert len(terminals) == 2
        assert all(frame["stop_reason"] == "done" and frame["steps"] == 1 for frame in terminals)
        assert len(built) == 2
        assert built[0].state is not built[1].state
        assert bound == ["rpc-0", "rpc-1"]
        assert len(closed) == 1
    finally:
        server._close_preset()
    assert len(closed) == 2


# ── StopReason enum + classifier unit tests ──────────────────────────


def test_stop_reasons_tuple_matches_enum() -> None:
    assert STOP_REASONS == tuple(r.value for r in StopReason)
    assert set(STOP_REASONS) == {
        "done",
        "max_steps",
        "budget",
        "stagnated",
        "cancelled",
        "error",
    }


def test_map_stop_reason_known_internal_values() -> None:
    assert map_stop_reason("done") is StopReason.DONE
    # The loop's natural step-budget exhaustion is the contract's max_steps.
    assert map_stop_reason("budget_exhausted") is StopReason.MAX_STEPS
    assert map_stop_reason(None) is StopReason.MAX_STEPS
    assert map_stop_reason("") is StopReason.MAX_STEPS
    assert map_stop_reason("cancelled") is StopReason.CANCELLED
    assert map_stop_reason("deadline_exceeded") is StopReason.CANCELLED


def test_map_stop_reason_hook_categories() -> None:
    # A resource/cost budget hook stop is distinct from step exhaustion.
    assert map_stop_reason("budget_exceeded") is StopReason.BUDGET
    assert map_stop_reason("token_budget") is StopReason.BUDGET
    assert map_stop_reason("stagnated") is StopReason.STAGNATED
    assert map_stop_reason("stagnation") is StopReason.STAGNATED


def test_map_stop_reason_error_flag() -> None:
    assert map_stop_reason(None, errored=True) is StopReason.ERROR
    assert map_stop_reason("done", errored=True) is StopReason.ERROR


# ── done_output helper ───────────────────────────────────────────────


def _accepted_done_state() -> DefaultState:
    state = DefaultState(max_steps=5)
    state.steps.append(
        Step(
            number=1,
            tool_call=ToolCall(tool="done", args={"summary": "ok"}, reasoning="", call_id="c1"),
            tool_result=ToolResult(
                tool="done", args_summary="", data={"summary": "ok", "done": True}, error=None
            ),
        )
    )
    return state


def test_done_output_returns_accepted_payload() -> None:
    state = _accepted_done_state()
    assert last_accepted_done(state) is not None
    assert done_output(state) == {"summary": "ok", "done": True}


def test_done_output_none_when_no_accepted_done() -> None:
    state = DefaultState(max_steps=5)
    assert done_output(state) is None
