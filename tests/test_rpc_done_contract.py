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
