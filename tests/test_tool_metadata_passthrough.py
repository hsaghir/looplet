"""Tests for the additive ``metadata`` dicts on ``ToolCall`` and ``ToolResult``.

External hooks can attach arbitrary annotations to a tool call (in
``pre_dispatch``) or a tool result (in ``post_dispatch``); the metadata
flows through ``to_dict()`` and into the saved trajectory under
``trajectory.steps[N].tool_call.metadata`` and
``trajectory.steps[N].tool_result.metadata`` respectively.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.provenance import TrajectoryRecorder
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec
from looplet.types import ToolCall, ToolResult


def test_tool_call_metadata_default_is_empty() -> None:
    tc = ToolCall(tool="x")
    assert tc.metadata == {}


def test_tool_result_metadata_default_is_empty() -> None:
    tr = ToolResult(tool="x", args_summary="", data=None)
    assert tr.metadata == {}


def test_tool_call_metadata_round_trips_through_to_dict() -> None:
    tc = ToolCall(tool="x", args={"k": 1})
    tc.metadata["ledger_node_id"] = "obs_001"
    tc.metadata["policy_version"] = 7
    d = tc.to_dict()
    assert d["metadata"] == {"ledger_node_id": "obs_001", "policy_version": 7}
    # Round-trip through JSON for safety
    assert json.loads(json.dumps(d))["metadata"]["ledger_node_id"] == "obs_001"


def test_tool_result_metadata_only_in_to_dict_when_non_empty() -> None:
    """ToolResult.to_dict() omits the metadata key when empty (kept compact
    for LLM context); includes it when the hook attached anything."""
    tr_empty = ToolResult(tool="x", args_summary="", data={"ok": True})
    assert "metadata" not in tr_empty.to_dict()

    tr_tagged = ToolResult(tool="x", args_summary="", data={"ok": True})
    tr_tagged.metadata["credit_score"] = 0.9
    d = tr_tagged.to_dict()
    assert d["metadata"] == {"credit_score": 0.9}


def test_tool_call_metadata_survives_in_saved_trajectory() -> None:
    """A pre_dispatch hook that tags the ToolCall sees the tag preserved
    in trajectory.json after the run."""

    class Tagger:
        def pre_dispatch(self, state, session_log, tool_call, step_num):
            tool_call.metadata["tag"] = f"step_{step_num}"
            return None

    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={"x": "str"},
            execute=lambda *, x: {"echoed": x},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="done",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        rec = TrajectoryRecorder(output_dir=out_dir)
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"echo","args":{"x":"hi"},"reasoning":""}',
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
                    ]
                ),
                tools=reg,
                state=DefaultState(max_steps=5),
                hooks=[Tagger(), rec],
                config=LoopConfig(max_steps=5),
            )
        )

        traj = json.loads((out_dir / "trajectory.json").read_text())
        # echo step should carry the tag set by Tagger.pre_dispatch
        echo_step = next(s for s in traj["steps"] if s["tool_call"]["tool"] == "echo")
        assert echo_step["tool_call"]["metadata"].get("tag") is not None


def test_tool_result_metadata_survives_in_saved_trajectory() -> None:
    """A post_dispatch hook that tags the ToolResult sees the tag preserved
    in trajectory.json after the run."""

    class Scorer:
        def post_dispatch(self, state, session_log, tool_call, tool_result, step_num):
            if tool_call.tool == "echo":
                tool_result.metadata["credit_score"] = 0.42
            return None

    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={"x": "str"},
            execute=lambda *, x: {"echoed": x},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="done",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        rec = TrajectoryRecorder(output_dir=out_dir)
        list(
            composable_loop(
                llm=MockLLMBackend(
                    responses=[
                        '{"tool":"echo","args":{"x":"hi"},"reasoning":""}',
                        '{"tool":"done","args":{"answer":"ok"},"reasoning":""}',
                    ]
                ),
                tools=reg,
                state=DefaultState(max_steps=5),
                hooks=[Scorer(), rec],
                config=LoopConfig(max_steps=5),
            )
        )

        traj = json.loads((out_dir / "trajectory.json").read_text())
        echo_step = next(s for s in traj["steps"] if s["tool_call"]["tool"] == "echo")
        # ToolResult.to_dict only emits 'metadata' when non-empty
        assert echo_step["tool_result"].get("metadata") == {"credit_score": 0.42}
