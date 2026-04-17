"""Smoke tests for replay_loop and the `python -m openharness show` CLI."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from openharness import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    ProvenanceSink,
    composable_loop,
    replay_loop,
)
from openharness.__main__ import main as cli_main
from openharness.testing import MockLLMBackend
from openharness.tools import ToolSpec


def _make_tools() -> BaseToolRegistry:
    tools = BaseToolRegistry()
    tools.register(ToolSpec(
        name="add",
        description="Add two integers",
        parameters={"a": "int", "b": "int"},
        execute=lambda *, a, b: {"sum": a + b},
    ))
    tools.register(ToolSpec(
        name="done",
        description="Finish",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return tools


def _captured_dir(tmp_path: Path) -> Path:
    """Capture a small scripted run and return the trace directory."""
    responses = [
        '{"tool":"add","args":{"a":1,"b":2},"reasoning":"s1"}',
        '{"tool":"add","args":{"a":3,"b":4},"reasoning":"s2"}',
        '{"tool":"done","args":{"answer":"ok"},"reasoning":"end"}',
    ]
    sink = ProvenanceSink(dir=tmp_path / "run_1")
    llm = sink.wrap_llm(MockLLMBackend(responses=responses))
    for _ in composable_loop(
        llm=llm,
        tools=_make_tools(),
        state=DefaultState(max_steps=5),
        hooks=[sink.trajectory_hook()],
        config=LoopConfig(max_steps=5),
    ):
        pass
    return sink.flush()


class TestReplayLoopSmoke:
    def test_replays_captured_run(self, tmp_path: Path):
        trace_dir = _captured_dir(tmp_path)
        out_steps = []
        for step in replay_loop(trace_dir, tools=_make_tools()):
            out_steps.append(step)
        assert [s.tool_call.tool for s in out_steps] == ["add", "add", "done"]
        # Same totals as the captured run.
        assert out_steps[-1].tool_call.tool == "done"

    def test_missing_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(replay_loop(tmp_path / "does-not-exist", tools=_make_tools()))

    def test_empty_dir_raises(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            list(replay_loop(tmp_path / "empty", tools=_make_tools()))

    def test_hooks_fire_during_replay(self, tmp_path: Path):
        trace_dir = _captured_dir(tmp_path)

        class Counter:
            def __init__(self):
                self.post = 0

            def post_dispatch(
                self, state, session_log, tool_call, tool_result, step_num,
            ):
                self.post += 1
                return None

        counter = Counter()
        steps = list(replay_loop(trace_dir, tools=_make_tools(), hooks=[counter]))
        assert counter.post == len(steps) - 1  # done step skips post_dispatch

    def test_fallback_without_manifest(self, tmp_path: Path):
        """Replay should work from call_NN_response.txt even without manifest."""
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "manifest.jsonl").unlink()
        steps = list(replay_loop(trace_dir, tools=_make_tools()))
        assert len(steps) == 3


class TestShowCLISmoke:
    def test_show_renders_summary(self, tmp_path: Path, capsys):
        trace_dir = _captured_dir(tmp_path)
        rc = cli_main(["show", str(trace_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "done" in out
        assert "3 steps" in out
        assert "LLM:" in out
        assert "add" in out

    def test_show_missing_dir(self, tmp_path: Path, capsys):
        rc = cli_main(["show", str(tmp_path / "no-such-dir")])
        assert rc == 1
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_show_empty_dir(self, tmp_path: Path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = cli_main(["show", str(empty)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not a trace directory" in err

    def test_show_tolerates_missing_trajectory(self, tmp_path: Path, capsys):
        """With only manifest.jsonl present, `show` still prints LLM summary."""
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "trajectory.json").unlink()
        rc = cli_main(["show", str(trace_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "LLM:" in out
