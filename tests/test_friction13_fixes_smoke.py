"""Round-13 friction fixes: TrajectoryRecorder output_dir + Trajectory.task +
Step.to_dict() key consistency."""

from __future__ import annotations

import json
import tempfile

import pytest

from looplet.provenance import StepRecord, Trajectory, TrajectoryRecorder
from looplet.types import Step, ToolCall, ToolResult

pytestmark = pytest.mark.smoke


# ── TrajectoryRecorder output_dir auto-save ──────────────────────


class TestTrajectoryRecorderOutputDir:
    def test_output_dir_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            rec = TrajectoryRecorder(output_dir=d)
            assert rec._output_dir is not None

    def test_output_dir_none_by_default(self):
        rec = TrajectoryRecorder()
        assert rec._output_dir is None

    def test_auto_save_on_loop_end(self):
        with tempfile.TemporaryDirectory() as d:
            rec = TrajectoryRecorder(output_dir=d)
            # Simulate pre_loop + on_loop_end
            rec.pre_loop(None, None, None)
            rec.on_loop_end(None, None, None, None)
            traj_path = f"{d}/trajectory.json"
            assert open(traj_path).read()
            data = json.loads(open(traj_path).read())
            assert "run_id" in data


# ── Trajectory.task ──────────────────────────────────────────────


class TestTrajectoryTask:
    def test_task_in_to_dict(self):
        t = Trajectory(run_id="abc", started_at=0.0, task={"description": "test"})
        d = t.to_dict()
        assert d["task"] == {"description": "test"}

    def test_task_defaults_empty(self):
        t = Trajectory(run_id="abc", started_at=0.0)
        assert t.task == {}
        assert t.to_dict()["task"] == {}

    def test_recorder_captures_task_from_state(self):
        class FakeState:
            task = {"description": "hello"}
            steps = []

        rec = TrajectoryRecorder()
        rec.pre_loop(FakeState(), None, None)
        assert rec.trajectory.task == {"description": "hello"}


# ── Step.to_dict() key consistency ───────────────────────────────


class TestStepToDictKeys:
    def test_uses_tool_call_and_tool_result_keys(self):
        tc = ToolCall(tool="search", reasoning="r")
        tr = ToolResult(tool="search", args_summary="q=x", data=[])
        step = Step(number=1, tool_call=tc, tool_result=tr)
        d = step.to_dict()
        assert "tool_call" in d, (
            f"Step.to_dict() should use 'tool_call', got keys: {list(d.keys())}"
        )
        assert "tool_result" in d, (
            f"Step.to_dict() should use 'tool_result', got keys: {list(d.keys())}"
        )
        # Old keys should NOT be present
        assert "call" not in d
        assert "result" not in d

    def test_matches_step_record_keys(self):
        """Step.to_dict() and StepRecord.to_dict() should use the same keys
        for tool_call and tool_result so trajectory JSON is consistent."""
        sr = StepRecord(
            step_num=1,
            timestamp=0,
            duration_ms=0,
            pretty="",
            tool_call={"tool": "x"},
            tool_result={"tool": "x"},
        )
        sr_keys = set(sr.to_dict().keys())
        assert "tool_call" in sr_keys
        assert "tool_result" in sr_keys
