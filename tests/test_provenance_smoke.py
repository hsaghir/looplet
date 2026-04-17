"""Smoke tests for openharness.provenance."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from openharness.provenance import (
    AsyncRecordingLLMBackend,
    LLMCall,
    ProvenanceSink,
    RecordingLLMBackend,
    StepRecord,
    Trajectory,
    TrajectoryRecorder,
)
from openharness.testing import AsyncMockLLMBackend, MockLLMBackend
from openharness.types import LLMBackend, Step, ToolCall, ToolResult


class TestRecordingLLMBackendSmoke:
    def test_satisfies_llm_backend_protocol(self):
        rec = RecordingLLMBackend(MockLLMBackend(responses=["ok"]))
        assert isinstance(rec, LLMBackend)

    def test_records_prompt_and_response(self):
        rec = RecordingLLMBackend(MockLLMBackend(responses=["hi there"]))
        out = rec.generate("hello", system_prompt="sys", temperature=0.1, max_tokens=50)
        assert out == "hi there"
        assert len(rec.calls) == 1
        c = rec.calls[0]
        assert c.index == 0
        assert c.method == "generate"
        assert c.prompt == "hello"
        assert c.system_prompt == "sys"
        assert c.response == "hi there"
        assert c.temperature == 0.1
        assert c.max_tokens == 50
        assert c.duration_ms >= 0.0
        assert c.error is None

    def test_records_error_and_reraises(self):
        class BoomBackend:
            def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                raise RuntimeError("boom")

        rec = RecordingLLMBackend(BoomBackend())
        with pytest.raises(RuntimeError):
            rec.generate("x")
        assert len(rec.calls) == 1
        assert rec.calls[0].error is not None
        assert "boom" in rec.calls[0].error

    def test_generate_with_tools_only_when_wrapped_supports_it(self):
        plain = MockLLMBackend(responses=["ok"])
        rec = RecordingLLMBackend(plain)
        assert not hasattr(rec, "generate_with_tools")

        class NativeBackend:
            def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
                return ""

            def generate_with_tools(
                self, prompt, *, tools, max_tokens=2000, system_prompt="", temperature=0.2
            ):
                return [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}]

        rec2 = RecordingLLMBackend(NativeBackend())
        assert hasattr(rec2, "generate_with_tools")
        blocks = rec2.generate_with_tools("q", tools=[{"name": "search"}])
        assert isinstance(blocks, list)
        assert rec2.calls[0].method == "generate_with_tools"
        assert rec2.calls[0].tools == [{"name": "search"}]

    def test_truncates_large_prompt(self):
        huge = "x" * 1000
        rec = RecordingLLMBackend(MockLLMBackend(responses=["ok"]), max_chars_per_call=100)
        rec.generate(huge)
        assert len(rec.calls[0].prompt) <= 100
        assert "truncated" in rec.calls[0].prompt

    def test_redact_callable_is_applied(self):
        rec = RecordingLLMBackend(
            MockLLMBackend(responses=["token=SECRET"]),
            redact=lambda s: s.replace("SECRET", "***"),
        )
        rec.generate("password=SECRET")
        assert "SECRET" not in rec.calls[0].prompt
        assert "SECRET" not in rec.calls[0].response
        assert "***" in rec.calls[0].prompt

    def test_save_writes_files(self, tmp_path: Path):
        rec = RecordingLLMBackend(MockLLMBackend(responses=["r1", "r2"]))
        rec.generate("p1", system_prompt="sys")
        rec.generate("p2")
        out = rec.save(tmp_path / "traces")
        assert (out / "call_00_prompt.txt").exists()
        assert (out / "call_00_response.txt").exists()
        assert (out / "call_01_prompt.txt").exists()
        assert (out / "manifest.jsonl").exists()
        text = (out / "call_00_prompt.txt").read_text()
        assert "p1" in text and "sys" in text
        # manifest has one line per call
        lines = (out / "manifest.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["index"] == 0


class TestAsyncRecordingLLMBackendSmoke:
    def test_records_async_call(self):
        rec = AsyncRecordingLLMBackend(AsyncMockLLMBackend(responses=["async ok"]))

        async def run():
            return await rec.generate("hi")

        out = asyncio.run(run())
        assert out == "async ok"
        assert len(rec.calls) == 1
        assert rec.calls[0].method == "generate"


class TestTrajectoryRecorderSmoke:
    def _make_step(self, n: int, tool: str = "search", data=None) -> Step:
        tc = ToolCall(tool=tool)
        tr = ToolResult(tool=tool, args_summary=f"n={n}", data=data or [], duration_ms=10.0)
        return Step(number=n, tool_call=tc, tool_result=tr)

    def test_captures_steps_and_termination(self):
        rec_llm = RecordingLLMBackend(MockLLMBackend(responses=["a"]))
        hook = TrajectoryRecorder(recording_llm=rec_llm, capture_context=True)

        class DummyState:
            def __init__(self):
                self.steps: list[Step] = []

        state = DummyState()
        hook.pre_loop(state, None, "ctx")
        hook.pre_prompt(state, None, "briefing #1", step_num=1)
        rec_llm.generate("p1")  # linked to step 1
        step1 = self._make_step(1, tool="search", data=[1, 2, 3])
        state.steps.append(step1)
        hook.post_dispatch(state, None, step1.tool_call, step1.tool_result, step_num=1)

        hook.pre_prompt(state, None, "briefing #2", step_num=2)
        step2 = self._make_step(2, tool="done")
        state.steps.append(step2)
        hook.post_dispatch(state, None, step2.tool_call, step2.tool_result, step_num=2)

        hook.on_loop_end(state, None, "ctx", rec_llm)

        traj = hook.trajectory
        assert len(traj.steps) == 2
        assert traj.steps[0].context_before == "briefing #1"
        assert traj.steps[0].llm_call_indices == [0]
        assert traj.termination_reason == "done"
        assert len(traj.llm_calls) == 1
        assert traj.ended_at is not None and traj.ended_at >= traj.started_at

    def test_sweeps_missed_done_step(self):
        """Loop's done-handling path bypasses post_dispatch — on_loop_end
        must sweep state.steps to catch it."""
        hook = TrajectoryRecorder()

        class DummyState:
            def __init__(self):
                self.steps: list[Step] = []

        state = DummyState()
        hook.pre_loop(state, None, None)
        hook.pre_prompt(state, None, None, step_num=1)
        step1 = self._make_step(1, tool="search")
        state.steps.append(step1)
        hook.post_dispatch(state, None, step1.tool_call, step1.tool_result, step_num=1)
        # The `done` step is appended to state.steps but post_dispatch is
        # never called — simulate exactly that.
        step2 = self._make_step(2, tool="done")
        state.steps.append(step2)
        hook.on_loop_end(state, None, None, None)

        traj = hook.trajectory
        assert [s.step_num for s in traj.steps] == [1, 2]
        assert traj.steps[-1].tool_call["tool"] == "done"
        assert traj.termination_reason == "done"

    def test_save_writes_trajectory_and_steps(self, tmp_path: Path):
        hook = TrajectoryRecorder()

        class DummyState:
            def __init__(self):
                self.steps: list[Step] = []

        state = DummyState()
        hook.pre_loop(state, None, None)
        hook.pre_prompt(state, None, "ctx", step_num=1)
        step = self._make_step(1, tool="done")
        state.steps.append(step)
        hook.post_dispatch(state, None, step.tool_call, step.tool_result, step_num=1)
        hook.on_loop_end(state, None, None, None)

        out = hook.save(tmp_path / "traj")
        assert (out / "trajectory.json").exists()
        assert (out / "steps" / "step_01.json").exists()
        doc = json.loads((out / "trajectory.json").read_text())
        assert doc["step_count"] == 1
        assert doc["termination_reason"] == "done"


class TestProvenanceSinkSmoke:
    def test_wrap_llm_and_flush(self, tmp_path: Path):
        sink = ProvenanceSink(dir=tmp_path / "run")
        llm = sink.wrap_llm(MockLLMBackend(responses=["r1"]))
        assert isinstance(llm, RecordingLLMBackend)
        hook = sink.trajectory_hook()
        assert isinstance(hook, TrajectoryRecorder)

        class DummyState:
            def __init__(self):
                self.steps: list[Step] = []

        state = DummyState()
        hook.pre_loop(state, None, None)
        hook.pre_prompt(state, None, "ctx", step_num=1)
        llm.generate("hi")
        tc = ToolCall(tool="done")
        tr = ToolResult(tool="done", args_summary="", data=None)
        step = Step(number=1, tool_call=tc, tool_result=tr)
        state.steps.append(step)
        hook.post_dispatch(state, None, tc, tr, step_num=1)
        hook.on_loop_end(state, None, None, llm)

        out = sink.flush()
        assert (out / "trajectory.json").exists()
        assert (out / "call_00_prompt.txt").exists()

    def test_wrap_llm_detects_async(self):
        sink = ProvenanceSink(dir=Path("/tmp/does-not-matter"))
        wrapped = sink.wrap_llm(AsyncMockLLMBackend(responses=["r"]))
        assert isinstance(wrapped, AsyncRecordingLLMBackend)


class TestDataclassSmoke:
    def test_llm_call_to_dict(self):
        c = LLMCall(
            index=0, timestamp=1.0, duration_ms=5.0, method="generate",
            prompt="p", system_prompt="", response="r",
        )
        d = c.to_dict()
        assert d["index"] == 0
        assert d["method"] == "generate"
        assert d["prompt_chars"] == 1
        assert d["response_chars"] == 1

    def test_step_record_to_dict(self):
        s = StepRecord(
            step_num=1, timestamp=1.0, duration_ms=2.0, pretty="#1 ✓",
            tool_call={"tool": "x"}, tool_result={"tool": "x"},
        )
        assert s.to_dict()["step_num"] == 1

    def test_trajectory_to_dict_empty(self):
        t = Trajectory(run_id="abc", started_at=0.0)
        d = t.to_dict()
        assert d["run_id"] == "abc"
        assert d["step_count"] == 0
