"""Smoke tests for replay_loop and the `python -m looplet show` CLI."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.__main__ import main as cli_main
from looplet.provenance import ProvenanceSink, replay_loop
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec


def _make_tools() -> BaseToolRegistry:
    tools = BaseToolRegistry()
    tools.register(
        ToolSpec(
            name="add",
            description="Add two integers",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    tools.register(
        ToolSpec(
            name="done",
            description="Finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
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
                self,
                state,
                session_log,
                tool_call,
                tool_result,
                step_num,
            ):
                self.post += 1
                return None

        counter = Counter()
        steps = list(replay_loop(trace_dir, tools=_make_tools(), hooks=[counter]))
        assert counter.post == len(steps)  # post_dispatch fires for done() too

    def test_fallback_without_manifest(self, tmp_path: Path):
        """Replay should work from call_NN_response.txt even without manifest."""
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "manifest.jsonl").unlink()
        steps = list(replay_loop(trace_dir, tools=_make_tools()))
        assert len(steps) == 3

    @pytest.mark.parametrize(
        "manifest, expected",
        [
            ("not-json\n", "invalid"),
            ("42\n", "expected an object"),
            ('{"method":"generate"}\n', "expected index 0"),
            (
                '{"index":0,"method":"generate"}\n{"index":0,"method":"generate"}\n',
                "expected index 1",
            ),
            ('{"index":0,"method":"unknown"}\n', "unsupported method"),
        ],
    )
    def test_replay_rejects_invalid_manifest(
        self,
        tmp_path: Path,
        manifest: str,
        expected: str,
    ):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text(manifest)
        (trace_dir / "call_00_response.txt").write_text("response")

        with pytest.raises(ValueError, match=expected):
            list(replay_loop(trace_dir, tools=_make_tools()))

    def test_replay_rejects_missing_manifest_response(self, tmp_path: Path):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text('{"index":0,"method":"generate"}\n')

        with pytest.raises(FileNotFoundError, match="call_00_response.txt"):
            list(replay_loop(trace_dir, tools=_make_tools()))

    @pytest.mark.parametrize("payload", ["not-json", "[42]"])
    def test_replay_rejects_malformed_native_response(self, tmp_path: Path, payload: str):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text('{"index":0,"method":"generate_with_tools"}\n')
        (trace_dir / "call_00_response.txt").write_text(
            f"# call 00 - method=generate_with_tools\n\n## RESPONSE (content blocks)\n{payload}\n"
        )

        with pytest.raises(ValueError, match="invalid recorded response"):
            list(replay_loop(trace_dir, tools=_make_tools()))

    def test_replays_valid_native_responses(self, tmp_path: Path):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text(
            '{"index":0,"method":"generate_with_tools"}\n'
            '{"index":1,"method":"generate_with_tools"}\n'
        )
        responses = [
            [{"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1, "b": 2}}],
            [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "done",
                    "input": {"answer": "3"},
                }
            ],
        ]
        for index, response in enumerate(responses):
            (trace_dir / f"call_{index:02d}_response.txt").write_text(
                f"# call {index:02d} - method=generate_with_tools\n\n"
                "## RESPONSE (content blocks)\n"
                f"{json.dumps(response)}\n"
            )

        steps = list(
            replay_loop(
                trace_dir,
                tools=_make_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3, use_native_tools=True),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["add", "done"]


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

    def test_show_json_preserves_trace_artifacts(self, tmp_path: Path, capsys):
        trace_dir = _captured_dir(tmp_path)

        rc = cli_main(["show", str(trace_dir), "--json"])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["trajectory"] == json.loads((trace_dir / "trajectory.json").read_text())
        assert payload["manifest"] == [
            json.loads(line)
            for line in (trace_dir / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        ]

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

    def test_show_json_tolerates_missing_trajectory(self, tmp_path: Path, capsys):
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "trajectory.json").unlink()

        rc = cli_main(["show", str(trace_dir), "--json"])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["trajectory"] is None
        assert len(payload["manifest"]) == 3

    def test_show_rejects_nonobject_trajectory(self, tmp_path: Path, capsys):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "trajectory.json").write_text("[1]\n")

        rc = cli_main(["show", str(trace_dir)])

        assert rc == 1
        captured = capsys.readouterr()
        assert "expected a JSON object" in captured.err

    def test_show_rejects_malformed_manifest_line(self, tmp_path: Path, capsys):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text('{"index": 0}\nnot-json\n')

        rc = cli_main(["show", str(trace_dir), "--json"])

        assert rc == 1
        captured = capsys.readouterr()
        assert "manifest.jsonl line 2" in captured.err
        assert captured.out == ""

    def test_show_rejects_nonobject_manifest_record(self, tmp_path: Path, capsys):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text("42\n")

        rc = cli_main(["show", str(trace_dir), "--json"])

        assert rc == 1
        captured = capsys.readouterr()
        assert "expected a JSON object" in captured.err
        assert captured.out == ""

    @pytest.mark.parametrize(
        "trajectory, expected",
        [
            ({"run_id": "r", "steps": 1}, "trajectory.steps must be an array"),
            ({"run_id": "r", "steps": [42]}, "trajectory.steps[0] must be an object"),
            (
                {"run_id": "r", "steps": [{"tool_call": 42}]},
                "trajectory.steps[0].tool_call must be an object",
            ),
            (
                {"run_id": "r", "steps": [{"tool_result": 42}]},
                "trajectory.steps[0].tool_result must be an object",
            ),
            (
                {"run_id": "r", "steps": [{"llm_call_indices": 1}]},
                "trajectory.steps[0].llm_call_indices must be an array",
            ),
            (
                {"run_id": "r", "steps": [{"duration_ms": "slow"}]},
                "trajectory.steps[0].duration_ms must be a finite number",
            ),
            (
                {"run_id": "r", "termination_reason": {}, "steps": []},
                "trajectory.termination_reason must be a string",
            ),
            (
                {"run_id": "r", "step_count": [], "steps": []},
                "trajectory.step_count must be a non-negative integer",
            ),
            (
                {"run_id": "r", "failure_modes": 1, "steps": []},
                "trajectory.failure_modes must be an array",
            ),
            (
                {"run_id": "r", "steps": [{"duration_ms": 10**1000}]},
                "trajectory.steps[0].duration_ms must be a finite number",
            ),
        ],
    )
    @pytest.mark.parametrize("json_mode", [False, True])
    def test_show_rejects_malformed_nested_trajectory(
        self,
        tmp_path: Path,
        capsys,
        trajectory,
        expected: str,
        json_mode: bool,
    ):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "trajectory.json").write_text(json.dumps(trajectory))
        args = ["show", str(trace_dir)]
        if json_mode:
            args.append("--json")

        rc = cli_main(args)

        assert rc == 1
        captured = capsys.readouterr()
        assert expected in captured.err
        assert "traceback" not in captured.err.lower()
        assert captured.out == ""

    def test_show_rejects_malformed_manifest_summary(self, tmp_path: Path, capsys):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text('{"duration_ms":"slow"}\n')

        rc = cli_main(["show", str(trace_dir)])

        assert rc == 1
        captured = capsys.readouterr()
        assert "manifest[0].duration_ms must be a finite number" in captured.err
        assert "traceback" not in captured.err.lower()


class TestDoctorCLISmoke:
    def test_doctor_no_backend_renders_local_checks(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        rc = cli_main(["doctor", "--no-backend"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "looplet doctor" in out
        assert "python" in out
        assert "backend_probe" in out

    def test_doctor_json_output(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        rc = cli_main(["doctor", "--no-backend", "--json"])

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "checks" in payload
        assert any(check["name"] == "looplet" for check in payload["checks"])

    def test_doctor_strict_fails_on_missing_env_warnings(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        rc = cli_main(["doctor", "--no-backend", "--strict"])

        assert rc == 1
        assert "OPENAI_BASE_URL" in capsys.readouterr().out
