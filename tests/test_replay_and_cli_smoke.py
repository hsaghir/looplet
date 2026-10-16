"""Smoke tests for replay_loop and the `python -m looplet show` CLI."""

from __future__ import annotations

import io
import json
import sys
import warnings
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.__main__ import main as cli_main
from looplet.artifact_compat import ARTIFACT_DESCRIPTOR
from looplet.evals import EvalHook, EvalResult, eval_mark, save_eval_run
from looplet.provenance import (
    ProvenanceSink,
    _load_trace_calls,
    _ReplayLLMBackend,
    replay_loop,
)
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


def _captured_dir(tmp_path: Path, *, task: dict | None = None) -> Path:
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
        config=LoopConfig(max_steps=5, use_native_tools=False),
        task=task,
    ):
        pass
    return sink.flush()


class TestReplayLoopSmoke:
    def test_inspect_trace_uses_existing_show_contract(self, tmp_path, capsys):
        trace_dir = _captured_dir(tmp_path)
        assert cli_main(["inspect", str(trace_dir), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["trajectory"]["termination_reason"] == "done"
        assert len(payload["manifest"]) == 3

    def test_omitted_replay_state_uses_configured_limit_without_warning(self, tmp_path):
        trace_dir = _captured_dir(tmp_path)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            steps = list(
                replay_loop(
                    trace_dir,
                    tools=_make_tools(),
                    config=LoopConfig(max_steps=7, use_native_tools=False),
                )
            )
        assert len(steps) == 3
        assert not recorded

    def test_replays_captured_run(self, tmp_path: Path):
        trace_dir = _captured_dir(tmp_path)
        out_steps = []
        for step in replay_loop(trace_dir, tools=_make_tools()):
            out_steps.append(step)
        assert [s.tool_call.tool for s in out_steps] == ["add", "add", "done"]
        # Same totals as the captured run.
        assert out_steps[-1].tool_call.tool == "done"

    @pytest.mark.parametrize("capture_trajectory", [False, True])
    @pytest.mark.parametrize("single_call", [False, True])
    def test_native_replay_budget_counts_tool_steps(
        self, tmp_path, capture_trajectory, single_call
    ):
        calls = [
            {"type": "tool_use", "id": "add-1", "name": "add", "input": {"a": 1, "b": 2}},
            {"type": "tool_use", "id": "add-2", "name": "add", "input": {"a": 3, "b": 4}},
            {"type": "tool_use", "id": "done-1", "name": "done", "input": {"answer": "7"}},
        ]

        class Backend:
            def __init__(self):
                self.responses = [calls] if single_call else [calls[:2], calls[2:]]

            def generate(self, prompt, **kwargs):
                raise AssertionError("Expected native calls")

            def generate_with_tools(self, prompt, **kwargs):
                return self.responses.pop(0)

        sink = ProvenanceSink(dir=tmp_path / "native")
        original = list(
            composable_loop(
                llm=sink.wrap_llm(Backend()),
                tools=_make_tools(),
                config=LoopConfig(max_steps=3, use_native_tools=True),
                hooks=[sink.trajectory_hook()] if capture_trajectory else [],
            )
        )
        trace_dir = sink.flush()
        if not capture_trajectory:
            (trace_dir / "trajectory.json").write_text(
                json.dumps({"step_count": 1, "task": {"goal": "undeclared sidecar"}})
            )
        before = {path: path.read_bytes() for path in trace_dir.rglob("*") if path.is_file()}
        state = DefaultState(max_steps=3)

        replayed = list(replay_loop(trace_dir, tools=_make_tools(), state=state))

        assert [step.tool_call.tool for step in original] == ["add", "add", "done"]
        assert [step.tool_call.tool for step in replayed] == ["add", "add", "done"]
        assert state.task == {}
        assert state._stop_reason == "done"
        assert {
            path: path.read_bytes() for path in trace_dir.rglob("*") if path.is_file()
        } == before

    @pytest.mark.parametrize("override", [None, {}, {"goal": "replacement"}])
    def test_replay_inherits_saved_task_unless_overridden(self, tmp_path, override):
        task = {"goal": "Compute both sums.", "context": {"labels": ["original"]}}
        trace_dir = _captured_dir(tmp_path, task=task)
        before = {path: path.read_bytes() for path in trace_dir.rglob("*") if path.is_file()}
        state = DefaultState(max_steps=3)

        steps = list(replay_loop(trace_dir, tools=_make_tools(), state=state, task=override))

        assert [step.tool_call.tool for step in steps] == ["add", "add", "done"]
        assert state.task == (task if override is None else override)
        assert {
            path: path.read_bytes() for path in trace_dir.rglob("*") if path.is_file()
        } == before

    def test_explicit_replay_config_overrides_saved_budget(self, tmp_path):
        trace_dir = _captured_dir(tmp_path)

        steps = list(
            replay_loop(
                trace_dir,
                tools=_make_tools(),
                config=LoopConfig(max_steps=1, use_native_tools=False),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["add"]

    @pytest.mark.parametrize(
        "payload, expected",
        [
            ("not-json", "Expecting value"),
            ("[]", "must contain an object"),
            ('{"step_count": true}', "step_count must be a non-negative integer"),
            ('{"step_count": -1}', "step_count must be a non-negative integer"),
            ('{"task": "wrong type"}', "task must be an object"),
        ],
    )
    def test_replay_rejects_invalid_defaults_before_running(
        self, tmp_path, monkeypatch, payload, expected
    ):
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "trajectory.json").write_text(payload)

        def unexpected_loop(**kwargs):
            raise AssertionError("Invalid defaults must fail before execution")

        monkeypatch.setattr("looplet.loop.composable_loop", unexpected_loop)
        with pytest.raises(ValueError, match=expected):
            list(replay_loop(trace_dir, tools=_make_tools()))

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
        (trace_dir / ARTIFACT_DESCRIPTOR).unlink()
        steps = list(replay_loop(trace_dir, tools=_make_tools()))
        assert len(steps) == 3

    def test_fallback_rejects_missing_middle_response(self, tmp_path: Path):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "call_00_response.txt").write_text("response")
        (trace_dir / "call_02_response.txt").write_text("response")

        with pytest.raises(ValueError, match="expected index 1, got 2"):
            list(replay_loop(trace_dir, tools=_make_tools()))

    def test_replay_preserves_recorded_llm_error(self, tmp_path: Path):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "index": 0,
                    "method": "generate_with_tools",
                    "error": "RuntimeError: provider unavailable",
                }
            )
            + "\n"
        )
        (trace_dir / "call_00_response.txt").write_text(
            "# call 00 - error=yes\n\n"
            "## ERROR\nRuntimeError: provider unavailable\n\n"
            "## RESPONSE (content blocks)\n[]\n"
        )

        calls = _load_trace_calls(trace_dir)
        assert calls[0]["error"] == "RuntimeError: provider unavailable"
        backend = _ReplayLLMBackend(calls)
        with pytest.raises(RuntimeError, match="replayed LLM call failed"):
            backend.generate_with_tools("prompt", tools=[])

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
                state=DefaultState(max_steps=2),
            )
        )

        assert [step.tool_call.tool for step in steps] == ["add", "done"]


class TestShowCLISmoke:
    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize(
        "step_durations,call_durations,total_ms",
        [([None, 4], [8, None], 4), ([None], [None, 8], 8), ([None], [None], 0)],
    )
    def test_nullable_timings_render_without_changing_trace(
        self, tmp_path, capsys, command, step_durations, call_durations, total_ms
    ):
        trajectory = {
            "run_id": "nullable-timing",
            "steps": [
                {"step_num": index, "duration_ms": duration}
                for index, duration in enumerate(step_durations, start=1)
            ],
        }
        calls = [
            {"duration_ms": duration, "prompt_chars": None, "response_chars": None}
            for duration in call_durations
        ]
        (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))
        (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(call) for call in calls))

        assert cli_main([command, str(tmp_path)]) == 0
        assert f"{total_ms}ms" in capsys.readouterr().out.splitlines()[0]
        assert cli_main([command, str(tmp_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == {
            "trajectory": trajectory,
            "manifest": calls,
        }
        assert json.loads((tmp_path / "trajectory.json").read_text()) == trajectory

    def test_show_renders_summary(self, tmp_path: Path, capsys):
        trace_dir = _captured_dir(tmp_path)
        rc = cli_main(["show", str(trace_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "done" in out
        assert "3 steps" in out
        assert "LLM:" in out
        assert "add" in out

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize("json_output", [False, True])
    @pytest.mark.parametrize("legacy", [False, True])
    def test_saved_eval_evidence_is_visible_without_execution(
        self, tmp_path, capsys, command, json_output, legacy
    ):
        calls = {"grader": 0, "collector": 0}

        @eval_mark("required")
        def eval_content(ctx):
            calls["grader"] += 1
            return EvalResult(
                name="eval_content",
                score=0.0,
                label="fail",
                explanation="The output file is missing.",
                details=["Expected hello in result.txt."],
            )

        def collect_content(state):
            calls["collector"] += 1
            return {"content": None}

        hook = EvalHook(evaluators=[eval_content], collectors=[collect_content])
        sink = ProvenanceSink(dir=tmp_path / "trace")
        list(
            composable_loop(
                llm=sink.wrap_llm(
                    MockLLMBackend(['{"tool":"done","args":{"answer":"Task complete."}}'])
                ),
                tools=_make_tools(),
                hooks=[sink.trajectory_hook(), hook],
                config=LoopConfig(use_native_tools=False),
            )
        )
        trace_dir = sink.flush()
        save_eval_run(trace_dir, eval_hook=hook)
        if legacy:
            (trace_dir / ARTIFACT_DESCRIPTOR).unlink()
        original = {path.name: path.read_bytes() for path in trace_dir.iterdir() if path.is_file()}
        arguments = [command, str(trace_dir), *(["--json"] if json_output else [])]

        assert cli_main(arguments) == 0
        rendered = capsys.readouterr().out
        if json_output:
            payload = json.loads(rendered)
            assert payload["evals"] == json.loads((trace_dir / "evals.json").read_text())
            assert payload["artifacts"] == {"content": None}
        else:
            assert "done" in rendered
            assert "saved evals:" in rendered
            assert "eval_content" in rendered
            assert "fail" in rendered
            assert "The output file is missing." in rendered
            assert "Expected hello in result.txt." in rendered
            assert '"content": null' in rendered
        assert calls == {"grader": 1, "collector": 1}
        assert {
            path.name: path.read_bytes() for path in trace_dir.iterdir() if path.is_file()
        } == original

    @pytest.mark.parametrize("json_output", [False, True])
    def test_saved_eval_evidence_accepts_forward_compatible_records(self, capsys, json_output):
        trace_dir = Path(__file__).parent / "fixtures" / "artifact_compat" / "v1_eval_run"
        arguments = ["inspect", str(trace_dir), *(["--json"] if json_output else [])]

        assert cli_main(arguments) == 0
        rendered = capsys.readouterr().out
        if json_output:
            assert json.loads(rendered)["evals"] == json.loads(
                (trace_dir / "evals.json").read_text()
            )
        else:
            assert "saved evals:" in rendered
            assert "eval_tests_pass" in rendered

    @pytest.mark.parametrize("json_output", [False, True])
    def test_saved_eval_evidence_rejects_corrupt_files(self, tmp_path, capsys, json_output):
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / ARTIFACT_DESCRIPTOR).unlink()
        (trace_dir / "evals.json").write_text("not JSON")
        arguments = ["inspect", str(trace_dir), *(["--json"] if json_output else [])]

        assert cli_main(arguments) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "could not read saved eval evidence" in captured.err
        assert "evals.json" in captured.err
        assert "Traceback" not in captured.err

    @pytest.mark.parametrize("json_output", [False, True])
    def test_provenance_ignores_undeclared_eval_files(self, tmp_path, capsys, json_output):
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "evals.json").write_text("not JSON")
        (trace_dir / "artifacts.json").write_text("not JSON")
        arguments = ["inspect", str(trace_dir), *(["--json"] if json_output else [])]

        assert cli_main(arguments) == 0
        rendered = capsys.readouterr().out
        if json_output:
            assert set(json.loads(rendered)) == {"trajectory", "manifest"}
        else:
            assert "saved evals:" not in rendered
            assert "artifacts:" not in rendered

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize("terminal_tool", ["done", "finish"])
    def test_renders_complete_final_result(self, tmp_path, capsys, command, terminal_tool):
        output = {"answer": "The payment service is owned by fintech-platform."}
        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name=terminal_tool,
                description="Finish the task.",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )
        sink = ProvenanceSink(dir=tmp_path / "trace")
        backend = sink.wrap_llm(
            MockLLMBackend([json.dumps({"tool": terminal_tool, "args": output})])
        )
        list(
            composable_loop(
                llm=backend,
                tools=tools,
                config=LoopConfig(done_tool=terminal_tool, use_native_tools=False),
                hooks=[sink.trajectory_hook()],
            )
        )
        trace_dir = sink.flush()
        original = (trace_dir / "trajectory.json").read_bytes()

        assert cli_main([command, str(trace_dir)]) == 0
        rendered = capsys.readouterr().out
        assert "result:" in rendered
        assert json.dumps(output, indent=2) in rendered
        assert cli_main([command, str(trace_dir), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["trajectory"] == json.loads(original)
        assert (trace_dir / "trajectory.json").read_bytes() == original

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize(
        "output", ["A complete answer with more than thirty characters.", [], 0, False]
    )
    def test_renders_scalar_and_list_final_results(self, tmp_path, capsys, command, output):
        trajectory = {
            "termination_reason": "done",
            "steps": [{"tool_result": {"data": output}}],
        }
        (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))

        assert cli_main([command, str(tmp_path)]) == 0
        rendered = capsys.readouterr().out
        expected = output if isinstance(output, str) else json.dumps(output, indent=2)
        assert f"result:\n{expected}\n" in rendered

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize(
        "reason, output, result_error, step_error",
        [
            ("budget", {"answer": "partial"}, None, None),
            ("cancelled", {"answer": "partial"}, None, None),
            ("error", {"answer": "partial"}, None, None),
            ("done", {"rejected": True, "reason": "Check failed."}, None, None),
            ("done", {"answer": "partial"}, "Tool failed.", None),
            ("done", {"answer": "partial"}, None, "Step failed."),
            ("done", None, None, None),
        ],
    )
    def test_does_not_label_nonfinal_output_as_result(
        self, tmp_path, capsys, command, reason, output, result_error, step_error
    ):
        trajectory = {
            "termination_reason": reason,
            "steps": [
                {"tool_result": {"data": output, "error": result_error}, "error": step_error}
            ],
        }
        (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))

        assert cli_main([command, str(tmp_path)]) == 0
        assert "result:" not in capsys.readouterr().out

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
        (trace_dir / ARTIFACT_DESCRIPTOR).unlink()
        rc = cli_main(["show", str(trace_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "LLM:" in out

    def test_show_json_tolerates_missing_trajectory(self, tmp_path: Path, capsys):
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "trajectory.json").unlink()
        (trace_dir / ARTIFACT_DESCRIPTOR).unlink()

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

    def test_show_rejects_manifest_directory(self, tmp_path: Path, capsys):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()
        (trace_dir / "manifest.jsonl").mkdir()

        rc = cli_main(["show", str(trace_dir)])

        assert rc == 1
        captured = capsys.readouterr()
        assert "could not read" in captured.err
        assert "Traceback" not in captured.err
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
    @pytest.mark.parametrize("json_output", [False, True])
    @pytest.mark.parametrize(
        ("provider", "variable", "endpoint", "origin"),
        [
            ("openai", "OPENAI_BASE_URL", "http://localhost:1234/v1", "http://localhost:1234"),
            ("openai", "OPENAI_BASE_URL", "http://[::1]:1234/v1", "http://[::1]:1234"),
            (
                "openai",
                "OPENAI_BASE_URL",
                "https://review_user:review_password@provider.invalid:8443/private-path?key=review_query#review_fragment",
                "https://provider.invalid:8443",
            ),
            (
                "anthropic",
                "ANTHROPIC_BASE_URL",
                "https://anthropic-review.invalid/private-path",
                "https://anthropic-review.invalid",
            ),
            ("proxy", "COPILOT_PROXY_URL", "http://localhost:4321/v1", "http://localhost:4321"),
            ("openai", None, None, "https://api.openai.com"),
        ],
    )
    def test_doctor_reports_effective_endpoint_without_credentials(
        self, capsys, json_output, provider, variable, endpoint, origin
    ):
        import os
        from unittest.mock import patch

        environment = {
            "LOOPLET_PROVIDER": provider,
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY": "review_key",
        }
        if variable is not None:
            environment[variable] = endpoint
            if provider in {"anthropic", "proxy"}:
                environment["OPENAI_BASE_URL"] = endpoint
        else:
            environment["COPILOT_PROXY_URL"] = "https://api.openai.com/v1"
        arguments = ["doctor", "--no-backend", "--strict"]
        if json_output:
            arguments.append("--json")
        with patch.dict(os.environ, environment, clear=True):
            assert cli_main(arguments) == 0
        output = capsys.readouterr()
        assert not output.err
        if json_output:
            checks = json.loads(output.out)["checks"]
            detail = next(check["detail"] for check in checks if check["name"] == "endpoint")
        else:
            detail = next(
                line for line in output.out.splitlines() if line.startswith("OK endpoint:")
            )
        assert origin in detail
        assert (variable or "provider client default") in detail
        assert "origin only" in detail
        assert all(
            secret not in output.out
            for secret in (
                "review_user",
                "review_password",
                "private-path",
                "review_query",
                "review_fragment",
                "review_key",
            )
        )

    @pytest.mark.parametrize("json_output", [False, True])
    @pytest.mark.parametrize("provider", [None, "openai", "anthropic"])
    def test_doctor_reports_ambiguous_provider_selection(self, capsys, provider, json_output):
        import os
        from unittest.mock import patch

        environment = {
            "OPENAI_BASE_URL": "http://localhost:1234/v1",
            "ANTHROPIC_API_KEY": "review-placeholder",
        }
        if provider is not None:
            environment["LOOPLET_PROVIDER"] = provider
        arguments = ["doctor", "--no-backend", "--strict"]
        if json_output:
            arguments.append("--json")
        with patch.dict(os.environ, environment, clear=True):
            assert cli_main(arguments) == (1 if provider is None else 0)
        output = capsys.readouterr()
        assert not output.err
        if json_output:
            checks = json.loads(output.out)["checks"]
            warnings = [check for check in checks if check["name"] == "provider_selection"]
            assert len(warnings) == (1 if provider is None else 0)
            if warnings:
                assert warnings[0]["status"] == "warn"
                assert "LOOPLET_PROVIDER=openai" in warnings[0]["detail"]
        else:
            assert ("WARN provider_selection:" in output.out) is (provider is None)

    @pytest.mark.parametrize(
        "environment",
        [
            {"OPENAI_API_KEY": "test"},
            {"OPENAI_BASE_URL": "http://localhost:1234/v1"},
            {"ANTHROPIC_API_KEY": "test"},
        ],
    )
    def test_doctor_strict_accepts_provider_defaults(self, capsys, environment):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, environment, clear=True):
            assert cli_main(["doctor", "--no-backend", "--strict", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert all(check["status"] == "ok" for check in payload["checks"])

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
