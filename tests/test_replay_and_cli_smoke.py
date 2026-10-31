"""Smoke tests for replay_loop and the `python -m looplet show` CLI."""

from __future__ import annotations

import gc
import io
import json
import sys
import warnings
from contextlib import aclosing
from pathlib import Path

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    HookDecision,
    LoopConfig,
    async_composable_loop,
    composable_loop,
    tools_from,
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
from looplet.testing import AsyncMockLLMBackend, MockLLMBackend
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("scenario", ["missing", "matching", "mismatch", "override"])
async def test_assistant_replay_requires_binding_and_checks_decisions(
    tmp_path, asynchronous, native, scenario
):
    from looplet import AssistantTurnDecision, RunResult
    from looplet.provenance import async_replay_loop

    effects = []
    responses = [
        "A retained intermediate message",
        '{"tool":"add","args":{"a":1,"b":2}}',
        '{"tool":"done","args":{"answer":"checked"}}',
    ]
    backend = MockLLMBackend(responses)
    original_generate = backend.generate

    def generate_with_tools(prompt, **kwargs):
        text = original_generate(prompt)
        if text.startswith("{"):
            call = json.loads(text)
            return [
                {
                    "type": "tool_use",
                    "id": call["tool"],
                    "name": call["tool"],
                    "input": call["args"],
                }
            ]
        return [{"type": "text", "text": text}]

    backend.generate_with_tools = generate_with_tools
    sink = ProvenanceSink(tmp_path / "source")
    source_policy = lambda context: AssistantTurnDecision("continue", "Check the observation")
    list(
        composable_loop(
            llm=sink.wrap_llm(backend),
            tools=_make_tools(),
            hooks=[sink.trajectory_hook()],
            config=LoopConfig(
                max_steps=2,
                max_model_calls=3,
                use_native_tools=native,
                assistant_turn_policy=source_policy,
            ),
        )
    )
    source = sink.flush()
    assert sink.trajectory_hook().trajectory.termination_reason == "done"
    calls = _load_trace_calls(source)
    assert all(call["schema"] == "looplet.model-call.v2" for call in calls)
    assert calls[0]["execution"]["model_turn_id"] == "run:turn-1"
    assert calls[0]["execution"]["model_call_id"] == "run:turn-1:call-1"

    def add(a: int, b: int):
        effects.append((a, b))
        return {"sum": a + b + 100}

    def done(answer: str):
        return {"answer": answer}

    state = DefaultState(max_steps=2)
    kwargs = dict(trace_dir=source, state=state, tools=tools_from([add, done]))
    if scenario != "missing":
        kwargs["assistant_turn_policy"] = (
            (lambda context: AssistantTurnDecision("stop"))
            if scenario == "override"
            else (lambda context: AssistantTurnDecision("continue", "Changed feedback"))
            if scenario == "mismatch"
            else source_policy
        )
    if scenario == "override":
        kwargs["policy_override_name"] = "counterfactual-stop"

    async def replay():
        return (
            [step async for step in async_replay_loop(**kwargs)]
            if asynchronous
            else list(replay_loop(**kwargs))
        )

    if scenario == "missing":
        with pytest.raises(ValueError, match="explicit host policy binding"):
            await replay()
        assert effects == []
        return
    steps = await replay()
    result = RunResult.from_state(state)
    if scenario == "matching":
        assert result.completed
        assert effects == [(1, 2)]
        assert steps[0].tool_result.data == {"sum": 103}
        assert state.metadata["looplet_assistant_turns"]["model_calls"] == 3
    else:
        assert effects == []
        assert steps == []
        assert not result.completed
        assert result.termination_reason == (
            "assistant_stop" if scenario == "override" else "assistant_policy_error"
        )
        if scenario == "override":
            assert state.metadata["replay_policy_override"] == "counterfactual-stop"


class TestReplayLoopSmoke:
    @pytest.mark.parametrize("native", [False, True])
    def test_structured_capture_reconstructs_without_parsing_text_projections(
        self, tmp_path, native
    ):
        from looplet.provenance import RecordingLLMBackend

        content = "\n## RESPONSE\n" + "synthetic reasoning \u2028 " * 100
        response = [{"type": "thinking", "thinking": content, "signature": "opaque"}]

        class Backend:
            last_stop_reason = "max_tokens"
            last_usage = {"output_tokens": 12}
            last_response = {"id": "saved-response", "content": response}

            def generate(self, prompt, **kwargs):
                return content

            def generate_with_tools(self, prompt, **kwargs):
                return response

        recorder = RecordingLLMBackend(Backend(), max_chars_per_call=100)
        if native:
            recorder.generate_with_tools("full prompt " * 100, tools=[])
        else:
            recorder.generate("full prompt " * 100)
        recorder.save(tmp_path)
        (tmp_path / "call_00_prompt.txt").write_text("Bounded request projection")
        (tmp_path / "call_00_response.txt").write_text("Bounded response projection")
        calls = _load_trace_calls(tmp_path)
        assert calls[0]["request"]["prompt"] == "full prompt " * 100
        assert calls[0]["response"] == (response if native else content)
        replay = _ReplayLLMBackend(calls)
        actual = (
            replay.generate_with_tools("ignored", tools=[])
            if native
            else replay.generate("ignored")
        )
        assert actual == (response if native else content)
        assert replay.last_stop_reason == "max_tokens"
        assert replay.last_usage == {"output_tokens": 12}
        assert replay.last_response["id"] == "saved-response"

    def test_native_batch_terminal_step_links_to_its_complete_model_call(self, tmp_path):
        response = [
            {"type": "thinking", "thinking": "Synthetic decision", "signature": "opaque"},
            {"type": "text", "text": "Compute and finish"},
            {"type": "tool_use", "id": "add-1", "name": "add", "input": {"a": 1, "b": 2}},
            {"type": "tool_use", "id": "done-1", "name": "done", "input": {"answer": "3"}},
        ]
        backend = MockLLMBackend(["unused"])
        backend.generate_with_tools = lambda prompt, **kwargs: response
        sink = ProvenanceSink(tmp_path)
        steps = list(
            composable_loop(
                llm=sink.wrap_llm(backend),
                tools=_make_tools(),
                config=LoopConfig(max_steps=2),
                hooks=[sink.trajectory_hook()],
            )
        )
        assert [step.tool_call.call_id for step in steps] == ["add-1", "done-1"]
        document = json.loads((sink.flush() / "trajectory.json").read_text())
        assert [step["llm_call_indices"] for step in document["steps"]] == [[0], [0]]
        assert document["llm_calls"][0]["capture"]["response"] == response
        assert document["steps"][0]["tool_result"]["data"] == {"sum": 3}

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("slot", ["check_permission", "pre_dispatch"])
    async def test_async_replay_preserves_async_tool_denials(self, tmp_path, native, slot, recwarn):
        from looplet import async_replay_loop

        effects = []
        gate_calls = []

        async def record_probe(value: str) -> dict:
            effects.append(value)
            return {"value": value}

        class PermissionGate:
            async def check_permission(self, tool_call, state):
                gate_calls.append(tool_call.tool)
                return tool_call.tool != "record_probe"

        class DispatchGate:
            async def pre_dispatch(self, state, session_log, tool_call, step_num):
                gate_calls.append(tool_call.tool)
                if tool_call.tool == "record_probe":
                    return HookDecision(permission="deny", block="Synthetic test denial")
                return None

        gate_class = PermissionGate if slot == "check_permission" else DispatchGate
        responses = [
            '{"tool":"record_probe","args":{"value":"synthetic-only"}}',
            '{"tool":"done","args":{"summary":"Finished."}}',
        ]
        backend = AsyncMockLLMBackend(responses, cycle=False)
        if native:
            blocks = iter(
                [
                    [
                        {
                            "type": "tool_use",
                            "id": "probe-1",
                            "name": "record_probe",
                            "input": {"value": "synthetic-only"},
                        }
                    ],
                    [
                        {
                            "type": "tool_use",
                            "id": "done-1",
                            "name": "done",
                            "input": {"summary": "Finished."},
                        }
                    ],
                ]
            )

            async def generate_with_tools(prompt, **kwargs):
                return next(blocks)

            backend.generate_with_tools = generate_with_tools
        sink = ProvenanceSink(dir=tmp_path / "async-capture")
        tools = tools_from([record_probe], include_done=True, validate=True)
        task = {"goal": "Synthetic replay permission test"}
        captured = [
            step
            async for step in async_composable_loop(
                llm=sink.wrap_llm(backend),
                tools=tools,
                config=LoopConfig(max_steps=2, use_native_tools=native),
                task=task,
                hooks=[gate_class(), sink.trajectory_hook()],
            )
        ]
        trace = sink.flush()
        before = {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()}
        assert effects == [] and "record_probe" in gate_calls
        assert "Permission denied" in captured[0].tool_result.error
        gate_calls.clear()
        state = DefaultState(max_steps=2)

        replayed = [
            step
            async for step in async_replay_loop(
                trace,
                tools=tools_from([record_probe], include_done=True, validate=True),
                state=state,
                hooks=[gate_class()],
            )
        ]

        assert effects == [] and "record_probe" in gate_calls
        assert "Permission denied" in replayed[0].tool_result.error
        assert [step.tool_call.tool for step in replayed] == ["record_probe", "done"]
        assert state.task == task and state._stop_reason == "done"
        assert {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()} == before
        gc.collect()
        assert not any(issubclass(warning.category, RuntimeWarning) for warning in recwarn)

    @pytest.mark.parametrize("early_stop", [False, True])
    async def test_async_replay_awaits_tools_and_closes_the_loop(self, tmp_path, early_stop):
        from looplet import async_replay_loop

        trace = _captured_dir(tmp_path)
        effects = []
        cleaned = []

        async def add(a: int, b: int) -> dict:
            effects.append((a, b))
            return {"sum": a + b}

        def done(answer: str) -> dict:
            return {"answer": answer}

        class CleanupHook:
            async def on_loop_end(self, state, session_log, context, llm):
                cleaned.append(state._stop_reason)
                return 0

        state = DefaultState(max_steps=3)
        steps = []
        async with aclosing(
            async_replay_loop(
                trace,
                tools=tools_from([add, done], validate=True),
                state=state,
                hooks=[CleanupHook()],
            )
        ) as iterator:
            async for step in iterator:
                steps.append(step)
                if early_stop:
                    break

        assert effects == ([(1, 2)] if early_stop else [(1, 2), (3, 4)])
        assert len(steps) == (1 if early_stop else 3)
        assert cleaned == (["stream_closed"] if early_stop else ["done"])

    @pytest.mark.parametrize("limit", [None, 1, 7])
    @pytest.mark.parametrize("override", [None, {}, {"goal": "replacement"}])
    async def test_async_replay_preserves_task_and_budget_overrides(
        self, tmp_path, limit, override
    ):
        from looplet import async_replay_loop

        task = {"goal": "Compute both sums.", "context": {"labels": ["original"]}}
        trace = _captured_dir(tmp_path, task=task)
        state = DefaultState(max_steps=limit if limit is not None else 3)
        config = None if limit is None else LoopConfig(max_steps=limit, use_native_tools=False)

        steps = [
            step
            async for step in async_replay_loop(
                trace, tools=_make_tools(), state=state, config=config, task=override
            )
        ]

        assert state.task == (task if override is None else override)
        assert len(steps) == (1 if limit == 1 else 3)
        assert state._stop_reason == ("budget_exhausted" if limit == 1 else "done")

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
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_native_replay_budget_counts_tool_steps(
        self, tmp_path, capture_trajectory, single_call, asynchronous
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

        if asynchronous:
            from looplet import async_replay_loop

            replayed = [
                step
                async for step in async_replay_loop(trace_dir, tools=_make_tools(), state=state)
            ]
        else:
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

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("override", [False, True])
    async def test_replay_preserves_alternate_terminal_unless_overridden(
        self, tmp_path, asynchronous, native, override
    ):
        from looplet import RunResult, async_replay_loop, run_result, run_result_async

        submissions = []

        def submit(answer: str) -> dict:
            submissions.append(answer)
            return {"answer": answer}

        class Backend:
            calls = 0

            def generate(self, prompt, **kwargs):
                self.calls += 1
                return '{"tool":"submit","args":{"answer":"42"}}'

            def generate_with_tools(self, prompt, **kwargs):
                self.calls += 1
                return [
                    {
                        "type": "tool_use",
                        "id": "submit-1",
                        "name": "submit",
                        "input": {"answer": "42"},
                    }
                ]

        backend = Backend()
        sink = ProvenanceSink(dir=tmp_path / "capture")
        config = LoopConfig(
            harness_profile="minimal",
            max_steps=2,
            use_native_tools=native,
            done_tools=["submit"],
        )
        options = {
            "tools": tools_from([submit], include_done=True),
            "config": config,
            "hooks": [sink.trajectory_hook()],
        }
        recorded = sink.wrap_llm(backend)
        original = (
            await run_result_async(recorded, **options)
            if asynchronous
            else run_result(recorded, **options)
        )
        assert original.completed and original.output == {"answer": "42"}
        assert backend.calls == 1 and submissions == ["42"]
        trace = sink.flush()
        before = {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()}
        state = DefaultState(max_steps=1)
        replay_options = {
            "tools": tools_from([submit], include_done=True),
            "state": state,
        }
        if override:
            replay_options["config"] = LoopConfig(
                harness_profile="minimal", max_steps=1, use_native_tools=native
            )
        if asynchronous:
            replayed = [step async for step in async_replay_loop(trace, **replay_options)]
        else:
            replayed = list(replay_loop(trace, **replay_options))
        result = RunResult.from_state(state)
        assert result.completed is not override
        assert result.termination_reason == ("budget_exhausted" if override else "done")
        assert result.output == (None if override else {"answer": "42"})
        assert submissions == ["42", "42"]
        assert len(replayed) == 1
        assert config.done_tools == ["submit"]
        assert {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()} == before

    @pytest.mark.parametrize(
        "payload, expected",
        [
            ("not-json", "Expecting value"),
            ("[]", "must contain an object"),
            ('{"step_count": true}', "step_count must be a non-negative integer"),
            ('{"step_count": -1}', "step_count must be a non-negative integer"),
            ('{"task": "wrong type"}', "task must be an object"),
            ('{"metadata": null}', "metadata must be an object"),
            ('{"metadata": {"run_stats": []}}', "run_stats must be an object"),
            (
                '{"metadata": {"run_stats": {"model_turn_config": []}}}',
                "Invalid recorded model_turn_config",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"unknown": 1}}}}',
                "Invalid recorded model_turn_config",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"allow_text_completion": 1}}}}',
                "allow_text_completion must be boolean",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"max_tokens": false}}}}',
                "max_tokens must be an integer",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"max_tokens": 0}}}}',
                "max_tokens must be an integer",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"max_turn_continuations": -1}}}}',
                "max_turn_continuations must be an integer",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"done_tool": ""}}}}',
                "done_tool must be a nonempty string",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"done_tools": "submit"}}}}',
                "done_tools must be a list of nonempty strings",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"done_tools": [""]}}}}',
                "done_tools must be a list of nonempty strings",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"done_tools": [1]}}}}',
                "done_tools must be a list of nonempty strings",
            ),
            (
                '{"metadata": {"run_stats": {"model_turn_config": {"done_tools": null}}}}',
                "done_tools must be a list of nonempty strings",
            ),
        ],
    )
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_replay_rejects_invalid_defaults_before_running(
        self, tmp_path, monkeypatch, payload, expected, asynchronous
    ):
        trace_dir = _captured_dir(tmp_path)
        (trace_dir / "trajectory.json").write_text(payload)

        def unexpected_loop(**kwargs):
            raise AssertionError("Invalid defaults must fail before execution")

        monkeypatch.setattr("looplet.loop.composable_loop", unexpected_loop)
        monkeypatch.setattr("looplet.async_loop.async_composable_loop", unexpected_loop)
        with pytest.raises(ValueError, match=expected):
            if asynchronous:
                from looplet import async_replay_loop

                [step async for step in async_replay_loop(trace_dir, tools=_make_tools())]
            else:
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

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_replay_exhaustion_stops_without_retry(
        self, tmp_path, monkeypatch, native, asynchronous
    ):
        from looplet import RunResult, async_replay_loop
        from looplet.run_records import run_diagnostics

        response = '{"tool":"add","args":{"a":1,"b":2}}'
        backend = MockLLMBackend([response], cycle=False)
        if native:
            backend.generate_with_tools = lambda prompt, **kwargs: [
                {"type": "tool_use", "id": "add-1", "name": "add", "input": {"a": 1, "b": 2}}
            ]
        capture = ProvenanceSink(dir=tmp_path / "capture")
        recorded = capture.wrap_llm(backend)
        if native:
            recorded.generate_with_tools("prompt", tools=[])
        else:
            recorded.generate("prompt")
        trace = capture.flush()
        before = {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()}

        def unexpected_backoff(delay):
            pytest.fail(f"Exhausted replay must not retry after {delay}s")

        async def unexpected_async_backoff(delay):
            pytest.fail(f"Exhausted async replay must not retry after {delay}s")

        monkeypatch.setattr("looplet.scaffolding.time.sleep", unexpected_backoff)
        monkeypatch.setattr("looplet.async_loop.asyncio.sleep", unexpected_async_backoff)
        state = DefaultState(max_steps=2)
        replayed = ProvenanceSink(dir=tmp_path / "replay")
        options = {
            "tools": _make_tools(),
            "state": state,
            "recording_sink": replayed,
            "max_steps": 2,
        }
        if asynchronous:
            steps = [step async for step in async_replay_loop(trace, **options)]
        else:
            steps = list(replay_loop(trace, **options))
        result = RunResult.from_state(state)
        assert result.termination_reason == "replay_exhausted"
        assert not result.completed
        assert result.status.value == "failed"
        assert run_diagnostics(result)["termination_reason"] == "replay_exhausted"
        assert "replay exhausted" in steps[-1].tool_result.error
        saved = replayed.flush()
        assert json.loads((saved / "trajectory.json").read_text())["termination_reason"] == (
            "replay_exhausted"
        )
        assert {path: path.read_bytes() for path in trace.rglob("*") if path.is_file()} == before

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
    @pytest.mark.parametrize("json_output", [False, True])
    @pytest.mark.parametrize(
        "field",
        [
            "run_id",
            "termination_reason",
            "tool",
            "args_summary",
            "error",
            "result",
            "failure_mode",
            "call_index",
            "step_number",
            "total_items",
            "step_error",
            "result_args",
        ],
    )
    def test_saved_result_controls_are_inert_without_changing_evidence(
        self, tmp_path, capsys, command, json_output, field
    ):
        trace_dir = _captured_dir(tmp_path)
        payload = "probe\x1b[2K\r\b\a\x9b2K\x7f\x00\u03bb\n\tok"
        visible = "probe\\x1b[2K\\r\\x08\\x07\\x9b2K\\x7f\\x00\u03bb\n\tok"
        trajectory_path = trace_dir / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        terminal = trajectory["steps"][-1]
        if field in {"run_id", "termination_reason"}:
            trajectory[field] = payload
        elif field == "tool":
            terminal["tool_call"]["tool"] = payload
        elif field == "args_summary":
            terminal["args_summary"] = payload
        elif field == "error":
            terminal["tool_result"]["error"] = payload
        elif field == "result":
            terminal["tool_result"]["data"] = payload
        elif field == "failure_mode":
            trajectory["failure_modes"] = [payload]
        elif field == "step_number":
            terminal["step_num"] = payload
        elif field == "total_items":
            terminal["tool_result"]["data"] = {"observed": "marker"}
            terminal["tool_result"]["total_items"] = payload
        elif field == "step_error":
            terminal["error"] = payload
        elif field == "result_args":
            terminal.pop("args_summary", None)
            terminal["tool_result"]["args"] = payload
        else:
            terminal["llm_call_indices"] = [payload]
        trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
        before = {path.name: path.read_bytes() for path in trace_dir.iterdir() if path.is_file()}

        assert cli_main([command, str(trace_dir), *(["--json"] if json_output else [])]) == 0
        output = capsys.readouterr().out
        assert not any(
            (ord(character) < 32 and character not in "\n\t") or 127 <= ord(character) < 160
            for character in output
        )
        if json_output:
            assert json.loads(output)["trajectory"] == trajectory
        else:
            assert visible in output
        assert {
            path.name: path.read_bytes() for path in trace_dir.iterdir() if path.is_file()
        } == before

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize("json_output", [False, True])
    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("relative", [False, True])
    @pytest.mark.parametrize(
        "filename", ["trajectory.json", "manifest.jsonl", "evals.json", "artifacts.json"]
    )
    @pytest.mark.parametrize("destination", ["outside", "inside", "broken"])
    def test_trace_component_links_are_contained_before_open(
        self,
        tmp_path,
        capsys,
        monkeypatch,
        command,
        json_output,
        legacy,
        relative,
        filename,
        destination,
    ):
        trace_dir = _captured_dir(tmp_path)
        if legacy:
            (trace_dir / ARTIFACT_DESCRIPTOR).unlink()
        selected = trace_dir / filename
        if filename in {"evals.json", "artifacts.json"}:
            selected.write_text("[]" if filename == "evals.json" else "{}", encoding="utf-8")
        content = selected.read_bytes()
        parent = (
            tmp_path / f"{trace_dir.name}-private"
            if destination == "outside"
            else trace_dir / "payloads"
        )
        parent.mkdir()
        target = parent / filename
        if destination != "broken":
            target.write_bytes(content)
        selected.unlink()
        link = (
            Path("..") / parent.name / filename
            if destination == "outside"
            else Path("payloads") / filename
        )
        selected.symlink_to(link if relative else target)
        before = {
            path.relative_to(trace_dir).as_posix(): path.read_bytes()
            for path in trace_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        opened = []
        open_path = Path.open

        def record_open(path, *arguments, **keywords):
            if path.resolve() == target.resolve():
                opened.append(str(path))
            return open_path(path, *arguments, **keywords)

        with monkeypatch.context() as patcher:
            patcher.setattr(Path, "open", record_open)
            status = cli_main([command, str(trace_dir), *(["--json"] if json_output else [])])
        captured = capsys.readouterr()
        if not legacy and filename in {"evals.json", "artifacts.json"}:
            assert status == 0
            assert opened == []
            if json_output:
                assert set(json.loads(captured.out)) == {"trajectory", "manifest"}
            else:
                assert "saved evals:" not in captured.out
        elif destination == "inside":
            assert status == 0
            assert opened
            if json_output:
                assert json.loads(captured.out)["trajectory"]["termination_reason"] == "done"
        else:
            assert status == 1, {"opened": opened}
            assert opened == []
            assert captured.out == ""
            assert captured.err
        assert selected.readlink() == (link if relative else target)
        assert {
            path.relative_to(trace_dir).as_posix(): path.read_bytes()
            for path in trace_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        } == before
        if destination != "broken":
            assert target.read_bytes() == content

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize("json_output", [False, True])
    def test_all_terminal_controls_keep_json_and_layout_lossless(
        self, tmp_path, capsys, command, json_output
    ):
        payload = "unicode \u03bb\n\t" + "".join(
            chr(number) for number in (*range(32), *range(127, 160))
        )
        trajectory = {
            "termination_reason": "done",
            "steps": [{"tool_result": {"data": payload}}],
        }
        selected = tmp_path / "trajectory.json"
        selected.write_text(json.dumps(trajectory), encoding="utf-8")
        before = selected.read_bytes()

        assert cli_main([command, str(tmp_path), *(["--json"] if json_output else [])]) == 0
        output = capsys.readouterr().out
        assert not any(
            (ord(character) < 32 and character not in "\n\t") or 127 <= ord(character) < 160
            for character in output
        )
        if json_output:
            assert json.loads(output)["trajectory"] == trajectory
        else:
            assert "result:\nunicode \u03bb\n\t" in output
            for escaped in ("\\x00", "\\x1b", "\\x7f", "\\x9b", "\\x9f", "\\r"):
                assert escaped in output
        assert selected.read_bytes() == before

    @pytest.mark.parametrize("command", ["show", "inspect"])
    @pytest.mark.parametrize("json_output", [False, True])
    @pytest.mark.parametrize(
        "fault",
        [
            "missing",
            "empty",
            "trajectory_parse",
            "trajectory_shape",
            "manifest_parse",
            "manifest_shape",
            "manifest_read",
            "descriptor",
            "eval_parse",
        ],
    )
    def test_trace_diagnostic_controls_are_inert(
        self, tmp_path, capsys, command, json_output, fault
    ):
        trace_dir = tmp_path / "trace\x1b[2K\r\a"
        if fault != "missing":
            trace_dir.mkdir()
        if fault == "trajectory_parse":
            (trace_dir / "trajectory.json").write_text("not JSON", encoding="utf-8")
        elif fault == "trajectory_shape":
            (trace_dir / "trajectory.json").write_text("[]", encoding="utf-8")
        elif fault == "manifest_parse":
            (trace_dir / "manifest.jsonl").write_text("not JSON", encoding="utf-8")
        elif fault == "manifest_shape":
            (trace_dir / "manifest.jsonl").write_text("[]", encoding="utf-8")
        elif fault == "manifest_read":
            (trace_dir / "manifest.jsonl").mkdir()
        elif fault == "descriptor":
            (trace_dir / ARTIFACT_DESCRIPTOR).write_text("{}", encoding="utf-8")
        elif fault == "eval_parse":
            (trace_dir / "trajectory.json").write_text('{"steps":[]}', encoding="utf-8")
            (trace_dir / "evals.json").write_text("not JSON", encoding="utf-8")
        before = {
            path.relative_to(trace_dir).as_posix(): path.read_bytes()
            for path in trace_dir.rglob("*")
            if path.is_file()
        }

        assert cli_main([command, str(trace_dir), *(["--json"] if json_output else [])]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err
        assert not any(
            (ord(character) < 32 and character not in "\n\t") or 127 <= ord(character) < 160
            for character in captured.err
        )
        assert {
            path.relative_to(trace_dir).as_posix(): path.read_bytes()
            for path in trace_dir.rglob("*")
            if path.is_file()
        } == before

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
    @pytest.mark.parametrize("controls", [False, True])
    def test_saved_eval_evidence_is_visible_without_execution(
        self, tmp_path, capsys, command, json_output, legacy, controls
    ):
        calls = {"grader": 0, "collector": 0}
        suffix = " \x1b[2K\r\a\x9b2K\u03bb" if controls else ""

        @eval_mark("required")
        def eval_content(ctx):
            calls["grader"] += 1
            return EvalResult(
                name="eval_content" + suffix,
                score=0.0,
                label="fail" + suffix,
                explanation="The output file is missing." + suffix,
                details=["Expected hello in result.txt." + suffix],
                metrics={"observed" + suffix: 0.0},
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
        assert not any(
            (ord(character) < 32 and character not in "\n\t") or 127 <= ord(character) < 160
            for character in rendered
        )
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
            if controls:
                assert "\\x1b[2K\\r\\x07\\x9b2K\u03bb" in rendered
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


@pytest.mark.parametrize("use_sys_argv", [False, True])
def test_version_flag_needs_no_command_or_backend(tmp_path, capsys, monkeypatch, use_sys_argv):
    import looplet
    from looplet.__main__ import _build_check

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOOPLET_PROVIDER", "invalid-provider")
    monkeypatch.setattr(
        "looplet.backends.make_backend",
        lambda **kwargs: pytest.fail("Version lookup must not build a backend"),
    )
    monkeypatch.setattr(sys, "argv", ["looplet", "--version"])
    with pytest.raises(SystemExit) as stopped:
        cli_main(None if use_sys_argv else ["--version"])
    assert stopped.value.code == 0
    captured = capsys.readouterr()
    identity = _build_check(Path(looplet.__file__).parent)["detail"]
    assert identity.startswith("sha256:")
    assert " ".join(captured.out.split()) == f"looplet {looplet.__version__} ({identity})"
    assert captured.err == ""


def test_version_reports_unavailable_fingerprint_without_traceback(capsys, monkeypatch):
    def denied_read(source):
        raise PermissionError("private-path")

    monkeypatch.setattr(Path, "read_bytes", denied_read)
    with pytest.raises(SystemExit) as stopped:
        cli_main(["--version"])
    assert stopped.value.code == 0
    captured = capsys.readouterr()
    assert "Source fingerprint unavailable (PermissionError)" in captured.out
    assert "private-path" not in captured.out
    assert captured.err == ""


def test_help_does_not_compute_build_fingerprint(capsys, monkeypatch):
    def unexpected_build_check(package_root):
        pytest.fail("Help must not fingerprint library sources")

    monkeypatch.setattr("looplet.__main__._build_check", unexpected_build_check)
    with pytest.raises(SystemExit) as stopped:
        cli_main(["--help"])
    assert stopped.value.code == 0


@pytest.mark.parametrize("command", ["run", "blueprint"])
def test_step_budget_help_clarifies_free_calls(command, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli_main([command, "--help"])
    assert stopped.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "Charged-step budget" in output
    assert "successful free calls do not count" in output


class TestDoctorCLISmoke:
    @pytest.mark.parametrize("json_output", [False, True])
    def test_doctor_reports_build_fingerprint(self, capsys, json_output):
        arguments = ["doctor", "--no-backend"]
        if json_output:
            arguments.append("--json")
        assert cli_main(arguments) == 0
        output = capsys.readouterr().out
        if json_output:
            checks = json.loads(output)["checks"]
            build = next(check for check in checks if check["name"] == "build")
            assert build["status"] == "ok"
            detail = build["detail"]
        else:
            detail = next(line for line in output.splitlines() if line.startswith("OK build:"))
            detail = detail.removeprefix("OK build: ")
        assert detail.startswith("sha256:")
        assert len(detail.removeprefix("sha256:")) == 64
        int(detail.removeprefix("sha256:"), 16)

    def test_build_fingerprint_tracks_sources_without_git(self, tmp_path):
        from looplet.__main__ import _build_check

        first = tmp_path / "first" / "looplet"
        second = tmp_path / "second" / "looplet"
        for package_root in (first, second):
            (package_root / "nested").mkdir(parents=True)
            (package_root / "__init__.py").write_text("VERSION = 1\n")
            (package_root / "nested" / "tools.py").write_text("VALUE = 2\n")
        original = _build_check(first)
        assert original["status"] == "ok"
        assert _build_check(second) == original
        (second / "__pycache__").mkdir()
        (second / "__pycache__" / "tools.pyc").write_bytes(b"cache")
        assert _build_check(second) == original
        (second / "_bundled").mkdir()
        (second / "_bundled" / "example.py").write_text("EXAMPLE = 1\n")
        assert _build_check(second) == original
        source = second / "nested" / "tools.py"
        source.write_text("VALUE = 3\n")
        changed = _build_check(second)
        assert changed != original
        source.rename(source.with_name("renamed.py"))
        assert _build_check(second) != changed

    def test_build_fingerprint_reports_unreadable_sources(self, tmp_path, monkeypatch):
        from looplet.__main__ import _build_check

        (tmp_path / "__init__.py").write_text("VERSION = 1\n")

        def denied_read(source):
            raise PermissionError("private-path")

        monkeypatch.setattr(Path, "read_bytes", denied_read)
        assert _build_check(tmp_path) == {
            "name": "build",
            "status": "warn",
            "detail": "Source fingerprint unavailable (PermissionError)",
        }

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
