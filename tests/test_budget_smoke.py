"""Smoke tests for context budgeting + threshold-based compaction."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    ContextBudget,
    DefaultState,
    LoopConfig,
    ThresholdCompactHook,
    TruncateCompact,
    composable_loop,
)
from looplet.budget import BudgetTelemetry, classify_tier
from looplet.session import SessionLog
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


class TestDefaultHarness:
    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("explicit_service", [False, True])
    async def test_setup_failure_preserves_caller_compact_service(
        self, tmp_path, asynchronous, explicit_service
    ):
        from looplet import MockLLMBackend, async_composable_loop, tools_from

        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        service = TruncateCompact() if explicit_service else None
        config = LoopConfig(run_dir=str(blocked), compact_service=service)
        backend = MockLLMBackend([])
        arguments = dict(llm=backend, tools=tools_from([], include_done=True), config=config)
        with pytest.raises(FileExistsError):
            if asynchronous:
                async for _step in async_composable_loop(**arguments):
                    pass
            else:
                for _step in composable_loop(**arguments):
                    pass
        assert config.compact_service is service
        assert config.run_dir == str(blocked)
        assert config.checkpoint_dir is None
        assert backend.calls == 0

    @pytest.mark.parametrize("context_window", [4, 4096, 32768, 262144])
    def test_execution_defaults_do_not_prescribe_strategy(self, context_window):
        from looplet import DefaultCompactService, ToolCallBudgetHook, resolve_harness
        from looplet.stagnation import StagnationHook

        resolved = resolve_harness(LoopConfig(context_window=context_window))

        assert resolved.describe()["profile"] == "execution-v1"
        assert isinstance(resolved.config.compact_service, DefaultCompactService)
        assert sum(isinstance(hook, ThresholdCompactHook) for hook in resolved.hooks) == 1
        assert not any(
            isinstance(hook, (StagnationHook, ToolCallBudgetHook)) for hook in resolved.hooks
        )
        assert resolved.describe()["max_tool_calls"] is None

    def test_execution_profile_preserves_explicit_policies(self):
        from looplet import ToolCallBudgetHook, resolve_harness
        from looplet.stagnation import StagnationHook

        supplied = [StagnationHook(), ToolCallBudgetHook(max_calls=7)]
        resolved = resolve_harness(LoopConfig(max_model_calls=12), supplied)

        assert resolved.hooks[:2] == supplied
        assert resolved.describe()["max_model_calls"] == 12
        assert resolved.describe()["max_tool_calls"] == 7

    def test_standard_defaults_are_composed_without_mutating_hooks(self):
        from looplet import DefaultCompactService, resolve_harness
        from looplet.stagnation import StagnationHook

        supplied_hooks = []
        resolved = resolve_harness(
            config=LoopConfig(harness_profile="standard-v1"), hooks=supplied_hooks
        )

        assert isinstance(resolved.config.compact_service, DefaultCompactService)
        assert sum(isinstance(hook, ThresholdCompactHook) for hook in resolved.hooks) == 1
        assert sum(isinstance(hook, StagnationHook) for hook in resolved.hooks) == 1
        assert resolved.config.max_model_calls is None
        assert resolved.describe()["max_tool_calls"] == 60
        assert supplied_hooks == []

    def test_explicit_compaction_policy_replaces_default(self):
        from looplet import resolve_harness

        service = TruncateCompact(keep_recent=4)
        hook = ThresholdCompactHook(
            ContextBudget(
                context_window=32_000,
                warning_at=16_000,
                error_at=24_000,
                compact_buffer=3200,
            )
        )
        config = LoopConfig(compact_service=service)
        resolved = resolve_harness(config=config, hooks=[hook])

        assert resolved.config.compact_service is service
        assert [item for item in resolved.hooks if isinstance(item, ThresholdCompactHook)] == [hook]

    def test_minimal_and_individual_opt_outs_preserve_explicit_policies(self):
        from looplet import resolve_harness

        minimal = resolve_harness(LoopConfig(harness_profile="minimal"))
        assert minimal.config.compact_service is None
        assert minimal.hooks == []

        no_compaction = resolve_harness(LoopConfig(disabled_defaults=["compaction"]))
        assert no_compaction.config.compact_service is None
        assert not any(isinstance(hook, ThresholdCompactHook) for hook in no_compaction.hooks)

        service = TruncateCompact()
        explicit = resolve_harness(
            LoopConfig(disabled_defaults=["compaction"], compact_service=service)
        )
        assert explicit.config.compact_service is service

    def test_unknown_defaults_fail_loudly(self):
        from looplet import resolve_harness

        with pytest.raises(ValueError, match="harness_profile"):
            resolve_harness(LoopConfig(harness_profile="standard-v99"))
        with pytest.raises(ValueError, match="disabled_defaults"):
            resolve_harness(LoopConfig(disabled_defaults=["compcat"]))

    def test_persistence_resolution_does_not_create_files(self, tmp_path):
        from looplet import resolve_harness

        directory = tmp_path / "not-created"
        resolved = resolve_harness(LoopConfig(run_dir=str(directory)))
        assert not directory.exists()
        assert resolved.describe()["persistence"] == {
            "enabled": True,
            "run_dir": str(directory),
            "defaults": ["history", "result_retention", "provenance", "checkpointing", "branching"],
        }

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize(
        "profile,disabled", [("minimal", []), ("execution-v1", ["persistence"])]
    )
    async def test_persistence_opt_out_performs_no_automatic_writes(
        self, tmp_path, asynchronous, profile, disabled
    ):
        from looplet import AsyncMockLLMBackend, MockLLMBackend, async_composable_loop, tools_from

        directory = tmp_path / "not-created"
        state = DefaultState(max_steps=1)
        history = state.steps
        tools = tools_from([], include_done=True)
        responses = ['{"tool":"done","args":{"summary":"ok"}}']
        arguments = dict(
            tools=tools,
            state=state,
            config=LoopConfig(
                max_steps=1,
                use_native_tools=False,
                run_dir=str(directory),
                harness_profile=profile,
                disabled_defaults=disabled,
            ),
        )
        if asynchronous:
            async for _ in async_composable_loop(llm=AsyncMockLLMBackend(responses), **arguments):
                pass
        else:
            list(composable_loop(llm=MockLLMBackend(responses), **arguments))
        assert state.run_status == "completed"
        assert state.steps is history
        assert tools.resources == {}
        assert not directory.exists()

    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_explicit_persistence_components_replace_defaults(self, tmp_path, asynchronous):
        from looplet import AsyncMockLLMBackend, MockLLMBackend, async_composable_loop, tools_from
        from looplet.provenance import (
            AsyncRecordingLLMBackend,
            RecordingLLMBackend,
            TrajectoryRecorder,
        )
        from looplet.result_store import FileToolResultStore
        from looplet.session_tree import SessionTree, SessionTreeRecorder

        class HostState(DefaultState):
            pass

        class HostCheckpoint:
            checkpoint_owner = True

            def __init__(self):
                self.steps = []

            def post_step(self, state, session_log, step_num):
                self.steps.append(step_num)

        state = HostState(max_steps=1)
        history = state.steps
        tools = tools_from([], include_done=True)
        store = FileToolResultStore(str(tmp_path / "host-results"))
        tools.set_resources({"tool_result_store": store})
        responses = ['{"tool":"done","args":{"summary":"ok"}}']
        backend = (
            AsyncRecordingLLMBackend(AsyncMockLLMBackend(responses))
            if asynchronous
            else RecordingLLMBackend(MockLLMBackend(responses))
        )
        recorder = TrajectoryRecorder(recording_llm=backend)
        tree = SessionTree()
        directory = tmp_path / "defaults"
        config = LoopConfig(max_steps=1, use_native_tools=False, run_dir=str(directory))
        checkpoint = HostCheckpoint()
        hooks = [recorder, SessionTreeRecorder(tree), checkpoint]
        arguments = dict(llm=backend, tools=tools, state=state, config=config, hooks=hooks)
        if asynchronous:
            async for _ in async_composable_loop(**arguments):
                pass
        else:
            list(composable_loop(**arguments))
        assert state.run_status == "completed"
        assert state.steps is history
        assert tools.resources["tool_result_store"] is store
        assert len(recorder.trajectory.steps) == 1
        assert len(recorder.trajectory.llm_calls) == 1
        assert len(tree.nodes) == 2
        assert checkpoint.steps == [1]
        assert not (directory / "traces").exists()
        assert not (directory / "checkpoints").exists()
        assert not (directory / "session.jsonl").exists()
        assert config.checkpoint_dir is None
        assert config.initial_checkpoint is None
        assert len(hooks) == 3

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("host_replaces_service", [False, True])
    async def test_run_service_ownership_and_completion_at_dispatch_cap(
        self, is_async, host_replaces_service
    ):
        from looplet import (
            AsyncMockLLMBackend,
            DefaultCompactService,
            MockLLMBackend,
            RunResult,
            ToolCallBudgetHook,
            async_composable_loop,
            tools_from,
        )

        config = LoopConfig(max_steps=1)
        replacement = TruncateCompact(keep_recent=7)
        active_services = []

        class ObserveConfig:
            def bind(self, context):
                assert context.config is config
                active_services.append(context.config.compact_service)
                if host_replaces_service:
                    config.compact_service = replacement

        state = DefaultState(max_steps=1)
        arguments = {
            "tools": tools_from([], include_done=True),
            "state": state,
            "config": config,
            "hooks": [ObserveConfig(), ToolCallBudgetHook(max_calls=1)],
        }
        responses = ['{"tool":"done","args":{"summary":"ok"}}']
        if is_async:
            async for _ in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=responses), **arguments
            ):
                pass
        else:
            list(composable_loop(llm=MockLLMBackend(responses=responses), **arguments))
        assert isinstance(active_services[0], DefaultCompactService)
        assert config.compact_service is (replacement if host_replaces_service else None)
        assert RunResult.from_state(state).completed

    def test_loop_compacts_under_pressure_without_a_supplied_hook(self):
        from looplet import MockLLMBackend, tools_from

        class CountingCompact(TruncateCompact):
            def __init__(self):
                super().__init__()
                self.reasons = []

            def compact(self, **kwargs):
                self.reasons.append(kwargs["reason"])
                return super().compact(**kwargs)

        def echo(message: str) -> dict:
            return {"message": message}

        import json

        responses = [
            json.dumps({"tool": "echo", "args": {"message": "x" * 6000}}) for _ in range(3)
        ]
        responses.append('{"tool":"done","args":{"answer":"ok"}}')
        service = CountingCompact()
        steps = list(
            composable_loop(
                llm=MockLLMBackend(responses=responses),
                tools=tools_from(
                    [echo], include_done=True, done_parameters={"answer": "Final answer"}
                ),
                config=LoopConfig(
                    max_steps=5,
                    context_window=8000,
                    build_prompt=lambda **kwargs: "short prompt",
                    compact_service=service,
                ),
            )
        )
        assert len(steps) == 4
        assert "proactive" in service.reasons

    def test_cartridge_defaults_and_explicit_minimal_round_trip(self, tmp_path):
        from looplet import cartridge_to_preset, preset_to_cartridge, scaffold_cartridge

        root = scaffold_cartridge(tmp_path / "original", name="defaults_agent", tools=[])
        with cartridge_to_preset(root) as preset:
            standard = preset.resolve_harness().describe()
            assert standard["profile"] == "execution-v1"
            assert standard["hooks"] == ["ThresholdCompactHook"]
            preset.config.harness_profile = "minimal"
            preset.config.disabled_defaults = ["stagnation"]
            preset.config.run_dir = str(tmp_path / "evidence")
            preset_to_cartridge(preset, tmp_path / "exported")

        with cartridge_to_preset(tmp_path / "exported") as restored:
            assert restored.config.harness_profile == "minimal"
            assert restored.config.disabled_defaults == ["stagnation"]
            assert restored.config.run_dir == str(tmp_path / "evidence")
            assert restored.resolve_harness().hooks == []

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_successful_free_calls_are_bounded(self, is_async):
        from looplet import AsyncMockLLMBackend, MockLLMBackend, RunResult, async_composable_loop
        from looplet.tools import register_done_tool

        registry = BaseToolRegistry()
        registry.register(
            ToolSpec("poll", "Poll progress", {}, lambda: {"status": "waiting"}, free=True)
        )
        register_done_tool(registry)
        responses = ['{"tool":"poll","args":{}}'] * 40
        state = DefaultState(max_steps=1)
        arguments = {
            "tools": registry,
            "state": state,
            "config": LoopConfig(max_steps=1, harness_profile="standard-v1"),
        }
        if is_async:
            steps = [
                step
                async for step in async_composable_loop(
                    llm=AsyncMockLLMBackend(responses=responses), **arguments
                )
            ]
        else:
            steps = list(composable_loop(llm=MockLLMBackend(responses=responses), **arguments))
        assert len(steps) == 16
        assert RunResult.from_state(state).termination_reason == "tool_call_budget"

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_concurrent_tool_budget_bounds_actual_side_effects(self, is_async):
        import json

        from looplet import (
            AsyncMockLLMBackend,
            MockLLMBackend,
            RunResult,
            ToolCallBudgetHook,
            async_composable_loop,
        )
        from looplet.tools import register_done_tool

        dispatched = []

        def poll(index):
            dispatched.append(index)
            return {"index": index}

        registry = BaseToolRegistry()
        register_done_tool(registry)
        registry.register(
            ToolSpec(
                "poll",
                "Poll progress",
                {"index": "Progress index"},
                poll,
                free=True,
                concurrent_safe=True,
            )
        )
        response = json.dumps(
            {"tools": [{"tool": "poll", "args": {"index": index}} for index in range(40)]}
        )
        state = DefaultState(max_steps=40)
        arguments = {
            "tools": registry,
            "state": state,
            "hooks": [ToolCallBudgetHook(max_calls=16)],
            "config": LoopConfig(max_steps=40, concurrent_dispatch=True, use_native_tools=False),
        }
        if is_async:
            async for _ in async_composable_loop(
                llm=AsyncMockLLMBackend(responses=[response]), **arguments
            ):
                pass
        else:
            list(composable_loop(llm=MockLLMBackend(responses=[response]), **arguments))

        assert sorted(dispatched) == list(range(16))
        assert RunResult.from_state(state).termination_reason == "tool_call_budget"


class TestContextBudget:
    def test_rejects_contradictory_thresholds(self):
        with pytest.raises(ValueError, match="compact_buffer"):
            ContextBudget(context_window=100, compact_buffer=100)
        with pytest.raises(ValueError, match="thresholds"):
            ContextBudget(context_window=20_000, warning_at=900, error_at=800)

    def test_tier_ordering(self):
        b = ContextBudget(
            context_window=1000,
            warning_at=500,
            error_at=700,
            compact_buffer=100,
        )
        assert b.blocking_at == 900
        assert b.classify(100) == "ok"
        assert b.classify(500) == "warning"
        assert b.classify(700) == "error"
        assert b.classify(900) == "blocking"

    def test_classify_tier_helper(self):
        b = ContextBudget(context_window=1000, warning_at=500, error_at=700, compact_buffer=100)
        log = SessionLog()
        for i in range(1, 6):
            log.record(step=i, theory="t", tool="x", reasoning="r")
        tier, est = classify_tier(b, session_log=log)
        assert tier in ("ok", "warning", "error", "blocking")
        assert est > 0


class TestThresholdCompactHook:
    def test_no_fire_when_under_threshold(self):
        b = ContextBudget(context_window=10_000_000, warning_at=5_000_000, error_at=8_000_000)
        h = ThresholdCompactHook(b)
        log = SessionLog()
        log.record(step=1, theory="t", tool="x", reasoning="r")
        assert h.should_compact(None, log, None, 1) is False
        assert h.fired_at == []

    def test_fires_when_over_error_threshold(self):
        # Tight budget so any small log crosses error tier.
        b = ContextBudget(
            context_window=200,
            warning_at=20,
            error_at=50,
            compact_buffer=10,
        )
        h = ThresholdCompactHook(b)
        log = SessionLog()
        for i in range(1, 20):
            log.record(
                step=i,
                theory="some theory",
                tool=f"tool_name_{i}",
                reasoning="long reasoning text " * 5,
                findings=[f"finding_{j}" for j in range(5)],
            )
        assert h.should_compact(None, log, None, 5) is True
        assert 5 in h.fired_at

    def test_warning_tier_fires_when_configured(self):
        b = ContextBudget(
            context_window=200,
            warning_at=10,
            error_at=100,
            compact_buffer=10,
        )
        h = ThresholdCompactHook(b, fire_tier="warning")
        log = SessionLog()
        for i in range(1, 5):
            log.record(step=i, theory="t", tool="x", reasoning="r " * 20)
        assert h.should_compact(None, log, None, 1) is True


class TestBudgetTelemetry:
    def test_samples_collected(self):
        b = ContextBudget()
        t = BudgetTelemetry(b)
        log = SessionLog()
        log.record(step=1, theory="t", tool="x", reasoning="r")
        t.pre_prompt(None, log, None, 1)
        t.pre_prompt(None, log, None, 2)
        assert len(t.samples) == 2
        assert t.peak_tier == "ok"


class TestLoopIntegration:
    def test_threshold_hook_triggers_compact(self):
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="echo",
                description="e",
                parameters={"msg": "str"},
                execute=lambda *, msg: {"msg": msg},
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"answer": "str"},
                execute=lambda *, answer: {"answer": answer},
            )
        )

        class _Backend:
            def __init__(self):
                self._r = [
                    '{"tool":"echo","args":{"msg":"a"},"reasoning":"r"}',
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]

            def generate(self, *a, **k):
                return self._r.pop(0)

        class _CountingCompact(TruncateCompact):
            calls = 0

            def compact(self, **kw):
                type(self).calls += 1
                return super().compact(**kw)

        # Very tight budget - forces fire on step 2.
        b = ContextBudget(
            context_window=100,
            warning_at=5,
            error_at=10,
            compact_buffer=5,
        )
        svc = _CountingCompact()
        list(
            composable_loop(
                llm=_Backend(),
                tools=reg,
                state=DefaultState(max_steps=3),
                hooks=[ThresholdCompactHook(b, fire_tier="warning")],
                config=LoopConfig(max_steps=3, compact_service=svc, use_native_tools=False),
            )
        )
        assert _CountingCompact.calls >= 1
