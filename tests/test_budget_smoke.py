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
    def test_standard_defaults_are_composed_without_mutating_hooks(self):
        from looplet import DefaultCompactService, resolve_harness
        from looplet.stagnation import StagnationHook

        supplied_hooks = []
        resolved = resolve_harness(config=LoopConfig(), hooks=supplied_hooks)

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
            assert standard["profile"] == "standard-v1"
            assert standard["hooks"] == [
                "ThresholdCompactHook",
                "StagnationHook",
                "ToolCallBudgetHook",
            ]
            preset.config.harness_profile = "minimal"
            preset.config.disabled_defaults = ["stagnation"]
            preset_to_cartridge(preset, tmp_path / "exported")

        with cartridge_to_preset(tmp_path / "exported") as restored:
            assert restored.config.harness_profile == "minimal"
            assert restored.config.disabled_defaults == ["stagnation"]
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
        arguments = {"tools": registry, "state": state, "config": LoopConfig(max_steps=1)}
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
