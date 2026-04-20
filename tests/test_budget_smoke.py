"""Smoke tests for context budgeting + threshold-based compaction."""
from __future__ import annotations

import pytest

from openharness import (
    BaseToolRegistry,
    ContextBudget,
    DefaultState,
    LoopConfig,
    ThresholdCompactHook,
    TruncateCompact,
    composable_loop,
)
from openharness.budget import BudgetTelemetry, classify_tier
from openharness.session import SessionLog
from openharness.tools import ToolSpec

pytestmark = pytest.mark.smoke


class TestContextBudget:
    def test_tier_ordering(self):
        b = ContextBudget(
            context_window=1000, warning_at=500, error_at=700, compact_buffer=100,
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
            context_window=200, warning_at=20, error_at=50, compact_buffer=10,
        )
        h = ThresholdCompactHook(b)
        log = SessionLog()
        for i in range(1, 20):
            log.record(
                step=i, theory="some theory",
                tool=f"tool_name_{i}",
                reasoning="long reasoning text " * 5,
                findings=[f"finding_{j}" for j in range(5)],
            )
        assert h.should_compact(None, log, None, 5) is True
        assert 5 in h.fired_at

    def test_warning_tier_fires_when_configured(self):
        b = ContextBudget(
            context_window=200, warning_at=10, error_at=100, compact_buffer=10,
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
        reg.register(ToolSpec(
            name="echo", description="e",
            parameters={"msg": "str"},
            execute=lambda *, msg: {"msg": msg},
        ))
        reg.register(ToolSpec(
            name="done", description="d",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        ))

        class _Backend:
            def __init__(self):
                self._r = [
                    '{"tool":"echo","args":{"msg":"a"},"reasoning":"r"}',
                    '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
                ]
            def generate(self, *a, **k): return self._r.pop(0)

        class _CountingCompact(TruncateCompact):
            calls = 0
            def compact(self, **kw):
                type(self).calls += 1
                return super().compact(**kw)

        # Very tight budget — forces fire on step 2.
        b = ContextBudget(
            context_window=100, warning_at=5, error_at=10, compact_buffer=5,
        )
        svc = _CountingCompact()
        list(composable_loop(
            llm=_Backend(), tools=reg, state=DefaultState(max_steps=3),
            hooks=[ThresholdCompactHook(b, fire_tier="warning")],
            config=LoopConfig(max_steps=3, compact_service=svc),
        ))
        assert _CountingCompact.calls >= 1
