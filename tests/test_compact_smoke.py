"""Smoke tests for :mod:`looplet.compact` and the compaction lifecycle."""
from __future__ import annotations

import pytest

from looplet import (
    CompactOutcome,
    CompactService,
    EventPayload,
    LifecycleEvent,
    TruncateCompact,
    run_compact,
)

pytestmark = pytest.mark.smoke


class TestDefaultCompactService:
    def test_default_service_is_compact_service(self):
        assert isinstance(TruncateCompact(), CompactService)

    def test_default_service_returns_outcome(self):
        svc = TruncateCompact()
        # Minimal mutable state + session_log stubs.
        state = type("S", (), {"steps": []})()
        from looplet.session import SessionLog
        sl = SessionLog()
        outcome = svc.compact(
            state=state, session_log=sl, llm=None,
            conversation=None, step_num=0, reason="test",
        )
        assert isinstance(outcome, CompactOutcome)
        assert outcome.reason == "test"


class TestRunCompactEvents:
    def test_pre_and_post_compact_events_fire(self):
        seen: list[LifecycleEvent] = []

        class Observer:
            def on_event(self, payload: EventPayload):
                seen.append(payload.event)
                return None

        state = type("S", (), {"steps": []})()
        from looplet.session import SessionLog
        sl = SessionLog()
        outcome = run_compact(
            TruncateCompact(),
            hooks=[Observer()],
            state=state, session_log=sl, llm=None, conversation=None,
            step_num=0, reason="test",
        )
        assert LifecycleEvent.PRE_COMPACT in seen
        assert LifecycleEvent.POST_COMPACT in seen
        assert outcome.reason == "test"

    def test_pre_compact_can_abort(self):
        from looplet import HookDecision

        class Aborter:
            def on_event(self, payload: EventPayload):
                if payload.event == LifecycleEvent.PRE_COMPACT:
                    return HookDecision(stop="do not compact")
                return None

        # A service that would raise if called — proves the abort worked.
        class AngryService:
            def compact(self, **kwargs):
                raise RuntimeError("should not run")

        state = type("S", (), {"steps": []})()
        from looplet.session import SessionLog
        sl = SessionLog()
        outcome = run_compact(
            AngryService(),
            hooks=[Aborter()],
            state=state, session_log=sl, llm=None, conversation=None,
            step_num=0, reason="x",
        )
        assert "aborted" in outcome.reason

    def test_custom_service_replaces_default(self):
        class Counter:
            def __init__(self):
                self.calls = 0

            def compact(self, **kwargs):
                self.calls += 1
                return CompactOutcome(reason="counter", llm_calls_spent=7)

        svc = Counter()
        state = type("S", (), {"steps": []})()
        from looplet.session import SessionLog
        sl = SessionLog()
        outcome = run_compact(
            svc, hooks=[],
            state=state, session_log=sl, llm=None, conversation=None,
            step_num=0, reason="x",
        )
        assert svc.calls == 1
        assert outcome.llm_calls_spent == 7


class TestLoopIntegration:
    def test_compact_service_is_a_LoopConfig_field(self):
        from looplet import LoopConfig
        cfg = LoopConfig(compact_service=TruncateCompact())
        assert cfg.compact_service is not None
