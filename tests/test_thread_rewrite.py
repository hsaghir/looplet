"""T8 dogfood: declarative ``rewrite_thread`` effect (portable compaction).

``CompactOutcome.cleanup`` is a Python closure — it cannot cross a
process boundary, so an out-of-process / cross-runtime compactor cannot
use it. T8 adds a declarative, JSON-safe equivalent:
:func:`~looplet.hook_decision.RewriteThread` /
``HookDecision.rewrite_thread`` and ``CompactOutcome.follow_up``, applied
by :func:`looplet.compact.apply_thread_rewrite`.
"""

from __future__ import annotations

import pytest

from looplet import DefaultState, HookDecision, RewriteThread
from looplet.compact import CompactOutcome, apply_thread_rewrite, run_compact

pytestmark = pytest.mark.smoke


class TestRewriteThreadEffect:
    def test_constructor_builds_spec(self):
        d = RewriteThread(
            reset_metadata_keys=["cache", "baseline"],
            metadata_updates={"token_baseline": 0},
        )
        assert d.rewrite_thread == {
            "reset_metadata_keys": ["cache", "baseline"],
            "metadata_updates": {"token_baseline": 0},
        }

    def test_wire_roundtrip_preserves_rewrite_thread(self):
        d = RewriteThread(reset_metadata_keys=["cache"], metadata_updates={"x": 1})
        wire = d.to_wire()
        # JSON-safe: only primitives / lists / dicts.
        assert wire["rewrite_thread"] == {
            "reset_metadata_keys": ["cache"],
            "metadata_updates": {"x": 1},
        }
        back = HookDecision.from_wire(wire)
        assert back is not None
        assert back.rewrite_thread == d.rewrite_thread

    def test_ergonomic_rewrite_thread_kind(self):
        back = HookDecision.from_wire(
            {"kind": "RewriteThread", "rewrite_thread": {"reset_metadata_keys": ["c"]}}
        )
        assert back is not None
        assert back.rewrite_thread == {"reset_metadata_keys": ["c"]}

    def test_rewrite_thread_not_noop(self):
        assert RewriteThread(reset_metadata_keys=["c"]).is_noop() is False


class TestApplyThreadRewrite:
    def test_reset_and_update_metadata(self):
        state = DefaultState()
        state.metadata["cache"] = {"big": "blob"}
        state.metadata["keep"] = "yes"
        apply_thread_rewrite(
            state,
            {"reset_metadata_keys": ["cache"], "metadata_updates": {"baseline": 0}},
        )
        assert "cache" not in state.metadata
        assert state.metadata["keep"] == "yes"
        assert state.metadata["baseline"] == 0

    def test_none_spec_is_safe(self):
        state = DefaultState()
        state.metadata["keep"] = "yes"
        apply_thread_rewrite(state, None)
        assert state.metadata == {"keep": "yes"}


class _NoopCompactService:
    """Minimal service returning a fixed outcome with a declarative follow-up."""

    def compact(self, *, state, session_log, llm, conversation, step_num, reason):
        return CompactOutcome(
            reason=reason,
            follow_up={
                "reset_metadata_keys": ["token_cache"],
                "metadata_updates": {"compacted": True},
            },
        )


class TestRunCompactAppliesFollowUp:
    def test_follow_up_applied_after_post_compact(self):
        state = DefaultState()
        state.metadata["token_cache"] = [1, 2, 3]
        outcome = run_compact(
            service=_NoopCompactService(),
            hooks=[],
            state=state,
            session_log=None,
            llm=None,
            conversation=None,
            step_num=1,
            reason="pressure",
        )
        assert outcome.reason == "pressure"
        assert "token_cache" not in state.metadata
        assert state.metadata["compacted"] is True

    def test_post_compact_hook_rewrite_thread_applied(self):
        class _RewriteHook:
            def on_event(self, payload):
                from looplet.events import LifecycleEvent

                if payload.event == LifecycleEvent.POST_COMPACT:
                    return RewriteThread(reset_metadata_keys=["hook_cache"])
                return None

        state = DefaultState()
        state.metadata["hook_cache"] = "x"
        run_compact(
            service=_NoopCompactService(),
            hooks=[_RewriteHook()],
            state=state,
            session_log=None,
            llm=None,
            conversation=None,
            step_num=1,
            reason="pressure",
        )
        assert "hook_cache" not in state.metadata
