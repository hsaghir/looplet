"""Smoke tests for :class:`SummarizeCompact`."""
from __future__ import annotations

import pytest

from looplet import SummarizeCompact
from looplet.session import SessionLog

pytestmark = pytest.mark.smoke


class _SummaryLLM:
    """Backend that returns a fixed summary."""
    def __init__(self, text="TASK: demo. FINDINGS: a, b. OPEN: c."):
        self.text = text
        self.calls = 0

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.calls += 1
        return self.text


class _FailingLLM:
    def generate(self, *a, **k):
        raise RuntimeError("boom")


def _populated_log() -> SessionLog:
    log = SessionLog()
    for i in range(1, 6):
        log.record(
            step=i, theory="t", tool=f"tool_{i}",
            reasoning=f"step {i}",
            findings=[f"f-{i}"],
        )
    return log


class TestLLMCompact:
    def test_happy_path_inserts_summary(self):
        svc = SummarizeCompact(keep_recent=2)
        log = _populated_log()
        before = len(log.entries)
        llm = _SummaryLLM()
        out = svc.compact(
            state=object(), session_log=log, llm=llm,
            conversation=None, step_num=5, reason="test",
        )
        assert llm.calls == 1
        assert out.llm_calls_spent == 1
        assert out.extra["mode"] == "llm_summary"
        # Keep-recent dropped the middle; summary was appended.
        summary_entries = [e for e in log.entries if e.tool == "__compact_summary__"]
        assert len(summary_entries) == 1
        # Net entries: keep_recent + 1 summary <= before
        assert len(log.entries) <= before

    def test_summary_fallback_on_llm_error(self):
        svc = SummarizeCompact(keep_recent=1)
        log = _populated_log()
        out = svc.compact(
            state=object(), session_log=log, llm=_FailingLLM(),
            conversation=None, step_num=5, reason="test",
        )
        # Fallback mode — no summary entry, but deterministic keep_recent ran.
        assert out.extra["mode"] == "llm_fallback"
        summary_entries = [e for e in log.entries if e.tool == "__compact_summary__"]
        assert summary_entries == []

    def test_empty_log_short_circuits(self):
        svc = SummarizeCompact()
        out = svc.compact(
            state=object(), session_log=SessionLog(),
            llm=_SummaryLLM(), conversation=None,
            step_num=0, reason="test",
        )
        assert out.extra["mode"] == "empty_fallback"
        assert out.llm_calls_spent == 0
