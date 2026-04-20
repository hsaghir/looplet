"""Tests for openharness.session — LogEntry and SessionLog."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# ── LogEntry tests ────────────────────────────────────────────────


class TestLogEntryDefaults:
    def test_creation_required_fields(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=1, theory="T1", tool="search", reasoning="find stuff")
        assert entry.step == 1
        assert entry.theory == "T1"
        assert entry.tool == "search"
        assert entry.reasoning == "find stuff"

    def test_optional_fields_default(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=1, theory="T", tool="t", reasoning="r")
        assert entry.entities_seen == []
        assert entry.findings == []
        assert entry.highlights == []
        assert entry.recall_key == ""

    def test_no_iocs_found_property(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=1, theory="T", tool="t", reasoning="r")
        assert not hasattr(entry, "iocs_found"), "iocs_found alias must be removed"

    def test_render_basic(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=2, theory="T", tool="query", reasoning="check events")
        text = entry.render()
        assert "S2" in text
        assert "query" in text

    def test_render_with_entities(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=1, theory="T", tool="t", reasoning="r",
                         entities_seen=["host-1", "user-bob"])
        text = entry.render()
        assert "host-1" in text
        assert "user-bob" in text

    def test_render_with_highlights(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=1, theory="T", tool="t", reasoning="r",
                         highlights=["important-item"])
        text = entry.render(highlights_label="notable")
        assert "notable" in text
        assert "important-item" in text

    def test_render_with_recall_key(self):
        from openharness.session import LogEntry
        entry = LogEntry(step=1, theory="T", tool="t", reasoning="r",
                         recall_key="result_001")
        text = entry.render()
        assert "result_001" in text


# ── SessionLog.record tests ───────────────────────────────────────


class TestSessionLogRecord:
    def test_record_creates_entry(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="H1", tool="search", reasoning="looking")
        assert len(log.entries) == 1
        assert log.entries[0].step == 1

    def test_record_persists_theory(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="H1", tool="a", reasoning="r")
        log.record(step=2, theory="", tool="b", reasoning="r2")
        assert log.entries[1].theory == "H1"

    def test_record_updates_current_theory(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="H1", tool="a", reasoning="r")
        assert log.current_theory == "H1"
        log.record(step=2, theory="H2", tool="b", reasoning="r2")
        assert log.current_theory == "H2"

    def test_record_entities_and_findings(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="T", tool="t", reasoning="r",
                   entities=["host-a"], findings=["found something"])
        e = log.entries[0]
        assert "host-a" in e.entities_seen
        assert "found something" in e.findings

    def test_record_highlights(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="T", tool="t", reasoning="r",
                   highlights=["key-item"])
        assert "key-item" in log.entries[0].highlights


# ── SessionLog.render tests ───────────────────────────────────────


class TestSessionLogRender:
    def test_render_empty(self):
        from openharness.session import SessionLog
        log = SessionLog()
        assert log.render() == ""

    def test_render_contains_title(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="T", tool="search", reasoning="r")
        text = log.render()
        assert "SESSION LOG" in text

    def test_render_custom_title(self):
        from openharness.session import SessionLog
        log = SessionLog(title="MY AGENT LOG")
        log.record(step=1, theory="T", tool="t", reasoning="r")
        assert "MY AGENT LOG" in log.render()

    def test_render_custom_highlights_label(self):
        from openharness.session import SessionLog
        log = SessionLog(highlights_label="notable items")
        log.record(step=1, theory="T", tool="t", reasoning="r",
                   highlights=["item-x"])
        assert "notable items" in log.render()

    def test_render_shows_theory(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="root cause is X", tool="t", reasoning="r")
        assert "root cause is X" in log.render()

    def test_render_no_legacy_terms(self):
        import openharness.session as sm
        from openharness.session import LogEntry, SessionLog
        src = open(sm.__file__).read().lower()
        for term in ["iocs_found"]:
            assert term not in src, f"Forbidden term '{term}' found in session.py"


# ── SessionLog.all_entities tests ────────────────────────────────


class TestAllEntities:
    def test_aggregates_across_steps(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="T", tool="a", reasoning="r",
                   entities=["host-1", "user-a"])
        log.record(step=2, theory="T", tool="b", reasoning="r",
                   entities=["host-2"])
        ents = log.all_entities()
        assert "host-1" in ents
        assert "user-a" in ents
        assert "host-2" in ents

    def test_includes_highlights(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="T", tool="t", reasoning="r",
                   highlights=["notable-thing"])
        ents = log.all_entities()
        assert "notable-thing" in ents

    def test_returns_set(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="T", tool="t", reasoning="r",
                   entities=["x", "x"])
        ents = log.all_entities()
        assert isinstance(ents, set)
        assert len(ents) == 1


# ── SessionLog.render_compact tests ──────────────────────────────


class TestRenderCompact:
    def _build_log(self):
        from openharness.session import SessionLog
        log = SessionLog()
        log.record(step=1, theory="H1", tool="search", reasoning="first step",
                   entities=["host-a"], findings=["found A"])
        log.record(step=2, theory="H1", tool="query", reasoning="second step",
                   entities=["host-b"], findings=["found B"])
        log.record(step=3, theory="H2", tool="search", reasoning="third step",
                   highlights=["key-item"])
        return log

    def test_compact_not_empty(self):
        log = self._build_log()
        text = log.render_compact()
        assert text

    def test_compact_contains_step_range(self):
        log = self._build_log()
        text = log.render_compact()
        assert "1" in text
        assert "3" in text

    def test_compact_contains_tools(self):
        log = self._build_log()
        text = log.render_compact()
        assert "search" in text
        assert "query" in text

    def test_compact_contains_findings(self):
        log = self._build_log()
        text = log.render_compact()
        assert "found A" in text
        assert "found B" in text

    def test_compact_deterministic(self):
        log = self._build_log()
        assert log.render_compact() == log.render_compact()

    def test_compact_empty_log(self):
        from openharness.session import SessionLog
        log = SessionLog()
        assert log.render_compact() == ""


# ── SessionLog.compact tests ─────────────────────────────────────


class TestCompact:
    def test_compact_not_triggered_when_few_entries(self):
        from openharness.session import SessionLog
        log = SessionLog()
        for i in range(3):
            log.record(step=i+1, theory="T", tool="t", reasoning="r")
        assert log.compact(max_entries_to_keep=5) is False

    def test_compact_triggered_when_many_entries(self):
        from openharness.session import SessionLog
        log = SessionLog()
        for i in range(10):
            log.record(step=i+1, theory="T", tool="search", reasoning="r",
                       entities=[f"host-{i}"])
        result = log.compact(max_entries_to_keep=5)
        assert result is True

    def test_compact_preserves_entity_count(self):
        from openharness.session import SessionLog
        log = SessionLog()
        for i in range(10):
            log.record(step=i+1, theory="T", tool="search", reasoning="r",
                       entities=[f"host-{i}"])
        all_before = log.all_entities()
        log.compact(max_entries_to_keep=5)
        all_after = log.all_entities()
        assert all_before == all_after

    def test_compact_reduces_entry_count(self):
        from openharness.session import SessionLog
        log = SessionLog()
        for i in range(10):
            log.record(step=i+1, theory="T", tool="search", reasoning="r")
        log.compact(max_entries_to_keep=5)
        assert len(log.entries) < 10


# ── Export check ──────────────────────────────────────────────────


class TestExports:
    def test_exported_from_openharness(self):
        import openharness as oh
        assert hasattr(oh, "SessionLog"), "SessionLog not exported"

    def test_logentry_importable_from_submodule(self):
        from openharness.session import LogEntry
        assert LogEntry is not None
