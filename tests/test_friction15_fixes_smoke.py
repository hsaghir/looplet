"""Round-15 friction fix: CompactOutcome.compacted property."""

from __future__ import annotations

import pytest

from looplet.compact import CompactOutcome

pytestmark = pytest.mark.smoke


class TestCompactOutcomeCompacted:
    def test_compacted_when_messages_reduced(self):
        o = CompactOutcome(reason="test", messages_before=20, messages_after=8)
        assert o.compacted is True

    def test_not_compacted_when_same_count(self):
        o = CompactOutcome(reason="test", messages_before=10, messages_after=10)
        assert o.compacted is False

    def test_compacted_when_cleared_positive(self):
        o = CompactOutcome(reason="prune", extra={"cleared": 5})
        assert o.compacted is True

    def test_not_compacted_when_cleared_zero(self):
        o = CompactOutcome(reason="prune", extra={"cleared": 0})
        assert o.compacted is False

    def test_not_compacted_when_no_info(self):
        o = CompactOutcome(reason="unknown")
        assert o.compacted is False

    def test_compacted_when_messages_before_none(self):
        # Only extra cleared matters
        o = CompactOutcome(reason="prune", extra={"cleared": 3})
        assert o.compacted is True
