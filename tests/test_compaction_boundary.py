"""Tests for HistoryRecorder.record_compaction_boundary.

Compaction boundaries are explicit markers in the conversation thread
that record "this is where the loop compacted older context into a
summary". Without them:

* The LLM cannot distinguish a summary it received from its own prior
  reasoning, which degrades coherence.
* Debuggers cannot tell why the trace jumps from step 3 to step 14.
* Subsequent compactions can't avoid re-compacting the same range.

Invariants:

1. The boundary is a SYSTEM-role message with a well-known
   ``metadata["kind"] == "compaction_boundary"``.
2. The metadata carries the summary plus the ``(first, last)`` step
   range that was compacted out.
3. ``Conversation.find_compaction_boundaries()`` returns them in order.
4. The boundary must never be dropped by subsequent compactions —
   ``Conversation.compact()`` skips over them.
"""

from __future__ import annotations

from looplet.conversation import Conversation, Message, MessageRole
from looplet.history import HistoryRecorder


class TestCompactionBoundary:
    def test_record_compaction_boundary_appends_system_message(self):
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv)
        rec.record_compaction_boundary(
            summary="Steps 1-10 compacted: found 3 IOCs, pivoted to host-47.",
            dropped_step_range=(1, 10),
        )
        assert len(conv.messages) == 1
        m = conv.messages[0]
        assert m.role == MessageRole.SYSTEM
        assert "Steps 1-10 compacted" in m.content
        assert m.metadata["kind"] == "compaction_boundary"
        assert m.metadata["dropped_step_range"] == (1, 10)
        assert m.metadata["summary"].startswith("Steps 1-10 compacted")

    def test_record_without_conversation_is_noop(self):
        rec = HistoryRecorder()  # no conversation attached
        rec.record_compaction_boundary(summary="x", dropped_step_range=(1, 2))
        # Nothing should raise and state/session_log untouched

    def test_find_compaction_boundaries_returns_in_order(self):
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv)
        rec.record_compaction_boundary(summary="S1", dropped_step_range=(1, 5))
        conv.append(Message(role=MessageRole.USER, content="mid"))
        rec.record_compaction_boundary(summary="S2", dropped_step_range=(6, 12))
        boundaries = conv.find_compaction_boundaries()
        assert len(boundaries) == 2
        assert boundaries[0].metadata["summary"] == "S1"
        assert boundaries[1].metadata["summary"] == "S2"

    def test_compaction_boundary_survives_subsequent_compact(self):
        """A boundary message must never be compacted away by
        ``Conversation.compact()``. Otherwise historical context would
        be silently lost on repeated compactions."""
        conv = Conversation()
        rec = HistoryRecorder(conversation=conv)
        # Fill with some compactable user/assistant traffic
        for i in range(10):
            conv.append(Message(role=MessageRole.USER, content=f"user-{i}"))
            conv.append(Message(role=MessageRole.ASSISTANT, content=f"asst-{i}"))
        rec.record_compaction_boundary(summary="keep-me", dropped_step_range=(1, 10))
        # Add more traffic AFTER the boundary
        for i in range(10, 20):
            conv.append(Message(role=MessageRole.USER, content=f"user-{i}"))
        # Compact aggressively (keep only last 5 messages)
        conv.compact(keep_recent=5)
        # Boundary must still be present — plus the new summary message
        # from compact() itself is now also marked as a boundary, so we
        # expect 2: the original 'keep-me' + the compact-generated one.
        boundaries = conv.find_compaction_boundaries()
        assert len(boundaries) == 2, "original boundary was dropped by compact()"
        assert boundaries[0].metadata["summary"] == "keep-me"
