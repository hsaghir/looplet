"""Tests for multimodal compaction strip.

Pre-compaction, heavy blocks (images, audio, etc.) are replaced by
short text placeholders so the summarizer never receives large binary
payloads.
"""

from __future__ import annotations

from openharness.conversation import (
    HEAVY_BLOCK_KINDS,
    ContentBlock,
    Conversation,
    Message,
    MessageRole,
)


class TestHeavyBlockConstants:
    def test_default_kinds_include_image_and_audio(self):
        assert "image" in HEAVY_BLOCK_KINDS
        assert "audio" in HEAVY_BLOCK_KINDS


class TestCompactionStripsHeavyBlocks:
    def _mk(self):
        conv = Conversation()
        for _ in range(3):
            conv.append(Message(role=MessageRole.USER, content=[
                ContentBlock(kind="text", data={"text": "plan: do thing"}),
                ContentBlock(kind="image", data={"url": "x"}),
            ]))
        conv.append(Message(role=MessageRole.ASSISTANT, content="ok"))
        conv.append(Message(role=MessageRole.ASSISTANT, content="done"))
        return conv

    def test_summarizer_never_sees_raw_image_payload(self):
        seen_kinds: list[str] = []

        def summarizer(msgs):
            for m in msgs:
                for b in m.blocks:
                    seen_kinds.append(b.kind)
            return "summary"

        conv = self._mk()
        conv.compact(summarizer=summarizer, keep_recent=2)

        # All image kinds should have been replaced before summarizer ran.
        assert "image" not in seen_kinds
        # The placeholder is a text block.
        assert "text" in seen_kinds

    def test_plain_string_messages_are_unchanged(self):
        seen_content = []

        def summarizer(msgs):
            seen_content.extend(m.text for m in msgs)
            return "s"

        conv = Conversation()
        for i in range(4):
            conv.append(Message(role=MessageRole.USER, content=f"msg{i}"))

        conv.compact(summarizer=summarizer, keep_recent=1)
        assert any("msg" in c for c in seen_content)
