"""Tests for multi-block ``Message.content``.

``Message.content`` widens from ``str`` to ``str | list[ContentBlock]``
where a ``ContentBlock`` has a ``kind`` (``"text"``, ``"image"``,
``"tool_use"``, ``"tool_result"``, etc.) and a free-form ``data`` dict.

Back-compat: anywhere the loop treats ``content`` as a string keeps
working because ``Message.text`` renders the blocks to a flat string
and existing constructors still accept plain strings.
"""

from __future__ import annotations

from looplet.conversation import (
    ContentBlock,
    Conversation,
    Message,
    MessageRole,
)


class TestContentBlock:
    def test_text_block(self):
        b = ContentBlock(kind="text", data={"text": "hi"})
        assert b.kind == "text"
        assert b.text == "hi"

    def test_image_block_has_no_text(self):
        b = ContentBlock(kind="image", data={"url": "http://x", "media_type": "image/png"})
        assert b.kind == "image"
        assert b.text == "[image attached]"

    def test_unknown_kind_falls_back_to_repr(self):
        b = ContentBlock(kind="custom", data={"x": 1})
        assert b.kind == "custom"
        assert "[custom]" in b.text


class TestMessageWithBlocks:
    def test_string_content_still_works(self):
        m = Message(role=MessageRole.USER, content="hello")
        assert m.text == "hello"
        assert m.blocks == [ContentBlock(kind="text", data={"text": "hello"})]

    def test_list_of_blocks_is_accepted(self):
        blocks = [
            ContentBlock(kind="text", data={"text": "look at this"}),
            ContentBlock(kind="image", data={"url": "http://x", "media_type": "image/png"}),
        ]
        m = Message(role=MessageRole.USER, content=blocks)
        assert m.text == "look at this\n[image attached]"
        assert len(m.blocks) == 2

    def test_text_only_helper(self):
        blocks = [
            ContentBlock(kind="text", data={"text": "a"}),
            ContentBlock(kind="image", data={"url": "u"}),
            ContentBlock(kind="text", data={"text": "b"}),
        ]
        m = Message(role=MessageRole.USER, content=blocks)
        txt_blocks = m.text_blocks()
        assert [b.text for b in txt_blocks] == ["a", "b"]


class TestConversationWithBlocks:
    def test_render_flattens_multimodal_content(self):
        conv = Conversation()
        conv.append(
            Message(
                role=MessageRole.USER,
                content=[
                    ContentBlock(kind="text", data={"text": "check screenshot"}),
                    ContentBlock(kind="image", data={"url": "x"}),
                ],
            )
        )
        rendered = conv.render()
        assert "check screenshot" in rendered
        assert "[image attached]" in rendered

    def test_fork_preserves_multimodal(self):
        conv = Conversation()
        conv.append(
            Message(
                role=MessageRole.USER,
                content=[
                    ContentBlock(kind="image", data={"url": "x"}),
                ],
            )
        )
        f = conv.fork()
        assert f.messages[0].blocks[0].kind == "image"
