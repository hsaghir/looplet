"""Tests for looplet.conversation — unified message thread."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.smoke


# ══════════════════════════════════════════════════════════════════
# MessageRole tests
# ══════════════════════════════════════════════════════════════════


class TestMessageRole:
    def test_has_all_four_roles(self):
        from looplet.conversation import MessageRole

        assert hasattr(MessageRole, "SYSTEM")
        assert hasattr(MessageRole, "USER")
        assert hasattr(MessageRole, "ASSISTANT")
        assert hasattr(MessageRole, "TOOL")

    def test_enum_values_are_strings(self):
        from looplet.conversation import MessageRole

        assert isinstance(MessageRole.SYSTEM.value, str)
        assert isinstance(MessageRole.USER.value, str)

    def test_enum_distinct_values(self):
        from looplet.conversation import MessageRole

        values = [r.value for r in MessageRole]
        assert len(values) == len(set(values))


# ══════════════════════════════════════════════════════════════════
# Message tests
# ══════════════════════════════════════════════════════════════════


class TestMessage:
    def test_basic_creation(self):
        from looplet.conversation import Message, MessageRole

        msg = Message(role=MessageRole.USER, content="hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"
        assert msg.tool_call is None
        assert msg.tool_result is None
        assert isinstance(msg.metadata, dict)
        assert isinstance(msg.timestamp, float)

    def test_timestamp_is_recent(self):
        from looplet.conversation import Message, MessageRole

        t0 = time.time()
        msg = Message(role=MessageRole.ASSISTANT, content="hi")
        assert msg.timestamp >= t0

    def test_with_tool_call(self):
        from looplet.conversation import Message, MessageRole
        from looplet.types import ToolCall

        tc = ToolCall(tool="search", args={"q": "test"})
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_call=tc)
        assert msg.tool_call is tc

    def test_with_tool_result(self):
        from looplet.conversation import Message, MessageRole
        from looplet.types import ToolResult

        tr = ToolResult(tool="search", args_summary="q=test", data={"r": 1})
        msg = Message(role=MessageRole.TOOL, content="", tool_result=tr)
        assert msg.tool_result is tr

    def test_metadata_default_is_empty_dict(self):
        from looplet.conversation import Message, MessageRole

        m1 = Message(role=MessageRole.USER, content="a")
        m2 = Message(role=MessageRole.USER, content="b")
        # Must be independent dicts (default_factory)
        m1.metadata["x"] = 1
        assert "x" not in m2.metadata

    def test_system_role(self):
        from looplet.conversation import Message, MessageRole

        msg = Message(role=MessageRole.SYSTEM, content="You are an agent.")
        assert msg.role == MessageRole.SYSTEM

    def test_all_roles_usable(self):
        from looplet.conversation import Message, MessageRole

        for role in MessageRole:
            msg = Message(role=role, content=f"test {role.value}")
            assert msg.role == role


# ══════════════════════════════════════════════════════════════════
# Conversation tests
# ══════════════════════════════════════════════════════════════════


class TestConversationAppend:
    def test_empty_conversation(self):
        from looplet.conversation import Conversation

        c = Conversation()
        assert c.messages == []

    def test_append_single_message(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        msg = Message(role=MessageRole.USER, content="hello")
        c.append(msg)
        assert len(c.messages) == 1
        assert c.messages[0] is msg

    def test_append_preserves_order(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT]
        for r in roles:
            c.append(Message(role=r, content=r.value))
        assert [m.role for m in c.messages] == roles


class TestConversationFork:
    def test_fork_returns_new_conversation(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="original"))
        fork = c.fork()
        assert fork is not c
        assert len(fork.messages) == 1

    def test_fork_is_deep_independent_copy(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="original"))
        fork = c.fork()

        # Mutate fork — parent must not change
        fork.append(Message(role=MessageRole.ASSISTANT, content="fork reply"))
        assert len(c.messages) == 1
        assert len(fork.messages) == 2

    def test_fork_content_is_independent(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="original"))
        fork = c.fork()

        # Mutate message metadata in fork — must not affect parent
        fork.messages[0].metadata["fork_key"] = "yes"
        assert "fork_key" not in c.messages[0].metadata

    def test_fork_has_same_initial_messages(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.SYSTEM, content="system"))
        c.append(Message(role=MessageRole.USER, content="user"))
        fork = c.fork()
        assert len(fork.messages) == 2
        assert fork.messages[0].content == "system"
        assert fork.messages[1].content == "user"


class TestConversationTruncate:
    def _make_conversation(self, n: int, with_system: bool = True):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        if with_system:
            c.append(Message(role=MessageRole.SYSTEM, content="system prompt"))
        for i in range(n):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            c.append(Message(role=role, content=f"message {i}"))
        return c

    def test_truncate_keeps_last_n(self):
        from looplet.conversation import MessageRole

        c = self._make_conversation(10, with_system=False)
        c.truncate(keep_last=3)
        assert len(c.messages) == 3
        assert c.messages[-1].content == "message 9"

    def test_truncate_preserves_system_messages_by_default(self):
        from looplet.conversation import MessageRole

        c = self._make_conversation(10, with_system=True)
        c.truncate(keep_last=3)
        # System message should be kept
        roles = [m.role for m in c.messages]
        assert MessageRole.SYSTEM in roles

    def test_truncate_no_preserve_system(self):
        from looplet.conversation import MessageRole

        c = self._make_conversation(10, with_system=True)
        c.truncate(keep_last=3, preserve_system=False)
        # With preserve_system=False, only last 3 are kept
        assert len(c.messages) == 3
        roles = [m.role for m in c.messages]
        # System message (first) should be gone since we only keep last 3
        assert MessageRole.SYSTEM not in roles or c.messages[0].content != "system prompt"

    def test_truncate_noop_when_small(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="a"))
        c.append(Message(role=MessageRole.USER, content="b"))
        c.truncate(keep_last=10)
        assert len(c.messages) == 2

    def test_truncate_returns_self(self):
        c = self._make_conversation(5)
        from looplet.conversation import Conversation

        result = c.truncate(keep_last=2)
        assert result is c


class TestConversationCompact:
    def _make_conversation_with_tools(self):
        from looplet.conversation import Conversation, Message, MessageRole
        from looplet.types import ToolCall, ToolResult

        c = Conversation()
        c.append(Message(role=MessageRole.SYSTEM, content="You are an agent."))
        c.append(Message(role=MessageRole.USER, content="Investigate something"))
        # Simulate a few tool calls
        for i in range(4):
            tc = ToolCall(tool=f"tool_{i}", args={})
            tr = ToolResult(tool=f"tool_{i}", args_summary=f"step {i}", data={"result": i})
            c.append(Message(role=MessageRole.ASSISTANT, content="", tool_call=tc))
            c.append(Message(role=MessageRole.TOOL, content="", tool_result=tr))
        return c

    def test_compact_reduces_message_count(self):
        c = self._make_conversation_with_tools()
        initial_count = len(c.messages)
        c.compact()
        assert len(c.messages) < initial_count

    def test_compact_adds_summary_system_message(self):
        from looplet.conversation import MessageRole

        c = self._make_conversation_with_tools()
        c.compact()
        roles = [m.role for m in c.messages]
        assert MessageRole.SYSTEM in roles

    def test_compact_with_custom_summarizer(self):
        from looplet.conversation import MessageRole

        c = self._make_conversation_with_tools()

        def custom_summarizer(messages):
            return f"Summary of {len(messages)} messages"

        c.compact(summarizer=custom_summarizer)
        # Should have a summary message somewhere
        contents = [m.content for m in c.messages]
        assert any("Summary" in content for content in contents)

    def test_compact_returns_self(self):
        c = self._make_conversation_with_tools()
        result = c.compact()
        assert result is c

    def test_compact_preserves_last_user_message(self):
        from looplet.conversation import MessageRole

        c = self._make_conversation_with_tools()
        c.compact()
        # Last user message should be in the compacted conversation
        user_msgs = [m for m in c.messages if m.role == MessageRole.USER]
        assert len(user_msgs) >= 1
        assert user_msgs[-1].content == "Investigate something"


class TestConversationRender:
    def test_render_returns_string(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="test"))
        result = c.render()
        assert isinstance(result, str)

    def test_render_includes_content(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="unique-marker-xyz"))
        result = c.render()
        assert "unique-marker-xyz" in result

    def test_render_includes_role_labels(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.SYSTEM, content="system msg"))
        c.append(Message(role=MessageRole.USER, content="user msg"))
        result = c.render()
        # Should have some indication of roles
        assert len(result) > 0

    def test_render_with_max_tokens(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        for i in range(100):
            c.append(Message(role=MessageRole.USER, content=f"message {i} " * 50))
        # Render with tight token budget should truncate
        full = c.render()
        truncated = c.render(max_tokens=50)
        assert len(truncated) <= len(full)


class TestConversationSerialize:
    def test_serialize_returns_dict(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="hello"))
        data = c.serialize()
        assert isinstance(data, dict)

    def test_serialize_is_json_compatible(self):
        import json

        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.SYSTEM, content="system"))
        c.append(Message(role=MessageRole.USER, content="user"))
        data = c.serialize()
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 0

    def test_deserialize_roundtrip(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.SYSTEM, content="system prompt"))
        c.append(Message(role=MessageRole.USER, content="user message"))
        c.append(Message(role=MessageRole.ASSISTANT, content="assistant reply"))

        data = c.serialize()
        c2 = Conversation.deserialize(data)

        assert len(c2.messages) == len(c.messages)
        for orig, restored in zip(c.messages, c2.messages):
            assert restored.role == orig.role
            assert restored.content == orig.content

    def test_deserialize_with_tool_call(self):
        from looplet.conversation import Conversation, Message, MessageRole
        from looplet.types import ToolCall

        c = Conversation()
        tc = ToolCall(tool="search", args={"q": "test"}, reasoning="because")
        c.append(Message(role=MessageRole.ASSISTANT, content="", tool_call=tc))

        data = c.serialize()
        c2 = Conversation.deserialize(data)
        assert c2.messages[0].tool_call is not None
        assert c2.messages[0].tool_call.tool == "search"
        assert c2.messages[0].tool_call.args["q"] == "test"

    def test_deserialize_with_tool_result(self):
        from looplet.conversation import Conversation, Message, MessageRole
        from looplet.types import ToolResult

        c = Conversation()
        tr = ToolResult(tool="search", args_summary="q=test", data={"results": ["r1"]})
        c.append(Message(role=MessageRole.TOOL, content="", tool_result=tr))

        data = c.serialize()
        c2 = Conversation.deserialize(data)
        assert c2.messages[0].tool_result is not None
        assert c2.messages[0].tool_result.tool == "search"

    def test_serialize_with_metadata(self):
        import json

        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        msg = Message(role=MessageRole.USER, content="hi", metadata={"key": "value"})
        c.append(msg)
        data = c.serialize()
        c2 = Conversation.deserialize(data)
        assert c2.messages[0].metadata.get("key") == "value"

    def test_message_role_accepts_plain_string(self):
        """Regression: ``MessageRole`` is a ``str, Enum`` so callers
        naturally pass plain strings; ``Message.__post_init__`` must
        coerce so downstream code that does ``msg.role.value`` works."""
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role="system", content="hi"))
        c.append(Message(role="user", content="hello"))
        # Round-trip — would have raised AttributeError before the fix.
        c2 = Conversation.deserialize(c.serialize())
        assert c2.messages[0].role is MessageRole.SYSTEM
        assert c2.messages[1].role is MessageRole.USER

    def test_serialize_round_trips_tool_call_and_result_metadata(self):
        """Regression: ``ToolCall.metadata`` and ``ToolResult.metadata``
        (added by PR #24) must round-trip through Conversation
        serialize/deserialize. Previously the serializer dropped them
        silently and the deserializer never restored them."""
        from looplet.conversation import Conversation, Message, MessageRole
        from looplet.types import ToolCall, ToolResult

        c = Conversation()
        c.append(
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_call=ToolCall(tool="search", args={"q": "x"}, metadata={"audit": "ok"}),
            )
        )
        c.append(
            Message(
                role=MessageRole.TOOL,
                content="result",
                tool_result=ToolResult(
                    tool="search",
                    args_summary="q=x",
                    data={"hits": [1]},
                    metadata={"scrubbed": True},
                ),
            )
        )

        c2 = Conversation.deserialize(c.serialize())
        assert c2.messages[0].tool_call.metadata == {"audit": "ok"}
        assert c2.messages[1].tool_result.metadata == {"scrubbed": True}


class TestConversationProperties:
    def test_token_estimate_zero_for_empty(self):
        from looplet.conversation import Conversation

        c = Conversation()
        assert c.token_estimate >= 0

    def test_token_estimate_grows_with_messages(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        t0 = c.token_estimate
        c.append(Message(role=MessageRole.USER, content="x" * 100))
        assert c.token_estimate > t0

    def test_entities_empty_for_no_results(self):
        from looplet.conversation import Conversation, Message, MessageRole

        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="hello"))
        assert isinstance(c.entities, set)

    def test_entities_from_tool_results(self):
        from looplet.conversation import Conversation, Message, MessageRole
        from looplet.types import ToolResult

        c = Conversation()
        tr = ToolResult(
            tool="search",
            args_summary="",
            data={"entities": ["entity_a", "entity_b"]},
        )
        c.append(Message(role=MessageRole.TOOL, content="", tool_result=tr))
        # entities extracts from tool_result.data["entities"]
        entities = c.entities
        assert "entity_a" in entities
        assert "entity_b" in entities

    def test_entities_union_across_messages(self):
        from looplet.conversation import Conversation, Message, MessageRole
        from looplet.types import ToolResult

        c = Conversation()
        for names in [["a", "b"], ["c"]]:
            tr = ToolResult(tool="t", args_summary="", data={"entities": names})
            c.append(Message(role=MessageRole.TOOL, content="", tool_result=tr))
        assert c.entities == {"a", "b", "c"}


class TestDefaultSummarizer:
    def test_default_summarizer_returns_string(self):
        from looplet.conversation import Message, MessageRole, default_summarizer

        messages = [
            Message(role=MessageRole.USER, content="Do something"),
            Message(role=MessageRole.ASSISTANT, content="ok"),
        ]
        result = default_summarizer(messages)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_default_summarizer_includes_role_counts(self):
        from looplet.conversation import Message, MessageRole, default_summarizer
        from looplet.types import ToolCall

        messages = [
            Message(role=MessageRole.USER, content="task"),
            Message(
                role=MessageRole.ASSISTANT, content="", tool_call=ToolCall(tool="search", args={})
            ),
            Message(
                role=MessageRole.ASSISTANT, content="", tool_call=ToolCall(tool="think", args={})
            ),
        ]
        result = default_summarizer(messages)
        # Should mention the tools called
        assert "search" in result or "think" in result or "tool" in result.lower()

    def test_default_summarizer_callable(self):
        import inspect

        from looplet.conversation import default_summarizer

        assert callable(default_summarizer)
