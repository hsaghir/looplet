"""Tests for Conversation integration with composable_loop."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from openharness.conversation import Conversation, Message, MessageRole
from openharness.loop import LoopConfig, composable_loop
from openharness.tools import BaseToolRegistry, ToolSpec

# ── Helpers ──────────────────────────────────────────────────────


class _ScriptedLLM:
    """Mock LLM returning scripted JSON responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp


@dataclass
class _SimpleState:
    steps: list = field(default_factory=list)
    queries_used: int = 0
    _max: int = 10

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max - len(self.steps))

    def context_summary(self) -> str:
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {}


def _make_registry():
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="echo", description="Echo input",
        parameters={"text": "text to echo"},
        execute=lambda text="": {"echoed": text},
    ))
    reg.register(ToolSpec(
        name="done", description="Finish",
        parameters={},
        execute=lambda **kw: {"status": "done"},
    ))
    return reg


# ── Tests ────────────────────────────────────────────────────────


class TestConversationIntegration:
    def test_conversation_records_messages(self):
        """Conversation receives USER/ASSISTANT/TOOL messages during loop."""
        llm = _ScriptedLLM([
            '{"tool": "echo", "args": {"text": "hello"}}',
            '{"tool": "done", "args": {}}',
        ])
        state = _SimpleState()
        conv = Conversation()
        config = LoopConfig(max_steps=5, done_tool="done")

        steps = list(composable_loop(
            llm, tools=_make_registry(), config=config,
            state=state, conversation=conv,
        ))

        assert len(steps) >= 2
        # Conversation should have messages: at least USER+ASSISTANT for each LLM call
        # plus ASSISTANT(tool_call)+TOOL(tool_result) for each tool dispatch
        assert len(conv.messages) > 0

        roles = [m.role for m in conv.messages]
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles
        assert MessageRole.TOOL in roles

    def test_conversation_has_tool_calls(self):
        """Tool calls and results are recorded in conversation."""
        llm = _ScriptedLLM([
            '{"tool": "echo", "args": {"text": "test"}}',
            '{"tool": "done", "args": {}}',
        ])
        state = _SimpleState()
        conv = Conversation()
        config = LoopConfig(max_steps=5, done_tool="done")

        list(composable_loop(
            llm, tools=_make_registry(), config=config,
            state=state, conversation=conv,
        ))

        # Find tool call messages
        tool_call_msgs = [m for m in conv.messages if m.tool_call is not None]
        tool_result_msgs = [m for m in conv.messages if m.tool_result is not None]
        assert len(tool_call_msgs) >= 1
        assert len(tool_result_msgs) >= 1
        assert tool_call_msgs[0].tool_call.tool == "echo"

    def test_conversation_none_doesnt_break(self):
        """Loop works fine without conversation (default behavior)."""
        llm = _ScriptedLLM([
            '{"tool": "done", "args": {}}',
        ])
        state = _SimpleState()
        config = LoopConfig(max_steps=5, done_tool="done")

        steps = list(composable_loop(
            llm, tools=_make_registry(), config=config,
            state=state, conversation=None,
        ))
        assert len(steps) >= 1

    def test_conversation_serialize_after_loop(self):
        """Conversation can be serialized after a loop run."""
        llm = _ScriptedLLM([
            '{"tool": "echo", "args": {"text": "x"}}',
            '{"tool": "done", "args": {}}',
        ])
        state = _SimpleState()
        conv = Conversation()
        config = LoopConfig(max_steps=5, done_tool="done")

        list(composable_loop(
            llm, tools=_make_registry(), config=config,
            state=state, conversation=conv,
        ))

        data = conv.serialize()
        restored = Conversation.deserialize(data)
        assert len(restored.messages) == len(conv.messages)

    def test_conversation_fork_for_subagent(self):
        """Forked conversation is independent from parent."""
        conv = Conversation()
        conv.append(Message(role=MessageRole.SYSTEM, content="You are helpful"))

        forked = conv.fork()
        forked.append(Message(role=MessageRole.USER, content="child msg"))

        assert len(conv.messages) == 1
        assert len(forked.messages) == 2
