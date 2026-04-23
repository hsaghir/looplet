"""Round-12 friction fixes: backend convenience kwargs, register_done_tool,
EvalResult.passed, Conversation.compact() boundary marking."""

from __future__ import annotations

import pytest

from looplet.conversation import Conversation, Message, MessageRole
from looplet.evals import EvalResult
from looplet.tools import BaseToolRegistry, register_done_tool

pytestmark = pytest.mark.smoke


# ── register_done_tool ───────────────────────────────────────────


class TestRegisterDoneTool:
    def test_registers_done_by_default(self):
        reg = BaseToolRegistry()
        register_done_tool(reg)
        assert "done" in reg._tools
        assert reg._tools["done"].description

    def test_custom_name(self):
        reg = BaseToolRegistry()
        register_done_tool(reg, name="finish")
        assert "finish" in reg._tools
        assert "done" not in reg._tools

    def test_dispatch_returns_status(self):
        from looplet.types import ToolCall

        reg = BaseToolRegistry()
        register_done_tool(reg)
        call = ToolCall(tool="done", args={"summary": "all good"}, reasoning="r")
        result = reg.dispatch(call)
        assert result.data["status"] == "completed"
        assert result.data["summary"] == "all good"


# ── OpenAIBackend convenience kwargs ─────────────────────────────


class TestBackendConvenienceKwargs:
    def test_openai_backend_requires_client_or_base_url(self):
        from looplet.backends import OpenAIBackend

        with pytest.raises(TypeError, match="base_url"):
            OpenAIBackend()

    def test_openai_backend_base_url_creates_client(self):
        from looplet.backends import OpenAIBackend

        # This should NOT raise — it auto-creates the client
        llm = OpenAIBackend(base_url="http://localhost:9999/v1", api_key="test")
        assert llm._client is not None
        assert llm._model == "gpt-4o"

    def test_openai_backend_explicit_client_still_works(self):
        from unittest.mock import MagicMock

        from looplet.backends import OpenAIBackend

        mock_client = MagicMock()
        llm = OpenAIBackend(mock_client, model="gpt-4o-mini")
        assert llm._client is mock_client
        assert llm._model == "gpt-4o-mini"

    def test_async_openai_backend_requires_client_or_base_url(self):
        from looplet.backends import AsyncOpenAIBackend

        with pytest.raises(TypeError, match="base_url"):
            AsyncOpenAIBackend()


# ── EvalResult.passed ────────────────────────────────────────────


class TestEvalResultPassed:
    def test_score_above_threshold(self):
        assert EvalResult(score=0.8).passed is True

    def test_score_at_threshold(self):
        assert EvalResult(score=0.5).passed is True

    def test_score_below_threshold(self):
        assert EvalResult(score=0.3).passed is False

    def test_label_pass(self):
        assert EvalResult(label="pass").passed is True
        assert EvalResult(label="Pass").passed is True
        assert EvalResult(label="PASS").passed is True

    def test_label_correct(self):
        assert EvalResult(label="correct").passed is True

    def test_label_fail(self):
        assert EvalResult(label="fail").passed is False

    def test_neither_score_nor_label(self):
        assert EvalResult().passed is False

    def test_score_takes_precedence_over_label(self):
        # score=0.1 + label="pass" → passed should be False (score wins)
        assert EvalResult(score=0.1, label="pass").passed is False


# ── Conversation.compact() boundary marking ──────────────────────


class TestCompactBoundaryMarking:
    def test_compact_creates_boundary(self):
        c = Conversation()
        for i in range(6):
            c.append(Message(role=MessageRole.USER, content=f"q{i}"))
            c.append(Message(role=MessageRole.ASSISTANT, content=f"a{i}"))
        c.compact(keep_recent=2)
        boundaries = c.find_compaction_boundaries()
        assert len(boundaries) == 1
        assert boundaries[0].metadata["kind"] == "compaction_boundary"
        assert "dropped_message_count" in boundaries[0].metadata

    def test_compact_boundary_survives_recompact(self):
        c = Conversation()
        for i in range(10):
            c.append(Message(role=MessageRole.USER, content=f"q{i}"))
            c.append(Message(role=MessageRole.ASSISTANT, content=f"a{i}"))
        c.compact(keep_recent=4)
        b1 = c.find_compaction_boundaries()
        assert len(b1) == 1
        # Add more traffic + compact again
        for i in range(10, 15):
            c.append(Message(role=MessageRole.USER, content=f"q{i}"))
            c.append(Message(role=MessageRole.ASSISTANT, content=f"a{i}"))
        c.compact(keep_recent=2)
        b2 = c.find_compaction_boundaries()
        # Both boundaries survive
        assert len(b2) == 2

    def test_no_boundary_when_nothing_to_compact(self):
        c = Conversation()
        c.append(Message(role=MessageRole.USER, content="hi"))
        c.compact(keep_recent=2)
        assert len(c.find_compaction_boundaries()) == 0
