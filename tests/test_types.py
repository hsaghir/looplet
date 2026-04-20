"""Tests for openharness.types — core data types and protocols."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.smoke


# ── ToolCall tests ────────────────────────────────────────────────


class TestToolCallDefaults:
    def test_creation_with_required_fields(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="search")
        assert tc.tool == "search"

    def test_args_default_is_empty_dict(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="search")
        assert tc.args == {}

    def test_args_are_independent_per_instance(self):
        from openharness.types import ToolCall
        tc1 = ToolCall(tool="a")
        tc2 = ToolCall(tool="b")
        tc1.args["key"] = "val"
        assert "key" not in tc2.args

    def test_reasoning_defaults_to_empty_string(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="search")
        assert tc.reasoning == ""

    def test_call_id_auto_generated(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="search")
        assert tc.call_id
        assert isinstance(tc.call_id, str)

    def test_call_id_unique_per_instance(self):
        from openharness.types import ToolCall
        tc1 = ToolCall(tool="search")
        tc2 = ToolCall(tool="search")
        assert tc1.call_id != tc2.call_id


class TestToolCallCustomArgs:
    def test_custom_args(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="query", args={"limit": 10, "filter": "active"})
        assert tc.args["limit"] == 10
        assert tc.args["filter"] == "active"

    def test_custom_reasoning(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="query", reasoning="Need recent events")
        assert tc.reasoning == "Need recent events"

    def test_custom_call_id(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="query", call_id="my-id-123")
        assert tc.call_id == "my-id-123"

    def test_to_dict(self):
        from openharness.types import ToolCall
        tc = ToolCall(tool="search", args={"q": "test"}, reasoning="why", call_id="abc")
        d = tc.to_dict()
        assert d["tool"] == "search"
        assert d["args"] == {"q": "test"}
        assert d["reasoning"] == "why"
        assert d["call_id"] == "abc"


# ── ToolResult tests ──────────────────────────────────────────────


class TestToolResultWithoutError:
    def test_basic_construction(self):
        from openharness.types import ToolResult
        tr = ToolResult(tool="search", args_summary="q=test", data=["result1"])
        assert tr.tool == "search"
        assert tr.args_summary == "q=test"
        assert tr.data == ["result1"]

    def test_error_defaults_to_none(self):
        from openharness.types import ToolResult
        tr = ToolResult(tool="search", args_summary="q=test", data=[])
        assert tr.error is None

    def test_duration_ms_defaults_to_zero(self):
        from openharness.types import ToolResult
        tr = ToolResult(tool="search", args_summary="q=test", data=[])
        assert tr.duration_ms == 0.0

    def test_result_key_defaults_to_none(self):
        from openharness.types import ToolResult
        tr = ToolResult(tool="search", args_summary="q=test", data=[])
        assert tr.result_key is None

    def test_call_id_defaults_to_none(self):
        from openharness.types import ToolResult
        tr = ToolResult(tool="search", args_summary="q=test", data=[])
        assert tr.call_id is None


class TestToolResultWithError:
    def test_error_field(self):
        from openharness.types import ToolResult
        tr = ToolResult(tool="search", args_summary="q=test", data=None, error="timeout")
        assert tr.error == "timeout"

    def test_all_optional_fields(self):
        from openharness.types import ToolResult
        tr = ToolResult(
            tool="search",
            args_summary="q=test",
            data={"items": []},
            error=None,
            duration_ms=42.5,
            result_key="search_result",
            call_id="abc123",
        )
        assert tr.duration_ms == 42.5
        assert tr.result_key == "search_result"
        assert tr.call_id == "abc123"


# ── Step tests ────────────────────────────────────────────────────


class TestStepConstruction:
    def test_basic_construction(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="search", call_id="id1")
        tr = ToolResult(tool="search", args_summary="q=test", data=[], call_id="id1")
        step = Step(number=1, tool_call=tc, tool_result=tr)
        assert step.number == 1
        assert step.tool_call is tc
        assert step.tool_result is tr

    def test_to_dict(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="search", call_id="id1")
        tr = ToolResult(tool="search", args_summary="q=test", data=[], call_id="id1")
        step = Step(number=2, tool_call=tc, tool_result=tr)
        d = step.to_dict()
        assert d["step"] == 2
        assert "call" in d
        assert "result" in d

    def test_summary_success(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="search")
        tr = ToolResult(tool="search", args_summary="q=test", data=["a", "b"])
        step = Step(number=1, tool_call=tc, tool_result=tr)
        summary = step.summary()
        assert "S1" in summary
        assert "search" in summary

    def test_summary_error(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="search")
        tr = ToolResult(tool="search", args_summary="q=test", data=None, error="timeout")
        step = Step(number=3, tool_call=tc, tool_result=tr)
        summary = step.summary()
        assert "✗" in summary
        assert "ERROR" in summary

    def test_pretty_success_with_duration(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="search")
        tr = ToolResult(
            tool="search", args_summary="q=test", data=["a", "b", "c"],
            duration_ms=182.4,
        )
        step = Step(number=1, tool_call=tc, tool_result=tr)
        pretty = step.pretty()
        assert pretty.startswith("#1 ✓ search(q=test)")
        assert "3 items" in pretty
        assert "[182ms]" in pretty

    def test_pretty_error(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="shell")
        tr = ToolResult(
            tool="shell", args_summary="cmd=ls", data=None, error="permission denied",
        )
        step = Step(number=2, tool_call=tc, tool_result=tr)
        pretty = step.pretty()
        assert pretty.startswith("#2 ✗ shell(cmd=ls)")
        assert "permission denied" in pretty

    def test_pretty_no_duration_when_zero(self):
        from openharness.types import Step, ToolCall, ToolResult
        tc = ToolCall(tool="noop")
        tr = ToolResult(tool="noop", args_summary="", data=None)
        step = Step(number=1, tool_call=tc, tool_result=tr)
        assert "ms" not in step.pretty()


# ── AgentState Protocol compliance ───────────────────────────────


class ConcreteAgentState:
    """Minimal concrete implementation of AgentState Protocol."""

    def __init__(self):
        self.steps = []
        self.queries_used = 0

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def budget_remaining(self) -> int:
        return 10 - self.queries_used

    def context_summary(self) -> str:
        return f"steps={self.step_count}"

    def snapshot(self) -> dict:
        return {"step_count": self.step_count, "queries_used": self.queries_used}


class TestAgentStateProtocol:
    def test_isinstance_check(self):
        from openharness.types import AgentState
        state = ConcreteAgentState()
        assert isinstance(state, AgentState)

    def test_missing_steps_fails(self):
        from openharness.types import AgentState

        class BadState:
            queries_used = 0

            @property
            def step_count(self): return 0

            @property
            def budget_remaining(self): return 0

            def context_summary(self): return ""

            def snapshot(self): return {}

        # Protocol runtime check relies on attributes — missing 'steps'
        bad = BadState()
        assert not isinstance(bad, AgentState)

    def test_properties_work(self):
        state = ConcreteAgentState()
        assert state.step_count == 0
        assert state.budget_remaining == 10
        assert state.context_summary() == "steps=0"
        assert state.snapshot() == {"step_count": 0, "queries_used": 0}


# ── LLMBackend Protocol compliance ───────────────────────────────


class ConcreteLLMBackend:
    """Minimal concrete implementation of LLMBackend Protocol."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        return f"response to: {prompt[:20]}"


class TestLLMBackendProtocol:
    def test_isinstance_check(self):
        from openharness.types import LLMBackend
        backend = ConcreteLLMBackend()
        assert isinstance(backend, LLMBackend)

    def test_missing_generate_fails(self):
        from openharness.types import LLMBackend

        class BadBackend:
            pass

        assert not isinstance(BadBackend(), LLMBackend)

    def test_generate_called(self):
        backend = ConcreteLLMBackend()
        result = backend.generate("hello world", max_tokens=100, temperature=0.5)
        assert "response to" in result


# ── Export check ──────────────────────────────────────────────────


class TestExports:
    def test_all_types_exported_from_openharness(self):
        import openharness as oh
        for name in ["ToolCall", "ToolResult", "Step", "DefaultState", "LLMBackend"]:
            assert hasattr(oh, name), f"{name} not exported from openharness"

    def test_agent_state_importable(self):
        from openharness.types import AgentState
        assert AgentState is not None
