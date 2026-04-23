"""Tests for friction-discovery fixes (2026-04-23).

Covers three improvements that emerged from building real sample
agents against looplet:

1. Pre-validation of required tool args surfaces a clean
   ``ErrorKind.VALIDATION`` with a schema hint, instead of a raw
   Python ``TypeError`` the LLM cannot easily parse.
2. ``run_sub_loop`` accepts ``config=LoopConfig(...)`` for API parity
   with ``composable_loop``.
3. ``Step.pretty()`` shows list lengths for ``{"files": [...]}``-
   shaped dict results instead of the uninformative "1 keys".
"""

from __future__ import annotations

import pytest

from looplet import BaseToolRegistry, DefaultState, LoopConfig, ToolSpec
from looplet.types import ErrorKind, Step, ToolCall, ToolResult

pytestmark = pytest.mark.smoke


# ── Fix 1: missing-arg validation ───────────────────────────────────


class TestMissingArgValidation:
    def test_missing_required_arg_returns_validation_error(self) -> None:
        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="greet",
                description="Greet someone.",
                parameters={"name": "str"},
                execute=lambda *, name: {"greeting": f"Hello, {name}!"},
            )
        )

        result = tools.dispatch(ToolCall(tool="greet", args={}))

        assert result.error is not None
        assert result.error_detail is not None
        assert result.error_detail.kind == ErrorKind.VALIDATION
        assert "greet" in result.error
        assert "name" in result.error
        # Hint should reference the parameter schema.
        assert "Expected:" in result.error

    def test_missing_arg_error_is_not_truncated_midword(self) -> None:
        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="tool",
                description="x",
                parameters={"verylongargumentname": "str"},
                execute=lambda *, verylongargumentname: {},
            )
        )

        result = tools.dispatch(ToolCall(tool="tool", args={}))

        # Full arg name is present — no mid-word truncation.
        assert "verylongargumentname" in (result.error or "")

    def test_supplied_args_still_execute(self) -> None:
        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="greet",
                description="x",
                parameters={"name": "str"},
                execute=lambda *, name: {"greeting": f"Hello, {name}!"},
            )
        )

        result = tools.dispatch(ToolCall(tool="greet", args={"name": "Alice"}))

        assert result.error is None
        assert result.data == {"greeting": "Hello, Alice!"}

    def test_json_schema_required_args_validated(self) -> None:
        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="fetch",
                description="x",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["url"],
                },
                execute=lambda *, url, timeout=30: {"ok": True},
            )
        )

        # url missing — should fail
        r1 = tools.dispatch(ToolCall(tool="fetch", args={"timeout": 10}))
        assert r1.error is not None
        assert "url" in r1.error

        # url present — should work
        r2 = tools.dispatch(ToolCall(tool="fetch", args={"url": "http://x"}))
        assert r2.error is None


# ── Fix 2: run_sub_loop accepts config= ─────────────────────────────


class TestRunSubLoopConfigParity:
    def test_config_kwarg_accepted(self) -> None:
        """Passing a LoopConfig via config= must not raise."""
        from looplet import OpenAIBackend, run_sub_loop
        from looplet.testing import MockLLMBackend

        # The point is API parity — we only need to assert that the
        # signature accepts ``config=`` without raising. A tiny mock
        # backend lets us reach the loop body.
        llm = MockLLMBackend(responses=['call tool: {"name":"done","args":{}}'])
        tools = BaseToolRegistry()
        tools.register(
            ToolSpec(
                name="done",
                description="x",
                parameters={},
                execute=lambda: {"ok": True},
            )
        )

        # Must not raise TypeError("unexpected keyword argument 'config'").
        result = run_sub_loop(
            llm=llm,
            tools=tools,
            task={"goal": "test"},
            config=LoopConfig(max_steps=2),
        )
        assert isinstance(result, dict)
        _ = OpenAIBackend  # referenced to keep import from being pruned


# ── Fix 3: pretty() for dict-with-list ──────────────────────────────


class TestPrettyWithListValue:
    def _step(self, data: object) -> Step:
        return Step(
            number=1,
            tool_call=ToolCall(tool="list_files", args={}),
            tool_result=ToolResult(tool="list_files", args_summary="", data=data),
        )

    def test_single_list_value_shows_list_length(self) -> None:
        s = self._step({"files": ["a.txt", "b.txt", "c.txt"]})
        out = s.pretty()
        assert "3 files" in out
        assert "1 keys" not in out

    def test_multiple_lists_falls_back_to_key_count(self) -> None:
        s = self._step({"files": ["a"], "errors": ["x"]})
        out = s.pretty()
        assert "2 keys" in out

    def test_non_list_dict_still_shows_key_count(self) -> None:
        s = self._step({"name": "Alice", "age": 30})
        out = s.pretty()
        assert "2 keys" in out
