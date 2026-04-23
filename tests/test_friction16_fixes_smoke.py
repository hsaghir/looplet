"""Round-16 friction fix: tool validation error shows provided args."""

from __future__ import annotations

import pytest

from looplet.tools import BaseToolRegistry, ToolSpec, register_done_tool
from looplet.types import ToolCall

pytestmark = pytest.mark.smoke


class TestValidationErrorShowsProvided:
    def test_missing_arg_shows_what_was_provided(self):
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="bash",
                description="Run a command",
                parameters={"command": "str"},
                execute=lambda *, command: {"ok": True},
            )
        )
        # LLM sends wrong arg name
        call = ToolCall(tool="bash", args={"cmd": "ls"}, reasoning="r")
        result = reg.dispatch(call)
        assert result.error is not None
        # Error should mention what was provided
        assert "provided" in result.error.lower() or "cmd" in result.error

    def test_done_missing_summary_shows_provided(self):
        reg = BaseToolRegistry()
        register_done_tool(reg)  # expects "summary"
        call = ToolCall(tool="done", args={"answer": "all done"}, reasoning="r")
        result = reg.dispatch(call)
        assert result.error is not None
        # Should show the user provided "answer" instead of "summary"
        assert "answer" in result.error or "provided" in result.error.lower()

    def test_empty_args_shows_empty(self):
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="search",
                description="Search",
                parameters={"query": "str"},
                execute=lambda *, query: {},
            )
        )
        call = ToolCall(tool="search", args={}, reasoning="r")
        result = reg.dispatch(call)
        assert result.error is not None
        assert "provided" in result.error.lower() or "[]" in result.error
