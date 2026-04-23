"""Round-4 friction fix (2026-04-24).

``_summarize_args_dict`` is now the single source of truth for
``args_summary`` formatting. Previously, loop.py, validation.py and
streaming.py produced Python dict-repr strings (``{'cmd': 'ls'}``)
while the happy path in ``BaseToolRegistry.dispatch`` produced
``cmd=ls``. Step logs were visibly inconsistent: denied/intercepted
calls looked different from allowed ones.
"""

from __future__ import annotations

import pytest

from looplet import BaseToolRegistry, ToolSpec
from looplet.tools import _summarize_args_dict
from looplet.types import ToolCall

pytestmark = pytest.mark.smoke


class TestSummarizeArgsDict:
    def test_basic_key_value_format(self) -> None:
        assert _summarize_args_dict({"cmd": "ls"}) == "cmd=ls"

    def test_multiple_args(self) -> None:
        assert _summarize_args_dict({"a": 1, "b": "x"}) == "a=1, b=x"

    def test_long_value_truncated(self) -> None:
        out = _summarize_args_dict({"cmd": "x" * 100})
        assert out.startswith("cmd=")
        assert "..." in out
        # Value part <= 50 chars + "..."
        assert len(out) < 70

    def test_empty_dict(self) -> None:
        assert _summarize_args_dict({}) == ""


class TestConsistentRendering:
    def test_unknown_tool_uses_kv_format(self) -> None:
        reg = BaseToolRegistry()
        reg.register(ToolSpec(name="known", description="d", parameters={}, execute=lambda: {}))
        call = ToolCall(tool="nope", args={"cmd": "ls /tmp"}, reasoning="")
        result = reg.dispatch(call)
        assert result.error is not None
        # Must be "cmd=ls /tmp", not "{'cmd': 'ls /tmp'}".
        assert result.args_summary == "cmd=ls /tmp"
        assert "{" not in result.args_summary
