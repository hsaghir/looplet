"""Round-17 friction fix: Step.summary() shows dict preview instead of '?'."""

from __future__ import annotations

import pytest

from looplet.types import Step, ToolCall, ToolResult

pytestmark = pytest.mark.smoke


class TestStepSummaryDictPreview:
    def test_dict_with_total_key(self):
        tr = ToolResult(tool="q", args_summary="x", data={"total": 42, "rows": [1, 2]})
        step = Step(number=1, tool_call=ToolCall(tool="q", reasoning="r"), tool_result=tr)
        s = step.summary()
        assert "42" in s
        # Should NOT show the key preview since total is present
        assert "?" not in s

    def test_dict_without_total_shows_preview(self):
        tr = ToolResult(tool="search", args_summary="q=x", data={"hits": 5, "results": ["a", "b"]})
        step = Step(number=1, tool_call=ToolCall(tool="search", reasoning="r"), tool_result=tr)
        s = step.summary()
        assert "hits: 5" in s
        assert "results: (2)" in s
        assert "?" not in s

    def test_dict_single_scalar(self):
        tr = ToolResult(tool="calc", args_summary="a=1", data={"result": 99})
        step = Step(number=1, tool_call=ToolCall(tool="calc", reasoning="r"), tool_result=tr)
        s = step.summary()
        assert "result: 99" in s

    def test_large_dict_truncated(self):
        data = {f"key_{i}": i for i in range(10)}
        tr = ToolResult(tool="t", args_summary="x", data=data)
        step = Step(number=1, tool_call=ToolCall(tool="t", reasoning="r"), tool_result=tr)
        s = step.summary()
        assert "10 keys" in s

    def test_empty_dict(self):
        tr = ToolResult(tool="t", args_summary="x", data={})
        step = Step(number=1, tool_call=ToolCall(tool="t", reasoning="r"), tool_result=tr)
        s = step.summary()
        # Should not crash, and should not show "?"
        assert "?" not in s
