"""Round-3 friction fixes (2026-04-24).

1. ``Step.pretty()`` now surfaces approval-gated tool calls with a
   ``⏸ awaiting approval: <desc>`` tail instead of a silent ``✓``.
2. ``Step.pretty()`` renders single-scalar-value dicts as
   ``key: value`` instead of the uninformative ``1 keys``.
"""

from __future__ import annotations

import pytest

from looplet.types import Step, ToolCall, ToolResult

pytestmark = pytest.mark.smoke


def _step(data: object) -> Step:
    return Step(
        number=3,
        tool_call=ToolCall(tool="t", args={}, reasoning=""),
        tool_result=ToolResult(
            tool="t",
            args_summary="path=/x",
            data=data,
            duration_ms=0.0,
        ),
    )


class TestApprovalPretty:
    def test_needs_approval_surfaces_in_pretty(self) -> None:
        s = _step(
            {
                "needs_approval": True,
                "approval_description": "Delete /tmp/x?",
            }
        )
        out = s.pretty()
        assert "⏸ awaiting approval" in out
        assert "Delete /tmp/x?" in out

    def test_approval_without_description(self) -> None:
        s = _step({"needs_approval": True})
        out = s.pretty()
        assert "⏸ awaiting approval" in out

    def test_approval_description_is_truncated(self) -> None:
        s = _step(
            {
                "needs_approval": True,
                "approval_description": "x" * 200,
            }
        )
        out = s.pretty()
        assert "⏸ awaiting approval" in out
        # Whole line shouldn't grow unboundedly.
        assert len(out) < 250


class TestScalarDictPretty:
    def test_single_string_value_dict(self) -> None:
        s = _step({"answer": "hello world"})
        out = s.pretty()
        assert "answer: hello world" in out
        assert "1 keys" not in out

    def test_single_int_value_dict(self) -> None:
        s = _step({"count": 42})
        out = s.pretty()
        assert "count: 42" in out

    def test_single_long_string_is_truncated(self) -> None:
        s = _step({"answer": "x" * 200})
        out = s.pretty()
        assert "answer: " in out
        assert "..." in out

    def test_two_scalar_keys_still_shows_key_count(self) -> None:
        s = _step({"answer": "hi", "meta": "there"})
        out = s.pretty()
        assert "2 keys" in out
