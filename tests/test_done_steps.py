"""Tests for :mod:`looplet.done_steps`."""

from __future__ import annotations

import pytest

from looplet.done_steps import (
    is_rejected_done,
    iter_done_steps,
    last_accepted_done,
    last_rejected_done,
)
from looplet.types import DefaultState, Step, ToolCall, ToolResult


def _make_step(
    number: int,
    tool: str,
    args: dict | None = None,
    data=None,
) -> Step:
    return Step(
        number=number,
        tool_call=ToolCall(tool=tool, args=args or {}, reasoning=""),
        tool_result=ToolResult(tool=tool, args_summary="", data=data),
    )


def _accepted_done(number: int, verdict: str = "escalate") -> Step:
    return _make_step(
        number,
        "done",
        args={"verdict": verdict, "confidence": 0.9},
        data={"verdict": verdict, "confidence": 0.9, "done": True},
    )


def _rejected_done(number: int, verdict: str = "escalate", reason: str = "gaps remain") -> Step:
    return _make_step(
        number,
        "done",
        args={"verdict": verdict, "confidence": 0.9},
        data={"rejected": True, "reason": reason},
    )


def _tool_step(number: int, tool: str = "search") -> Step:
    return _make_step(number, tool, args={"q": "anything"}, data={"hits": []})


class TestIsRejectedDone:
    def test_rejected_marker_detected(self) -> None:
        assert is_rejected_done(_rejected_done(1)) is True

    def test_accepted_done_not_rejected(self) -> None:
        assert is_rejected_done(_accepted_done(1)) is False

    def test_non_done_step_not_rejected(self) -> None:
        assert is_rejected_done(_tool_step(1)) is False

    def test_data_none_not_rejected(self) -> None:
        step = _make_step(1, "done", args={"verdict": "escalate"}, data=None)
        assert is_rejected_done(step) is False

    def test_custom_tool_name(self) -> None:
        step = _make_step(
            1,
            "finish",
            args={"verdict": "escalate"},
            data={"rejected": True, "reason": "nope"},
        )
        assert is_rejected_done(step, tool_name="finish") is True
        assert is_rejected_done(step, tool_name="done") is False


class TestIterDoneSteps:
    def test_reverse_order(self) -> None:
        state = DefaultState()
        state.steps = [
            _tool_step(1),
            _accepted_done(2, "dismiss"),
            _tool_step(3),
            _rejected_done(4, "escalate"),
        ]
        numbers = [s.number for s in iter_done_steps(state)]
        assert numbers == [4, 2]

    def test_empty_state(self) -> None:
        state = DefaultState()
        assert list(iter_done_steps(state)) == []

    def test_no_done_steps(self) -> None:
        state = DefaultState()
        state.steps = [_tool_step(1), _tool_step(2)]
        assert list(iter_done_steps(state)) == []

    def test_state_without_steps_attr(self) -> None:
        class Bare:
            pass

        assert list(iter_done_steps(Bare())) == []


class TestLastAcceptedDone:
    def test_returns_most_recent_accepted(self) -> None:
        state = DefaultState()
        state.steps = [
            _accepted_done(1, "dismiss"),
            _rejected_done(2),
            _accepted_done(3, "escalate"),
        ]
        step = last_accepted_done(state)
        assert step is not None
        assert step.number == 3
        assert step.tool_result.data["verdict"] == "escalate"

    def test_skips_rejected(self) -> None:
        state = DefaultState()
        state.steps = [
            _accepted_done(1, "dismiss"),
            _rejected_done(2),
            _rejected_done(3),
        ]
        step = last_accepted_done(state)
        assert step is not None
        assert step.number == 1

    def test_none_when_only_rejections(self) -> None:
        state = DefaultState()
        state.steps = [_rejected_done(1), _rejected_done(2)]
        assert last_accepted_done(state) is None

    def test_none_when_no_done(self) -> None:
        state = DefaultState()
        state.steps = [_tool_step(1)]
        assert last_accepted_done(state) is None


class TestLastRejectedDone:
    def test_returns_most_recent_rejected(self) -> None:
        state = DefaultState()
        state.steps = [
            _rejected_done(1, "escalate", "too early"),
            _tool_step(2),
            _rejected_done(3, "dismiss", "need more evidence"),
        ]
        step = last_rejected_done(state)
        assert step is not None
        assert step.number == 3
        assert step.tool_call.args["verdict"] == "dismiss"
        assert step.tool_result.data["reason"] == "need more evidence"

    def test_skips_accepted(self) -> None:
        state = DefaultState()
        state.steps = [
            _rejected_done(1, "escalate"),
            _accepted_done(2, "dismiss"),
        ]
        step = last_rejected_done(state)
        assert step is not None
        assert step.number == 1

    def test_none_when_only_accepted(self) -> None:
        state = DefaultState()
        state.steps = [_accepted_done(1), _accepted_done(2)]
        assert last_rejected_done(state) is None


class TestIntegration:
    def test_intent_recovery_pattern(self) -> None:
        """The canonical use case: recover agent intent from a rejected
        done() when no accepted done() was issued before budget ran out."""
        state = DefaultState()
        state.steps = [
            _tool_step(1),
            _rejected_done(2, "escalate", "no findings recorded"),
            _tool_step(3),  # agent added findings but ran out of budget
        ]
        accepted = last_accepted_done(state)
        rejected = last_rejected_done(state)
        assert accepted is None
        assert rejected is not None
        # Intent is recoverable from args even though data was rejection marker
        assert rejected.tool_call.args["verdict"] == "escalate"
        assert rejected.tool_result.data["reason"] == "no findings recorded"

    def test_accepted_overrides_rejected(self) -> None:
        """When the agent was rejected but later re-emitted a valid
        done(), last_accepted_done should win."""
        state = DefaultState()
        state.steps = [
            _rejected_done(1, "escalate"),
            _tool_step(2),  # agent added findings
            _accepted_done(3, "escalate"),  # re-emitted, accepted
        ]
        accepted = last_accepted_done(state)
        assert accepted is not None
        assert accepted.number == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
