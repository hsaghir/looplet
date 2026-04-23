"""Tests for :mod:`looplet.stagnation`."""

from __future__ import annotations

import pytest

from looplet.stagnation import (
    StagnationHook,
    result_size_fingerprint,
    tool_call_fingerprint,
)
from looplet.types import DefaultState, ToolCall, ToolResult


def _call(tool: str, **args) -> ToolCall:
    return ToolCall(tool=tool, args=args, reasoning="")


def _result(tool: str, data=None) -> ToolResult:
    return ToolResult(tool=tool, args_summary="", data=data)


class TestToolCallFingerprint:
    def test_same_tool_same_args_matches(self) -> None:
        a = _call("search", q="foo")
        b = _call("search", q="foo")
        assert tool_call_fingerprint(None, a, None) == tool_call_fingerprint(None, b, None)

    def test_different_args_differ(self) -> None:
        a = _call("search", q="foo")
        b = _call("search", q="bar")
        assert tool_call_fingerprint(None, a, None) != tool_call_fingerprint(None, b, None)

    def test_arg_order_irrelevant(self) -> None:
        a = _call("search", q="foo", limit=5)
        b = _call("search", limit=5, q="foo")
        assert tool_call_fingerprint(None, a, None) == tool_call_fingerprint(None, b, None)

    def test_unhashable_args_fall_back_to_repr(self) -> None:
        a = _call("search", tags=["a", "b"])
        # Should not raise.
        fp = tool_call_fingerprint(None, a, None)
        assert fp is not None


class TestResultSizeFingerprint:
    def test_empty_vs_nonempty_differ(self) -> None:
        c = _call("search", q="x")
        empty = result_size_fingerprint(None, c, _result("search", data={"hits": []}))
        full = result_size_fingerprint(None, c, _result("search", data={"hits": [1]}))
        assert empty != full


class TestStagnationHookBasic:
    def _hook(self, **kw) -> StagnationHook:
        return StagnationHook(threshold=3, nudge="STAGNATION", **kw)

    def test_single_repeat_no_nudge(self) -> None:
        hook = self._hook()
        st = DefaultState()
        r1 = hook.post_dispatch(st, None, _call("search", q="x"), _result("search"), 1)
        assert r1 is None
        assert hook.stagnant_steps == 1

    def test_fires_on_threshold(self) -> None:
        hook = self._hook()
        st = DefaultState()
        results = []
        for i in range(3):
            results.append(
                hook.post_dispatch(st, None, _call("search", q="x"), _result("search"), i + 1)
            )
        assert results == [None, None, "STAGNATION"]

    def test_different_action_resets(self) -> None:
        hook = self._hook()
        st = DefaultState()
        hook.post_dispatch(st, None, _call("search", q="x"), _result("search"), 1)
        hook.post_dispatch(st, None, _call("search", q="x"), _result("search"), 2)
        # Break the pattern
        r = hook.post_dispatch(st, None, _call("fetch", id=1), _result("fetch"), 3)
        assert r is None
        assert hook.stagnant_steps == 1

    def test_reset_after_nudge_default(self) -> None:
        hook = self._hook()
        st = DefaultState()
        for i in range(3):
            hook.post_dispatch(st, None, _call("search", q="x"), _result("search"), i + 1)
        # Next step same — should NOT fire again immediately
        r = hook.post_dispatch(st, None, _call("search", q="x"), _result("search"), 4)
        assert r is None

    def test_reset_after_nudge_false_fires_every_step(self) -> None:
        hook = StagnationHook(threshold=2, nudge="X", reset_after_nudge=False)
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        r2 = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 2)
        r3 = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 3)
        assert r2 == "X" and r3 == "X"


class TestProgressCounter:
    def test_progress_increase_resets_streak(self) -> None:
        counter = {"n": 0}

        hook = StagnationHook(
            threshold=3,
            nudge="STAGNATION",
            progress=lambda _state: counter["n"],
        )
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 2)
        # Real progress made
        counter["n"] = 1
        r = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 3)
        assert r is None
        assert hook.stagnant_steps == 0

    def test_progress_flat_does_not_reset(self) -> None:
        hook = StagnationHook(
            threshold=2,
            nudge="X",
            progress=lambda _s: 0,
        )
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        r = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 2)
        assert r == "X"

    def test_progress_decrease_does_not_reset(self) -> None:
        # Non-monotonic counter shouldn't break the hook.
        counter = {"n": 5}
        hook = StagnationHook(
            threshold=2,
            nudge="X",
            progress=lambda _s: counter["n"],
        )
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        counter["n"] = 2
        r = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 2)
        assert r == "X"

    def test_bad_progress_callable_does_not_crash(self) -> None:
        def bad(_state):
            raise RuntimeError("boom")

        hook = StagnationHook(threshold=2, nudge="X", progress=bad)
        st = DefaultState()
        # Should not raise — defensive
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        r = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 2)
        assert r == "X"


class TestIgnoreTools:
    def test_ignored_tool_does_not_count(self) -> None:
        hook = StagnationHook(
            threshold=2,
            nudge="X",
            ignore_tools={"done", "note"},
        )
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        # done() call in between shouldn't reset or advance the streak
        r1 = hook.post_dispatch(st, None, _call("done"), _result("done"), 2)
        assert r1 is None
        r2 = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 3)
        assert r2 == "X"


class TestCallableNudge:
    def test_callable_nudge_receives_state_and_streak(self) -> None:
        seen: dict = {}

        def nudge(state, streak):
            seen["streak"] = streak
            return f"stuck for {streak} steps"

        hook = StagnationHook(threshold=2, nudge=nudge)
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 1)
        r = hook.post_dispatch(st, None, _call("s", q=1), _result("s"), 2)
        assert r == "stuck for 2 steps"
        assert seen["streak"] == 2


class TestCustomFingerprint:
    def test_fingerprint_returning_none_resets(self) -> None:
        # Caller-controlled fingerprint that sometimes declares progress
        counter = {"n": 0}

        def fp(_s, _c, _r):
            counter["n"] += 1
            return None if counter["n"] == 3 else "same"

        hook = StagnationHook(threshold=2, nudge="X", fingerprint=fp)
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s"), _result("s"), 1)
        r2 = hook.post_dispatch(st, None, _call("s"), _result("s"), 2)
        assert r2 == "X"
        # Fingerprint returns None on step 3 — streak resets
        r3 = hook.post_dispatch(st, None, _call("s"), _result("s"), 3)
        assert r3 is None
        assert hook.stagnant_steps == 0


class TestValidation:
    def test_threshold_must_be_at_least_2(self) -> None:
        with pytest.raises(ValueError):
            StagnationHook(threshold=1, nudge="X")


class TestReset:
    def test_reset_clears_state(self) -> None:
        hook = StagnationHook(threshold=3, nudge="X")
        st = DefaultState()
        hook.post_dispatch(st, None, _call("s"), _result("s"), 1)
        hook.post_dispatch(st, None, _call("s"), _result("s"), 2)
        assert hook.stagnant_steps == 2
        hook.reset()
        assert hook.stagnant_steps == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
