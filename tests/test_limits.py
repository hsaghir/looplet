"""Tests for :mod:`looplet.limits`."""

from __future__ import annotations

import pytest

from looplet.limits import BudgetWarningHook, PerToolLimitHook
from looplet.types import DefaultState, ToolCall


def _call(tool: str, **args) -> ToolCall:
    return ToolCall(tool=tool, args=args, reasoning="")


# ── PerToolLimitHook ────────────────────────────────────────────


class TestPerToolLimitHook:
    def test_under_limit_returns_none(self) -> None:
        hook = PerToolLimitHook(limits={"search": 3})
        st = DefaultState()
        for i in range(3):
            r = hook.pre_dispatch(st, None, _call("search", q="x"), i + 1)
            assert r is None
        assert hook.counts == {"search": 3}

    def test_at_limit_short_circuits(self) -> None:
        hook = PerToolLimitHook(limits={"search": 2})
        st = DefaultState()
        hook.pre_dispatch(st, None, _call("search", q="a"), 1)
        hook.pre_dispatch(st, None, _call("search", q="b"), 2)
        # Third call hits the cap
        r = hook.pre_dispatch(st, None, _call("search", q="c"), 3)
        assert r is not None
        assert r.error is not None
        assert "search" in r.error
        assert "2/2" in r.error
        assert r.error_detail is not None
        assert r.error_detail.retriable is False
        assert r.error_detail.context["per_tool_limit"] == 2

    def test_unlisted_tools_unlimited(self) -> None:
        hook = PerToolLimitHook(limits={"search": 1})
        st = DefaultState()
        # fetch not in limits → any count OK
        for _ in range(10):
            r = hook.pre_dispatch(st, None, _call("fetch", id=1), 1)
            assert r is None

    def test_multiple_tools_tracked_separately(self) -> None:
        hook = PerToolLimitHook(limits={"search": 2, "fetch": 3})
        st = DefaultState()
        for _ in range(2):
            hook.pre_dispatch(st, None, _call("search"), 1)
        for _ in range(3):
            hook.pre_dispatch(st, None, _call("fetch"), 1)
        # Both at cap
        assert hook.pre_dispatch(st, None, _call("search"), 1) is not None
        assert hook.pre_dispatch(st, None, _call("fetch"), 1) is not None

    def test_reset_clears_counts(self) -> None:
        hook = PerToolLimitHook(limits={"search": 1})
        st = DefaultState()
        hook.pre_dispatch(st, None, _call("search"), 1)
        assert hook.counts == {"search": 1}
        hook.reset()
        assert hook.counts == {}
        r = hook.pre_dispatch(st, None, _call("search"), 1)
        assert r is None

    def test_custom_message(self) -> None:
        hook = PerToolLimitHook(
            limits={"search": 1},
            message="No more {tool} calls (used {used} of {limit})",
        )
        st = DefaultState()
        hook.pre_dispatch(st, None, _call("search"), 1)
        r = hook.pre_dispatch(st, None, _call("search"), 2)
        assert r is not None
        assert r.error == "No more search calls (used 1 of 1)"

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            PerToolLimitHook(limits={"search": -1})

    def test_zero_limit_blocks_all(self) -> None:
        hook = PerToolLimitHook(limits={"search": 0})
        st = DefaultState()
        r = hook.pre_dispatch(st, None, _call("search"), 1)
        assert r is not None


# ── BudgetWarningHook ───────────────────────────────────────────


class TestBudgetWarningHook:
    def _state(self, used: int, max_steps: int) -> DefaultState:
        st = DefaultState(max_steps=max_steps)
        # Populate 'steps' with placeholder objects to advance step_count.
        st.steps = [object()] * used
        return st

    def test_no_warning_above_threshold(self) -> None:
        hook = BudgetWarningHook(thresholds=(0.5,))
        st = self._state(used=2, max_steps=10)  # 80% remaining
        r = hook.post_dispatch(st, None, _call("x"), None, 2)
        assert r is None

    def test_fires_at_threshold(self) -> None:
        hook = BudgetWarningHook(thresholds=(0.5,))
        st = self._state(used=5, max_steps=10)  # 50% remaining
        r = hook.post_dispatch(st, None, _call("x"), None, 5)
        assert r is not None
        assert "50%" in r

    def test_fires_once_per_threshold(self) -> None:
        hook = BudgetWarningHook(thresholds=(0.5,))
        st = self._state(used=5, max_steps=10)
        r1 = hook.post_dispatch(st, None, _call("x"), None, 5)
        assert r1 is not None
        # Still under threshold next step
        st.steps = [object()] * 6
        r2 = hook.post_dispatch(st, None, _call("x"), None, 6)
        assert r2 is None

    def test_multiple_thresholds_escalate(self) -> None:
        hook = BudgetWarningHook(thresholds=(0.5, 0.2))
        seen: list[str] = []
        for used in range(1, 10):
            st = self._state(used=used, max_steps=10)
            r = hook.post_dispatch(st, None, _call("x"), None, used)
            if r is not None:
                seen.append(r)
        # Crossed 50% (at used=5) then 20% (at used=8)
        assert len(seen) == 2
        assert hook.fired_thresholds == {0.5, 0.2}

    def test_callable_message(self) -> None:
        def fn(frac: float, remaining: int) -> str:
            return f"frac={frac:.2f} left={remaining}"

        hook = BudgetWarningHook(thresholds=(0.5,), message=fn)
        st = self._state(used=5, max_steps=10)
        r = hook.post_dispatch(st, None, _call("x"), None, 5)
        assert r == "frac=0.50 left=5"

    def test_template_substitution(self) -> None:
        hook = BudgetWarningHook(
            thresholds=(0.5,),
            message="left {remaining_steps} ({remaining_pct:.1%})",
        )
        st = self._state(used=5, max_steps=10)
        r = hook.post_dispatch(st, None, _call("x"), None, 5)
        assert r == "left 5 (50.0%)"

    def test_zero_budget_state_skipped(self) -> None:
        hook = BudgetWarningHook(thresholds=(0.5,))
        st = DefaultState(max_steps=0)
        r = hook.post_dispatch(st, None, _call("x"), None, 0)
        assert r is None

    def test_reset(self) -> None:
        hook = BudgetWarningHook(thresholds=(0.5,))
        st = self._state(used=5, max_steps=10)
        hook.post_dispatch(st, None, _call("x"), None, 5)
        assert hook.fired_thresholds == {0.5}
        hook.reset()
        assert hook.fired_thresholds == set()
        # Should fire again
        r = hook.post_dispatch(st, None, _call("x"), None, 5)
        assert r is not None

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            BudgetWarningHook(thresholds=(0.0,))
        with pytest.raises(ValueError):
            BudgetWarningHook(thresholds=(1.0,))
        with pytest.raises(ValueError):
            BudgetWarningHook(thresholds=(1.5,))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
