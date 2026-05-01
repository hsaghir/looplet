"""Hooks that police how often and how deep the agent uses its tools.

Two universal footguns any agent harness has to guard against:

1. **Runaway tool repetition** — an agent calls the same tool 50 times,
   exhausting budget without making progress.  :class:`PerToolLimitHook`
   caps cumulative calls per tool name and blocks further invocations
   with an informative ``ToolResult.error`` once the cap is reached.

2. **Silent budget exhaustion** — the loop's ``max_steps`` runs out
   mid-task and the agent has no warning that it should start
   wrapping up.  :class:`BudgetWarningHook` injects a briefing nudge
   when the remaining fraction of the budget drops below a threshold.

Neither hook knows anything about tool semantics or task domain; both
operate purely on :class:`AgentState`, :class:`ToolCall`, and counters
the loop already maintains.

Typical use::

    from looplet.limits import BudgetWarningHook, PerToolLimitHook

    limit_hook = PerToolLimitHook(limits={"search": 20, "fetch": 50})
    warn_hook = BudgetWarningHook(thresholds=(0.5, 0.2))

    config = LoopConfig(hooks=[limit_hook, warn_hook], ...)

The module has zero third-party dependencies.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from looplet.types import ErrorKind, ToolCall, ToolError, ToolResult

__all__ = ["PerToolLimitHook", "BudgetWarningHook"]


# ── Per-tool usage cap ──────────────────────────────────────────


class PerToolLimitHook:
    """Cap how many times each tool may be invoked over a single loop.

    Wired through the loop's ``pre_dispatch`` slot: once a tool's
    cumulative invocation count reaches its limit, subsequent calls
    short-circuit with a ``ToolResult.error`` explaining the cap.
    The agent sees the error in the next prompt and can adjust its
    strategy.

    Args:
        limits: Mapping of tool name to max invocations.  Tools not
            listed fall back to ``default_limit``.
        default_limit: Max invocations for any tool not in ``limits``.
            ``None`` means unlimited.  At least one of ``limits`` or
            ``default_limit`` must be provided.
        message: Message template for the short-circuit error.
            Substitutions available: ``{tool}``, ``{limit}``,
            ``{used}``.

    Attributes:
        counts: Read-only counter of calls seen so far.
    """

    def __init__(
        self,
        *,
        limits: dict[str, int] | None = None,
        default_limit: int | None = None,
        message: str = (
            "Tool '{tool}' reached its per-loop limit ({used}/{limit}). Try a different approach."
        ),
    ) -> None:
        if limits is not None and any(v < 0 for v in limits.values()):
            raise ValueError("limits values must be >= 0")
        if default_limit is not None and default_limit < 0:
            raise ValueError("default_limit must be >= 0")
        if limits is None and default_limit is None:
            raise TypeError(
                "PerToolLimitHook requires at least one of limits={...} "
                "or default_limit=N. Example:\n"
                "  PerToolLimitHook(default_limit=10)\n"
                '  PerToolLimitHook(limits={"search": 3, "bash": 5})\n'
                '  PerToolLimitHook(default_limit=10, limits={"search": 3})'
            )
        self._limits = dict(limits or {})
        self._default_limit = default_limit
        self._message = message
        self._counts: Counter[str] = Counter()

    @property
    def counts(self) -> dict[str, int]:
        """Snapshot of cumulative call counts per tool."""
        return dict(self._counts)

    def reset(self) -> None:
        """Clear all counters — useful between runs."""
        self._counts.clear()

    def to_config(self) -> dict[str, Any]:
        """Round-trip kwargs for ``preset_to_workspace``."""
        return {
            "limits": dict(self._limits),
            "default_limit": self._default_limit,
            "message": self._message,
        }

    # ── LoopHook slot ─────────────────────────────────────────

    def pre_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: ToolCall,
        step_num: int,
    ) -> ToolResult | None:
        tool = getattr(tool_call, "tool", "")
        limit = self._limits.get(tool, self._default_limit)
        if limit is None:
            return None  # unlimited

        used = self._counts[tool]
        if used >= limit:
            msg = self._message.format(tool=tool, limit=limit, used=used)
            return ToolResult(
                tool=tool,
                args_summary=getattr(tool_call, "args_summary", "") or "",
                data=None,
                error=msg,
                error_detail=ToolError(
                    kind=ErrorKind.VALIDATION,
                    message=msg,
                    retriable=False,
                    context={"per_tool_limit": limit, "used": used},
                ),
            )

        # Count this call.  We increment here (pre-dispatch) so that
        # even tool errors consume budget — the cap is on attempts,
        # not successes.
        self._counts[tool] = used + 1
        return None


# ── Low-budget warning ──────────────────────────────────────────


class BudgetWarningHook:
    """Inject a briefing warning when remaining budget crosses a threshold.

    Fires once per threshold — e.g. with ``thresholds=(0.5, 0.25, 0.1)``
    the agent sees three escalating nudges as ``max_steps`` runs out.
    Thresholds are fractions of the total budget (``state.step_count``
    relative to an inferred ``total``), monotonically decreasing.

    The ``total`` is read from ``state.step_count + state.budget_remaining``
    on first fire so the hook works with any ``AgentState`` that
    satisfies the protocol — no configuration required.

    Args:
        thresholds: Fractions in ``(0, 1)``, one warning per fraction.
            Ordered high-to-low internally.
        message: Callable ``(remaining_fraction, remaining_steps) -> str``
            producing the nudge text, or a plain string.  A plain
            string gets ``{remaining_pct}`` and ``{remaining_steps}``
            substitutions.

    Example::

        warn = BudgetWarningHook(
            thresholds=(0.5, 0.25, 0.1),
            message=(
                "[budget] {remaining_pct:.0%} of your step budget left "
                "({remaining_steps} steps). Start consolidating."
            ),
        )
    """

    def __init__(
        self,
        *,
        thresholds: tuple[float, ...] = (0.5, 0.2),
        message: str | Callable[[float, int], str] = (
            "[low budget] {remaining_pct:.0%} of step budget remaining "
            "({remaining_steps} steps). Start consolidating."
        ),
    ) -> None:
        for t in thresholds:
            if not 0.0 < t < 1.0:
                raise ValueError("thresholds must be in (0, 1)")
        # Descending so we check the largest (earliest) first.
        self._thresholds = tuple(sorted(set(thresholds), reverse=True))
        self._message = message
        self._fired: set[float] = set()
        self._total: int | None = None

    @property
    def fired_thresholds(self) -> set[float]:
        """Snapshot of thresholds that have already fired."""
        return set(self._fired)

    def reset(self) -> None:
        """Clear fired-threshold memory — useful between runs."""
        self._fired.clear()
        self._total = None

    def to_config(self) -> dict[str, Any]:
        """Round-trip kwargs for ``preset_to_workspace``.

        Round-trips when ``message`` is a string (the common case);
        when it's a callable, returns the default-message form so the
        reloaded hook is at least functional. Callable messages are
        not portable across processes by design.
        """
        return {
            "thresholds": list(self._thresholds),
            "message": self._message
            if isinstance(self._message, str)
            else (
                "[low budget] {remaining_pct:.0%} of step budget remaining "
                "({remaining_steps} steps). Start consolidating."
            ),
        }

    # ── LoopHook slot ─────────────────────────────────────────

    def post_dispatch(
        self,
        state: Any,
        session_log: Any,
        tool_call: Any,
        tool_result: Any,
        step_num: int,
    ) -> str | None:
        used = int(getattr(state, "step_count", 0) or 0)
        remaining = int(getattr(state, "budget_remaining", 0) or 0)

        if self._total is None:
            total = used + remaining
            if total <= 0:
                return None  # no budget to warn against
            self._total = total

        if self._total <= 0:
            return None

        frac = remaining / self._total
        for t in self._thresholds:
            if frac <= t and t not in self._fired:
                self._fired.add(t)
                if callable(self._message):
                    return self._message(frac, remaining)
                return self._message.format(
                    remaining_pct=frac,
                    remaining_steps=remaining,
                )
        return None
