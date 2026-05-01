"""Threshold-tier context budgeting + default compaction trigger.

Production agent loops manage context pressure with explicit budget
tiers (warning, error, blocking thresholds + a compaction buffer).
Looplet historically had only a single ``max_briefing_tokens``
knob and a ``4 chars/token`` estimator; proactive compaction was
delegated entirely to user hooks.

This module ships threshold-tier budgeting as first-class config:

* :class:`ContextBudget` — declarative tier thresholds.
* :class:`ThresholdCompactHook` — ready-to-register
  :class:`looplet.loop.LoopHook` whose ``should_compact()`` returns
  ``True`` when estimated prompt tokens cross the configured tier.
* :class:`BudgetTelemetry` — observer for production dashboards.

All tiers are **opt-in**. The loop still runs without any budget set
(identical to prior behaviour). Attach :class:`ThresholdCompactHook`
to get auto-compaction once you've configured a
:class:`CompactService`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from looplet.scaffolding import estimate_prompt_tokens
from looplet.session import SessionLog

if TYPE_CHECKING:
    from looplet.types import AgentState

__all__ = [
    "BudgetTier",
    "ContextBudget",
    "ThresholdCompactHook",
    "BudgetTelemetry",
    "classify_tier",
]

logger = logging.getLogger(__name__)

BudgetTier = Literal["ok", "warning", "error", "blocking"]
"""Classification of current context pressure.

* ``ok`` — plenty of headroom.
* ``warning`` — approaching limit; compaction would be cheap but
  isn't required yet.
* ``error`` — close to limit; compaction strongly recommended.
* ``blocking`` — past the compaction-buffer boundary; reactive
  recovery (prompt-too-long) is imminent if no action is taken.
"""


@dataclass
class ContextBudget:
    """Threshold-tier budgeting for a loop's context window.

    All values are in tokens. Defaults use sensible production
    constants for a 200K-token window with 13K buffer.
    Adjust to match your backend — the tiers are proportional, not
    absolute. A sensible rule of thumb:

    * ``warning_at`` ≈ 60% of ``context_window``
    * ``error_at`` ≈ 80%
    * ``blocking_at`` = ``context_window`` − ``compact_buffer``

    Ordering invariant: ``warning_at < error_at < blocking_at``.
    """

    context_window: int = 200_000
    """Total model context window. Match your backend's capacity."""

    warning_at: int = 120_000
    """Log a warning when estimated tokens exceed this."""

    error_at: int = 160_000
    """Log an error AND trigger compaction (via
    :class:`ThresholdCompactHook`) when estimated tokens exceed this."""

    compact_buffer: int = 13_000
    """Reserved headroom for the turn's output. Compaction must leave
    at least this much slack — otherwise the next LLM call will fail
    with prompt-too-long on a slightly larger response."""

    @property
    def blocking_at(self) -> int:
        """Hard ceiling: past this, prompt-too-long is imminent."""
        return self.context_window - self.compact_buffer

    def classify(self, estimated_tokens: int) -> BudgetTier:
        """Return the tier for a given token estimate."""
        if estimated_tokens >= self.blocking_at:
            return "blocking"
        if estimated_tokens >= self.error_at:
            return "error"
        if estimated_tokens >= self.warning_at:
            return "warning"
        return "ok"


def classify_tier(
    budget: ContextBudget,
    *,
    session_log: SessionLog,
    conversation: Any = None,
    extra_estimate: int = 0,
) -> tuple[BudgetTier, int]:
    """Estimate current tokens from available context and classify.

    When ``conversation`` is provided its serialized size is used
    (much closer to actual prompt tokens than the session log alone).
    Falls back to ``session_log.render()`` when no conversation is
    threaded. ``extra_estimate`` lets callers fold in expected
    add-ons (e.g. rendered memory, briefing tail). Returns
    ``(tier, estimated_tokens)``.
    """
    # Prefer conversation — it's the closest approximation of what
    # the LLM actually sees (tool catalog, briefing, etc. are still
    # excluded, but message history dominates in long sessions).
    if conversation is not None and hasattr(conversation, "messages"):
        total_chars = 0
        for m in conversation.messages:
            c = getattr(m, "content", "")
            if isinstance(c, str):
                total_chars += len(c)
            elif isinstance(c, list):
                # ContentBlock list — sum text of each block.
                for blk in c:
                    total_chars += len(getattr(blk, "text", "") or "")
        est = max(1, total_chars // 4)
    else:
        rendered = session_log.render() if session_log is not None else ""
        est = estimate_prompt_tokens(rendered)
    est += int(extra_estimate)
    return budget.classify(est), est


# ── Hooks ─────────────────────────────────────────────────────────


class ThresholdCompactHook:
    """Proactive-compaction hook driven by a :class:`ContextBudget`.

    Register this in ``hooks=[...]`` alongside a
    :class:`looplet.compact.CompactService` on
    :class:`looplet.loop.LoopConfig` to get threshold-based
    auto-compaction:

    >>> from looplet import (
    ...     LoopConfig, DefaultCompactService,
    ...     ContextBudget, ThresholdCompactHook,
    ... )
    >>> budget = ContextBudget(context_window=200_000)
    >>> config = LoopConfig(compact_service=DefaultCompactService())
    >>> hooks = [ThresholdCompactHook(budget)]

    ``fire_tier`` controls how aggressive the trigger is:

    * ``"error"`` (default) — compact at the error tier and above.
      Rare, only when the session is genuinely large.
    * ``"warning"`` — compact earlier; trades a bit of summary cost
      for guaranteed zero prompt-too-long incidents.
    """

    def __init__(
        self,
        budget: ContextBudget,
        *,
        fire_tier: Literal["warning", "error"] = "error",
    ) -> None:
        # Accept a dict form for workspace workspace round-trip — workspace
        # config.yaml stores constructor kwargs as primitives, so the
        # ``budget`` field arrives as a plain dict from to_config().
        if isinstance(budget, dict):
            budget = ContextBudget(**budget)
        self.budget = budget
        self.fire_tier = fire_tier
        self._fired_at: list[int] = []

    @property
    def fired_at(self) -> list[int]:
        """Step numbers where this hook requested compaction. Useful
        for tests and production telemetry."""
        return list(self._fired_at)

    def to_config(self) -> dict[str, Any]:
        """Round-trip kwargs for ``preset_to_workspace``.

        Returns the constructor kwargs needed to rebuild this hook —
        the budget is unpacked into its scalar fields so the workspace
        can serialise it as plain JSON.
        """
        from dataclasses import asdict  # noqa: PLC0415

        return {
            "budget": asdict(self.budget),
            "fire_tier": self.fire_tier,
        }

    def should_compact(
        self,
        state: AgentState,
        session_log: SessionLog,
        conversation: Any,
        step_num: int,
    ) -> bool:
        tier, est = classify_tier(
            self.budget,
            session_log=session_log,
            conversation=conversation,
        )
        if (
            tier == "blocking"
            or (self.fire_tier == "warning" and tier in ("warning", "error"))
            or (self.fire_tier == "error" and tier == "error")
        ):
            logger.info(
                "threshold_compact step=%d tier=%s est=%d blocking_at=%d",
                step_num,
                tier,
                est,
                self.budget.blocking_at,
            )
            self._fired_at.append(step_num)
            return True
        return False

    # LoopHook Protocol no-ops so this can be registered directly.


class BudgetTelemetry:
    """Observer hook that records tier transitions per step.

    Use in production for dashboards — "what fraction of steps ran
    at ``warning`` or above?". Does not trigger compaction; pair with
    :class:`ThresholdCompactHook` if you want both telemetry and
    action.
    """

    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget
        self.samples: list[tuple[int, BudgetTier, int]] = []

    def pre_prompt(
        self,
        state: AgentState,
        session_log: SessionLog,
        context: Any,
        step_num: int,
    ) -> None:
        # Prefer conversation from state (stashed by composable_loop)
        # for a more accurate token estimate — in long sessions
        # session_log-only estimates can be off by 2-3×.
        conversation = getattr(state, "conversation", None)
        tier, est = classify_tier(self.budget, session_log=session_log, conversation=conversation)
        self.samples.append((step_num, tier, est))
        return None

    @property
    def peak_tier(self) -> BudgetTier:
        """The highest tier seen in this session."""
        order = {"ok": 0, "warning": 1, "error": 2, "blocking": 3}
        if not self.samples:
            return "ok"
        return max((s[1] for s in self.samples), key=lambda t: order[t])
