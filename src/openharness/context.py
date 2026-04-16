"""Progressive context management hook.

Tracks context token usage and proactively manages it via:
  1. Result aging: old tool results progressively compacted
  2. Proactive compaction: deterministic session log summarization
     approaching token threshold (optional LLM refinement via subclass)
  3. Result budget enforcement: per-result and aggregate limits

This hook implements the proactive layers. Reactive compaction
(on prompt-too-long errors) is triggered from the loop itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openharness.scaffolding import compress_session_log, enforce_result_budget

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

DEFAULT_CONTEXT_WINDOW = 128_000    # tokens
DEFAULT_RESULT_MAX_AGE_FULL = 3     # steps before result is compacted

# Multi-tier thresholds (inspired by Claude Code's 4-tier system).
# Each tier is a buffer subtracted from the context window.
# Larger buffer = triggers earlier.
DEFAULT_COMPACT_BUFFER = 20_000     # tokens reserved before compaction fires
DEFAULT_WARNING_BUFFER = 30_000     # soft warning to hooks
DEFAULT_BLOCKING_BUFFER = 5_000     # refuse to send LLM call (prevent wasted API call)

# Sentinel marking a result as already compacted (skip re-processing)
_COMPACTED_MARKER = "__compacted__"


class ContextManagerHook:
    """Progressive context management as a composable loop hook.

    Applied as a pre_prompt hook. Each step:
      1. Ages tool results older than threshold (idempotent)
      2. Enforces per-result and aggregate budgets (skips already-compacted)
      3. Compresses session log if approaching token limit
    """

    def __init__(
        self,
        llm: Any,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        compact_buffer: int = DEFAULT_COMPACT_BUFFER,
        warning_buffer: int = DEFAULT_WARNING_BUFFER,
        blocking_buffer: int = DEFAULT_BLOCKING_BUFFER,
        result_max_age_full: int = DEFAULT_RESULT_MAX_AGE_FULL,
        per_result_chars: int = 50_000,
        aggregate_chars: int = 500_000,
        must_preserve: Any = None,
    ) -> None:
        self._llm = llm
        self._context_window = context_window
        # Multi-tier thresholds (absolute token counts subtracted from window ceiling)
        self._compact_threshold = context_window - compact_buffer
        self._warning_threshold = context_window - warning_buffer
        self._blocking_threshold = context_window - blocking_buffer
        self._result_max_age_full = result_max_age_full
        self._per_result_chars = per_result_chars
        self._aggregate_chars = aggregate_chars
        self._must_preserve = must_preserve
        self._extra_llm_calls = 0
        self._compact_failures = 0
        self._max_compact_failures = 3  # circuit breaker

    def pre_prompt(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        step_num: int,
    ) -> str | None:
        """Age results, enforce budgets, compact if needed."""
        # Layer 1: Age old tool results (idempotent — skips already-compacted)
        self._age_results(state, step_num)

        # Layer 2: Enforce result budgets (skips already-compacted)
        if hasattr(state, "steps") and state.steps:
            enforce_result_budget(
                state.steps,
                per_result_chars=self._per_result_chars,
                aggregate_chars=self._aggregate_chars,
            )

        # Layer 3: Multi-tier context management
        estimated = self._estimate_context_tokens(state, session_log)

        # Tier 3a: Blocking check — refuse to proceed if context is nearly full
        if estimated >= self._blocking_threshold:
            logger.warning(
                "Context at ~%d tokens (blocking threshold %d) — forcing emergency compact",
                estimated, self._blocking_threshold,
            )
            # Emergency: compact session log + clear old results
            compress_session_log(session_log, must_preserve=self._must_preserve)
            if hasattr(state, "steps"):
                for step in state.steps[:-2]:
                    step.tool_result.data = None
            return (
                "⚠ CONTEXT LIMIT: Context was near capacity. "
                "Old results cleared. Continue with current findings."
            )

        # Tier 3b: Proactive compaction if approaching compact threshold
        # Circuit breaker: stop trying after 3 consecutive failures
        if self._compact_failures < self._max_compact_failures:
            if estimated >= self._compact_threshold:
                logger.info(
                    "Context at ~%d tokens (compact threshold %d) — compacting",
                    estimated, self._compact_threshold,
                )
                result = compress_session_log(session_log, llm=self._llm,
                                              must_preserve=self._must_preserve)
                if result is not None:
                    if self._llm is not None:
                        self._extra_llm_calls += 1
                    self._compact_failures = 0  # reset on success
                    # Health probe: verify entities survived compaction
                    probe_text = self._health_probe(state, session_log)
                    if probe_text:
                        return probe_text
                else:
                    self._compact_failures += 1
                    if self._compact_failures >= self._max_compact_failures:
                        logger.warning("Compaction circuit breaker tripped after %d failures",
                                       self._compact_failures)

        return None

    def on_loop_end(
        self,
        state: Any,
        session_log: Any,
        context: Any,
        llm: Any,
    ) -> int:
        return self._extra_llm_calls

    def _age_results(self, state: Any, step_num: int) -> None:
        """Progressively age tool results based on step distance.

        Idempotent: skips results that are already compacted or None.
        Does NOT clear data to None — keeps compact summary forever
        so the LLM can still see what was found, with recall key.
        """
        if not hasattr(state, "steps"):
            return
        for step in state.steps:
            age = step_num - step.number
            r = step.tool_result
            if r.data is None or r.error:
                continue
            # Skip already-compacted results (idempotent)
            if isinstance(r.data, dict) and r.data.get(_COMPACTED_MARKER):
                continue

            if age > self._result_max_age_full:
                r.data = _compact_data(r.data, r.result_key)

    def _health_probe(self, state: Any, session_log: Any) -> str | None:
        """Verify context integrity after compaction.

        Checks whether entities tracked by the session log are still
        mentioned in at least the session log render.  If the compaction
        reduced entity visibility, injects a brief reminder.

        Domain-agnostic: works with any session log that has all_entities().
        """
        if not hasattr(session_log, "all_entities"):
            return None
        all_ents = session_log.all_entities()
        if not all_ents or len(all_ents) <= 3:
            return None

        # Check: does the session log render still mention these entities?
        rendered = session_log.render() if hasattr(session_log, "render") else ""
        visible = {e for e in all_ents if e.lower() in rendered.lower()}
        lost = all_ents - visible
        if not lost or len(lost) < 3:
            return None

        sample = sorted(lost)[:8]
        return (
            f"⚠ CONTEXT NOTE: {len(lost)} entities from earlier steps "
            f"were compressed. Key notable items still tracked: "
            f"{', '.join(sample)}. Use result_key to retrieve full data."
        )

    def _estimate_context_tokens(self, state: Any, session_log: Any) -> int:
        """Estimate total prompt tokens.

        Counts step data + session log + realistic prompt overhead.
        Uses 4 chars/token heuristic.
        """
        total_chars = 0
        if hasattr(state, "steps"):
            for step in state.steps:
                if step.tool_result.data is not None:
                    try:
                        total_chars += len(json.dumps(step.tool_result.data, default=str))
                    except (TypeError, ValueError):
                        total_chars += len(str(step.tool_result.data))
                total_chars += len(step.tool_call.reasoning) + 50
        if hasattr(session_log, "render"):
            total_chars += len(session_log.render())
        # Realistic prompt overhead: task(500) + tool_catalog(3000) +
        # state_summary(1000) + briefing+hooks(3000) + context_history(3000)
        # + system_prompt(2500)
        total_chars += 13_000
        return total_chars // 4


def _compact_data(data: Any, result_key: str | None) -> dict:
    """Compact result data to a summary dict.

    Always returns a dict with __compacted__=True so that:
    - _age_results skips it on subsequent calls (idempotent)
    - enforce_result_budget skips it (already small)
    - isinstance(data, dict) is True (no type confusion)
    """
    if isinstance(data, list):
        return {
            _COMPACTED_MARKER: True,
            "type": "list",
            "original_count": len(data),
            "sample": data[:3],
            **({
                "recall_key": result_key,
                "note": f"Full data available via result_key '{result_key}'",
            } if result_key else {}),
        }
    if isinstance(data, dict):
        rows = data.get("rows")
        if isinstance(rows, list):
            summary: dict[str, Any] = {
                _COMPACTED_MARKER: True,
                "total_rows": len(rows),
                "sample_rows": rows[:3],
            }
            for k, v in data.items():
                if k != "rows":
                    summary[k] = v
            if result_key:
                summary["recall_key"] = result_key
            return summary
        # Already compact or no rows — mark and pass through
        return {_COMPACTED_MARKER: True, **data}
    # String or other — wrap in dict
    return {
        _COMPACTED_MARKER: True,
        "summary": str(data)[:500],
        **({"recall_key": result_key} if result_key else {}),
    }
