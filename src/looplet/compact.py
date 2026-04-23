"""Context compaction service + lifecycle event plumbing.

The composable loop's reactive-recovery path compresses agent state
when a prompt grows past the context window. Historically this lived
as a hardcoded chain inside ``loop.py`` (:func:`_recovery_chain`);
this module exposes it as a **service** so users can:

* Observe every compaction via :attr:`LifecycleEvent.PRE_COMPACT` and
  :attr:`LifecycleEvent.POST_COMPACT` on :meth:`LoopHook.on_event`.
* Swap the strategy wholesale by passing a
  :class:`CompactService` to :attr:`LoopConfig.compact_service`.
* Trigger compaction manually from a hook at any time by calling
  :func:`run_compact`.

The default service :class:`DefaultCompactService` preserves the
existing three-strategy chain (aggressive budget → reactive compact →
clear old results) so existing behavior is unchanged when no service
is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from looplet.session import SessionLog
    from looplet.types import AgentState, LLMBackend

__all__ = [
    "CompactService",
    "CompactOutcome",
    "TruncateCompact",
    "SummarizeCompact",
    "PruneToolResults",
    "compact_chain",
    "run_compact",
]


@dataclass
class CompactOutcome:
    """Result of a single compaction invocation.

    All fields are optional; the default service populates what it
    knows. Custom services may add :attr:`extra` for domain-specific
    metrics (e.g. tokens freed, summary LLM calls spent).
    """

    reason: str = ""
    messages_before: int | None = None
    messages_after: int | None = None
    llm_calls_spent: int = 0
    extra: dict[str, Any] | None = None
    cleanup: "Callable[[], None] | None" = None
    """Optional post-compact callback. When set, :func:`run_compact`
    invokes it after firing the ``POST_COMPACT`` event. Use for
    domain-specific state resets (clear caches, re-inject file
    context, reset token baselines) that the loop shouldn't know
    about."""

    @property
    def compacted(self) -> bool:
        """True when the compaction actually reduced context size.

        Checks ``messages_before`` vs ``messages_after`` when both are
        set; also returns True when ``extra`` contains a positive
        ``"cleared"`` count (from :class:`PruneToolResults`).
        """
        if (
            self.messages_before is not None
            and self.messages_after is not None
            and self.messages_after < self.messages_before
        ):
            return True
        if self.extra and self.extra.get("cleared", 0) > 0:
            return True
        return False


@runtime_checkable
class CompactService(Protocol):
    """Swap-in service that compresses state when the loop hits token pressure.

    A service is called with the same ``(state, session_log, llm,
    step_num)`` signature as the legacy recovery strategies plus a
    ``conversation`` (optional — None for loops that do not thread
    one). It must mutate those surfaces in place and return a
    :class:`CompactOutcome` describing what it did.
    """

    def compact(
        self,
        *,
        state: AgentState,
        session_log: SessionLog,
        llm: LLMBackend,
        conversation: Any | None,
        step_num: int,
        reason: str,
    ) -> CompactOutcome: ...


class TruncateCompact:
    """Drop old entries, keep the N most recent. Zero LLM calls.

    Session-log side: calls
    :func:`looplet.scaffolding.emergency_truncate`.
    Conversation side: calls :meth:`Conversation.compact` with the
    default deterministic summarizer.

    Fast, free, and deterministic — but anything in the dropped
    middle is gone. Use when speed matters more than context
    retention, or as the last-resort stage in a :func:`compact_chain`.
    """

    def __init__(self, *, keep_recent: int = 2) -> None:
        self.keep_recent = keep_recent

    def compact(
        self,
        *,
        state: AgentState,
        session_log: SessionLog,
        llm: LLMBackend,
        conversation: Any | None,
        step_num: int,
        reason: str,
    ) -> CompactOutcome:
        # Session-log side.
        from looplet.scaffolding import emergency_truncate  # noqa: PLC0415

        messages_before = len(conversation.messages) if conversation is not None else None
        emergency_truncate(state, session_log, keep_recent=self.keep_recent)

        # Conversation side (optional — most domains don't thread one).
        if conversation is not None and hasattr(conversation, "compact"):
            conversation.compact(keep_recent=self.keep_recent)
        messages_after = len(conversation.messages) if conversation is not None else None

        return CompactOutcome(
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_after,
            llm_calls_spent=0,
        )


def run_compact(
    service: CompactService,
    *,
    hooks: list[Any],
    state: AgentState,
    session_log: SessionLog,
    llm: LLMBackend,
    conversation: Any | None,
    step_num: int,
    reason: str,
) -> CompactOutcome:
    """Invoke a :class:`CompactService` with pre/post lifecycle events.

    Fires :attr:`LifecycleEvent.PRE_COMPACT` before the service runs
    and :attr:`LifecycleEvent.POST_COMPACT` after. Event hooks
    returning :class:`HookDecision` with ``stop=...`` abort the
    compaction before the service is called.

    Returns the :class:`CompactOutcome` from the service, or a
    synthetic one with ``reason="aborted_by_hook"`` if a pre-compact
    hook requested stop.
    """
    from looplet.events import LifecycleEvent  # noqa: PLC0415

    # Pre-compact: observers can block.
    pre_decisions = _emit_compact_event(
        hooks,
        LifecycleEvent.PRE_COMPACT,
        state=state,
        session_log=session_log,
        step_num=step_num,
        messages_before=(len(conversation.messages) if conversation is not None else None),
        reason=reason,
    )
    for d in pre_decisions:
        if d.stop is not None:
            return CompactOutcome(reason=f"aborted: {d.stop}")

    outcome = service.compact(
        state=state,
        session_log=session_log,
        llm=llm,
        conversation=conversation,
        step_num=step_num,
        reason=reason,
    )

    _emit_compact_event(
        hooks,
        LifecycleEvent.POST_COMPACT,
        state=state,
        session_log=session_log,
        step_num=step_num,
        messages_before=outcome.messages_before,
        messages_after=outcome.messages_after,
        reason=reason,
    )

    # Post-compact cleanup callback — domain-specific state resets.
    if outcome.cleanup is not None:
        try:
            outcome.cleanup()
        except Exception:  # noqa: BLE001
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).exception(
                "CompactOutcome.cleanup raised; continuing",
            )

    return outcome


def _emit_compact_event(
    hooks: list[Any],
    event: Any,
    *,
    state: AgentState,
    session_log: SessionLog,
    step_num: int,
    messages_before: int | None,
    messages_after: int | None = None,
    reason: str,
) -> list[Any]:
    """Dispatch a compact lifecycle event via ``on_event``.

    Kept inline to avoid importing from :mod:`looplet.loop` (that
    would create a cycle; the loop imports from us).
    """
    from looplet.events import EventPayload  # noqa: PLC0415
    from looplet.hook_decision import HookDecision  # noqa: PLC0415

    payload = EventPayload(
        event=event,
        step_num=step_num,
        state=state,
        session_log=session_log,
        messages_before=messages_before,
        messages_after=messages_after,
        extra={"reason": reason},
    )
    decisions: list[HookDecision] = []
    for hook in hooks:
        fn = getattr(hook, "on_event", None)
        if fn is None:
            continue
        try:
            result = fn(payload)
        except Exception:  # noqa: BLE001
            # Compaction must never break the loop — log and continue.
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).exception(
                "on_event hook raised during %s; continuing",
                event,
            )
            continue
        if isinstance(result, HookDecision):
            decisions.append(result)
    return decisions


# ── LLM-driven compaction ─────────────────────────────────────────


_DEFAULT_SUMMARY_PROMPT = """You are summarising an AI agent's working
session so the agent can continue working with a much shorter context.

Produce a single concise summary covering ONLY these four sections,
in order, in plain text (no markdown headers, no code fences):

1. Task goal: one sentence restating what the agent is trying to
   accomplish.
2. Key findings: facts the agent has established, as a compact
   bulleted list. Preserve identifiers (IDs, paths, hashes, host
   names) verbatim — downstream reasoning depends on them.
3. Open questions: what remains to investigate, as a compact
   bulleted list.
4. Recent decisions: the last few tool calls and their outcomes, one
   short line each — enough for the agent to not repeat work.

Hard constraints:
* Never invent facts not present in the transcript.
* Never drop identifiers (IDs, paths, hashes, URLs, host names).
* Stay under {budget} characters.

Transcript:
{transcript}

Summary:"""


class SummarizeCompact:
    """Ask the LLM to summarise the session, then keep N recent entries.

    Spends one LLM call to produce a dense 4-section summary (goal,
    findings, open questions, recent decisions), spliced into the
    session log before the recent entries. Falls back to deterministic
    keep-recent on any summariser error — compaction always succeeds.

    Prefer for long-running autonomous sessions where reasoning-chain
    preservation matters. Avoid for sub-second latency budgets or
    deterministic/offline runs.
    """

    def __init__(
        self,
        *,
        keep_recent: int = 2,
        summary_prompt: str | None = None,
        summary_max_chars: int = 4000,
        summary_max_tokens: int = 1200,
        summary_temperature: float = 0.1,
    ) -> None:
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt or _DEFAULT_SUMMARY_PROMPT
        self.summary_max_chars = summary_max_chars
        self.summary_max_tokens = summary_max_tokens
        self.summary_temperature = summary_temperature

    def compact(
        self,
        *,
        state: AgentState,
        session_log: SessionLog,
        llm: LLMBackend,
        conversation: Any | None,
        step_num: int,
        reason: str,
    ) -> CompactOutcome:
        from looplet.scaffolding import (  # noqa: PLC0415
            emergency_truncate,
            llm_call_with_retry,
        )

        messages_before = len(conversation.messages) if conversation is not None else None

        # 1. Build transcript text: session_log.render() is the
        #    single source of truth for what the agent has seen.
        transcript = ""
        if hasattr(session_log, "render"):
            try:
                transcript = session_log.render() or ""
            except Exception:  # noqa: BLE001
                transcript = ""

        # Short-circuit: nothing to compact.
        if not transcript.strip():
            emergency_truncate(state, session_log, keep_recent=self.keep_recent)
            if conversation is not None and hasattr(conversation, "compact"):
                conversation.compact(keep_recent=self.keep_recent)
            return CompactOutcome(
                reason=reason,
                messages_before=messages_before,
                messages_after=(len(conversation.messages) if conversation is not None else None),
                llm_calls_spent=0,
                extra={"mode": "empty_fallback"},
            )

        # Escape curly braces in transcript before str.format() — tool
        # results routinely contain JSON with {/} which would cause
        # KeyError/ValueError from the format call.
        _safe_transcript = transcript.replace("{", "{{").replace("}", "}}")
        prompt = self.summary_prompt.format(
            budget=self.summary_max_chars,
            transcript=_safe_transcript,
        )

        # 2. Ask the LLM for a summary. Recovery-tier call — no retry
        #    on prompt-too-long since that's exactly what we're
        #    compacting in response to; just fall back to deterministic.
        llm_calls_spent = 0
        summary_text: str | None = None
        try:
            result = llm_call_with_retry(
                llm,
                prompt,
                max_tokens=self.summary_max_tokens,
                system_prompt="",
                temperature=self.summary_temperature,
                max_retries=0,
            )
            llm_calls_spent = 1
            if result.ok and isinstance(result.text, str):
                summary_text = result.text.strip()[: self.summary_max_chars]
        except Exception:  # noqa: BLE001
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).exception(
                "LLMCompactService summary call raised; falling back",
            )

        # 3. Apply compaction. Deterministic keep-recent runs either
        #    way (so the log is actually shorter); the summary is
        #    spliced in as a synthetic entry.
        emergency_truncate(state, session_log, keep_recent=self.keep_recent)

        if summary_text and hasattr(session_log, "entries"):
            try:
                from looplet.session import LogEntry  # noqa: PLC0415

                summary_entry = LogEntry(
                    step=step_num,
                    theory="",
                    tool="__compact_summary__",
                    reasoning=f"[compaction summary @ step {step_num}]",
                    entities_seen=[],
                    findings=[summary_text],
                )
                # Insert BEFORE recent entries so chronological order
                # is correct: [old summaries, LLM summary, recent].
                # emergency_truncate left keep_recent entries at the
                # tail; splice our summary just above them.
                entries = session_log.entries
                insert_pos = max(0, len(entries) - self.keep_recent)
                entries.insert(insert_pos, summary_entry)
            except Exception:  # noqa: BLE001
                # Session log shape varies across domains — never let
                # a splice failure break the loop.
                pass

        if conversation is not None and hasattr(conversation, "compact"):
            conversation.compact(keep_recent=self.keep_recent)

        messages_after = len(conversation.messages) if conversation is not None else None
        return CompactOutcome(
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_after,
            llm_calls_spent=llm_calls_spent,
            extra={
                "mode": "llm_summary" if summary_text else "llm_fallback",
                "summary_chars": len(summary_text) if summary_text else 0,
            },
        )


# ── Tool-result pruning ──────────────────────────────────────────


_CLEARED_MARKER = "[tool result cleared by compact]"


class PruneToolResults:
    """Clear old tool-result content, keep conversation structure intact.

    Iterates :attr:`Conversation.messages`, finds TOOL messages older
    than the last ``keep_recent`` tool results, and replaces their
    ``content`` with a short marker string. Zero LLM calls, zero
    structure changes — the message count stays the same, only the
    payload shrinks.

    Use as the cheapest first stage in a :func:`compact_chain`::

        compact_chain(
            PruneToolResults(keep_recent=5),
            SummarizeCompact(keep_recent=2),
        )

    ``compactable_tools``: when non-empty, only tool results whose
    ``tool_result.tool`` is in this set are cleared. Leave empty
    (default) to prune all tool results. Some agent frameworks
    restrict pruning to file_read, shell, grep, glob, web_search,
    web_fetch; looplet lets you decide.
    """

    def __init__(
        self,
        *,
        keep_recent: int = 5,
        compactable_tools: frozenset[str] | None = None,
        cleared_marker: str = _CLEARED_MARKER,
    ) -> None:
        self.keep_recent = keep_recent
        self.compactable_tools = compactable_tools or frozenset()
        self.cleared_marker = cleared_marker

    def compact(
        self,
        *,
        state: AgentState,
        session_log: SessionLog,
        llm: LLMBackend,
        conversation: Any | None,
        step_num: int,
        reason: str,
    ) -> CompactOutcome:
        if conversation is None or not hasattr(conversation, "messages"):
            return CompactOutcome(reason=reason, extra={"mode": "no_conversation"})

        from looplet.conversation import MessageRole  # noqa: PLC0415

        msgs = conversation.messages
        messages_before = len(msgs)

        # Collect indices of TOOL messages whose content is eligible.
        tool_indices: list[int] = []
        for i, m in enumerate(msgs):
            if m.role != MessageRole.TOOL:
                continue
            # Already cleared?
            if isinstance(m.content, str) and m.content == self.cleared_marker:
                continue
            # Filter by tool name if configured.
            if self.compactable_tools:
                tool_name = getattr(m.tool_result, "tool", "") if m.tool_result else ""
                if tool_name not in self.compactable_tools:
                    continue
            tool_indices.append(i)

        # Keep the last N; clear the rest.
        to_clear = tool_indices[: -self.keep_recent] if len(tool_indices) > self.keep_recent else []

        cleared = 0
        for idx in to_clear:
            msgs[idx].content = self.cleared_marker
            cleared += 1

        return CompactOutcome(
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_before,  # structure unchanged
            llm_calls_spent=0,
            extra={"mode": "prune", "cleared": cleared},
        )


# ── Chained compaction ───────────────────────────────────────────


def compact_chain(*services: CompactService) -> CompactService:
    """Combine multiple :class:`CompactService` implementations into
    a first-success chain.

    Each service runs in order. After each one the chain checks if
    the conversation shrank (``messages_after < messages_before``),
    or if tool results were pruned (``extra.cleared > 0``). If the
    stage had an effect, the chain stops and returns a merged
    :class:`CompactOutcome`. If not, the next stage runs.

    The last stage always runs and its outcome is returned even if
    nothing changed — this lets a terminal :class:`TruncateCompact`
    guarantee progress.

    Usage::

        config = LoopConfig(
            compact_service=compact_chain(
                PruneToolResults(keep_recent=5),
                SummarizeCompact(keep_recent=2),
                TruncateCompact(keep_recent=1),
            ),
        )
    """
    if not services:
        raise ValueError("compact_chain requires at least one service")

    return _CompactChain(list(services))


class _CompactChain:
    """Internal implementation for :func:`compact_chain`."""

    def __init__(self, stages: list[Any]) -> None:
        self._stages = stages

    def compact(
        self,
        *,
        state: AgentState,
        session_log: SessionLog,
        llm: LLMBackend,
        conversation: Any | None,
        step_num: int,
        reason: str,
    ) -> CompactOutcome:
        total_llm = 0
        for i, svc in enumerate(self._stages):
            outcome = svc.compact(
                state=state,
                session_log=session_log,
                llm=llm,
                conversation=conversation,
                step_num=step_num,
                reason=reason,
            )
            total_llm += outcome.llm_calls_spent

            # Did this stage have an effect?
            _shrank = (
                outcome.messages_before is not None
                and outcome.messages_after is not None
                and outcome.messages_after < outcome.messages_before
            )
            _pruned = outcome.extra is not None and outcome.extra.get("cleared", 0) > 0
            if _shrank or _pruned or i == len(self._stages) - 1:
                outcome.llm_calls_spent = total_llm
                if outcome.extra is None:
                    outcome.extra = {}
                outcome.extra["chain_stage"] = i
                outcome.extra["chain_stage_count"] = len(self._stages)
                return outcome

        # Unreachable — the loop always returns on the last stage.
        return CompactOutcome(reason=reason)  # pragma: no cover
