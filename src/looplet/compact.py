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

The default service :class:`DefaultCompactService` is the recommended
production starting point: it prunes old bulky tool payloads, preserves
the older working session in a compact summary, and falls back to
deterministic truncation when the summariser cannot help.
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
    "DefaultCompactService",
    "TruncateCompact",
    "SummarizeCompact",
    "PruneToolResults",
    "compact_chain",
    "default_compact_service",
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
    session_entries_before: int | None = None
    session_entries_after: int | None = None
    compacted_step_range: tuple[int, int] | None = None
    summary: str = ""
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

        Checks conversation message counts, session-log entry counts,
        explicit compacted step ranges, and tool-result pruning counts.
        This keeps chained services honest when a stage mutates only
        one history surface.
        """
        if (
            self.messages_before is not None
            and self.messages_after is not None
            and self.messages_after < self.messages_before
        ):
            return True
        if (
            self.session_entries_before is not None
            and self.session_entries_after is not None
            and self.session_entries_after < self.session_entries_before
        ):
            return True
        if self.compacted_step_range is not None:
            return True
        if self.extra and self.extra.get("cleared", 0) > 0:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return a small JSON-able summary for lifecycle events and tests."""
        return {
            "reason": self.reason,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "session_entries_before": self.session_entries_before,
            "session_entries_after": self.session_entries_after,
            "compacted_step_range": self.compacted_step_range,
            "summary": self.summary,
            "llm_calls_spent": self.llm_calls_spent,
            "compacted": self.compacted,
            "extra": dict(self.extra or {}),
        }


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


def _message_count(conversation: Any | None) -> int | None:
    if conversation is None or not hasattr(conversation, "messages"):
        return None
    return len(conversation.messages)


def _session_entry_count(session_log: Any | None) -> int | None:
    if session_log is None or not hasattr(session_log, "entries"):
        return None
    return len(session_log.entries)


def _older_entries(entries: list[Any], keep_recent: int) -> list[Any]:
    if keep_recent <= 0:
        return list(entries)
    return list(entries[:-keep_recent])


def _compacted_step_range(session_log: Any | None, keep_recent: int) -> tuple[int, int] | None:
    if session_log is None or not hasattr(session_log, "entries"):
        return None
    entries = _older_entries(list(session_log.entries), keep_recent)
    compactable = [
        entry
        for entry in entries
        if getattr(entry, "tool", "") not in {"__summary__", "__compact_summary__"}
        and isinstance(getattr(entry, "step", None), int)
        and getattr(entry, "step") > 0
    ]
    if not compactable:
        return None
    return (compactable[0].step, compactable[-1].step)


def _session_log_was_compacted(
    *,
    summary: str,
    entries_before: int | None,
    entries_after: int | None,
) -> bool:
    return bool(
        summary
        or (
            entries_before is not None
            and entries_after is not None
            and entries_after < entries_before
        )
    )


def _render_transcript(session_log: Any | None, conversation: Any | None) -> tuple[str, str]:
    if session_log is not None and hasattr(session_log, "render"):
        try:
            rendered = session_log.render() or ""
        except Exception:  # noqa: BLE001
            rendered = ""
        if rendered.strip():
            return rendered, "session_log"
    if conversation is not None and hasattr(conversation, "render"):
        try:
            rendered = conversation.render() or ""
        except Exception:  # noqa: BLE001
            rendered = ""
        if rendered.strip():
            return rendered, "conversation"
    return "", "empty"


def _compact_conversation_if_smaller(
    conversation: Any | None,
    *,
    keep_recent: int,
    summarizer: Callable[[list[Any]], str] | None = None,
) -> int | None:
    """Compact a conversation only when the message count shrinks."""
    if conversation is None or not hasattr(conversation, "compact"):
        return _message_count(conversation)
    before = _message_count(conversation)
    original_messages = list(getattr(conversation, "messages", []))
    if summarizer is None:
        conversation.compact(keep_recent=keep_recent)
    else:
        conversation.compact(summarizer=summarizer, keep_recent=keep_recent)
    after = _message_count(conversation)
    if before is not None and after is not None and after >= before:
        conversation.messages = original_messages
        return before
    return after


def _stage_report(name: str, outcome: CompactOutcome) -> dict[str, Any]:
    return {
        "name": name,
        "compacted": outcome.compacted,
        "messages_before": outcome.messages_before,
        "messages_after": outcome.messages_after,
        "session_entries_before": outcome.session_entries_before,
        "session_entries_after": outcome.session_entries_after,
        "compacted_step_range": outcome.compacted_step_range,
        "llm_calls_spent": outcome.llm_calls_spent,
        "mode": (outcome.extra or {}).get("mode"),
        "cleared": (outcome.extra or {}).get("cleared", 0),
    }


def _merge_outcomes(reason: str, outcomes: list[tuple[str, CompactOutcome]]) -> CompactOutcome:
    """Combine several stage outcomes into one production-facing report."""
    if not outcomes:
        return CompactOutcome(reason=reason)
    first = outcomes[0][1]
    last = outcomes[-1][1]
    step_ranges = [out.compacted_step_range for _, out in outcomes if out.compacted_step_range]
    compacted_step_range = None
    if step_ranges:
        compacted_step_range = (min(r[0] for r in step_ranges), max(r[1] for r in step_ranges))
    summary = next((out.summary for _, out in outcomes if out.summary), "")
    stages = [_stage_report(name, out) for name, out in outcomes]
    extra: dict[str, Any] = {
        "mode": "default",
        "stages": stages,
        "stage_count": len(outcomes),
    }
    for _, out in outcomes:
        if out.extra:
            for key in ("cleared",):
                if key in out.extra:
                    extra[key] = extra.get(key, 0) + out.extra[key]
    return CompactOutcome(
        reason=reason,
        messages_before=first.messages_before,
        messages_after=last.messages_after,
        session_entries_before=first.session_entries_before,
        session_entries_after=last.session_entries_after,
        compacted_step_range=compacted_step_range,
        summary=summary,
        llm_calls_spent=sum(out.llm_calls_spent for _, out in outcomes),
        extra=extra,
    )


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

        messages_before = _message_count(conversation)
        session_entries_before = _session_entry_count(session_log)
        candidate_step_range = _compacted_step_range(session_log, self.keep_recent)
        summary = emergency_truncate(state, session_log, keep_recent=self.keep_recent) or ""
        session_entries_after = _session_entry_count(session_log)
        session_was_compacted = _session_log_was_compacted(
            summary=summary,
            entries_before=session_entries_before,
            entries_after=session_entries_after,
        )
        compacted_step_range = candidate_step_range if session_was_compacted else None

        # Conversation side (optional — most domains don't thread one).
        messages_after = _compact_conversation_if_smaller(
            conversation,
            keep_recent=self.keep_recent,
        )

        return CompactOutcome(
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_after,
            session_entries_before=session_entries_before,
            session_entries_after=session_entries_after,
            compacted_step_range=compacted_step_range,
            summary=summary,
            llm_calls_spent=0,
            extra={"mode": "truncate", "keep_recent": self.keep_recent},
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
        outcome=outcome,
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
    outcome: CompactOutcome | None = None,
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
        extra={"reason": reason, **({"outcome": outcome.to_dict()} if outcome else {})},
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
    findings, open questions, recent decisions). When the session log
    is long enough to shorten, the summary is spliced in before the
    recent entries. Falls back to deterministic keep-recent on any
    summariser error — compaction always succeeds.

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

        messages_before = _message_count(conversation)
        session_entries_before = _session_entry_count(session_log)
        candidate_step_range = _compacted_step_range(session_log, self.keep_recent)

        # 1. Build transcript text: session_log.render() is the
        #    preferred source of truth for what the agent has seen;
        #    conversation.render() is the fallback for loops that do
        #    not thread a SessionLog.
        transcript, transcript_source = _render_transcript(session_log, conversation)

        # Short-circuit: nothing to compact.
        if not transcript.strip():
            summary = emergency_truncate(state, session_log, keep_recent=self.keep_recent) or ""
            session_entries_after = _session_entry_count(session_log)
            session_was_compacted = _session_log_was_compacted(
                summary=summary,
                entries_before=session_entries_before,
                entries_after=session_entries_after,
            )
            compacted_step_range = candidate_step_range if session_was_compacted else None
            messages_after = _compact_conversation_if_smaller(
                conversation,
                keep_recent=self.keep_recent,
            )
            return CompactOutcome(
                reason=reason,
                messages_before=messages_before,
                messages_after=messages_after,
                session_entries_before=session_entries_before,
                session_entries_after=session_entries_after,
                compacted_step_range=compacted_step_range,
                summary=summary,
                llm_calls_spent=0,
                extra={"mode": "empty_fallback", "transcript_source": transcript_source},
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
                "SummarizeCompact summary call raised; falling back",
            )

        # 3. Apply compaction. Deterministic keep-recent runs either
        #    way (so the log is actually shorter); the summary is
        #    spliced in as a synthetic entry.
        fallback_summary = (
            emergency_truncate(state, session_log, keep_recent=self.keep_recent) or ""
        )
        session_entries_after_truncate = _session_entry_count(session_log)
        session_was_compacted = _session_log_was_compacted(
            summary=fallback_summary,
            entries_before=session_entries_before,
            entries_after=session_entries_after_truncate,
        )
        compacted_step_range = candidate_step_range if session_was_compacted else None

        if summary_text and session_was_compacted and hasattr(session_log, "entries"):
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

        messages_after = _compact_conversation_if_smaller(
            conversation,
            keep_recent=self.keep_recent,
            summarizer=(lambda _messages: summary_text) if summary_text else None,
        )
        final_summary = summary_text or fallback_summary
        return CompactOutcome(
            reason=reason,
            messages_before=messages_before,
            messages_after=messages_after,
            session_entries_before=session_entries_before,
            session_entries_after=_session_entry_count(session_log),
            compacted_step_range=compacted_step_range,
            summary=final_summary,
            llm_calls_spent=llm_calls_spent,
            extra={
                "mode": "llm_summary" if summary_text else "llm_fallback",
                "summary_chars": len(summary_text) if summary_text else 0,
                "transcript_source": transcript_source,
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
        session_entries_before = _session_entry_count(session_log)
        if conversation is None or not hasattr(conversation, "messages"):
            return CompactOutcome(
                reason=reason,
                session_entries_before=session_entries_before,
                session_entries_after=_session_entry_count(session_log),
                extra={"mode": "no_conversation"},
            )

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
            session_entries_before=session_entries_before,
            session_entries_after=_session_entry_count(session_log),
            llm_calls_spent=0,
            extra={"mode": "prune", "cleared": cleared},
        )


# ── Production default ───────────────────────────────────────────


class DefaultCompactService:
    """Recommended production compaction service.

    The service keeps looplet's core simple by composing ordinary
    compaction stages:

    1. :class:`PruneToolResults` clears old bulky tool payloads while
       keeping conversation structure intact.
    2. :class:`SummarizeCompact` preserves older working context in a
       concise summary and keeps recent entries verbatim.
    3. :class:`TruncateCompact` runs only if the first two stages had
       no effect, providing a deterministic last-resort fallback.

    Use this when you want the normal, production-ready behaviour and
    do not need to choose individual stages yourself. Use
    :func:`compact_chain` or your own :class:`CompactService` when you
    want a different policy.
    """

    def __init__(
        self,
        *,
        keep_recent: int = 2,
        keep_recent_tool_results: int = 5,
        use_llm_summary: bool = True,
        summary_max_chars: int = 4000,
        summary_max_tokens: int = 1200,
        compactable_tools: frozenset[str] | None = None,
    ) -> None:
        self.keep_recent = keep_recent
        self.keep_recent_tool_results = keep_recent_tool_results
        self.use_llm_summary = use_llm_summary
        self.prune = PruneToolResults(
            keep_recent=keep_recent_tool_results,
            compactable_tools=compactable_tools,
        )
        self.summarize: CompactService
        if use_llm_summary:
            self.summarize = SummarizeCompact(
                keep_recent=keep_recent,
                summary_max_chars=summary_max_chars,
                summary_max_tokens=summary_max_tokens,
            )
        else:
            self.summarize = TruncateCompact(keep_recent=keep_recent)
        self.fallback = TruncateCompact(keep_recent=max(1, keep_recent))

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
        outcomes: list[tuple[str, CompactOutcome]] = []

        prune_outcome = self.prune.compact(
            state=state,
            session_log=session_log,
            llm=llm,
            conversation=conversation,
            step_num=step_num,
            reason=reason,
        )
        outcomes.append(("prune_tool_results", prune_outcome))

        summarize_outcome = self.summarize.compact(
            state=state,
            session_log=session_log,
            llm=llm,
            conversation=conversation,
            step_num=step_num,
            reason=reason,
        )
        outcomes.append(
            ("summarize_context" if self.use_llm_summary else "truncate_context", summarize_outcome)
        )

        if not any(outcome.compacted for _, outcome in outcomes):
            fallback_outcome = self.fallback.compact(
                state=state,
                session_log=session_log,
                llm=llm,
                conversation=conversation,
                step_num=step_num,
                reason=reason,
            )
            outcomes.append(("fallback_truncate", fallback_outcome))

        merged = _merge_outcomes(reason, outcomes)
        if merged.extra is None:
            merged.extra = {}
        merged.extra.update(
            {
                "mode": "default",
                "keep_recent": self.keep_recent,
                "keep_recent_tool_results": self.keep_recent_tool_results,
                "use_llm_summary": self.use_llm_summary,
            }
        )
        return merged


def default_compact_service(**kwargs: Any) -> DefaultCompactService:
    """Return the recommended production compaction service.

    This small factory is convenient in workspace resource builders::

        from looplet import default_compact_service

        def build(runtime=None):
            return default_compact_service(keep_recent=3)
    """

    return DefaultCompactService(**kwargs)


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
            if outcome.compacted or i == len(self._stages) - 1:
                outcome.llm_calls_spent = total_llm
                if outcome.extra is None:
                    outcome.extra = {}
                outcome.extra["chain_stage"] = i
                outcome.extra["chain_stage_count"] = len(self._stages)
                return outcome

        # Unreachable — the loop always returns on the last stage.
        return CompactOutcome(reason=reason)  # pragma: no cover
