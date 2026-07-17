"""Unified history write path for the agent loop.

``HistoryRecorder`` is the single call site that writes a completed step
or an LLM turn into any combination of the three history surfaces:

* ``Conversation`` - authoritative message thread (user / assistant / tool).
* ``SessionLog`` - per-step narrative log used in prompt assembly.
* ``state.steps`` - the ``AgentState`` list consumed by
  ``context_summary()`` / ``snapshot()``.

Before this module existed the composable loop wrote to each surface
directly, producing three parallel records of the same events and making
any future migration (e.g. making ``Conversation`` the sole source of
truth) invasive. The recorder gives us one site to change.

All three surfaces are optional. Passing only the subset the caller
cares about keeps the recorder usable from sub-agents, tests, and
domains that run without a SessionLog.
"""

from __future__ import annotations

from typing import Any

from looplet.conversation import Conversation, Message, MessageRole
from looplet.session import SessionLog
from looplet.types import Step


class HistoryRecorder:
    """Route every step and LLM turn through a single write path.

    Args:
        state: Agent state whose ``steps`` list should receive completed
            ``Step`` objects. Optional - omit when the domain has no
            state (e.g. sub-agents with their own minimal state).
        session_log: SessionLog that should receive one ``LogEntry`` per
            step. Optional.
        conversation: Conversation thread to append tool-use /
            tool-result / user / assistant messages to. Optional.
        max_message_chars: Upper bound on how much LLM prompt or response
            text is stored verbatim on ``Conversation`` messages. Protects
            long-running agents from blowing out message storage.
    """

    def __init__(
        self,
        *,
        state: Any | None = None,
        session_log: SessionLog | None = None,
        conversation: Conversation | None = None,
        max_message_chars: int = 5000,
    ) -> None:
        self._state = state
        self._session_log = session_log
        self._conversation = conversation
        self._max_chars = max_message_chars

    # ── Step recording ───────────────────────────────────────────

    def record_step(
        self,
        step: Step,
        *,
        theory: str = "",
        entities: list[str] | None = None,
        findings: list[str] | None = None,
        highlights: list[str] | None = None,
        recall_key: str = "",
    ) -> None:
        """Record a completed Step across every attached history surface.

        Idempotent with respect to ``state.steps``: if the caller already
        appended the step before calling us (common during a gradual
        migration) we do not duplicate it.
        """
        if self._state is not None and hasattr(self._state, "steps"):
            if not self._state.steps or self._state.steps[-1] is not step:
                self._state.steps.append(step)

        if self._session_log is not None:
            self._session_log.record(
                step=step.number,
                theory=theory,
                tool=step.tool_call.tool,
                reasoning=step.tool_call.reasoning,
                entities=list(entities or []),
                findings=list(findings or []),
                highlights=list(highlights or []),
                recall_key=recall_key or (step.tool_result.result_key or ""),
            )

        if self._conversation is not None:
            self._conversation.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_call=step.tool_call,
                )
            )
            self._conversation.append(
                Message(
                    role=MessageRole.TOOL,
                    content="",
                    tool_result=step.tool_result,
                )
            )

    # ── Compaction boundary recording ───────────────────────────

    def record_compaction_boundary(
        self,
        *,
        summary: str,
        dropped_step_range: tuple[int, int],
    ) -> None:
        """Mark where the loop compacted older context into a summary.

        Appends a SYSTEM-role message with
        ``metadata["kind"] == "compaction_boundary"`` and the summary +
        dropped step range on metadata. The boundary is preserved by
        ``Conversation.compact()`` so it survives subsequent
        compactions - giving the LLM (and debuggers) an explicit
        record of what was compressed and when.

        No-op if no ``Conversation`` is attached. Does not touch
        ``state.steps`` or the ``SessionLog``.
        """
        if self._conversation is None:
            return
        self._conversation.append(
            Message(
                role=MessageRole.SYSTEM,
                content=summary,
                metadata={
                    "kind": "compaction_boundary",
                    "summary": summary,
                    "dropped_step_range": dropped_step_range,
                },
            )
        )

    # ── LLM turn recording ──────────────────────────────────────

    def record_llm_turn(self, *, prompt: str, response: Any) -> None:
        """Record a prompt/response pair on the conversation thread.

        Non-string responses (for example the list-of-blocks shape
        returned by native tool-calling backends) are coerced to their
        string repr so the message thread stays JSON-serialisable and
        inspectable. Both prompt and response are truncated to
        ``max_message_chars`` before storage.
        """
        if self._conversation is None:
            return
        self._conversation.append(
            Message(
                role=MessageRole.USER,
                content=str(prompt)[: self._max_chars],
            )
        )
        if response is None:
            return
        response_text = response if isinstance(response, str) else str(response)
        self._conversation.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=response_text[: self._max_chars],
            )
        )
