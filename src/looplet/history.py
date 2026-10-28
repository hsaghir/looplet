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

from collections.abc import Callable
from typing import Any

from looplet._backend_contract import json_snapshot
from looplet.conversation import ContentBlock, Conversation, Message, MessageRole
from looplet.session import SessionLog
from looplet.types import Step, ToolCall, ToolResult


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
        typed_turns: bool = False,
        project_tool_result: Callable[[ToolResult], ToolResult] | None = None,
        observe_tool_result: Callable[[ToolCall, ToolResult], None] | None = None,
    ) -> None:
        self._state = state
        self._session_log = session_log
        self._conversation = conversation
        self._max_chars = max_message_chars
        self._typed_turns = typed_turns
        self._project_tool_result = project_tool_result
        self._observe_tool_result = observe_tool_result

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

        if (
            self._observe_tool_result is not None
            and step.tool_call.call_id
            and not step.tool_call.tool.startswith("__")
            and step.tool_call.metadata.get("completion_source") != "assistant_message"
        ):
            self._observe_tool_result(step.tool_call, step.tool_result)

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
            if (
                self._typed_turns
                and step.tool_call.metadata.get("completion_source") == "assistant_message"
            ):
                result = step.tool_result
                data = result.data if isinstance(result.data, dict) else {}
                if result.error or data.get("rejected") is True or data.get("status") == "rejected":
                    reason = str(
                        data.get("reason")
                        or data.get("error")
                        or result.error
                        or "Final answer was not accepted"
                    )
                    model_turn_id = next(
                        (
                            message.metadata.get("model_turn_id", "")
                            for message in reversed(self._conversation.messages)
                            if message.metadata.get("origin") == "model"
                        ),
                        "",
                    )
                    self.record_host_feedback(reason, model_turn_id=model_turn_id)
                return
            if not self._typed_turns:
                self._conversation.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content="",
                        tool_call=step.tool_call,
                        metadata=dict(step.metadata),
                    )
                )
            self._conversation.append(
                Message(
                    role=MessageRole.TOOL,
                    content="",
                    tool_result=(
                        self._project_tool_result(step.tool_result)
                        if self._project_tool_result is not None
                        else step.tool_result
                    ),
                    metadata={
                        **step.metadata,
                        "tool_call_id": step.tool_call.call_id,
                        **(
                            {
                                "model_turn_id": next(
                                    (
                                        message.metadata.get("model_turn_id")
                                        for message in reversed(self._conversation.messages)
                                        if message.metadata.get("origin") == "model"
                                    ),
                                    None,
                                )
                            }
                            if self._typed_turns
                            else {}
                        ),
                    },
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

    def record_llm_turn(
        self, *, prompt: str, response: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a prompt/response pair on the conversation thread.

        Non-string responses (for example the list-of-blocks shape
        returned by native tool-calling backends) are coerced to their
        string repr so the message thread stays JSON-serialisable and
        inspectable. Both prompt and response are truncated to
        ``max_message_chars`` before storage.
        """
        if self._conversation is None:
            return
        if self._typed_turns:
            if response is not None:
                snapshot = json_snapshot(response)
                content = (
                    [
                        ContentBlock(
                            kind=block.get("type", "unknown"),
                            data={key: value for key, value in block.items() if key != "type"},
                        )
                        for block in snapshot
                        if isinstance(block, dict)
                    ]
                    if isinstance(snapshot, list)
                    else str(snapshot)
                )
                self._conversation.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=content,
                        metadata={**(metadata or {}), "origin": "model"},
                    )
                )
            return
        self._conversation.append(
            Message(
                role=MessageRole.USER,
                content=str(prompt)[: self._max_chars],
                metadata=dict(metadata or {}),
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

    def record_host_feedback(self, feedback: str, *, model_turn_id: str) -> None:
        """Retain explicit host feedback without raising its instruction priority."""
        if feedback and self._conversation is not None:
            self._conversation.append(
                Message(
                    role=MessageRole.USER,
                    content=feedback,
                    metadata={
                        "origin": "host",
                        "kind": "assistant_feedback",
                        "model_turn_id": model_turn_id,
                    },
                )
            )
