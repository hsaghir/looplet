"""Unified message thread for agentic conversations.

Provides a structured alternative to the three parallel state representations
(state.steps, SessionLog, context_history text). Can be used standalone or
alongside existing structures during migration.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from looplet.types import ToolCall, ToolResult

__all__ = [
    "MessageRole",
    "ContentBlock",
    "Message",
    "Conversation",
    "default_summarizer",
]


class MessageRole(str, Enum):
    """Role of a message in a conversation thread."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ContentBlock:
    """One block of multimodal message content.

    ``kind`` is a free-form discriminator — common values are ``"text"``,
    ``"image"``, ``"tool_use"``, ``"tool_result"``. ``data`` holds the
    payload in a shape native to the block kind (``{"text": "..."}`` for
    text, ``{"url": "...", "media_type": "image/png"}`` for images, etc.).

    Keeping this deliberately loose — providers disagree on exact
    shapes — so the loop remains vendor-agnostic.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Render block as plain text for prompt inclusion / fallbacks."""
        if self.kind == "text":
            return str(self.data.get("text", ""))
        if self.kind == "image":
            return "[image attached]"
        if self.kind == "tool_use":
            name = self.data.get("name", "?")
            return f"[tool_use: {name}]"
        if self.kind == "tool_result":
            return f"[tool_result: {str(self.data.get('content', ''))[:200]}]"
        return f"[{self.kind}]"


@dataclass
class Message:
    """A single message in a conversation.

    ``content`` is either a plain string (legacy / text-only case) or a
    list of :class:`ContentBlock` for multimodal / tool-native traffic.
    Use :meth:`text` or :meth:`blocks` to access content uniformly.
    """

    role: MessageRole
    content: str | list[ContentBlock]
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Flatten content to a single string (multi-block → newlines)."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(b.text for b in self.content if b.text)

    @property
    def blocks(self) -> list[ContentBlock]:
        """Normalise content to a list of :class:`ContentBlock`."""
        if isinstance(self.content, str):
            return [ContentBlock(kind="text", data={"text": self.content})]
        return list(self.content)

    def text_blocks(self) -> list[ContentBlock]:
        """Return only the text blocks — useful when stripping images
        for summarisation or for backends that don't support multimodal."""
        return [b for b in self.blocks if b.kind == "text"]


class Conversation:
    """Unified message thread for an agentic session.

    Provides append, fork, truncate, compact, render, serialize/deserialize,
    and computed properties (token_estimate, entities).

    Designed to be a drop-in complement (or eventual replacement) for
    state.steps + SessionLog + context_history text.
    """

    def __init__(self, messages: list[Message] | None = None) -> None:
        self.messages: list[Message] = list(messages) if messages else []

    # ── Core operations ──────────────────────────────────────────

    def append(self, msg: Message) -> "Conversation":
        """Append a message to the thread. Returns self for chaining."""
        self.messages.append(msg)
        return self

    def fork(self) -> "Conversation":
        """Create a deep-independent copy of this conversation.

        Mutations to the fork (append, truncate, metadata edits) do not
        affect the parent, and vice versa. Use for sub-agent branching.
        """
        return Conversation(messages=copy.deepcopy(self.messages))

    def truncate(self, keep_last: int, preserve_system: bool = True) -> "Conversation":
        """Remove old messages, keeping the last ``keep_last`` and optionally
        all SYSTEM messages.

        Args:
            keep_last: Number of most-recent messages to keep.
            preserve_system: If True, SYSTEM messages are always kept
                regardless of their position.

        Returns self for chaining.
        """
        if len(self.messages) <= keep_last:
            return self

        if preserve_system:
            system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            recent = self.messages[-keep_last:]
            # Merge: system messages + recent, deduplicating by object identity
            recent_ids = {id(m) for m in recent}
            merged = [m for m in system_msgs if id(m) not in recent_ids]
            merged.extend(recent)
            self.messages = merged
        else:
            self.messages = self.messages[-keep_last:]

        return self

    def compact(
        self,
        summarizer: Callable[[list[Message]], str] | None = None,
        keep_recent: int = 2,
    ) -> "Conversation":
        """Replace older messages with a single SYSTEM summary message.

        Keeps the most recent ``keep_recent`` messages plus the last USER
        message (if not already in the recent window). All older messages
        are replaced by a deterministic or LLM-refined summary.

        Compaction boundary messages (``metadata["kind"] ==
        "compaction_boundary"``) are preserved verbatim — dropping them
        would silently erase the record of earlier compactions.

        Args:
            summarizer: Optional callable(list[Message]) -> str. If None,
                ``default_summarizer`` is used.
            keep_recent: Number of most-recent messages to keep verbatim
                after the summary.

        Returns self for chaining.
        """
        if len(self.messages) <= keep_recent + 1:
            return self

        fn = summarizer or default_summarizer

        # Keep the last `keep_recent` messages verbatim
        split_idx = len(self.messages) - keep_recent
        to_compact = self.messages[:split_idx]
        to_keep = self.messages[split_idx:]

        # Preserve the last USER message if it's in to_compact
        last_user_in_compact = None
        for msg in to_compact:
            if msg.role == MessageRole.USER:
                last_user_in_compact = msg

        # Preserve every compaction boundary — they must survive re-compaction
        boundaries_to_keep = [
            m for m in to_compact if m.metadata.get("kind") == "compaction_boundary"
        ]

        if not to_compact:
            return self

        # Don't include boundary messages in the summary input; they're
        # already a compressed form of prior context.
        summary_input = [
            _strip_large_content(m)
            for m in to_compact
            if m.metadata.get("kind") != "compaction_boundary"
        ]
        summary_text = fn(summary_input) if summary_input else ""
        summary_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Summary of prior context]\n{summary_text}",
            metadata={
                "kind": "compaction_boundary",
                "summary": summary_text,
                "dropped_message_count": len(to_compact) - len(boundaries_to_keep),
            },
        )

        head: list[Message] = list(boundaries_to_keep)
        if summary_input:
            head.append(summary_msg)
        if last_user_in_compact is not None and last_user_in_compact not in head:
            head.append(last_user_in_compact)
        self.messages = head + to_keep
        return self

    def find_compaction_boundaries(self) -> list[Message]:
        """Return all compaction-boundary messages in order of insertion."""
        return [m for m in self.messages if m.metadata.get("kind") == "compaction_boundary"]

    # ── Rendering ────────────────────────────────────────────────

    def render(self, max_tokens: int | None = None) -> str:
        """Produce a formatted text representation for LLM prompt inclusion.

        Args:
            max_tokens: Optional token budget. Messages are truncated from
                the oldest (non-system) end when the budget is exceeded.
                Uses a 4-chars-per-token heuristic.

        Returns a multi-line string with role labels and content.
        """
        lines: list[str] = []
        for msg in self.messages:
            role_label = msg.role.value.upper()
            if msg.tool_call:
                tc = msg.tool_call
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in tc.args.items() if not k.startswith("__")
                )
                lines.append(f"[{role_label}] → {tc.tool}({args_str})")
                if tc.reasoning:
                    lines.append(f"  reasoning: {tc.reasoning}")
            elif msg.tool_result:
                tr = msg.tool_result
                if tr.error:
                    lines.append(f"[{role_label}] ✗ {tr.tool}: {tr.error}")
                else:
                    lines.append(f"[{role_label}] ✓ {tr.tool}: {str(tr.data)[:200]}")
            elif msg.content:
                lines.append(f"[{role_label}] {msg.text}")

        text = "\n".join(lines)

        if max_tokens is not None:
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[-max_chars:]

        return text

    # ── Persistence ──────────────────────────────────────────────

    def serialize(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for persistence."""
        return {
            "version": 1,
            "messages": [_serialize_message(m) for m in self.messages],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "Conversation":
        """Reconstruct a Conversation from serialized data."""
        messages = [_deserialize_message(m) for m in data.get("messages", [])]
        return cls(messages=messages)

    # ── Computed properties ──────────────────────────────────────

    @property
    def token_estimate(self) -> int:
        """Rough token count across all messages (4 chars per token)."""
        total_chars = sum(
            len(m.text)
            + (len(str(m.tool_call.args)) if m.tool_call else 0)
            + (len(str(m.tool_result.data)) if m.tool_result else 0)
            for m in self.messages
        )
        return max(0, total_chars // 4)

    @property
    def entities(self) -> set[str]:
        """Union of all entity strings from tool results.

        Looks for a ``entities`` key in tool_result.data (list of strings).
        """
        result: set[str] = set()
        for msg in self.messages:
            if msg.tool_result and isinstance(msg.tool_result.data, dict):
                ents = msg.tool_result.data.get("entities", [])
                if isinstance(ents, list):
                    result.update(str(e) for e in ents)
        return result


# ── Default summarizer ───────────────────────────────────────────


LARGE_CONTENT_TYPES: frozenset[str] = frozenset({"image", "audio", "video", "binary"})
"""Block kinds stripped before summarization — large payloads that
shouldn't be sent to a text summarizer. This is a ``frozenset`` and
cannot be mutated in place. To customize, reassign the module attribute
with a new frozenset::

    from looplet import conversation
    conversation.LARGE_CONTENT_TYPES = conversation.LARGE_CONTENT_TYPES | {"pdf"}
"""


def _strip_large_content(msg: Message) -> Message:
    """Return a copy of ``msg`` with heavy (binary / multimodal) blocks
    replaced by short text placeholders. Plain-string content is
    returned unchanged. This is applied pre-summarization so large
    payloads never reach the summarizer LLM's context.
    """
    if isinstance(msg.content, str):
        return msg
    # Short-circuit: if no heavy blocks, return the original unchanged.
    if not any(b.kind in LARGE_CONTENT_TYPES for b in msg.content):
        return msg
    stripped: list[ContentBlock] = []
    for b in msg.content:
        if b.kind in LARGE_CONTENT_TYPES:
            stripped.append(
                ContentBlock(
                    kind="text",
                    data={"text": f"[{b.kind} omitted during compaction]"},
                )
            )
        else:
            stripped.append(b)
    return Message(
        role=msg.role,
        content=stripped,
        tool_call=msg.tool_call,
        tool_result=msg.tool_result,
        metadata=msg.metadata,
        timestamp=msg.timestamp,
    )


def default_summarizer(messages: list[Message]) -> str:
    """Deterministic summarizer — no LLM required.

    Counts messages by role, lists tools called, and preserves the last
    user request if present.
    """
    from collections import Counter

    role_counts: Counter[str] = Counter()
    tools_called: list[str] = []
    last_user_content: str = ""

    for msg in messages:
        role_counts[msg.role.value] += 1
        if msg.tool_call:
            tools_called.append(msg.tool_call.tool)
        if msg.role == MessageRole.USER and msg.content:
            last_user_content = msg.text

    parts: list[str] = []

    if role_counts:
        counts_str = ", ".join(f"{count} {role}" for role, count in sorted(role_counts.items()))
        parts.append(f"Prior context: {counts_str} messages.")

    if tools_called:
        unique_tools = list(dict.fromkeys(tools_called))  # preserve order, dedup
        parts.append(f"Tools called: {', '.join(unique_tools)}.")

    if last_user_content:
        parts.append(f"Last request: {last_user_content[:200]}")

    return " ".join(parts) if parts else "Prior conversation context."


# ── Serialization helpers ────────────────────────────────────────


def _serialize_message(msg: Message) -> dict[str, Any]:
    if isinstance(msg.content, str):
        content_repr: Any = msg.content
    else:
        content_repr = [{"kind": b.kind, "data": b.data} for b in msg.content]
    d: dict[str, Any] = {
        "role": msg.role.value,
        "content": content_repr,
        "timestamp": msg.timestamp,
        "metadata": msg.metadata,
    }
    if msg.tool_call:
        tc = msg.tool_call
        d["tool_call"] = {
            "tool": tc.tool,
            "args": tc.args,
            "reasoning": tc.reasoning,
            "call_id": tc.call_id,
        }
    if msg.tool_result:
        tr = msg.tool_result
        d["tool_result"] = {
            "tool": tr.tool,
            "args_summary": tr.args_summary,
            "data": tr.data,
            "error": tr.error,
            "error_kind": tr.error_detail.kind.value if tr.error_detail else None,
            "error_retriable": tr.error_detail.retriable if tr.error_detail else None,
            "error_context": tr.error_detail.context if tr.error_detail else None,
            "duration_ms": tr.duration_ms,
            "result_key": tr.result_key,
            "call_id": tr.call_id,
        }
    return d


def _deserialize_message(d: dict[str, Any]) -> Message:
    role = MessageRole(d["role"])
    content_raw = d.get("content", "")
    content: str | list[ContentBlock]
    if isinstance(content_raw, list):
        content = [
            ContentBlock(kind=b.get("kind", "text"), data=b.get("data", {})) for b in content_raw
        ]
    else:
        content = content_raw
    timestamp = d.get("timestamp", time.time())
    metadata = d.get("metadata", {})

    tool_call: ToolCall | None = None
    if "tool_call" in d and d["tool_call"]:
        tc_data = d["tool_call"]
        tool_call = ToolCall(
            tool=tc_data["tool"],
            args=tc_data.get("args", {}),
            reasoning=tc_data.get("reasoning", ""),
            call_id=tc_data.get("call_id", ""),
        )

    tool_result: ToolResult | None = None
    if "tool_result" in d and d["tool_result"]:
        tr_data = d["tool_result"]
        # Reconstruct error_detail from serialized error_kind/error_retriable.
        _error_detail = None
        if tr_data.get("error_kind"):
            from looplet.types import ErrorKind, ToolError  # noqa: PLC0415

            _error_detail = ToolError(
                kind=ErrorKind(tr_data["error_kind"]),
                message=tr_data.get("error", ""),
                retriable=tr_data.get("error_retriable", False),
                context=tr_data.get("error_context") or {},
            )
        tool_result = ToolResult(
            tool=tr_data["tool"],
            args_summary=tr_data.get("args_summary", ""),
            data=tr_data.get("data"),
            error=tr_data.get("error"),
            error_detail=_error_detail,
            duration_ms=tr_data.get("duration_ms", 0.0),
            result_key=tr_data.get("result_key"),
            call_id=tr_data.get("call_id"),
        )

    return Message(
        role=role,
        content=content,
        tool_call=tool_call,
        tool_result=tool_result,
        metadata=metadata,
        timestamp=timestamp,
    )
