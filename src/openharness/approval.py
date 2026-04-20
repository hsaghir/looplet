"""Async approval for tool actions that need human sign-off.

The pattern:
  1. A tool detects a high-risk action and sets
     ``{"needs_approval": True, "approval_description": "..."}``
     in its result data.
  2. ``ApprovalHook.post_dispatch`` detects this and stops the loop
     with ``stop_reason="waiting_for_approval"``.
  3. The loop checkpoints (if ``checkpoint_dir`` is configured) and
     the generator terminates.
  4. An external system (webhook, Slack bot, email, human on CLI)
     records the approval.
  5. The caller restarts the loop — ``checkpoint_dir`` auto-resumes
     from the last step, and the approval is injected into context
     via a ``StaticMemorySource`` or ``pre_prompt`` hook.

Usage::

    from openharness import ApprovalHook, LoopConfig

    config = LoopConfig(
        checkpoint_dir="./checkpoints",  # auto-save + auto-resume
        approval_handler=my_sync_handler,  # or None for async-only
    )
    hooks = [ApprovalHook()]

For **sync approval** (blocks until human responds)::

    config = LoopConfig(
        approval_handler=lambda prompt, opts: input(f"{prompt} {opts}: "),
    )
    # No ApprovalHook needed — tools call ctx.approve() inline.

For **async approval** (suspend → external approval → resume)::

    config = LoopConfig(
        checkpoint_dir="./ckpt",
        approval_handler=None,   # tools get None → set needs_approval
    )
    hooks = [ApprovalHook()]
    # Loop stops at approval request.
    # Resume: restart the loop, approval injected via memory_sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openharness.session import SessionLog
    from openharness.types import AgentState, ToolCall, ToolResult

__all__ = ["ApprovalHook", "ApprovalRequest"]

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ApprovalRequest:
    """Record of a pending approval request.

    Stored by :class:`ApprovalHook` when a tool result contains
    ``needs_approval=True``. Callers can inspect ``pending`` after
    the loop stops to build notifications, webhooks, or UIs.
    """

    step: int
    tool: str
    description: str = ""
    options: list[str] = field(default_factory=lambda: ["approve", "deny"])

class ApprovalHook:
    """Hook that stops the loop when a tool requests approval.

    Detects ``needs_approval=True`` in any tool result's ``data``
    dict and stops the loop so an external system can provide the
    approval before the loop resumes.

    When the loop stops, ``self.pending`` contains the
    :class:`ApprovalRequest` — use it to send a notification,
    create a webhook, or prompt a human.

    Combine with ``LoopConfig(checkpoint_dir=...)`` for full
    crash-safe async approval::

        hook = ApprovalHook()
        config = LoopConfig(checkpoint_dir="./ckpt")
        gen = composable_loop(..., hooks=[hook], config=config)
        for step in gen:
            print(step)
        if hook.pending:
            print(f"Waiting for approval: {hook.pending}")
            # ... external approval happens ...
            # On next run, checkpoint_dir auto-resumes.
    """

    def __init__(self) -> None:
        self._pending: ApprovalRequest | None = None

    @property
    def pending(self) -> ApprovalRequest | None:
        """The pending approval request, or None if none is pending."""
        return self._pending

    def post_dispatch(
        self,
        state: AgentState,
        session_log: SessionLog,
        tool_call: ToolCall,
        tool_result: ToolResult,
        step_num: int,
    ) -> Any:
        """Detect ``needs_approval`` in tool result and stop the loop."""
        from openharness.hook_decision import HookDecision  # noqa: PLC0415

        data = getattr(tool_result, "data", None) or {}
        if not data.get("needs_approval"):
            return None

        self._pending = ApprovalRequest(
            step=step_num,
            tool=getattr(tool_call, "tool", ""),
            description=data.get("approval_description", ""),
            options=data.get("approval_options", ["approve", "deny"]),
        )
        logger.info(
            "approval_required step=%d tool=%s desc=%s",
            step_num, self._pending.tool, self._pending.description,
        )
        return HookDecision(stop="waiting_for_approval")

    def should_stop(self, state: AgentState, step_num: int, new_entities: int) -> bool:
        return False

