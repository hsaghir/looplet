"""Unified hook return type — one dataclass, every lifecycle slot.

Every hook method on :class:`looplet.loop.LoopHook` traditionally
had its own return shape: ``str | None`` for briefing injection,
``bool`` for permission, ``ToolResult | None`` for dispatch
intercept, etc. That sprawl makes it painful to add new capabilities
(mutating tool args, structured stop reasons, permission grants)
without breaking everyone.

``HookDecision`` collapses those slots into **one dataclass with
optional fields**. Every hook method now returns
``HookDecision | None``; ``None`` means "no opinion, proceed as
default". Fields that don't apply to the current call site are
silently ignored, so a hook can safely set fields that only matter
for one slot without guessing which method the loop will call.

The dataclass is intentionally flat — no inheritance, no variants,
one level of optional attributes. That keeps the API surface small
and makes it trivial to inspect a decision in logs.

All existing hook return types (``str``, ``bool``,
``ToolResult``) remain accepted at their call sites in this
release for backward compatibility. New code should return
:class:`HookDecision` for clarity and composability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from looplet.types import ToolResult

__all__ = [
    "HookDecision",
    "Allow",
    "Deny",
    "Block",
    "Stop",
    "Continue",
    "InjectContext",
    "normalize_hook_return",
]


@dataclass
class HookDecision:
    """The single unified return type for every hook method.

    Fields are evaluated per call site. A hook that runs in a slot the
    field doesn't apply to — e.g. setting ``updated_args`` in
    ``on_loop_end`` — is a silent no-op, never a crash. This lets one
    hook cover multiple lifecycle events without switching on method
    name.

    Attributes:
        block: When set, short-circuits a tool call (``pre_tool_use``)
            or a ``done()`` acceptance (``check_done``) with this
            message. The string is surfaced to the model in the next
            briefing. ``None`` means "allow".
        stop: When set on any hook during a step, signals the loop to
            terminate after the current step completes. The string is
            the ``termination_reason`` — captured in trajectories and
            logs. ``None`` means "continue".
        updated_args: When set on ``pre_tool_use``, replaces the tool
            call's arguments before dispatch. Enables auto-correction
            hooks without re-prompting the model. ``None`` means
            "use the model-provided args as-is".
        updated_result: When set on ``pre_tool_use``, **short-circuits
            the tool** and records this result instead (cache hit,
            mocked call, deterministic fixture). When set on
            ``post_tool_use``, **rewrites** the real result before it
            lands in history. ``None`` in either slot means "use the
            real tool output".
        permission: When set on ``pre_tool_use``, grants or refuses
            the call directly — ``"allow"`` proceeds to dispatch;
            ``"deny"`` converts to a ``ToolError(kind=PERMISSION_DENIED)``
            using ``block`` as the human-readable reason. Collapses
            the old ``check_permission`` + ``PermissionEngine`` duality
            into one field.
        additional_context: Plain text appended to the next briefing.
            Works on every hook slot — pre-prompt, post-dispatch,
            on_compact, etc. Subject to ``max_briefing_tokens``.
        metadata: Free-form dict preserved in trajectory records.
            Good for hook-specific telemetry that shouldn't leak
            into the prompt.
    """

    block: str | None = None
    stop: str | None = None
    updated_args: dict[str, Any] | None = None
    updated_result: ToolResult | None = None
    permission: str | None = None  # "allow" | "deny" | None
    additional_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Convenience predicates ───────────────────────────────────

    def is_block(self) -> bool:
        """True when this decision blocks (tool denial or done rejection)."""
        return self.block is not None or self.permission == "deny"

    def is_stop(self) -> bool:
        """True when this decision terminates the loop."""
        return self.stop is not None

    def is_noop(self) -> bool:
        """True when this decision carries no side effects."""
        return (
            self.block is None
            and self.stop is None
            and self.updated_args is None
            and self.updated_result is None
            and self.permission is None
            and self.additional_context is None
            and not self.metadata
        )


# ── Ergonomic constructors ──────────────────────────────────────
#
# These are thin wrappers around ``HookDecision(...)`` that read
# naturally at call sites:
#
#     return Deny("path outside sandbox")
#     return Stop("budget exceeded")
#     return InjectContext("remember: this is a dry run")
#
# Each returns a ``HookDecision`` — they're factories, not classes.


def Allow(updated_args: dict[str, Any] | None = None) -> HookDecision:
    """Grant a tool call, optionally rewriting its arguments."""
    return HookDecision(permission="allow", updated_args=updated_args)


def Deny(reason: str, *, retry: bool = False) -> HookDecision:
    """Refuse a tool call. The reason is surfaced to the model.

    ``retry=True`` signals that the model may legitimately try again
    with different args (recorded in ``metadata["retry"]`` for hooks
    and logs to observe).
    """
    return HookDecision(
        permission="deny",
        block=reason,
        metadata={"retry": retry} if retry else {},
    )


def Block(reason: str) -> HookDecision:
    """Reject a ``done()`` call or abort a tool without a permission
    judgement. The reason is surfaced to the model."""
    return HookDecision(block=reason)


def Stop(reason: str) -> HookDecision:
    """Terminate the loop cleanly after the current step."""
    return HookDecision(stop=reason)


def Continue(additional_context: str | None = None) -> HookDecision:
    """Explicit no-op. Useful when you want to attach ``additional_context``
    without any other effect."""
    return HookDecision(additional_context=additional_context)


def InjectContext(text: str) -> HookDecision:
    """Append ``text`` to the next briefing. Equivalent to returning a
    plain string from the legacy ``pre_prompt`` / ``post_dispatch``
    hook signatures."""
    return HookDecision(additional_context=text)


# ── Legacy → HookDecision coercion ─────────────────────────────


def normalize_hook_return(
    value: Any,
    *,
    slot: str,
) -> HookDecision | None:
    """Coerce a legacy hook return value into a :class:`HookDecision`.

    The loop still calls hooks with the legacy method names and return
    types. This helper folds the old shapes into the new one so the
    loop body can uniformly inspect a :class:`HookDecision`:

        * ``None`` → ``None``
        * :class:`HookDecision` → pass through
        * ``str`` → ``InjectContext(s)`` for briefing slots, or
          ``Block(s)`` for ``check_done``
        * ``bool`` → ``Allow()`` / ``Deny("permission denied")`` for
          ``check_permission`` slots; ``Stop("hook")`` / ``None`` for
          ``should_stop``
        * :class:`ToolResult` → ``HookDecision(updated_result=r)``
          (dispatch-intercept)

    Anything else raises ``TypeError`` — hooks that return garbage
    should fail loud, not silently drop.
    """
    if value is None:
        return None
    if isinstance(value, HookDecision):
        return value
    # ToolResult — dispatch intercept.
    if isinstance(value, ToolResult):
        return HookDecision(updated_result=value)
    if isinstance(value, bool):
        if slot == "check_permission":
            return Allow() if value else Deny("permission denied")
        if slot == "should_stop":
            return Stop("hook_requested_stop") if value else None
        # Bools from other slots are nonsense — signal cleanly.
        raise TypeError(
            f"hook slot {slot!r} received bool {value!r}; expected HookDecision | str | None"
        )
    if isinstance(value, str):
        if slot in ("pre_prompt", "post_dispatch"):
            return InjectContext(value)
        if slot == "check_done":
            return Block(value)
        # Strings from unexpected slots — accept as briefing rather
        # than crash; the old "return a string" behaviour was additive
        # in every case that shipped.
        return InjectContext(value)
    raise TypeError(
        f"hook slot {slot!r} returned {type(value).__name__} "
        f"{value!r}; expected HookDecision | str | bool | ToolResult | None"
    )
