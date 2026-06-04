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
    "RewriteThread",
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
    rewrite_thread: dict[str, Any] | None = None
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
            and self.rewrite_thread is None
            and not self.metadata
        )

    # ── Wire round-trip (Loop Effect Protocol §3) ───────────────
    #
    # ``to_wire`` / ``from_wire`` are the executable fidelity map: an
    # out-of-process hook returns an *effect* as JSON, and the host
    # reconstructs the identical :class:`HookDecision` it would have
    # gotten from an in-process hook. The form is loss-free for every
    # field a hook can set, which is what makes cartridge⇄library
    # translation behaviourally lossless for pure hooks (§5).

    def to_wire(self) -> dict[str, Any]:
        """Serialise to a JSON-safe effect dict (all fields preserved)."""
        return {
            "kind": "HookDecision",
            "block": self.block,
            "stop": self.stop,
            "updated_args": self.updated_args,
            "updated_result": _toolresult_to_wire(self.updated_result),
            "permission": self.permission,
            "additional_context": self.additional_context,
            "rewrite_thread": dict(self.rewrite_thread) if self.rewrite_thread else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, raw: Any) -> "HookDecision | None":
        """Reconstruct a :class:`HookDecision` from an effect dict.

        Accepts two shapes:

        * the canonical generic form emitted by :meth:`to_wire`
          (``{"kind": "HookDecision", ...}``), and
        * the *ergonomic* algebra forms a hand-written or non-Python
          policy server is likely to emit, keyed by effect constructor
          name (``Allow``/``Deny``/``Block``/``Stop``/``Continue``/
          ``InjectContext``/``UpdateArgs``/``UpdateResult``).

        ``None`` or an explicit ``{"kind": "Continue"}`` with no payload
        yields ``None`` (no-opinion), matching legacy hook semantics.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise TypeError(f"effect must be a dict, got {type(raw).__name__}")
        kind = raw.get("kind")

        if kind == "HookDecision":
            return cls(
                block=raw.get("block"),
                stop=raw.get("stop"),
                updated_args=raw.get("updated_args"),
                updated_result=_toolresult_from_wire(raw.get("updated_result")),
                permission=raw.get("permission"),
                additional_context=raw.get("additional_context"),
                rewrite_thread=(
                    dict(raw["rewrite_thread"])
                    if isinstance(raw.get("rewrite_thread"), dict)
                    else None
                ),
                metadata=dict(raw.get("metadata") or {}),
            )

        # Ergonomic algebra forms (the §3 constructors).
        if kind in (None, "Continue", "Allow") and not any(
            raw.get(k) for k in ("text", "block", "reason", "args", "result")
        ):
            if kind == "Allow":
                return Allow()
            ctx = raw.get("additional_context") or raw.get("text")
            return Continue(ctx) if ctx else None
        if kind == "Allow":
            return Allow(updated_args=raw.get("args") or raw.get("updated_args"))
        if kind == "Deny":
            return Deny(raw.get("block") or raw.get("reason") or "permission denied")
        if kind == "Block":
            return Block(raw.get("reason") or raw.get("block") or "blocked")
        if kind == "Stop":
            return Stop(raw.get("reason") or raw.get("stop") or "hook_requested_stop")
        if kind == "InjectContext":
            return InjectContext(raw.get("text") or raw.get("additional_context") or "")
        if kind == "UpdateArgs":
            return HookDecision(updated_args=raw.get("args") or raw.get("updated_args"))
        if kind == "UpdateResult":
            return HookDecision(
                updated_result=_toolresult_from_wire(raw.get("result") or raw.get("updated_result"))
            )
        if kind == "RewriteThread":
            spec = raw.get("rewrite_thread") or raw.get("spec")
            return HookDecision(rewrite_thread=dict(spec) if isinstance(spec, dict) else {})
        raise ValueError(f"unrecognised effect kind {kind!r}")


# ── ToolResult wire helpers ─────────────────────────────────────


def _toolresult_to_wire(result: "ToolResult | None") -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "tool": result.tool,
        "args_summary": result.args_summary,
        "data": result.data,
        "error": result.error,
        "duration_ms": result.duration_ms,
        "result_key": result.result_key,
        "call_id": result.call_id,
        "warnings": list(result.warnings),
        "metadata": dict(result.metadata),
    }


def _toolresult_from_wire(raw: Any) -> "ToolResult | None":
    if raw is None:
        return None
    if isinstance(raw, ToolResult):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"updated_result must be a dict, got {type(raw).__name__}")
    return ToolResult(
        tool=raw.get("tool", ""),
        args_summary=raw.get("args_summary", ""),
        data=raw.get("data"),
        error=raw.get("error"),
        duration_ms=raw.get("duration_ms", 0.0),
        result_key=raw.get("result_key"),
        call_id=raw.get("call_id"),
        warnings=list(raw.get("warnings") or []),
        metadata=dict(raw.get("metadata") or {}),
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


def RewriteThread(
    *,
    reset_metadata_keys: list[str] | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> HookDecision:
    """Declaratively rewrite run state after compaction.

    This is the portable, JSON-safe replacement for the imperative
    ``CompactOutcome.cleanup`` closure: instead of a Python callback
    (which an out-of-process / cross-runtime compactor cannot ship),
    a hook or compactor declares *which* metadata keys to clear and
    *what* to set. The host applies the spec via
    :func:`looplet.compact.apply_thread_rewrite`.
    """
    spec: dict[str, Any] = {}
    if reset_metadata_keys:
        spec["reset_metadata_keys"] = list(reset_metadata_keys)
    if metadata_updates:
        spec["metadata_updates"] = dict(metadata_updates)
    return HookDecision(rewrite_thread=spec)


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
