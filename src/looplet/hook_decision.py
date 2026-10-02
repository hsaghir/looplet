"""Unified hook return type - one dataclass, every lifecycle slot.

Every hook method on :class:`looplet.loop.LoopHook` traditionally
had its own return shape: ``str | None`` for briefing injection,
``bool`` for permission, ``ToolResult | None`` for dispatch
intercept, etc. That sprawl makes it painful to add new capabilities
(mutating tool args, structured stop reasons, permission grants)
without breaking everyone.

``HookDecision`` collapses those slots into **one dataclass with
optional fields**. Decision-returning slots accept
``HookDecision | None``; ``None`` means "no opinion, proceed as
default". Effect applicability is explicit per slot. Unsupported fields
remain ignored for compatibility, but normalization reports their names
without logging their values.

The dataclass is intentionally flat - no inheritance, no variants,
one level of optional attributes. That keeps the API surface small
and makes it trivial to inspect a decision in logs.

All existing hook return types (``str``, ``bool``,
``ToolResult``) remain accepted at their call sites in this
release for backward compatibility. New code should return
:class:`HookDecision` for clarity and composability.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, fields
from typing import Any
from weakref import WeakKeyDictionary

from looplet.types import PolicyDecision, ToolResult

__all__ = [
    "HookDecision",
    "Allow",
    "Deny",
    "Block",
    "Stop",
    "Continue",
    "InjectContext",
    "RewriteThread",
    "PolicyDecision",
    "normalize_hook_return",
    "hook_effect_fields",
]

logger = logging.getLogger(__name__)

_PRE_TOOL_FIELDS = frozenset(
    {"updated_args", "updated_result", "permission", "block", "additional_context"}
)
_POST_TOOL_FIELDS = frozenset({"updated_result", "additional_context", "stop"})
_MODEL_FIELDS = frozenset({"additional_context", "stop"})
_HOOK_SLOT_EFFECTS = {
    "pre_prompt": frozenset({"additional_context"}),
    "pre_dispatch": _PRE_TOOL_FIELDS,
    "check_permission": frozenset({"permission", "block"}),
    "post_dispatch": _POST_TOOL_FIELDS,
    "check_done": frozenset({"block", "permission"}),
    "should_stop": frozenset({"stop"}),
    "pre_tool_use": _PRE_TOOL_FIELDS,
    "post_tool_use": _POST_TOOL_FIELDS,
    "post_tool_failure": _POST_TOOL_FIELDS,
    "pre_llm_call": _MODEL_FIELDS,
    "post_llm_response": _MODEL_FIELDS,
    "pre_compact": frozenset({"stop"}),
    "post_compact": frozenset({"rewrite_thread"}),
    "session_start": frozenset(),
    "tool_progress": frozenset(),
    "hook_decision": frozenset(),
    "done_accepted": frozenset(),
    "stop": frozenset(),
    "subagent_start": frozenset(),
    "subagent_stop": frozenset(),
}


def hook_effect_fields(slot: str) -> frozenset[str]:
    """Return applicable effect fields; audit metadata is separate.

    At tool/permission gates, ``block`` supplies the reason only when
    ``permission='deny'``. ``pre_compact.stop`` aborts compaction rather
    than terminating the run. Unknown host-owned slots define their own
    applicability outside this loop contract.
    """
    return _HOOK_SLOT_EFFECTS.get(slot, frozenset())


@dataclass
class HookDecision:
    """The single unified return type for every hook method.

    Fields are evaluated per call site. A hook that runs in a slot the
    field doesn't apply to remains a no-op. :meth:`ignored_effects` and
    normalization diagnostics make that boundary inspectable. Prompt
    builders, compaction votes, and cleanup retain their dedicated return
    shapes; they are not general decision slots.

    Attributes:
        block: Rejects completion in ``check_done``. At tool and permission
            gates, supplies the reason for ``permission='deny'``; use
            ``Deny`` rather than a block-only decision there.
        stop: Stops after the current step from ``post_dispatch``,
            ``should_stop``, or model lifecycle events. It is unsupported
            in ``check_done``; ``pre_compact`` uses it to abort compaction.
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
        permission: ``"deny"`` refuses the call at tool/permission gates.
            ``"allow"`` does not override another hook's denial or grant
            host execution authority.
        additional_context: Plain text appended to the next briefing.
            Applies to prompt, tool, and model slots listed by
            :func:`hook_effect_fields`, not every lifecycle event.
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
    policy_decision: PolicyDecision | None = None
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
            and self.policy_decision is None
            and not self.metadata
        )

    def ignored_effects(self, slot: str) -> tuple[str, ...]:
        """List unapplied fields without changing the decision or its values."""
        applicable = _HOOK_SLOT_EFFECTS.get(slot)
        if applicable is None:
            return ()
        ignored = {
            descriptor.name
            for descriptor in fields(self)
            if descriptor.name not in applicable | {"metadata", "policy_decision"}
            and getattr(self, descriptor.name) is not None
        }
        if (
            self.block is not None
            and slot in {"pre_dispatch", "pre_tool_use", "check_permission"}
            and self.permission != "deny"
        ):
            ignored.add("block")
        if self.permission is not None and self.permission not in {"allow", "deny"}:
            ignored.add("permission")
        return tuple(sorted(ignored))

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
            "policy_decision": (
                self.policy_decision.to_dict() if self.policy_decision is not None else None
            ),
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
                policy_decision=(
                    PolicyDecision(**dict(raw["policy_decision"]))
                    if isinstance(raw.get("policy_decision"), dict)
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
# Each returns a ``HookDecision`` - they're factories, not classes.


def Allow(
    updated_args: dict[str, Any] | None = None,
    *,
    policy_decision: PolicyDecision | None = None,
) -> HookDecision:
    """Grant a tool call, optionally rewriting its arguments."""
    return HookDecision(
        permission="allow", updated_args=updated_args, policy_decision=policy_decision
    )


def Deny(
    reason: str,
    *,
    retry: bool = False,
    policy_decision: PolicyDecision | None = None,
) -> HookDecision:
    """Refuse a tool call. The reason is surfaced to the model.

    ``retry=True`` signals that the model may legitimately try again
    with different args (recorded in ``metadata["retry"]`` for hooks
    and logs to observe).
    """
    return HookDecision(
        permission="deny",
        block=reason,
        policy_decision=policy_decision,
        metadata={"retry": retry} if retry else {},
    )


def Block(reason: str) -> HookDecision:
    """Reject completion in ``check_done``. Use ``Deny`` for tool gates."""
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


_HOOK_CALL_PLANS: WeakKeyDictionary[Any, tuple[frozenset[str], bool]] = WeakKeyDictionary()


def _hook_accepts_keyword(method: Any, name: str) -> bool:
    """Inspect optional keyword support using callable identity, not method IDs."""
    key = getattr(method, "__func__", method)
    try:
        plan = _HOOK_CALL_PLANS.get(key)
    except TypeError:
        plan = None
    if plan is None:
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            plan = (frozenset(), False)
        else:
            plan = (
                frozenset(
                    parameter.name
                    for parameter in parameters
                    if parameter.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ),
                any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters),
            )
        try:
            _HOOK_CALL_PLANS[key] = plan
        except TypeError:
            pass
    return name in plan[0] or plan[1]


def _invoke_hook(
    hook: Any,
    slot: str,
    *args: Any,
    _optional_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Preserve transport decisions and legacy direct bootstrap calls.

    ``pre_loop`` is bootstrap, not another remote SESSION_START dispatch.
    Event transport remains owned by its dedicated lifecycle emission.
    """
    method = getattr(hook, slot)
    if _optional_kwargs:
        kwargs.update(
            {
                name: value
                for name, value in _optional_kwargs.items()
                if _hook_accepts_keyword(method, name)
            }
        )
    invoke = getattr(hook, "_invoke_hook_return", None)
    if slot != "pre_loop" and callable(invoke):
        return invoke(slot, *args, **kwargs)
    return method(*args, **kwargs)


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

    Anything else raises ``TypeError`` - hooks that return garbage
    should fail loud, not silently drop.
    """
    decision: HookDecision | None
    if value is None:
        return None
    if isinstance(value, HookDecision):
        decision = value
    elif isinstance(value, ToolResult):
        decision = HookDecision(updated_result=value)
    elif isinstance(value, bool):
        if slot == "check_permission":
            decision = Allow() if value else Deny("permission denied")
        elif slot == "should_stop":
            decision = Stop("hook_requested_stop") if value else None
        else:
            raise TypeError(
                f"hook slot {slot!r} received bool {value!r}; expected HookDecision | str | None"
            )
    elif isinstance(value, str):
        decision = Block(value) if slot == "check_done" else InjectContext(value)
    else:
        raise TypeError(
            f"hook slot {slot!r} returned {type(value).__name__} "
            f"{value!r}; expected HookDecision | str | bool | ToolResult | None"
        )
    if decision is not None:
        ignored = decision.ignored_effects(slot)
        if ignored:
            logger.warning("hook slot %r ignores effect fields: %s", slot, ", ".join(ignored))
    return decision
