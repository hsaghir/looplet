"""Declarative permission engine for tool calls.

Register a :class:`PermissionEngine` via ``hooks=[PermissionHook(engine)]``
to get:

* Four canonical decisions - ``allow``, ``deny``, ``ask``, ``default``
* Rule-based matching on ``(tool_name, arg_matcher)``
* Automatic audit trail of every denial, surfaced as a
  :class:`looplet.types.ToolError` with
  ``kind=ErrorKind.PERMISSION_DENIED``
* A single extension point - plug in a callable ``ask_handler`` to
  wire up human-in-the-loop prompts without touching the engine

This is the minimum needed to match modern agent permission semantics
while staying domain-agnostic.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from looplet.types import PolicyDecision, ToolCall

if TYPE_CHECKING:
    from looplet.types import AgentState

logger = logging.getLogger(__name__)


class PermissionDecision(str, Enum):
    """Four-way decision produced by a rule or the engine as a whole."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    DEFAULT = "default"  # no rule matched - caller decides fallback


ArgMatcher = Callable[[dict[str, Any]], bool]
"""A predicate that inspects the tool's args dict and returns True if
the rule should match. ``None`` rules match regardless of args."""


@dataclass
class PermissionRule:
    """A single rule in the engine's evaluation list.

    Rules are checked in order; the first matching rule wins. A rule
    matches when the tool name equals ``tool`` (``"*"`` matches any)
    and - if provided - ``arg_matcher(args)`` is truthy.
    """

    tool: str
    decision: PermissionDecision
    arg_matcher: ArgMatcher | None = None
    reason: str = ""

    def matches(self, call: ToolCall) -> bool:
        matched, _ = self._match(call)
        return matched

    def _match(self, call: ToolCall) -> tuple[bool, bool]:
        if self.tool != "*" and self.tool != call.tool:
            return False, False
        if self.arg_matcher is None:
            return True, False
        try:
            return bool(self.arg_matcher(call.args)), False
        except Exception as exc:
            # A buggy matcher must fail closed, which means different things
            # depending on the rule's decision:
            #   DENY  → act as if it matched (block the call)
            #   ALLOW → act as if it did NOT match (don't grant access)
            #   ASK   → deny without calling the approval handler
            #   DEFAULT → act as if it did NOT match
            fail_closed_match = self.decision in (PermissionDecision.DENY, PermissionDecision.ASK)
            logger.warning(
                "PermissionRule arg_matcher for '%s' (decision=%s) raised %s - "
                "failing closed (matches=%s)",
                self.tool,
                self.decision.value,
                exc,
                fail_closed_match,
            )
            return fail_closed_match, True


@dataclass
class PermissionOutcome:
    """Result of evaluating a tool call against the engine."""

    decision: PermissionDecision
    rule: PermissionRule | None = None
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == PermissionDecision.DENY


@dataclass
class PermissionEngine:
    """Evaluate tool calls against an ordered list of rules.

    ``default`` controls what happens when no rule matches. ``ask_handler``
    is an optional callable that turns an ``ASK`` outcome into a concrete
    ``ALLOW`` or ``DENY`` - typically by prompting a human or another
    agent. An explicit ``ASK`` without a handler denies the call; the
    ``default`` applies only when no rule matches.

    The engine keeps an append-only ``denials`` log for auditability;
    each entry captures the tool name, args, and the rule (if any)
    responsible for the deny.
    """

    rules: list[PermissionRule] = field(default_factory=list)
    default: PermissionDecision = PermissionDecision.ALLOW
    ask_handler: Callable[[ToolCall, PermissionRule], PermissionDecision] | None = None
    denials: list[dict[str, Any]] = field(default_factory=list)
    policy_version: str | int | None = None

    def _policy_snapshot(self) -> tuple[Any, ...]:
        return (
            self.policy_version,
            self.default,
            self.ask_handler,
            tuple(
                (id(rule), rule.tool, rule.decision, rule.arg_matcher, rule.reason)
                for rule in self.rules
            ),
        )

    def allow(
        self, tool: str, *, arg_matcher: ArgMatcher | None = None, reason: str = ""
    ) -> "PermissionEngine":
        self.rules.append(
            PermissionRule(
                tool=tool,
                decision=PermissionDecision.ALLOW,
                arg_matcher=arg_matcher,
                reason=reason,
            )
        )
        return self

    def deny(
        self, tool: str, *, arg_matcher: ArgMatcher | None = None, reason: str = ""
    ) -> "PermissionEngine":
        self.rules.append(
            PermissionRule(
                tool=tool,
                decision=PermissionDecision.DENY,
                arg_matcher=arg_matcher,
                reason=reason,
            )
        )
        return self

    def ask(
        self, tool: str, *, arg_matcher: ArgMatcher | None = None, reason: str = ""
    ) -> "PermissionEngine":
        self.rules.append(
            PermissionRule(
                tool=tool,
                decision=PermissionDecision.ASK,
                arg_matcher=arg_matcher,
                reason=reason,
            )
        )
        return self

    def evaluate(self, call: ToolCall) -> PermissionOutcome:
        """Run the call through all rules; first match wins.

        When a rule's decision is ``ASK``:
        - If an ``ask_handler`` is set, it is called and must return
          ``ALLOW`` or ``DENY``. Any other value (including ``ASK`` or
          ``DEFAULT``) is treated as ``DENY`` to fail closed.
        - Without a handler, the call is denied, regardless of ``default``.
        """
        for rule in self.rules:
            matched, matcher_failed = rule._match(call)
            if matched:
                decision = rule.decision
                if matcher_failed and decision == PermissionDecision.ASK:
                    decision = PermissionDecision.DENY
                elif decision == PermissionDecision.ASK:
                    if self.ask_handler is not None:
                        try:
                            original = (call.call_id, call.tool, deepcopy(call.args))
                            policy = self._policy_snapshot()
                            decision = self.ask_handler(call, rule)
                            if decision == PermissionDecision.ALLOW and (
                                (call.call_id, call.tool, call.args) != original
                                or self._policy_snapshot() != policy
                            ):
                                logger.warning(
                                    "Approval changed the action or policy for '%s'", call.tool
                                )
                                decision = PermissionDecision.DENY
                        except Exception:
                            logger.exception("Approval failed for '%s'", call.tool)
                            decision = PermissionDecision.DENY
                        # Guard: handler must return ALLOW or DENY.
                        if decision not in (PermissionDecision.ALLOW, PermissionDecision.DENY):
                            logger.warning(
                                "ask_handler returned %r for tool '%s' - "
                                "treating as DENY (must return ALLOW or DENY)",
                                decision,
                                call.tool,
                            )
                            decision = PermissionDecision.DENY
                    else:
                        decision = PermissionDecision.DENY
                outcome = PermissionOutcome(
                    decision=decision,
                    rule=rule,
                    reason=rule.reason,
                )
                if outcome.denied:
                    self._record_denial(call, rule, rule.reason)
                return outcome

        outcome = PermissionOutcome(decision=self._resolve_default(call), reason="no rule matched")
        if outcome.denied:
            self._record_denial(call, None, outcome.reason)
        return outcome

    def _resolve_default(self, call: ToolCall) -> PermissionDecision:
        """Collapse ``self.default`` to a concrete ALLOW/DENY.

        ``DEFAULT`` or ``ASK`` at the engine-default level are ambiguous
        outcomes that would otherwise leak into :class:`PermissionOutcome`
        and be silently treated as not-allowed-and-not-denied (effectively
        fail-open in some callers). Collapse them to ``DENY`` so the
        engine always produces a decisive outcome.
        """
        if self.default in (PermissionDecision.ALLOW, PermissionDecision.DENY):
            return self.default
        logger.warning(
            "PermissionEngine.default=%r is ambiguous for '%s' - "
            "collapsing to DENY (configure default=ALLOW or DENY to silence)",
            self.default,
            call.tool,
        )
        return PermissionDecision.DENY

    def _record_denial(self, call: ToolCall, rule: PermissionRule | None, reason: str) -> None:
        # Strip internal ``__…__`` scaffolding keys (e.g. ``__theory__``)
        # that parse.py stamps onto tool args - those are agent-internal
        # metadata, not user-visible args, and leaking them into an
        # audit log is noisy + potentially sensitive.
        clean_args = {k: v for k, v in call.args.items() if not k.startswith("__")}
        self.denials.append(
            {
                "tool": call.tool,
                "args": clean_args,
                "rule": rule.tool if rule else None,
                "reason": reason,
            }
        )


# ── Permission hook helper ─────────────────────────────────────


class PermissionHook:
    """Adapt a :class:`PermissionEngine` to the :class:`LoopHook` surface.

    This is the principled way to install declarative permission rules
    into the new ``HookDecision`` world: one hook class, registered in
    ``hooks=[...]`` alongside everything else, producing the same
    :class:`HookDecision` shape all other hook slots use.

    Example::

        engine = PermissionEngine(default=PermissionDecision.ALLOW)
        engine.deny("bash", arg_matcher=lambda a: "rm -rf" in a.get("cmd", ""))

        for step in composable_loop(
            llm=..., tools=..., state=...,
            hooks=[PermissionHook(engine)],
        ):
            ...
    """

    def __init__(self, engine: "PermissionEngine") -> None:
        self.engine = engine
        self._approved: tuple[int, str, str, dict[str, Any], tuple[Any, ...]] | None = None

    def to_config(self) -> dict:
        """Cartridge round-trip: emit ``engine`` as an ``@ref`` so the
        workspace writer auto-generates ``resources/<name>.py``.

        When ``self.engine`` was produced by a workspace resource
        builder (its class lives in ``_chw_resource_<name>``), the
        original ref name is preserved - e.g. a workspace declaring
        ``resources/sql_permissions.py`` round-trips as
        ``{"engine": "@sql_permissions"}``. Otherwise the writer
        emits the generic ``"@engine"`` name.
        """
        from looplet.refs import resource_ref_for  # noqa: PLC0415

        ref = resource_ref_for(self.engine)
        return {"engine": ref or "@engine"}

    def on_event(self, payload: Any) -> Any:
        """Fire on ``PRE_TOOL_USE`` and convert engine outcomes to decisions."""
        # Lazy import - avoids a hard cycle with looplet.events.
        from looplet.events import LifecycleEvent  # noqa: PLC0415
        from looplet.hook_decision import Allow, Deny  # noqa: PLC0415

        if payload.event != LifecycleEvent.PRE_TOOL_USE:
            return None
        if payload.tool_call is None:
            return None
        self._approved = None
        outcome = self.engine.evaluate(payload.tool_call)
        audit = PolicyDecision(
            decision=outcome.decision.value,
            tool=payload.tool_call.tool,
            policy_id=outcome.rule.tool if outcome.rule is not None else "default",
            reason=outcome.reason,
        )
        if outcome.allowed:
            if outcome.rule is not None and outcome.rule.decision == PermissionDecision.ASK:
                self._approved = (
                    id(payload.tool_call),
                    payload.tool_call.call_id,
                    payload.tool_call.tool,
                    deepcopy(payload.tool_call.args),
                    self.engine._policy_snapshot(),
                )
            return Allow(policy_decision=audit)
        if outcome.denied:
            return Deny(
                outcome.reason or f"permission denied for '{payload.tool_call.tool}'",
                policy_decision=audit,
            )
        # ALLOW by default for ASK/DEFAULT outcomes that slipped through.
        return None

    # Back-compat: many older hooks use check_permission. Expose it so
    # this hook behaves correctly even if someone calls the per-method
    # surface directly (e.g. in tests).
    def check_permission(self, tool_call: ToolCall, state: AgentState) -> bool:
        approved = self._approved
        self._approved = None
        if approved is not None:
            if approved == (
                id(tool_call),
                tool_call.call_id,
                tool_call.tool,
                tool_call.args,
                self.engine._policy_snapshot(),
            ):
                return True
            self.engine._record_denial(tool_call, None, "Approved action or policy changed")
            return False
        outcome = self.engine.evaluate(tool_call)
        return not outcome.denied
