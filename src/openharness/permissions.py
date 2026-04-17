"""Declarative permission engine for tool calls.

``check_permission`` hooks remain as a low-level escape hatch, but most
pipelines should attach a :class:`PermissionEngine` to
:class:`LoopConfig.permissions` to get:

* Four canonical decisions — ``allow``, ``deny``, ``ask``, ``default``
* Rule-based matching on ``(tool_name, arg_matcher)``
* Automatic audit trail of every denial, surfaced as a
  :class:`openharness.types.ToolError` with
  ``kind=ErrorKind.PERMISSION_DENIED``
* A single extension point — plug in a callable ``ask_handler`` to
  wire up human-in-the-loop prompts without touching the engine

This is the minimum needed to match claude-code's permission semantics
while staying domain-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from openharness.types import ToolCall

logger = logging.getLogger(__name__)


class PermissionDecision(str, Enum):
    """Four-way decision produced by a rule or the engine as a whole."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    DEFAULT = "default"  # no rule matched — caller decides fallback


ArgMatcher = Callable[[dict[str, Any]], bool]
"""A predicate that inspects the tool's args dict and returns True if
the rule should match. ``None`` rules match regardless of args."""


@dataclass
class PermissionRule:
    """A single rule in the engine's evaluation list.

    Rules are checked in order; the first matching rule wins. A rule
    matches when the tool name equals ``tool`` (``"*"`` matches any)
    and — if provided — ``arg_matcher(args)`` is truthy.
    """

    tool: str
    decision: PermissionDecision
    arg_matcher: ArgMatcher | None = None
    reason: str = ""

    def matches(self, call: ToolCall) -> bool:
        if self.tool != "*" and self.tool != call.tool:
            return False
        if self.arg_matcher is None:
            return True
        try:
            return bool(self.arg_matcher(call.args))
        except Exception as exc:
            # A buggy matcher must fail closed, which means different things
            # depending on the rule's decision:
            #   DENY  → act as if it matched (block the call)
            #   ALLOW → act as if it did NOT match (don't grant access)
            #   ASK   → act as if it did NOT match (don't escalate to human)
            #   DEFAULT → act as if it did NOT match
            fail_closed_match = self.decision == PermissionDecision.DENY
            logger.warning(
                "PermissionRule arg_matcher for '%s' (decision=%s) raised %s — "
                "failing closed (matches=%s)",
                self.tool, self.decision.value, exc, fail_closed_match,
            )
            return fail_closed_match


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
    ``ALLOW`` or ``DENY`` — typically by prompting a human or another
    agent. Without a handler, ``ASK`` falls back to ``default`` so the
    engine never blocks indefinitely.

    The engine keeps an append-only ``denials`` log for auditability;
    each entry captures the tool name, args, and the rule (if any)
    responsible for the deny.
    """

    rules: list[PermissionRule] = field(default_factory=list)
    default: PermissionDecision = PermissionDecision.ALLOW
    ask_handler: Callable[[ToolCall, PermissionRule], PermissionDecision] | None = None
    denials: list[dict[str, Any]] = field(default_factory=list)

    def allow(self, tool: str, *, arg_matcher: ArgMatcher | None = None,
              reason: str = "") -> "PermissionEngine":
        self.rules.append(PermissionRule(
            tool=tool, decision=PermissionDecision.ALLOW,
            arg_matcher=arg_matcher, reason=reason,
        ))
        return self

    def deny(self, tool: str, *, arg_matcher: ArgMatcher | None = None,
             reason: str = "") -> "PermissionEngine":
        self.rules.append(PermissionRule(
            tool=tool, decision=PermissionDecision.DENY,
            arg_matcher=arg_matcher, reason=reason,
        ))
        return self

    def ask(self, tool: str, *, arg_matcher: ArgMatcher | None = None,
            reason: str = "") -> "PermissionEngine":
        self.rules.append(PermissionRule(
            tool=tool, decision=PermissionDecision.ASK,
            arg_matcher=arg_matcher, reason=reason,
        ))
        return self

    def evaluate(self, call: ToolCall) -> PermissionOutcome:
        """Run the call through all rules; first match wins.

        When a rule's decision is ``ASK``:
        - If an ``ask_handler`` is set, it is called and must return
          ``ALLOW`` or ``DENY``. Any other value (including ``ASK`` or
          ``DEFAULT``) is treated as ``DENY`` to fail closed.
        - Without a handler, the engine's ``default`` is used.
        """
        for rule in self.rules:
            if rule.matches(call):
                decision = rule.decision
                if decision == PermissionDecision.ASK:
                    if self.ask_handler is not None:
                        decision = self.ask_handler(call, rule)
                        # Guard: handler must return ALLOW or DENY.
                        if decision not in (PermissionDecision.ALLOW, PermissionDecision.DENY):
                            logger.warning(
                                "ask_handler returned %r for tool '%s' — "
                                "treating as DENY (must return ALLOW or DENY)",
                                decision, call.tool,
                            )
                            decision = PermissionDecision.DENY
                    else:
                        decision = self._resolve_default(call)
                outcome = PermissionOutcome(
                    decision=decision, rule=rule, reason=rule.reason,
                )
                if outcome.denied:
                    self._record_denial(call, rule, rule.reason)
                return outcome

        outcome = PermissionOutcome(decision=self._resolve_default(call),
                                    reason="no rule matched")
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
            "PermissionEngine.default=%r is ambiguous for '%s' — "
            "collapsing to DENY (configure default=ALLOW or DENY to silence)",
            self.default, call.tool,
        )
        return PermissionDecision.DENY

    def _record_denial(self, call: ToolCall, rule: PermissionRule | None,
                       reason: str) -> None:
        self.denials.append({
            "tool": call.tool,
            "args": dict(call.args),
            "rule": rule.tool if rule else None,
            "reason": reason,
        })
