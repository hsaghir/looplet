"""Pluggable recovery strategies for agent loop failure scenarios.

``RecoveryRegistry`` maps each ``FailureScenario`` to a ``RecoveryRecipe``
that decides what to do when a failure occurs.  The registry enforces
``max_attempts`` per scenario and exposes a ``reset()`` method for new
sessions.

Typical usage::

    registry = build_default_registry()

    # In a loop exception handler:
    action = registry.attempt_recovery(
        FailureScenario.PARSE_ERROR,
        {"error": str(exc), "step": step_num},
    )
    if action is None:
        break  # max attempts exceeded - abort
    if action.action_type == "modify_prompt":
        prompt = build_simplified_prompt(...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── FailureScenario ──────────────────────────────────────────────


class FailureScenario(Enum):
    """Enumeration of common agent loop failure modes."""

    PARSE_ERROR = "parse_error"
    PROMPT_TOO_LONG = "prompt_too_long"
    TOOL_ERROR = "tool_error"
    EMPTY_RESULT = "empty_result"
    REPEATED_TOOL = "repeated_tool"
    STAGNATION = "stagnation"
    LLM_ERROR = "llm_error"
    TIMEOUT = "timeout"


# ── RecoveryAction ───────────────────────────────────────────────


@dataclass
class RecoveryAction:
    """The action the loop should take to recover from a failure.

    Args:
        action_type: One of ``'retry'``, ``'modify_prompt'``, ``'skip'``,
            ``'abort'``, ``'inject_guidance'``.
        payload: Optional key-value data for the action handler
            (e.g. ``{"hint": "simplify"}``).
        message: Human-readable explanation / guidance text injected into
            the next prompt when action_type is ``'inject_guidance'`` or
            ``'modify_prompt'``.
    """

    action_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    message: str = ""


# ── RecoveryRecipe ───────────────────────────────────────────────


@dataclass
class RecoveryRecipe:
    """Associates a ``FailureScenario`` with a recovery handler.

    Args:
        scenario: The failure scenario this recipe handles.
        handler: ``Callable[[dict], RecoveryAction]`` - receives a context
            dict with state/error info and returns the recovery action.
        max_attempts: Maximum number of times this recipe fires before the
            registry returns ``None`` (default 3).
        description: Human-readable description of the recovery strategy.
    """

    scenario: FailureScenario
    handler: Callable[[dict[str, Any]], RecoveryAction]
    max_attempts: int = 3
    description: str = ""


# ── RecoveryRegistry ─────────────────────────────────────────────


class RecoveryRegistry:
    """Stores recipes and enforces per-scenario attempt limits.

    Usage::

        registry = RecoveryRegistry()
        registry.register(RecoveryRecipe(
            scenario=FailureScenario.PARSE_ERROR,
            handler=lambda ctx: RecoveryAction("retry"),
            max_attempts=2,
        ))
        action = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    """

    def __init__(self) -> None:
        self._recipes: dict[FailureScenario, RecoveryRecipe] = {}
        self._attempt_counts: dict[FailureScenario, int] = {}

    def register(self, recipe: RecoveryRecipe) -> None:
        """Register a recipe for a failure scenario.

        Warns when a recipe for the same scenario is already registered.
        Silent overwrites when composing registries are a real
        footgun (e.g. ``build_default_registry()`` followed by a
        caller adding their own ``PARSE_ERROR`` recipe would overwrite
        the default without any signal). The overwrite still happens
        so existing behaviour is preserved; only the warning is new.
        """
        if recipe.scenario in self._recipes:
            logger.warning(
                "Recovery recipe for %s is already registered - overwriting. "
                "If this is intentional, silence this warning by clearing the "
                "registry before re-registering.",
                recipe.scenario.name,
            )
        self._recipes[recipe.scenario] = recipe

    def attempt_recovery(
        self,
        scenario: FailureScenario,
        context: dict[str, Any],
    ) -> RecoveryAction | None:
        """Try to recover from a failure scenario.

        Returns:
            A ``RecoveryAction`` if the recipe exists and has not exceeded
            ``max_attempts``; ``None`` otherwise.

        Both "no recipe registered" and "max_attempts exceeded" return
        ``None``, but they are distinguishable in the logs: the
        former emits an ``info`` line (an unregistered scenario is
        often intentional), while the latter emits a ``warning``.
        Callers who need to branch programmatically can check
        :meth:`has_recipe` before calling.
        """
        recipe = self._recipes.get(scenario)
        if recipe is None:
            logger.info(
                "No recovery recipe registered for %s - aborting recovery.",
                scenario.name,
            )
            return None

        current = self._attempt_counts.get(scenario, 0)
        if current >= recipe.max_attempts:
            logger.warning(
                "Recovery for %s exhausted (%d/%d attempts)",
                scenario.name,
                current,
                recipe.max_attempts,
            )
            return None

        self._attempt_counts[scenario] = current + 1
        return recipe.handler(context)

    def has_recipe(self, scenario: FailureScenario) -> bool:
        """Return True iff a recipe is registered for ``scenario``."""
        return scenario in self._recipes

    def reset(self) -> None:
        """Clear all attempt counts - call at the start of a new session."""
        self._attempt_counts.clear()

    def get_attempts(self, scenario: FailureScenario) -> int:
        """Return how many recovery attempts have been made for a scenario."""
        return self._attempt_counts.get(scenario, 0)


# ── Default Registry ─────────────────────────────────────────────


def build_default_registry() -> RecoveryRegistry:
    """Create a ``RecoveryRegistry`` with sensible defaults for all scenarios.

    Default strategies:
    - PARSE_ERROR → modify_prompt with simplified instructions (max 2)
    - PROMPT_TOO_LONG → inject_guidance to compact context
    - TOOL_ERROR → retry with backoff hint
    - EMPTY_RESULT → inject_guidance with alternative approaches
    - REPEATED_TOOL → inject_guidance with dedup warning
    - STAGNATION → inject_guidance to pivot or conclude
    - LLM_ERROR → retry (max 3)
    - TIMEOUT → skip with warning
    """
    registry = RecoveryRegistry()

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.PARSE_ERROR,
            handler=lambda ctx: RecoveryAction(
                action_type="modify_prompt",
                payload={"simplify": True},
                message=(
                    "Your previous response could not be parsed as a tool call. "
                    "Please respond using ONLY the JSON tool-call format with no "
                    "extra text before or after the JSON."
                ),
            ),
            max_attempts=2,
            description="Simplify prompt when LLM output cannot be parsed",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.PROMPT_TOO_LONG,
            handler=lambda ctx: RecoveryAction(
                action_type="inject_guidance",
                payload={"compact": True},
                message=(
                    "The context has grown too large. "
                    "Please summarise what you have found so far and continue "
                    "with a focused next step rather than repeating prior context."
                ),
            ),
            max_attempts=3,
            description="Compact context when prompt exceeds token limit",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.TOOL_ERROR,
            handler=lambda ctx: RecoveryAction(
                action_type="retry",
                payload={"backoff": True},
                message=(
                    "The last tool call returned an error. "
                    "You may retry with different arguments or choose an alternative tool."
                ),
            ),
            max_attempts=3,
            description="Retry tool calls that return errors",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.EMPTY_RESULT,
            handler=lambda ctx: RecoveryAction(
                action_type="inject_guidance",
                payload={"suggest_alternative": True},
                message=(
                    "The last tool returned no results. "
                    "Try a broader query, different keywords, or a different tool "
                    "to find the information you need."
                ),
            ),
            max_attempts=3,
            description="Guide agent to try alternative approaches on empty results",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.REPEATED_TOOL,
            handler=lambda ctx: RecoveryAction(
                action_type="inject_guidance",
                payload={"dedup": True},
                message=(
                    "You have called the same tool with the same arguments more "
                    "than once. Please avoid repeating queries you have already made "
                    "and instead try a different approach or tool."
                ),
            ),
            max_attempts=3,
            description="Warn agent about duplicate tool calls",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.STAGNATION,
            handler=lambda ctx: RecoveryAction(
                action_type="inject_guidance",
                payload={"pivot": True},
                message=(
                    "Progress has stalled - no new information found in recent steps. "
                    "Consider pivoting to a different angle, summarising current "
                    "findings, or calling done() if the task is complete."
                ),
            ),
            max_attempts=3,
            description="Encourage the agent to pivot or conclude when stagnant",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.LLM_ERROR,
            handler=lambda ctx: RecoveryAction(
                action_type="retry",
                payload={},
                message=("The language model returned an error. Retrying the request."),
            ),
            max_attempts=3,
            description="Retry on transient LLM API errors",
        )
    )

    registry.register(
        RecoveryRecipe(
            scenario=FailureScenario.TIMEOUT,
            handler=lambda ctx: RecoveryAction(
                action_type="skip",
                payload={"warn": True},
                message=(
                    "A tool call timed out and has been skipped. "
                    "Continue with the information you have."
                ),
            ),
            max_attempts=3,
            description="Skip timed-out tool calls",
        )
    )

    return registry
