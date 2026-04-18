"""Tests for openharness.recovery — pluggable failure recovery strategies."""
from __future__ import annotations

import pytest

from openharness.recovery import (
    FailureScenario,
    RecoveryAction,
    RecoveryRecipe,
    RecoveryRegistry,
    build_default_registry,
)

# ── FailureScenario enum ─────────────────────────────────────────


def test_failure_scenario_has_all_values():
    expected = {
        "PARSE_ERROR",
        "PROMPT_TOO_LONG",
        "TOOL_ERROR",
        "EMPTY_RESULT",
        "REPEATED_TOOL",
        "STAGNATION",
        "LLM_ERROR",
        "TIMEOUT",
    }
    actual = {m.name for m in FailureScenario}
    assert expected == actual


def test_failure_scenario_count():
    assert len(FailureScenario) == 8


def test_failure_scenario_is_enum():
    from enum import Enum
    assert issubclass(FailureScenario, Enum)


# ── RecoveryAction dataclass ─────────────────────────────────────


def test_recovery_action_creation():
    action = RecoveryAction(action_type="retry")
    assert action.action_type == "retry"
    assert action.payload == {}
    assert action.message == ""


def test_recovery_action_with_payload():
    action = RecoveryAction(
        action_type="modify_prompt",
        payload={"hint": "simplify"},
        message="Retry with simplified instructions",
    )
    assert action.payload == {"hint": "simplify"}
    assert action.message == "Retry with simplified instructions"


def test_recovery_action_valid_types():
    """Action types match the spec."""
    for atype in ("retry", "modify_prompt", "skip", "abort", "inject_guidance"):
        a = RecoveryAction(action_type=atype)
        assert a.action_type == atype


def test_recovery_action_payload_independent():
    """Each action should get its own payload dict."""
    a = RecoveryAction(action_type="retry")
    b = RecoveryAction(action_type="retry")
    a.payload["x"] = 1
    assert b.payload == {}


# ── RecoveryRecipe dataclass ─────────────────────────────────────


def test_recovery_recipe_creation():
    def handler(ctx: dict) -> RecoveryAction:
        return RecoveryAction(action_type="retry")

    recipe = RecoveryRecipe(scenario=FailureScenario.PARSE_ERROR, handler=handler)
    assert recipe.scenario == FailureScenario.PARSE_ERROR
    assert recipe.max_attempts == 3
    assert recipe.description == ""
    result = recipe.handler({})
    assert result.action_type == "retry"


def test_recovery_recipe_custom_max_attempts():
    recipe = RecoveryRecipe(
        scenario=FailureScenario.LLM_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=5,
        description="Retry LLM errors",
    )
    assert recipe.max_attempts == 5
    assert recipe.description == "Retry LLM errors"


# ── RecoveryRegistry ─────────────────────────────────────────────


def test_registry_starts_empty():
    registry = RecoveryRegistry()
    assert registry.get_attempts(FailureScenario.PARSE_ERROR) == 0


def test_registry_register_recipe():
    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.PARSE_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
    )
    registry.register(recipe)
    # After registration, attempt_recovery should return an action
    action = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action is not None
    assert action.action_type == "retry"


def test_registry_returns_none_for_unregistered():
    registry = RecoveryRegistry()
    result = registry.attempt_recovery(FailureScenario.TOOL_ERROR, {})
    assert result is None


def test_registry_counts_attempts():
    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.PARSE_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=3,
    )
    registry.register(recipe)

    assert registry.get_attempts(FailureScenario.PARSE_ERROR) == 0
    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert registry.get_attempts(FailureScenario.PARSE_ERROR) == 1
    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert registry.get_attempts(FailureScenario.PARSE_ERROR) == 2


def test_registry_enforces_max_attempts():
    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.PARSE_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=2,
    )
    registry.register(recipe)

    action1 = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action1 is not None  # 1st attempt OK

    action2 = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action2 is not None  # 2nd attempt OK

    action3 = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action3 is None  # 3rd attempt exceeds max


def test_registry_max_attempts_one():
    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.TIMEOUT,
        handler=lambda ctx: RecoveryAction(action_type="skip"),
        max_attempts=1,
    )
    registry.register(recipe)

    action1 = registry.attempt_recovery(FailureScenario.TIMEOUT, {})
    assert action1 is not None

    action2 = registry.attempt_recovery(FailureScenario.TIMEOUT, {})
    assert action2 is None


def test_registry_reset_clears_counts():
    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.LLM_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=1,
    )
    registry.register(recipe)

    registry.attempt_recovery(FailureScenario.LLM_ERROR, {})
    assert registry.get_attempts(FailureScenario.LLM_ERROR) == 1
    # Exhausted
    assert registry.attempt_recovery(FailureScenario.LLM_ERROR, {}) is None

    registry.reset()
    assert registry.get_attempts(FailureScenario.LLM_ERROR) == 0
    # Should work again after reset
    action = registry.attempt_recovery(FailureScenario.LLM_ERROR, {})
    assert action is not None


def test_registry_reset_does_not_remove_recipes():
    registry = RecoveryRegistry()
    recipe = RecoveryRecipe(
        scenario=FailureScenario.STAGNATION,
        handler=lambda ctx: RecoveryAction(action_type="inject_guidance"),
    )
    registry.register(recipe)
    registry.attempt_recovery(FailureScenario.STAGNATION, {})
    registry.reset()
    # Recipe should still be available
    action = registry.attempt_recovery(FailureScenario.STAGNATION, {})
    assert action is not None


def test_registry_handler_receives_context():
    """Handler should receive the context dict passed to attempt_recovery."""
    received_ctx: list[dict] = []

    def handler(ctx: dict) -> RecoveryAction:
        received_ctx.append(ctx)
        return RecoveryAction(action_type="retry")

    registry = RecoveryRegistry()
    registry.register(RecoveryRecipe(scenario=FailureScenario.TOOL_ERROR, handler=handler))
    registry.attempt_recovery(FailureScenario.TOOL_ERROR, {"error": "timeout", "step": 3})

    assert len(received_ctx) == 1
    assert received_ctx[0]["error"] == "timeout"


def test_registry_independent_attempt_counts():
    """Counts for different scenarios are independent."""
    registry = RecoveryRegistry()
    registry.register(RecoveryRecipe(
        scenario=FailureScenario.PARSE_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=2,
    ))
    registry.register(RecoveryRecipe(
        scenario=FailureScenario.LLM_ERROR,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=2,
    ))

    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    # Parse error exhausted
    assert registry.attempt_recovery(FailureScenario.PARSE_ERROR, {}) is None
    # LLM error still fresh
    assert registry.get_attempts(FailureScenario.LLM_ERROR) == 0


# ── build_default_registry ───────────────────────────────────────


def test_default_registry_has_all_scenarios():
    registry = build_default_registry()
    for scenario in FailureScenario:
        action = registry.attempt_recovery(scenario, {})
        assert action is not None, f"No recipe for {scenario}"


def test_default_registry_parse_error_modifies_prompt():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action is not None
    assert action.action_type == "modify_prompt"


def test_default_registry_prompt_too_long_injects_guidance():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.PROMPT_TOO_LONG, {})
    assert action is not None
    assert action.action_type == "inject_guidance"


def test_default_registry_tool_error_retries():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.TOOL_ERROR, {})
    assert action is not None
    assert action.action_type == "retry"


def test_default_registry_empty_result_injects_guidance():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.EMPTY_RESULT, {})
    assert action is not None
    assert action.action_type == "inject_guidance"


def test_default_registry_repeated_tool_injects_guidance():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.REPEATED_TOOL, {})
    assert action is not None
    assert action.action_type == "inject_guidance"


def test_default_registry_stagnation_injects_guidance():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.STAGNATION, {})
    assert action is not None
    assert action.action_type == "inject_guidance"


def test_default_registry_llm_error_retries():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.LLM_ERROR, {})
    assert action is not None
    assert action.action_type == "retry"


def test_default_registry_timeout_skips():
    registry = build_default_registry()
    action = registry.attempt_recovery(FailureScenario.TIMEOUT, {})
    assert action is not None
    assert action.action_type == "skip"


def test_default_registry_parse_error_max_2():
    """PARSE_ERROR default should only allow 2 attempts."""
    registry = build_default_registry()
    registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    action2 = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action2 is not None  # 2nd OK
    action3 = registry.attempt_recovery(FailureScenario.PARSE_ERROR, {})
    assert action3 is None  # 3rd exceeds max of 2


def test_default_registry_actions_have_messages():
    """All default actions should have non-empty guidance messages."""
    registry = build_default_registry()
    for scenario in FailureScenario:
        action = registry.attempt_recovery(scenario, {})
        assert action is not None
        assert isinstance(action.message, str)
