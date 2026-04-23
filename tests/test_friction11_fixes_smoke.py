"""Round-11 friction: RecoveryRegistry.register warns on overwrite + has_recipe()."""

from __future__ import annotations

import logging

import pytest

from looplet.recovery import (
    FailureScenario,
    RecoveryAction,
    RecoveryRecipe,
    RecoveryRegistry,
    build_default_registry,
)

pytestmark = pytest.mark.smoke


def _recipe(scenario: FailureScenario) -> RecoveryRecipe:
    return RecoveryRecipe(
        scenario=scenario,
        handler=lambda ctx: RecoveryAction(action_type="retry"),
        max_attempts=2,
    )


class TestRecoveryRegistryOverwriteWarning:
    def test_duplicate_register_logs_warning(self, caplog):
        reg = RecoveryRegistry()
        reg.register(_recipe(FailureScenario.PARSE_ERROR))
        with caplog.at_level(logging.WARNING, logger="looplet.recovery"):
            reg.register(_recipe(FailureScenario.PARSE_ERROR))
        assert any("already registered" in rec.message for rec in caplog.records)

    def test_default_registry_plus_user_override_warns(self, caplog):
        reg = build_default_registry()
        with caplog.at_level(logging.WARNING, logger="looplet.recovery"):
            reg.register(_recipe(FailureScenario.PARSE_ERROR))
        assert any("PARSE_ERROR" in rec.message for rec in caplog.records)

    def test_first_register_no_warning(self, caplog):
        reg = RecoveryRegistry()
        with caplog.at_level(logging.WARNING, logger="looplet.recovery"):
            reg.register(_recipe(FailureScenario.PARSE_ERROR))
        assert not any("already registered" in rec.message for rec in caplog.records)


class TestHasRecipe:
    def test_has_recipe_true(self):
        reg = RecoveryRegistry()
        reg.register(_recipe(FailureScenario.TOOL_ERROR))
        assert reg.has_recipe(FailureScenario.TOOL_ERROR)

    def test_has_recipe_false(self):
        reg = RecoveryRegistry()
        assert not reg.has_recipe(FailureScenario.TOOL_ERROR)

    def test_has_recipe_distinguishes_no_recipe_from_exhausted(self):
        reg = RecoveryRegistry()
        reg.register(_recipe(FailureScenario.TOOL_ERROR))
        # Exhaust
        for _ in range(2):
            reg.attempt_recovery(FailureScenario.TOOL_ERROR, {})
        # Still registered, just exhausted — previously indistinguishable from
        # "never registered". Now callers can check has_recipe.
        assert reg.attempt_recovery(FailureScenario.TOOL_ERROR, {}) is None
        assert reg.has_recipe(FailureScenario.TOOL_ERROR)
        assert not reg.has_recipe(FailureScenario.LLM_ERROR)
