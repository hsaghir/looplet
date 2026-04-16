"""Tests for cadence.flags — _CadenceFlags and FLAGS singleton."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


class TestFlagDefaults:
    def test_singleton_exists(self):
        from openharness.flags import FLAGS, _CadenceFlags
        assert isinstance(FLAGS, _CadenceFlags)

    def test_concurrent_dispatch_default_false(self):
        from openharness.flags import FLAGS
        assert FLAGS.concurrent_dispatch is False

    def test_sub_agents_default_false(self):
        from openharness.flags import FLAGS
        assert FLAGS.sub_agents is False

    def test_sub_agent_max_steps_default_5(self):
        from openharness.flags import FLAGS
        assert FLAGS.sub_agent_max_steps == 5

    def test_sub_agent_max_spawns_default_2(self):
        from openharness.flags import FLAGS
        assert FLAGS.sub_agent_max_spawns == 2

    def test_context_management_default_true(self):
        from openharness.flags import FLAGS
        assert FLAGS.context_management is True

    def test_reactive_recovery_default_true(self):
        from openharness.flags import FLAGS
        assert FLAGS.reactive_recovery is True

    def test_native_tools_default_false(self):
        from openharness.flags import FLAGS
        assert FLAGS.native_tools is False

    def test_result_budgeting_default_true(self):
        from openharness.flags import FLAGS
        assert FLAGS.result_budgeting is True


class TestFlagEnvOverrides:
    def test_concurrent_dispatch_env_true(self, monkeypatch):
        monkeypatch.setenv("CADENCE_CONCURRENT_DISPATCH", "1")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().concurrent_dispatch is True

    def test_concurrent_dispatch_env_false(self, monkeypatch):
        monkeypatch.setenv("CADENCE_CONCURRENT_DISPATCH", "false")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().concurrent_dispatch is False

    def test_sub_agents_env_on(self, monkeypatch):
        monkeypatch.setenv("CADENCE_SUB_AGENTS", "on")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().sub_agents is True

    def test_context_management_env_off(self, monkeypatch):
        monkeypatch.setenv("CADENCE_CONTEXT_MANAGEMENT", "0")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().context_management is False

    def test_reactive_recovery_env_no(self, monkeypatch):
        monkeypatch.setenv("CADENCE_REACTIVE_RECOVERY", "no")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().reactive_recovery is False

    def test_native_tools_env_yes(self, monkeypatch):
        monkeypatch.setenv("CADENCE_NATIVE_TOOLS", "yes")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().native_tools is True

    def test_result_budgeting_env_off(self, monkeypatch):
        monkeypatch.setenv("CADENCE_RESULT_BUDGETING", "off")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().result_budgeting is False


class TestIntFlags:
    def test_sub_agent_max_steps_env(self, monkeypatch):
        monkeypatch.setenv("CADENCE_SUB_AGENT_MAX_STEPS", "10")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().sub_agent_max_steps == 10

    def test_sub_agent_max_spawns_env(self, monkeypatch):
        monkeypatch.setenv("CADENCE_SUB_AGENT_MAX_SPAWNS", "5")
        from openharness.flags import _CadenceFlags
        assert _CadenceFlags().sub_agent_max_spawns == 5

    def test_sub_agent_max_steps_returns_int(self):
        from openharness.flags import FLAGS
        assert isinstance(FLAGS.sub_agent_max_steps, int)

    def test_sub_agent_max_spawns_returns_int(self):
        from openharness.flags import FLAGS
        assert isinstance(FLAGS.sub_agent_max_spawns, int)


class TestExports:
    def test_flags_exported_from_cadence(self):
        import openharness as cadence
        assert hasattr(cadence, "FLAGS")

    def test_no_harness_prefix_in_flags(self):
        """Verify we use CADENCE_ prefix for env vars, not HARNESS_."""
        import openharness.flags as fm
        src = open(fm.__file__).read()
        # Only check for HARNESS_ as an env var prefix (e.g. _flag("HARNESS_..."))
        import re
        harness_env_vars = re.findall(r'_flag\(["\']HARNESS_', src)
        assert not harness_env_vars, f"HARNESS_ env var prefix found: {harness_env_vars}"
        assert "primal_security" not in src, "primal_security import found"
