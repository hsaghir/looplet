"""Tests for looplet example agents — verify they run without errors."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# ── hello_world ──────────────────────────────────────────────────


def test_hello_world_importable():
    import looplet.examples.hello_world  # noqa: F401


def test_hello_world_has_main():
    import looplet.examples.hello_world as m
    assert hasattr(m, "main") and callable(m.main)


# ── coding_agent ─────────────────────────────────────────────────


def test_coding_agent_importable():
    import looplet.examples.coding_agent  # noqa: F401


def test_coding_agent_has_run_function():
    import looplet.examples.coding_agent as m
    assert hasattr(m, "run_coding_agent") and callable(m.run_coding_agent)


def test_coding_agent_has_build_tools():
    import looplet.examples.coding_agent as m
    assert hasattr(m, "build_tools") and callable(m.build_tools)


def test_coding_agent_has_guardrail_hook():
    import looplet.examples.coding_agent as m
    assert hasattr(m, "CodingGuardrailHook")
