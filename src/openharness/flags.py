"""Feature flags for openharness pipeline capabilities.

All flags read from environment variables with sensible defaults.
Non-essential or expensive features are off by default.

Usage:
    from openharness.flags import FLAGS

    if FLAGS.concurrent_dispatch:
        results = tools.dispatch_batch(calls)
    else:
        results = [tools.dispatch(c) for c in calls]
"""

from __future__ import annotations

import os


def _flag(name: str, default: bool, *, legacy_name: str | None = None) -> bool:
    """Read a boolean flag from environment.

    Checks ``name`` first (OPENHARNESS_*); falls back to ``legacy_name``
    (CADENCE_*) for one release cycle of backward compatibility.
    """
    for env_name in (name, legacy_name):
        if env_name is None:
            continue
        val = os.environ.get(env_name, "").lower()
        if val in ("1", "true", "yes", "on"):
            return True
        if val in ("0", "false", "no", "off"):
            return False
    return default


def _int_flag(name: str, default: int, *, legacy_name: str | None = None) -> int:
    """Read an integer flag from environment with legacy fallback."""
    for env_name in (name, legacy_name):
        if env_name is None:
            continue
        val = os.environ.get(env_name)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return default


class _Flags:
    """openharness feature flags — read from environment at access time.

    Environment variables use the ``OPENHARNESS_`` prefix.  The legacy
    ``CADENCE_`` prefix is accepted as a fallback for one release cycle.
    """

    @property
    def concurrent_dispatch(self) -> bool:
        """Run concurrent-safe tools in parallel via ThreadPoolExecutor.
        Default OFF — some backends are not thread-safe."""
        return _flag("OPENHARNESS_CONCURRENT_DISPATCH", False,
                      legacy_name="CADENCE_CONCURRENT_DISPATCH")

    @property
    def sub_agents(self) -> bool:
        """Enable sub-agent spawning for focused sub-tasks.
        Default OFF — adds LLM calls and execution time."""
        return _flag("OPENHARNESS_SUB_AGENTS", False,
                      legacy_name="CADENCE_SUB_AGENTS")

    @property
    def sub_agent_max_steps(self) -> int:
        """Max tool calls per sub-agent. Default 5."""
        return _int_flag("OPENHARNESS_SUB_AGENT_MAX_STEPS", 5,
                          legacy_name="CADENCE_SUB_AGENT_MAX_STEPS")

    @property
    def sub_agent_max_spawns(self) -> int:
        """Max sub-agent spawns per parent loop. Default 2."""
        return _int_flag("OPENHARNESS_SUB_AGENT_MAX_SPAWNS", 2,
                          legacy_name="CADENCE_SUB_AGENT_MAX_SPAWNS")

    @property
    def context_management(self) -> bool:
        """Progressive result aging + budget enforcement + proactive compaction.
        Default ON — prevents context degradation on longer runs."""
        return _flag("OPENHARNESS_CONTEXT_MANAGEMENT", True,
                      legacy_name="CADENCE_CONTEXT_MANAGEMENT")

    @property
    def reactive_recovery(self) -> bool:
        """Multi-strategy recovery on prompt-too-long errors.
        Default ON — essential for reliability."""
        return _flag("OPENHARNESS_REACTIVE_RECOVERY", True,
                      legacy_name="CADENCE_REACTIVE_RECOVERY")

    @property
    def native_tools(self) -> bool:
        """Use API native tool_use protocol instead of JSON text parsing.
        Default OFF — requires LLM backend support."""
        return _flag("OPENHARNESS_NATIVE_TOOLS", False,
                      legacy_name="CADENCE_NATIVE_TOOLS")

    @property
    def result_budgeting(self) -> bool:
        """Per-result and aggregate context budget enforcement.
        Default ON (part of context_management when enabled)."""
        return _flag("OPENHARNESS_RESULT_BUDGETING", True,
                      legacy_name="CADENCE_RESULT_BUDGETING")


FLAGS = _Flags()

# Backward-compat aliases
HARNESS_FLAGS = FLAGS
