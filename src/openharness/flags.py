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


def _flag(name: str, default: bool) -> bool:
    """Read a boolean flag from the environment."""
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _int_flag(name: str, default: int) -> int:
    """Read an integer flag from the environment."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    return default


class _Flags:
    """openharness feature flags — read from environment at access time.

    Environment variables use the ``OPENHARNESS_`` prefix.
    """

    @property
    def concurrent_dispatch(self) -> bool:
        """Run concurrent-safe tools in parallel via ThreadPoolExecutor.
        Default OFF — some backends are not thread-safe."""
        return _flag("OPENHARNESS_CONCURRENT_DISPATCH", False)

    @property
    def sub_agents(self) -> bool:
        """Enable sub-agent spawning for focused sub-tasks.
        Default OFF — adds LLM calls and execution time."""
        return _flag("OPENHARNESS_SUB_AGENTS", False)

    @property
    def sub_agent_max_steps(self) -> int:
        """Max tool calls per sub-agent. Default 5."""
        return _int_flag("OPENHARNESS_SUB_AGENT_MAX_STEPS", 5)

    @property
    def sub_agent_max_spawns(self) -> int:
        """Max sub-agent spawns per parent loop. Default 2."""
        return _int_flag("OPENHARNESS_SUB_AGENT_MAX_SPAWNS", 2)

    @property
    def context_management(self) -> bool:
        """Progressive result aging + budget enforcement + proactive compaction.
        Default ON — prevents context degradation on longer runs."""
        return _flag("OPENHARNESS_CONTEXT_MANAGEMENT", True)

    @property
    def reactive_recovery(self) -> bool:
        """Multi-strategy recovery on prompt-too-long errors.
        Default ON — essential for reliability."""
        return _flag("OPENHARNESS_REACTIVE_RECOVERY", True)

    @property
    def native_tools(self) -> bool:
        """Use API native tool_use protocol instead of JSON text parsing.
        Default OFF — requires LLM backend support."""
        return _flag("OPENHARNESS_NATIVE_TOOLS", False)

    @property
    def result_budgeting(self) -> bool:
        """Per-result and aggregate context budget enforcement.
        Default ON (part of context_management when enabled)."""
        return _flag("OPENHARNESS_RESULT_BUDGETING", True)


FLAGS = _Flags()
