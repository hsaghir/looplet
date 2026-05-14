"""Declaration of which modules constitute the *cartridge spec*.

This file marks a deliberate, small internal boundary inside
:mod:`looplet`: the schema and the loader are spec material; the loop,
backends, examples, CLI, and friends are runtime material. The split
exists so that:

* the ``SPEC.md`` document can point at a stable, narrow surface;
* a future repo split (Phase 2 of the agents-as-files roadmap) can
  carve these modules out without dragging in the runtime;
* third-party loaders in other languages can study a small set of
  Python files instead of the whole package.

A test (``tests/test_spec_boundary.py``) enforces that no module in
:data:`SPEC_MODULES` adds a top-level dependency on a non-spec
runtime module. New imports inside SPEC modules must either be
external (stdlib, typing-only ``TYPE_CHECKING``) or lazy (function
local). When a new spec slot is added, list its module here too.
"""

from __future__ import annotations

# ── The cartridge spec surface ──────────────────────────────────────
#
# Modules whose public types and behaviour define the cartridge.
# Keep this set deliberately small.
SPEC_MODULES: frozenset[str] = frozenset(
    {
        "memory",  # PersistentMemorySource protocol + impls
        "validation",  # OutputSchema, DoneValidator
        "permissions",  # PermissionEngine + rule semantics
        "hook_decision",  # the small return-type vocabulary hooks share
        "skills",  # agentskills.io SKILL.md format support
    }
)

# Runtime modules that SPEC modules MAY NOT top-level import. Listed
# explicitly so the boundary check can give a precise diagnostic.
# (TYPE_CHECKING-only imports and function-local imports are always
# allowed.)
SPEC_FORBIDDEN_TOP_LEVEL_IMPORTS: frozenset[str] = frozenset(
    {
        # The loop itself and everything it pulls in.
        "loop",
        "async_loop",
        "scaffolding",
        "resilient",
        # I/O backends.
        "backends",
        "router",
        # Application-level features.
        "presets",
        "subagent",
        "compact",
        "checkpoint",
        "evals",
        "provenance",
        "cache",
        "telemetry",
        "streaming",
        "context",
        "context_budget",
        "limits",
        "stagnation",
        "session",
        "session_tree",
        "history",
        "conversation",
        "prompts",
        "recovery",
        "recovery_strategies",
        "approval",
        "blueprints",
        "bundles",
        "cost",
        "steering",
        "events",
        "rpc",
        "harness_snapshot",
        "done_steps",
        "flags",
        "parse",
        "native_tools",
        "testing",
        "builtin_tools",
        "builtin_hooks",
        "examples",
        "cli",
        "mcp",
    }
)


__all__ = ["SPEC_MODULES", "SPEC_FORBIDDEN_TOP_LEVEL_IMPORTS"]
