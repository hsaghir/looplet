"""Layout constants, errors, and the preset-origin tracker.

This module is intentionally tiny and dependency-free (stdlib only).
Everything here is consumed by every other ``looplet.cartridge.*``
submodule; keeping it self-contained avoids cycles.
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


# ── Layout constants ────────────────────────────────────────────


class CartridgeLayout:
    """Fixed mount points inside a workspace directory."""

    WORKSPACE_JSON = "workspace.json"
    # Cartridge Spec v1.0 alias for ``workspace.json``. The loader
    # accepts either filename so cartridges authored against the spec
    # terminology load without renaming.
    CARTRIDGE_JSON = "cartridge.json"
    CONFIG_YAML = "config.yaml"
    PROMPTS_DIR = "prompts"
    SYSTEM_PROMPT_MD = "prompts/system.md"
    BRIEFING_MD = "prompts/briefing.md"
    RECOVERY_MD = "prompts/recovery.md"
    TOOLS_DIR = "tools"
    HOOKS_DIR = "hooks"
    MEMORY_DIR = "memory"
    RESOURCES_DIR = "resources"
    SETUP_PY = "setup.py"

    # ``LoopConfig`` field names that round-trip via ``config.yaml``.
    # NOTE: ``tool_metadata`` IS here but is auto-populated by the loader
    # (e.g. with the resolved model identity for cost tracking); user
    # cartridges should not author it directly. If you want acceptance
    # gates, write a ``check_done`` hook under ``hooks/<name>/`` that
    # reads its criteria from ``hook.config.yaml`` — same as any other
    # policy. See ``examples/snippets/11_quality_gate/``.
    SERIALIZABLE_CONFIG_FIELDS: tuple[str, ...] = (
        "max_steps",
        "max_tokens",
        "temperature",
        "recovery_temperature",
        "done_tool",
        "done_tools",
        "max_turn_continuations",
        "use_native_tools",
        "concurrent_dispatch",
        "reactive_recovery",
        "context_window",
        "max_briefing_tokens",
        "checkpoint_dir",
        "tool_metadata",
        "generate_kwargs",
        "context_window_steps",
        "context_inline_per_step_chars",
        "context_window_total_chars",
    )

    # ``LoopConfig`` callable / opaque fields that cannot round-trip.
    NON_SERIALIZABLE_CONFIG_FIELDS: tuple[str, ...] = (
        "build_briefing",
        "extract_entities",
        "build_trace",
        "build_prompt",
        "extract_step_metadata",
        "domain",
        "router",
        "tracer",
        "recovery_registry",
        "compact_service",
        "output_schema",
        "initial_checkpoint",
        "cache_policy",
        "cancel_token",
        "approval_handler",
        "render_messages_override",
    )

    # ── Field tiering (cartridge spec v2 prep) ──────────────────
    # Three tiers carve ``LoopConfig`` into "what the agent does"
    # (CONTRACT), "how the runtime executes it" (RUNTIME), and
    # "what the host application provides" (HOST). The cartridge
    # spec v2 will move RUNTIME and HOST keys out of
    # ``config.yaml`` into a sibling ``runtime.yaml`` (RUNTIME)
    # and host-supplied :class:`LoopConfig` patches (HOST). v1.x
    # accepts both shapes; runtime keys placed in ``config.yaml``
    # raise a :class:`DeprecationWarning` pointing at the new home.
    #
    # See ``paper/principled_cartridge_v2.md`` for the rationale.

    RUNTIME_TIER_FIELDS: frozenset[str] = frozenset(
        {
            # Sampling — host-tunable defaults.
            "max_tokens",
            "temperature",
            "recovery_temperature",
            "max_turn_continuations",
            # Backend kwargs passthrough — same family as the sampling
            # knobs above (``top_p``, ``frequency_penalty``, etc.); pure
            # "how to sample", not "what the agent does".
            "generate_kwargs",
            # Engine knobs.
            "use_native_tools",
            "concurrent_dispatch",
            "reactive_recovery",
            # Context / window management.
            "context_window",
            "context_window_steps",
            "context_inline_per_step_chars",
            "context_window_total_chars",
            "max_briefing_tokens",
            # Wired capabilities — runtime-specific implementations.
            "router",
            "tracer",
            "recovery_registry",
            "compact_service",
            "cache_policy",
            # Persistence — operational, not behavioural.
            "checkpoint_dir",
            "initial_checkpoint",
            "tool_result_persist_dir",
        }
    )

    HOST_TIER_FIELDS: frozenset[str] = frozenset(
        {
            "approval_handler",
            "cancel_token",
            "render_messages_override",
        }
    )

    @classmethod
    def contract_tier_fields(cls) -> frozenset[str]:
        """Fields that legitimately live in ``config.yaml``.

        Computed as everything serialisable that isn't tagged
        RUNTIME, plus the v1.0 declarative slots and the system
        prompt that the loader populates from ``prompts/system.md``.
        """
        return frozenset(cls.SERIALIZABLE_CONFIG_FIELDS) - cls.RUNTIME_TIER_FIELDS


# ── Errors ──────────────────────────────────────────────────────


class CartridgeSerializationError(RuntimeError):
    """Raised when a workspace component cannot be round-tripped.

    Use ``strict=False`` on :func:`preset_to_cartridge` to demote these
    into recorded warnings on the resulting :class:`Workspace`.
    """


# ── Preset origin tracker ───────────────────────────────────────
#
# Maps ``id(preset)`` → source workspace root for presets returned by
# :func:`cartridge_to_preset`. Read by :func:`preset_to_cartridge` so
# it can copy any top-level ``*.py`` helper modules from the source
# workspace into the snapshot.
_preset_origin: dict[int, Path] = {}


def _stamp_preset_origin(preset: Any, root: Path) -> None:
    """Record the source workspace root for ``preset``.

    Registers a finalizer that drops the entry when ``preset`` is
    garbage-collected so this map can't leak memory.
    """
    key = id(preset)
    _preset_origin[key] = root.resolve()
    weakref.finalize(preset, _preset_origin.pop, key, None)


def _preset_origin_root(preset: Any) -> Path | None:
    """Return the source workspace root for ``preset`` (or ``None``
    when the preset wasn't built via :func:`cartridge_to_preset`)."""
    return _preset_origin.get(id(preset))
