"""Cartridge v1.x backward-compat helpers — slated for removal in v2.0.

This module is the **single home** for code paths that exist solely to
keep v1.x (schema_version=1) cartridges loading. Everything here is
called from :mod:`looplet.cartridge._load`, gated on
``schema_version < 2``. When v2.0 ships and the v1.x grace period
ends, this module will be deleted in one commit and the call sites in
``_load.py`` will reduce to bare ``CartridgeSerializationError`` raises
(already in place under the ``if is_v2:`` branches).

Concretely, v1.x compat covers three magic behaviours that v2.0
removes:

* **Runtime keys in ``config.yaml``** (sampling, context windows,
  caching, telemetry). v2 requires them in ``runtime.yaml``.
* **Magic ``prompts/briefing.md`` / ``prompts/recovery.md``
  auto-load.** v2 requires explicit declaration via
  ``builtin_hooks: - static_briefing: { path: ... }``.
* **``setup.py`` escape hatch.** v2 requires the equivalent wiring
  via ``resources/`` + ``builtin_hooks:`` + ``@ref`` strings.

Each helper here emits a ``DeprecationWarning`` (pointing at the new
home) when invoked, so v1 cartridges produce a clear migration trail.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from looplet.cartridge._layout import CartridgeLayout
from looplet.cartridge._yaml import _load_yaml

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def warn_v1_stray_runtime_keys(
    *,
    cfg_kwargs: dict[str, Any],
    cfg_path: Path,
    runtime_yaml_path: Path,
) -> None:
    """Warn when v1 ``config.yaml`` declares RUNTIME-tier keys.

    v2 cartridges hard-fail on the same condition (handled inline at
    the call site under ``if is_v2``). This helper is invoked only
    when ``schema_version < 2`` and emits a single
    ``DeprecationWarning`` listing every offending key.
    """
    if runtime_yaml_path.is_file():
        runtime_yaml_keys_set = set(
            _load_yaml(runtime_yaml_path.read_text(encoding="utf-8"), source_path=runtime_yaml_path)
            or {}
        )
        stray = sorted(
            k
            for k in cfg_kwargs
            if k in CartridgeLayout.RUNTIME_TIER_FIELDS and k not in runtime_yaml_keys_set
        )
    else:
        stray = sorted(k for k in cfg_kwargs if k in CartridgeLayout.RUNTIME_TIER_FIELDS)
    if not stray:
        return
    msg = (
        f"config.yaml at {cfg_path} declares runtime-tier key(s) "
        f"{stray}. Cartridge spec v2 moves runtime configuration into a "
        f"sibling ``runtime.yaml`` so the cartridge stays runtime-agnostic. "
        f"Move these keys to ``{runtime_yaml_path}``. "
        f"v1.x continues to accept them in config.yaml; v2.0 will hard-fail. "
        f"Run ``looplet migrate <cartridge>`` to upgrade."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=4)


def load_v1_magic_prompt_hooks(
    *,
    root: Path,
    builtin_hook_specs: list[Any],
) -> list[Any]:
    """Auto-load v1 magic ``prompts/briefing.md`` / ``prompts/recovery.md``.

    Returns the list of prompt hooks to PREPEND to the user's hook
    chain (so they fire BEFORE user-declared hooks). Empty list when
    neither magic file is present, OR when the cartridge has already
    declared the corresponding ``builtin_hook`` explicitly (in which
    case the magic file is intentional, not a v1 leftover).

    v2 cartridges raise ``CartridgeSerializationError`` for the same
    condition (handled inline at the call site under ``if is_v2``).
    Each hook returned here emits a ``DeprecationWarning`` pointing at
    the explicit ``builtin_hooks`` declaration the author should use.
    """
    from looplet.cartridge.prompt_files import (  # noqa: PLC0415
        RecoveryHintHook,
        StaticBriefingHook,
    )

    briefing_path = root / CartridgeLayout.BRIEFING_MD
    recovery_path = root / CartridgeLayout.RECOVERY_MD

    def _builtin_hook_declared(name: str) -> bool:
        for entry in builtin_hook_specs:
            if isinstance(entry, str) and entry == name:
                return True
            if isinstance(entry, dict) and name in entry:
                return True
        return False

    briefing_is_magic = briefing_path.is_file() and not _builtin_hook_declared("static_briefing")
    recovery_is_magic = recovery_path.is_file() and not _builtin_hook_declared("recovery_hint")
    if not (briefing_is_magic or recovery_is_magic):
        return []

    prompt_hooks: list[Any] = []
    if briefing_is_magic:
        text = briefing_path.read_text(encoding="utf-8")
        prompt_hooks.append(StaticBriefingHook(text=text))
        warnings.warn(
            f"Cartridge {root}: magic ``prompts/briefing.md`` auto-load is "
            f"deprecated (cartridge spec v2). Declare it explicitly via "
            f"``builtin_hooks: - static_briefing: {{ path: prompts/briefing.md }}`` "
            f"in config.yaml. v1.x continues to auto-load; v2.0 will drop "
            f"the magic-filename behaviour.",
            DeprecationWarning,
            stacklevel=2,
        )
    if recovery_is_magic:
        text = recovery_path.read_text(encoding="utf-8")
        prompt_hooks.append(RecoveryHintHook(text=text))
        warnings.warn(
            f"Cartridge {root}: magic ``prompts/recovery.md`` auto-load is "
            f"deprecated (cartridge spec v2). Declare it explicitly via "
            f"``builtin_hooks: - recovery_hint: {{ path: prompts/recovery.md }}`` "
            f"in config.yaml. v1.x continues to auto-load; v2.0 will drop "
            f"the magic-filename behaviour.",
            DeprecationWarning,
            stacklevel=2,
        )
    return prompt_hooks
