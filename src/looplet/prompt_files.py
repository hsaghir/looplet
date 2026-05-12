"""Back-compat alias module for :mod:`looplet.cartridge.prompt_files`.

The auto-attached briefing/recovery hooks now live under the
cartridge package. This shim preserves the historical import path.
"""

from __future__ import annotations

from looplet.cartridge.prompt_files import *  # noqa: F401, F403
from looplet.cartridge.prompt_files import (  # noqa: F401
    RecoveryHintHook,
    StaticBriefingHook,
)
