"""Back-compat alias module for :mod:`looplet.cartridge.hot_reload`.

WorkspaceWatcher now lives under the cartridge package. This shim
preserves the historical import path.
"""

from __future__ import annotations

from looplet.cartridge.hot_reload import *  # noqa: F401, F403
from looplet.cartridge.hot_reload import WorkspaceWatcher  # noqa: F401
