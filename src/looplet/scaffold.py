"""Back-compat alias module for :mod:`looplet.cartridge.scaffold`.

The cartridge package now lives at :mod:`looplet.cartridge` (a
package, not a module). This shim preserves the historical
``looplet.scaffold`` import path so existing code keeps working.
New code SHOULD import from :mod:`looplet.cartridge.scaffold`.
"""

from __future__ import annotations

from looplet.cartridge.scaffold import *  # noqa: F401, F403
from looplet.cartridge.scaffold import (  # noqa: F401
    scaffold_cartridge,
    scaffold_workspace,
)
