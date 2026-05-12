"""Back-compat alias module for :mod:`looplet.cartridge.spec_slots`.

The declarative-slot compilers (model:, permissions:, output_schema)
now live under the cartridge package. This shim preserves the
historical import path so existing code keeps working.
"""

from __future__ import annotations

from looplet.cartridge.spec_slots import *  # noqa: F401, F403
from looplet.cartridge.spec_slots import (  # noqa: F401
    compile_model_block,
    compile_output_schema,
    compile_permissions_block,
    default_long_term_memory_path,
)
