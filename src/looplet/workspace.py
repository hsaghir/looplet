"""Back-compat alias module for :mod:`looplet.cartridge`.

The agent-definition loader was historically named ``looplet.workspace``;
the Cartridge Spec v1.0 renamed the artifact to a "cartridge", and the
canonical module is now :mod:`looplet.cartridge`. This module simply
re-exports every public symbol so that ``from looplet.workspace import X``
keeps working unchanged.

New code SHOULD import from :mod:`looplet.cartridge`. Existing code
that imports from :mod:`looplet.workspace` continues to work; both
forms refer to identical objects (``is``-equal).
"""

from __future__ import annotations

# Re-export everything from the canonical module so ``from
# looplet.workspace import *`` and attribute access both work.
from looplet.cartridge import *  # noqa: F401, F403

# Re-export private symbols that existing tests reach into directly.
# Names beginning with ``_`` are NOT re-exported by ``import *``,
# so they need to be named explicitly.
from looplet.cartridge import (  # noqa: F401
    SCHEMA_VERSION,
    Workspace,
    WorkspaceLayout,
    WorkspaceSerializationError,
    preset_to_workspace,
    resource_ref_for,
    workspace_to_preset,
)

# These private registry helpers live in :mod:`looplet.refs`; the
# ``looplet.cartridge`` package no longer re-exports them after the
# round-2 cleanup. Import directly from the source module.
from looplet.refs import _register_resource_origin, _resource_origin  # noqa: F401
