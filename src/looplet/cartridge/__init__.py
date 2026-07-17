"""Cartridge format: bidirectional ``AgentPreset`` and directory round-trip.

A cartridge makes an agent harness reviewable as ordinary files. The supported
JSON-able subset of an :class:`AgentPreset` can be serialized with
:func:`preset_to_cartridge` and loaded with :func:`cartridge_to_preset`.
Importable Python modules provide the explicit code boundary for tools, hooks,
resources, and custom state.

Current schema-v2 layout
------------------------

::

        agent.cartridge/
        ├── cartridge.json
        ├── config.yaml
        ├── runtime.yaml
        ├── prompts/system.md
        ├── tools/<name>/{tool.yaml, execute.py}
        ├── hooks/<order>_<name>/{config.yaml, hook.py}
        ├── resources/<name>.py
        ├── memory/*.md
        └── evals/{cases/, collect_*.py, eval_*.py}

The loader accepts schema version 2 only. Contract fields belong in
``config.yaml``; host/runtime fields belong in ``runtime.yaml``. References use
one explicit grammar:

* ``${ref:name}`` resolves a resource built from ``resources/name.py``.
* ``${py:module:symbol}`` imports a Python object.
* ``${runtime.field}`` reads host-supplied runtime data and supports defaults.

The historical ``@name`` spelling remains an alias for ``${ref:name}``.
Imperative root-level ``setup.py`` wiring is rejected by schema v2.

Round-trip boundary
-------------------

Primitive configuration fields, top-level tool functions, serializable hooks,
and static memory sources round-trip directly. Opaque runtime objects and
closures either produce a serialization warning or raise
:class:`CartridgeSerializationError` when ``strict=True``. Declarative
references and resource builders are the supported way to reconstruct those
objects at load time.

Extraction contract
-------------------

This package is kept extractable into a future standalone cartridge package.
Top-level imports from the Looplet umbrella are restricted to the allowlist
locked by ``tests/test_cartridge_extraction_contract.py``:

* ``looplet.refs``
* ``looplet.hook_decision``
* ``looplet.permissions``
* ``looplet.validation``

Everything else is imported lazily inside function bodies. Update the contract
test intentionally if this allowlist changes.
"""

from __future__ import annotations

import logging

__all__ = [
    "CartridgeLayout",
    "Cartridge",
    "CartridgeSerializationError",
    "preset_to_cartridge",
    "cartridge_to_preset",
    "CartridgePortabilityReport",
    "ComponentReport",
    "analyse_cartridge",
]

logger = logging.getLogger(__name__)

# Layout constants and errors live in :mod:`looplet.cartridge._layout`.
# Re-exported here so ``looplet.cartridge.X`` keeps resolving for the
# public surface only. Other internal helpers (``_load_resources``,
# ``_load_single_file_tool``, ``_stamp_preset_origin``, ...) used to
# be re-exported here for back-compat; they had zero out-of-package
# callers as of round-2 cleanup and are now imported from their
# defining module directly when needed.
from looplet.cartridge._imports import _import_module_from_path  # noqa: E402, F401
from looplet.cartridge._layout import (  # noqa: E402, F401
    SCHEMA_VERSION,
    CartridgeLayout,
    CartridgeSerializationError,
)

# Loader (directory → :class:`AgentPreset`) lives in :mod:`looplet.cartridge._load`.
from looplet.cartridge._load import (  # noqa: E402, F401
    _workspace_to_preset_inner,
    cartridge_to_preset,
)

# Cartridge dataclass + manifest helpers live in :mod:`looplet.cartridge._manifest`.
from looplet.cartridge._manifest import (  # noqa: E402, F401
    Cartridge,
    _manifest_present,
)
from looplet.cartridge._render import _apply_runtime_substitutions  # noqa: E402, F401

# Resource registry helpers - ``_resolve_refs`` is the only one with
# out-of-package callers (see ``test_cartridge_extraction_contract``).
from looplet.cartridge._resources import _resolve_refs  # noqa: E402, F401

# Serialiser (preset → directory) lives in :mod:`looplet.cartridge._serialise`.
from looplet.cartridge._serialise import preset_to_cartridge  # noqa: E402

# YAML reader/writer lives in :mod:`looplet.cartridge._yaml`. The
# parser is a deliberately minimal stdlib-only subset; full PyYAML
# would be overkill. Re-imported here so existing callers can still
# use ``looplet.cartridge._load_yaml``.
from looplet.cartridge._yaml import _dump_yaml, _load_yaml  # noqa: E402, F401

# Whole-cartridge portability report (static analyser).
from looplet.cartridge.portability import (  # noqa: E402, F401
    CartridgePortabilityReport,
    ComponentReport,
    analyse_cartridge,
)

# ``resource_ref_for`` is the public entry; the underlying registry
# (``_REF_PREFIX``, ``_register_resource_origin``, ``_resource_origin``)
# is private to :mod:`looplet.refs` and not re-exported here.
from looplet.refs import resource_ref_for  # noqa: E402, F401
