"""Cartridge — bidirectional ``AgentPreset`` ↔ directory round-trip.

A *workspace* is a directory layout that round-trips with an
:class:`AgentPreset` losslessly for the JSON-able subset of the harness
and provides a clean code-escape hatch for the rest. It is the missing
inverse of :class:`looplet.bundles.SkillBundle`, which can be loaded
from disk but not written back from a live preset.

Extraction contract (loader independence)
-----------------------------------------

``looplet.cartridge`` is structured so it can be extracted into a
standalone ``looplet-cartridge`` package without touching the rest
of looplet. To preserve that property, this package's TOP-LEVEL
(import-time) dependencies on the looplet umbrella are pinned to
the following allowlist; everything else MUST be imported lazily
inside function bodies. The list is locked by
``tests/test_cartridge_extraction_contract.py`` — if it grows,
update both places intentionally.

Allowlisted top-level looplet imports inside cartridge/**.py:

* ``looplet.refs``           — tiny shared registry (~117 LOC);
                               designed as the cross-package bridge.
* ``looplet.hook_decision``  — small leaf module used by
                               ``prompt_files.py``.
* ``looplet.permissions``    — used by ``spec_slots.py`` to compile
                               the declarative ``permissions:`` block.
* ``looplet.validation``     — used by ``spec_slots.py`` to compile
                               the declarative ``output_schema:`` block.

Everything else — ``loop``, ``presets``, ``tools``, ``types``,
``memory``, ``compact``, ``builtin_*``, ``backends`` — is imported
lazily (function-local), so a future split package can replace
those imports with Protocol-typed runtime hooks without any
ripple in the cartridge code itself.

Design goal
-----------

Make the agent harness an editable artifact on disk so external tools
(harness search, GEPA-style evolution, diff/review workflows) can
mutate components by file diff, version-control the result with git,
and re-materialise an :class:`AgentPreset` for execution — without
anyone forking the loop or replacing the workspace mechanism.

Layout
------

::

    my_workspace/
    ├── workspace.json           # schema_version, name, description, version bookkeeping
    ├── prompts/
    │   └── system.md            # config.system_prompt (file body)
    ├── config.yaml              # LoopConfig JSON-able subset (max_steps, etc.)
    ├── tools/
    │   └── grep/
    │       ├── tool.yaml        # name, description, parameters, concurrent_safe, free, timeout_s
    │       └── execute.py       # def execute(*, ...) -> Any
    ├── hooks/
    │   └── 00_done_gate/        # leading number = sort order = hook list order
    │       ├── hook.py          # exposes either `class HookClass` or `def build()`
    │       └── config.yaml      # optional kwargs for HookClass(**kwargs)
    └── memory/
        └── lessons.md           # one StaticMemorySource per file; filename = source name

What is round-trippable
-----------------------

* ``LoopConfig``: every primitive scalar field (``max_steps``,
  ``max_tokens``, ``temperature``, ``recovery_temperature``,
  ``done_tool``, ``max_turn_continuations``, ``use_native_tools``,
  ``concurrent_dispatch``, ``reactive_recovery``, ``context_window``,
  ``max_briefing_tokens``, ``checkpoint_dir``); ``generate_kwargs``
  (JSON-able dict, RUNTIME-tier sampling overrides). ``tool_metadata``
  is auto-populated by the loader (e.g. with the resolved model
  identity for cost tracking) and should not be authored by hand.
  Acceptance gates are not a config field \u2014 declare them in the
  ``config.yaml`` of a ``hooks/<name>/`` ``check_done`` hook that
  enforces them (see ``examples/snippets/11_quality_gate/``).
* Every :class:`ToolSpec` whose ``execute`` is a top-level function
  (closures cannot be re-imported from disk).
* Every hook that either: (a) implements an opt-in
  ``to_config() -> dict`` returning JSON-able kwargs for its
  constructor, OR (b) is a top-level class importable from a written
  ``hook.py`` module, OR (c) ships its own ``hook.py`` source via the
  code-escape hatch.
* :class:`StaticMemorySource` instances; other memory sources land
  under the code-escape hatch.

What is NOT round-trippable (raises ``CartridgeSerializationError``
when ``preset_to_cartridge`` is called with ``strict=True``)
-----------------------------------------------------------------------

Callable / opaque ``LoopConfig`` fields (``build_briefing``,
``router``, ``tracer``, ``compact_service``, ``recovery_registry``,
``output_schema``, ``initial_checkpoint``, ``cache_policy``,
``cancel_token``, ``approval_handler``, ``render_messages_override``,
``domain``). When ``strict=False`` (default), they are silently
omitted from the serialized config and a list of skipped fields is
returned in the resulting :class:`Cartridge.serialization_warnings`.

These fields can still be wired **declaratively on load** by
hand-authoring ``config.yaml`` with the workspace reference grammar.
Three reference forms are supported, applied uniformly to every
string value the loader processes:

* ``${ref:name}``           — resolve from the resource registry
                              (``resources/name.py::build()``)
* ``${py:module:symbol}``   — import a Python object by dotted path
* ``${runtime.field}``      — read the per-invocation runtime dict;
                              supports nested ``${runtime.a.b}`` and
                              defaults via ``${runtime.x:-default}``

The legacy ``"@name"`` form continues to work as an alias for
``${ref:name}`` so older workspaces keep loading unchanged.

Example::

    # config.yaml
    max_steps: ${runtime.max_steps:-20}
    compact_service: ${ref:compact_service}
    tracer: ${py:my.module:make_tracer}
    state: ${py:my.app.state:MyAgentState}

    # resources/compact_service.py
    from looplet.compact import default_compact_service
    def build(runtime=None):
        return default_compact_service(keep_recent=2)

This eliminates the ``setup.py`` detour for the common case of
attaching callable ``LoopConfig`` services or custom state classes.
Tool dependency injection also goes through the same resource
registry: ``tool.yaml`` declares ``requires: [<name>, ...]`` and the
dispatcher hands the resolved instances to the tool's
``execute(ctx, ...)`` via ``ctx.resources[name]``. Memory sources
accept the same references (``memory_sources: ['${ref:project_memory}']``).

Cartridge authors retrieve loaded resources from
:attr:`AgentPreset.resources` after calling
:func:`cartridge_to_preset` — useful for callers (benchmarks,
evidence-bundle writers) that need post-load access to live objects
without writing a setup.py themselves.

``setup.py`` remains as an opt-in escape hatch for users with
truly imperative load-time wiring needs, but no shipped example
needs one — every published workspace is fully declarative.

Why this is in Looplet (not in a research extension)
----------------------------------------------------

The disk format is generic infrastructure: anyone can use it for
workspace editing, agent diffing, code review, packaging, or
between-round harness search. The research-specific layer
(manifests with ``predicted_fixes``/``predicted_regressions``,
the evolve agent, the search loop) lives in downstream packages
that consume :class:`Cartridge`.
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

# Resource registry helpers — ``_resolve_refs`` is the only one with
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
