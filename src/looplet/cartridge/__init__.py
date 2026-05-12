"""Workspace — bidirectional ``AgentPreset`` ↔ directory round-trip.

A *workspace* is a directory layout that round-trips with an
:class:`AgentPreset` losslessly for the JSON-able subset of the harness
and provides a clean code-escape hatch for the rest. It is the missing
inverse of :class:`looplet.bundles.SkillBundle`, which can be loaded
from disk but not written back from a live preset.

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
  ``max_briefing_tokens``, ``checkpoint_dir``); ``acceptance_criteria``;
  ``tool_metadata`` and ``generate_kwargs`` (JSON-able dicts).
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
returned in the resulting :class:`Workspace.serialization_warnings`.

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

Workspace authors retrieve loaded resources from
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
that consume :class:`Workspace`.
"""

from __future__ import annotations

import logging

__all__ = [
    "CartridgeLayout",
    "Cartridge",
    "CartridgeSerializationError",
    "preset_to_cartridge",
    "cartridge_to_preset",
]

logger = logging.getLogger(__name__)

# Layout constants, errors, and the preset-origin tracker live in
# :mod:`looplet.cartridge._layout` so the rest of the package can
# import them without bringing in the heavy serialiser / loader.
# Re-exported here so ``looplet.cartridge.X`` keeps resolving.
# Hook + tool + resource source loading lives in :mod:`looplet.cartridge._imports`.
from looplet.cartridge._imports import _import_module_from_path  # noqa: E402, F401
from looplet.cartridge._layout import (  # noqa: E402, F401
    SCHEMA_VERSION,
    CartridgeLayout,
    CartridgeSerializationError,
    _stamp_preset_origin,
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

# Resource registry, ref resolution, and the v1.1 single-file tool
# loader live in :mod:`looplet.cartridge._resources`. Re-imported here
# so external callers reaching into ``looplet.cartridge.X`` keep
# resolving.
from looplet.cartridge._resources import (  # noqa: E402, F401
    _load_resources,
    _load_single_file_tool,
    _resolve_refs,
)

# Serialiser (preset → directory) lives in :mod:`looplet.cartridge._serialise`.
from looplet.cartridge._serialise import preset_to_cartridge  # noqa: E402

# YAML reader/writer lives in :mod:`looplet.cartridge._yaml`. The
# parser is a deliberately minimal stdlib-only subset; full PyYAML
# would be overkill. Re-imported here so existing callers can still
# use ``looplet.cartridge._load_yaml``.
from looplet.cartridge._yaml import _dump_yaml, _load_yaml  # noqa: E402, F401

# Resource ref registry — the small standalone module that lets
# core hooks (permissions / streaming / evals / telemetry) call
# resource_ref_for without importing the full cartridge loader.
# Re-imported here for back-compat; consumers that historically
# reached ``looplet.cartridge.resource_ref_for`` keep working.
from looplet.refs import (  # noqa: E402, F401
    _REF_PREFIX,
    _register_resource_origin,
    _resource_origin,
    resource_ref_for,
)

# ── Back-compat aliases ────────────────────────────────────────
#
# The Cartridge Spec v1.0 renamed the artifact from "workspace" to
# "cartridge". The canonical names above are the ones exposed in
# SPEC.md and recommended for new code. The aliases below preserve
# the historical ``looplet.workspace.*`` API — same objects,
# different names — so existing imports continue to work.

Workspace = Cartridge
WorkspaceLayout = CartridgeLayout
WorkspaceSerializationError = CartridgeSerializationError
workspace_to_preset = cartridge_to_preset
preset_to_workspace = preset_to_cartridge
