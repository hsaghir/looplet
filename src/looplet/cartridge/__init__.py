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

import inspect
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from looplet.memory import PersistentMemorySource
    from looplet.presets import AgentPreset

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
from looplet.cartridge._layout import (  # noqa: E402
    SCHEMA_VERSION,  # noqa: F401  -- re-exported for back-compat
    CartridgeLayout,
    CartridgeSerializationError,
    _stamp_preset_origin,
)

# Cartridge dataclass + manifest helpers live in :mod:`looplet.cartridge._manifest`.
from looplet.cartridge._manifest import (  # noqa: E402, F401
    Cartridge,
    _manifest_present,
)
from looplet.cartridge._render import _apply_runtime_substitutions  # noqa: E402

# Resource registry, ref resolution, and the v1.1 single-file tool
# loader live in :mod:`looplet.cartridge._resources`. Serialiser-side
# renderers live in :mod:`looplet.cartridge._render`. Both are
# re-imported here so the rest of ``__init__.py`` (and external
# callers reaching into ``looplet.cartridge.X``) keep resolving.
from looplet.cartridge._resources import (  # noqa: E402
    _load_resources,
    _load_single_file_tool,
    _resolve_refs,  # noqa: F401  -- re-exported for tests
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

# ── Deserialise: directory → AgentPreset ───────────────────────


def cartridge_to_preset(
    workspace_dir: str | Path,
    *,
    state_factory: Callable[[int], Any] | None = None,
    strict: bool = False,
    runtime: dict[str, Any] | None = None,
) -> "AgentPreset":
    """Read a workspace directory and materialise an :class:`AgentPreset`.

    Args:
        workspace_dir: Path to the workspace root.
        state_factory: Builds the runtime ``state`` from ``max_steps``.
            Defaults to ``DefaultState(max_steps=...)``.
        strict: When ``True``, raise :class:`CartridgeSerializationError`
            on any tool / hook that fails to load (e.g. a hook whose
            ``config.yaml`` lacks the kwargs its constructor needs).
            When ``False`` (default), drop the offender, log a warning,
            and continue. Use ``strict=True`` for round-trip
            verification and CI lint.
        runtime: Optional dict of host-supplied runtime values
            (e.g. ``{"workspace": "/tmp/myrepo"}`` for the coder
            coder workspace). Three integration points read it:
              * ``${runtime.<key>}`` placeholders in ``config.yaml``
                are substituted before constructing ``LoopConfig``.
              * ``resources/<name>.py`` builders that declare
                ``def build(runtime)`` (or ``**kwargs``) receive it.
              * ``setup.py``'s ``setup(...)`` receives it via the
                ``runtime`` kwarg when its signature accepts it.

    Self-contained workspaces may co-locate helper modules (by
    convention ``<workspace>/lib.py``); the loader pushes the workspace
    root onto :data:`sys.path` for the duration of the load so
    ``from lib import X`` resolves cleanly inside tools / hooks /
    resources / setup.py without forcing every workspace to register an
    import shim.
    """
    root = Path(workspace_dir)
    # Accept both the historical ``workspace.json`` and the spec
    # alias ``cartridge.json``. ``workspace.json`` wins if both exist.
    if not _manifest_present(root):
        raise FileNotFoundError(
            f"cartridge metadata not found at "
            f"{root / CartridgeLayout.CARTRIDGE_JSON}; "
            f"is this a Cartridge directory?"
        )

    # ``extends:`` resolution. If config.yaml declares
    # ``extends: <path>``, we merge the parent workspace under the
    # current one — for every file under tools/, hooks/, resources/,
    # prompts/, memory/, and any top-level *.py the child doesn't
    # provide, we transparently inherit from the parent. Implemented by
    # materializing a merged directory in a tempdir and loading from
    # there. Multi-level extends is supported: the parent's own
    # ``extends:`` is resolved recursively before the merge.
    extended_root = _resolve_extends(root)
    if extended_root is not root:
        # Re-validate metadata in the merged dir.
        if not _manifest_present(extended_root):
            raise CartridgeSerializationError(
                f"merged workspace at {extended_root} missing workspace.json (or cartridge.json)"
            )
        root = extended_root

    # Shared-resource registry — built once, referenced by ``@<name>``
    # strings throughout hook / tool kwargs. Lets two hooks share the
    # same live object (e.g. a FileCache) on reload, instead of
    # silently splitting into two independent instances.
    runtime_dict = dict(runtime or {})

    # Workspaces are self-contained — their tools / hooks / resources
    # may reference co-located helper modules (conventionally
    # ``<workspace>/<wsname>_lib.py`` or similar; pick a name unique to
    # this workspace so two workspaces loaded back-to-back don't share
    # a cached ``lib`` module). The loader pushes the workspace root
    # AND its ``resources/`` subdirectory onto ``sys.path`` for the
    # duration of this load so:
    #   * ``from <wsname>_lib import X`` resolves to the workspace root
    #   * ``from <resource_name> import helper`` resolves to a
    #     ``resources/<resource_name>.py`` module (lets tools call back
    #     into resource modules without forcing a setup.py shim).
    import sys as _sys  # noqa: PLC0415

    root_str = str(root)
    resources_dir_str = str(root / CartridgeLayout.RESOURCES_DIR)
    pushed_paths: list[str] = []
    for p in (root_str, resources_dir_str):
        if (root / CartridgeLayout.RESOURCES_DIR).is_dir() or p == root_str:
            if p not in _sys.path:
                _sys.path.insert(0, p)
                pushed_paths.append(p)
    try:
        preset = _workspace_to_preset_inner(
            root, runtime_dict, state_factory=state_factory, strict=strict
        )
        # Stamp the preset with its origin so a subsequent
        # ``preset_to_cartridge`` call can copy any top-level ``*.py``
        # helper modules from the source workspace into the snapshot.
        # Tracked in a module-level WeakKey-style dict (keyed by
        # ``id(preset)`` with a finalizer) so the public AgentPreset
        # dataclass surface stays clean.
        _stamp_preset_origin(preset, root)
        return preset
    finally:
        for p in pushed_paths:
            try:
                _sys.path.remove(p)
            except ValueError:
                pass


_EXTENDS_TEMPDIRS: list[Path] = []


def _cleanup_extends_tempdirs() -> None:
    """Best-effort rmtree of every merged-extends tempdir at exit.

    Registered as an :mod:`atexit` hook at module import time so the
    process leaves no per-load tempdirs behind. Cleared in LIFO order
    (children before parents) which matches the order they were
    materialized in :func:`_resolve_extends`.
    """

    while _EXTENDS_TEMPDIRS:
        d = _EXTENDS_TEMPDIRS.pop()
        shutil.rmtree(d, ignore_errors=True)


import atexit as _atexit  # noqa: E402, PLC0415

_atexit.register(_cleanup_extends_tempdirs)


def _resolve_extends(root: Path, *, _seen: set[Path] | None = None) -> Path:
    """Resolve ``extends:`` in a workspace's config.yaml.

    If ``config.yaml`` declares ``extends: <path>`` (relative to ``root``
    or absolute), the parent workspace is recursively resolved and a
    *merged* workspace directory is materialized under a tempdir:

    * Parent files are copied first (recursively).
    * Child files overlay on top — any same-relative-path file in the
      child wins.

    Returns the path to the merged workspace, or ``root`` itself when
    there is no ``extends:`` to resolve. Multi-level inheritance works
    (parent can itself ``extends:`` a grandparent).

    Cycle detection: ``_seen`` carries the resolved paths visited so
    far; revisiting raises :class:`CartridgeSerializationError`.

    The merged workspace dir is created via :func:`tempfile.mkdtemp`
    with a stable prefix and tracked in :data:`_EXTENDS_TEMPDIRS`,
    which means it survives until interpreter exit (the loaded preset's
    modules may continue importing from it) and is then cleaned up
    via the module-level ``atexit`` hook.
    """
    cfg_path = root / CartridgeLayout.CONFIG_YAML
    if not cfg_path.is_file():
        return root
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg = _load_yaml(cfg_text, source_path=cfg_path) or {}
    extends_val = cfg.get("extends")
    if not extends_val:
        return root

    seen = set(_seen) if _seen is not None else set()
    root_resolved = root.resolve()
    if root_resolved in seen:
        raise CartridgeSerializationError(f"circular ``extends:`` detected at {root}")
    seen.add(root_resolved)

    parent_path = Path(extends_val)
    if not parent_path.is_absolute():
        parent_path = (root / parent_path).resolve()
    if not parent_path.is_dir():
        raise CartridgeSerializationError(
            f"extends: {extends_val!r} → {parent_path} does not exist or is not a directory"
        )

    # Resolve parent's own extends first (recursive).
    parent_root = _resolve_extends(parent_path, _seen=seen)

    import shutil as _shutil  # noqa: PLC0415
    import tempfile as _tempfile  # noqa: PLC0415

    merged = Path(_tempfile.mkdtemp(prefix=f"looplet_extends_{root.name}_"))
    # Register for cleanup at interpreter exit so the OS doesn't have
    # to (the merged dir holds .pyc caches and may import modules that
    # are still bound to its path until process end).
    _EXTENDS_TEMPDIRS.append(merged)
    # Copy parent first, then overlay child. Directory overlay is
    # correct for tools/, hooks/, resources/, prompts/, memory/ —
    # each entry lives in its own subdirectory and the child either
    # adds a new entry or replaces an entry of the same name.
    _shutil.copytree(parent_root, merged, dirs_exist_ok=True, symlinks=False)
    _shutil.copytree(root, merged, dirs_exist_ok=True, symlinks=False)

    # config.yaml needs *key-level* merging, not file-level overlay.
    # File-level overlay would cause the child's config.yaml to wholly
    # replace the parent's, silently dropping every parent key the child
    # didn't redeclare (e.g. ``max_tokens``, ``max_steps``, ``model:``,
    # ``permissions:``, ...). Builders observing this from outside the
    # merge would see ``extends:`` as a partial inheritance mechanism
    # — which is the opposite of what the schema promises.
    #
    # The merge strategy is shallow: top-level scalars and lists are
    # replaced wholesale by the child if redeclared, while top-level
    # mappings (``model:``, ``permissions:``, ``memory:``,
    # ``tool_metadata:``) are recursively shallow-merged so a child
    # can override a single key inside a block (e.g. just
    # ``model.reasoning_effort``) without losing siblings. Lists are
    # NOT concatenated — wholesale-replace mirrors how layered config
    # files (Kubernetes overlays, Terraform locals, Hydra) handle them.
    parent_cfg_path = parent_root / CartridgeLayout.CONFIG_YAML
    child_cfg_path = root / CartridgeLayout.CONFIG_YAML
    parent_cfg = (
        _load_yaml(parent_cfg_path.read_text(encoding="utf-8"))
        if parent_cfg_path.is_file()
        else None
    ) or {}
    child_cfg = (
        _load_yaml(child_cfg_path.read_text(encoding="utf-8")) if child_cfg_path.is_file() else None
    ) or {}
    # Drop the child's ``extends:`` so the inner loader doesn't re-resolve.
    child_cfg.pop("extends", None)
    parent_cfg.pop("extends", None)  # belt and braces; should already be gone
    merged_cfg = _shallow_merge_config(parent_cfg, child_cfg)
    merged_cfg_path = merged / CartridgeLayout.CONFIG_YAML
    merged_cfg_path.write_text(_dump_yaml(merged_cfg) + "\n", encoding="utf-8")
    return merged


def _shallow_merge_config(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge two parsed config.yaml dicts.

    Top-level scalars and lists from ``child`` win wholesale. Top-level
    mappings are recursively shallow-merged so a child can override one
    sub-key (``model.reasoning_effort``) without erasing siblings
    (``model.provider``). Used by :func:`_resolve_extends`.
    """
    out: dict[str, Any] = dict(parent)
    for key, child_val in child.items():
        parent_val = out.get(key)
        if isinstance(parent_val, dict) and isinstance(child_val, dict):
            out[key] = _shallow_merge_config(parent_val, child_val)
        else:
            out[key] = child_val
    return out


def _workspace_to_preset_inner(
    root: Path,
    runtime_dict: dict[str, Any],
    *,
    state_factory: Callable[[int], Any] | None,
    strict: bool,
) -> "AgentPreset":
    """Inner workspace loader. Assumes ``sys.path`` already includes
    the workspace root so co-located ``lib.py`` modules import cleanly.
    Split out from :func:`cartridge_to_preset` to keep the path-management
    boilerplate at the public boundary.
    """
    from looplet.loop import LoopConfig  # noqa: PLC0415
    from looplet.presets import AgentPreset  # noqa: PLC0415
    from looplet.tools import BaseToolRegistry, ToolSpec  # noqa: PLC0415
    from looplet.types import DefaultState  # noqa: PLC0415

    resources = _load_resources(root, runtime_dict)

    # Config
    cfg_kwargs: dict[str, Any] = {}
    cfg_path = root / CartridgeLayout.CONFIG_YAML
    if cfg_path.is_file():
        raw_cfg_text = cfg_path.read_text(encoding="utf-8")
        # Apply ``${runtime.<key>}`` substitution before YAML parsing so
        # workspace authors can parameterise config.yaml without needing
        # a setup.py for the common cases.
        raw_cfg_text = _apply_runtime_substitutions(raw_cfg_text, runtime_dict)
        cfg_kwargs.update(_load_yaml(raw_cfg_text, source_path=cfg_path) or {})

    sys_prompt_path = root / CartridgeLayout.SYSTEM_PROMPT_MD
    if sys_prompt_path.is_file():
        cfg_kwargs["system_prompt"] = sys_prompt_path.read_text(encoding="utf-8")

    # Memory sources — three sources, combined in the following order:
    #   1. File-based ``memory/*.md`` → StaticMemorySource (sorted)
    #   2. File-based ``memory/*.py`` → CallableMemorySource around the
    #      module's ``load`` attr (sorted alongside .md files)
    #   3. Any ``memory_sources: ["@x", ...]`` list declared in
    #      ``config.yaml`` — each ``@ref`` is resolved against the
    #      resource registry and must return a ``PersistentMemorySource``
    #      (typically a ``CallableMemorySource`` whose builder closes
    #      over ``runtime`` values like ``workspace`` / ``max_steps``).
    # The yaml-declared list runs through the existing
    # ``_resolve_refs(cfg_kwargs, resources)`` pass below; here we just
    # ensure file-based entries are appended on top of whatever the
    # yaml declared, preserving any explicit ordering the author chose.
    file_memory_sources: list[PersistentMemorySource] = []
    memory_dir = root / CartridgeLayout.MEMORY_DIR
    if memory_dir.is_dir():
        from looplet.memory import (  # noqa: PLC0415
            CallableMemorySource,
            StaticMemorySource,
        )

        memory_files = sorted(
            p for p in memory_dir.iterdir() if p.is_file() and p.suffix in (".md", ".py")
        )
        for memory_file in memory_files:
            if memory_file.suffix == ".md":
                file_memory_sources.append(
                    StaticMemorySource(text=memory_file.read_text(encoding="utf-8"))
                )
            else:
                module = _import_module_from_path(memory_file, f"_chw_memory_{memory_file.stem}")
                load_fn = getattr(module, "load", None)
                if not callable(load_fn):
                    msg = (
                        f"memory module {memory_file.name!r} must export a ``load(state)`` callable"
                    )
                    if strict:
                        raise CartridgeSerializationError(msg)
                    logger.warning("%s; skipping", msg)
                    continue
                file_memory_sources.append(CallableMemorySource(fn=load_fn))  # type: ignore[arg-type]

    # Merge file-based memory with any yaml-declared @ref list. The
    # yaml entries (still ``@ref`` strings at this point) get appended
    # after the file-based ones; ``_resolve_refs`` below converts each
    # ``@ref`` string in the list into the live resource instance.
    yaml_declared_memory = cfg_kwargs.get("memory_sources") or []
    if yaml_declared_memory or file_memory_sources:
        cfg_kwargs["memory_sources"] = list(file_memory_sources) + list(yaml_declared_memory)

    # Resolve ``"@<name>"`` references in config kwargs against the
    # shared-resource registry so callable / opaque LoopConfig fields
    # (tracer, router, compact_service, recovery_registry, cache_policy,
    # approval_handler, domain, build_briefing, output_schema, …) can be
    # wired declaratively from ``resources/<name>.py`` builders instead
    # of forcing every workspace into a ``setup.py`` detour. Symmetric
    # with the hook-kwargs ref resolution below.
    cfg_kwargs = _resolve_refs(
        cfg_kwargs,
        resources,
        runtime=runtime_dict,
        source_path=str(cfg_path) if cfg_path.is_file() else "config.yaml",
    )

    # ``builtin_tools:`` is a workspace-loader directive, not a
    # ``LoopConfig`` field — pop it before constructing the config so
    # ``LoopConfig(**cfg_kwargs)`` doesn't choke on an unknown kwarg.
    # The popped list is consumed below where the tool registry is
    # populated.
    _builtin_tool_names: list[str] = list(cfg_kwargs.pop("builtin_tools", None) or [])

    # Symmetric ``builtin_hooks:`` directive — opt into looplet-shipped
    # hooks (``skill_activation``, ...) without writing a hooks/<name>/
    # directory. Each entry is either a string (name) or a single-key
    # dict ``{name: {kwarg: value}}``. ``${ref:...}`` and ``${runtime.x}``
    # syntax in kwargs are resolved against the live resource registry.
    _builtin_hook_specs: list[Any] = list(cfg_kwargs.pop("builtin_hooks", None) or [])

    # ``state:`` is a workspace-loader directive that lets a workspace
    # describe the state object declaratively (any reference: a
    # ``${ref:...}`` resource, a ``${py:...}`` factory callable, or a
    # pre-resolved instance from ``_resolve_refs``). Falls back to the
    # ``state_factory`` constructor arg, and finally to ``DefaultState``.
    _state_directive = cfg_kwargs.pop("state", None)

    # ── v1.0 declarative slots (SPEC.md): model, permissions, memory.
    # Each is consumed here, *not* by ``LoopConfig``. We pop them off
    # cfg_kwargs so the LoopConfig constructor doesn't see unknown
    # kwargs, then process them after the config is built.
    _model_block = cfg_kwargs.pop("model", None)
    _permissions_block = cfg_kwargs.pop("permissions", None)
    _memory_block = cfg_kwargs.pop("memory", None)

    # ``output_schema:`` belongs inside ``tools/done/tool.yaml`` per
    # SPEC.md; declaring it at the top of ``config.yaml`` is a common
    # authoring mistake. Catch it here with an actionable message
    # rather than silently dropping a raw dict into
    # ``LoopConfig.output_schema`` (which expects an ``OutputSchema``
    # instance, not a JSON-Schema-ish dict).
    if isinstance(cfg_kwargs.get("output_schema"), dict):
        msg = (
            f"top-level ``output_schema:`` in {cfg_path} must be a "
            f"compiled ``OutputSchema`` instance (e.g. supplied via a "
            f"resource), not a raw dict. To declare an output schema "
            f"declaratively, put the ``output_schema:`` block inside "
            f"``tools/done/tool.yaml`` instead — see SPEC.md "
            f"'Output contract on done'."
        )
        if strict:
            raise CartridgeSerializationError(msg)
        logger.warning("%s; dropping the top-level output_schema", msg)
        cfg_kwargs.pop("output_schema", None)

    # Apply ``model:`` overrides BEFORE constructing LoopConfig so the
    # structured block wins over flat ``temperature:`` / ``max_tokens:``
    # at the top level. ``compile_model_block`` returns a dict of
    # LoopConfig field overrides plus an updated ``tool_metadata``
    # carrying provider / name / reasoning_effort for downstream
    # tooling to read.
    if _model_block is not None:
        from looplet.cartridge.spec_slots import compile_model_block  # noqa: PLC0415

        try:
            model_overrides = compile_model_block(_model_block, existing_cfg=cfg_kwargs)
        except ValueError as exc:
            msg = f"invalid 'model' block in {cfg_path}: {exc}"
            if strict:
                raise CartridgeSerializationError(msg) from exc
            logger.warning("%s; ignoring model block", msg)
            model_overrides = {}
        cfg_kwargs.update(model_overrides)

    # Catch unknown top-level keys in ``config.yaml`` BEFORE the
    # ``LoopConfig`` constructor would silently raise an obscure
    # ``TypeError`` (or, worse, the loader would silently drop them
    # because the known-directive list missed them). Anything left in
    # ``cfg_kwargs`` after the v1.0 slot pops above MUST correspond to
    # a real ``LoopConfig`` field; otherwise it's an authoring mistake.
    # In practice this catches ``output_schema:`` placed at the top
    # level instead of inside ``tools/done/tool.yaml``, hand-rolled
    # ``rules:`` / ``arg_matcher:`` shapes for ``permissions:`` (the
    # spec uses ``deny:``/``ask:``/``allow:`` lists), and typos like
    # ``temprature:`` / ``done-tool:``.
    _allowed_cfg_keys = (
        set(CartridgeLayout.SERIALIZABLE_CONFIG_FIELDS)
        | set(CartridgeLayout.NON_SERIALIZABLE_CONFIG_FIELDS)
        | {"system_prompt", "memory_sources", "extends"}
    )
    _unknown = sorted(k for k in cfg_kwargs if k not in _allowed_cfg_keys)
    if _unknown:
        msg = (
            f"unknown top-level key(s) in {cfg_path}: {_unknown}. "
            f"Recognized v1.0 slots: ``model:``, ``permissions:``, "
            f"``memory:``, ``builtin_tools:``, ``builtin_hooks:``, "
            f"``state:``, ``extends:`` plus LoopConfig fields. "
            f"Note ``output_schema:`` belongs inside ``tools/done/tool.yaml``, "
            f"not in config.yaml."
        )
        if strict:
            raise CartridgeSerializationError(msg)
        logger.warning("%s; dropping the unknown keys", msg)
        for k in _unknown:
            cfg_kwargs.pop(k, None)

    config = LoopConfig(**cfg_kwargs)

    # Track tool + hook modules so setup.py can wire shared resources
    # into them after the declarative load (see ``setup.py`` block below).
    tool_modules: dict[str, Any] = {}
    hook_modules: dict[str, Any] = {}

    # Tools
    registry = BaseToolRegistry()
    tools_dir = root / CartridgeLayout.TOOLS_DIR
    if tools_dir.is_dir():
        # ── Detect single-file ↔ multi-file collisions early ──
        # ``tools/foo.py`` (single-file form) and ``tools/foo/`` (multi-
        # file form) refer to the same tool name. Both forms loading
        # the same name produces a confusing dispatch race; an empty
        # ``tools/foo/`` next to ``tools/foo.py`` (a common authoring
        # mistake from leftover ``mkdir`` setup) used to fail much
        # later with "missing tool.yaml or execute.py", masking the
        # real problem (the empty dir).
        single_file_stems = {
            p.stem
            for p in tools_dir.iterdir()
            if p.is_file() and p.suffix == ".py" and not p.name.startswith("_")
        }
        multi_file_names = {p.name for p in tools_dir.iterdir() if p.is_dir()}
        collisions = sorted(single_file_stems & multi_file_names)
        if collisions:
            msg = (
                f"tool name collision in {tools_dir}: both single-file "
                f"(tools/<name>.py) and multi-file (tools/<name>/) forms "
                f"present for {collisions}. Pick one form per tool. If "
                f"the directory is an empty leftover from setup, ``rmdir`` "
                f"the empty tools/<name>/ directory."
            )
            if strict:
                raise CartridgeSerializationError(msg)
            logger.warning("%s; preferring single-file form", msg)
            # In loose mode, drop the colliding multi-file dirs so the
            # walk doesn't pick them up downstream.
            multi_file_names -= set(collisions)

        # ── v1.1 single-file tool form ────────────────────────
        # ``tools/<name>.py`` (no surrounding directory) is a tool
        # whose metadata lives in module-level dunders:
        #   __name__         (defaults to the file stem)
        #   __description__  (defaults to first docstring line)
        #   __parameters__   (dict; defaults to {} = no params)
        #   __tags__         (list[str]; optional, v1.1)
        #   __render__       (dict; optional, v1.1 render hints)
        #   __requires__     (list[str]; optional)
        #   __concurrent_safe__, __free__, __timeout_s__ (optional)
        # The module MUST export a callable named ``execute``.
        # Cuts boilerplate for trivial tools without changing the
        # multi-file form (still preferred for tools that need
        # extensive YAML-side metadata or substantial code).
        for tool_file in sorted(
            p
            for p in tools_dir.iterdir()
            if p.is_file() and p.suffix == ".py" and not p.name.startswith("_")
        ):
            spec = _load_single_file_tool(tool_file, strict=strict, tool_modules=tool_modules)
            if spec is not None:
                registry.register(spec)

        # ── multi-file tool form (existing) ───────────────────
        for tool_dir in sorted(
            p for p in tools_dir.iterdir() if p.is_dir() and p.name in multi_file_names
        ):
            spec_path = tool_dir / "tool.yaml"
            execute_path = tool_dir / "execute.py"
            if not spec_path.is_file() or not execute_path.is_file():
                msg = (
                    f"malformed tool dir {tool_dir} (missing tool.yaml or execute.py). "
                    f"If this directory is an empty leftover, ``rmdir`` it; "
                    f"if you meant a single-file tool, place the body at "
                    f"``{tool_dir.with_suffix('.py')}`` instead."
                )
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("skipping %s", msg)
                continue
            yaml_payload = (
                _load_yaml(
                    _apply_runtime_substitutions(
                        spec_path.read_text(encoding="utf-8"), runtime_dict
                    ),
                    source_path=spec_path,
                )
                or {}
            )
            module = _import_module_from_path(execute_path, f"_chw_tool_{tool_dir.name}")
            tool_modules[tool_dir.name] = module
            execute_fn = getattr(module, "execute", None)
            if execute_fn is None:
                # Fall back to the function whose name matches the YAML name.
                execute_fn = getattr(module, str(yaml_payload.get("name", "")), None)
            if not callable(execute_fn):
                msg = (
                    f"tool {tool_dir.name!r} has no callable execute "
                    f"(looked for `execute` and `{yaml_payload.get('name', '')}` in {execute_path})"
                )
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("%s; skipping", msg)
                continue
            raw_parameters = yaml_payload.get("parameters", {}) or {}
            if not isinstance(raw_parameters, dict):
                msg = (
                    f"tool {yaml_payload.get('name', tool_dir.name)!r} (in "
                    f"{tool_dir.name!r}): tool.yaml ``parameters:`` must be a "
                    f'mapping ("name: {{type: ...}}" entries), got '
                    f"{type(raw_parameters).__name__}. Source: {spec_path}."
                )
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("%s; skipping tool", msg)
                continue
            spec = ToolSpec(
                name=str(yaml_payload.get("name", tool_dir.name)),
                description=str(yaml_payload.get("description", "")),
                parameters=dict(raw_parameters),
                execute=execute_fn,
                concurrent_safe=bool(yaml_payload.get("concurrent_safe", False)),
                free=bool(yaml_payload.get("free", False)),
                timeout_s=yaml_payload.get("timeout_s"),
                requires=list(yaml_payload.get("requires", []) or []),
                tags=list(yaml_payload.get("tags", []) or []),
                render=dict(yaml_payload.get("render", {}) or {}),
            )
            # Surface tool name ↔ directory name mismatches. The agent's
            # system prompt and tool catalog use ``spec.name``, but
            # users debug via the directory layout. A typo in
            # ``name:`` silently registers the wrong tool name and
            # makes the agent lose access to the tool. Warn (loose
            # mode) or raise (strict mode) so the mismatch is caught
            # at load time.
            if spec.name != tool_dir.name:
                msg = (
                    f"tool name mismatch: directory is {tool_dir.name!r} but "
                    f"tool.yaml declares name: {spec.name!r}. The registered "
                    f"tool name will be {spec.name!r} (yaml wins). Rename "
                    f"the directory or fix the yaml so they match."
                )
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("%s", msg)
            # Surface ``requires:`` typos at load time. Without this,
            # a tool that declares ``requires: [my_resoruce]`` (typo)
            # silently receives ``ctx.resources["my_resoruce"] = None``
            # at dispatch time and crashes deep inside its body with
            # ``AttributeError`` — forcing the user to read a stack
            # trace inside their own tool to find the typo. Validating
            # here points at the ``tool.yaml`` line directly.
            if spec.requires:
                missing = [r for r in spec.requires if r not in resources]
                if missing:
                    available = sorted(resources)
                    msg = (
                        f"tool {spec.name!r} (in {tool_dir.name!r}) declares "
                        f"requires: {missing} but no such resource is defined. "
                        f"Available resources: {available}. "
                        f"Add a ``resources/<name>.py`` builder or fix the "
                        f"``requires:`` list in ``tool.yaml``."
                    )
                    if strict:
                        raise CartridgeSerializationError(msg)
                    logger.warning("%s; tool will receive None for missing resources", msg)
            # Surface ``parameters: {}`` mismatches with the execute.py
            # signature. The most common scaffold-then-edit friction:
            # ``scaffold_workspace`` writes ``parameters: {}`` and
            # ``def execute(ctx, **kwargs)`` together. Users replace
            # ``**kwargs`` with explicit keyword params (``*, name: str``)
            # but forget to also fill in ``parameters:``. The dispatcher
            # then rejects every call with VALIDATION because the schema
            # advertises zero parameters. We detect the mismatch here and
            # warn pointing at the tool.yaml. Detection is deliberately
            # conservative: we only flag declared-empty parameters paired
            # with a non-``**kwargs`` signature that has at least one
            # required keyword-only parameter beyond ``ctx``.
            if not spec.parameters:
                try:
                    sig = inspect.signature(execute_fn)
                    explicit = [
                        p
                        for p in sig.parameters.values()
                        if p.kind == inspect.Parameter.KEYWORD_ONLY and p.name != "ctx"
                    ]
                    has_var_keyword = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                except (TypeError, ValueError):
                    explicit = []
                    has_var_keyword = True
                if explicit and not has_var_keyword:
                    declared = [p.name for p in explicit]
                    msg = (
                        f"tool {spec.name!r} (in {tool_dir.name!r}): "
                        f"execute.py declares keyword params {declared} but "
                        f"tool.yaml has empty ``parameters: {{}}``. The "
                        f"dispatcher will reject every call with a VALIDATION "
                        f"error. Add the parameters block to {spec_path.name}, "
                        f"or accept ``**kwargs`` in execute.py."
                    )
                    if strict:
                        raise CartridgeSerializationError(msg)
                    logger.warning("%s", msg)
            registry.register(spec)

            # v1.0: ``output_schema:`` on the done tool installs an
            # OutputSchema validator on the LoopConfig so the loop
            # validates the agent's done() args before terminating.
            # Only the configured *primary* ``done_tool`` is treated
            # specially; output_schema on other tools (including
            # additional v1.1 ``done_tools``) is recorded in tool
            # metadata. Multi-sentinel cartridges where each sentinel
            # has a different schema should put a schema on each
            # tool.yaml and add a hook for cross-sentinel cross-checks
            # — there is no v1.0/v1.1 loader feature for a per-sentinel
            # schema map (deliberate: keeps the loader contract small).
            if (
                spec.name == config.done_tool
                and config.output_schema is None
                and isinstance(yaml_payload.get("output_schema"), dict)
            ):
                from looplet.cartridge.spec_slots import compile_output_schema  # noqa: PLC0415

                try:
                    config.output_schema = compile_output_schema(
                        dict(yaml_payload["output_schema"])
                    )
                except (ValueError, TypeError) as exc:
                    msg = f"invalid output_schema in {spec_path}: {exc}"
                    if strict:
                        raise CartridgeSerializationError(msg) from exc
                    logger.warning("%s; ignoring output_schema", msg)

    # Built-in tools — opt-in via ``builtin_tools:`` in config.yaml.
    # These are looplet-shipped tools (``subagent``, future helpers)
    # that any workspace can enable without writing a tools/<name>/
    # directory. Looked up in :mod:`looplet.builtin_tools` by name.
    if _builtin_tool_names:
        from looplet.builtin_tools import get_builtin_tool  # noqa: PLC0415

        for bname in _builtin_tool_names:
            spec = get_builtin_tool(bname)
            if spec is None:
                msg = (
                    f"unknown builtin tool {bname!r}; "
                    f"available built-ins: see ``looplet.builtin_tools.AVAILABLE``"
                )
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("%s; skipping", msg)
                continue
            registry.register(spec)
    # Hand the resource registry to the tool registry so any tool whose
    # ``ToolSpec.requires`` lists a resource name receives the live
    # instance via ``ctx.resources[name]`` at dispatch time.
    if resources:
        registry.set_resources(resources)

    # Hooks (explicit ``order:`` in config.yaml wins; ties + missing
    # values fall back to alphabetical-by-dirname).
    #
    # The directory-name convention (``00_FirstHook``, ``01_SecondHook``)
    # works for small hook chains but doesn't scale — inserting a hook
    # between positions 5 and 6 in a chain of 30 means renaming 24
    # directories with ``git mv``, destroying history and producing
    # noisy merge conflicts. The ``order:`` directive lets workspace
    # authors keep stable directory names while still controlling
    # execution order via a small integer in each hook's config.yaml.
    hooks: list[Any] = []
    hooks_dir = root / CartridgeLayout.HOOKS_DIR
    if hooks_dir.is_dir():
        # First pass: collect (order_key, hook_dir) pairs so we can sort
        # by explicit ``order:`` field with directory name as the
        # tie-breaker.
        hook_entries: list[tuple[Any, Path]] = []
        for hook_dir in (p for p in hooks_dir.iterdir() if p.is_dir()):
            cfg_yaml = hook_dir / "config.yaml"
            order_value: int | float = float("inf")  # un-ordered hooks sort last
            enabled = True
            if cfg_yaml.is_file():
                try:
                    raw = cfg_yaml.read_text(encoding="utf-8")
                    parsed = (
                        _load_yaml(
                            _apply_runtime_substitutions(raw, runtime_dict),
                            source_path=cfg_yaml,
                        )
                        or {}
                    )
                    if isinstance(parsed, dict):
                        if "order" in parsed:
                            order_value = int(parsed["order"])
                        if "enabled" in parsed:
                            enabled = bool(parsed["enabled"])
                except (CartridgeSerializationError, ValueError, TypeError):
                    # Malformed config.yaml will be re-raised below
                    # with full context — here we just fall back to
                    # alphabetical ordering for this hook.
                    pass
            # ``enabled: false`` lets workspace authors ablate a hook
            # without renaming or deleting its directory — essential
            # for ``extends:``-based ablation cells, where a child
            # workspace toggles individual parent hooks off to measure
            # their contribution.
            if not enabled:
                logger.info("skipping disabled hook %s", hook_dir.name)
                continue
            # Tuple sort: explicit order first, then directory name.
            # When ``order_value == inf`` (no ``order:`` field), hooks
            # sort by dirname only — the legacy behaviour.
            hook_entries.append(((order_value, hook_dir.name), hook_dir))
        hook_entries.sort(key=lambda x: x[0])
        for _key, hook_dir in hook_entries:
            hook_py = hook_dir / "hook.py"
            cfg_yaml = hook_dir / "config.yaml"
            if not hook_py.is_file():
                msg = f"malformed hook dir {hook_dir} (missing hook.py)"
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("skipping %s", msg)
                continue
            module = _import_module_from_path(hook_py, f"_chw_hook_{hook_dir.name}")
            hook_modules[hook_dir.name] = module
            hook_cfg = (
                _load_yaml(
                    _apply_runtime_substitutions(
                        cfg_yaml.read_text(encoding="utf-8"), runtime_dict
                    ),
                    source_path=cfg_yaml,
                )
                if cfg_yaml.is_file()
                else {}
            ) or {}
            class_name = str(hook_cfg.get("class_name") or "")
            if not class_name:
                # Pick the first class defined in the module.
                classes = [
                    obj
                    for name, obj in inspect.getmembers(module, inspect.isclass)
                    if obj.__module__ == module.__name__
                ]
                if not classes:
                    msg = f"hook {hook_dir.name!r} has no class in {hook_py}"
                    if strict:
                        raise CartridgeSerializationError(msg)
                    logger.warning("%s; skipping", msg)
                    continue
                cls = classes[0]
            else:
                cls = getattr(module, class_name, None)
                if cls is None:
                    msg = (
                        f"hook {hook_dir.name!r} declares class_name={class_name!r} "
                        f"but module has no such class"
                    )
                    if strict:
                        raise CartridgeSerializationError(msg)
                    logger.warning("%s; skipping", msg)
                    continue
            kwargs = dict(hook_cfg.get("kwargs", {}) or {})
            # Resolve ``"@<name>"`` references against the shared-resource
            # registry so hooks can share live objects on reload.
            kwargs = _resolve_refs(
                kwargs,
                resources,
                runtime=runtime_dict,
                source_path=str(cfg_yaml),
            )
            try:
                hooks.append(cls(**kwargs))
            except TypeError as exc:
                msg = (
                    f"hook {hook_dir.name!r} ({class_name or cls.__name__}) could not be "
                    f"instantiated with kwargs={kwargs} (in {cfg_yaml}): {exc}. "
                    f"Implement to_config(self) -> dict on the hook class so the "
                    f"workspace round-trip can capture its constructor kwargs."
                )
                if strict:
                    raise CartridgeSerializationError(msg) from exc
                logger.warning("%s; skipping hook", msg)

    # Built-in hooks — symmetric to ``builtin_tools:``. Each entry is
    # either a string (name) or a single-key dict ``{name: {kwargs...}}``.
    if _builtin_hook_specs:
        from looplet.builtin_hooks import build_builtin_hook  # noqa: PLC0415

        for entry in _builtin_hook_specs:
            if isinstance(entry, str):
                hname, raw_kwargs = entry, {}
            elif isinstance(entry, dict) and len(entry) == 1:
                hname, raw_kwargs = next(iter(entry.items()))
                raw_kwargs = dict(raw_kwargs or {})
            else:
                msg = f"builtin_hooks entry must be a string or single-key dict, got {entry!r}"
                if strict:
                    raise CartridgeSerializationError(msg)
                logger.warning("%s; skipping", msg)
                continue
            kwargs = _resolve_refs(
                raw_kwargs,
                resources,
                runtime=runtime_dict,
                source_path="config.yaml:builtin_hooks",
            )
            try:
                hook = build_builtin_hook(hname, resources=resources, kwargs=kwargs)
            except KeyError as exc:
                msg = (
                    f"unknown builtin hook {hname!r}: {exc}; "
                    f"see looplet.builtin_hooks.AVAILABLE for the registry"
                )
                if strict:
                    raise CartridgeSerializationError(msg) from exc
                logger.warning("%s; skipping", msg)
                continue
            except Exception as exc:  # noqa: BLE001
                msg = f"builtin hook {hname!r} could not be built: {exc}"
                if strict:
                    raise CartridgeSerializationError(msg) from exc
                logger.warning("%s; skipping", msg)
                continue
            hooks.append(hook)

    # State — priority: declarative ``state:`` directive in config.yaml,
    # then ``state_factory`` constructor arg, then ``DefaultState``.
    max_steps = int(getattr(config, "max_steps", 15))
    state: Any
    if _state_directive is not None:
        # ``_resolve_refs`` already turned ``${...}`` values into objects.
        # If the result is callable, call it with no args (factory protocol);
        # if it has ``__init__`` taking ``max_steps``, pass it; otherwise
        # treat as a pre-built instance.
        if callable(_state_directive) and not isinstance(_state_directive, type):
            # Plain callable factory — call with no args.
            state = _state_directive()
        elif inspect.isclass(_state_directive):
            # Class reference — try ``Class(max_steps=...)`` first,
            # fall back to ``Class()`` for stateless types.
            try:
                state = _state_directive(max_steps=max_steps)
            except TypeError:
                state = _state_directive()
        else:
            # Pre-built instance — use as-is.
            state = _state_directive
    elif state_factory is not None:
        state = state_factory(max_steps)
    else:
        state = DefaultState(max_steps=max_steps)

    # Sanity check: warn when ``done_tool`` doesn't point at any
    # registered tool. We deliberately only warn (not raise even under
    # strict) because some legitimate use cases construct ``done``
    # later (sub-agent presets where the parent injects a done tool,
    # workspaces extended by host code, test fixtures). The runtime
    # will surface the missing tool when the LLM tries to dispatch
    # it; this warning lets a builder catch the typo at load time
    # without breaking those use cases.
    if config.done_tool and config.done_tool not in registry.tool_names:
        # Suppress the warning when v1.1 ``done_tools:`` is in play —
        # the cartridge has explicitly opted into a different terminal
        # set, and the legacy default ``done`` may be irrelevant.
        if not config.done_tools:
            msg = (
                f"config.yaml declares ``done_tool: {config.done_tool!r}`` but no "
                f"such tool is registered. Available tools: {sorted(registry.tool_names)}. "
                f"Add a ``tools/{config.done_tool}/`` directory or fix the "
                f"``done_tool:`` field. The loop will fail at the agent's first "
                f"done() call."
            )
            logger.warning("%s", msg)
    # v1.1 ``done_tools:`` plural — same sanity check applied to each
    # extra terminal sentinel.
    for extra_done in config.done_tools:
        if extra_done and extra_done not in registry.tool_names:
            logger.warning(
                "config.yaml declares ``done_tools: [..., %r, ...]`` but no such tool "
                "is registered. Available tools: %s.",
                extra_done,
                sorted(registry.tool_names),
            )

    # ── v1.1 prompt files: briefing + recovery ──────────────────────
    # ``prompts/briefing.md`` (auto-prepended every step) and
    # ``prompts/recovery.md`` (injected after a tool error) are
    # opt-in: when present, the loader auto-instantiates a
    # StaticBriefingHook / RecoveryHintHook and prepends them so
    # they fire BEFORE any user hooks. Absent files mean no hook is
    # attached (zero overhead).
    briefing_path = root / CartridgeLayout.BRIEFING_MD
    recovery_path = root / CartridgeLayout.RECOVERY_MD
    if briefing_path.is_file() or recovery_path.is_file():
        from looplet.cartridge.prompt_files import (  # noqa: PLC0415
            RecoveryHintHook,
            StaticBriefingHook,
        )

        prompt_hooks: list[Any] = []
        if briefing_path.is_file():
            text = briefing_path.read_text(encoding="utf-8")
            prompt_hooks.append(StaticBriefingHook(text=text))
        if recovery_path.is_file():
            text = recovery_path.read_text(encoding="utf-8")
            prompt_hooks.append(RecoveryHintHook(text=text))
        # Prepend so these fire before user-declared hooks (which may
        # build on the briefing) and ahead of permission/built-in hooks
        # (which fire on dispatch boundaries; relative order doesn't
        # matter there).
        hooks = [*prompt_hooks, *hooks]

    preset = AgentPreset(
        config=config,
        hooks=hooks,
        tools=registry,
        state=state,
        resources=dict(resources),
    )

    # ── v1.0 declarative slots: permissions, memory.long_term ───────
    # Both are processed AFTER hooks/state/preset are built so the
    # auto-installed hook lands at the end of the hook list (after any
    # user-defined permission policy in ``hooks/``) and the long-term
    # memory file appends to file-based memory sources already loaded.
    # ``setup.py`` (below) can still override either, by design.

    if _permissions_block is not None:
        from looplet.cartridge.spec_slots import compile_permissions_block  # noqa: PLC0415

        try:
            permission_hook = compile_permissions_block(_permissions_block)
        except ValueError as exc:
            msg = f"invalid 'permissions' block in {cfg_path}: {exc}"
            if strict:
                raise CartridgeSerializationError(msg) from exc
            logger.warning("%s; ignoring permissions block", msg)
        else:
            preset.hooks.append(permission_hook)

    # Long-term memory: explicit ``memory: { long_term: <path> }`` wins;
    # otherwise, auto-load ``memory/long_term.md`` if present. Either
    # path resolves relative to the cartridge root and is appended to
    # the existing memory_sources so existing static files are not
    # disturbed.
    long_term_path: Path | None = None
    if isinstance(_memory_block, dict):
        explicit = _memory_block.get("long_term")
        if isinstance(explicit, str) and explicit:
            long_term_path = (root / explicit).resolve()
    if long_term_path is None:
        from looplet.cartridge.spec_slots import default_long_term_memory_path  # noqa: PLC0415

        candidate = root / default_long_term_memory_path()
        if candidate.is_file():
            long_term_path = candidate.resolve()
    if long_term_path is not None and long_term_path.is_file():
        from looplet.memory import StaticMemorySource  # noqa: PLC0415

        long_term_source = StaticMemorySource(
            text=long_term_path.read_text(encoding="utf-8"),
        )
        if preset.config.memory_sources is None:
            preset.config.memory_sources = []
        preset.config.memory_sources = list(preset.config.memory_sources) + [long_term_source]

    # ``setup.py`` escape hatch — runs after the declarative load to
    # let the workspace attach callable / opaque fields that don't
    # round-trip via JSON (e.g. ``LoopConfig.tracer``,
    # ``LoopConfig.compact_service``, custom domain adapters), or
    # inject shared resources into top-level tool/hook modules.
    setup_path = root / CartridgeLayout.SETUP_PY
    if setup_path.is_file():
        # Module name is derived from the workspace directory so two
        # workspaces loaded in the same process don't collide in
        # ``sys.modules`` (the legacy ``_chw_setup`` constant did).
        ws_slug = re.sub(r"\W+", "_", root.name).strip("_") or "workspace"
        module = _import_module_from_path(setup_path, f"looplet_setup_{ws_slug}")
        setup_fn = getattr(module, "setup", None)
        if not callable(setup_fn):
            raise CartridgeSerializationError(
                f"workspace setup.py at {setup_path} must define "
                f"`def setup(preset, resources, tool_modules, hook_modules)`"
            )
        # Modern signature accepts (preset, resources, tool_modules,
        # hook_modules); the older 2-arg signature still works for
        # forward compatibility — inspect.signature picks the right one.
        import inspect as _i  # noqa: PLC0415

        sig_params = _i.signature(setup_fn).parameters
        kwargs: dict[str, Any] = {}
        if "tool_modules" in sig_params:
            kwargs["tool_modules"] = tool_modules
        if "hook_modules" in sig_params:
            kwargs["hook_modules"] = hook_modules
        if "runtime" in sig_params:
            kwargs["runtime"] = runtime_dict
        result = setup_fn(preset, resources, **kwargs)
        if isinstance(result, AgentPreset):
            preset = result

    return preset


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
