"""Serialise an :class:`AgentPreset` to a cartridge directory.

The public entry point is :func:`preset_to_cartridge`. The module
also exports the helpers it uses (``_write_*``, ``_iter_tool_specs``,
``_infer_hook_kwargs_from_init``, ``_render_hook_source``,
``_copy_workspace_helpers``, ``_write_resources_for_refs``,
``_write_memory``) so the loader / round-trip tests can call into
them directly when needed.

Logically the inverse of :mod:`looplet.cartridge._load` — a
preset that came in via ``cartridge_to_preset`` should round-trip
to an equivalent directory through this module.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from looplet.presets import AgentPreset
    from looplet.tools import BaseToolRegistry

from looplet.cartridge._layout import (
    CartridgeLayout,
    CartridgeSerializationError,
    _preset_origin_root,
)
from looplet.cartridge._manifest import Cartridge
from looplet.cartridge._render import (
    _DataclassReprFailed,
    _hook_class,
    _render_dataclass_kwargs,
    _safe_filename,
)
from looplet.cartridge._yaml import _dump_yaml
from looplet.refs import _REF_PREFIX, _resource_origin, resource_ref_for

logger = logging.getLogger(__name__)

# ── Serialise: AgentPreset → directory ─────────────────────────


def preset_to_cartridge(
    preset: "AgentPreset",
    out_dir: str | Path,
    *,
    name: str | None = None,
    description: str = "",
    overwrite: bool = False,
    strict: bool = False,
) -> Cartridge:
    """Write an :class:`AgentPreset` to a workspace directory.

    Args:
        preset: The harness to serialise.
        out_dir: Target directory. Created if missing. If it already
            exists and is non-empty, ``overwrite=True`` is required.
        name: Cartridge name. Defaults to the directory basename.
        description: Free-form description stored in
            ``workspace.json``.
        overwrite: Allow writing into a non-empty existing directory
            (its workspace-managed subdirectories are wiped first).
        strict: When ``True``, raise
            :class:`CartridgeSerializationError` on any non-round-trippable
            component. When ``False`` (default), record warnings on the
            returned workspace and skip the offending field.

    Returns:
        The :class:`Cartridge` describing the newly-written directory.
    """
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} is not empty; pass overwrite=True to wipe and rewrite")
    if root.exists() and overwrite:
        for sub in (
            CartridgeLayout.PROMPTS_DIR,
            CartridgeLayout.TOOLS_DIR,
            CartridgeLayout.HOOKS_DIR,
            CartridgeLayout.MEMORY_DIR,
            CartridgeLayout.RESOURCES_DIR,
        ):
            sub_path = root / sub
            if sub_path.is_dir():
                shutil.rmtree(sub_path)
        for stale in (
            CartridgeLayout.CARTRIDGE_JSON,
            CartridgeLayout.WORKSPACE_JSON,
            CartridgeLayout.CONFIG_YAML,
            "runtime.yaml",
        ):
            stale_path = root / stale
            if stale_path.is_file():
                stale_path.unlink()
    root.mkdir(parents=True, exist_ok=True)

    workspace = Cartridge(
        path=root,
        name=name or root.name,
        description=description,
    )
    warnings: list[str] = []

    # 1. config — write JSON-able subset; emit warnings for the rest.
    cfg = preset.config
    serialized_cfg: dict[str, Any] = {}
    for fname in CartridgeLayout.SERIALIZABLE_CONFIG_FIELDS:
        if fname == "system_prompt":
            continue  # written as a separate prompts/system.md file
        if not hasattr(cfg, fname):
            continue
        value = getattr(cfg, fname)
        try:
            json.dumps(value)
        except TypeError:
            msg = f"config.{fname} ({type(value).__name__!r}) is not JSON-able; skipping"
            if strict:
                raise CartridgeSerializationError(msg)
            warnings.append(msg)
            continue
        serialized_cfg[fname] = value

    # Auto-emit ``@ref`` strings into config.yaml for any non-serializable
    # LoopConfig field that is set, mirroring how hook kwargs ride the
    # resource-builder machinery. The actual instances get collected here
    # and passed to ``_write_resources_for_refs`` below so the writer
    # auto-generates ``resources/<field>.py`` placeholders the loader
    # resolves declaratively.
    config_field_refs: dict[str, Any] = {}
    for fname in CartridgeLayout.NON_SERIALIZABLE_CONFIG_FIELDS:
        if not hasattr(cfg, fname):
            continue
        value = getattr(cfg, fname)
        if value is None:
            continue
        # Round-trip via ``@<fname>`` ref + auto-generated resource stub.
        # The user can replace the stub with a real builder later; for
        # in-process snapshot+reload (harness search / GEPA evolution)
        # the auto-emitted builder rebuilds the type from its module
        # the same way hook resources do.
        serialized_cfg[fname] = f"{_REF_PREFIX}{fname}"
        config_field_refs[fname] = value

    # Re-emit declarative tool directives so the round-tripped cartridge
    # re-derives these tools from the loader instead of carrying inlined
    # specs. ``builtin_tools`` re-registers looplet-shipped tools (which
    # need loader-provided resources like ``workspace_config``);
    # ``mcp_servers`` re-spawns the MCP subprocess (whose tool executors
    # are closures that cannot serialise to disk).
    builtin_tool_names = list(getattr(preset, "builtin_tool_names", None) or [])
    if builtin_tool_names:
        serialized_cfg["builtin_tools"] = builtin_tool_names
    mcp_servers = dict(getattr(preset, "mcp_servers", None) or {})
    if mcp_servers:
        serialized_cfg["mcp_servers"] = _relativise_mcp_servers(mcp_servers, preset, root, warnings)

    if serialized_cfg:
        # ── Cartridge spec v2 split: contract → config.yaml,
        # runtime knobs → runtime.yaml. Field tiering lives on
        # CartridgeLayout (RUNTIME_TIER_FIELDS / contract_tier_fields).
        # The split is purely about file destination; both files
        # round-trip via the loader's merge in _load.py.
        runtime_keys = CartridgeLayout.RUNTIME_TIER_FIELDS
        runtime_payload = {k: v for k, v in serialized_cfg.items() if k in runtime_keys}
        contract_payload = {k: v for k, v in serialized_cfg.items() if k not in runtime_keys}

        if contract_payload:
            (root / CartridgeLayout.CONFIG_YAML).write_text(
                _dump_yaml(contract_payload) + "\n",
                encoding="utf-8",
            )
        if runtime_payload:
            (root / "runtime.yaml").write_text(
                _dump_yaml(runtime_payload) + "\n",
                encoding="utf-8",
            )

    # 2. system prompt
    prompts_dir = root / CartridgeLayout.PROMPTS_DIR
    prompts_dir.mkdir(exist_ok=True)
    (root / CartridgeLayout.SYSTEM_PROMPT_MD).write_text(
        getattr(cfg, "system_prompt", "") or "",
        encoding="utf-8",
    )

    # 3. tools — one subdir per tool with tool.yaml + execute.py
    tools_root = root / CartridgeLayout.TOOLS_DIR
    tools_root.mkdir(exist_ok=True)
    _builtin_names = set(builtin_tool_names)
    for spec in _iter_tool_specs(preset.tools):
        # Skip tools that are re-derived from declarative directives:
        # builtin tools come back via ``builtin_tools:`` and MCP tools via
        # ``mcp_servers:`` (their closure executors cannot round-trip).
        if getattr(spec, "name", None) in _builtin_names:
            continue
        if _is_mcp_tool(spec):
            continue
        _write_tool(spec, tools_root, warnings, strict)

    # 4. hooks — one subdir per hook, ordered by index for deterministic load
    hooks_root = root / CartridgeLayout.HOOKS_DIR
    hooks_root.mkdir(exist_ok=True)
    for idx, hook in enumerate(preset.hooks):
        _write_hook(hook, hooks_root, idx, warnings, strict)

    # 5. memory sources — three emission paths:
    #    StaticMemorySource              → memory/<idx>_static.md
    #    CallableMemorySource(top-level) → memory/<idx>_callable.py
    #    CallableMemorySource(_chw_resource_X.<lambda>)
    #                                    → config.yaml ``memory_sources:
    #                                      ["@X"]`` + resources/<X>.py
    #                                      (auto-emitted via the
    #                                      regular resource pipeline)
    # The third branch lets workspaces wire load-time-parameterised
    # memory through the same ``runtime=``-aware resource builder
    # mechanism the rest of the harness uses, instead of forcing a
    # ``setup.py`` detour for the ``CallableMemorySource(lambda ...)``
    # closure pattern.
    memory_root = root / CartridgeLayout.MEMORY_DIR
    memory_root.mkdir(exist_ok=True)
    memory_ref_entries: list[str] = []
    for idx, source in enumerate(getattr(cfg, "memory_sources", []) or []):
        ref_name = _write_memory(source, memory_root, idx, warnings, strict)
        if ref_name is not None:
            ref_string = f"{_REF_PREFIX}{ref_name}"
            memory_ref_entries.append(ref_string)
            # Stash a value the resource pipeline's pre-pass can trace
            # back to ``_chw_resource_<name>``. The wrapper itself
            # (CallableMemorySource) lives in ``looplet.memory`` so we
            # stash the wrapped fn whose ``__module__`` is the workspace
            # resource module — that's what the pre-pass uses to copy
            # the original ``resources/<name>.py`` file verbatim.
            from looplet.memory import CallableMemorySource  # noqa: PLC0415

            stash_value = source.fn if isinstance(source, CallableMemorySource) else source
            config_field_refs.setdefault(ref_name, stash_value)

    # When ``CallableMemorySource`` round-tripped via @ref, append the
    # ref strings to ``config.yaml``'s ``memory_sources`` list so the
    # loader's @ref resolution rebuilds them on load. File-based memory
    # is loaded first by the loader and these @refs append after it.
    if memory_ref_entries:
        existing = serialized_cfg.get("memory_sources") or []
        serialized_cfg["memory_sources"] = list(existing) + memory_ref_entries
        # Re-write config.yaml to include the new memory_sources entries.
        # Apply the same contract/runtime split as the initial write so
        # we don't accidentally re-introduce runtime keys into config.yaml.
        runtime_keys = CartridgeLayout.RUNTIME_TIER_FIELDS
        contract_payload = {k: v for k, v in serialized_cfg.items() if k not in runtime_keys}
        (root / CartridgeLayout.CONFIG_YAML).write_text(
            _dump_yaml(contract_payload) + "\n",
            encoding="utf-8",
        )

    # 6. resources — for any @<name> ref found in written hook configs,
    # emit a placeholder ``resources/<name>.py`` so the snapshot loads
    # without unresolved-reference errors. The placeholder returns the
    # actual instance the hook is currently bound to so the reload
    # behaves identically to the source preset (within process — across
    # processes the user must replace the placeholder with a real
    # builder).
    #
    # Also collect resources referenced by tool ``requires:`` lists so
    # the round-tripped cartridge stays self-contained. Without this,
    # a tool whose ``requires: [support_data]`` resolves at load time
    # via the source cartridge's ``resources/support_data.py`` ends up
    # without that resource on reload, and dispatch quietly receives
    # ``None`` from ``ctx.resources["support_data"]``.
    tool_required_refs: dict[str, Any] = {}
    source_resources = dict(getattr(preset, "resources", {}) or {})
    for spec in _iter_tool_specs(preset.tools):
        for req_name in getattr(spec, "requires", []) or []:
            if req_name in tool_required_refs:
                continue
            if req_name in ("runtime",):
                continue  # reserved; auto-injected from runtime dict
            inst = source_resources.get(req_name)
            if inst is not None:
                tool_required_refs[req_name] = inst
    if tool_required_refs:
        if config_field_refs is None:
            config_field_refs = {}
        for k, v in tool_required_refs.items():
            config_field_refs.setdefault(k, v)

    _write_resources_for_refs(
        hooks_root,
        root,
        preset.hooks,
        warnings,
        strict,
        extra_refs=config_field_refs,
    )

    # 7. workspace-root helper modules — when the preset was loaded
    # from another workspace, copy any top-level ``*.py`` helper files
    # (e.g. ``coder_lib_tools.py``, ``threat_intel_lib.py``) so the
    # snapshot stays self-contained. Without this, hooks / tools /
    # resources that import from these helpers crash on cross-process
    # reload with ``ModuleNotFoundError``.
    _copy_workspace_helpers(preset, root, warnings)

    workspace.serialization_warnings = warnings
    workspace.write_metadata()
    return workspace


def _is_mcp_tool(spec: Any) -> bool:
    """True when ``spec``'s executor is an MCP adapter closure.

    MCP tool specs are produced by
    :meth:`looplet.mcp.MCPToolAdapter.tools`, whose ``execute`` is a
    per-tool closure (``MCPToolAdapter._make_executor.<locals>...``).
    Such closures cannot serialise to disk, so the serialiser skips them
    and relies on the re-emitted ``mcp_servers:`` directive to recreate
    the tools on reload.
    """
    fn = getattr(spec, "execute", None)
    qualname = getattr(fn, "__qualname__", "") or ""
    return "MCPToolAdapter" in qualname and "_make_executor" in qualname


def _relativise_mcp_servers(
    mcp_servers: dict[str, Any],
    preset: Any,
    dest_root: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Re-template resolved ``mcp_servers`` commands back to
    ``${runtime.cartridge_root}`` and vendor any bundled server files.

    The loader resolves ``${runtime.cartridge_root}`` to the source
    cartridge's absolute path before storing the block on the preset. To
    make the round-tripped cartridge relocatable, we copy the referenced
    bundle directory (e.g. ``_server/``) into ``dest_root`` and rewrite
    the absolute origin path back to the ``${runtime.cartridge_root}``
    placeholder. When the preset has no recorded origin (built in-process
    from an absolute path), the command is emitted verbatim so reload
    still re-spawns the server from its original location.
    """
    src_root = _preset_origin_root(preset)
    out: dict[str, Any] = {}
    for name, cfg in mcp_servers.items():
        if not isinstance(cfg, dict) or src_root is None:
            out[name] = cfg
            continue
        cfg = dict(cfg)
        src_str = str(Path(src_root).resolve())
        command = cfg.get("command")
        if isinstance(command, str) and src_str in command:
            _copy_mcp_bundle(command, src_str, Path(src_root), dest_root, warnings)
            cfg["command"] = command.replace(src_str, "${runtime.cartridge_root}")
        out[name] = cfg
    return out


def _copy_mcp_bundle(
    command: str,
    src_str: str,
    src_root: Path,
    dest_root: Path,
    warnings: list[str],
) -> None:
    """Copy the top-level bundle component referenced by an MCP server
    command (e.g. the ``_server/`` directory holding ``calc.py``) from
    the source cartridge into ``dest_root``."""
    for token in command.split():
        if not token.startswith(src_str):
            continue
        rel = Path(token[len(src_str) :].lstrip("/\\"))
        if not rel.parts:
            continue
        top = rel.parts[0]
        src_path = src_root / top
        dest_path = dest_root / top
        if not src_path.exists() or dest_path.exists():
            continue
        try:
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path)
            else:
                dest_path.write_bytes(src_path.read_bytes())
        except OSError as exc:  # noqa: BLE001
            warnings.append(f"mcp bundle copy: could not copy {top!r}: {exc}")


def _copy_workspace_helpers(preset: Any, dest_root: Path, warnings: list[str]) -> None:
    """Copy top-level ``*.py`` helper modules from the preset's source
    workspace (if any) into ``dest_root``.

    The source workspace is looked up via :func:`_preset_origin_root`,
    populated by :func:`cartridge_to_preset` for every preset it
    returns. When the preset was built in-process (no recorded
    origin), this is a no-op — there's nothing to vendor.
    """
    src_root = _preset_origin_root(preset)
    if src_root is None:
        return
    src_root = Path(src_root)
    if not src_root.is_dir() or src_root.resolve() == dest_root.resolve():
        return

    for helper in sorted(src_root.glob("*.py")):
        if helper.name in {"__init__.py", CartridgeLayout.SETUP_PY}:
            continue
        dest_file = dest_root / helper.name
        if dest_file.exists():
            continue
        try:
            dest_file.write_text(helper.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            warnings.append(f"workspace-helper copy: could not copy {helper.name}: {exc}")


def _write_resources_for_refs(
    hooks_root: Path,
    root: Path,
    hooks: list[Any],
    warnings: list[str],
    strict: bool,
    *,
    extra_refs: dict[str, Any] | None = None,
) -> None:
    """Emit ``resources/<name>.py`` placeholders for every ``@<name>``
    ref found in hook configs, plus any caller-supplied
    ``extra_refs`` (used by the writer to auto-emit callable
    LoopConfig fields like ``compact_service``).

    The placeholder's ``build()`` returns a stashed module-level
    reference to the actual object the hook is bound to in the
    source preset. This makes in-process snapshot+reload
    round-trip cleanly (the same FileCache instance gets re-used).
    For cross-process distribution the user must replace the
    placeholder with a real builder (the comment header explains
    how).
    """
    # Walk every hook's to_config() output and collect unique @<name> refs
    # paired with the actual constructor-arg value the hook holds.
    refs_seen: dict[str, Any] = {}
    for hook in hooks:
        if not hasattr(hook, "to_config") or not callable(hook.to_config):
            continue
        try:
            cfg = hook.to_config()
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(cfg, dict):
            continue
        for kwarg_name, kwarg_value in cfg.items():
            if not (isinstance(kwarg_value, str) and kwarg_value.startswith(_REF_PREFIX)):
                continue
            ref_name = kwarg_value[len(_REF_PREFIX) :]
            if ref_name in refs_seen:
                continue
            # Pull the actual instance from the hook's matching attribute
            # (try public attr name, then mangled private convention).
            actual = getattr(hook, kwarg_name, None)
            if actual is None:
                actual = getattr(hook, f"_{kwarg_name}", None)
            refs_seen[ref_name] = actual

    # Caller-supplied refs (e.g. config.compact_service) win over hook
    # refs when names collide — the LoopConfig field is the
    # authoritative live instance for that key.
    if extra_refs:
        for ref_name, instance in extra_refs.items():
            refs_seen[ref_name] = instance

    if not refs_seen:
        return

    resources_dir = root / CartridgeLayout.RESOURCES_DIR
    resources_dir.mkdir(exist_ok=True)
    import sys as _sys  # noqa: PLC0415

    def _origin_chw_resource_file(value: Any) -> Path | None:
        """If ``value`` (or every callable element of a list/tuple)
        traces back to a single workspace resource module, return the
        on-disk source file. Two detection paths:

        1. **Identity registry** (set up by :func:`_load_resources`):
           covers stock-class instances like ``PermissionEngine``
           whose own ``__module__`` is the framework, not the
           workspace.
        2. **Module name fallback**: covers closures defined inside
           the resource module (``def build(): def fn(...): ...``)
           and custom classes declared in the resource file.

        The original ``resources/<name>.py`` is the canonical builder
        — copying it round-trips builds that compute lists / closures
        / nested objects without losing the original ``def build()``
        signature.
        """
        # Path 1: identity registry → look up by id(), then read the
        # registered name's source file from sys.modules.
        candidates: list[Any] = list(value) if isinstance(value, (list, tuple)) else [value]
        if not candidates:
            return None
        registered_name = _resource_origin.get(id(value))
        if registered_name is None and isinstance(value, (list, tuple)):
            registered_name = _resource_origin.get(id(candidates[0]))
        if registered_name is not None:
            chw_mod = _sys.modules.get(f"_chw_resource_{registered_name}")
            chw_file = getattr(chw_mod, "__file__", None) if chw_mod is not None else None
            if chw_file and Path(chw_file).is_file():
                return Path(chw_file)

        # Path 2: every candidate's __module__ is the same _chw_resource_*
        chw_modules: set[str] = set()
        for item in candidates:
            mod = getattr(item, "__module__", "") or type(item).__module__ or ""
            if not mod.startswith("_chw_resource_"):
                return None
            chw_modules.add(mod)
        if len(chw_modules) != 1:
            return None
        chw_mod = _sys.modules.get(next(iter(chw_modules)))
        chw_file = getattr(chw_mod, "__file__", None) if chw_mod is not None else None
        if chw_file and Path(chw_file).is_file():
            return Path(chw_file)
        return None

    for ref_name, instance in refs_seen.items():
        # ── Universal pre-pass: copy original resources/<X>.py when
        # the live instance came from a workspace's own resource. The
        # original builder has the exact ``def build(runtime=None)``
        # the loader expects; reproducing it from the live instance
        # via dataclass / class auto-emit can lose the runtime= path,
        # the closure state, or the original list factory.
        chw_src = _origin_chw_resource_file(instance)
        if chw_src is not None:
            (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(
                chw_src.read_text(encoding="utf-8"), encoding="utf-8"
            )
            continue
        if instance is None:
            msg = (
                f"resource {ref_name!r}: could not locate the live instance on "
                f"any hook; emitted a stub builder that returns None"
            )
            if strict:
                raise CartridgeSerializationError(msg)
            warnings.append(msg)
            stub = (
                f"# AUTOGENERATED PLACEHOLDER for {ref_name!r}.\n"
                f"# No live instance was found on any hook. Replace this\n"
                f"# stub with a real ``def build(runtime=None)`` that\n"
                f"# constructs the resource the hooks expect.\n"
                "def build(runtime=None):\n"
                "    return None\n"
            )
            (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(stub, encoding="utf-8")
            continue

        # Special case: list / tuple of top-level callables. Common
        # for ``EvalHook(evaluators=[...])`` and similar collector
        # patterns. Emit a builder that re-imports each callable by
        # ``module:qualname``. ``__main__`` callables are accepted
        # (so script-driven dogfooding round-trips) but recorded as
        # a non-fatal warning in strict mode — cross-process loads
        # need the callables moved to a real module.
        if (
            isinstance(instance, (list, tuple))
            and instance
            and all(
                callable(item)
                and getattr(item, "__module__", "") not in ("", "builtins")
                and not getattr(item, "__module__", "").startswith("_chw_")
                and getattr(item, "__qualname__", "<lambda>") != "<lambda>"
                and "<locals>" not in getattr(item, "__qualname__", "<locals>")
                for item in instance
            )
        ):
            import_lines: list[str] = []
            ref_names: list[str] = []
            main_callables: list[str] = []
            for item in instance:
                mod = item.__module__
                fname = item.__name__
                import_lines.append(f"from {mod} import {fname}")
                ref_names.append(fname)
                if mod == "__main__":
                    main_callables.append(fname)
            if main_callables:
                # Non-fatal in strict mode: best-effort same-process
                # round-trip works, cross-process needs editing.
                warnings.append(
                    f"resource {ref_name!r}: contains __main__ callables "
                    f"{main_callables} — re-imports from ``__main__`` for "
                    f"same-process round-trip but cross-process loads will "
                    f"fail until these are moved to an importable module"
                )
            joined_imports = "\n".join(import_lines)
            joined_refs = ", ".join(ref_names)
            container = "list" if isinstance(instance, list) else "tuple"
            if container == "list":
                stub = (
                    f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                    f"# Returns a fresh list of top-level callables\n"
                    f"# re-imported from their original modules. If the\n"
                    f"# source preset depended on closures or instance state\n"
                    f"# carried by the callables, replace each entry with\n"
                    f"# the real construction.\n"
                    f"{joined_imports}\n"
                    "\n"
                    "def build(runtime=None):\n"
                    f"    return [{joined_refs}]\n"
                )
            else:
                stub = (
                    f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                    f"# Returns a fresh tuple of top-level callables\n"
                    f"# re-imported from their original modules.\n"
                    f"{joined_imports}\n"
                    "\n"
                    "def build(runtime=None):\n"
                    f"    return ({joined_refs},)\n"
                )
            (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(stub, encoding="utf-8")
            continue
        cls_module = type(instance).__module__ or ""
        cls_name = type(instance).__name__

        # Dataclass auto-emit: when the live instance is a dataclass we
        # reproduce its full field state in the builder (not just the
        # required ctor args). Common shape: ``PermissionEngine(rules=[
        # PermissionRule(...), ...])`` — the rules list is JSON-able-ish
        # only after we re-emit each PermissionRule's class import.
        # Falls through to the generic class branch when reproduction
        # fails (e.g. a field holds a closure).
        import dataclasses as _dc  # noqa: PLC0415

        if (
            _dc.is_dataclass(instance)
            and cls_module
            and cls_module not in {"__main__", "builtins"}
            and not cls_module.startswith("_chw_")
        ):
            try:
                imports_set: set[str] = {f"from {cls_module} import {cls_name}"}
                kwargs_src = _render_dataclass_kwargs(instance, imports_set)
            except _DataclassReprFailed as exc:
                kwargs_src = None
                logger.debug("dataclass auto-emit fell through for %s: %s", ref_name, exc)
            if kwargs_src is not None:
                joined_imports = "\n".join(sorted(imports_set))
                stub = (
                    f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                    f"# Reproduces the live ``{cls_name}`` instance field-by-\n"
                    f"# field. Replace any value here with a real construction\n"
                    f"# if you need different behaviour at load time.\n"
                    f"{joined_imports}\n"
                    "\n"
                    "def build(runtime=None):\n"
                    f"    return {cls_name}({kwargs_src})\n"
                )
                (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(
                    stub, encoding="utf-8"
                )
                continue

        # Best-effort: for installed classes, emit a builder that
        # imports the class and returns a fresh instance. This loses
        # cross-process state — explain that in the header.
        if (
            cls_module
            and cls_module not in {"__main__", "builtins"}
            and not cls_module.startswith("_chw_")
        ):
            # Inspect the constructor so we know which kwargs to pass —
            # FileCache(workspace=...) needs ``workspace`` from runtime,
            # while StreamingHook(emitter=...) won't survive at all.
            try:
                ctor_sig = inspect.signature(type(instance).__init__)
                required: list[str] = []
                for p_name, p in ctor_sig.parameters.items():
                    if p_name == "self":
                        continue
                    if p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    if p.default is inspect.Parameter.empty:
                        required.append(p_name)
            except (TypeError, ValueError):
                required = []

            ctor_kwargs_parts: list[str] = []
            extra_imports: list[str] = []
            best_effort_warnings: list[str] = []
            for kw in required:
                # Try to derive the kwarg from the live instance's
                # matching attribute (public ``kw`` first, then mangled
                # private ``_kw``). When the attr is a top-level
                # importable callable, generate a real ``from M import
                # F`` so the builder reproduces the original wiring
                # exactly. Otherwise fall back to ``runtime.get(kw)``
                # so the user can still inject the value at load time.
                live = getattr(instance, kw, None)
                if live is None:
                    live = getattr(instance, f"_{kw}", None)
                live_mod = getattr(live, "__module__", "") or ""
                live_name = getattr(live, "__name__", "")
                live_qual = getattr(live, "__qualname__", "<lambda>")
                if (
                    callable(live)
                    and live_mod
                    and live_mod not in ("builtins", "_chw_")
                    and not live_mod.startswith("_chw_")
                    and live_name
                    and live_qual != "<lambda>"
                    and "<locals>" not in live_qual
                ):
                    extra_imports.append(f"from {live_mod} import {live_name}")
                    ctor_kwargs_parts.append(f"{kw}={live_name}")
                    if live_mod == "__main__":
                        best_effort_warnings.append(
                            f"resource {ref_name!r}: kwarg {kw!r} re-imports "
                            f"from ``__main__`` ({live_name!r}); cross-process "
                            f"loads will fail until moved to a real module"
                        )
                else:
                    ctor_kwargs_parts.append(f"{kw}=runtime.get({kw!r})")

            if best_effort_warnings:
                warnings.extend(best_effort_warnings)
            ctor_kwargs = ", ".join(ctor_kwargs_parts)
            extra_import_block = ("\n".join(extra_imports) + "\n") if extra_imports else ""
            stub = (
                f"# AUTOGENERATED resource builder for {ref_name!r}.\n"
                f"# Returns a FRESH {cls_name} instance from {cls_module} on\n"
                f"# every workspace load. Required ctor kwargs are derived from\n"
                f"# the live source instance when possible (top-level callables\n"
                f"# get re-imported); the rest fall back to ``runtime.get(<kw>)``.\n"
                f"# Replace any unresolved kwargs with real construction before\n"
                f"# distributing the workspace.\n"
                f"from {cls_module} import {cls_name}\n"
                f"{extra_import_block}"
                "\n"
                "def build(runtime=None):\n"
                "    runtime = runtime or {}\n"
                f"    return {cls_name}({ctor_kwargs})\n"
            )
        else:
            msg = (
                f"resource {ref_name!r}: live instance class {cls_name!r} from "
                f"module {cls_module!r} is not importable; emitted a None-stub "
                f"builder \u2014 replace ``resources/{_safe_filename(ref_name)}.py`` "
                f"with a real builder before loading in a new process"
            )
            if strict:
                raise CartridgeSerializationError(msg)
            warnings.append(msg)
            stub = (
                f"# AUTOGENERATED PLACEHOLDER for {ref_name!r} (instance class\n"
                f"# {cls_name!r} from non-importable module {cls_module!r}).\n"
                f"# Replace with a real builder.\n"
                "def build(runtime=None):\n"
                "    return None\n"
            )
        (resources_dir / f"{_safe_filename(ref_name)}.py").write_text(stub, encoding="utf-8")


def _iter_tool_specs(tools: "BaseToolRegistry") -> Iterable[Any]:
    if hasattr(tools, "_tools"):
        return list(tools._tools.values())  # type: ignore[attr-defined]
    if hasattr(tools, "_specs"):
        return list(tools._specs.values())  # type: ignore[attr-defined]
    if hasattr(tools, "specs"):
        return list(tools.specs())  # type: ignore[attr-defined,operator]
    raise CartridgeSerializationError(
        f"tool registry {type(tools).__name__!r} does not expose tool specs"
    )


def _write_tool(spec: Any, tools_root: Path, warnings: list[str], strict: bool) -> None:
    name = spec.name
    tool_dir = tools_root / _safe_filename(name)
    tool_dir.mkdir(parents=True, exist_ok=True)

    yaml_payload: dict[str, Any] = {
        "name": name,
        "description": spec.description,
        "parameters": dict(spec.parameters or {}),
    }
    # Promote multi-paragraph descriptions to ``description.md`` so
    # the on-disk artefact stays legible and yaml-block-scalar-free
    # (paper §"Promoted to first-class"). Single-line descriptions
    # stay in ``tool.yaml``. The loader prefers ``description.md``
    # when both exist.
    description_text = str(spec.description or "")
    if "\n" in description_text.strip():
        (tool_dir / "description.md").write_text(description_text.rstrip() + "\n", encoding="utf-8")
        yaml_payload["description"] = description_text.splitlines()[0]
    for opt in ("concurrent_safe", "free", "timeout_s"):
        if hasattr(spec, opt):
            val = getattr(spec, opt)
            if val is not None:
                yaml_payload[opt] = val
    # Round-trip the resource-requirements list (workspace tool DI).
    requires = getattr(spec, "requires", None) or []
    if requires:
        yaml_payload["requires"] = list(requires)
    # Round-trip v1.1 advisory metadata (tags + render hints). Both
    # default to empty; emit only when non-empty so absent metadata
    # round-trips as absent (not as ``tags: []``).
    tags = getattr(spec, "tags", None) or []
    if tags:
        yaml_payload["tags"] = list(tags)
    render = getattr(spec, "render", None) or {}
    if render:
        yaml_payload["render"] = dict(render)
    (tool_dir / "tool.yaml").write_text(_dump_yaml(yaml_payload) + "\n", encoding="utf-8")

    fn = spec.execute
    qualname = getattr(fn, "__qualname__", "<lambda>")
    if "<locals>" in qualname or qualname == "<lambda>":
        msg = (
            f"tool {name!r} execute is a closure or lambda ({qualname}); cannot round-trip to disk"
        )
        if strict:
            raise CartridgeSerializationError(msg)
        warnings.append(msg)
        (tool_dir / "execute.py").write_text(
            "# AUTOGENERATED PLACEHOLDER\n"
            "# Original tool.execute was a closure/lambda and could not be\n"
            "# serialised. Re-implement here as a top-level ``execute`` function.\n"
            "def execute(**kwargs):\n"
            "    raise NotImplementedError('replace this stub')\n",
            encoding="utf-8",
        )
        return

    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        msg = f"tool {name!r} execute has no retrievable source"
        if strict:
            raise CartridgeSerializationError(msg)
        warnings.append(msg)
        return

    # Prefer re-importing the original function so its enclosing module's
    # imports stay in scope (typing.Any, helper functions, etc.). Falls
    # back to source-dump only when the function isn't importable.
    fn_name = getattr(fn, "__name__", "")
    module_name = getattr(fn, "__module__", "") or ""

    # Cartridge-loaded tools live in dynamic ``_chw_tool_<name>`` modules
    # registered in ``sys.modules``. Re-importing by that synthetic name
    # would silently fail at reload (the new process has no entry under
    # that key). Copy the original on-disk ``execute.py`` verbatim — it
    # already carries the correct shim or top-level function.
    if module_name.startswith("_chw_tool_"):
        import sys as _sys  # noqa: PLC0415

        chw_mod = _sys.modules.get(module_name)
        chw_file = getattr(chw_mod, "__file__", None) if chw_mod is not None else None
        if chw_file and Path(chw_file).is_file():
            (tool_dir / "execute.py").write_text(
                Path(chw_file).read_text(encoding="utf-8"), encoding="utf-8"
            )
            return

    if fn_name and module_name and module_name not in {"__main__", "builtins"}:
        try:
            mod = importlib.import_module(module_name)
            if getattr(mod, fn_name, None) is fn:
                (tool_dir / "execute.py").write_text(
                    "# AUTOGENERATED from preset_to_cartridge.\n"
                    "# Re-imported from the original module so the function's\n"
                    "# closure (typing imports, helpers) stays available.\n"
                    f"from {module_name} import {fn_name} as execute\n",
                    encoding="utf-8",
                )
                return
        except Exception:  # noqa: BLE001
            pass

    # Fallback: dump source. Add an `execute = <fn_name>` alias so the
    # loader finds it under the canonical name regardless of what the
    # original function was called.
    alias_line = f"execute = {fn_name}\n" if fn_name and fn_name != "execute" else ""
    (tool_dir / "execute.py").write_text(
        f"# AUTOGENERATED from preset_to_cartridge.\n{source}\n{alias_line}",
        encoding="utf-8",
    )


def _write_hook(hook: Any, hooks_root: Path, index: int, warnings: list[str], strict: bool) -> None:
    # ── kind: lep — out-of-process hook serialises to declarative data ──
    # An LEPHookAdapter carries no Python decision logic; its faithful
    # cartridge form is the launch command + declared view + failure
    # policy. Emitting that (instead of trying to pickle the adapter as a
    # class hook) is what makes the library→cartridge direction lossless
    # for out-of-process hooks (HOOK_CARTRIDGE_DESIGN.md §5.1).
    from looplet.lep import LEPHookAdapter  # noqa: PLC0415

    if isinstance(hook, LEPHookAdapter):
        # The loader sets ``_cartridge_id`` to the source hook dir name,
        # which already carries an ``NN_`` index prefix. Strip it before
        # re-prefixing so the dir name does not compound (``01_NameGuard``
        # → ``01_01_NameGuard`` → …) across successive round-trips.
        base_id = re.sub(r"^\d+_", "", hook._cartridge_id or "lep") or "lep"
        dir_name = f"{index:02d}_{_safe_filename(base_id)}"
        hook_dir = hooks_root / dir_name
        hook_dir.mkdir(parents=True, exist_ok=True)
        # Make the snapshot self-contained: any argv token *after* the
        # program/interpreter (argv[0]) that is an existing file (the
        # server script + its local helpers) is copied into the hook dir
        # and rewritten to a bare filename, which the loader resolves
        # relative to the hook dir. argv[0] (the interpreter/executable)
        # and non-file tokens (flags) pass through verbatim — copying the
        # interpreter binary would break it outside its install tree.
        emitted_cmd: list[str] = []
        for idx, tok in enumerate(hook._argv):
            src_path = Path(str(tok))
            if idx > 0 and src_path.is_file():
                dest = hook_dir / src_path.name
                try:
                    shutil.copy2(src_path, dest)
                    emitted_cmd.append(src_path.name)
                    continue
                except OSError as exc:  # pragma: no cover - defensive
                    warnings.append(f"lep hook server copy failed for {tok}: {exc}")
            emitted_cmd.append(str(tok))
        lep_cfg: dict[str, Any] = {
            "kind": "lep",
            "command": emitted_cmd,
            "view": hook._view.to_dict(),
            "on_failure": hook._on_failure,
            "run_id": hook._run_id,
        }
        (hook_dir / "config.yaml").write_text(_dump_yaml(lep_cfg) + "\n", encoding="utf-8")
        return

    cls = _hook_class(hook)
    cls_name = cls.__name__
    dir_name = f"{index:02d}_{_safe_filename(cls_name)}"
    hook_dir = hooks_root / dir_name
    hook_dir.mkdir(parents=True, exist_ok=True)

    # Strategy: write a hook.py that **re-imports** the original class from
    # its module by default, so the class's full closure (typing imports,
    # helpers, sibling utilities) stays intact. ``inspect.getsource(cls)``
    # alone returns just the class body — names like ``Any`` and the
    # tool-call decision helpers vanish on reload.
    #
    # When the original module is not importable from disk (anonymous /
    # closure / dynamically-defined class), fall back to the full module
    # source — which is heavier but preserves correctness.
    src = _render_hook_source(cls, warnings, strict)
    # Only prepend the AUTOGENERATED header when the rendered source
    # doesn't already carry it — branch 0 of ``_render_hook_source``
    # copies an existing source file verbatim and we don't want the
    # header to stack across repeat round-trips.
    header = "# AUTOGENERATED from preset_to_cartridge.\n"
    if src.startswith(header):
        body = src
    else:
        body = f"{header}{src}\n"
    (hook_dir / "hook.py").write_text(body, encoding="utf-8")

    # Constructor kwargs: prefer hook.to_config(); else introspect
    # the __init__ signature and read matching attributes off the
    # instance. Without this, hooks that take resource kwargs (the
    # canonical v1.1 declarative pattern) lose all their configuration
    # on round-trip because the writer falls back to ``kwargs: {}``.
    cfg_payload: dict[str, Any] = {"class_name": cls_name}
    if hasattr(hook, "to_config") and callable(hook.to_config):
        try:
            cfg_payload["kwargs"] = hook.to_config()
        except Exception as exc:  # noqa: BLE001
            msg = f"hook {cls_name!r}.to_config() raised: {exc!r}"
            if strict:
                raise CartridgeSerializationError(msg) from exc
            warnings.append(msg)
            cfg_payload["kwargs"] = {}
    else:
        cfg_payload["kwargs"] = _infer_hook_kwargs_from_init(hook)

    (hook_dir / "config.yaml").write_text(_dump_yaml(cfg_payload) + "\n", encoding="utf-8")


def _infer_hook_kwargs_from_init(hook: Any) -> dict[str, Any]:
    """Infer ``kwargs`` for a hook config by introspecting its
    ``__init__`` signature and reading matching attributes off the
    instance.

    For each non-self parameter, look for an attribute on the instance
    in this order: ``self.<name>``, ``self._<name>``. Resource-typed
    values (objects produced by a ``resources/<name>.py`` builder) are
    re-emitted as their original ``@<name>`` ref so the round-tripped
    cartridge resolves them via the same mechanism.

    Skips parameters whose value couldn't be located (defensive — the
    user can fix the hook config manually if needed).
    """
    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(type(hook).__init__)
    except (TypeError, ValueError):
        return kwargs
    for p_name, p in sig.parameters.items():
        if p_name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        # Find the value on the instance.
        if hasattr(hook, p_name):
            value = getattr(hook, p_name)
        elif hasattr(hook, f"_{p_name}"):
            value = getattr(hook, f"_{p_name}")
        else:
            continue  # can't recover; skip
        # Skip values equal to the parameter's default (keeps the
        # round-tripped config minimal).
        if p.default is not inspect.Parameter.empty and value is p.default:
            continue
        # Resource-typed value? Emit as @<name> ref.
        ref = resource_ref_for(value)
        if ref is not None:
            kwargs[p_name] = ref
            continue
        # Plain JSON-serialisable scalar? Pass through.
        if value is None or isinstance(value, (bool, int, float, str)):
            kwargs[p_name] = value
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(x, (bool, int, float, str)) for x in value
        ):
            kwargs[p_name] = list(value)
            continue
        if isinstance(value, dict) and all(
            isinstance(v, (bool, int, float, str, type(None))) for v in value.values()
        ):
            kwargs[p_name] = dict(value)
            continue
        # Otherwise: skip (can't safely round-trip). The user can
        # supply a manual config or implement to_config().
    return kwargs


def _render_hook_source(cls: type, warnings: list[str], strict: bool) -> str:
    """Render a self-contained hook.py source for ``cls``.

    Preference order:
      0. Loaded from a workspace ``_chw_hook_*`` dynamic module whose
         source file is on disk → copy that file verbatim. Preserves
         workspace-local subclasses (e.g. one that adds ``to_config()``)
         that the MRO-walk fallback would silently drop by aliasing
         the importable base class.
      1. Importable module → re-import the class by name.
      2. Loaded from a workspace ``_chw_hook_*`` dynamic module with no
         on-disk source → walk the MRO to find an importable base class
         and re-import that.
      3. Module source available → dump full module (preserves imports).
      4. Class source only → dump source with a typing-import fallback.
    """
    cls_name = cls.__name__
    module_name = cls.__module__ or ""

    # 0. Cartridge ``_chw_hook_*`` source on disk → copy verbatim. This
    #    is essential for byte-identity round-trip and for preserving
    #    any methods (commonly ``to_config()``) the workspace-local
    #    subclass adds beyond an installed base class.
    if module_name.startswith("_chw_hook_"):
        import sys as _sys  # noqa: PLC0415

        chw_mod = _sys.modules.get(module_name)
        chw_file = getattr(chw_mod, "__file__", None) if chw_mod is not None else None
        if chw_file and Path(chw_file).is_file():
            return Path(chw_file).read_text(encoding="utf-8")

    # 1. Try to re-import — works for installed packages, top-level classes
    #    in importable modules, and anything addressable by ``module:name``.
    if (
        module_name
        and module_name not in {"__main__", "builtins"}
        and not module_name.startswith("_chw_")
    ):
        try:
            mod = importlib.import_module(module_name)
            if getattr(mod, cls_name, None) is cls:
                # Canonical "as <cls_name>" form so this branch produces
                # byte-identical output to the MRO-walk branch below —
                # the snapshot writer needs to be idempotent under repeat
                # round-trips, and that requires both code paths to emit
                # the same source for the same hook class.
                return (
                    "# Re-imported from the original module so the class's\n"
                    "# full closure (typing imports, helpers) stays available.\n"
                    "# Edit this file (or replace the import with a class\n"
                    "# definition) to customise the hook in this workspace.\n"
                    f"from {module_name} import {cls_name} as {cls_name}\n"
                )
        except Exception:  # noqa: BLE001
            pass

    # 2. Class came from a workspace ``_chw_hook_*`` dynamic module
    #    (re-loaded from disk). Walk the MRO to find an importable
    #    parent class — workspace hook files commonly subclass an
    #    installed class to add ``to_config()``, so the ancestor is
    #    a stable re-import target.
    for ancestor in cls.__mro__[1:]:
        anc_module = ancestor.__module__ or ""
        anc_name = ancestor.__name__
        if (
            not anc_module
            or anc_module in {"__main__", "builtins", "object"}
            or anc_module.startswith("_chw_")
        ):
            continue
        try:
            mod = importlib.import_module(anc_module)
            if getattr(mod, anc_name, None) is ancestor:
                # Same canonical form as branch 1 — see comment there.
                return (
                    "# Re-imported from the original module so the class's\n"
                    "# full closure (typing imports, helpers) stays available.\n"
                    "# Edit this file (or replace the import with a class\n"
                    "# definition) to customise the hook in this workspace.\n"
                    f"from {anc_module} import {anc_name} as {cls_name}\n"
                )
        except Exception:  # noqa: BLE001
            continue

    # 3. Fall back to dumping the FULL module source (captures imports,
    #    sibling helpers). Heavier but correct for hooks defined inline
    #    in a script the user might still want to edit on disk.
    try:
        mod = inspect.getmodule(cls)
        if mod is not None:
            mod_src = inspect.getsource(mod)
            return (
                "# Full module source preserved so all imports / helpers\n"
                "# the hook references stay available on reload.\n"
                f"{mod_src}"
            )
    except (OSError, TypeError):
        pass

    # 4. Last resort: just the class source. Likely needs hand-editing on
    #    reload to add `from typing import Any` etc.
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        # ``TypeError: <class X> is a built-in class`` is what
        # ``inspect.getsource`` raises for classes loaded from the
        # workspace's own ``_chw_hook_*`` dynamic modules — those
        # have no on-disk source ``inspect`` can find.
        msg = f"hook class {cls_name!r} has no retrievable source"
        if strict:
            raise CartridgeSerializationError(msg)
        warnings.append(msg)
        return f"# AUTOGENERATED PLACEHOLDER\nclass {cls_name}:\n    pass\n"


def _write_memory(
    source: Any, memory_root: Path, index: int, warnings: list[str], strict: bool
) -> str | None:
    """Emit a memory source to disk.

    Returns ``None`` when the source was written as a file in
    ``memory/`` (StaticMemorySource → ``*.md``, top-level
    CallableMemorySource → ``*.py``). Returns a resource ref name
    (e.g. ``"project_memory"``) when the wrapped fn came from a
    workspace-loaded ``_chw_resource_<name>`` module — the caller adds
    the corresponding ``"@<name>"`` entry to ``config.yaml``'s
    ``memory_sources`` list and the resource auto-emit machinery
    copies the original on-disk source verbatim.
    """
    from looplet.memory import (  # noqa: PLC0415
        CallableMemorySource,
        StaticMemorySource,
    )

    if isinstance(source, StaticMemorySource):
        (memory_root / f"{index:02d}_static.md").write_text(source.text, encoding="utf-8")
        return None
    # CallableMemorySource: three branches in priority order.
    if isinstance(source, CallableMemorySource):
        fn = source.fn
        fn_name = getattr(fn, "__name__", "")
        fn_mod = getattr(fn, "__module__", "") or ""
        fn_qual = getattr(fn, "__qualname__", "<lambda>")

        # Branch A: fn was created inside a workspace-loaded
        # ``_chw_resource_<name>`` builder. The resource pipeline
        # already knows how to copy the original ``resources/<name>.py``
        # file verbatim — route through @ref to reuse that machinery
        # and to keep the closure (which closes over runtime values
        # like ``workspace`` / ``max_steps``) intact across reload.
        if fn_mod.startswith("_chw_resource_"):
            return fn_mod[len("_chw_resource_") :]

        # Branch B: fn is a top-level importable function. Emit a
        # ``<idx>_callable.py`` shim that re-imports it.
        if (
            fn_name
            and fn_mod
            and fn_mod not in ("builtins",)
            and not fn_mod.startswith("_chw_")
            and fn_qual != "<lambda>"
            and "<locals>" not in fn_qual
        ):
            (memory_root / f"{index:02d}_callable.py").write_text(
                "# AUTOGENERATED CallableMemorySource builder.\n"
                "# The exported ``load`` callable receives the loop's\n"
                "# ``state`` on every turn and returns the memory text\n"
                "# (or ``None`` to skip). Re-imported from the source\n"
                "# module so its closure stays intact.\n"
                f"from {fn_mod} import {fn_name} as load\n",
                encoding="utf-8",
            )
            if fn_mod == "__main__":
                warnings.append(
                    f"memory source {index!r}: CallableMemorySource wraps "
                    f"a ``__main__`` callable {fn_name!r}; cross-process "
                    f"loads will fail until it is moved to a real module"
                )
            return None

        # Branch C: lambda / closure / non-importable. Warn and skip —
        # the caller should move the closure into a
        # ``resources/<name>.py`` builder so branch A can route it.
        msg = (
            f"memory source 'CallableMemorySource' wraps a non-importable "
            f"callable ({fn_qual!r} from {fn_mod!r}); skipping. Move the "
            f"closure into a ``resources/<name>.py`` builder and reference "
            f"it from ``config.yaml`` via ``memory_sources: ['@<name>']`` "
            f"to make it round-trip cleanly."
        )
        if strict:
            raise CartridgeSerializationError(msg)
        warnings.append(msg)
        return None

    name = type(source).__name__
    msg = f"memory source {name!r} is not a StaticMemorySource; skipping"
    if strict:
        raise CartridgeSerializationError(msg)
    warnings.append(msg)
    return None
