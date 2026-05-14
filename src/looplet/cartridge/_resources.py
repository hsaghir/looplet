"""Resource registry, reference resolution, and the v1.1 single-file tool loader.

Owns the four reference-resolution mechanisms shared across the
loader / serialiser:

* :func:`_load_resources` — build the per-run resource registry by
  importing every ``resources/<name>.py`` and calling its
  ``build()``. Each instance is registered in
  :data:`looplet.refs._resource_origin` so hooks / writers can
  recover the original ``@<name>`` ref later.
* :func:`_resolve_refs` — walk a YAML-loaded dict / list / scalar
  and replace every ``@name`` / ``${ref:name}`` /
  ``${py:mod:sym}`` / ``${runtime.x}`` token with the live
  object the cartridge author meant.
* :func:`_resolve_py_ref` and :func:`_resolve_runtime_ref` —
  the per-form helpers `_resolve_refs` delegates into for
  ``${py:...}`` and ``${runtime.x}`` respectively.
* :func:`_load_single_file_tool` — load the v1.1
  ``tools/<name>.py`` form (module-level dunders + ``execute``
  callable) into a :class:`ToolSpec`.
* :func:`_coerce_default` — best-effort cast of a runtime default
  text into a typed value.
"""

from __future__ import annotations

import importlib
import inspect as _inspect
import logging
import re
from pathlib import Path
from typing import Any

from looplet.cartridge._imports import _import_module_from_path
from looplet.cartridge._layout import CartridgeLayout, CartridgeSerializationError
from looplet.refs import _REF_PREFIX, _register_resource_origin

logger = logging.getLogger(__name__)


def _load_resources(root: Path, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the shared-resource registry from ``resources/<name>.py`` files.

    Each resource module must define a builder named ``build``. The
    loader inspects the signature: ``build()`` (zero-arg) is the
    short form; ``build(runtime)`` lets the resource read the
    host-supplied ``runtime`` dict (e.g. ``runtime['workspace']`` for
    the coder workspace). Resources are shared by every ``"@<name>"``
    reference in hook / tool kwargs.

    Each built instance is also registered in
    :data:`_resource_origin` (keyed by ``id``, with a finalizer that
    drops the entry when the instance is garbage-collected) so that
    :func:`resource_ref_for` can recover the original ref name on
    workspace round-trip — even when the instance's class lives in a
    third-party module (e.g. ``looplet.permissions.PermissionEngine``)
    rather than the workspace's own resource module.

    Reserved key: ``"runtime"`` is auto-injected with the host-supplied
    runtime dict so tools can ``requires: [runtime]`` and read
    ``ctx.resources["runtime"]`` directly, without the boilerplate of
    a one-line ``resources/runtime.py`` builder. A workspace that
    ships its own ``resources/runtime.py`` overrides this default
    (the explicit file wins).
    """

    runtime_dict = dict(runtime or {})
    # Pre-seed the reserved ``runtime`` resource. Workspace-defined
    # ``resources/runtime.py`` (if any) overwrites this below.
    resources: dict[str, Any] = {"runtime": runtime_dict}

    resources_dir = root / CartridgeLayout.RESOURCES_DIR
    if not resources_dir.is_dir():
        return resources
    for resource_file in sorted(resources_dir.glob("*.py")):
        name = resource_file.stem
        module = _import_module_from_path(resource_file, f"_chw_resource_{name}")
        builder = getattr(module, "build", None)
        if not callable(builder):
            raise CartridgeSerializationError(
                f"resource {name!r} ({resource_file}) must define `def build() -> Any`"
            )
        # Pass runtime only when the builder accepts it, so the
        # zero-arg short form keeps working unchanged.
        try:
            sig = _inspect.signature(builder)
            accepts_runtime = "runtime" in sig.parameters or any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_runtime = False
        instance = builder(runtime=runtime_dict) if accepts_runtime else builder()
        resources[name] = instance
        _register_resource_origin(instance, name)
        # Cartridge spec v2: resources may declare ``THREAD_SAFE = True``
        # / ``False`` at module level. The loader stashes this in a
        # parallel registry keyed under the reserved name
        # ``_resource_thread_safety`` so the runtime can refuse
        # ``concurrent_dispatch`` of tools whose ``requires:`` includes
        # an unsafe resource. Resources that don't declare a value are
        # treated as "unknown" (stricter hosts can choose to fail; the
        # default runtime behaviour is to allow with a warning).
        thread_safe = getattr(module, "THREAD_SAFE", None)
        if thread_safe is not None:
            if not isinstance(thread_safe, bool):
                raise CartridgeSerializationError(
                    f"resource {name!r} ({resource_file}) declares "
                    f"THREAD_SAFE={thread_safe!r}; must be ``True`` or ``False``"
                )
            resources.setdefault("_resource_thread_safety", {})[name] = thread_safe
    return resources


# Identity-keyed map from a resource-built instance's ``id()`` to its
# ref name. Keyed by ``id`` because the instance may be unhashable (a
# list, a dict) or its hash semantics may collide with another
# instance. Entries are dropped on instance GC via ``weakref.finalize``;
# instances that don't support weak references (built-in containers
# like ``list`` / ``tuple`` / ``dict``) are NOT registered here at all
# because their ``id()`` would be reused by a later same-typed object
# whose source bears no relation to this resource. The
# ``__module__``-based fallback in :func:`resource_ref_for` covers
# those cases (closures created inside the resource module's
# ``build()`` carry the synthetic ``_chw_resource_<name>`` module).
# ``_REF_PREFIX``, ``_resource_origin``, ``_register_resource_origin`` and
# ``resource_ref_for`` live in :mod:`looplet.refs` so that core modules
# (permissions, streaming, evals, telemetry) can call ``resource_ref_for``
# without importing this 3000-line workspace loader. Re-exported here for
# backwards compatibility — existing tests reach
# ``looplet.workspace._resource_origin`` directly, and that still works.
from looplet.refs import (  # noqa: E402, F401, I001, PLC0415
    _resource_origin,
)


# Unified ``${kind:value}`` reference grammar — applied to every
# string value the workspace loader resolves. Three kinds today:
#
#   ${ref:name}           → looked up in the resources registry
#   ${py:module:symbol}   → importlib.import_module + getattr
#   ${runtime.field}      → looked up in the per-invocation runtime dict
#
# Anything not matching this pattern (and not the legacy ``@name``
# form) passes through unchanged. The grammar is intentionally
# *closed* — there is no escape hatch for arbitrary expressions; if
# a workspace needs imperative wiring it falls back to ``setup.py``.
#
# Same syntax for ``${runtime.x}`` as ``_apply_runtime_substitutions``
# uses on raw text below; this resolver handles the structured-value
# path (preserves the resolved object's identity instead of stringifying).
_REF_PATTERN = re.compile(r"^\$\{(?P<kind>ref|py|runtime)(?::|\.)(?P<value>[^}]+)\}$")


def _resolve_py_ref(spec: str) -> Any:
    """Resolve a ``module:symbol`` (or ``module.symbol``) string to a Python object.

    Accepts both colon and dot as the module/attribute separator on
    the rightmost split. ``module`` may itself contain dots
    (``a.b.c:Class``). Symbol may be nested via dots
    (``module:Class.factory``).
    """
    if ":" in spec:
        module_path, _, attr_path = spec.partition(":")
    elif "." in spec:
        module_path, _, attr_path = spec.rpartition(".")
    else:
        raise CartridgeSerializationError(
            f"py: reference {spec!r} must be 'module:symbol' or 'module.symbol'"
        )
    if not module_path or not attr_path:
        raise CartridgeSerializationError(f"py: reference {spec!r} has empty module or symbol part")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise CartridgeSerializationError(
            f"py: reference {spec!r} — could not import {module_path!r}: {exc}"
        ) from exc
    obj: Any = module
    for attr in attr_path.split("."):
        if not hasattr(obj, attr):
            raise CartridgeSerializationError(
                f"py: reference {spec!r} — {attr!r} not found on {obj!r}"
            )
        obj = getattr(obj, attr)
    return obj


def _resolve_runtime_ref(field: str, runtime: dict[str, Any] | None) -> Any:
    """Resolve a ``${runtime.field}`` reference. Field may use
    ``a.b`` dotted lookup into nested dicts.

    Supports a default with ``:-`` syntax: ``${runtime.x:-default}``
    (matches the shell convention; consistent with text-mode
    substitution conventions).
    """
    # Default form: "field:-default"
    default_marker = ":-"
    has_default = default_marker in field
    default_str: str | None = None
    if has_default:
        field, _, default_str = field.partition(default_marker)
    runtime = runtime or {}
    parts = field.split(".")
    val: Any = runtime
    for part in parts:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            if has_default:
                # Bare strings are returned as-is; numeric defaults
                # are parsed best-effort.
                return _coerce_default(default_str)
            raise CartridgeSerializationError(
                f"unresolved ${{runtime.{field}}} reference; known runtime keys: {sorted(runtime)}"
            )
    return val


def _load_single_file_tool(
    tool_file: Path,
    *,
    strict: bool,
    tool_modules: dict[str, Any],
    is_v2: bool = False,
) -> Any:
    """Load a single-file tool (``tools/<name>.py``) into a ToolSpec.

    The module declares its metadata via dunders:

    * ``__name__`` (defaults to file stem)
    * ``__description__`` (defaults to first docstring line)
    * ``__parameters__`` (dict; defaults to ``{}``)
    * ``__concurrent_safe__`` / ``__free__`` / ``__timeout_s__`` (optional)

    Required: an ``execute`` callable.

    **Cartridge spec v2 constraint.** The single-file form is for
    *trivial* tools only — a v2 cartridge that declares
    ``__requires__``, ``__render__``, or ``__tags__`` on a single-file
    tool is rejected at load time. Use the multi-file form
    (``tools/<name>/{tool.yaml, execute.py}``) for any tool that needs
    a shared resource. Two ways to express the same thing was a
    legibility failure of v1.1.

    Returns ``None`` (with a warning) when ``strict=False`` and the
    module is malformed; raises ``CartridgeSerializationError`` under
    ``strict=True``.
    """
    from looplet.tools import ToolSpec  # noqa: PLC0415

    name_stem = tool_file.stem
    module = _import_module_from_path(tool_file, f"_chw_tool_{name_stem}")
    tool_modules[name_stem] = module
    execute_fn = getattr(module, "execute", None)
    if not callable(execute_fn):
        msg = (
            f"single-file tool {tool_file} declares no callable ``execute``. "
            f"Add ``def execute(ctx, *, ...) -> dict``."
        )
        if strict:
            raise CartridgeSerializationError(msg)
        logger.warning("%s; skipping", msg)
        return None

    # ``__name__`` is a real dunder Python sets for every module, so
    # we can't rely on its presence as user intent. Detect by
    # comparing to the auto-set value (``_chw_tool_<stem>``); if the
    # user didn't override it, fall back to the file stem.
    declared_name = getattr(module, "__name__", "")
    if not declared_name or declared_name.startswith("_chw_tool_"):
        declared_name = name_stem

    # First non-empty line of the module docstring is a sensible
    # default description.
    raw_doc = (module.__doc__ or "").strip()
    default_desc = raw_doc.splitlines()[0] if raw_doc else f"Call {declared_name}."
    description = getattr(module, "__description__", default_desc)

    raw_parameters = getattr(module, "__parameters__", {}) or {}
    if not isinstance(raw_parameters, dict):
        msg = (
            f"single-file tool {tool_file}: ``__parameters__`` must be a dict, "
            f"got {type(raw_parameters).__name__}."
        )
        if strict:
            raise CartridgeSerializationError(msg)
        logger.warning("%s; skipping", msg)
        return None

    requires_dunder = list(getattr(module, "__requires__", []) or [])
    tags_dunder = list(getattr(module, "__tags__", []) or [])
    render_dunder = dict(getattr(module, "__render__", {}) or {})

    if is_v2:
        offenders: list[str] = []
        if requires_dunder:
            offenders.append("__requires__")
        if tags_dunder:
            offenders.append("__tags__")
        if render_dunder:
            offenders.append("__render__")
        if offenders:
            raise CartridgeSerializationError(
                f"single-file tool {tool_file} declares {offenders} but "
                f"cartridge spec v2 restricts the single-file form to trivial "
                f"tools (no shared resources, no rendering hints, no catalog "
                f"tags). Convert to the multi-file form: move the body to "
                f"``tools/{name_stem}/execute.py`` and put the metadata in "
                f"``tools/{name_stem}/tool.yaml``. Render hints belong in "
                f"``runtime.yaml: tool_render_hints:``; tags are a hook "
                f"concern (see docs/cartridge.md exclusion table)."
            )

    return ToolSpec(
        name=declared_name,
        description=str(description),
        parameters=dict(raw_parameters),
        execute=execute_fn,
        concurrent_safe=bool(getattr(module, "__concurrent_safe__", False)),
        free=bool(getattr(module, "__free__", False)),
        timeout_s=getattr(module, "__timeout_s__", None),
        requires=requires_dunder,
        tags=tags_dunder,
        render=render_dunder,
    )


def _coerce_default(text: str | None) -> Any:
    """Best-effort coerce a default string from ``${runtime.x:-default}``.

    Tries int → float → bool → string in that order. Bare ``null``
    becomes ``None``. Workspace authors who need richer defaults
    should set them in the factory function signature instead.
    """
    if text is None:
        return None
    s = text.strip()
    if s in ("null", "None"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ``resource_ref_for`` is defined in :mod:`looplet.refs` and re-imported
# above; kept available here as ``looplet.workspace.resource_ref_for`` for
# backwards compatibility.


def _resolve_refs(
    value: Any,
    resources: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
    is_v2: bool = False,
) -> Any:
    """Replace structured references in ``value`` with resolved objects.

    Supports four reference forms (all in string values; recurses into
    dicts and lists):

    * ``${ref:name}``         → ``resources[name]`` (raises if missing)
    * ``${py:module:symbol}`` → imported Python object
    * ``${runtime.field}``    → looked up in ``runtime`` (supports
      ``${runtime.field:-default}`` for missing keys)
    * ``"@name"`` (legacy)    → equivalent to ``${ref:name}``

    Other types pass through unchanged.

    The ``runtime`` parameter is keyword-only and defaults to ``None``
    so existing callers that only need ref-resolution work unchanged.

    The optional ``source_path`` is appended to every error message
    so a typo in ``hooks/05_QualityGate/config.yaml`` reports its
    location instead of leaving the user to grep for the bad ref.
    """
    src = f" (in {source_path})" if source_path else ""
    if isinstance(value, str):
        m = _REF_PATTERN.match(value)
        if m:
            kind = m.group("kind")
            spec = m.group("value")
            if kind == "ref":
                if spec not in resources:
                    raise CartridgeSerializationError(
                        f"unresolved ${{ref:{spec}}} reference{src}; "
                        f"known resources: {sorted(resources)}"
                    )
                return resources[spec]
            if kind == "py":
                if is_v2:
                    raise CartridgeSerializationError(
                        f"${{py:{spec}}} reference grammar is removed in "
                        f"cartridge spec v2{src}. The cartridge must not "
                        f"reach into arbitrary Python at load time. Wrap "
                        f"the symbol in a one-line ``resources/<name>.py`` "
                        f"builder (``def build(): from {spec.split(':')[0]} "
                        f"import {spec.split(':')[-1]}; return "
                        f"{spec.split(':')[-1]}``) and reference it as "
                        f"``${{ref:<name>}}``. Reference grammar in v2 is "
                        f"``${{ref:...}}`` / ``${{runtime.x}}`` only."
                    )
                try:
                    return _resolve_py_ref(spec)
                except CartridgeSerializationError as exc:
                    if src:
                        raise CartridgeSerializationError(f"{exc}{src}") from exc
                    raise
            if kind == "runtime":
                try:
                    return _resolve_runtime_ref(spec, runtime)
                except CartridgeSerializationError as exc:
                    if src:
                        raise CartridgeSerializationError(f"{exc}{src}") from exc
                    raise
        # Legacy ``@name`` form.
        if value.startswith(_REF_PREFIX):
            name = value[len(_REF_PREFIX) :]
            if name not in resources:
                raise CartridgeSerializationError(
                    f"unresolved resource reference {value!r}{src}; "
                    f"known resources: {sorted(resources)}"
                )
            return resources[name]
        return value
    if isinstance(value, dict):
        return {
            k: _resolve_refs(v, resources, runtime=runtime, source_path=source_path, is_v2=is_v2)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_refs(item, resources, runtime=runtime, source_path=source_path, is_v2=is_v2)
            for item in value
        ]
    return value
