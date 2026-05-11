"""Resource-reference helpers shared between core and the workspace loader.

This module owns the small registry that lets hooks like
:class:`looplet.permissions.PermissionHook` ask "was this engine handed
to me by a workspace ``resources/<name>.py`` builder?" — and if so,
return ``"@<name>"`` so :func:`looplet.workspace.preset_to_workspace`
can serialise the original reference instead of a placeholder.

It lives in its own tiny module (rather than inside ``workspace.py``)
so that core modules — ``permissions``, ``streaming``, ``evals``,
``telemetry`` — can call :func:`resource_ref_for` without importing
the 3000-line workspace loader. The workspace loader is then cleanly
extractable into a separate package without leaving reverse imports
behind.

Public symbol: :func:`resource_ref_for`. Everything else is treated as
internal but is re-exported from :mod:`looplet.workspace` for
backwards compatibility with tests written against the old location.
"""

from __future__ import annotations

from typing import Any

_REF_PREFIX = "@"

# id() -> resource name, populated by the workspace loader's
# ``_load_resources`` via ``_register_resource_origin``. Empty in
# pure-Python (no-workspace) usage; that path simply makes
# ``resource_ref_for`` return ``None`` and callers fall back to a
# default ref name.
_resource_origin: dict[int, str] = {}


def _register_resource_origin(instance: Any, name: str) -> None:
    """Record that ``instance`` came from ``resources/<name>.py``.

    Only registers instances that support :mod:`weakref`. Built-in
    containers (``list``, ``tuple``, ``dict``, ``set``) and other
    types that reject weak references would otherwise pin a stale
    entry forever — and Python's ``id()`` can be reused once the
    original object is garbage-collected, leading to false positives
    in :func:`resource_ref_for`. Skip them here; their identity is
    recovered via the ``__module__``-based fallback path.
    """
    import weakref  # noqa: PLC0415

    key = id(instance)
    try:
        weakref.finalize(instance, _resource_origin.pop, key, None)
    except TypeError:
        # Object can't be weak-referenced (list / tuple / dict / set /
        # bare int / etc.) — skip the identity registration entirely
        # so a future object at the same ``id()`` can't pick up this
        # name by accident.
        return
    _resource_origin[key] = name


def resource_ref_for(value: Any) -> str | None:
    """Return ``"@<name>"`` when ``value`` was produced by a workspace
    ``resources/<name>.py`` builder; ``None`` otherwise.

    Two detection paths:

    1. **Identity registry.** :func:`_register_resource_origin`
       stamps every built instance into :data:`_resource_origin`
       keyed by ``id``. This handles the common case where the
       builder returns a stock third-party object (e.g.
       ``PermissionEngine``, ``MetricsCollector``) whose own
       ``__module__`` is *not* in the workspace.
    2. **Module name fallback.** When the value (or its first list
       element) was *defined* inside the workspace's resource module
       (e.g. a closure built by ``def build(): def fn(...): ...``),
       its ``__module__`` starts with ``_chw_resource_``.

    Used by hook ``to_config()`` implementations to round-trip the
    *original* resource ref name (e.g. ``"@sql_permissions"``) instead
    of a hardcoded fallback (e.g. ``"@engine"``).

    Returns ``None`` for in-process objects so callers can fall back
    to a default ref name. In a pure-Python (no-workspace) program,
    :data:`_resource_origin` is empty and this always returns ``None``,
    which is the correct answer.
    """
    # Path 1: identity registry — handles instances of stock classes
    # like ``looplet.permissions.PermissionEngine`` whose own
    # ``__module__`` is the framework, not the workspace.
    by_id = _resource_origin.get(id(value))
    if by_id is not None:
        return f"{_REF_PREFIX}{by_id}"
    # Also probe list/tuple elements for identity matches — common
    # for ``EvalHook(evaluators=[...])`` whose list is rebuilt each
    # call by the resource builder.
    if isinstance(value, (list, tuple)) and value:
        elem_id = _resource_origin.get(id(value[0]))
        if elem_id is not None:
            return f"{_REF_PREFIX}{elem_id}"

    # Path 2: module-name fallback — handles closures defined inside
    # the resource module (CallableMemorySource(fn=lambda ...)) and
    # custom classes declared in the resource file.
    candidate_modules: list[str] = []
    cls_mod = getattr(type(value), "__module__", "") or ""
    if cls_mod:
        candidate_modules.append(cls_mod)
    direct_mod = getattr(value, "__module__", "") or ""
    if direct_mod and direct_mod != cls_mod:
        candidate_modules.append(direct_mod)
    if isinstance(value, (list, tuple)) and value:
        item_mod = getattr(value[0], "__module__", "") or getattr(type(value[0]), "__module__", "")
        if item_mod and item_mod not in candidate_modules:
            candidate_modules.append(item_mod)
    for mod in candidate_modules:
        if mod.startswith("_chw_resource_"):
            return f"{_REF_PREFIX}{mod[len('_chw_resource_') :]}"
    return None
