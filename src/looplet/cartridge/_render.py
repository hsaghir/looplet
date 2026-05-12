"""Serialiser-side renderers and helpers.

* :func:`_safe_filename` — sanitise a string into a filesystem-safe name.
* :class:`_DataclassReprFailed` — raised when a dataclass instance can't
  be reproduced on disk (closure, lambda, non-importable type).
* :func:`_render_value_literal` — emit a Python literal for any
  JSON-able / dataclass / importable callable value.
* :func:`_render_dataclass_kwargs` — emit ``field=value, ...`` for a
  dataclass instance, recursing through nested dataclasses.
* :func:`_apply_runtime_substitutions` — pre-pass substituting
  ``${runtime.x}`` placeholders into a YAML source text BEFORE the
  YAML parser sees it (lets cartridges reference runtime values in
  positions where YAML scalars don't survive structured replacement).
* :func:`_hook_class` — `hook if isclass(hook) else type(hook)`.

These are split out from the bigger `_serialise` /
`preset_to_cartridge` so they can be re-used and unit-tested in
isolation.
"""

from __future__ import annotations

import dataclasses as _dc
import enum as _enum
import inspect
import re
from typing import Any

from looplet.cartridge._layout import CartridgeSerializationError


def _safe_filename(name: str) -> str:
    """Sanitise an arbitrary string into a directory-safe filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "unnamed"


class _DataclassReprFailed(Exception):
    """Raised by ``_render_dataclass_kwargs`` when a field cannot be
    reproduced in source form (closure, lambda, opaque object, …).

    The auto-emit machinery catches this and falls through to the
    generic class branch which writes the safer ``Cls(...)`` shell.
    """


def _render_value_literal(value: Any, imports: set[str]) -> str:
    """Render ``value`` as a Python source expression usable in a builder.

    Supports JSON-able scalars/lists/dicts, top-level importable
    callables (emitted as ``from M import F`` + bare name), and
    nested dataclasses (recurses). Mutates ``imports`` so the caller
    can collect every needed import line.

    Raises :class:`_DataclassReprFailed` for closures, lambdas, opaque
    instances, or anything else that can't be re-emitted in source.
    """

    # Enum check first — string-backed enums (``class X(str, Enum)``)
    # would otherwise match the scalar branch and ``repr()`` would
    # emit invalid ``<EnumClass.MEMBER: 'value'>`` source.
    if isinstance(value, _enum.Enum):
        ecls = type(value)
        emod = ecls.__module__
        ename = ecls.__name__
        if emod and emod not in ("builtins",) and not emod.startswith("_chw_"):
            imports.add(f"from {emod} import {ename}")
            return f"{ename}.{value.name}"
        raise _DataclassReprFailed(f"non-importable enum: {ecls!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        parts = [_render_value_literal(v, imports) for v in value]
        if isinstance(value, list):
            return "[" + ", ".join(parts) + "]"
        # tuple: keep trailing comma for single-element tuples
        if len(parts) == 1:
            return "(" + parts[0] + ",)"
        return "(" + ", ".join(parts) + ")"
    if isinstance(value, dict):
        parts = [
            f"{_render_value_literal(k, imports)}: {_render_value_literal(v, imports)}"
            for k, v in value.items()
        ]
        return "{" + ", ".join(parts) + "}"
    # Top-level importable callable / class → emit bare name + import.
    if callable(value):
        # Special case: ``permissions:`` arg matchers carry their
        # source spec as ``__looplet_arg_matcher_spec__`` so the
        # round-trip writer can rebuild the matcher declaratively
        # instead of failing on the closure. Re-emit a call to
        # ``_make_arg_matcher({...})`` that produces an equivalent
        # closure on reload.
        spec_attr = getattr(value, "__looplet_arg_matcher_spec__", None)
        if isinstance(spec_attr, dict):
            imports.add("from looplet.cartridge.spec_slots import _make_arg_matcher")
            spec_literal = _render_value_literal(spec_attr, imports)
            return f"_make_arg_matcher({spec_literal})"
        mod = getattr(value, "__module__", "") or ""
        name = getattr(value, "__name__", "")
        qual = getattr(value, "__qualname__", "<lambda>")
        if (
            mod
            and mod not in ("builtins",)
            and not mod.startswith("_chw_")
            and name
            and qual != "<lambda>"
            and "<locals>" not in qual
        ):
            imports.add(f"from {mod} import {name}")
            return name
        raise _DataclassReprFailed(f"non-importable callable: {qual!r} from {mod!r}")
    # Nested dataclass instance → recurse.
    if _dc.is_dataclass(value):
        v_mod = type(value).__module__
        v_name = type(value).__name__
        if v_mod and v_mod not in ("builtins", "__main__") and not v_mod.startswith("_chw_"):
            imports.add(f"from {v_mod} import {v_name}")
            inner_kwargs = _render_dataclass_kwargs(value, imports)
            return f"{v_name}({inner_kwargs})"
    raise _DataclassReprFailed(f"unrenderable value of type {type(value).__name__!r}")


def _render_dataclass_kwargs(instance: Any, imports: set[str]) -> str:
    """Return ``"k1=v1, k2=v2, ..."`` reproducing ``instance``'s fields.

    Skips fields whose current value equals the dataclass-declared
    default (or default_factory output) so the rendered builder stays
    compact and matches the source preset's expressed configuration.
    """

    parts: list[str] = []
    for f in _dc.fields(instance):
        val = getattr(instance, f.name)
        # Skip fields holding their default — keeps builder readable
        # and matches dataclass __repr__ semantics.
        if f.default is not _dc.MISSING and val == f.default:
            continue
        if f.default_factory is not _dc.MISSING:  # type: ignore[misc]
            try:
                if val == f.default_factory():  # type: ignore[misc]
                    continue
            except Exception:  # noqa: BLE001
                pass
        rendered = _render_value_literal(val, imports)
        parts.append(f"{f.name}={rendered}")
    return ", ".join(parts)


_RUNTIME_PLACEHOLDER = re.compile(
    # Match ``${runtime.field}`` or ``${runtime.field:-default}`` —
    # ``field`` may be dotted for nested lookup.  Same shape as the
    # structured grammar in ``_resolve_refs`` so workspace authors
    # can use one syntax for both raw text (config.yaml interpolation
    # before YAML parse) and structured values (after parse).
    r"\$\{runtime\.([A-Za-z_][A-Za-z0-9_.]*)(:-[^}]*)?\}"
)


def _apply_runtime_substitutions(text: str, runtime: dict[str, Any]) -> str:
    """Replace ``${runtime.<key>}`` placeholders in ``text`` with the
    string form of ``runtime[<key>]``.

    Supports the same ``:-default`` form as the structured grammar
    (``${runtime.x:-15}``). Dotted keys descend nested dicts. Unknown
    keys without a default raise so a typo fails loudly at load time.

    **Scalar-only at text time.** When the resolved runtime value is
    a non-scalar (dict, list, custom object), the placeholder is
    *left intact* and the structured-value pass (``_resolve_refs``,
    run after YAML parsing) handles it — so callers passing structured
    runtime values like ``runtime={'alert': {...}}`` see them as the
    original object, not as a stringified ``"{'id': 'X', ...}"``.
    """

    _SCALAR_TYPES = (str, int, float, bool, type(None))

    def _sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        default_part = match.group(2) or ""
        has_default = default_part.startswith(":-")
        default_text = default_part[2:] if has_default else None

        # Walk dotted path.
        parts = key.split(".")
        val: Any = runtime
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                if has_default:
                    return default_text or ""
                raise CartridgeSerializationError(
                    f"unresolved ${{runtime.{key}}} placeholder; "
                    f"known runtime keys: {sorted(runtime)}"
                )
        # Only stringify scalars at text-pass time. Non-scalars stay
        # as the original placeholder so the structured pass can
        # resolve them with identity preserved.
        if not isinstance(val, _SCALAR_TYPES):
            return match.group(0)
        return str(val)

    # Strip full-line comments before substitution so the regex
    # doesn't fire on placeholder-shaped text inside YAML # comments
    # (the rest of the loader strips them anyway in ``_load_yaml``).
    text_no_comments = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    return _RUNTIME_PLACEHOLDER.sub(_sub, text_no_comments)


def _hook_class(hook: Any) -> type:
    return hook if inspect.isclass(hook) else type(hook)
