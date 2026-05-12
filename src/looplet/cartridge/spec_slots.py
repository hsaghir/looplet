"""Compilers for the v1.0 declarative slots in ``config.yaml``.

These functions translate the declarative ``model:``, ``permissions:``,
and ``memory:`` blocks documented in ``SPEC.md`` into the live
runtime objects the loop expects (``LoopConfig`` field overrides and
``PermissionHook`` instances).

Kept here rather than inlined in :mod:`looplet.workspace` so the
permission / model / memory shape can be tested in isolation and
ported to a future spec-only package without dragging the loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from looplet.permissions import (
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    PermissionRule,
)
from looplet.validation import FieldSpec, OutputSchema

if TYPE_CHECKING:
    pass


__all__ = [
    "compile_permissions_block",
    "compile_model_block",
    "compile_output_schema",
    "default_long_term_memory_path",
    "SCHEMA_BLOCK_PERMISSIONS",
    "SCHEMA_BLOCK_MODEL",
    "SCHEMA_BLOCK_MEMORY",
]


SCHEMA_BLOCK_PERMISSIONS = "permissions"
SCHEMA_BLOCK_MODEL = "model"
SCHEMA_BLOCK_MEMORY = "memory"


# ── permissions: ───────────────────────────────────────────────────


_DECISION_BY_KEY: dict[str, PermissionDecision] = {
    "allow": PermissionDecision.ALLOW,
    "deny": PermissionDecision.DENY,
    "ask": PermissionDecision.ASK,
}


def _make_arg_matcher(spec: dict[str, Any]) -> Any:
    """Build an ``ArgMatcher`` from the ``matches:`` / ``contains:`` shorthand.

    * ``matches:`` requires an equality match against the named arg.
    * ``contains:`` requires the named arg to be a string containing
      the substring.

    Returning ``None`` means the rule matches regardless of args.
    """
    matches = spec.get("matches") or {}
    contains = spec.get("contains") or {}
    if not matches and not contains:
        return None

    def _match(args: dict[str, Any]) -> bool:
        for key, expected in matches.items():
            if args.get(key) != expected:
                return False
        for key, needle in contains.items():
            value = args.get(key)
            if not isinstance(value, str) or needle not in value:
                return False
        return True

    # Round-trip metadata: stamp the closure with the spec that
    # produced it so ``preset_to_cartridge`` can reproduce the matcher
    # declaratively (re-emitting ``matches:`` / ``contains:`` instead
    # of failing on the closure). Without this, every cartridge that
    # uses ``permissions: deny: contains:`` loses its rules on
    # round-trip because the closure isn't importable.
    _match.__looplet_arg_matcher_spec__ = {  # type: ignore[attr-defined]
        "matches": dict(matches) if matches else {},
        "contains": dict(contains) if contains else {},
    }
    return _match


def _normalise_rule_entry(entry: Any) -> dict[str, Any]:
    """Accept either a bare tool name or a structured rule dict.

    Returns a dict with at least ``tool``; raises ``ValueError`` on
    malformed input so the loader can attribute the error to the
    offending file.
    """
    if isinstance(entry, str):
        return {"tool": entry}
    if isinstance(entry, dict):
        if "tool" not in entry:
            raise ValueError(f"permission rule missing 'tool' field: {entry!r}")
        return entry
    raise ValueError(f"permission rule must be a string (tool name) or a dict, got {entry!r}")


def compile_permissions_block(block: dict[str, Any]) -> PermissionHook:
    """Compile a ``permissions:`` block into a ``PermissionHook``.

    The block accepts ``default``, ``allow``, ``deny``, ``ask`` keys.
    Each rule list contains either bare tool names (matching any args)
    or structured dicts with ``matches:`` / ``contains:`` shorthand.

    Empty / missing blocks return a hook with no rules and the
    documented default of ``allow`` so the result is always safe to
    install.
    """
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(
            f"permissions block must be a mapping, got {type(block).__name__}: {block!r}"
        )

    default_key = (block.get("default") or "allow").lower()
    if default_key not in _DECISION_BY_KEY:
        raise ValueError(
            f"permissions.default must be one of {sorted(_DECISION_BY_KEY)}, got {default_key!r}"
        )
    engine = PermissionEngine(default=_DECISION_BY_KEY[default_key])

    # Order matters: deny list is evaluated first so explicit denials
    # cannot be silently overridden by a broader allow rule. ask runs
    # second; allow is last and acts as a positive override.
    for slot_key in ("deny", "ask", "allow"):
        rules = block.get(slot_key) or []
        if not isinstance(rules, list):
            raise ValueError(
                f"permissions.{slot_key} must be a list, got {type(rules).__name__}: {rules!r}"
            )
        decision = _DECISION_BY_KEY[slot_key]
        for entry in rules:
            normalised = _normalise_rule_entry(entry)
            engine.rules.append(
                PermissionRule(
                    tool=normalised["tool"],
                    decision=decision,
                    arg_matcher=_make_arg_matcher(normalised),
                    reason=str(normalised.get("reason", "")),
                )
            )

    return PermissionHook(engine)


# ── model: ─────────────────────────────────────────────────────────


_MODEL_TO_LOOPCONFIG: dict[str, str] = {
    "max_tokens": "max_tokens",
    "temperature": "temperature",
}


def compile_model_block(block: dict[str, Any], *, existing_cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate a structured ``model:`` block into ``LoopConfig`` overrides.

    Returns a dict of overrides to merge into the existing config
    kwargs. When both a structured block and a flat field
    (e.g. ``temperature: 0.2`` at the top level) are present, the
    structured block wins.

    Provider / model name / reasoning effort / extras are *not* part
    of the v1.0 ``LoopConfig`` surface; they are surfaced through
    ``tool_metadata['model']`` so that downstream tooling (cost
    estimators, routers, governance) can read them without parsing
    the cartridge again.
    """
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"model block must be a mapping, got {type(block).__name__}: {block!r}")

    overrides: dict[str, Any] = {}

    for source_key, cfg_key in _MODEL_TO_LOOPCONFIG.items():
        if source_key in block:
            overrides[cfg_key] = block[source_key]

    metadata: dict[str, Any] = {}
    for key in ("provider", "name", "reasoning_effort", "top_p", "extra"):
        if key in block:
            metadata[key] = block[key]

    if metadata:
        existing_metadata = dict(existing_cfg.get("tool_metadata") or {})
        existing_metadata.setdefault("model", {}).update(metadata)
        overrides["tool_metadata"] = existing_metadata

    return overrides


# ── memory: ────────────────────────────────────────────────────────


def default_long_term_memory_path() -> str:
    """Path of the default long-term memory file relative to the cartridge root."""
    return "memory/long_term.md"


# ── output_schema: on done ────────────────────────────────


_JSONSCHEMA_TYPE_TO_FIELD_TYPE: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def compile_output_schema(block: dict[str, Any]) -> OutputSchema:
    """Translate a (subset of) JSON Schema into an :class:`OutputSchema`.

    Supported shape (matches what ``tools/done/tool.yaml`` realistically
    declares today): top-level ``type: object`` with ``properties``
    and an optional ``required`` list. Each property may declare a
    JSON Schema ``type``, an ``enum`` (mapped to ``allowed_values``),
    and a ``description``.

    Anything outside this subset is ignored with a deliberate silent
    pass: future spec versions may widen the supported shape, and
    rejecting unknown keys would prevent forward-compatible cartridges.
    """
    if not isinstance(block, dict):
        raise ValueError(f"output_schema must be a mapping, got {type(block).__name__}: {block!r}")
    block_type = block.get("type", "object")
    if block_type != "object":
        raise ValueError(
            f"output_schema.type must be 'object' (got {block_type!r}); "
            f"flat schemas are out of scope for v1.0"
        )

    properties = block.get("properties") or {}
    if not isinstance(properties, dict):
        raise ValueError(
            f"output_schema.properties must be a mapping, got {type(properties).__name__}"
        )
    required = set(block.get("required") or [])

    fields: dict[str, FieldSpec] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            raise ValueError(f"output_schema.properties[{name!r}] must be a mapping")
        json_type = prop.get("type", "any")
        if isinstance(json_type, list):
            # JSON Schema allows ["string", "null"] etc. We collapse
            # to the first non-null type for OutputSchema's flat type tag.
            non_null = [t for t in json_type if t != "null"]
            json_type = non_null[0] if non_null else "any"
        field_type = _JSONSCHEMA_TYPE_TO_FIELD_TYPE.get(json_type, "any")
        enum = prop.get("enum")
        allowed_values = [str(v) for v in enum] if isinstance(enum, list) and enum else None
        fields[name] = FieldSpec(
            name=name,
            field_type=field_type,
            required=name in required,
            description=str(prop.get("description", "") or ""),
            allowed_values=allowed_values,
        )

    return OutputSchema(fields=fields, strict=False)
