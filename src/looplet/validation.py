"""Validation — schema enforcement for tool call arguments and done() payloads.

Provides:
  - FieldSpec: description of one field in a schema
  - OutputSchema: collection of FieldSpecs with strict/non-strict mode
  - ValidationResult: outcome of a validation check (valid, errors, warnings)
  - validate_args: validate a dict against an OutputSchema
  - ValidatingToolRegistry: BaseToolRegistry subclass that validates before dispatch
  - DoneValidator: Protocol for done-payload validators
  - SimpleDoneValidator: validates that required fields are present in done()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Map field_type string to Python types for isinstance checks
_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "float": (float, int),  # int is a valid float
    "bool": bool,
    "list": list,
    "dict": dict,
}


# ── FieldSpec ────────────────────────────────────────────────────────


@dataclass
class FieldSpec:
    """Description of one field in an OutputSchema.

    Args:
        name: Field name (must match the arg key).
        field_type: Type tag — one of 'str'|'int'|'float'|'bool'|'list'|'dict'|'any'.
        required: Whether the field must be present.
        description: Human-readable description for documentation.
        allowed_values: Restrict to a fixed set of string values (enum-like).
    """

    name: str
    field_type: str
    required: bool = True
    description: str = ""
    allowed_values: list[str] | None = None


# ── OutputSchema ─────────────────────────────────────────────────────


@dataclass
class OutputSchema:
    """Collection of FieldSpecs describing expected tool call arguments.

    Args:
        fields: Mapping of field_name -> FieldSpec.
        strict: If True, unknown fields are errors rather than warnings.
    """

    fields: dict[str, FieldSpec]
    strict: bool = False


# ── ValidationResult ─────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Outcome of a validation check.

    Args:
        valid: True if all required constraints passed.
        errors: List of hard constraint violations.
        warnings: List of soft advisories (e.g. unknown fields).
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── validate_args ─────────────────────────────────────────────────────


def validate_args(schema: OutputSchema, args: dict[str, Any]) -> ValidationResult:
    """Validate a dict of tool call arguments against an OutputSchema.

    Checks:
    1. Required fields are present.
    2. Field types match (basic isinstance).
    3. allowed_values constraint respected.
    4. Unknown fields: warning (non-strict) or error (strict).

    Returns a ValidationResult with all discovered issues.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required fields and types
    for name, spec in schema.fields.items():
        if name not in args:
            if spec.required:
                errors.append(f"missing required field: {name!r}")
            continue

        value = args[name]

        # Type check (skip for 'any')
        if spec.field_type != "any":
            expected = _TYPE_MAP.get(spec.field_type)
            if expected is not None:
                # bool is a subclass of int — check bool before int to avoid false positives
                if spec.field_type == "int" and isinstance(value, bool):
                    errors.append(f"field {name!r}: expected int, got {type(value).__name__}")
                elif not isinstance(value, expected):
                    errors.append(
                        f"field {name!r}: expected {spec.field_type}, got {type(value).__name__}"
                    )

        # allowed_values check
        if spec.allowed_values is not None and value not in spec.allowed_values:
            errors.append(
                f"field {name!r}: value {value!r} not in allowed values {spec.allowed_values}"
            )

    # Unknown fields
    known = set(schema.fields.keys())
    unknown = set(args.keys()) - known
    for unk in sorted(unknown):
        if schema.strict:
            errors.append(f"unknown field: {unk!r}")
        else:
            warnings.append(f"unknown field: {unk!r}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── ValidatingToolRegistry ────────────────────────────────────────────


class ValidatingToolRegistry:
    """BaseToolRegistry subclass that validates args against a schema before dispatch.

    Tools registered with register_with_schema() have their args validated
    before execution. Invalid args produce a ToolResult with an error message.
    Tools registered normally (via _register()) bypass validation.
    """

    def __init__(self) -> None:
        # Import lazily to avoid circular dependencies and allow tools.py to be optional
        from looplet.tools import BaseToolRegistry

        self._base = BaseToolRegistry.__new__(BaseToolRegistry)
        BaseToolRegistry.__init__(self._base)
        self._schemas: dict[str, OutputSchema] = {}

    def register_with_schema(self, spec: Any, schema: OutputSchema) -> None:
        """Register a ToolSpec with an accompanying validation schema."""
        self._base.register(spec)
        self._schemas[spec.name] = schema

    def register(self, spec: Any) -> None:
        """Register a tool without a schema (no validation applied)."""
        self._base.register(spec)

    # Backward-compat alias
    _register = register

    @property
    def tool_names(self) -> list[str]:
        return self._base.tool_names

    def tool_catalog_text(self) -> str:
        return self._base.tool_catalog_text()

    @property
    def _tools(self) -> dict:
        """Proxy to the underlying registry's tool dict for budget accounting."""
        return self._base._tools

    def tool_schemas(self) -> list:
        """Proxy for native tool calling."""
        return self._base.tool_schemas()

    def dispatch(self, call: Any, **kwargs: Any) -> Any:
        """Validate args (if schema exists) then dispatch. Returns error ToolResult on failure."""
        from looplet.types import ToolResult

        clean_args = {k: v for k, v in call.args.items() if not k.startswith("__")}

        if call.tool in self._schemas:
            schema = self._schemas[call.tool]
            result = validate_args(schema, clean_args)
            if not result.valid:
                from looplet.tools import _summarize_args_dict  # noqa: PLC0415

                error_msg = "Validation failed: " + "; ".join(result.errors)
                return ToolResult(
                    tool=call.tool,
                    args_summary=_summarize_args_dict(clean_args),
                    data=None,
                    error=error_msg,
                    call_id=call.call_id,
                )

        return self._base.dispatch(call, **kwargs)

    def dispatch_batch(self, calls: list[Any], **kwargs: Any) -> list[Any]:
        """Validate all calls, dispatch valid ones in batch, return errors for invalid.

        Preserves concurrent dispatch optimization from BaseToolRegistry for
        calls that pass validation.
        """
        from looplet.types import ToolResult

        results: dict[int, Any] = {}
        valid_calls: list[tuple[int, Any]] = []

        for i, call in enumerate(calls):
            clean_args = {k: v for k, v in call.args.items() if not k.startswith("__")}
            if call.tool in self._schemas:
                schema = self._schemas[call.tool]
                validation = validate_args(schema, clean_args)
                if not validation.valid:
                    error_msg = "Validation failed: " + "; ".join(validation.errors)
                    results[i] = ToolResult(
                        tool=call.tool,
                        args_summary=str(clean_args)[:100],
                        data=None,
                        error=error_msg,
                        call_id=call.call_id,
                    )
                    continue
            valid_calls.append((i, call))

        if valid_calls:
            batch_results = self._base.dispatch_batch([c for _, c in valid_calls], **kwargs)
            for (idx, _), result in zip(valid_calls, batch_results):
                results[idx] = result

        return [results[i] for i in range(len(calls))]


# ── DoneValidator Protocol ────────────────────────────────────────────


@runtime_checkable
class DoneValidator(Protocol):
    """Protocol for done-payload validators.

    Implementations check that the agent's final done() call payload
    contains all required fields and meets any domain-specific constraints.
    """

    def validate_done(self, payload: dict[str, Any]) -> ValidationResult: ...


# ── SimpleDoneValidator ───────────────────────────────────────────────


class SimpleDoneValidator:
    """Validates that required_fields are present in the done() payload.

    Optional fields are accepted without warnings. Any other fields
    in the payload produce a warning.
    """

    def __init__(
        self,
        required_fields: list[str],
        optional_fields: list[str] | None = None,
    ) -> None:
        self.required_fields = list(required_fields)
        self.optional_fields = list(optional_fields or [])

    def validate_done(self, payload: dict[str, Any]) -> ValidationResult:
        """Check that all required_fields are present; warn about unexpected fields."""
        errors: list[str] = []
        warnings: list[str] = []

        for fname in self.required_fields:
            if fname not in payload:
                errors.append(f"missing required done field: {fname!r}")

        known = set(self.required_fields) | set(self.optional_fields)
        for key in sorted(payload.keys()):
            if key not in known:
                warnings.append(f"unexpected done field: {key!r}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
