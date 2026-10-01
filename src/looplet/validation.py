"""Validation - schema enforcement for tool call arguments and done() payloads.

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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from looplet.tools import BaseToolRegistry, _summarize_args_dict
from looplet.types import ErrorKind, ToolCall, ToolContext, ToolError, ToolResult

if TYPE_CHECKING:
    from looplet.tools import _PreparedToolCall

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
        field_type: Type tag - one of 'str'|'int'|'float'|'bool'|'list'|'dict'|'any',
            or a Python type for backwards compatibility.
        required: Whether the field must be present.
        description: Human-readable description for documentation.
        allowed_values: Restrict to a fixed set of string values (enum-like).
    """

    name: str
    field_type: str | type
    required: bool = True
    description: str = ""
    allowed_values: list[str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.field_type, type):
            return
        if self.field_type != "any" and self.field_type not in _TYPE_MAP:
            raise ValueError(f"unsupported field type: {self.field_type!r}")


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
            expected = (
                spec.field_type
                if isinstance(spec.field_type, type)
                else _TYPE_MAP.get(spec.field_type)
            )
            if expected is not None:
                # bool is a subclass of int - check bool before int to avoid false positives
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


class ValidatingToolRegistry(BaseToolRegistry):
    """BaseToolRegistry subclass that validates args against a schema before dispatch.

    Tools registered with register_with_schema() have their args validated
    before execution. Invalid args produce a ToolResult with an error message.
    Tools registered normally (via _register()) bypass validation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._schemas: dict[str, OutputSchema] = {}

    def register_with_schema(self, spec: Any, schema: OutputSchema) -> None:
        """Register a ToolSpec with an accompanying validation schema."""
        self.register(spec)
        self._schemas[spec.name] = schema

    @property
    def _base(self) -> BaseToolRegistry:
        """Compatibility access to the registry that owns execution and results."""
        return self

    def _prepare_dispatch(
        self, call: ToolCall, *, ctx: ToolContext | None
    ) -> _PreparedToolCall | ToolResult:
        """Apply optional schemas once before the shared execution contract."""
        clean_args = {k: v for k, v in call.args.items() if not k.startswith("__")}
        if call.tool in self._schemas:
            schema = self._schemas[call.tool]
            result = validate_args(schema, clean_args)
            if not result.valid:
                error_msg = "Validation failed: " + "; ".join(result.errors)
                return ToolResult(
                    tool=call.tool,
                    args_summary=_summarize_args_dict(clean_args),
                    data=None,
                    error=error_msg,
                    error_detail=ToolError(
                        kind=ErrorKind.VALIDATION, message=error_msg, retriable=False
                    ),
                    call_id=call.call_id,
                )
        return super()._prepare_dispatch(call, ctx=ctx)


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
