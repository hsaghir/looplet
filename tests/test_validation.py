"""Tests for looplet.validation - schema enforcement for tool call args and done payloads."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from looplet.types import ToolCall, ToolResult
from looplet.validation import (
    DoneValidator,
    FieldSpec,
    OutputSchema,
    SimpleDoneValidator,
    ValidatingToolRegistry,
    ValidationResult,
    validate_args,
)

# ── FieldSpec ────────────────────────────────────────────────────────


class TestFieldSpec:
    def test_name_and_type_required(self) -> None:
        fs = FieldSpec(name="query", field_type="str")
        assert fs.name == "query"
        assert fs.field_type == "str"

    def test_required_defaults_true(self) -> None:
        fs = FieldSpec(name="x", field_type="int")
        assert fs.required is True

    def test_description_defaults_empty(self) -> None:
        fs = FieldSpec(name="x", field_type="str")
        assert fs.description == ""

    def test_allowed_values_defaults_none(self) -> None:
        fs = FieldSpec(name="x", field_type="str")
        assert fs.allowed_values is None

    def test_optional_field(self) -> None:
        fs = FieldSpec(name="limit", field_type="int", required=False)
        assert fs.required is False

    def test_allowed_values_set(self) -> None:
        fs = FieldSpec(name="color", field_type="str", allowed_values=["red", "blue"])
        assert fs.allowed_values == ["red", "blue"]

    def test_all_field_types_creatable(self) -> None:
        for ft in ("str", "int", "float", "bool", "list", "dict", "any"):
            fs = FieldSpec(name="x", field_type=ft)
            assert fs.field_type == ft


# ── OutputSchema ────────────────────────────────────────────────────


class TestOutputSchema:
    def test_fields_dict(self) -> None:
        schema = OutputSchema(fields={"q": FieldSpec(name="q", field_type="str")})
        assert "q" in schema.fields

    def test_strict_defaults_false(self) -> None:
        schema = OutputSchema(fields={})
        assert schema.strict is False

    def test_strict_mode(self) -> None:
        schema = OutputSchema(fields={}, strict=True)
        assert schema.strict is True


# ── ValidationResult ────────────────────────────────────────────────


class TestValidationResult:
    def test_valid_true(self) -> None:
        vr = ValidationResult(valid=True)
        assert vr.valid is True

    def test_errors_default_empty(self) -> None:
        vr = ValidationResult(valid=True)
        assert vr.errors == []

    def test_warnings_default_empty(self) -> None:
        vr = ValidationResult(valid=True)
        assert vr.warnings == []

    def test_invalid_with_errors(self) -> None:
        vr = ValidationResult(valid=False, errors=["missing field: x"])
        assert not vr.valid
        assert "missing field: x" in vr.errors


# ── validate_args ───────────────────────────────────────────────────


class TestValidateArgs:
    def _schema(self, **specs: tuple[str, bool]) -> OutputSchema:
        """Build a simple schema. specs: field_name -> (type, required)."""
        fields = {
            name: FieldSpec(name=name, field_type=ft, required=req)
            for name, (ft, req) in specs.items()
        }
        return OutputSchema(fields=fields)

    def test_valid_args_returns_valid(self) -> None:
        schema = self._schema(query=("str", True))
        result = validate_args(schema, {"query": "hello"})
        assert result.valid is True
        assert result.errors == []

    def test_missing_required_field_invalid(self) -> None:
        schema = self._schema(query=("str", True))
        result = validate_args(schema, {})
        assert result.valid is False
        assert any("query" in e for e in result.errors)

    def test_missing_optional_field_valid(self) -> None:
        schema = self._schema(limit=("int", False))
        result = validate_args(schema, {})
        assert result.valid is True

    def test_wrong_type_invalid(self) -> None:
        schema = self._schema(count=("int", True))
        result = validate_args(schema, {"count": "not-an-int"})
        assert result.valid is False
        assert any("count" in e for e in result.errors)

    def test_correct_int_type_valid(self) -> None:
        schema = self._schema(count=("int", True))
        result = validate_args(schema, {"count": 42})
        assert result.valid is True

    def test_correct_float_type_valid(self) -> None:
        schema = self._schema(score=("float", True))
        result = validate_args(schema, {"score": 3.14})
        assert result.valid is True

    def test_int_passes_float_check(self) -> None:
        # int is a subtype of float in Python, should be valid
        schema = self._schema(score=("float", True))
        result = validate_args(schema, {"score": 5})
        assert result.valid is True

    def test_correct_bool_type_valid(self) -> None:
        schema = self._schema(flag=("bool", True))
        result = validate_args(schema, {"flag": True})
        assert result.valid is True

    def test_correct_list_type_valid(self) -> None:
        schema = self._schema(items=("list", True))
        result = validate_args(schema, {"items": [1, 2, 3]})
        assert result.valid is True

    def test_wrong_list_type_invalid(self) -> None:
        schema = self._schema(items=("list", True))
        result = validate_args(schema, {"items": "not-a-list"})
        assert result.valid is False

    def test_correct_dict_type_valid(self) -> None:
        schema = self._schema(data=("dict", True))
        result = validate_args(schema, {"data": {"k": "v"}})
        assert result.valid is True

    def test_any_type_accepts_anything(self) -> None:
        schema = self._schema(value=("any", True))
        for v in (1, "s", [1], {"k": "v"}, True, 3.14):
            result = validate_args(schema, {"value": v})
            assert result.valid is True

    def test_allowed_values_valid(self) -> None:
        schema = OutputSchema(
            fields={
                "color": FieldSpec(name="color", field_type="str", allowed_values=["red", "blue"])
            }
        )
        result = validate_args(schema, {"color": "red"})
        assert result.valid is True

    def test_allowed_values_invalid(self) -> None:
        schema = OutputSchema(
            fields={
                "color": FieldSpec(name="color", field_type="str", allowed_values=["red", "blue"])
            }
        )
        result = validate_args(schema, {"color": "green"})
        assert result.valid is False
        assert any("green" in e or "allowed" in e.lower() for e in result.errors)

    def test_unknown_fields_warning_non_strict(self) -> None:
        schema = self._schema(query=("str", True))
        result = validate_args(schema, {"query": "x", "extra": "y"})
        assert result.valid is True
        assert any("extra" in w for w in result.warnings)

    def test_unknown_fields_error_strict(self) -> None:
        schema = OutputSchema(
            fields={"query": FieldSpec(name="query", field_type="str")},
            strict=True,
        )
        result = validate_args(schema, {"query": "x", "extra": "y"})
        assert result.valid is False
        assert any("extra" in e for e in result.errors)

    def test_multiple_errors_reported(self) -> None:
        schema = self._schema(a=("str", True), b=("int", True))
        result = validate_args(schema, {})
        assert result.valid is False
        assert len(result.errors) >= 2

    def test_empty_schema_empty_args_valid(self) -> None:
        schema = OutputSchema(fields={})
        result = validate_args(schema, {})
        assert result.valid is True


# ── ValidatingToolRegistry ──────────────────────────────────────────


class TestValidatingToolRegistry:
    def _make_registry(self) -> "ValidatingToolRegistry":
        from looplet.tools import ToolSpec
        from looplet.validation import FieldSpec, OutputSchema, ValidatingToolRegistry

        registry = ValidatingToolRegistry()
        spec = ToolSpec(
            name="search",
            description="Search for data",
            parameters={"query": "search query"},
            execute=lambda query="": {"rows": [{"q": query}]},
        )
        schema = OutputSchema(
            fields={
                "query": FieldSpec(name="query", field_type="str", required=True),
            }
        )
        registry.register_with_schema(spec, schema)
        return registry

    def test_valid_call_executes_tool(self) -> None:
        registry = self._make_registry()
        call = ToolCall(tool="search", args={"query": "test"}, reasoning="r")
        result = registry.dispatch(call)
        assert result.error is None
        assert result.data is not None

    def test_invalid_call_returns_error_result(self) -> None:
        registry = self._make_registry()
        call = ToolCall(tool="search", args={}, reasoning="r")
        result = registry.dispatch(call)
        assert result.error is not None
        assert "query" in result.error.lower() or "validation" in result.error.lower()

    def test_invalid_call_does_not_execute_tool(self) -> None:
        executed: list[bool] = []
        from looplet.tools import ToolSpec
        from looplet.validation import FieldSpec, OutputSchema, ValidatingToolRegistry

        registry = ValidatingToolRegistry()
        spec = ToolSpec(
            name="op",
            description="op",
            parameters={"x": "an int"},
            execute=lambda x=0: executed.append(True) or {"ok": True},
        )
        schema = OutputSchema(
            fields={
                "x": FieldSpec(name="x", field_type="int", required=True),
            }
        )
        registry.register_with_schema(spec, schema)

        call = ToolCall(tool="op", args={"x": "not-int"}, reasoning="r")
        registry.dispatch(call)
        assert len(executed) == 0

    def test_unknown_tool_still_returns_error(self) -> None:
        registry = self._make_registry()
        call = ToolCall(tool="nonexistent", args={}, reasoning="r")
        result = registry.dispatch(call)
        assert result.error is not None

    def test_tool_without_schema_dispatches_normally(self) -> None:
        from looplet.tools import ToolSpec
        from looplet.validation import ValidatingToolRegistry

        registry = ValidatingToolRegistry()
        spec = ToolSpec(
            name="think",
            description="think",
            parameters={"analysis": "reasoning"},
            execute=lambda analysis="": {"ok": True},
        )
        registry.register(spec)  # No schema attached
        call = ToolCall(tool="think", args={"analysis": "hmm"}, reasoning="r")
        result = registry.dispatch(call)
        assert result.error is None

    def test_register_with_schema_adds_to_tools(self) -> None:
        registry = self._make_registry()
        assert "search" in registry.tool_names

    def test_error_result_has_tool_name(self) -> None:
        registry = self._make_registry()
        call = ToolCall(tool="search", args={}, reasoning="r")
        result = registry.dispatch(call)
        assert result.tool == "search"


# ── SimpleDoneValidator ─────────────────────────────────────────────


class TestSimpleDoneValidator:
    def test_required_fields_all_present_valid(self) -> None:
        validator = SimpleDoneValidator(required_fields=["summary", "entities"])
        result = validator.validate_done({"summary": "done", "entities": ["host-a"]})
        assert result.valid is True

    def test_missing_required_field_invalid(self) -> None:
        validator = SimpleDoneValidator(required_fields=["summary"])
        result = validator.validate_done({})
        assert result.valid is False
        assert any("summary" in e for e in result.errors)

    def test_optional_fields_dont_cause_warnings(self) -> None:
        validator = SimpleDoneValidator(
            required_fields=["summary"],
            optional_fields=["notes"],
        )
        result = validator.validate_done({"summary": "done", "notes": "extra"})
        assert result.valid is True
        assert not any("notes" in w for w in result.warnings)

    def test_unknown_fields_produce_warnings(self) -> None:
        validator = SimpleDoneValidator(
            required_fields=["summary"],
            optional_fields=["notes"],
        )
        result = validator.validate_done({"summary": "done", "unexpected": "x"})
        assert result.valid is True
        assert any("unexpected" in w for w in result.warnings)

    def test_empty_required_fields_always_valid(self) -> None:
        validator = SimpleDoneValidator(required_fields=[])
        result = validator.validate_done({"anything": "goes"})
        assert result.valid is True

    def test_multiple_missing_fields_all_reported(self) -> None:
        validator = SimpleDoneValidator(required_fields=["a", "b", "c"])
        result = validator.validate_done({})
        assert result.valid is False
        assert len(result.errors) == 3

    def test_returns_validation_result_type(self) -> None:
        validator = SimpleDoneValidator(required_fields=[])
        result = validator.validate_done({})
        assert isinstance(result, ValidationResult)


# ── DoneValidator Protocol ──────────────────────────────────────────


class TestDoneValidatorProtocol:
    def test_simple_done_validator_is_instance_of_protocol(self) -> None:
        validator = SimpleDoneValidator(required_fields=["summary"])
        assert isinstance(validator, DoneValidator)

    def test_protocol_has_validate_done_method(self) -> None:
        validator = SimpleDoneValidator(required_fields=[])
        assert hasattr(validator, "validate_done")
        assert callable(validator.validate_done)
