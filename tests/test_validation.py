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

    def test_unknown_field_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported field type"):
            FieldSpec(name="x", field_type="integer")

    def test_python_type_field_type_remains_supported(self) -> None:
        schema = OutputSchema(fields={"x": FieldSpec(name="x", field_type=str)})
        assert validate_args(schema, {"x": "ok"}).valid
        assert not validate_args(schema, {"x": 1}).valid


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
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("native", [False, True])
    async def test_real_loops_share_validated_execution(self, async_mode, native):
        import json

        from looplet import LoopConfig, composable_loop
        from looplet.async_loop import async_composable_loop
        from looplet.tools import ToolSpec

        executed = []
        registry = ValidatingToolRegistry()
        registry.register_with_schema(
            ToolSpec(
                name="work",
                description="work",
                parameters={"value": "value"},
                execute=lambda value: executed.append(value) or {"value": value},
            ),
            OutputSchema(fields={"value": FieldSpec("value", "int")}),
        )
        registry.register(
            ToolSpec(
                name="done",
                description="finish",
                parameters={"answer": "answer"},
                execute=lambda answer: {"answer": answer},
            )
        )

        class Backend:
            def __init__(self):
                self.calls = iter(
                    [("work", {"value": "bad"}), ("work", {"value": 7}), ("done", {"answer": "7"})]
                )

            def generate(self, prompt, **kwargs):
                name, args = next(self.calls)
                return json.dumps({"tool": name, "args": args, "reasoning": "check"})

            def generate_with_tools(self, prompt, **kwargs):
                name, args = next(self.calls)
                return [{"type": "tool_use", "id": name, "name": name, "input": args}]

        kwargs = {
            "llm": Backend(),
            "tools": registry,
            "config": LoopConfig(max_steps=3, use_native_tools=native),
        }
        steps = (
            [step async for step in async_composable_loop(**kwargs)]
            if async_mode
            else list(composable_loop(**kwargs))
        )

        assert executed == [7]
        assert len(steps) == 3
        assert steps[0].tool_result.error_detail.kind.value == "validation"
        assert steps[-1].tool_call.tool == "done"

    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_resources_cancellation_and_views_keep_base_contract(self, async_mode):
        from looplet.tools import ToolSpec
        from looplet.types import CancelToken, ErrorKind, ToolContext

        registry = ValidatingToolRegistry()
        resource = object()
        observed = []

        def execute(ctx, value):
            observed.append(ctx.resources["shared"])
            return value

        registry.set_resources({"shared": resource})
        registry.register_with_schema(
            ToolSpec(
                name="work",
                description="work",
                parameters={"value": "value"},
                execute=execute,
                requires=("shared",),
            ),
            OutputSchema(fields={"value": FieldSpec("value", "int")}),
        )
        registry.register(
            ToolSpec(name="plain", description="plain", parameters={}, execute=lambda: True)
        )
        assert registry.tool_view(["plain"]).names == ("plain",)
        call = ToolCall(
            tool="work", args={"value": 7, "__internal": "hidden"}, reasoning="", call_id="work"
        )
        result = await registry.async_dispatch(call) if async_mode else registry.dispatch(call)
        assert result.error is None and observed == [resource]
        token = CancelToken()
        token.cancel()
        ctx = ToolContext(cancel_token=token)
        result = (
            await registry.async_dispatch(call, ctx=ctx)
            if async_mode
            else registry.dispatch(call, ctx=ctx)
        )
        assert result.error_detail.kind is ErrorKind.CANCELLED
        assert observed == [resource]

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("batch", [False, True])
    async def test_validation_and_context_use_one_dispatch_contract(self, async_mode, batch):
        from looplet.tools import ToolSpec
        from looplet.types import ErrorKind, ToolContext

        observed = []
        registry = ValidatingToolRegistry()

        def execute(ctx, value):
            observed.append((value, ctx.metadata["slot"]))
            return {"value": value, "step": ctx.metadata["slot"]}

        spec = ToolSpec(
            name="work",
            description="work",
            parameters={"value": "value"},
            execute=execute,
            concurrent_safe=True,
        )
        registry.register_with_schema(
            spec, OutputSchema(fields={"value": FieldSpec("value", "int")})
        )
        calls = [
            ToolCall(
                tool="work",
                args={"value": "bad", "__note": "internal"},
                reasoning="",
                call_id="bad",
            ),
            ToolCall(tool="work", args={"value": 7}, reasoning="", call_id="good"),
        ]
        contexts = [ToolContext(metadata={"slot": 10}), ToolContext(metadata={"slot": 20})]
        if async_mode:
            results = (
                await registry.async_dispatch_batch(calls, ctx=contexts)
                if batch
                else [
                    await registry.async_dispatch(call, ctx=ctx)
                    for call, ctx in zip(calls, contexts)
                ]
            )
        else:
            results = (
                registry.dispatch_batch(calls, ctx=contexts)
                if batch
                else [registry.dispatch(call, ctx=ctx) for call, ctx in zip(calls, contexts)]
            )

        assert observed == [(7, 20)]
        assert [result.call_id for result in results] == ["bad", "good"]
        assert results[0].error_detail.kind is ErrorKind.VALIDATION
        assert results[1].data == {"value": 7, "step": 20}

    def test_result_store_checkpoint_methods_delegate(self) -> None:
        registry = ValidatingToolRegistry()
        assert registry.snapshot_results() == {}
        registry.restore_results({"k": {"value": 1}})

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
