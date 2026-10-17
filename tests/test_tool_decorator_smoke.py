"""Tests for decorator-first tool construction."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, Literal, Union

import pytest

from looplet import ToolContext, ToolSpec, tool, tools_from
from looplet.types import ToolCall

pytestmark = pytest.mark.smoke


class TestToolDecorator:
    def test_decorator_with_args_returns_tool_spec(self) -> None:
        @tool(description="Search the docs.", concurrent_safe=True, timeout_s=3)
        def search_docs(query: str, limit: int = 5) -> dict:
            return {"query": query, "limit": limit}

        assert isinstance(search_docs, ToolSpec)
        assert search_docs.name == "search_docs"
        assert search_docs.description == "Search the docs."
        assert search_docs.concurrent_safe is True
        assert search_docs.timeout_s == 3
        schema = search_docs.to_json_schema()
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["limit"]["default"] == 5
        assert schema["required"] == ["query"]

    def test_decorator_without_parentheses_uses_docstring(self) -> None:
        @tool
        def ping(message: str) -> dict:
            """Echo a message."""
            return {"message": message}

        assert isinstance(ping, ToolSpec)
        assert ping.description == "Echo a message."

    def test_custom_name_and_literal_enum(self) -> None:
        @tool(name="set_mode")
        def set_mode(mode: Literal["fast", "safe"]) -> dict:
            return {"mode": mode}

        schema = set_mode.to_json_schema()
        assert set_mode.name == "set_mode"
        assert schema["properties"]["mode"]["enum"] == ["fast", "safe"]

    def test_tools_from_registers_decorated_and_plain_callables(self) -> None:
        @tool(description="Increment a value.")
        def increment(value: int) -> dict:
            return {"value": value + 1}

        def shout(text: str) -> dict:
            """Uppercase text."""
            return {"text": text.upper()}

        registry = tools_from([increment, shout], include_done=True)

        assert registry.tool_names == ["increment", "shout", "done"]
        result = registry.dispatch(ToolCall(tool="increment", args={"value": 2}))
        assert result.error is None
        assert result.data == {"value": 3}
        shout_result = registry.dispatch(ToolCall(tool="shout", args={"text": "hi"}))
        assert shout_result.data == {"text": "HI"}

    @pytest.mark.parametrize("async_dispatch", [False, True])
    @pytest.mark.parametrize("kind", ["partial", "nested_partial", "instance", "partial_instance"])
    async def test_bound_callables_preserve_signature_and_metadata(self, kind, async_dispatch):
        effects = []
        owners = {"payments": "fintech-platform"}

        def lookup_owner(owners: dict, service: str, limit: int = 1) -> dict:
            """Find the team that owns a service."""
            effects.append((service, limit))
            return {"owner": owners.get(service), "limit": limit}

        class OwnerLookup:
            def __call__(self, service: str, limit: int = 1) -> dict:
                """Find the team that owns a service."""
                return lookup_owner(owners, service, limit)

        candidates = {
            "partial": (partial(lookup_owner, owners), "lookup_owner", 1),
            "nested_partial": (partial(partial(lookup_owner, owners), limit=2), "lookup_owner", 2),
            "instance": (OwnerLookup(), "OwnerLookup", 1),
            "partial_instance": (partial(OwnerLookup(), limit=2), "OwnerLookup", 2),
        }
        candidate, name, limit = candidates[kind]
        registry = tools_from([candidate], validate=True)
        spec = registry.tool_specs[name]
        assert spec.description == "Find the team that owns a service."
        assert spec.to_json_schema() == {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "limit": {"type": "integer", "default": limit},
            },
            "required": ["service"],
        }
        call = ToolCall(tool=name, args={"service": "payments"})
        result = await registry.async_dispatch(call) if async_dispatch else registry.dispatch(call)
        assert result.error is None
        assert result.data == {"owner": "fintech-platform", "limit": limit}
        assert effects == [("payments", limit)]
        invalid = ToolCall(tool=name, args={"service": "payments", "limit": "two"})
        rejected = (
            await registry.async_dispatch(invalid) if async_dispatch else registry.dispatch(invalid)
        )
        assert rejected.error is not None
        assert effects == [("payments", limit)]
        assert candidate("payments") == result.data

    def test_explicit_callable_metadata_overrides_inference(self):
        class Increment:
            def __call__(self, value: int, offset: int = 1) -> dict:
                return {"value": value + offset}

        candidate = partial(Increment(), offset=2)
        spec = tool(candidate, name="add_two", description="Add two to a value.")
        assert spec.name == "add_two"
        assert spec.description == "Add two to a value."
        result = tools_from([spec], validate=True).dispatch(
            ToolCall(tool="add_two", args={"value": 3})
        )
        assert result.error is None
        assert result.data == {"value": 5}

    def test_callable_instance_without_docstring_has_a_description(self):
        class Echo:
            def __call__(self, value: str) -> dict:
                return {"value": value}

        spec = tool(Echo())
        assert spec.name == "Echo"
        assert spec.description == "Call Echo."

    @pytest.mark.parametrize("kind", ["partial", "instance", "partial_instance"])
    async def test_async_bound_callables_are_awaited(self, kind):
        effects = []

        async def lookup_owner(prefix: str, service: str) -> dict:
            effects.append(service)
            return {"owner": prefix + service}

        class OwnerLookup:
            async def __call__(self, service: str) -> dict:
                return await lookup_owner("owner-", service)

        candidates = {
            "partial": (partial(lookup_owner, "owner-"), "lookup_owner"),
            "instance": (OwnerLookup(), "OwnerLookup"),
            "partial_instance": (partial(OwnerLookup()), "OwnerLookup"),
        }
        candidate, name = candidates[kind]
        registry = tools_from([candidate], validate=True)
        result = await registry.async_dispatch(ToolCall(tool=name, args={"service": "payments"}))
        assert result.error is None
        assert result.data == {"owner": "owner-payments"}
        assert effects == ["payments"]

    def test_tools_from_can_customize_done_and_include_think(self) -> None:
        @tool(description="Audit a package.")
        def audit_package(name: str) -> dict:
            return {"name": name}

        registry = tools_from(
            [audit_package],
            include_think=True,
            include_done=True,
            done_parameters={"report": "Structured final report"},
        )

        assert registry.tool_names == ["audit_package", "think", "done"]
        done_result = registry.dispatch(ToolCall(tool="done", args={"report": "all clear"}))
        assert done_result.data == {"status": "completed", "report": "all clear"}
        think_result = registry.dispatch(ToolCall(tool="think", args={"analysis": "plan"}))
        assert think_result.data == {"acknowledged": True, "analysis": "plan"}

    def test_ctx_parameter_is_not_exposed_but_is_injected(self) -> None:
        @tool(description="Warn through context.")
        def warn_user(message: str, ctx: ToolContext) -> dict:
            ctx.warn(f"saw {message}")
            return {"message": message}

        assert "ctx" not in warn_user.parameter_names()
        registry = tools_from([warn_user])
        ctx = ToolContext()
        result = registry.dispatch(ToolCall(tool="warn_user", args={"message": "hello"}), ctx=ctx)

        assert result.error is None
        assert result.warnings == ["saw hello"]

    def test_optional_none_annotation_is_nullable(self) -> None:
        @tool(description="Maybe tag an item.")
        def tag_item(name: str, tag: str | None = None) -> dict:
            return {"name": name, "tag": tag}

        schema = tag_item.to_json_schema()
        assert schema["properties"]["tag"]["type"] == ["string", "null"]
        assert schema["required"] == ["name"]

    @pytest.mark.parametrize("async_dispatch", [False, True])
    @pytest.mark.parametrize(
        ("annotation", "value", "schema"),
        [
            (Annotated[int, "Maximum results"], 3, {"type": "integer"}),
            (str | int, 3, {"type": ["string", "integer"]}),
            (Union[str, int], "item-3", {"type": ["string", "integer"]}),
            (str | int | None, None, {"type": ["string", "integer", "null"]}),
            (Any, {"count": 3}, {}),
            (Any, [3], {}),
            (Any, None, {}),
            (Annotated[Any, "Opaque JSON"], {"count": 3}, {}),
            (
                Literal["fast", "safe"] | None,
                None,
                {"enum": ["fast", "safe", None], "type": ["string", "null"]},
            ),
            (
                Literal["auto", 1, 2],
                1,
                {"enum": ["auto", 1, 2], "type": ["string", "integer"]},
            ),
            (
                Literal["auto", 1, 2],
                2,
                {"enum": ["auto", 1, 2], "type": ["string", "integer"]},
            ),
            (
                Literal[1, 2, "auto"],
                "auto",
                {"enum": [1, 2, "auto"], "type": ["integer", "string"]},
            ),
            (
                Literal["auto", None],
                None,
                {"enum": ["auto", None], "type": ["string", "null"]},
            ),
            (
                Literal[None, "auto"],
                "auto",
                {"enum": [None, "auto"], "type": ["null", "string"]},
            ),
            (Literal[True, 2], 2, {"enum": [True, 2], "type": ["boolean", "integer"]}),
            (Literal[1, False], False, {"enum": [1, False], "type": ["integer", "boolean"]}),
            (Literal[1, False], 1, {"enum": [1, False], "type": ["integer", "boolean"]}),
        ],
    )
    async def test_annotation_schema_and_dispatch_agree(
        self, async_dispatch, annotation, value, schema
    ):
        effects = []

        def echo(value):
            effects.append(value)
            return {"value": value}

        echo.__annotations__["value"] = annotation
        registry = tools_from([echo], validate=True)
        assert registry.tool_specs["echo"].parameters["properties"]["value"] == schema
        call = ToolCall(tool="echo", args={"value": value})

        result = await registry.async_dispatch(call) if async_dispatch else registry.dispatch(call)

        assert result.error is None
        assert result.data == {"value": value}
        assert effects == [value]
        assert registry.dispatch(ToolCall(tool="echo", args={})).error is not None
        assert effects == [value]

    @pytest.mark.parametrize("async_dispatch", [False, True])
    @pytest.mark.parametrize(
        ("annotation", "value"),
        [
            (Annotated[int, "Maximum results"], "3"),
            (str | int, 3.5),
            (str | int, True),
            (Literal["fast", "safe"] | None, "unknown"),
            (Literal["auto", 1, 2], 3),
            (Literal["auto", 1, 2], True),
            (Literal["auto", None], "unknown"),
            (Literal[1, False], True),
            (Literal[True, 2], 1),
            (Literal[0, True], False),
        ],
    )
    async def test_annotation_validation_rejects_invalid_values(
        self, async_dispatch, annotation, value
    ):
        effects = []

        def echo(value):
            effects.append(value)

        echo.__annotations__["value"] = annotation
        registry = tools_from([echo], validate=True)
        call = ToolCall(tool="echo", args={"value": value})

        result = await registry.async_dispatch(call) if async_dispatch else registry.dispatch(call)

        assert result.error is not None
        assert effects == []

    @pytest.mark.parametrize("annotation", [complex, str | list[int]])
    def test_unsupported_annotations_fail_before_registration(self, annotation):
        def echo(value):
            return value

        echo.__annotations__["value"] = annotation
        with pytest.raises(TypeError, match="explicit ToolSpec.parameters schema"):
            tools_from([echo])

    @pytest.mark.parametrize("validate", [False, True])
    @pytest.mark.parametrize("explicit_spec", [False, True])
    def test_positional_only_parameters_fail_before_registration(self, validate, explicit_spec):
        effects = []

        def lookup(value: str, /) -> dict:
            effects.append(value)
            return {"value": value}

        item = (
            ToolSpec(
                name="lookup",
                description="Look up a value.",
                parameters={"value": "str"},
                execute=lookup,
            )
            if explicit_spec
            else lookup
        )
        with pytest.raises(TypeError, match="positional-only.*keyword-compatible"):
            tools_from([item], validate=validate)
        assert effects == []

    def test_contract_audit_reports_positional_only_parameters(self):
        def lookup(value: str, /) -> dict:
            return {"value": value}

        spec = ToolSpec(
            name="lookup",
            description="Look up a value.",
            parameters={"value": "str"},
            execute=lookup,
        )
        assert any("positional-only" in error for error in spec.contract_errors())

    @pytest.mark.parametrize("validate", [False, True])
    @pytest.mark.parametrize(
        "annotation",
        [
            tuple,
            tuple[str, ...],
            set,
            set[str],
            frozenset,
            frozenset[str],
            list[set[str]],
            set[str] | None,
        ],
    )
    def test_non_json_container_annotations_fail_before_registration(self, annotation, validate):
        effects = []

        def overlap(values):
            effects.append(values)
            return values.intersection({"blue"})

        overlap.__annotations__["values"] = annotation
        with pytest.raises(TypeError, match="Python lists.*list parameter.*wrapper"):
            tools_from([overlap], validate=validate)
        assert effects == []

    @pytest.mark.parametrize("async_dispatch", [False, True])
    async def test_json_list_wrapper_can_convert_before_calling_domain_code(self, async_dispatch):
        effects = []

        def overlap(values: set[str]) -> list[str]:
            effects.append(values)
            return sorted(values.intersection({"blue"}))

        def lookup(values: list[str]) -> dict:
            return {"overlap": overlap(set(values))}

        registry = tools_from([lookup], validate=True)
        call = ToolCall(tool="lookup", args={"values": ["blue", "red", "blue"]})
        result = await registry.async_dispatch(call) if async_dispatch else registry.dispatch(call)

        assert result.error is None
        assert result.data == {"overlap": ["blue"]}
        assert effects == [{"blue", "red"}]

    @pytest.mark.parametrize("async_dispatch", [False, True])
    @pytest.mark.parametrize("quantity", ["not-an-integer", True])
    async def test_opt_in_validation_rejects_wrong_types_before_effects(
        self, async_dispatch, quantity
    ):
        effects = []

        @tool
        def echo_quantity(quantity: int) -> dict:
            effects.append(quantity)
            return {"quantity": quantity}

        registry = tools_from([echo_quantity], validate=True)
        call = ToolCall(tool="echo_quantity", args={"quantity": quantity})
        result = await registry.async_dispatch(call) if async_dispatch else registry.dispatch(call)

        assert result.error is not None
        assert "expected int" in result.error
        assert result.error_detail.kind.value == "validation"
        assert effects == []

    @pytest.mark.parametrize(
        ("args", "valid"),
        [
            ({"quantity": None, "mode": "fast"}, True),
            ({"quantity": 0, "mode": "safe"}, True),
            ({"quantity": "bad", "mode": "fast"}, False),
            ({"quantity": 1, "mode": "unknown"}, False),
            ({"mode": "fast"}, False),
        ],
    )
    def test_validation_preserves_nullability_enums_and_required_arguments(self, args, valid):
        def configure(quantity: int | None, mode: Literal["fast", "safe"]) -> dict:
            return {"quantity": quantity, "mode": mode}

        registry = tools_from([configure], validate=True)
        result = registry.dispatch(ToolCall(tool="configure", args=args))

        assert (result.error is None) is valid

    def test_minimal_preset_can_validate_without_changing_the_base_registry(self):
        from looplet import minimal_preset

        @tool
        def echo_quantity(quantity: int) -> dict:
            return {"quantity": quantity}

        call = ToolCall(tool="echo_quantity", args={"quantity": "bad"})
        assert tools_from([echo_quantity]).dispatch(call).data == {"quantity": "bad"}
        with minimal_preset(tools=[echo_quantity], validate=True) as run:
            assert run.tools.dispatch(call).error is not None
