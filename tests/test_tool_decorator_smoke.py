"""Tests for decorator-first tool construction."""

from __future__ import annotations

from typing import Literal

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
