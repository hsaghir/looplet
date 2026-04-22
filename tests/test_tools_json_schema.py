"""Tests for ToolSpec JSON Schema support and introspect()."""

import pytest

pytestmark = pytest.mark.smoke


class TestToolSpecJsonSchema:
    """Test that ToolSpec accepts both simple and JSON Schema parameters."""

    def test_simple_format_is_not_json_schema(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="read", description="Read file",
            parameters={"file_path": "str"},
            execute=lambda *, file_path: {},
        )
        assert not spec.is_json_schema

    def test_json_schema_format_detected(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="read", description="Read file",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to read"},
                },
                "required": ["file_path"],
            },
            execute=lambda *, file_path: {},
        )
        assert spec.is_json_schema

    def test_parameter_names_simple(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="test", description="Test",
            parameters={"a": "str", "b": "int"},
            execute=lambda *, a, b: {},
        )
        assert spec.parameter_names() == ["a", "b"]

    def test_parameter_names_json_schema(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="test", description="Test",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            execute=lambda *, query, limit=10: {},
        )
        assert spec.parameter_names() == ["query", "limit"]

    def test_required_parameters_simple(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="test", description="Test",
            parameters={"a": "str", "b": "int"},
            execute=lambda *, a, b: {},
        )
        # Simple format: all required
        assert spec.required_parameters() == ["a", "b"]

    def test_required_parameters_json_schema(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="test", description="Test",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            execute=lambda *, query, limit=10: {},
        )
        assert spec.required_parameters() == ["query"]

    def test_spec_text_simple(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="read", description="Read a file",
            parameters={"path": "str"},
            execute=lambda *, path: {},
        )
        text = spec.spec_text()
        assert "read" in text
        assert "path: str" in text

    def test_spec_text_json_schema(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="read", description="Read a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string"},
                },
                "required": ["path"],
            },
            execute=lambda *, path, encoding="utf-8": {},
        )
        text = spec.spec_text()
        assert "read" in text
        assert "path: string" in text
        assert "encoding?: string" in text  # optional marked with ?

    def test_to_api_schema_simple(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="test", description="Test tool",
            parameters={"q": "search query"},
            execute=lambda *, q: {},
        )
        schema = spec.to_api_schema()
        assert schema["name"] == "test"
        assert schema["description"] == "Test tool"
        assert "q" in schema["input_schema"]["properties"]

    def test_to_api_schema_json_schema_passthrough(self):
        from looplet.tools import ToolSpec
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }
        spec = ToolSpec(
            name="search", description="Search",
            parameters=params,
            execute=lambda *, query, limit=10: {},
        )
        schema = spec.to_api_schema()
        assert schema["input_schema"] is params  # direct passthrough

    def test_to_json_schema_from_simple(self):
        from looplet.tools import ToolSpec
        spec = ToolSpec(
            name="test", description="Test",
            parameters={"name": "person name", "age": "integer"},
            execute=lambda *, name, age: {},
        )
        schema = spec.to_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["required"] == ["name", "age"]

    def test_to_json_schema_from_json_schema(self):
        from looplet.tools import ToolSpec
        params = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        spec = ToolSpec(
            name="test", description="Test",
            parameters=params,
            execute=lambda *, q: {},
        )
        schema = spec.to_json_schema()
        assert schema == params

    def test_dispatch_works_with_json_schema(self):
        from looplet.tools import BaseToolRegistry, ToolSpec
        from looplet.types import ToolCall
        reg = BaseToolRegistry()
        reg.register(ToolSpec(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            execute=lambda *, query: {"results": [query]},
        ))
        result = reg.dispatch(ToolCall(tool="search", args={"query": "hello"}))
        assert result.error is None
        assert result.data == {"results": ["hello"]}


class TestRegistryIntrospect:
    """Test BaseToolRegistry.introspect() for machine-readable discovery."""

    def test_introspect_empty_registry(self):
        from looplet.tools import BaseToolRegistry
        reg = BaseToolRegistry()
        info = reg.introspect()
        assert info["tool_count"] == 0
        assert info["tools"] == []

    def test_introspect_with_tools(self):
        from looplet.tools import BaseToolRegistry, ToolSpec
        reg = BaseToolRegistry()
        reg.register(ToolSpec(
            name="echo", description="Echo input",
            parameters={"text": "str"},
            execute=lambda *, text: {"echoed": text},
            concurrent_safe=True,
        ))
        reg.register(ToolSpec(
            name="done", description="Finish",
            parameters={"summary": "str"},
            execute=lambda *, summary: {},
        ))
        info = reg.introspect()
        assert info["tool_count"] == 2
        assert len(info["tools"]) == 2

        echo_info = info["tools"][0]
        assert echo_info["name"] == "echo"
        assert echo_info["description"] == "Echo input"
        assert echo_info["concurrent_safe"] is True
        assert echo_info["free"] is False
        assert echo_info["parameters"]["type"] == "object"
        assert "text" in echo_info["parameters"]["properties"]

    def test_introspect_json_schema_passthrough(self):
        from looplet.tools import BaseToolRegistry, ToolSpec
        reg = BaseToolRegistry()
        params = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }
        reg.register(ToolSpec(
            name="search", description="Search items",
            parameters=params,
            execute=lambda *, query, limit=10: {},
        ))
        info = reg.introspect()
        tool_info = info["tools"][0]
        assert tool_info["parameters"] == params
