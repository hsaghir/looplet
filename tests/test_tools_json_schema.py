"""Tests for ToolSpec JSON Schema support and introspect()."""

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("shape", ["schema", "callable", "shorthand"])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("content", ["", " ", "\n"])
async def test_blank_content_follows_declared_schema(
    tmp_path, shape, validate, async_mode, content
):
    from looplet import ToolCall, ToolSpec, tools_from

    destination = tmp_path / "written.txt"
    observed = []

    def write_text(content: str) -> dict:
        observed.append(content)
        destination.write_text(content, encoding="utf-8")
        return {"bytes": len(content.encode("utf-8"))}

    if shape == "callable":
        registry = tools_from([write_text], validate=validate)
    else:
        parameters = {"content": "Text to write"}
        if shape == "schema":
            parameters = {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            }
        registry = tools_from(
            [ToolSpec("write_text", "Write exact content", parameters, write_text)],
            validate=validate,
        )
    call = ToolCall(tool="write_text", args={"content": content})
    result = await registry.async_dispatch(call) if async_mode else registry.dispatch(call)
    if shape == "shorthand":
        assert result.error_detail.kind.value == "validation"
        assert observed == []
        assert not destination.exists()
    else:
        assert result.error is None
        assert observed == [content]
        assert destination.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize("async_mode", [False, True])
async def test_nonblank_content_is_a_tool_owned_validation_rule(async_mode):
    from looplet import ToolCall, ToolValidationError, tools_from

    observed = []

    def publish(content: str) -> dict:
        if not content.strip():
            raise ToolValidationError("content must contain non-whitespace text")
        observed.append(content)
        return {"content": content}

    registry = tools_from([publish], validate=True)
    for content in ("", " ", "\n"):
        call = ToolCall(tool="publish", args={"content": content})
        result = await registry.async_dispatch(call) if async_mode else registry.dispatch(call)
        assert result.error_detail.kind.value == "validation"
        assert result.error_detail.retriable is False
        assert observed == []
    call = ToolCall(tool="publish", args={"content": "actual content"})
    result = await registry.async_dispatch(call) if async_mode else registry.dispatch(call)
    assert result.error is None
    assert observed == ["actual content"]


@pytest.mark.parametrize("shape", ["function", "partial", "instance"])
def test_input_inference_ignores_unused_context_and_return_annotations(shape):
    from functools import partial

    from looplet import ToolCall, tool, tools_from

    namespace = {"__name__": __name__}
    exec(
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from datetime import date as ReturnOnly\n"
        "    from looplet import ToolContext as ContextOnly\n"
        "def lookup(query: str, prefix: str = 'plain', ctx: ContextOnly = None) -> ReturnOnly:\n"
        "    return {'query': query}\n"
        "class Lookup:\n"
        "    def __call__(self, query: str, ctx: ContextOnly = None) -> ReturnOnly:\n"
        "        return {'query': query}\n",
        namespace,
    )
    target = namespace["lookup"]
    if shape == "partial":
        target = partial(target, prefix="bound")
    elif shape == "instance":
        target = namespace["Lookup"]()
    spec = tool(target)
    schema = spec.to_json_schema()
    assert schema["properties"]["query"] == {"type": "string"}
    assert schema["required"] == ["query"]
    assert "ctx" not in schema["properties"]
    result = tools_from([spec], validate=True).dispatch(
        ToolCall(tool=spec.name, args={"query": "actual input"})
    )
    assert result.error is None
    assert result.data == {"query": "actual input"}


@pytest.mark.parametrize("shape", ["function", "partial", "instance"])
def test_input_inference_uses_wrapped_callable_annotation_globals(shape):
    from functools import partial

    from looplet import ToolCall, tool, tools_from

    decorators = {"__name__": __name__ + ".decorators"}
    exec(
        "from functools import wraps\n"
        "def decorate(fn):\n"
        "    @wraps(fn)\n"
        "    def wrapped(*args, **kwargs):\n"
        "        return fn(*args, **kwargs)\n"
        "    return wrapped\n",
        decorators,
    )
    owner = {"__name__": __name__, "decorate": decorators["decorate"]}
    exec(
        "from __future__ import annotations\n"
        "from typing import Literal\n"
        "Choice = Literal['first', 'second']\n"
        "@decorate\n"
        "def choose(value: Choice, prefix: str = 'plain') -> dict:\n"
        "    return {'choice': value}\n"
        "class Choose:\n"
        "    @decorate\n"
        "    def __call__(self, value: Choice) -> dict:\n"
        "        return {'choice': value}\n",
        owner,
    )
    target = owner["choose"]
    if shape == "partial":
        target = partial(target, prefix="bound")
    elif shape == "instance":
        target = owner["Choose"]()
    spec = tool(target)
    assert spec.to_json_schema()["properties"]["value"] == {
        "type": "string",
        "enum": ["first", "second"],
    }
    result = tools_from([spec]).dispatch(ToolCall(tool=spec.name, args={"value": "first"}))
    assert result.error is None
    assert result.data == {"choice": "first"}


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("width", [2, 8, 32])
def test_dispatch_normalizes_parameter_schema_once(monkeypatch, validate, width):
    from looplet import ToolCall, ToolSpec, tools_from

    spec = ToolSpec(
        name="work",
        description="Return named fields",
        parameters={f"field_{index}": {"type": "integer"} for index in range(width)},
        execute=lambda **values: values,
    )
    registry = tools_from([spec], validate=validate)
    normalize = spec.to_json_schema
    calls = []

    def counted_schema():
        calls.append(True)
        return normalize()

    monkeypatch.setattr(spec, "to_json_schema", counted_schema)
    arguments = {f"field_{index}": index for index in range(width)}
    result = registry.dispatch(ToolCall(tool="work", args=arguments))
    assert result.error is None
    assert result.data == arguments
    assert len(calls) == 1


@pytest.mark.parametrize("async_mode", [False, True])
async def test_inferred_argument_policy_survives_cartridge_round_trip(tmp_path, async_mode):
    from looplet import ToolCall, cartridge_to_preset, preset_to_cartridge, tools_from

    source = tmp_path / "source"
    source.mkdir()
    (source / "cartridge.json").write_text('{"name":"typed","schema_version":2}\n')
    (source / "config.yaml").write_text("max_steps: 2\ndone_tool: work\n")
    tool_directory = source / "tools" / "work"
    tool_directory.mkdir(parents=True)
    (tool_directory / "tool.yaml").write_text("name: work\nparameters:\n  value: {type: string}\n")
    (tool_directory / "execute.py").write_text("def execute(*, value): return {'value': value}\n")
    with cartridge_to_preset(source, strict=True) as preset:
        spec = preset.tools.tool_specs["work"]
        preset.tools = tools_from([spec], validate=True)
        exported = tmp_path / "exported"
        preset_to_cartridge(preset, exported, name="typed", strict=True)

    with cartridge_to_preset(exported, strict=True) as loaded:
        spec = loaded.tools.tool_specs["work"]
        spec.parameters["value"]["type"] = "integer"
        accepted = ToolCall(tool="work", args={"value": 7})
        result = (
            await loaded.tools.async_dispatch(accepted)
            if async_mode
            else loaded.tools.dispatch(accepted)
        )
        assert result.error is None
        assert result.data == {"value": 7}
        rejected = ToolCall(tool="work", args={"value": "old"})
        result = (
            await loaded.tools.async_dispatch(rejected)
            if async_mode
            else loaded.tools.dispatch(rejected)
        )
        assert result.error_detail.kind.value == "validation"


class TestToolSpecJsonSchema:
    """Test that ToolSpec accepts both simple and JSON Schema parameters."""

    def test_simple_format_is_not_json_schema(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="read",
            description="Read file",
            parameters={"file_path": "str"},
            execute=lambda *, file_path: {},
        )
        assert not spec.is_json_schema

    def test_json_schema_format_detected(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="read",
            description="Read file",
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
            name="test",
            description="Test",
            parameters={"a": "str", "b": "int"},
            execute=lambda *, a, b: {},
        )
        assert spec.parameter_names() == ["a", "b"]

    def test_parameter_names_json_schema(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="test",
            description="Test",
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
            name="test",
            description="Test",
            parameters={"a": "str", "b": "int"},
            execute=lambda *, a, b: {},
        )
        # Simple format: all required
        assert spec.required_parameters() == ["a", "b"]

    def test_required_parameters_json_schema(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="test",
            description="Test",
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

    def test_contract_errors_accept_matching_optional_signature(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
            execute=lambda *, query, limit=10: {"query": query, "limit": limit},
        )

        assert spec.contract_errors() == []

    def test_contract_errors_reject_mismatched_requiredness(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            execute=lambda *, query="default": {"query": query},
        )

        errors = spec.contract_errors()

        assert errors == ["callable parameter 'query' has a default but schema marks it required"]

    def test_contract_errors_reject_undeclared_callable_parameter(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="search",
            description="Search",
            parameters={"query": "query"},
            execute=lambda *, query, limit: {"query": query, "limit": limit},
        )

        assert spec.contract_errors() == ["callable parameter 'limit' is not declared"]

    def test_retry_contract_defaults_are_conservative(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec("read", "Read", {}, lambda: {})

        assert spec.idempotency == "unknown"
        assert spec.retryable is False

    def test_retry_contract_round_trips_on_cartridge_tool(self, tmp_path):
        from looplet import ToolSpec, cartridge_to_preset, preset_to_cartridge

        source = tmp_path / "source"
        source.mkdir()
        (source / "cartridge.json").write_text('{"name":"x","schema_version":2}\n')
        (source / "config.yaml").write_text("max_steps: 2\ndone_tool: done\n")
        (source / "tools" / "done").mkdir(parents=True)
        (source / "tools" / "done" / "tool.yaml").write_text(
            "name: done\nparameters:\n  summary: {type: string}\n"
        )
        (source / "tools" / "done" / "execute.py").write_text(
            "def execute(*, summary): return {'summary': summary}\n"
        )
        preset = cartridge_to_preset(source)
        preset.tools._tools["done"].idempotency = "safe"
        preset.tools._tools["done"].retryable = True
        output = tmp_path / "output"
        preset_to_cartridge(preset, output, name="x")
        text = (output / "tools" / "done" / "tool.yaml").read_text()
        assert "idempotency: safe" in text
        assert "retryable: true" in text

    def test_spec_text_simple(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="read",
            description="Read a file",
            parameters={"path": "str"},
            execute=lambda *, path: {},
        )
        text = spec.spec_text()
        assert "read" in text
        assert "path: str" in text

    def test_spec_text_json_schema(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="read",
            description="Read a file",
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
            name="test",
            description="Test tool",
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
            name="search",
            description="Search",
            parameters=params,
            execute=lambda *, query, limit=10: {},
        )
        schema = spec.to_api_schema()
        assert schema["input_schema"] is params  # direct passthrough

    def test_to_json_schema_from_simple(self):
        from looplet.tools import ToolSpec

        spec = ToolSpec(
            name="test",
            description="Test",
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
            name="test",
            description="Test",
            parameters=params,
            execute=lambda *, q: {},
        )
        schema = spec.to_json_schema()
        assert schema == params

    def test_compatibility_schemas_share_requiredness_and_rich_types(self):
        from looplet import BaseToolRegistry, ToolCall, ToolSpec

        parameters = {
            "mode": {"type": "string", "enum": ["safe", "fast"]},
            "rows": {"type": "array", "items": {"type": "integer"}},
            "limit": {"type": "integer", "minimum": 1, "default": 10},
            "options": "(optional) object",
        }
        spec = ToolSpec(
            "query",
            "Query rows",
            parameters,
            lambda *, mode, rows, limit=10, options=None: {"rows": rows, "limit": limit},
        )
        registry = BaseToolRegistry()
        registry.register(spec)
        canonical = spec.to_json_schema()

        assert canonical["required"] == spec.required_parameters() == ["mode", "rows"]
        assert canonical["properties"]["rows"] == parameters["rows"]
        assert canonical["properties"]["mode"] == parameters["mode"]
        assert canonical["properties"]["limit"] == parameters["limit"]
        assert canonical["properties"]["options"]["type"] == "object"
        assert registry.tool_schemas()[0]["input_schema"] == canonical
        assert registry.introspect()["tools"][0]["parameters"] == canonical
        assert registry.dispatch(ToolCall("query", {"mode": "safe", "rows": [1]})).data == {
            "rows": [1],
            "limit": 10,
        }

    @pytest.mark.parametrize(
        ("descriptor", "expected"),
        [("int", "integer"), ("bool", "boolean"), ("list", "array"), ("file path", "string")],
    )
    def test_shorthand_type_normalization(self, descriptor, expected):
        from looplet import ToolSpec

        spec = ToolSpec("echo", "Echo", {"value": descriptor}, lambda *, value: value)

        assert spec.to_json_schema()["properties"]["value"]["type"] == expected

    def test_signature_optionality_and_edits_reach_every_schema_view(self):
        from looplet import BaseToolRegistry, ToolSpec

        spec = ToolSpec(
            "echo",
            "Echo",
            {"value": "str", "limit": "int"},
            lambda *, value, limit=10: value,
            infer_optional_from_signature=True,
        )
        registry = BaseToolRegistry()
        registry.register(spec)
        assert spec.required_parameters() == ["value"]
        assert spec.to_api_schema()["input_schema"]["required"] == ["value"]
        spec.parameters["limit"] = {"type": "integer", "default": 5, "maximum": 20}

        assert spec.to_json_schema()["properties"]["limit"]["maximum"] == 20
        assert registry.introspect()["tools"][0]["parameters"] == spec.to_json_schema()
        assert spec.contract_errors() == []

    def test_dispatch_works_with_json_schema(self):
        from looplet.tools import BaseToolRegistry, ToolSpec
        from looplet.types import ToolCall

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
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
            )
        )
        result = reg.dispatch(ToolCall(tool="search", args={"query": "hello"}))
        assert result.error is None
        assert result.data == {"results": ["hello"]}

    def test_required_nullable_callable_accepts_explicit_null(self):
        from looplet import tools_from
        from looplet.types import ToolCall

        received = []

        def nullable(value: int | None) -> dict:
            received.append(value)
            return {"value": value}

        registry = tools_from([nullable])
        spec = registry.tool_specs["nullable"]
        assert spec.to_json_schema()["properties"]["value"]["type"] == ["integer", "null"]
        assert spec.contract_errors() == []

        result = registry.dispatch(ToolCall(tool="nullable", args={"value": None}))

        assert result.error is None
        assert result.data == {"value": None}
        assert received == [None]
        missing = registry.dispatch(ToolCall(tool="nullable", args={}))
        assert missing.error is not None
        assert received == [None]

    @pytest.mark.parametrize("parameter_type", ["null", ["integer", "null"]])
    def test_explicit_nullable_schema_accepts_null(self, parameter_type):
        from looplet import BaseToolRegistry, ToolSpec
        from looplet.types import ToolCall

        registry = BaseToolRegistry()
        registry.register(
            ToolSpec(
                "nullable",
                "Accept an explicit null.",
                {"value": {"type": parameter_type}},
                lambda *, value: {"value": value},
            )
        )

        result = registry.dispatch(ToolCall(tool="nullable", args={"value": None}))

        assert result.error is None
        assert result.data == {"value": None}

    @pytest.mark.parametrize("arguments", [{}, {"value": None}])
    def test_non_nullable_required_input_rejects_missing_or_null(self, arguments):
        from looplet import tools_from
        from looplet.types import ToolCall

        received = []

        def echo(value: str) -> dict:
            received.append(value)
            return {"value": value}

        result = tools_from([echo]).dispatch(ToolCall(tool="echo", args=arguments))

        assert result.error is not None
        assert received == []


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
        reg.register(
            ToolSpec(
                name="echo",
                description="Echo input",
                parameters={"text": "str"},
                execute=lambda *, text: {"echoed": text},
                concurrent_safe=True,
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="Finish",
                parameters={"summary": "str"},
                execute=lambda *, summary: {},
            )
        )
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
        reg.register(
            ToolSpec(
                name="search",
                description="Search items",
                parameters=params,
                execute=lambda *, query, limit=10: {},
            )
        )
        info = reg.introspect()
        tool_info = info["tools"][0]
        assert tool_info["parameters"] == params
