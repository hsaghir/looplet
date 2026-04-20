"""Tests for openharness.tools — ToolSpec, BaseToolRegistry, register_think_tool."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


# ── ToolSpec tests ────────────────────────────────────────────────


class TestToolSpec:
    def test_creation(self):
        from openharness.tools import ToolSpec
        spec = ToolSpec(
            name="echo",
            description="Echo the input back",
            parameters={"text": "The text to echo"},
            execute=lambda text="": text,
        )
        assert spec.name == "echo"
        assert spec.description == "Echo the input back"
        assert "text" in spec.parameters
        assert spec.concurrent_safe is False
        assert spec.free is False

    def test_concurrent_safe_default(self):
        from openharness.tools import ToolSpec
        spec = ToolSpec(name="x", description="d", parameters={}, execute=lambda: None)
        assert spec.concurrent_safe is False

    def test_free_default(self):
        from openharness.tools import ToolSpec
        spec = ToolSpec(name="x", description="d", parameters={}, execute=lambda: None)
        assert spec.free is False

    def test_to_api_schema(self):
        from openharness.tools import ToolSpec
        spec = ToolSpec(
            name="search",
            description="Search for events",
            parameters={"query": "SQL query string", "limit": "Max rows"},
            execute=lambda query="", limit=10: [],
        )
        schema = spec.to_api_schema()
        assert schema["name"] == "search"
        assert schema["description"] == "Search for events"
        assert "input_schema" in schema
        props = schema["input_schema"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "limit" in props

    def test_spec_text(self):
        from openharness.tools import ToolSpec
        spec = ToolSpec(
            name="lookup",
            description="Look up a value",
            parameters={"key": "The key"},
            execute=lambda key="": key,
        )
        text = spec.spec_text()
        assert "lookup" in text
        assert "key" in text
        assert "Look up a value" in text


# ── BaseToolRegistry tests ────────────────────────────────────────


class TestBaseToolRegistry:
    def _make_registry_with_echo(self):
        from openharness.tools import BaseToolRegistry, ToolSpec
        reg = BaseToolRegistry()
        reg.register(ToolSpec(
            name="echo",
            description="Echo back",
            parameters={"text": "text to echo"},
            execute=lambda text="": {"echoed": text},
        ))
        return reg

    def test_register_and_tool_names(self):
        reg = self._make_registry_with_echo()
        assert "echo" in reg.tool_names

    def test_dispatch_success(self):
        from openharness.types import ToolCall
        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "hello"})
        result = reg.dispatch(call)
        assert result.error is None
        assert result.data == {"echoed": "hello"}
        assert result.tool == "echo"

    def test_dispatch_timing(self):
        from openharness.types import ToolCall
        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "hi"})
        result = reg.dispatch(call)
        assert result.duration_ms >= 0.0

    def test_dispatch_unknown_tool(self):
        from openharness.tools import BaseToolRegistry
        from openharness.types import ToolCall
        reg = BaseToolRegistry()
        call = ToolCall(tool="nonexistent", args={})
        result = reg.dispatch(call)
        assert result.error is not None
        assert "Unknown tool" in result.error

    def test_dispatch_error_handling(self):
        from openharness.tools import BaseToolRegistry, ToolSpec
        from openharness.types import ToolCall
        reg = BaseToolRegistry()
        reg.register(ToolSpec(
            name="boom",
            description="Always raises",
            parameters={},
            execute=lambda: (_ for _ in ()).throw(ValueError("kaboom")),
        ))
        call = ToolCall(tool="boom", args={})
        result = reg.dispatch(call)
        assert result.error is not None
        assert "ValueError" in result.error
        assert "kaboom" in result.error
        assert result.data is None

    def test_dispatch_call_id_propagated(self):
        from openharness.types import ToolCall
        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "x"}, call_id="my-id-123")
        result = reg.dispatch(call)
        assert result.call_id == "my-id-123"

    def test_dispatch_strips_dunder_args(self):
        from openharness.types import ToolCall
        reg = self._make_registry_with_echo()
        call = ToolCall(tool="echo", args={"text": "hi", "__internal": "secret"})
        result = reg.dispatch(call)
        assert result.error is None

    def test_tool_catalog_text(self):
        reg = self._make_registry_with_echo()
        catalog = reg.tool_catalog_text()
        assert "Available tools" in catalog
        assert "echo" in catalog

    def test_to_api_schema(self):
        from openharness.tools import BaseToolRegistry
        reg = self._make_registry_with_echo()
        schemas = reg.tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"


# ── Batch dispatch tests ──────────────────────────────────────────


class TestBatchDispatch:
    def _make_registry(self):
        from openharness.tools import BaseToolRegistry, ToolSpec
        reg = BaseToolRegistry()
        reg.register(ToolSpec(
            name="read",
            description="Read-only concurrent-safe",
            parameters={"key": "key"},
            execute=lambda key="": f"value:{key}",
            concurrent_safe=True,
        ))
        reg.register(ToolSpec(
            name="write",
            description="Write — serial only",
            parameters={"key": "key", "val": "value"},
            execute=lambda key="", val="": f"wrote:{key}={val}",
            concurrent_safe=False,
        ))
        return reg

    def test_dispatch_batch_empty(self):
        from openharness.tools import BaseToolRegistry
        reg = BaseToolRegistry()
        assert reg.dispatch_batch([]) == []

    def test_dispatch_batch_returns_all_results(self):
        from openharness.types import ToolCall
        reg = self._make_registry()
        calls = [
            ToolCall(tool="read", args={"key": "a"}),
            ToolCall(tool="read", args={"key": "b"}),
            ToolCall(tool="write", args={"key": "c", "val": "1"}),
        ]
        results = reg.dispatch_batch(calls)
        assert len(results) == 3
        assert all(r.error is None for r in results)

    def test_partition_concurrent_vs_serial(self):
        from openharness.types import ToolCall
        reg = self._make_registry()
        calls = [
            ToolCall(tool="read", args={"key": "a"}),
            ToolCall(tool="read", args={"key": "b"}),
            ToolCall(tool="write", args={"key": "c", "val": "1"}),
        ]
        batches = reg._partition_calls(calls)
        # First batch: 2 concurrent reads
        assert batches[0]["concurrent"] is True
        assert len(batches[0]["calls"]) == 2
        # Second batch: 1 serial write
        assert batches[1]["concurrent"] is False
        assert len(batches[1]["calls"]) == 1

    def test_concurrent_batch_dispatches_parallel(self):
        from openharness.types import ToolCall
        reg = self._make_registry()
        calls = [ToolCall(tool="read", args={"key": str(i)}) for i in range(5)]
        results = reg._dispatch_concurrent_batch(calls)
        assert len(results) == 5
        assert all(r.error is None for r in results)


# ── Think tool tests ──────────────────────────────────────────────


class TestThinkTool:
    def test_think_tool_registered(self):
        from openharness.tools import BaseToolRegistry, register_think_tool
        reg = BaseToolRegistry()
        register_think_tool(reg)
        assert "think" in reg.tool_names

    def test_think_tool_is_free(self):
        from openharness.tools import BaseToolRegistry, register_think_tool
        reg = BaseToolRegistry()
        register_think_tool(reg)
        spec = reg._tools["think"]
        assert spec.free is True

    def test_think_tool_is_concurrent_safe(self):
        from openharness.tools import BaseToolRegistry, register_think_tool
        reg = BaseToolRegistry()
        register_think_tool(reg)
        spec = reg._tools["think"]
        assert spec.concurrent_safe is True

    def test_think_tool_dispatch(self):
        from openharness.tools import BaseToolRegistry, register_think_tool
        from openharness.types import ToolCall
        reg = BaseToolRegistry()
        register_think_tool(reg)
        call = ToolCall(tool="think", args={"analysis": "weighing pros and cons"})
        result = reg.dispatch(call)
        assert result.error is None
        assert result.data == {"acknowledged": True, "analysis": "weighing pros and cons"}

    def test_think_docstring_no_security_terms(self):
        import inspect

        from openharness.tools import register_think_tool
        doc = inspect.getdoc(register_think_tool) or ""
        lower_doc = doc.lower()
        for term in ["brute force", "vpn", "lateral movement", "security"]:
            assert term not in lower_doc, f"Security term '{term}' found in think tool docstring"

    def test_think_description_no_security_terms(self):
        from openharness.tools import BaseToolRegistry, register_think_tool
        reg = BaseToolRegistry()
        register_think_tool(reg)
        desc = reg._tools["think"].description.lower()
        for term in ["brute force", "vpn", "lateral movement"]:
            assert term not in desc, f"Security term '{term}' in think tool description"


# ── Export check ──────────────────────────────────────────────────


class TestExports:
    def test_all_tools_exported_from_openharness(self):
        import openharness as oh
        for name in ["ToolSpec", "BaseToolRegistry"]:
            assert hasattr(oh, name), f"{name} not exported from openharness"

    def test_register_think_tool_importable(self):
        from openharness.tools import register_think_tool
        assert register_think_tool is not None
