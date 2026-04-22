"""Smoke tests for MCPToolAdapter — MCP protocol adapter."""
from __future__ import annotations

import pytest

from looplet.mcp import MCPToolAdapter

pytestmark = pytest.mark.smoke


class TestMCPToolAdapter:
    def test_importable(self):
        assert MCPToolAdapter is not None

    def test_extract_params_from_schema(self):
        schema = {
            "name": "read_file",
            "description": "Read a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "encoding": {"type": "string", "description": "Encoding"},
                },
                "required": ["path"],
            },
        }
        params = MCPToolAdapter._extract_params(schema)
        assert params == {"path": "string", "encoding": "string"}

    def test_extract_params_empty(self):
        schema = {"name": "noop", "inputSchema": {"type": "object"}}
        params = MCPToolAdapter._extract_params(schema)
        assert params == {}

    def test_context_manager_without_server(self):
        """MCPToolAdapter can be constructed without starting a server."""
        adapter = MCPToolAdapter("echo test")
        assert adapter._started is False

    def test_close_idempotent(self):
        adapter = MCPToolAdapter("echo test")
        adapter.close()  # Should not raise
        adapter.close()  # Idempotent

    def test_from_looplet_import(self):
        from looplet import MCPToolAdapter as MCP
        assert MCP is MCPToolAdapter
