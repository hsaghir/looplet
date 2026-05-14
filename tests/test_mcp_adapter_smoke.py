"""Smoke tests for MCPToolAdapter — MCP protocol adapter."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

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


# ── Real-subprocess regression test for NDJSON framing ──────────
#
# Per MCP spec (https://modelcontextprotocol.io/.../basic/transports#stdio):
# "Messages are delimited by newlines, and MUST NOT contain embedded newlines."
# Earlier looplet versions used LSP-style ``Content-Length:`` framing, which
# silently failed against every real MCP server. This test spawns a tiny
# Python stdio MCP server and round-trips ``initialize`` + ``tools/list`` +
# ``tools/call`` so the wire format is exercised end-to-end.

_TINY_MCP_SERVER = textwrap.dedent(
    """\
    import json, sys
    TOOLS = [{
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]},
    }]
    def respond(msg_id, result=None, error=None):
        out = {"jsonrpc": "2.0", "id": msg_id}
        if error is not None: out["error"] = error
        else: out["result"] = result
        sys.stdout.write(json.dumps(out) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        req = json.loads(line); m = req.get("method"); i = req.get("id")
        if m == "initialize":
            respond(i, {"protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "tiny", "version": "0.1"}})
        elif m == "notifications/initialized":
            continue
        elif m == "tools/list":
            respond(i, {"tools": TOOLS})
        elif m == "tools/call":
            args = req["params"]["arguments"]
            respond(i, {"content": [{"type": "text",
                        "text": str(int(args["a"]) + int(args["b"]))}],
                       "isError": False})
        else:
            respond(i, error={"code": -32601, "message": "unknown"})
    """
)


def test_real_subprocess_ndjson_round_trip(tmp_path: Path) -> None:
    """End-to-end: spawn a real Python stdio server, discover one tool,
    dispatch it, get the right answer back. Pinned regression for the
    NDJSON framing fix (was Content-Length: which broke every server)."""
    server = tmp_path / "tiny_mcp.py"
    server.write_text(_TINY_MCP_SERVER)
    cmd = f"{sys.executable} {server}"

    with MCPToolAdapter(cmd, timeout=10.0) as adapter:
        specs = adapter.tools()
        assert [s.name for s in specs] == ["add"]
        # Dispatch through the real adapter callable.
        result = specs[0].execute(a=7, b=5)
        # Result text is "12"; the adapter MAY wrap or unwrap the
        # MCP content envelope — assert the answer survives in any case.
        assert "12" in str(result)
