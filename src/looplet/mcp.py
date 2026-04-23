"""MCP (Model Context Protocol) tool adapter.

Wraps tools from an MCP server as looplet :class:`ToolSpec` instances
so they can be registered in a :class:`BaseToolRegistry` and used in
``composable_loop`` like any other tool.

MCP servers expose tools via JSON-RPC over stdio. This adapter:

1. Starts the server as a subprocess.
2. Sends ``initialize`` + ``tools/list`` to discover available tools.
3. Creates a :class:`ToolSpec` for each tool with the server's name,
   description, and JSON-schema parameters.
4. Tool execution sends ``tools/call`` and returns the result.

Usage::

    from looplet.mcp import MCPToolAdapter

    # Start an MCP server and load its tools
    adapter = MCPToolAdapter("npx @modelcontextprotocol/server-filesystem /tmp")
    adapter.register_all(my_registry)

    # Or selectively
    for spec in adapter.tools():
        if spec.name in ("read_file", "write_file"):
            my_registry.register(spec)

    # Clean up when done
    adapter.close()

Works with any MCP-compliant server (filesystem, GitHub, Slack, etc.).
No MCP SDK dependency — communicates via raw JSON-RPC over stdio.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from typing import Any

from looplet.tools import BaseToolRegistry, ToolSpec

__all__ = ["MCPToolAdapter"]

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Adapt an MCP server's tools into looplet ToolSpecs.

    Args:
        command: Shell command to start the MCP server
            (e.g. ``"npx @modelcontextprotocol/server-filesystem /tmp"``).
        env: Optional environment variables for the subprocess.
        timeout: Seconds to wait for server responses (default 30).

    The adapter starts the server on first use and keeps it running
    until :meth:`close` is called. Use as a context manager::

        with MCPToolAdapter("npx @mcp/server-fs /tmp") as mcp:
            mcp.register_all(registry)
            ...
    """

    def __init__(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._command = command
        self._env = env
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._tool_schemas: list[dict[str, Any]] = []
        self._started = False

    def __enter__(self) -> "MCPToolAdapter":
        self._ensure_started()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Public API ───────────────────────────────────────────────

    def tools(self) -> list[ToolSpec]:
        """Return a ToolSpec for each tool the MCP server exposes."""
        self._ensure_started()
        specs = []
        for schema in self._tool_schemas:
            name = schema["name"]
            desc = schema.get("description", "")
            params = self._extract_params(schema)
            specs.append(
                ToolSpec(
                    name=name,
                    description=desc,
                    parameters=params,
                    execute=self._make_executor(name),
                )
            )
        return specs

    def register_all(self, registry: BaseToolRegistry) -> int:
        """Register all MCP tools into the given registry.

        Returns the number of tools registered.
        """
        specs = self.tools()
        for spec in specs:
            registry.register(spec)
        return len(specs)

    def close(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()
            self._proc = None
            self._started = False

    # ── Internals ────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._proc = subprocess.Popen(
            self._command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
        )
        try:
            # Initialize the MCP session
            init_resp = self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "looplet", "version": "0.1"},
                },
            )
            if init_resp is None:
                raise RuntimeError(f"MCP server failed to initialize: {self._command}")
            # Send initialized notification
            self._send_notification("notifications/initialized", {})
            # List tools
            tools_resp = self._send_request("tools/list", {})
            if tools_resp and "tools" in tools_resp:
                self._tool_schemas = tools_resp["tools"]
                logger.info("MCP: loaded %d tools from %s", len(self._tool_schemas), self._command)
            self._started = True
        except BaseException:
            # Any failure during init leaves a live subprocess —
            # clean it up so callers don't leak a server per retry.
            self.close()
            raise

    def _send_request(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and wait for the response."""
        with self._lock:
            self._request_id += 1
            msg = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            return self._send_and_receive(msg)

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        with self._lock:
            msg = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
            self._write_message(msg)

    def _send_and_receive(self, msg: dict) -> dict | None:
        """Write a message and read the response."""
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return None
        self._write_message(msg)
        return self._read_message()

    def _write_message(self, msg: dict) -> None:
        """Write a JSON-RPC message with Content-Length header."""
        if self._proc is None or self._proc.stdin is None:
            return
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)
        self._proc.stdin.flush()

    def _read_message(self) -> dict | None:
        """Read a JSON-RPC message with Content-Length header."""
        if self._proc is None or self._proc.stdout is None:
            return None
        # Read headers
        content_length = 0
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            line_str = line.decode("utf-8").strip()
            if not line_str:
                break  # Empty line = end of headers
            if line_str.startswith("Content-Length:"):
                content_length = int(line_str.split(":")[1].strip())
        if content_length == 0:
            return None
        body = self._proc.stdout.read(content_length)
        data = json.loads(body.decode("utf-8"))
        if "error" in data:
            logger.warning("MCP error: %s", data["error"])
            return None
        return data.get("result", data)

    def _make_executor(self, tool_name: str) -> Any:
        """Create an execute function that calls the MCP server."""

        def execute(**kwargs: Any) -> dict:
            resp = self._send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": kwargs,
                },
            )
            if resp is None:
                return {"error": f"MCP tool '{tool_name}' returned no response"}
            # MCP returns content as list of content blocks
            content = resp.get("content", [])
            if isinstance(content, list) and len(content) == 1:
                block = content[0]
                if isinstance(block, dict) and block.get("type") == "text":
                    # Try to parse as JSON for structured data
                    text = block.get("text", "")
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        return {"text": text}
            return {"content": content}

        return execute

    @staticmethod
    def _extract_params(schema: dict) -> dict[str, str]:
        """Convert MCP JSON-schema parameters to ToolSpec parameter dict."""
        input_schema = schema.get("inputSchema", {})
        props = input_schema.get("properties", {})
        params: dict[str, str] = {}
        for name, prop in props.items():
            params[name] = prop.get("type", "str")
        return params
