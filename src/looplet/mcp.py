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
No MCP SDK dependency - communicates via raw JSON-RPC over stdio.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
from collections.abc import Callable
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
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self._command = command
        self._env = env
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._reader_lock = threading.Lock()
        self._reader_threads: set[threading.Thread] = set()
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
                    execute=self._make_executor(name, params),
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
        proc = self._proc
        self._proc = None
        self._started = False
        if proc is not None:
            wait_timeout = min(5.0, max(0.1, self._timeout))
            self._stop_process_tree(proc, wait_timeout=wait_timeout)
            with self._reader_lock:
                readers = list(self._reader_threads)
            for reader in readers:
                reader.join(timeout=wait_timeout)
            readers_alive = any(reader.is_alive() for reader in readers)

            for stream in (
                proc.stdin,
                None if readers_alive else proc.stdout,
                None if readers_alive else proc.stderr,
            ):
                if stream is not None:
                    try:
                        stream.close()
                    except (BrokenPipeError, OSError):
                        pass

    # ── Internals ────────────────────────────────────────────────

    def _ensure_started(self) -> None:
        if self._started:
            return
        process_options: dict[str, Any] = {}
        if os.name == "posix":
            process_options["start_new_session"] = True
        elif os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        self._proc = subprocess.Popen(
            self._command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            **process_options,
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
                proc = self._proc
                rc = proc.poll() if proc is not None else None
                stderr_tail = self._stderr_tail_if_exited(proc)
                raise RuntimeError(
                    f"MCP server failed to initialize: {self._command!r} "
                    f"(exit_code={rc}); stderr: {stderr_tail!r}"
                )
            # Send initialized notification
            self._send_notification("notifications/initialized", {})
            # List tools
            tools_resp = self._send_request("tools/list", {})
            if tools_resp and "tools" in tools_resp:
                self._tool_schemas = tools_resp["tools"]
                logger.info("MCP: loaded %d tools from %s", len(self._tool_schemas), self._command)
            self._started = True
        except BaseException as exc:
            # Any failure during init leaves a live subprocess -
            # clean it up so callers don't leak a server per retry.
            proc = self._proc
            rc = proc.poll() if proc is not None else None
            stderr_tail = self._stderr_tail_if_exited(proc)
            self.close()
            if isinstance(exc, (BrokenPipeError, OSError, TimeoutError)):
                raise RuntimeError(
                    f"MCP server failed to initialize: {self._command!r} "
                    f"(exit_code={rc}); stderr: {stderr_tail!r}"
                ) from exc
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
        try:
            self._write_message(msg)
            return self._read_message()
        except OSError:
            self.close()
            raise

    def _write_message(self, msg: dict) -> None:
        """Write a JSON-RPC message as a single newline-delimited line.

        MCP stdio transport uses newline-delimited JSON (NDJSON), per the
        spec: "Messages are delimited by newlines, and MUST NOT contain
        embedded newlines." The line MUST be UTF-8.
        """
        if self._proc is None or self._proc.stdin is None:
            return
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _read_message(self) -> dict | None:
        """Read one newline-delimited JSON-RPC message from the server.

        Returns the parsed ``result`` payload, or ``None`` on EOF / a
        JSON-RPC error response (errors are logged at WARNING).
        """
        if self._proc is None or self._proc.stdout is None:
            return None
        line = self._run_io_with_timeout(
            self._proc.stdout.readline,
            operation="response",
        )
        if not line:
            return None
        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("MCP non-JSON line on stdout (%s): %r", exc, line[:200])
            return None
        if "error" in data:
            logger.warning("MCP error: %s", data["error"])
            return None
        return data.get("result", data)

    def _stderr_tail_if_exited(self, proc: subprocess.Popen | None) -> str:
        """Read one bounded stderr chunk after a child has exited."""
        if proc is None or proc.stderr is None or proc.poll() is None:
            return ""
        stderr = proc.stderr
        try:
            line = self._run_io_with_timeout(
                lambda: stderr.read(4096),
                operation="stderr",
                timeout=min(self._timeout, 0.1),
            )
        except (OSError, TimeoutError):
            return ""
        return line.decode("utf-8", errors="replace").strip()

    def _run_io_with_timeout(
        self,
        operation_fn: Callable[[], bytes],
        *,
        operation: str,
        timeout: float | None = None,
    ) -> bytes:
        """Run one blocking pipe read with a portable wall-clock bound."""
        done = threading.Event()
        values: list[bytes] = []
        errors: list[BaseException] = []

        def read() -> None:
            try:
                values.append(operation_fn())
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                with self._reader_lock:
                    self._reader_threads.discard(threading.current_thread())
                done.set()

        reader = threading.Thread(target=read, daemon=True, name=f"looplet-mcp-{operation}")
        with self._reader_lock:
            self._reader_threads.add(reader)
        reader.start()
        limit = self._timeout if timeout is None else timeout
        if not done.wait(limit):
            if operation == "response" and self._proc is not None:
                self._stop_process_tree(self._proc, wait_timeout=min(5.0, max(0.1, limit)))
                done.wait(min(5.0, max(0.1, limit)))
            raise TimeoutError(f"MCP server {operation} timed out after {limit:g}s")
        if errors:
            raise errors[0]
        return values[0] if values else b""

    @staticmethod
    def _stop_process_tree(proc: subprocess.Popen, *, wait_timeout: float) -> None:
        """Terminate a shell command and any child retaining its stdio pipes."""
        if proc.poll() is not None:
            return

        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            elif os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                proc.terminate()
            proc.wait(timeout=wait_timeout)
            return
        except (OSError, subprocess.SubprocessError):
            pass

        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=wait_timeout)
        except (OSError, subprocess.SubprocessError):
            pass

    def _make_executor(self, tool_name: str, param_types: dict[str, str] | None = None) -> Any:
        """Create an execute function that calls the MCP server.

        ``param_types`` maps each declared parameter to its flattened
        JSON-Schema type string (e.g. ``"array"``, ``"object"``,
        ``"(optional) array"``). It is used to coerce structured
        arguments that arrive double-encoded as JSON strings.
        """
        param_types = param_types or {}

        def execute(**kwargs: Any) -> dict:
            arguments = self._coerce_structured_args(kwargs, param_types)
            resp = self._send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments,
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
    def _coerce_structured_args(
        kwargs: dict[str, Any], param_types: dict[str, str]
    ) -> dict[str, Any]:
        """Parse JSON-string args for params declared as array/object.

        A model driving the loop via JSON-text tool calls sometimes
        double-encodes a structured argument - e.g. it emits
        ``"edits": "[{\\"old_string\\": ...}]"`` (a JSON string) instead
        of a real list. In-process dispatch tolerates this loosely, but
        an MCP server validating against its inputSchema rejects the
        stringified value. When a parameter's declared type is
        ``array``/``object`` and its value arrived as a ``str`` that
        parses to the matching Python type, substitute the parsed value.
        Anything that doesn't cleanly match is passed through untouched.
        """
        coerced = dict(kwargs)
        for key, value in kwargs.items():
            if not isinstance(value, str):
                continue
            type_str = param_types.get(key, "")
            wants_array = "array" in type_str
            wants_object = "object" in type_str
            if not (wants_array or wants_object):
                continue
            stripped = value.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                continue
            if wants_array and isinstance(parsed, list):
                coerced[key] = parsed
            elif wants_object and isinstance(parsed, dict):
                coerced[key] = parsed
        return coerced

    @staticmethod
    def _extract_params(schema: dict) -> dict[str, str]:
        """Convert MCP JSON-schema parameters to ToolSpec parameter dict."""
        input_schema = schema.get("inputSchema", {})
        props = input_schema.get("properties", {})
        params: dict[str, str] = {}
        for name, prop in props.items():
            params[name] = prop.get("type", "str")
        return params
