"""Smoke tests for MCPToolAdapter - MCP protocol adapter."""

from __future__ import annotations

import io
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from looplet import BaseToolRegistry
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

    def test_register_all_registers_discovered_tools(self):
        adapter = MCPToolAdapter("echo test")
        adapter._started = True
        adapter._tool_schemas = [
            {
                "name": "echo",
                "description": "Echo text.",
                "inputSchema": {"properties": {"text": {"type": "string"}}},
            }
        ]
        registry = BaseToolRegistry()

        assert adapter.register_all(registry) == 1
        assert registry.tool_names == ["echo"]

    def test_protocol_guards_without_process(self):
        adapter = MCPToolAdapter("echo test")

        assert adapter._send_and_receive({}) is None
        assert adapter._read_message() is None
        adapter._write_message({})

    def test_json_rpc_error_response_returns_none(self, caplog):
        adapter = MCPToolAdapter("echo test")
        adapter._proc = SimpleNamespace(
            stdout=io.BytesIO(b'{"error":{"code":-1,"message":"bad"}}\n')
        )

        assert adapter._read_message() is None
        assert "MCP error" in caplog.text

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_timeout_must_be_positive(self, timeout):
        with pytest.raises(ValueError, match="greater than zero"):
            MCPToolAdapter("echo test", timeout=timeout)

    def test_reader_rethrows_pipe_error(self):
        adapter = MCPToolAdapter("echo test")

        def fail():
            raise OSError("closed")

        with pytest.raises(OSError, match="closed"):
            adapter._run_io_with_timeout(fail, operation="test")

    def test_stderr_tail_reads_only_exited_process(self):
        adapter = MCPToolAdapter("echo test")
        proc = SimpleNamespace(stderr=io.BytesIO(b"startup failed\n"), poll=lambda: 2)

        assert adapter._stderr_tail_if_exited(proc) == "startup failed"

    def test_stderr_tail_ignores_read_error(self):
        adapter = MCPToolAdapter("echo test")

        class BrokenStream:
            def read(self, _size):
                raise OSError("closed")

        proc = SimpleNamespace(stderr=BrokenStream(), poll=lambda: 2)
        assert adapter._stderr_tail_if_exited(proc) == ""

    def test_close_ignores_broken_stream(self, monkeypatch):
        adapter = MCPToolAdapter("echo test")

        class BrokenStream:
            def close(self):
                raise BrokenPipeError

        proc = SimpleNamespace(
            stdin=BrokenStream(),
            stdout=BrokenStream(),
            stderr=BrokenStream(),
            poll=lambda: 0,
        )
        adapter._proc = proc
        monkeypatch.setattr(adapter, "_stop_process_tree", lambda *_args, **_kwargs: None)

        adapter.close()
        assert adapter._proc is None

    def test_close_joins_active_reader(self, monkeypatch):
        adapter = MCPToolAdapter("echo test")
        calls = []

        class Reader:
            def join(self, *, timeout):
                calls.append(timeout)

            def is_alive(self):
                return False

        proc = SimpleNamespace(
            stdin=io.BytesIO(),
            stdout=io.BytesIO(),
            stderr=io.BytesIO(),
            poll=lambda: 0,
        )
        adapter._proc = proc
        adapter._reader_threads.add(Reader())
        monkeypatch.setattr(adapter, "_stop_process_tree", lambda *_args, **_kwargs: None)

        adapter.close()
        assert calls == [5.0]

    def test_stop_process_tree_returns_for_exited_process(self, monkeypatch):
        from looplet import mcp

        monkeypatch.setattr(
            mcp.os,
            "killpg",
            lambda *_args: pytest.fail("killpg should not run for an exited process"),
        )
        proc = SimpleNamespace(poll=lambda: 0)
        MCPToolAdapter._stop_process_tree(proc, wait_timeout=0.1)

    def test_stop_process_tree_escalates_from_term_to_kill(self, monkeypatch):
        from looplet import mcp

        signals = []

        def killpg(_pid, sent_signal):
            signals.append(sent_signal)
            if sent_signal == signal.SIGTERM:
                raise OSError("term failed")

        proc = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: None)
        monkeypatch.setattr(mcp.os, "name", "posix")
        monkeypatch.setattr(mcp.os, "killpg", killpg)

        MCPToolAdapter._stop_process_tree(proc, wait_timeout=0.1)
        assert signals == [signal.SIGTERM, signal.SIGKILL]

    def test_stop_process_tree_uses_taskkill_on_windows(self, monkeypatch):
        from looplet import mcp

        calls = []
        proc = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: None)
        monkeypatch.setattr(mcp.os, "name", "nt")
        monkeypatch.setattr(mcp.subprocess, "run", lambda args, **kwargs: calls.append(args))

        MCPToolAdapter._stop_process_tree(proc, wait_timeout=0.1)
        assert calls == [["taskkill", "/PID", "123", "/T", "/F"]]

    def test_windows_start_sets_new_process_group(self, monkeypatch):
        from looplet import mcp

        options = {}
        proc = SimpleNamespace(stdin=object(), stdout=object(), stderr=object())

        def fake_popen(*_args, **kwargs):
            options.update(kwargs)
            return proc

        adapter = MCPToolAdapter("echo test")
        monkeypatch.setattr(mcp.os, "name", "nt")
        monkeypatch.setattr(mcp.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
        monkeypatch.setattr(mcp.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            adapter,
            "_send_request",
            lambda method, _params: {"tools": []} if method == "tools/list" else {"ok": True},
        )
        monkeypatch.setattr(adapter, "_send_notification", lambda *_args: None)

        adapter._ensure_started()
        assert options["creationflags"] == 512
        assert adapter._started is True

    def test_stop_process_tree_falls_back_to_process_methods(self, monkeypatch):
        from looplet import mcp

        calls = []
        proc = SimpleNamespace(
            poll=lambda: None,
            terminate=lambda: calls.append("terminate"),
            kill=lambda: calls.append("kill"),
            wait=lambda timeout: None,
        )
        monkeypatch.setattr(mcp.os, "name", "other")

        MCPToolAdapter._stop_process_tree(proc, wait_timeout=0.1)
        assert calls == ["terminate"]

    def test_stop_process_tree_escalates_process_methods(self, monkeypatch):
        from looplet import mcp

        calls = []

        def terminate():
            calls.append("terminate")
            raise subprocess.SubprocessError

        proc = SimpleNamespace(
            poll=lambda: None,
            terminate=terminate,
            kill=lambda: calls.append("kill"),
            wait=lambda timeout: None,
        )
        monkeypatch.setattr(mcp.os, "name", "other")

        MCPToolAdapter._stop_process_tree(proc, wait_timeout=0.1)
        assert calls == ["terminate", "kill"]

    def test_stop_process_tree_ignores_failed_kill(self, monkeypatch):
        from looplet import mcp

        proc = SimpleNamespace(
            poll=lambda: None,
            terminate=lambda: (_ for _ in ()).throw(subprocess.SubprocessError()),
            kill=lambda: (_ for _ in ()).throw(OSError()),
            wait=lambda timeout: None,
        )
        monkeypatch.setattr(mcp.os, "name", "other")

        MCPToolAdapter._stop_process_tree(proc, wait_timeout=0.1)

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            (None, {"error": "MCP tool 'echo' returned no response"}),
            ({"content": [{"type": "text", "text": "hello"}]}, {"text": "hello"}),
            (
                {"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
                {"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
            ),
        ],
    )
    def test_executor_response_shapes(self, monkeypatch, response, expected):
        adapter = MCPToolAdapter("echo test")
        monkeypatch.setattr(adapter, "_send_request", lambda *_args, **_kwargs: response)

        assert adapter._make_executor("echo")() == expected

    @pytest.mark.parametrize("value", ["", "not-json"])
    def test_coerce_structured_args_ignores_empty_or_invalid_json(self, value):
        assert MCPToolAdapter._coerce_structured_args({"items": value}, {"items": "array"}) == {
            "items": value
        }

    def test_from_looplet_import(self):
        from looplet import MCPToolAdapter as MCP

        assert MCP is MCPToolAdapter

    def test_coerce_structured_args_parses_json_string_array(self):
        """A double-encoded JSON-string array arg is parsed when the
        param's declared type is ``array`` (the multi_edit ``edits`` gap)."""
        param_types = {"edits": "array", "file_path": "string"}
        kwargs = {
            "file_path": "a.py",
            "edits": '[{"old_string": "x", "new_string": "y"}]',
        }
        out = MCPToolAdapter._coerce_structured_args(kwargs, param_types)
        assert out["edits"] == [{"old_string": "x", "new_string": "y"}]
        assert out["file_path"] == "a.py"  # plain string untouched

    def test_coerce_structured_args_object_and_optional(self):
        param_types = {"opts": "(optional) object", "tags": "(optional) array"}
        kwargs = {"opts": '{"a": 1}', "tags": '["x", "y"]'}
        out = MCPToolAdapter._coerce_structured_args(kwargs, param_types)
        assert out["opts"] == {"a": 1}
        assert out["tags"] == ["x", "y"]

    def test_coerce_structured_args_passes_through_non_matching(self):
        param_types = {"edits": "array", "count": "integer", "name": "string"}
        kwargs = {
            "edits": [{"old_string": "x", "new_string": "y"}],  # already a list
            "count": 3,  # not a str
            "name": "not json",  # string param, leave alone
            "missing": "[1,2]",  # unknown param, no declared type
        }
        out = MCPToolAdapter._coerce_structured_args(kwargs, param_types)
        assert out["edits"] == [{"old_string": "x", "new_string": "y"}]
        assert out["count"] == 3
        assert out["name"] == "not json"
        assert out["missing"] == "[1,2]"

    def test_coerce_structured_args_type_mismatch_passthrough(self):
        """A JSON string that parses to the wrong shape is left as-is."""
        param_types = {"edits": "array"}
        kwargs = {"edits": '{"not": "a list"}'}
        out = MCPToolAdapter._coerce_structured_args(kwargs, param_types)
        assert out["edits"] == '{"not": "a list"}'


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
        # MCP content envelope - assert the answer survives in any case.
        assert "12" in str(result)
