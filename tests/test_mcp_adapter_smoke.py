"""Smoke tests for MCPToolAdapter - MCP protocol adapter."""

from __future__ import annotations

import io
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from looplet import BaseToolRegistry, RunEnvelope
from looplet.mcp import MCPToolAdapter, MCPToolError

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
        assert params == {"path": "string", "encoding": "(optional) string"}

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

    def test_explicit_alias_preserves_preparation_and_server_name(self, monkeypatch):
        from dataclasses import replace

        from looplet import ToolCall, ToolSpec

        trusted = ToolSpec("echo", "Trusted local echo.", {}, lambda: {"trusted": True})
        registry = BaseToolRegistry()
        registry.register(trusted)
        adapter = MCPToolAdapter("synthetic server never started")
        adapter._started = True
        adapter._tool_schemas = [
            {
                "name": "echo",
                "inputSchema": {
                    "type": "object",
                    "properties": {"targets": {"type": "array"}},
                    "required": ["targets"],
                },
            }
        ]
        sent = []
        monkeypatch.setattr(
            adapter, "_send_request", lambda method, params: sent.append(params) or {"content": []}
        )
        registry.register(replace(adapter.tools()[0], name="server__echo"))

        result = registry.dispatch(ToolCall("server__echo", {"targets": '["allowed"]'}))

        assert result.error is None
        assert registry.tool_specs["echo"] is trusted
        assert sent == [{"name": "echo", "arguments": {"targets": ["allowed"]}}]

    @pytest.mark.parametrize("names", [("fresh", "echo"), ("echo", "fresh"), ("fresh", "fresh")])
    def test_register_all_rejects_collisions_atomically(self, names):
        from looplet import ToolSpec

        trusted = ToolSpec(
            name="echo",
            description="Trusted local echo.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            execute=lambda text: {"trusted": text},
        )
        registry = BaseToolRegistry()
        registry.register(trusted)
        adapter = MCPToolAdapter("echo test")
        adapter._started = True
        adapter._tool_schemas = [
            {"name": name, "inputSchema": {"type": "object"}} for name in names
        ]

        with pytest.raises(ValueError, match="MCP tool name collision"):
            adapter.register_all(registry)

        assert registry.tool_specs == {"echo": trusted}

    @pytest.mark.parametrize("mode", ["sync", "async"])
    @pytest.mark.parametrize("validate", [False, True])
    @pytest.mark.parametrize("encoded", [False, True])
    @pytest.mark.parametrize("restricted", [False, True])
    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("shape", ["array", "object"])
    async def test_structured_arguments_are_authorized_before_dispatch(
        self, monkeypatch, mode, validate, encoded, restricted, native, shape
    ):
        from looplet import (
            AsyncMockLLMBackend,
            LoopConfig,
            MockLLMBackend,
            PermissionDecision,
            PermissionEngine,
            PermissionHook,
            async_composable_loop,
            composable_loop,
            tools_from,
        )

        adapter = MCPToolAdapter("synthetic server never started")
        adapter._started = True
        adapter._tool_schemas = [
            {
                "name": "read_items",
                "inputSchema": {
                    "type": "object",
                    "properties": {"targets": {"type": shape}},
                    "required": ["targets"],
                },
            }
        ]
        sent = []
        seen = []
        monkeypatch.setattr(
            adapter, "_send_request", lambda method, params: sent.append(params) or {"content": []}
        )

        def permitted(arguments):
            seen.append(arguments["targets"])
            values = arguments["targets"]
            if shape == "object":
                values = values.values()
            return all(target != "restricted" for target in values)

        engine = PermissionEngine(default=PermissionDecision.DENY)
        engine.allow("read_items", arg_matcher=permitted)
        target = "restricted" if restricted else "allowed"
        targets = {"item": target} if shape == "object" else [target]
        response = json.dumps(
            {"tool": "read_items", "args": {"targets": json.dumps(targets) if encoded else targets}}
        )
        options = {
            "tools": tools_from(adapter.tools(), include_done=True, validate=validate),
            "task": {"goal": "Check synthetic argument authorization."},
            "config": LoopConfig(max_steps=1, use_native_tools=native),
            "hooks": [PermissionHook(engine)],
        }
        if mode == "async":
            steps = [
                step
                async for step in async_composable_loop(
                    llm=AsyncMockLLMBackend([response], cycle=False), **options
                )
            ]
        else:
            steps = list(composable_loop(llm=MockLLMBackend([response], cycle=False), **options))

        assert seen and all(arguments == targets for arguments in seen)
        if restricted:
            assert sent == []
            assert "Permission denied" in steps[0].tool_result.error
        else:
            assert steps[0].tool_result.error is None
            assert sent == [{"name": "read_items", "arguments": {"targets": targets}}]

    @pytest.mark.parametrize("clone", [False, True])
    def test_discovered_schema_keeps_constraints_and_structured_repair(self, monkeypatch, clone):
        from looplet import ToolCall

        input_schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["safe", "fast"]},
                "rows": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                "limit": {"type": "integer", "minimum": 1, "default": 10},
            },
            "required": ["mode", "rows"],
            "additionalProperties": False,
        }
        adapter = MCPToolAdapter("echo test")
        adapter._started = True
        adapter._tool_schemas = [{"name": "query", "inputSchema": input_schema}]
        sent = []
        monkeypatch.setattr(
            adapter,
            "_send_request",
            lambda method, params: sent.append((method, params)) or {"content": []},
        )
        registry = BaseToolRegistry()
        adapter.register_all(registry)

        if clone:
            from looplet.subagent import clone_tools_excluding

            registry = clone_tools_excluding(registry, [])

        assert registry.tool_schemas()[0]["input_schema"] == input_schema
        assert registry.introspect()["tools"][0]["parameters"] == input_schema
        assert registry.tool_specs["query"].required_parameters() == ["mode", "rows"]
        assert (
            registry.dispatch(ToolCall("query", {"mode": "safe", "rows": "[1, 2]"})).error is None
        )
        assert sent == [
            ("tools/call", {"name": "query", "arguments": {"mode": "safe", "rows": [1, 2]}})
        ]

    @pytest.mark.parametrize(
        ("type_name", "value", "expected"),
        [(["array", "null"], "[1]", [1]), (["object", "null"], '{"a": 1}', {"a": 1})],
    )
    def test_nullable_structured_schema_keeps_repair(self, monkeypatch, type_name, value, expected):
        from looplet import ToolCall

        adapter = MCPToolAdapter("echo test")
        adapter._started = True
        adapter._tool_schemas = [
            {
                "name": "echo",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": type_name}},
                    "required": ["value"],
                },
            }
        ]
        sent = []
        monkeypatch.setattr(
            adapter, "_send_request", lambda method, params: sent.append(params) or {"content": []}
        )
        spec = adapter.tools()[0]

        assert spec.to_json_schema()["properties"]["value"]["type"] == type_name
        registry = BaseToolRegistry()
        registry.register(spec)
        assert registry.dispatch(ToolCall("echo", {"value": value})).error is None
        assert sent[0]["arguments"]["value"] == expected

    def test_prepared_call_is_not_repaired_again_after_hooks_change_arguments(self, monkeypatch):
        from looplet import ToolCall

        adapter = MCPToolAdapter("synthetic server never started")
        adapter._started = True
        adapter._tool_schemas = [
            {
                "name": "read_items",
                "inputSchema": {
                    "type": "object",
                    "properties": {"targets": {"type": "array"}},
                },
            }
        ]
        sent = []
        monkeypatch.setattr(
            adapter, "_send_request", lambda method, params: sent.append(params) or {"content": []}
        )
        registry = BaseToolRegistry()
        adapter.register_all(registry)
        call = ToolCall("read_items", {"targets": '["allowed"]'})
        registry.prepare_call(call)
        assert call.args == {"targets": ["allowed"]}
        call.args = {"targets": '["restricted"]'}

        result = registry.dispatch(call)

        assert "prepared before dispatch" in result.error
        assert sent == []

    def test_legacy_optional_type_marker_is_normalized_without_mutating_server_schema(self):
        adapter = MCPToolAdapter("echo test")
        adapter._started = True
        original = {
            "type": "object",
            "properties": {"items": {"type": "(optional) array"}},
            "required": ["items"],
        }
        adapter._tool_schemas = [{"name": "echo", "inputSchema": original}]
        spec = adapter.tools()[0]

        assert spec.to_json_schema()["properties"]["items"]["type"] == "array"
        assert spec.required_parameters() == []
        assert original["properties"]["items"]["type"] == "(optional) array"
        assert original["required"] == ["items"]

    def test_protocol_guards_without_process(self):
        adapter = MCPToolAdapter("echo test")

        assert adapter._send_and_receive({}) is None
        assert adapter._read_message() is None
        adapter._write_message({})

    def test_envelope_is_added_to_wire_request_params(self, monkeypatch):
        adapter = MCPToolAdapter("echo test", run_envelope=RunEnvelope(run_id="run-1"))
        sent = []
        monkeypatch.setattr(
            adapter, "_send_and_receive", lambda message: sent.append(message) or {}
        )

        adapter._send_request("tools/list", {})

        assert sent[0]["params"]["run_envelope"] == {"run_id": "run-1"}

    def test_json_rpc_error_response_is_a_tool_failure(self, caplog):
        adapter = MCPToolAdapter("echo test")
        adapter._proc = SimpleNamespace(
            stdout=io.BytesIO(b'{"error":{"code":-1,"message":"bad"}}\n')
        )

        with pytest.raises(MCPToolError, match="MCP request failed"):
            adapter._read_message()
        assert "MCP error" in caplog.text

    def test_discovered_tools_preserve_legacy_undeclared_authority_by_default(self):
        adapter = MCPToolAdapter("echo test")
        adapter._started = True
        adapter._tool_schemas = [{"name": "echo", "inputSchema": {"type": "object"}}]

        assert adapter.tools()[0].capabilities == []

    def test_custom_capabilities_are_attached_to_discovered_tools(self):
        adapter = MCPToolAdapter("echo test", capabilities=["workspace.read"])
        adapter._started = True
        adapter._tool_schemas = [{"name": "read", "inputSchema": {"type": "object"}}]

        assert adapter.tools()[0].capabilities == ["workspace.read"]

    def test_missing_result_or_error_is_rejected(self):
        adapter = MCPToolAdapter("echo test")
        adapter._proc = SimpleNamespace(stdout=io.BytesIO(b'{"jsonrpc":"2.0","id":1}\n'))

        with pytest.raises(RuntimeError, match="result or error"):
            adapter._read_message(expected_id=1)

    def test_reader_skips_notifications_and_queues_other_responses(self):
        adapter = MCPToolAdapter("echo test")
        adapter._proc = SimpleNamespace(
            stdout=io.BytesIO(
                b'{"jsonrpc":"2.0","method":"notifications/progress"}\n'
                b'{"jsonrpc":"2.0","id":2,"result":{"value":"later"}}\n'
                b'{"jsonrpc":"2.0","id":1,"result":{"value":"now"}}\n'
            )
        )

        assert adapter._read_message(expected_id=1) == {"value": "now"}
        assert adapter._read_message(expected_id=2) == {"value": "later"}

    @pytest.mark.parametrize("payload", [b"[]\n", b"1\n"])
    def test_nonobject_response_returns_none(self, payload, caplog):
        adapter = MCPToolAdapter("echo test")
        adapter._proc = SimpleNamespace(stdout=io.BytesIO(payload))

        assert adapter._read_message() is None
        assert "must be a JSON object" in caplog.text

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_timeout_must_be_positive(self, timeout):
        with pytest.raises(ValueError, match="greater than zero"):
            MCPToolAdapter("echo test", timeout=timeout)

    @pytest.mark.parametrize("timeout", [float("inf"), float("nan"), True])
    def test_timeout_must_be_finite(self, timeout):
        with pytest.raises(ValueError, match="finite"):
            MCPToolAdapter("echo test", timeout=timeout)

    @pytest.mark.parametrize(
        "name", ["max_message_bytes", "max_pending_messages", "max_skipped_messages"]
    )
    @pytest.mark.parametrize("value", [0, -1, True, 1.5])
    def test_protocol_limits_must_be_positive_integers(self, name, value):
        with pytest.raises(ValueError, match=name):
            MCPToolAdapter("echo test", **{name: value})

    def test_reader_bounds_message_size(self):
        from looplet.mcp import MCPProtocolError

        adapter = MCPToolAdapter("echo test", max_message_bytes=64)
        adapter._proc = SimpleNamespace(stdout=io.BytesIO(b"x" * 65))
        with pytest.raises(MCPProtocolError, match="max_message_bytes"):
            adapter._read_message()

    def test_writer_rejects_large_messages_before_pipe_effects(self):
        from looplet.mcp import MCPProtocolError

        stdin = io.BytesIO()
        adapter = MCPToolAdapter("echo test", max_message_bytes=64)
        adapter._proc = SimpleNamespace(stdin=stdin)
        with pytest.raises(MCPProtocolError, match="max_message_bytes"):
            adapter._write_message({"payload": "x" * 64})
        assert stdin.getvalue() == b""

    def test_reader_bounds_pending_responses(self):
        from looplet.mcp import MCPProtocolError

        adapter = MCPToolAdapter("echo test", max_pending_messages=1)
        adapter._proc = SimpleNamespace(
            stdout=io.BytesIO(b'{"id":2,"result":{}}\n{"id":3,"result":{}}\n')
        )
        with pytest.raises(MCPProtocolError, match="max_pending_messages"):
            adapter._read_message(expected_id=1)
        assert len(adapter._pending_messages) == 1

    @pytest.mark.parametrize(
        "line", [b"invalid\n", b"[]\n", b'{"method":"notifications/progress"}\n']
    )
    def test_reader_bounds_skipped_messages(self, line):
        from looplet.mcp import MCPProtocolError

        adapter = MCPToolAdapter("echo test", max_skipped_messages=2)
        adapter._proc = SimpleNamespace(stdout=io.BytesIO(line * 3))
        with pytest.raises(MCPProtocolError, match="max_skipped_messages"):
            adapter._read_message(expected_id=1)

    def test_notifications_do_not_reset_request_deadline(self, monkeypatch):
        from looplet import mcp

        clock = [0.0]

        class ProgressStream:
            def readline(self, limit):
                clock[0] += 0.4
                return b'{"method":"notifications/progress"}\n'

        monkeypatch.setattr(mcp.time, "monotonic", lambda: clock[0])
        adapter = MCPToolAdapter("echo test", timeout=1.0)
        adapter._proc = SimpleNamespace(stdout=ProgressStream())
        with pytest.raises(TimeoutError, match="request deadline"):
            adapter._read_message(expected_id=1)
        assert clock[0] < 1.7

    def test_invalid_utf8_is_a_protocol_error(self):
        from looplet.mcp import MCPProtocolError

        adapter = MCPToolAdapter("echo test")
        adapter._proc = SimpleNamespace(stdout=io.BytesIO(b"\xff\n"))
        with pytest.raises(MCPProtocolError, match="UTF-8"):
            adapter._read_message()

    def test_close_discards_previous_session_messages(self):
        adapter = MCPToolAdapter("echo test")
        adapter._pending_messages = [{"id": 1, "result": {"old": True}}]
        adapter._tool_schemas = [{"name": "old"}]
        adapter.close()
        assert adapter._pending_messages == []
        assert adapter._tool_schemas == []

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

    def test_server_env_extends_host_environment(self, monkeypatch):
        from looplet import mcp

        options = {}
        proc = SimpleNamespace(stdin=object(), stdout=object(), stderr=object())

        def fake_popen(*_args, **kwargs):
            options.update(kwargs)
            return proc

        monkeypatch.setenv("LOOPLET_LLM_SOCKET", "/tmp/gateway.sock")
        adapter = MCPToolAdapter("echo test", env={"EXTRA": "value"})
        monkeypatch.setattr(mcp.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            adapter,
            "_send_request",
            lambda method, _params: {"tools": []} if method == "tools/list" else {"ok": True},
        )
        monkeypatch.setattr(adapter, "_send_notification", lambda *_args: None)

        adapter._ensure_started()

        assert options["env"]["LOOPLET_LLM_SOCKET"] == "/tmp/gateway.sock"
        assert options["env"]["EXTRA"] == "value"

    @pytest.mark.parametrize("env", [None, {}, {"EXTRA": "value"}])
    def test_exact_environment_does_not_inherit_host_values(self, monkeypatch, env):
        from looplet import mcp

        options = {}
        proc = SimpleNamespace(stdin=object(), stdout=object(), stderr=object())
        monkeypatch.setenv("SYNTHETIC_HOST_TOKEN", "not-a-real-secret")
        monkeypatch.setattr(
            mcp.subprocess, "Popen", lambda *args, **kwargs: options.update(kwargs) or proc
        )
        adapter = MCPToolAdapter(
            [sys.executable, "synthetic_server.py"], env=env, inherit_env=False
        )
        monkeypatch.setattr(
            adapter,
            "_send_request",
            lambda method, params: {"tools": []} if method == "tools/list" else {},
        )
        monkeypatch.setattr(adapter, "_send_notification", lambda *args: None)

        adapter._ensure_started()

        assert options["shell"] is False
        assert options["env"] == (env or {})
        assert "SYNTHETIC_HOST_TOKEN" not in options["env"]

    @pytest.mark.parametrize("command", ["", [], ["python", ""], ["python", 1]])
    def test_invalid_commands_are_rejected_before_start(self, command):
        with pytest.raises(ValueError, match="command"):
            MCPToolAdapter(command)

    def test_environment_options_are_validated_before_start(self):
        with pytest.raises(TypeError, match="inherit_env"):
            MCPToolAdapter("echo test", inherit_env="false")
        with pytest.raises(TypeError, match="string mapping"):
            MCPToolAdapter("echo test", env={"PORT": 1234})

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

    def test_executor_no_response_is_a_tool_failure(self, monkeypatch):
        adapter = MCPToolAdapter("echo test")
        monkeypatch.setattr(adapter, "_send_request", lambda *_args, **_kwargs: None)

        with pytest.raises(MCPToolError, match="returned no response"):
            adapter._make_executor("echo")()

    def test_executor_is_error_is_a_tool_failure(self, monkeypatch):
        adapter = MCPToolAdapter("echo test")
        monkeypatch.setattr(
            adapter,
            "_send_request",
            lambda *_args, **_kwargs: {
                "isError": True,
                "content": [{"type": "text", "text": "permission denied"}],
            },
        )

        with pytest.raises(MCPToolError, match="permission denied"):
            adapter._make_executor("echo")()

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
