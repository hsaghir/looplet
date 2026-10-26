"""Unit tests for the Model Gateway Protocol (MGP) primitive.

The model gateway is the host-side sibling of SSP: a 1:N ``AF_UNIX``
server that exposes the loop's LLM backend to out-of-process MCP tool /
LEP hook servers so they can call ``ctx.llm.generate(...)`` instead of
degrading. These tests exercise the in-host server, the in-process
client proxy, the handle lifecycle, and the late-binding / no-backend
semantics that let a headless dispatch fall back gracefully.
"""

from __future__ import annotations

import json
import os
import socket

import pytest

from looplet.model_gateway import (
    LLM_SOCKET_ENV_VAR,
    MGP_VERSION,
    ModelGatewayClient,
    ModelGatewayError,
    ModelGatewayHandle,
    ModelGatewayServer,
)

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="model gateway requires AF_UNIX sockets"
)


class _ScriptedLLM:
    """A backend that records prompts and returns canned replies in order."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return self._replies.pop(0) if self._replies else "DONE"


@pytest.mark.parametrize("prior", [None, "/caller/gateway.sock"])
def test_handle_exports_socket_env_var(monkeypatch, prior):
    if prior is None:
        monkeypatch.delenv(LLM_SOCKET_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(LLM_SOCKET_ENV_VAR, prior)
    handle = ModelGatewayHandle.start()
    try:
        assert os.environ[LLM_SOCKET_ENV_VAR] == handle.socket_path
        assert os.path.exists(handle.socket_path)
    finally:
        handle.close()
    # close() removes the env var it owned and the socket.
    assert os.environ.get(LLM_SOCKET_ENV_VAR) == prior
    assert not os.path.exists(handle.socket_path)


def test_generate_round_trip_with_backend():
    backend = _ScriptedLLM(["HIGH"])
    handle = ModelGatewayHandle.start(backend=backend)
    try:
        client = ModelGatewayClient(handle.socket_path)
        assert client.ready is True
        out = client.generate("classify this", max_tokens=20)
        assert out == "HIGH"
        # kwargs are forwarded verbatim to the backend.
        assert backend.prompts == ["classify this"]
        assert backend.kwargs == [{"max_tokens": 20}]
        client.close()
    finally:
        handle.close()


def test_close_wakes_idle_listener(monkeypatch):
    import threading

    original_accept = socket.socket.accept
    idle = threading.Event()
    accepts = 0

    def accept(listener):
        nonlocal accepts
        accepts += 1
        listener.settimeout(60)
        if accepts >= 3:
            idle.set()
        return original_accept(listener)

    monkeypatch.setattr(socket.socket, "accept", accept)
    handle = ModelGatewayHandle.start(backend=_ScriptedLLM(["ok"]), export_env=False)
    duplicate = handle.server._srv.dup()
    try:
        client = ModelGatewayClient(handle.socket_path)
        try:
            assert client.generate("probe") == "ok"
            assert idle.wait(2)
        finally:
            client.close()
        handle.close()
        assert not handle._thread.is_alive()
    finally:
        try:
            duplicate.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        duplicate.close()
        handle.close()


def test_stop_keeps_listener_when_serve_clears_reference(monkeypatch):
    server = ModelGatewayServer()
    closed = []

    class Listener:
        def shutdown(self, how):
            assert how == socket.SHUT_RDWR
            monkeypatch.setattr(server, "_srv", None)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(server, "_srv", Listener())
    server.stop()
    assert closed == [True]
    assert server._srv is None
    assert server._stop.is_set()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_tool_generation_preserves_blocks_and_arguments(asynchronous):
    import asyncio

    content = [{"type": "tool_use", "id": "native-1", "name": "done", "input": {"summary": "ok"}}]
    tools = [{"name": "done", "input_schema": {"type": "object"}}]
    requests = []

    class Backend(_ScriptedLLM):
        def generate_with_tools(self, prompt, **kwargs):
            requests.append((prompt, kwargs))
            return content

    class AsyncBackend(Backend):
        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    handle = ModelGatewayHandle.start(backend=AsyncBackend([]) if asynchronous else Backend([]))
    try:
        client = ModelGatewayClient(handle.socket_path)
        try:
            actual = await asyncio.to_thread(
                client.generate_with_tools, "dispatch", tools=tools, max_tokens=2000
            )
            assert actual == content
            assert requests == [("dispatch", {"tools": tools, "max_tokens": 2000})]
        finally:
            client.close()
    finally:
        handle.close()


def test_no_backend_raises_so_callers_degrade():
    # Started without a backend (the load-time default): generate must
    # surface an error so out-of-process tools fall back to ctx.llm-None.
    handle = ModelGatewayHandle.start()
    try:
        client = ModelGatewayClient(handle.socket_path)
        assert client.ready is False
        with pytest.raises(ModelGatewayError):
            client.generate("anything")
        client.close()
    finally:
        handle.close()


@pytest.mark.parametrize("max_model_calls", [None, 2])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_portable_tool_calls_share_admission_and_isolate_conversation(
    max_model_calls, asynchronous
):
    import copy
    from pathlib import Path

    from looplet import cartridge_to_preset

    class Backend(_ScriptedLLM):
        conversation_mode = "messages"

        def __init__(self):
            super().__init__(["[]"])
            self.conversation = None
            self.requests = []
            self.helper_conversations = []
            self.native_calls = 0

        def set_conversation_input(self, snapshot):
            self.conversation = copy.deepcopy(snapshot)

        def generate(self, prompt, **kwargs):
            self.requests.append("helper")
            self.helper_conversations.append(copy.deepcopy(self.conversation))
            return super().generate(prompt, **kwargs)

        def generate_with_tools(self, prompt, **kwargs):
            self.requests.append("parent")
            self.native_calls += 1
            name = "find_alternatives" if self.native_calls == 1 else "done"
            arguments = (
                {"package_name": "requests"}
                if name == "find_alternatives"
                else {"summary": "checked"}
            )
            return [
                {
                    "type": "tool_use",
                    "id": f"call-{self.native_calls}",
                    "name": name,
                    "input": arguments,
                }
            ]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    cartridge = Path(__file__).resolve().parents[1] / "examples/dep_doctor_portable.cartridge"
    backend = AsyncBackend() if asynchronous else Backend()
    with cartridge_to_preset(cartridge) as preset:
        preset.config.max_model_calls = max_model_calls
        task = {"description": "Check alternatives and finish"}
        result = (
            await preset.run_result_async(backend, task=task)
            if asynchronous
            else preset.run_result(backend, task=task)
        )
        expected_calls = 3 if max_model_calls is None else 2
        assert len(backend.requests) == expected_calls
        assert getattr(preset.state, "_looplet_run_stats")["llm_calls"] == expected_calls
        assert backend.helper_conversations == [None]
        assert result.completed is (max_model_calls is None)


def test_late_binding_backend():
    # The loader starts the gateway with no backend (before any run); the
    # backend is bound later at run time. A client that connected early
    # must see the backend once it is set.
    handle = ModelGatewayHandle.start()
    try:
        client = ModelGatewayClient(handle.socket_path)
        with pytest.raises(ModelGatewayError):
            client.generate("before")
        handle.set_backend(_ScriptedLLM(["AFTER"]))
        assert client.generate("after") == "AFTER"
        client.close()
    finally:
        handle.close()


async def test_async_backend_delivery_uses_its_bound_host_loop():
    import asyncio

    owner = asyncio.get_running_loop()

    class Backend:
        async def generate(self, prompt, **kwargs):
            assert asyncio.get_running_loop() is owner
            assert prompt == "async request" and kwargs == {"max_tokens": 20}
            return "async response"

    handle = ModelGatewayHandle.start()
    try:
        handle.set_backend(Backend())
        client = ModelGatewayClient(handle.socket_path)
        try:
            assert await asyncio.to_thread(client.generate, "async request", max_tokens=20) == (
                "async response"
            )
        finally:
            client.close()
    finally:
        handle.close()


async def test_async_gateway_deadline_cancels_backend_delivery(monkeypatch):
    import asyncio

    from looplet.model_gateway import ModelGatewayServer

    closed = asyncio.Event()
    never = asyncio.Event()

    class Backend:
        async def generate(self, prompt, **kwargs):
            try:
                await never.wait()
            finally:
                closed.set()

    server = ModelGatewayServer(Backend())
    monkeypatch.setattr("looplet.model_gateway.remaining_deadline", lambda envelope: 0.05)
    with pytest.raises(ModelGatewayError, match="deadline"):
        await asyncio.to_thread(server._generate, "request", {})
    await asyncio.wait_for(closed.wait(), timeout=2)


def test_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv(LLM_SOCKET_ENV_VAR, raising=False)
    assert ModelGatewayClient.from_env() is None


def test_from_env_connects_when_set():
    handle = ModelGatewayHandle.start(backend=_ScriptedLLM(["X"]))
    try:
        client = ModelGatewayClient.from_env()
        assert client is not None
        assert client.generate("p") == "X"
        client.close()
    finally:
        handle.close()


def test_multiple_concurrent_clients_share_backend():
    backend = _ScriptedLLM(["A", "B", "C"])
    handle = ModelGatewayHandle.start(backend=backend)
    try:
        c1 = ModelGatewayClient(handle.socket_path)
        c2 = ModelGatewayClient(handle.socket_path)
        assert c1.generate("p1") == "A"
        assert c2.generate("p2") == "B"
        assert c1.generate("p3") == "C"
        c1.close()
        c2.close()
    finally:
        handle.close()


def test_initialize_reports_version():
    handle = ModelGatewayHandle.start(backend=_ScriptedLLM([]))
    try:
        # Talk the raw wire protocol to confirm the documented shape.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(handle.socket_path)
        sock.sendall((json.dumps({"id": 1, "method": "llm/initialize"}) + "\n").encode())
        data = b""
        while b"\n" not in data:
            data += sock.recv(4096)
        reply = json.loads(data.split(b"\n", 1)[0].decode())
        assert reply["id"] == 1
        assert reply["result"]["mgp_version"] == MGP_VERSION
        assert reply["result"]["ready"] is True
        sock.close()
    finally:
        handle.close()


def test_unknown_method_returns_error():
    handle = ModelGatewayHandle.start()
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(handle.socket_path)
        sock.sendall((json.dumps({"id": 9, "method": "llm/bogus"}) + "\n").encode())
        data = b""
        while b"\n" not in data:
            data += sock.recv(4096)
        reply = json.loads(data.split(b"\n", 1)[0].decode())
        assert reply["id"] == 9
        assert "error" in reply
        sock.close()
    finally:
        handle.close()


def test_stop_closes_connected_clients():
    handle = ModelGatewayHandle.start(backend=_ScriptedLLM(["X"]))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    sock.connect(handle.socket_path)
    try:
        handle.close()
        try:
            assert sock.recv(1) == b""
        except ConnectionResetError:
            pass
    finally:
        sock.close()


@pytest.mark.parametrize("payload", [b"not-json\n", b"[]\n"])
def test_malformed_frame_gets_protocol_error(payload):
    handle = ModelGatewayHandle.start()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    sock.connect(handle.socket_path)
    try:
        sock.sendall(payload)
        data = sock.recv(4096)
        reply = json.loads(data.split(b"\n", 1)[0].decode())
        assert reply["id"] is None
        assert "error" in reply
    finally:
        sock.close()
        handle.close()


def test_client_rejects_mismatched_response_id():
    import threading
    from types import SimpleNamespace

    client = ModelGatewayClient.__new__(ModelGatewayClient)
    client._lock = threading.Lock()
    client._next_id = 0
    client._run_envelope = None
    client._sock = SimpleNamespace(sendall=lambda _payload: None)
    client._reader = SimpleNamespace(readline=lambda: '{"id":99,"result":{}}')

    with pytest.raises(ModelGatewayError, match="id mismatch"):
        client._rpc("llm/initialize", {})


def test_client_deadline_bounds_silent_response_and_closes_connection():
    import threading
    import time

    from looplet.model_gateway import _LineReader
    from looplet.types import RunEnvelope

    local, peer = socket.socketpair()
    client = ModelGatewayClient.__new__(ModelGatewayClient)
    client._lock = threading.Lock()
    client._next_id = 0
    client._run_envelope = RunEnvelope(run_id="deadline", deadline_at=time.time() + 0.05)
    client._sock = local
    client._reader = _LineReader(local)
    started = time.monotonic()
    try:
        with pytest.raises(ModelGatewayError, match="deadline"):
            client._rpc("llm/generate", {})
        assert time.monotonic() - started < 1.0
        assert local.fileno() == -1
    finally:
        client.close()
        peer.close()


def test_client_restores_socket_timeout_after_successful_deadline_call():
    import threading
    import time

    from looplet.model_gateway import _LineReader
    from looplet.types import RunEnvelope

    local, peer = socket.socketpair()
    local.settimeout(2.0)
    peer.sendall(b'{"id":1,"result":{"text":"ok"}}\n')
    client = ModelGatewayClient.__new__(ModelGatewayClient)
    client._lock = threading.Lock()
    client._next_id = 0
    client._run_envelope = RunEnvelope(run_id="deadline", deadline_at=time.time() + 1.0)
    client._sock = local
    client._reader = _LineReader(local)
    try:
        assert client._rpc("llm/generate", {}) == {"text": "ok"}
        assert local.gettimeout() == 2.0
    finally:
        client.close()
        peer.close()


@pytest.mark.parametrize("has_deadline", [False, True])
def test_client_socket_timeout_is_not_reported_as_run_deadline(has_deadline):
    import threading
    import time

    from looplet.model_gateway import _LineReader
    from looplet.types import RunEnvelope

    local, peer = socket.socketpair()
    local.settimeout(0.02)
    client = ModelGatewayClient.__new__(ModelGatewayClient)
    client._lock = threading.Lock()
    client._next_id = 0
    client._run_envelope = (
        RunEnvelope(run_id="deadline", deadline_at=time.time() + 10.0) if has_deadline else None
    )
    client._sock = local
    client._reader = _LineReader(local)
    try:
        with pytest.raises(ModelGatewayError, match="gateway transport failed") as failure:
            client._rpc("llm/generate", {})
        assert "deadline" not in str(failure.value)
        assert local.fileno() == -1
    finally:
        client.close()
        peer.close()


def test_client_deadline_bounds_trickled_response(monkeypatch):
    import threading

    from looplet.model_gateway import _LineReader

    class TrickleSocket:
        def __init__(self):
            self.timeout = None
            self.reads = 0
            self.closed = False

        def gettimeout(self):
            return self.timeout

        def settimeout(self, timeout):
            self.timeout = timeout

        def sendall(self, payload):
            pass

        def recv(self, size):
            self.reads += 1
            return b"{"

        def close(self):
            self.closed = True

    clock = iter([0.0, 0.0, 0.04, 0.06, 0.06])
    monkeypatch.setattr("looplet.model_gateway.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("looplet.model_gateway.remaining_deadline", lambda _envelope: 0.05)
    transport = TrickleSocket()
    client = ModelGatewayClient.__new__(ModelGatewayClient)
    client._lock = threading.Lock()
    client._next_id = 0
    client._run_envelope = None
    client._sock = transport
    client._reader = _LineReader(transport)

    with pytest.raises(ModelGatewayError, match="deadline"):
        client._rpc("llm/generate", {})

    assert transport.reads == 2
    assert transport.timeout == pytest.approx(0.01)
    assert transport.closed


def test_start_fails_when_server_never_binds(monkeypatch, tmp_path):
    socket_dir = tmp_path / "gateway"
    socket_dir.mkdir()

    class DeadThread:
        def start(self):
            return None

        def join(self, timeout):
            assert timeout == 2.0

    times = iter([0.0, 6.0])
    monkeypatch.setattr("looplet.model_gateway.tempfile.mkdtemp", lambda **_kwargs: str(socket_dir))
    monkeypatch.setattr("looplet.model_gateway.threading.Thread", lambda **_kwargs: DeadThread())
    monkeypatch.setattr("looplet.model_gateway.time.monotonic", lambda: next(times))

    with pytest.raises(ModelGatewayError, match="failed to create"):
        ModelGatewayHandle.start()
    assert not socket_dir.exists()


def test_server_refuses_live_socket(tmp_path):
    path = str(tmp_path / "live.sock")
    owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    owner.bind(path)
    owner.listen(1)
    try:
        from looplet.model_gateway import ModelGatewayServer

        with pytest.raises(ModelGatewayError, match="already in use"):
            ModelGatewayServer().serve(path)
    finally:
        owner.close()
        os.unlink(path)
