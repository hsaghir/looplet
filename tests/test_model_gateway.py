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


def test_handle_exports_socket_env_var():
    prior = os.environ.get(LLM_SOCKET_ENV_VAR)
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
