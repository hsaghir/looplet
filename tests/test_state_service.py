"""Dogfood tests for the State Service Protocol (``looplet.state_service``).

The State Service is the out-of-process replacement for the in-process
``@ref`` shared-resource pattern: a small server owns a piece of shared
mutable state behind a Unix-domain socket, and any number of clients -
each potentially in a *separate* process - read and mutate the SAME
state through a :class:`StateServiceClient` proxy.

These tests witness the two properties the primitive must guarantee:

* **round-trip** - a client can call the server's public methods and get
  results back over the wire (method discovery, mutation, query).
* **1:N sharing** - two independent clients connected to the same socket
  observe each other's writes (the cross-process shared-state contract
  that makes the ``@ref`` pattern portable).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

import looplet
from looplet.state_service import (
    StateServiceBase,
    StateServiceClient,
    StateServiceError,
    StateServiceHandle,
    state_server_argv,
)

_SRC = str(Path(looplet.__file__).resolve().parent.parent)


def _write_server(tmp_path: Path, body: str, name: str = "server.py") -> Path:
    """Write a StateServiceBase server that can import looplet.

    ``body`` is the source of the class body (methods), dedented and
    re-indented under the class.
    """
    class_body = textwrap.indent(textwrap.dedent(body).strip("\n"), "    ")
    src = (
        "import sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "from looplet.state_service import StateServiceBase\n"
        "\n"
        "class Service(StateServiceBase):\n"
        f"{class_body}\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(Service().serve())\n"
    )
    path = tmp_path / name
    path.write_text(src, encoding="utf-8")
    return path


_COUNTER_BODY = """
    def __init__(self):
        super().__init__()
        self._n = 0
        self._log = []

    def incr(self, by=1):
        self._n += int(by)
        return self._n

    def value(self):
        return self._n

    def append(self, item):
        self._log.append(item)

    def items(self):
        return list(self._log)
"""


def test_round_trip_method_discovery_and_calls(tmp_path: Path) -> None:
    server = _write_server(tmp_path, _COUNTER_BODY)
    handle = StateServiceHandle.spawn(state_server_argv(str(server)), name="counter")
    try:
        client = handle.client
        # Public methods are discovered at initialize time.
        assert set(client.methods) == {"incr", "value", "append", "items"}
        assert client.value() == 0
        assert client.incr() == 1
        assert client.incr(by=4) == 5
        assert client.value() == 5
        client.append("a")
        client.append("b")
        assert client.items() == ["a", "b"]
    finally:
        handle.close()


def test_two_clients_share_state_one_to_n(tmp_path: Path) -> None:
    """A second independent client sees the first client's writes."""
    server = _write_server(tmp_path, _COUNTER_BODY)
    handle = StateServiceHandle.spawn(state_server_argv(str(server)), name="counter")
    try:
        c1 = handle.client
        c1.incr(by=3)
        c1.append("hello")

        # A *separate* client process-handle, same socket → same state.
        c2 = StateServiceClient(handle.socket_path)
        try:
            assert c2.value() == 3
            assert c2.items() == ["hello"]
            # And a write through c2 is visible to c1.
            c2.incr(by=10)
            assert c1.value() == 13
        finally:
            c2.close()
    finally:
        handle.close()


def test_unknown_method_raises_state_service_error(tmp_path: Path) -> None:
    server = _write_server(tmp_path, _COUNTER_BODY)
    handle = StateServiceHandle.spawn(state_server_argv(str(server)), name="counter")
    try:
        with pytest.raises(StateServiceError):
            handle.client.call("does_not_exist")
    finally:
        handle.close()


def test_reserved_methods_are_not_exposed(tmp_path: Path) -> None:
    server = _write_server(tmp_path, _COUNTER_BODY)
    handle = StateServiceHandle.spawn(state_server_argv(str(server)), name="counter")
    try:
        # ``serve`` is a framework method and must never be callable state.
        assert "serve" not in handle.client.methods
        with pytest.raises(StateServiceError):
            handle.client.call("serve")
    finally:
        handle.close()


@pytest.mark.parametrize("previous", [None, "/caller/service.sock"])
def test_close_terminates_server_and_cleans_socket_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous: str | None
) -> None:
    env_name = "LOOPLET_STATE_COUNTER"
    if previous is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, previous)
    server = _write_server(tmp_path, _COUNTER_BODY)
    handle = StateServiceHandle.spawn(state_server_argv(str(server)), name="counter")
    socket_path = handle.socket_path
    assert Path(socket_path).exists()
    handle.export_env()
    assert os.environ[env_name] == socket_path
    handle.close()
    # Socket file is removed on close.
    assert not Path(socket_path).exists()
    if previous is None:
        assert env_name not in os.environ
    else:
        assert os.environ[env_name] == previous


def test_close_reaps_stubborn_server_process(tmp_path: Path) -> None:
    class StubbornProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.reaped = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired(["stubborn"], timeout)
            self.reaped = True

        def kill(self):
            self.killed = True

    class Client:
        def _rpc(self, method, params):
            return {}

        def close(self):
            pass

    process = StubbornProcess()
    handle = StateServiceHandle(
        "stubborn",
        process,
        Client(),
        str(tmp_path / "service.sock"),
        str(tmp_path / "socket-dir"),
    )

    handle.close()

    assert process.terminated
    assert process.killed
    assert process.reaped


def test_spawn_failure_cleans_socket_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_dir = tmp_path / "state-socket"

    def fake_mkdtemp(*, prefix: str) -> str:
        assert prefix.startswith("looplet-state-missing-")
        socket_dir.mkdir()
        return str(socket_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)

    with pytest.raises(FileNotFoundError):
        StateServiceHandle.spawn(["/definitely/missing/state-server"], name="missing")
    assert not socket_dir.exists()


def test_failed_connection_attempt_closes_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class FakeSocket:
        def connect(self, path: str) -> None:
            raise OSError(f"missing {path}")

        def close(self) -> None:
            closed.append(True)

    times = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr("looplet.state_service.time.monotonic", lambda: next(times))
    monkeypatch.setattr("looplet.state_service.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("looplet.state_service.socket.socket", lambda *_args: FakeSocket())

    with pytest.raises(StateServiceError, match="could not connect"):
        StateServiceClient._connect("/missing.sock", 1.0)
    assert closed == [True]


def test_spawn_rejects_nonpositive_timeout() -> None:
    with pytest.raises(StateServiceError, match="greater than zero"):
        StateServiceHandle.spawn(["unused"], name="counter", timeout_s=0)


def test_server_refuses_live_socket(tmp_path: Path) -> None:
    path = str(tmp_path / "live.sock")
    owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    owner.bind(path)
    owner.listen(1)
    try:
        with pytest.raises(StateServiceError, match="already in use"):
            StateServiceBase().serve(path)
    finally:
        owner.close()
        Path(path).unlink(missing_ok=True)


def test_state_server_returns_error_for_malformed_request() -> None:
    left, right = socket.socketpair()
    try:
        service = StateServiceBase()
        service._handle_line(left, "not-json")
        reply = json.loads(right.recv(4096).split(b"\n", 1)[0])
        assert reply["id"] is None
        assert "error" in reply
    finally:
        left.close()
        right.close()


def test_state_server_reports_non_json_result() -> None:
    class Service(StateServiceBase):
        def bad(self):
            return object()

    left, right = socket.socketpair()
    try:
        Service()._handle_line(left, '{"id":1,"method":"state/call","params":{"name":"bad"}}')
        reply = json.loads(right.recv(4096).split(b"\n", 1)[0])
        assert reply["id"] == 1
        assert "not JSON-serializable" in reply["error"]["message"]
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("close_first", [True, False])
def test_export_env_stacks_same_name_ownership(tmp_path: Path, close_first: bool) -> None:
    class NoopProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    class NoopClient:
        def _rpc(self, method, params):
            return {}

        def close(self):
            pass

    first = StateServiceHandle.__new__(StateServiceHandle)
    first.name = "counter"
    first.proc = NoopProc()
    first.client = NoopClient()
    first.socket_path = str(tmp_path / "one.sock")
    first._socket_dir = str(tmp_path / "one-dir")
    first._exported_env = None
    second = StateServiceHandle.__new__(StateServiceHandle)
    second.name = "counter"
    second.proc = NoopProc()
    second.client = NoopClient()
    second.socket_path = str(tmp_path / "two.sock")
    second._socket_dir = str(tmp_path / "two-dir")
    second._exported_env = None
    os.environ["LOOPLET_STATE_COUNTER"] = "/caller/service.sock"
    try:
        first.export_env()
        second.export_env()
        assert os.environ["LOOPLET_STATE_COUNTER"] == second.socket_path
        if close_first:
            first.close()
            assert os.environ["LOOPLET_STATE_COUNTER"] == second.socket_path
            second.close()
        else:
            second.close()
            assert os.environ["LOOPLET_STATE_COUNTER"] == first.socket_path
            first.close()
        assert os.environ["LOOPLET_STATE_COUNTER"] == "/caller/service.sock"
    finally:
        os.environ.pop("LOOPLET_STATE_COUNTER", None)
