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

import textwrap
from pathlib import Path

import pytest

import looplet
from looplet.state_service import (
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


def test_close_terminates_server_and_cleans_socket(tmp_path: Path) -> None:
    server = _write_server(tmp_path, _COUNTER_BODY)
    handle = StateServiceHandle.spawn(state_server_argv(str(server)), name="counter")
    socket_path = handle.socket_path
    assert Path(socket_path).exists()
    handle.close()
    # Socket file is removed on close.
    assert not Path(socket_path).exists()
