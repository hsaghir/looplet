"""State Service Protocol (SSP) — the portable shared-mutable-state primitive.

``mcp_servers:`` makes *tools* portable (a tool body runs out-of-process
over MCP). ``kind: lep`` makes *hooks* portable (a hook's policy runs
out-of-process over LEP). Both are **1:1** stdio bridges: one parent, one
child. Neither can express the last in-process coupling the portability
report flags — **shared mutable state** that two *different* components
read and write (e.g. ``hello``'s greeting log, which the ``greet`` tool
appends to and the ``PolitenessGate`` hook reads to gate ``done()``).

A state service is the missing sibling: a **1:N** server that owns a piece
of mutable state in its own process and exposes named methods over a Unix
domain socket. The (now out-of-process) tool server and hook server both
*connect* to the same socket, so the state they share lives behind a
protocol rather than in one Python address space. That is exactly what
flips a ``@ref`` shared object from the ``inprocess`` tier (Python-pinned)
to the ``protocol`` tier (any conforming loader can spawn the server and
point its own tools/hooks at the socket).

Three pieces, mirroring :mod:`looplet.lep`:

* :class:`StateServiceBase` — subclass, define public methods, call
  :meth:`StateServiceBase.serve`. Holds the state; nothing else from
  looplet is required (a Rust/Go server would speak only the wire).
* :class:`StateServiceClient` — an in-process proxy. ``client.record(...)``
  forwards to the server over the socket; the loader injects one into the
  resource registry under the service's name, so existing ``@ref`` /
  ``requires:`` wiring resolves to it unchanged.
* :class:`StateServiceHandle` — spawns a server, waits for its socket,
  and owns the client + teardown.

Wire format (line-delimited JSON over ``AF_UNIX`` ``SOCK_STREAM``):

    → {"id": 1, "method": "state/initialize"}
    ← {"id": 1, "result": {"methods": ["record", "names", "entries"]}}
    → {"id": 2, "method": "state/call",
       "params": {"name": "record", "args": ["Ada"], "kwargs": {"text": "Hi"}}}
    ← {"id": 2, "result": {"value": null}}
    → {"id": 3, "method": "state/shutdown"}
    ← {"id": 3, "result": {"ok": true}}

The server reads its socket path from the ``LOOPLET_STATE_SOCKET`` env var
(set by the launcher) so the same ``command:`` works regardless of where
the loader places the socket.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "SSP_VERSION",
    "StateServiceBase",
    "StateServiceClient",
    "StateServiceError",
    "StateServiceHandle",
    "SOCKET_ENV_VAR",
]

SSP_VERSION = "0.1"

#: Env var the launcher sets so a spawned server knows where to bind and
#: so out-of-process tool/hook servers know where to connect.
SOCKET_ENV_VAR = "LOOPLET_STATE_SOCKET"

#: Per-service env var prefix: ``LOOPLET_STATE_<NAME>`` carries the socket
#: path for service ``<name>`` to clients that share several services.
PER_SERVICE_ENV_PREFIX = "LOOPLET_STATE_"


class StateServiceError(RuntimeError):
    """Raised when the state-service transport breaks or a call fails."""


def _send_line(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


class _LineReader:
    """Buffered line reader over a stream socket (one per connection)."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""

    def readline(self) -> str | None:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                if self._buf:
                    line, self._buf = self._buf, b""
                    return line.decode("utf-8")
                return None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("utf-8")


# ── server ────────────────────────────────────────────────────


class StateServiceBase:
    """Base class for a shared-mutable-state server.

    Subclass it, define public methods (any callable not starting with
    ``_`` and not one of the framework methods), and call :meth:`serve`
    from ``__main__``. Every method call is serialized under a single
    lock, so subclasses may keep ordinary (non-thread-safe) Python state
    — the lock makes concurrent client access safe by construction.
    """

    #: Method names that belong to the framework and are never exposed
    #: as callable state operations.
    _RESERVED = frozenset({"serve", "shutdown_requested"})

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # -- introspection -------------------------------------------------
    def _public_methods(self) -> list[str]:
        out = []
        for name in dir(self):
            if name.startswith("_") or name in self._RESERVED:
                continue
            if callable(getattr(self, name, None)):
                out.append(name)
        return sorted(out)

    def _dispatch(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if name.startswith("_") or name in self._RESERVED:
            raise StateServiceError(f"method {name!r} is not callable")
        fn: Callable[..., Any] | None = getattr(self, name, None)
        if not callable(fn):
            raise StateServiceError(f"unknown state method {name!r}")
        with self._lock:
            return fn(*args, **kwargs)

    # -- serving -------------------------------------------------------
    def serve(self, socket_path: str | None = None) -> int:
        """Bind a Unix domain socket and serve clients until shutdown.

        Args:
            socket_path: Where to bind. Defaults to ``$LOOPLET_STATE_SOCKET``.
        """
        if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - non-POSIX
            raise StateServiceError("state services require AF_UNIX sockets")
        path = socket_path or os.environ.get(SOCKET_ENV_VAR)
        if not path:
            raise StateServiceError(f"no socket path given and {SOCKET_ENV_VAR} is unset")
        if os.path.exists(path):
            os.unlink(path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(16)
        srv.settimeout(0.5)
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        finally:
            srv.close()
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:  # pragma: no cover - best effort
                    pass
        return 0

    def _serve_conn(self, conn: socket.socket) -> None:
        reader = _LineReader(conn)
        try:
            while not self._stop.is_set():
                line = reader.readline()
                if line is None:
                    return
                line = line.strip()
                if not line:
                    continue
                self._handle_line(conn, line)
        except (OSError, ValueError):  # pragma: no cover - client gone
            return
        finally:
            try:
                conn.close()
            except OSError:  # pragma: no cover - best effort
                pass

    def _handle_line(self, conn: socket.socket, line: str) -> None:
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(req, dict):
            return
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        try:
            if method == "state/initialize":
                result: dict[str, Any] = {
                    "ssp_version": SSP_VERSION,
                    "methods": self._public_methods(),
                }
            elif method == "state/call":
                value = self._dispatch(
                    str(params.get("name", "")),
                    list(params.get("args") or []),
                    dict(params.get("kwargs") or {}),
                )
                result = {"value": value}
            elif method == "state/shutdown":
                self._stop.set()
                result = {"ok": True}
            else:
                _send_line(
                    conn,
                    {"id": rid, "error": {"message": f"unknown method {method!r}"}},
                )
                return
        except Exception as exc:  # noqa: BLE001 — report, never crash the server
            _send_line(conn, {"id": rid, "error": {"message": str(exc)}})
            return
        _send_line(conn, {"id": rid, "result": result})


# ── client ────────────────────────────────────────────────────


class StateServiceClient:
    """In-process proxy to a :class:`StateServiceBase` over a socket.

    ``client.record("Ada", text="Hi")`` forwards to the server's
    ``record`` method. Calls are serialized with an internal lock so a
    single client instance can be shared across loop threads (e.g. under
    ``concurrent_dispatch``).
    """

    def __init__(self, socket_path: str, *, connect_timeout: float = 10.0) -> None:
        self._path = socket_path
        self._lock = threading.Lock()
        self._next_id = 0
        self._sock = self._connect(socket_path, connect_timeout)
        self._reader = _LineReader(self._sock)
        self.methods: list[str] = []
        try:
            init = self._rpc("state/initialize", {})
            self.methods = list(init.get("methods") or [])
        except StateServiceError:  # pragma: no cover - tolerated
            self.methods = []

    @staticmethod
    def _connect(path: str, timeout: float) -> socket.socket:
        if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - non-POSIX
            raise StateServiceError("state services require AF_UNIX sockets")
        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(path)
                return sock
            except OSError as exc:
                last_exc = exc
                time.sleep(0.02)
        raise StateServiceError(f"could not connect to state service at {path!r}: {last_exc}")

    @property
    def socket_path(self) -> str:
        return self._path

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            try:
                _send_line(self._sock, {"id": rid, "method": method, "params": params})
                line = self._reader.readline()
            except OSError as exc:
                raise StateServiceError(f"state transport failed: {exc}") from exc
            if line is None:
                raise StateServiceError("state service closed the connection")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise StateServiceError(f"state service emitted non-JSON: {line!r}") from exc
            if not isinstance(parsed, dict):  # pragma: no cover - defensive
                raise StateServiceError("malformed state-service response")
            if parsed.get("error"):
                raise StateServiceError(str(parsed["error"].get("message", "error")))
            return parsed.get("result") or {}

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a server method by name; return its value."""
        result = self._rpc("state/call", {"name": name, "args": list(args), "kwargs": kwargs})
        return result.get("value")

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # Only reached for attributes not found normally; expose every
        # server method as ``client.<method>(...)``.
        if name.startswith("_"):
            raise AttributeError(name)

        def _method(*args: Any, **kwargs: Any) -> Any:
            return self.call(name, *args, **kwargs)

        return _method

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:  # pragma: no cover - best effort
            pass


# ── launcher / handle ─────────────────────────────────────────


class StateServiceHandle:
    """Owns a spawned state-service process plus a connected client."""

    def __init__(
        self,
        name: str,
        proc: subprocess.Popen[bytes],
        client: StateServiceClient,
        socket_path: str,
        socket_dir: str,
    ) -> None:
        self.name = name
        self.proc = proc
        self.client = client
        self.socket_path = socket_path
        self._socket_dir = socket_dir

    @staticmethod
    def _server_env(socket_path: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        # Guarantee a Python SSP server can ``import looplet`` from a
        # source checkout (mirrors LEPHookAdapter._server_env).
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        parts = [pkg_parent] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env[SOCKET_ENV_VAR] = socket_path
        if extra:
            env.update(extra)
        return env

    @classmethod
    def spawn(
        cls,
        command: str | list[str],
        *,
        name: str,
        timeout_s: float = 10.0,
        env: dict[str, str] | None = None,
    ) -> "StateServiceHandle":
        """Spawn a state-service server and connect a client to it.

        Args:
            command: argv (list) or a shell-style string to launch the
                server. The server binds ``$LOOPLET_STATE_SOCKET``.
            name: The service name (used for the socket filename + the
                per-service env var passed to other servers).
            timeout_s: How long to wait for the socket to come up.
            env: Extra environment for the server process.
        """
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            raise StateServiceError(f"state service {name!r} has an empty command")
        socket_dir = tempfile.mkdtemp(prefix=f"looplet-state-{name}-")
        socket_path = os.path.join(socket_dir, f"{name}.sock")
        proc = subprocess.Popen(  # noqa: S603 — argv is operator-supplied
            argv,
            stdin=subprocess.DEVNULL,
            env=cls._server_env(socket_path, env),
        )
        try:
            client = StateServiceClient(socket_path, connect_timeout=timeout_s)
        except StateServiceError:
            proc.terminate()
            raise
        return cls(name, proc, client, socket_path, socket_dir)

    def close(self) -> None:
        try:
            self.client.call  # noqa: B018 — touch to ensure attr exists
            self.client._rpc("state/shutdown", {})  # noqa: SLF001
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            self.client.close()
        except Exception:  # pragma: no cover - best effort
            pass
        proc = self.proc
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover - best effort
            try:
                proc.kill()
            except Exception:
                pass
        for p in (self.socket_path, self._socket_dir):
            try:
                if os.path.isdir(p):
                    os.rmdir(p)
                elif os.path.exists(p):
                    os.unlink(p)
            except OSError:  # pragma: no cover - best effort
                pass


def state_server_argv(server_path: str, *args: str) -> list[str]:
    """Build argv to launch a Python SSP server with this interpreter."""
    return [sys.executable, server_path, *args]
