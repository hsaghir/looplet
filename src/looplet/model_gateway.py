"""Model Gateway Protocol (MGP) - the portable host-LLM access primitive.

``mcp_servers:`` makes *tools* portable (a tool body runs out-of-process
over MCP). ``kind: lep`` makes *hooks* portable (a hook's policy runs
out-of-process over LEP). ``state_services:`` (SSP) makes *shared mutable
state* portable (a 1:N socket server both tools and hooks connect to).

All three move a component out of the host's Python address space - and
in doing so they sever the one ambient capability an in-process tool took
for granted: **``ctx.llm``, the host's LLM backend.** An in-process tool
that wants to summarise/classify/extract just calls
``ctx.llm.generate(...)``; an out-of-process MCP tool has no such handle,
so it silently degrades (``dep_doctor``'s ``find_alternatives`` returns an
empty list, ``threat_intel``'s ``extract_iocs`` drops its severity field,
etc.). That degradation is exactly the gap this primitive closes.

A model gateway is the missing sibling: a **1:N server that owns the live
LLM backend in the *host* process** and exposes ``generate`` over a Unix
domain socket. Out-of-process tool servers (MCP) and hook servers (LEP)
both *connect* to the same socket, so the LLM they share lives behind a
protocol rather than in one Python address space. A portable tool can
therefore call back into the *same* model the loop is driving - restoring
functional and qualitative parity with the in-process original, with no
Python pinned on the tool side (a Rust/Go/TS MCP server would speak only
the wire).

Unlike SSP/LEP/MCP servers - which the loader *spawns* as subprocesses -
the gateway server runs **in the host process** (a daemon thread): the
LLM backend is a host object that cannot be serialised across a fork. The
loader allocates the socket path and exports it as ``LOOPLET_LLM_SOCKET``
*before* spawning any out-of-process server, so every child inherits it
and can connect lazily on its first ``generate`` call. The backend is
bound at *run* time (``AgentPreset.run(llm)``), so until a run is active -
or in a headless/test dispatch - the gateway reports "no backend" and the
tool falls back to its documented ``ctx.llm is None`` branch.

Three pieces, mirroring :mod:`looplet.state_service`:

* :class:`ModelGatewayServer` - owns the backend, serves clients. Holds a
  settable ``backend`` slot; nothing else from looplet is required to
  re-implement it in another language (the wire is the contract).
* :class:`ModelGatewayClient` - an in-process proxy. ``client.generate(
  prompt, max_tokens=...)`` forwards over the socket. Portable MCP/LEP
  servers that want to stay stdlib-only can inline an equivalent ~30-line
  client instead of importing looplet.
* :class:`ModelGatewayHandle` - allocates the socket, starts the host
  daemon thread, exports ``LOOPLET_LLM_SOCKET``, and owns teardown.

Wire format (line-delimited JSON over ``AF_UNIX`` ``SOCK_STREAM``):

    → {"id": 1, "method": "llm/initialize"}
    ← {"id": 1, "result": {"mgp_version": "0.1", "ready": true}}
    → {"id": 2, "method": "llm/generate",
       "params": {"prompt": "Classify ...", "kwargs": {"max_tokens": 20}}}
    ← {"id": 2, "result": {"text": "HIGH"}}
       (or {"id": 2, "error": {"message": "no LLM backend is bound"}})
    → {"id": 3, "method": "llm/shutdown"}
    ← {"id": 3, "result": {"ok": true}}

The gateway bridges ``generate`` only - the documented ``ctx.llm`` surface
for *tool-internal* single calls (summarise, classify, extract). The
loop's own multi-turn ``generate_with_tools`` orchestration is not a tool
concern and is intentionally out of scope.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MGP_VERSION",
    "LLM_SOCKET_ENV_VAR",
    "ModelGatewayError",
    "ModelGatewayServer",
    "ModelGatewayClient",
    "ModelGatewayHandle",
]

MGP_VERSION = "0.1"

#: Env var the loader sets so out-of-process MCP tool / LEP hook servers
#: know where to connect to reach the host's LLM backend.
LLM_SOCKET_ENV_VAR = "LOOPLET_LLM_SOCKET"


class ModelGatewayError(RuntimeError):
    """Raised when the model-gateway transport breaks or a call fails."""


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


class ModelGatewayServer:
    """Host-resident server exposing a (settable) LLM backend over a socket.

    Unlike :class:`looplet.state_service.StateServiceBase`, this server is
    not spawned as a subprocess - it runs in the *host* process (via
    :class:`ModelGatewayHandle`), because the LLM backend is a live host
    object. The :attr:`backend` slot is bound at run time; until then
    ``llm/generate`` returns an error and out-of-process callers degrade
    to their ``ctx.llm is None`` branch.

    Every call is serialised under a single lock so concurrent clients
    (multiple out-of-process tools) cannot interleave on a backend that
    may not be re-entrant.
    """

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._srv: socket.socket | None = None

    @property
    def backend(self) -> Any:
        return self._backend

    @backend.setter
    def backend(self, value: Any) -> None:
        self._backend = value

    def set_backend(self, backend: Any) -> None:
        """Bind (or replace) the live LLM backend the gateway exposes."""
        self._backend = backend

    # -- serving -------------------------------------------------------
    def serve(self, socket_path: str) -> int:
        """Bind a Unix domain socket and serve clients until shutdown."""
        if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - non-POSIX
            raise ModelGatewayError("model gateway requires AF_UNIX sockets")
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(socket_path)
        srv.listen(16)
        srv.settimeout(0.5)
        self._srv = srv
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:  # pragma: no cover - socket closed on stop
                    break
                threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()
        finally:
            srv.close()
            self._srv = None
            if os.path.exists(socket_path):
                try:
                    os.unlink(socket_path)
                except OSError:  # pragma: no cover - best effort
                    pass
        return 0

    def stop(self) -> None:
        self._stop.set()

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
            if method == "llm/initialize":
                result: dict[str, Any] = {
                    "mgp_version": MGP_VERSION,
                    "ready": self._backend is not None,
                }
            elif method == "llm/generate":
                text = self._generate(
                    str(params.get("prompt", "")),
                    dict(params.get("kwargs") or {}),
                )
                result = {"text": text}
            elif method == "llm/shutdown":
                self._stop.set()
                result = {"ok": True}
            else:
                _send_line(
                    conn,
                    {"id": rid, "error": {"message": f"unknown method {method!r}"}},
                )
                return
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            _send_line(conn, {"id": rid, "error": {"message": str(exc)}})
            return
        _send_line(conn, {"id": rid, "result": result})

    def _generate(self, prompt: str, kwargs: dict[str, Any]) -> str:
        with self._lock:
            backend = self._backend
            if backend is None:
                raise ModelGatewayError("no LLM backend is bound")
            out = backend.generate(prompt, **kwargs)
        return out if isinstance(out, str) else str(out)


# ── client ────────────────────────────────────────────────────


class ModelGatewayClient:
    """In-process proxy to a :class:`ModelGatewayServer` over a socket.

    ``client.generate(prompt, max_tokens=20)`` forwards to the host
    backend. Calls are serialised with an internal lock so a single
    client can be shared across threads. Construct from an explicit path
    or from the ``LOOPLET_LLM_SOCKET`` env var via :meth:`from_env`.
    """

    def __init__(self, socket_path: str, *, connect_timeout: float = 10.0) -> None:
        self._path = socket_path
        self._lock = threading.Lock()
        self._next_id = 0
        self._sock = self._connect(socket_path, connect_timeout)
        self._reader = _LineReader(self._sock)
        self.ready = False
        try:
            init = self._rpc("llm/initialize", {})
            self.ready = bool(init.get("ready"))
        except ModelGatewayError:  # pragma: no cover - tolerated
            self.ready = False

    @classmethod
    def from_env(cls, *, connect_timeout: float = 10.0) -> "ModelGatewayClient | None":
        """Build a client from ``$LOOPLET_LLM_SOCKET``; ``None`` if unset/down."""
        path = os.environ.get(LLM_SOCKET_ENV_VAR)
        if not path:
            return None
        try:
            return cls(path, connect_timeout=connect_timeout)
        except ModelGatewayError:
            return None

    @staticmethod
    def _connect(path: str, timeout: float) -> socket.socket:
        if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - non-POSIX
            raise ModelGatewayError("model gateway requires AF_UNIX sockets")
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
        raise ModelGatewayError(f"could not connect to model gateway at {path!r}: {last_exc}")

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
                raise ModelGatewayError(f"gateway transport failed: {exc}") from exc
            if line is None:
                raise ModelGatewayError("model gateway closed the connection")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ModelGatewayError(f"model gateway emitted non-JSON: {line!r}") from exc
            if not isinstance(parsed, dict):  # pragma: no cover - defensive
                raise ModelGatewayError("malformed model-gateway response")
            if parsed.get("error"):
                raise ModelGatewayError(str(parsed["error"].get("message", "error")))
            return parsed.get("result") or {}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Forward a single-call generation to the host backend."""
        result = self._rpc("llm/generate", {"prompt": prompt, "kwargs": kwargs})
        return str(result.get("text", ""))

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:  # pragma: no cover - best effort
            pass


# ── launcher / handle ─────────────────────────────────────────


class ModelGatewayHandle:
    """Owns a host-resident gateway server thread plus its socket.

    The loader creates one of these (via :meth:`start`) when a cartridge
    declares out-of-process tool servers, *before* spawning them, so the
    socket path is exported as ``LOOPLET_LLM_SOCKET`` and inherited by
    every child. The backend is bound later - at run time - via
    :meth:`set_backend`.
    """

    def __init__(
        self,
        server: ModelGatewayServer,
        socket_path: str,
        socket_dir: str,
        thread: threading.Thread,
    ) -> None:
        self.server = server
        self.socket_path = socket_path
        self._socket_dir = socket_dir
        self._thread = thread

    @classmethod
    def start(cls, *, backend: Any = None, export_env: bool = True) -> "ModelGatewayHandle":
        """Allocate a socket, start the host server thread, export the env var.

        Args:
            backend: Optional initial LLM backend (usually ``None``; bound
                later at run time).
            export_env: When true (default), set ``LOOPLET_LLM_SOCKET`` in
                ``os.environ`` so subsequently-spawned children inherit it.
        """
        if not hasattr(socket, "AF_UNIX"):  # pragma: no cover - non-POSIX
            raise ModelGatewayError("model gateway requires AF_UNIX sockets")
        socket_dir = tempfile.mkdtemp(prefix="looplet-llm-")
        socket_path = os.path.join(socket_dir, "gateway.sock")
        server = ModelGatewayServer(backend=backend)
        thread = threading.Thread(target=server.serve, args=(socket_path,), daemon=True)
        thread.start()
        # Wait briefly for the socket to appear so children that spawn
        # immediately can connect on first use.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not os.path.exists(socket_path):
            time.sleep(0.01)
        if export_env:
            os.environ[LLM_SOCKET_ENV_VAR] = socket_path
        return cls(server, socket_path, socket_dir, thread)

    def set_backend(self, backend: Any) -> None:
        """Bind the live LLM backend the gateway exposes to its clients."""
        self.server.set_backend(backend)

    def close(self) -> None:
        self.server.stop()
        try:
            self._thread.join(timeout=2.0)
        except Exception:  # pragma: no cover - best effort
            pass
        # Drop the env var if it still points at our socket.
        if os.environ.get(LLM_SOCKET_ENV_VAR) == self.socket_path:
            os.environ.pop(LLM_SOCKET_ENV_VAR, None)
        for p in (self.socket_path, self._socket_dir):
            try:
                if os.path.isdir(p):
                    os.rmdir(p)
                elif os.path.exists(p):
                    os.unlink(p)
            except OSError:  # pragma: no cover - best effort
                pass


def _pkg_parent() -> str:  # pragma: no cover - parity helper with SSP
    """Return the directory to add to PYTHONPATH for a source checkout."""
    return str(Path(__file__).resolve().parent.parent)
