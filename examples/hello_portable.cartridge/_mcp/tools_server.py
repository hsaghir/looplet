"""Out-of-process MCP stdio server exposing the ``greet`` and ``done``
tools — the portable replacement for the original cartridge's
``tools/greet/execute.py`` and ``tools/done/execute.py`` Python bodies.

Speaks the MCP stdio transport (newline-delimited JSON-RPC), so any
conforming loader (Rust/Go/TS/Python) can drive these tools with no
Python required on the host side.

The ``greet`` tool records each greeting in the SHARED greeting log that
lives in a separate State Service process. We reach it through a
:class:`looplet.state_service.StateServiceClient` connected to the socket
the loader exported as ``LOOPLET_STATE_GREETING_LOG``. This is the exact
moment the missing primitive pays off: a tool in THIS process and the
PolitenessGate hook in YET ANOTHER process both mutate/read the same log,
reproducing the in-process ``@ref`` sharing across the process boundary.

Self-contained: only the looplet client + stdlib. No npm/Node.
"""

import json
import os
import sys

from looplet.state_service import StateServiceClient

TOOLS = [
    {
        "name": "greet",
        "description": (
            "Greet a single person by name. Returns the greeting text. "
            "Records the greeting in the shared greeting log."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's name to greet.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "done",
        "description": (
            "Signal that the greeting task is complete. Provide a brief summary of who you greeted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line summary of who was greeted.",
                }
            },
            "required": ["summary"],
        },
    },
]

# Lazily-opened client to the shared greeting-log state service.
_LOG_CLIENT: StateServiceClient | None = None


def _log() -> StateServiceClient | None:
    global _LOG_CLIENT
    if _LOG_CLIENT is not None:
        return _LOG_CLIENT
    socket_path = os.environ.get("LOOPLET_STATE_GREETING_LOG")
    if not socket_path:
        return None
    try:
        _LOG_CLIENT = StateServiceClient(socket_path)
    except Exception:  # noqa: BLE001 - degrade gracefully if state is down
        _LOG_CLIENT = None
    return _LOG_CLIENT


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _call_greet(args):
    name = args["name"]
    text = f"Hello, {name}!"
    client = _log()
    if client is not None:
        client.record(name, text=text)
    return {"greeting": text}


def _call_done(args):
    return {"status": "completed", "summary": args["summary"]}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method = req.get("method")
        msg_id = req.get("id")
        if method == "initialize":
            respond(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hello-portable-tools", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            try:
                if name == "greet":
                    payload = _call_greet(args)
                elif name == "done":
                    payload = _call_done(args)
                else:
                    respond(
                        msg_id,
                        error={"code": -32601, "message": f"unknown tool {name!r}"},
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                respond(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": f"error: {exc}"}],
                        "isError": True,
                    },
                )
                continue
            respond(
                msg_id,
                {
                    "content": [{"type": "text", "text": json.dumps(payload)}],
                    "isError": False,
                },
            )
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
