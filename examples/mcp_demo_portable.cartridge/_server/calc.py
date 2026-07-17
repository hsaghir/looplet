"""A minimal stdio MCP (Model Context Protocol) server.

Portable twin of the ``mcp_demo`` calc server. Exposes TWO tools -
``add(a, b)`` and ``done(total)`` - over the MCP stdio transport
(newline-delimited JSON-RPC). Bundling the completion sentinel
(``done``) here, instead of as an in-process ``tools/done/execute.py``,
is what makes the ``mcp_demo_portable`` cartridge fully portable: no
Python tool body is required by the host.

Standard-library only, no external deps. Teaching artifact for the MCP
wire format; for real deployments use the official MCP SDK
(https://modelcontextprotocol.io/).

Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import sys

TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First addend."},
                "b": {"type": "integer", "description": "Second addend."},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "done",
        "description": "Report the computed total and finish.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "total": {
                    "type": "integer",
                    "description": "The sum returned by the `add` tool.",
                },
            },
            "required": ["total"],
        },
    },
]


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _content(payload):
    """Wrap a JSON payload as MCP tool-call text content."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}


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
                    "serverInfo": {"name": "mcp-demo-portable-calc", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue  # notifications carry no id, no response
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "add":
                total = int(args["a"]) + int(args["b"])
                respond(msg_id, _content({"sum": total}))
            elif name == "done":
                respond(
                    msg_id,
                    _content({"total": args.get("total"), "status": "completed"}),
                )
            else:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
