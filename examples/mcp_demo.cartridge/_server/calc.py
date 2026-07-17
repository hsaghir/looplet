"""A minimal stdio MCP (Model Context Protocol) server.

Exposes one tool - ``add(a, b)`` - and speaks the MCP stdio
transport (newline-delimited JSON-RPC). Bundled alongside the
``mcp_demo`` cartridge so the example is fully self-contained: no
``npm``, no Node, no external server install required.

This file deliberately uses only the Python standard library and is
under 60 lines. It is meant as a teaching artifact for how the MCP
wire format works, not as a production server. For real deployments,
use the official MCP SDK (https://modelcontextprotocol.io/).

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
    }
]


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


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
                    "serverInfo": {"name": "mcp-demo-calc", "version": "0.1"},
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
                respond(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": str(total)}],
                        "isError": False,
                    },
                )
            else:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
