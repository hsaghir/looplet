"""Minimal stdio MCP server exposing a single ``done`` completion tool.

Serving ``done`` over MCP (instead of an in-process tool body) keeps the
planner child fully portable. Standard-library only.
Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import sys

TOOLS = [
    {
        "name": "done",
        "description": "Return the finished plan. Pass the numbered list as `summary`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Numbered plan, 3-7 short steps.",
                },
            },
            "required": ["summary"],
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
                    "serverInfo": {"name": "planner-child-done", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "done":
                payload = {"status": "completed", "summary": args.get("summary")}
                respond(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": json.dumps(payload)}],
                        "isError": False,
                    },
                )
            else:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
