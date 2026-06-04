"""Minimal stdio MCP server exposing a single ``done`` completion tool.

Serving ``done`` from an out-of-process MCP server (instead of an
in-process ``tools/done/execute.py``) is what makes this cartridge fully
portable: any conforming loader spawns this command and registers the
tool over the MCP stdio transport — no Python tool body required by the
host.

Standard-library only. Teaching artifact for the MCP wire format.
Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import sys

TOOLS = [
    {
        "name": "done",
        "description": "Signal the planner finished. Pass the final cleaned-up "
        "plan as a numbered list in `summary`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Final plan as a numbered list of steps.",
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
                    "serverInfo": {"name": "planner-done", "version": "0.1"},
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
