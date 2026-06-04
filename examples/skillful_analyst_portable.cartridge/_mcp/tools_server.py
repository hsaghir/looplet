"""Stdio MCP server for the skillful_analyst_portable cartridge.

Serves the three tools that were in-process ``tools/*/execute.py`` bodies
in the original ``skillful_analyst`` cartridge — ``done``, ``read_text``,
and ``write_text`` — over the MCP stdio transport. Moving them out of
process is what makes the twin fully portable: no Python tool body is
required by the host.

Relative paths are resolved against the server's working directory
(``os.getcwd()``), which the loader sets to the host project root — the
portable, host-agnostic equivalent of the original tools'
``resolve_project_root(runtime)`` anchoring.

Standard-library only.
Spec: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio
"""

import json
import os
import sys
from pathlib import Path

TOOLS = [
    {
        "name": "done",
        "description": "Signal task completion. Pass a short summary in `summary`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-line summary of what was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "read_text",
        "description": (
            "Read a UTF-8 text file and return its contents. Returns "
            '{"content": str, "size": int, "lines": int}.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path (absolute or relative to the project root).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_text",
        "description": (
            "Write a UTF-8 text file, creating parent directories as needed. "
            'Returns {"path": str, "bytes": int}. Overwrites existing files.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path (absolute or relative to the project root).",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content, UTF-8.",
                },
            },
            "required": ["path", "content"],
        },
    },
]


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    return p


def _read_text(path):
    p = _resolve(path)
    if not p.is_file():
        return {"error": f"file not found: {p}"}
    text = p.read_text(encoding="utf-8", errors="replace")
    return {"content": text, "size": len(text), "lines": text.count("\n") + 1}


def _write_text(path, content):
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (content or "").encode("utf-8")
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data)}


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def _content(payload):
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
                    "serverInfo": {"name": "skillful-analyst-tools", "version": "0.1"},
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
            if name == "done":
                respond(msg_id, _content({"summary": args.get("summary")}))
            elif name == "read_text":
                respond(msg_id, _content(_read_text(args.get("path", ""))))
            elif name == "write_text":
                respond(
                    msg_id,
                    _content(_write_text(args.get("path", ""), args.get("content", ""))),
                )
            else:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
        else:
            respond(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
