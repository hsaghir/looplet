"""Out-of-process MCP stdio server exposing the coder cartridge's full
tool set — the portable replacement for the original cartridge's 16
in-process ``tools/<name>/execute.py`` bodies.

Speaks the MCP stdio transport (newline-delimited JSON-RPC), so any
conforming loader can drive these tools. The tool *logic* is vendored
unchanged from the original cartridge (``coder_lib_tools.py`` helpers +
each ``_tools/<name>.py`` body); this server is the protocol adapter
that wires three things the original got from its Python host:

* **workspace** — the original read ``ctx.resources['workspace_config'].path``
  (an in-process ``@ref`` resource). Here it comes from
  ``$LOOPLET_PROJECT_ROOT`` (or cwd), exposed through a tiny shim with
  the same ``.path`` attribute.

* **file_cache** — the original shared a single in-process ``FileCache``
  between the file tools and the StaleFile/FileCache hooks via
  ``@file_cache``. Here it lives in a separate **State Service** process;
  we reach it through a :class:`StateServiceClient` (socket exported as
  ``LOOPLET_STATE_FILE_CACHE``). A thin proxy gives it the exact
  FileCache method surface the tools call.

* **ctx.llm** — ``subagent`` and ``web_fetch`` need the host LLM. Here
  it is reached over the **Model Gateway Protocol**: a
  :class:`ModelGatewayClient` (socket ``LOOPLET_LLM_SOCKET``) forwards
  generation to the host's bound backend. When no backend is bound,
  ``ctx.llm`` degrades to ``None`` and the tools return their original
  no-LLM behaviour.

Each ``tools/call`` builds a lightweight :class:`ToolContext` wiring
those three pieces and invokes the vendored ``execute(ctx, **args)``.
"""

import importlib.util
import inspect
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
# Put this dir on the path so the vendored tool bodies can
# ``from coder_lib_tools import ...`` exactly as they did in-process.
sys.path.insert(0, _HERE)

from looplet.model_gateway import ModelGatewayClient  # noqa: E402
from looplet.state_service import StateServiceClient  # noqa: E402
from looplet.types import ToolContext  # noqa: E402

_TOOLS_DIR = os.path.join(_HERE, "_tools")

# Canonical tool order (matches the original cartridge surface).
_TOOL_NAMES = [
    "bash",
    "list_dir",
    "read_file",
    "write_file",
    "edit_file",
    "multi_edit",
    "notebook_edit",
    "glob",
    "grep",
    "git_inspect",
    "worktree",
    "web_fetch",
    "subagent",
    "todo",
    "think",
    "done",
]

# Map Python annotations to JSON-Schema types for the MCP inputSchema.
_ANNOT_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# The vendored tool modules use ``from __future__ import annotations`` (PEP
# 563), so ``inspect.signature(fn).parameters[...].annotation`` is the
# *string* name of the type (e.g. ``"list"``), not the type object. Map by
# name too, otherwise structured params (``edits: list``) silently fall back
# to ``"string"`` in the advertised schema — which makes the model encode
# them as a JSON string and trips the tool's list validation.
_ANNOT_TYPES_BY_NAME = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _json_type(annotation: object) -> str:
    """Resolve a parameter annotation (type object *or* PEP-563 string)
    to a JSON-Schema type name, defaulting to ``"string"``."""
    if isinstance(annotation, str):
        # Strip an optional ``X | None`` / ``Optional[X]`` wrapper to the
        # leading bare name so e.g. ``"list | None"`` → ``"array"``.
        head = annotation.split("|", 1)[0].strip()
        head = head.removeprefix("Optional[").rstrip("]").strip()
        return _ANNOT_TYPES_BY_NAME.get(head, "string")
    return _ANNOT_TYPES.get(annotation, "string")


# ── workspace shim (replaces the @workspace_config resource) ─────────
class _WorkspaceConfig:
    def __init__(self, path: str) -> None:
        self.path = path


def _project_root() -> str:
    return os.environ.get("LOOPLET_PROJECT_ROOT") or os.getcwd()


# ── file_cache proxy (replaces the @file_cache resource) ─────────────
class _FileCacheProxy:
    """Forwards the FileCache method surface to the State Service.

    Exposes exactly the methods the tools call on the in-process cache
    (``record``/``invalidate``/``is_unchanged``/``was_read``/
    ``record_bash``/``recent_bash_repeats``). The socket is shared with
    the StaleFile/FileCache LEP hooks, so all processes see one cache.
    """

    def __init__(self, client: "StateServiceClient | None") -> None:
        self._c = client

    def record(self, path: str) -> None:
        if self._c is not None:
            self._c.record(path)

    def invalidate(self, path: str) -> None:
        if self._c is not None:
            self._c.invalidate(path)

    def is_unchanged(self, path: str) -> bool:
        if self._c is None:
            return False
        return bool(self._c.is_unchanged(path))

    def was_read(self, path: str) -> bool:
        if self._c is None:
            return False
        return bool(self._c.was_read(path))

    def record_bash(self, command: str) -> int:
        if self._c is None:
            return 0
        return int(self._c.record_bash(command))

    def recent_bash_repeats(self, command: str) -> int:
        if self._c is None:
            return 0
        return int(self._c.recent_bash_repeats(command))


_CACHE_CLIENT: "StateServiceClient | None" = None
_CACHE_TRIED = False


def _file_cache() -> _FileCacheProxy:
    global _CACHE_CLIENT, _CACHE_TRIED
    if not _CACHE_TRIED:
        _CACHE_TRIED = True
        socket_path = os.environ.get("LOOPLET_STATE_FILE_CACHE")
        if socket_path:
            try:
                _CACHE_CLIENT = StateServiceClient(socket_path)
            except Exception:  # noqa: BLE001 - degrade gracefully
                _CACHE_CLIENT = None
    return _FileCacheProxy(_CACHE_CLIENT)


# ── host LLM (Model Gateway) ─────────────────────────────────────────
def _host_llm() -> "ModelGatewayClient | None":
    """Return a ready Model Gateway client, or None when no backend is bound.

    Re-checks readiness on every call so late backend binding (the host
    binds the LLM at ``AgentPreset.run``, after this server started) is
    picked up, and a running-but-unbound gateway degrades to ``None``.
    """
    client = ModelGatewayClient.from_env()
    if client is None:
        return None
    if not getattr(client, "ready", False):
        client.close()
        return None
    return client


# ── load vendored tool bodies + schemas ──────────────────────────────
def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_coder_tool_{name}", os.path.join(_TOOLS_DIR, f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.execute


def _schema_from_signature(name: str, fn) -> dict:
    """Derive the MCP inputSchema from ``execute``'s signature.

    Avoids any YAML dependency (the spawned interpreter reaches looplet
    over an injected ``PYTHONPATH`` but has no guaranteed PyYAML). The
    keyword-only params are the tool arguments; the leading positional
    ``ctx`` (when present) is the host wiring and is excluded.

    Optionality note: looplet's MCP adapter flattens an inputSchema to a
    simple ``{name: type_string}`` dict and DROPS the JSON-Schema
    ``required`` list — it then treats a param as optional only when its
    type/description string begins with ``(optional)``. So we encode a
    Python default as an ``"(optional) <type>"`` type string; that keeps
    the param *known* (passable) while marking it not-required, exactly
    matching the original ``tool.yaml`` ``default:`` semantics.
    """
    properties: dict = {}
    required: list[str] = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            continue  # ctx — host wiring, not a model-facing argument
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        jtype = _json_type(param.annotation)
        if param.default is inspect.Parameter.empty:
            properties[param.name] = {"type": jtype}
            required.append(param.name)
        else:
            properties[param.name] = {"type": f"(optional) {jtype}"}
    doc = (inspect.getdoc(fn) or name).strip().splitlines()
    return {
        "name": name,
        "description": doc[0] if doc else name,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


_EXECUTORS = {name: _load_tool(name) for name in _TOOL_NAMES}
_TOOLS = [_schema_from_signature(name, _EXECUTORS[name]) for name in _TOOL_NAMES]


def _wants_ctx(fn) -> bool:
    """True if ``execute`` takes a positional ``ctx`` (most tools do; the
    pure builtins ``think``/``done`` declare only keyword-only params)."""
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            return True
    return False


_WANTS_CTX = {name: _wants_ctx(fn) for name, fn in _EXECUTORS.items()}


def _make_ctx() -> ToolContext:
    return ToolContext(
        resources={
            "workspace_config": _WorkspaceConfig(_project_root()),
            "file_cache": _file_cache(),
        },
        llm=_host_llm(),
    )


# ── MCP stdio plumbing ───────────────────────────────────────────────
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
                    "serverInfo": {"name": "coder-portable-tools", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            respond(msg_id, {"tools": _TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {}) or {}
            executor = _EXECUTORS.get(name)
            if executor is None:
                respond(msg_id, error={"code": -32601, "message": f"unknown tool {name!r}"})
                continue
            try:
                if _WANTS_CTX.get(name, True):
                    payload = executor(_make_ctx(), **args)
                else:
                    payload = executor(**args)
            except Exception as exc:  # noqa: BLE001
                respond(
                    msg_id,
                    {
                        "content": [
                            {"type": "text", "text": f"error: {exc}\n{traceback.format_exc()}"}
                        ],
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
