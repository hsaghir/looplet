"""RPC mode — drive a looplet loop over stdio JSONL.

Lets non-Python clients (TypeScript, Rust, Go, shell) embed looplet
without FFI. The protocol is the smallest thing that works:

* **Frame:** one JSON object per line, LF-delimited (matching Pi's
  ``--mode rpc`` framing convention).
* **Inbound commands** (stdin → server):

  - ``{"cmd": "load_workspace", "path": "...", "runtime": {...}}``
    Load an :class:`AgentPreset` from a workspace. Required first.
  - ``{"cmd": "set_backend", "factory": "pkg.module:fn"}``
    Import a callable that returns an :class:`LLMBackend`. Required
    before ``run``.
  - ``{"cmd": "run", "task": {...}, "max_steps": 30}``
    Execute one loop. Streams ``step`` events back, then a ``done``
    event with the stop reason.
  - ``{"cmd": "quit"}`` — terminate cleanly.

* **Outbound events** (server → stdout):

  - ``{"event": "ready"}`` after each successful command.
  - ``{"event": "step", "step": <Step.to_dict>, "step_num": N}``
  - ``{"event": "done", "stop_reason": "...", "steps": N}``
  - ``{"event": "error", "message": "..."}`` on any error; the
    server then continues accepting commands.

The default entrypoint is::

    python -m looplet.rpc

By design this module does **not** ship a built-in backend, default
LLM, or auth flow. Callers wire those by pointing ``set_backend`` at
their own factory (e.g. ``mypkg.backends:make_anthropic``).
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from looplet.cartridge import cartridge_to_preset
from looplet.loop import composable_loop
from looplet.types import DefaultState

__all__ = [
    "RPCServer",
    "main",
]


def _emit(out: TextIO, payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, default=_json_default) + "\n")
    out.flush()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return repr(obj)


def _import_factory(spec: str) -> Callable[..., Any]:
    """Resolve ``"pkg.module:func"`` to a callable."""
    if ":" not in spec:
        raise ValueError(f"factory must be 'pkg.module:func', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise ValueError(f"factory {spec!r} is not callable")
    return fn


@dataclass
class RPCServer:
    """Stateful stdio RPC server. One instance per process.

    Args:
        in_stream: Where to read JSONL commands from. Defaults to stdin.
        out_stream: Where to emit JSONL events. Defaults to stdout.
    """

    in_stream: TextIO = field(default_factory=lambda: sys.stdin)
    out_stream: TextIO = field(default_factory=lambda: sys.stdout)
    preset: Any = None
    backend: Any = None

    # ── command handlers ─────────────────────────────────────────

    def cmd_load_workspace(self, msg: dict[str, Any]) -> None:
        path = msg.get("path")
        if not path:
            raise ValueError("load_workspace requires 'path'")
        runtime = msg.get("runtime") or None
        self.preset = cartridge_to_preset(Path(path), runtime=runtime)
        _emit(
            self.out_stream,
            {"event": "ready", "loaded": str(path), "tools": list(self.preset.tools.tool_names)},
        )

    def cmd_set_backend(self, msg: dict[str, Any]) -> None:
        factory = msg.get("factory")
        if not factory:
            raise ValueError("set_backend requires 'factory' (dotted 'pkg.module:fn')")
        kwargs = msg.get("kwargs") or {}
        fn = _import_factory(factory)
        self.backend = fn(**kwargs)
        _emit(self.out_stream, {"event": "ready", "backend": factory})

    def cmd_run(self, msg: dict[str, Any]) -> None:
        if self.preset is None:
            raise ValueError("call load_workspace before run")
        if self.backend is None:
            raise ValueError("call set_backend before run")
        task = msg.get("task") or {}
        max_steps = int(msg.get("max_steps") or self.preset.config.max_steps or 30)

        # Build a fresh state each run so RPC clients can re-use the
        # same loaded preset across many invocations.
        config = self.preset.config
        if config.max_steps != max_steps:
            from dataclasses import replace

            config = replace(config, max_steps=max_steps)
        state = DefaultState(max_steps=max_steps)

        steps_emitted = 0
        stop_reason: str | None = None
        try:
            for step in composable_loop(
                llm=self.backend,
                tools=self.preset.tools,
                state=state,
                config=config,
                hooks=self.preset.hooks,
                task=task,
            ):
                steps_emitted += 1
                _emit(
                    self.out_stream,
                    {"event": "step", "step_num": step.number, "step": step.to_dict()},
                )
            stop_reason = getattr(state, "stop_reason", None) or "exhausted"
        except Exception as exc:  # noqa: BLE001
            _emit(
                self.out_stream,
                {
                    "event": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(),
                },
            )
            return
        _emit(
            self.out_stream,
            {"event": "done", "stop_reason": stop_reason, "steps": steps_emitted},
        )

    # ── dispatch loop ────────────────────────────────────────────

    def serve_forever(self) -> int:
        """Block reading commands until ``quit`` or EOF."""
        handlers: dict[str, str] = {
            "load_workspace": "cmd_load_workspace",
            "set_backend": "cmd_set_backend",
            "run": "cmd_run",
        }
        for line in self._iter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                _emit(self.out_stream, {"event": "error", "message": f"bad JSON: {exc}"})
                continue
            cmd = msg.get("cmd")
            if cmd == "quit":
                _emit(self.out_stream, {"event": "ready", "quit": True})
                return 0
            handler_name = handlers.get(cmd or "")
            if handler_name is None:
                _emit(self.out_stream, {"event": "error", "message": f"unknown cmd: {cmd!r}"})
                continue
            handler = getattr(self, handler_name)
            try:
                handler(msg)
            except Exception as exc:  # noqa: BLE001
                _emit(
                    self.out_stream,
                    {"event": "error", "message": f"{type(exc).__name__}: {exc}"},
                )
        return 0

    def _iter_lines(self) -> Iterable[str]:
        return iter(self.in_stream.readline, "")


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001 - CLI parity
    return RPCServer().serve_forever()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
