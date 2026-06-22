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

  - ``{"event": "ready"}`` after each successful command. The ``ready``
    emitted after ``load_workspace`` additionally carries a
    ``capabilities`` dict (``{events, cancel, checkpoint, cost,
    permission_authority, stop_reasons[]}``) describing what the loaded
    agent supports, so an orchestrator can degrade gracefully.
  - ``{"event": "step", "step": <Step.to_dict>, "step_num": N}``
  - ``{"event": "event", "kind": <LifecycleEvent>, "step_num": N,
    "payload": {...}}`` for every in-process lifecycle event emitted
    during a ``run`` (the white-box feed and a fine-grained liveness
    signal — each event is a lease renewal for an orchestrator). The
    ``payload`` is a best-effort JSON-safe view
    (:meth:`looplet.events.EventPayload.to_jsonable`); non-JSON values
    are dropped, never raised on.
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


#: The frozen set of loop termination reasons advertised in the capability
#: handshake. Formalised as a ``StopReason`` enum in §1.3; kept here as the
#: single source of truth the ``done`` event maps onto.
STOP_REASONS: tuple[str, ...] = (
    "done",
    "max_steps",
    "budget",
    "stagnated",
    "cancelled",
    "error",
)


def _has_permission_authority(hooks: Iterable[Any]) -> bool:
    """True iff any hook can gate tool dispatch on a permission decision.

    Covers the PermissionEngine-backed
    :class:`~looplet.permissions.PermissionHook`, the out-of-process
    :class:`~looplet.lep.LEPHookAdapter` policy server, and any back-compat
    hook exposing ``check_permission`` — all of which advertise that surface.
    """
    return any(hasattr(hook, "check_permission") for hook in hooks)


def _has_cost_sink(preset: Any) -> bool:
    """True iff a cost sink is wired — a :class:`~looplet.cost.CostHook`
    feeding a tracker, or a bare :class:`~looplet.cost.CostTracker` published
    as a resource."""
    try:
        from looplet.cost import CostHook, CostTracker  # noqa: PLC0415
    except Exception:  # pragma: no cover - cost module is always importable
        return False
    hooks = getattr(preset, "hooks", None) or []
    if any(isinstance(h, CostHook) for h in hooks):
        return True
    resources = getattr(preset, "resources", None) or {}
    return any(isinstance(v, CostTracker) for v in resources.values())


def _capabilities(preset: Any) -> dict[str, Any]:
    """Describe what the loaded preset supports, for the handshake.

    An orchestrator reads this to degrade gracefully across heterogeneous
    agents:

    * ``events`` / ``cancel`` — always offered by the RPC server itself
      (every loop emits :class:`~looplet.events.LifecycleEvent`\\ s and
      accepts a cancel token), so they are unconditionally true.
    * ``checkpoint`` — true when the preset persists checkpoints
      (``config.checkpoint_dir`` is set).
    * ``cost`` — true when a cost sink is wired.
    * ``permission_authority`` — true when a permission hook (a
      ``PermissionEngine``-backed or LEP hook) is present.
    * ``stop_reasons`` — the frozen termination enum the ``done`` event
      reports.
    """
    config = getattr(preset, "config", None)
    hooks = getattr(preset, "hooks", None) or []
    return {
        "events": True,
        "cancel": True,
        "checkpoint": bool(getattr(config, "checkpoint_dir", None)),
        "cost": _has_cost_sink(preset),
        "permission_authority": _has_permission_authority(hooks),
        "stop_reasons": list(STOP_REASONS),
    }


class _RPCEventForwarder:
    """Pure ``on_event`` observer that streams lifecycle events to the wire.

    Appended to the run's hook list so it subscribes through the loop's
    existing ``on_event`` bus — ``composable_loop``'s core is untouched.
    It implements *only* ``on_event`` (no per-method slots) so the loop's
    on-event/per-method deduplication (``PRE_TOOL_USE``→``pre_dispatch``,
    ``POST_TOOL_USE``→``post_dispatch``) never skips it: a pure observer
    receives every lifecycle event, including those with a per-method
    equivalent.

    Each event becomes a ``{"event": "event", "kind": <name>, "step_num":
    N, "payload": {...}}`` frame. Serialisation is best-effort and
    self-contained: a malformed payload is dropped, never raised into the
    loop (the loop's ``emit_event`` also guards — this is belt-and-suspenders).
    """

    def __init__(self, out_stream: TextIO) -> None:
        self._out = out_stream

    def on_event(self, payload: Any) -> None:
        try:
            event = getattr(payload, "event", None)
            kind = getattr(event, "value", None) or str(event)
            to_jsonable = getattr(payload, "to_jsonable", None)
            body = to_jsonable() if callable(to_jsonable) else {}
            _emit(
                self._out,
                {
                    "event": "event",
                    "kind": kind,
                    "step_num": int(getattr(payload, "step_num", 0) or 0),
                    "payload": body,
                },
            )
        except Exception:  # noqa: BLE001 - event forwarding must never break the run
            return


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
            {
                "event": "ready",
                "loaded": str(path),
                "tools": list(self.preset.tools.tool_names),
                "capabilities": _capabilities(self.preset),
            },
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
        # Subscribe to the lifecycle-event bus by appending a pure observer
        # hook (do NOT mutate preset.hooks — it is reused across runs).
        run_hooks = [*self.preset.hooks, _RPCEventForwarder(self.out_stream)]
        try:
            for step in composable_loop(
                llm=self.backend,
                tools=self.preset.tools,
                state=state,
                config=config,
                hooks=run_hooks,
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
