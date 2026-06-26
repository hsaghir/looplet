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
  - ``{"cmd": "run", "task": {...}, "max_steps": 30,
    "checkpoint_dir": "..."}``
    Execute one loop. Streams ``step`` events back, then a ``done``
    event with the stop reason. When ``checkpoint_dir`` is given, the
    loop saves a :class:`~looplet.checkpoint.Checkpoint` after every step
    and the server emits a ``checkpoint`` frame for each write.
  - ``{"cmd": "resume", "checkpoint": "<id|path>", "task": {...},
    "max_steps": 30, "checkpoint_dir": "..."}``
    Start a run from a previously saved checkpoint. ``checkpoint`` is
    either an absolute path to a ``step_N.json`` file (e.g. the ``path``
    from a ``checkpoint`` frame) or a bare id (``"step_N"``) resolved
    against ``checkpoint_dir``. The loop restores the saved session log
    and continues at step ``N+1`` — even in a brand-new process.
  - ``{"cmd": "cancel"}`` — cooperatively cancel the *in-flight* ``run``.
    May arrive while a ``run`` is streaming: a small reader thread reads
    stdin during the run and trips the loop's
    :class:`~looplet.types.CancelToken`, so the loop stops between turns
    and ends with ``done`` carrying ``stop_reason="cancelled"``. A
    ``cancel`` received when no run is active is a harmless no-op.
  - ``{"cmd": "quit"}`` — terminate cleanly.

* **Outbound events** (server → stdout):

  - ``{"event": "ready"}`` after each successful command. The ``ready``
    emitted after ``load_workspace`` additionally carries a
    ``capabilities`` dict (``{events, cancel, checkpoint, cost,
    permission_authority, stop_reasons[]}``) describing what the loaded
    agent supports, so an orchestrator can degrade gracefully.
  - ``{"event": "step", "step": <Step.to_dict>, "step_num": N}``
  - ``{"event": "checkpoint", "id": "step_N", "step_num": N,
    "path": "..."}`` — emitted for each checkpoint the loop writes during
    a ``run``/``resume`` with ``checkpoint_dir`` set. ``id`` is the
    checkpoint key, ``path`` the on-disk JSON file (pass it back as a
    ``resume`` ``checkpoint`` to restart from that point).
  - ``{"event": "event", "kind": <LifecycleEvent>, "step_num": N,
    "payload": {...}}`` for every in-process lifecycle event emitted
    during a ``run`` (the white-box feed and a fine-grained liveness
    signal — each event is a lease renewal for an orchestrator). The
    ``payload`` is a best-effort JSON-safe view
    (:meth:`looplet.events.EventPayload.to_jsonable`); non-JSON values
    are dropped, never raised on.
  - ``{"event": "done", "stop_reason": "<StopReason>", "steps": N,
    "output": <obj|null>}`` — the terminal frame of every ``run``.
    ``stop_reason`` is a value from the frozen :class:`StopReason` enum
    (``done|max_steps|budget|stagnated|cancelled|error``); ``steps`` is the
    number of step frames emitted (retained for back-compat); ``output`` is
    the structured payload of the accepted ``done()`` sentinel
    (:func:`looplet.done_steps.done_output`) or ``null``.
  - ``{"event": "error", "message": "..."}`` on any error; the
    server emits this diagnostic frame and then still emits a terminal
    ``done`` frame with ``stop_reason="error"`` before continuing to
    accept commands.

The default entrypoint is::

    python -m looplet.rpc

By design this module does **not** ship a built-in backend, default
LLM, or auth flow. Callers wire those by pointing ``set_backend`` at
their own factory (e.g. ``mypkg.backends:make_anthropic``).
"""

from __future__ import annotations

import importlib
import json
import queue
import sys
import threading
import traceback
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from looplet.cartridge import cartridge_to_preset
from looplet.done_steps import done_output
from looplet.loop import composable_loop
from looplet.types import CancelToken, DefaultState

__all__ = [
    "RPCServer",
    "StopReason",
    "STOP_REASONS",
    "map_stop_reason",
    "main",
]

#: Sentinel pushed onto the inbound queue when stdin reaches EOF, so both the
#: command dispatcher and a run's cancel watcher can recognise end-of-stream
#: without confusing it with an empty line.
_EOF = object()


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


class StopReason(str, Enum):
    """Frozen enum of loop termination reasons reported by the ``done`` event.

    The six values are the complete contract surface an orchestrator can
    branch on; they are also advertised verbatim in the capability
    handshake's ``stop_reasons``. ``StopReason`` subclasses ``str`` so a
    member compares equal to (and JSON-serialises as) its wire value.

    * ``DONE`` — the agent called the accepted ``done()`` sentinel.
    * ``MAX_STEPS`` — the step budget was exhausted (the loop ran to its
      ``max_steps`` limit without an explicit stop).
    * ``BUDGET`` — a resource/cost budget hook stopped the loop.
    * ``STAGNATED`` — a stagnation guard stopped the loop (no new progress).
    * ``CANCELLED`` — a cancel token was observed between turns.
    * ``ERROR`` — the run raised before terminating normally.
    """

    DONE = "done"
    MAX_STEPS = "max_steps"
    BUDGET = "budget"
    STAGNATED = "stagnated"
    CANCELLED = "cancelled"
    ERROR = "error"


#: The frozen tuple of termination-reason wire values, derived from
#: :class:`StopReason`. Advertised in the capability handshake and the
#: single source of truth the ``done`` event maps onto.
STOP_REASONS: tuple[str, ...] = tuple(r.value for r in StopReason)

# Loop-internal ``stop_reason`` strings that denote step-budget exhaustion
# (the contract's ``max_steps``) rather than a resource budget. The loop seeds
# ``stop_reason = "budget_exhausted"`` and leaves it untouched when the
# ``while budget_remaining > 0`` guard trips, so this exact token must map to
# ``MAX_STEPS`` — NOT ``BUDGET`` — even though it contains the word "budget".
_MAX_STEPS_TOKENS = frozenset({"budget_exhausted", "max_steps", "exhausted", "step_budget"})


def map_stop_reason(raw: Any, *, errored: bool = False) -> StopReason:
    """Classify a loop-internal ``stop_reason`` into the frozen enum.

    Args:
        raw: The loop's internal ``state._stop_reason`` (e.g. ``"done"``,
            ``"cancelled"``, ``"budget_exhausted"``, or a hook-supplied
            ``HookDecision.stop`` string such as ``"budget_exceeded"`` /
            ``"stagnated"``). ``None``/empty means the loop fell out of its
            ``while budget_remaining > 0`` guard without an explicit reason.
        errored: When the run raised, short-circuits to :attr:`StopReason.ERROR`
            regardless of ``raw``.

    The mapping is order-sensitive: the explicit step-exhaustion tokens
    (:data:`_MAX_STEPS_TOKENS`) are checked *before* the generic ``"budget"``
    substring so the loop's ``"budget_exhausted"`` sentinel resolves to
    ``MAX_STEPS``, while a true resource-budget hook (``"budget_exceeded"``)
    resolves to ``BUDGET``. An unrecognised but intentional hook stop is
    treated as a policy/resource guard (``BUDGET``) — the dominant real use
    case (see the budget-hook recipe).
    """
    if errored:
        return StopReason.ERROR
    if raw is None or not str(raw).strip():
        return StopReason.MAX_STEPS
    text = str(raw).strip().lower()
    if text == StopReason.DONE.value:
        return StopReason.DONE
    if text in _MAX_STEPS_TOKENS:
        return StopReason.MAX_STEPS
    if "cancel" in text:
        return StopReason.CANCELLED
    if "stagn" in text:
        return StopReason.STAGNATED
    if "error" in text or "exception" in text or "fail" in text:
        return StopReason.ERROR
    if "budget" in text or "token" in text or "cost" in text or "quota" in text:
        return StopReason.BUDGET
    # An intentional hook stop we can't categorise (e.g. a bare ``should_stop``
    # → True with no reason). Treat as a policy/resource guard.
    return StopReason.BUDGET


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
    * ``checkpoint`` — always true: like events/cancel it is offered by the
      RPC server itself. The loop checkpoints *any* run when the caller passes
      a per-call ``checkpoint_dir`` to ``run``/``resume`` (``config.checkpoint_dir``
      only sets the auto-default), so an orchestrator must not gate crash-recovery
      on whether a cartridge happened to set ``checkpoint_dir``.
    * ``cost`` — true when a cost sink is wired.
    * ``permission_authority`` — true when a permission hook (a
      ``PermissionEngine``-backed or LEP hook) is present.
    * ``stop_reasons`` — the frozen termination enum the ``done`` event
      reports.
    """
    hooks = getattr(preset, "hooks", None) or []
    return {
        "events": True,
        "cancel": True,
        # A server capability like events/cancel: the loop checkpoints ANY run
        # given a per-call ``checkpoint_dir`` on run/resume (config.checkpoint_dir
        # is only the auto-default), so advertise it unconditionally. Gating it on
        # the cartridge config made orchestrators wrongly skip crash-recovery.
        "checkpoint": True,
        "cost": _has_cost_sink(preset),
        "permission_authority": _has_permission_authority(hooks),
        "stop_reasons": list(STOP_REASONS),
    }


# Terminal LLM-failure sentinel the loop yields when ``llm_call_with_retry``
# is exhausted (``loop.py`` / ``async_loop.py`` ``ToolCall(tool="__llm_error__")``).
# The loop ``break``s right after yielding it but leaves ``stop_reason`` at its
# seeded ``"budget_exhausted"``, so the RPC layer detects this sentinel to map
# the error-termination path onto :attr:`StopReason.ERROR`.
_LLM_ERROR_TOOL = "__llm_error__"


def _terminal_llm_error_step(state: Any) -> Any:
    """Return the trailing ``__llm_error__`` sentinel step, or ``None``.

    The loop swallows LLM failures (retried, then a terminal
    ``__llm_error__`` step) without setting an error ``stop_reason``; this
    lets the RPC layer recognise an error termination without changing the
    loop core.
    """
    steps = getattr(state, "steps", None) or []
    if not steps:
        return None
    last = steps[-1]
    tool = getattr(getattr(last, "tool_call", None), "tool", "")
    return last if tool == _LLM_ERROR_TOOL else None


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

    # ── stdin reader machinery (single source of truth for inbound lines) ──
    # A single daemon thread reads ``in_stream`` line-by-line into ``_inbox``.
    # Outside a run, :meth:`serve_forever` consumes the queue to dispatch
    # commands. *During* a run, a per-run cancel watcher consumes it to trip
    # the in-flight :class:`~looplet.types.CancelToken` on ``{"cmd":"cancel"}``;
    # any non-cancel line it pulls is deferred back for the dispatcher. Only
    # one consumer is ever active at a time (the run executes synchronously on
    # the dispatcher thread), so there is no read race on stdin.
    _inbox: "queue.Queue[Any] | None" = field(default=None, init=False, repr=False)
    _reader_thread: "threading.Thread | None" = field(default=None, init=False, repr=False)
    _active_cancel_token: Any = field(default=None, init=False, repr=False)
    _deferred: list[str] = field(default_factory=list, init=False, repr=False)

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
        self._execute_run(
            task=task,
            max_steps=max_steps,
            checkpoint_dir=msg.get("checkpoint_dir"),
        )

    def cmd_resume(self, msg: dict[str, Any]) -> None:
        """Resume a run from a previously saved checkpoint.

        ``msg["checkpoint"]`` is either an absolute path to a checkpoint
        JSON file (e.g. the ``path`` from a ``checkpoint`` frame) or a bare
        id (``"step_N"``) resolved against ``msg["checkpoint_dir"]``. The
        loaded :class:`~looplet.checkpoint.Checkpoint` is handed to the loop
        as ``initial_checkpoint`` so the continuation restores the saved
        session log and resumes at step ``N+1``.
        """
        if self.preset is None:
            raise ValueError("call load_workspace before resume")
        if self.backend is None:
            raise ValueError("call set_backend before resume")
        ref = msg.get("checkpoint")
        if not ref:
            raise ValueError("resume requires 'checkpoint' (an id or a path)")
        checkpoint_dir = msg.get("checkpoint_dir")
        checkpoint = self._load_checkpoint(str(ref), checkpoint_dir)
        task = msg.get("task") or {}
        max_steps = int(msg.get("max_steps") or self.preset.config.max_steps or 30)
        self._execute_run(
            task=task,
            max_steps=max_steps,
            checkpoint_dir=checkpoint_dir,
            initial_checkpoint=checkpoint,
        )

    # ── shared run engine ────────────────────────────────────────

    def _load_checkpoint(self, ref: str, checkpoint_dir: str | None) -> Any:
        """Resolve a checkpoint reference to a :class:`Checkpoint`.

        ``ref`` is tried first as a filesystem path (an absolute
        ``step_N.json``), then as a bare key resolved against
        ``checkpoint_dir`` via :class:`FileCheckpointStore` — both reuse
        ``checkpoint.py``'s serialisation verbatim.
        """
        from looplet.checkpoint import Checkpoint, FileCheckpointStore

        path = Path(ref)
        if path.is_file():
            try:
                return Checkpoint.from_dict(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"could not load checkpoint file {ref!r}: {exc}") from exc
        if checkpoint_dir:
            cp = FileCheckpointStore(checkpoint_dir).load(Path(ref).name)
            if cp is not None:
                return cp
        raise ValueError(
            f"checkpoint not found: {ref!r} "
            "(pass an existing checkpoint path, or a 'checkpoint_dir' plus its id)"
        )

    def _execute_run(
        self,
        *,
        task: dict[str, Any],
        max_steps: int,
        checkpoint_dir: str | None = None,
        initial_checkpoint: Any = None,
    ) -> None:
        """Run one loop and stream its frames; shared by ``run`` and ``resume``.

        Builds a fresh state each call so a loaded preset can back many runs.
        When ``checkpoint_dir`` is set the loop persists a checkpoint after
        every step (``checkpoint.py`` verbatim); we observe that directory and
        surface each new file as a ``checkpoint`` frame. When
        ``initial_checkpoint`` is set the loop restores it and continues at the
        saved step + 1.
        """
        config = self.preset.config
        changes: dict[str, Any] = {}
        if config.max_steps != max_steps:
            changes["max_steps"] = max_steps
        if checkpoint_dir is not None:
            changes["checkpoint_dir"] = checkpoint_dir
        if initial_checkpoint is not None:
            changes["initial_checkpoint"] = initial_checkpoint

        # Install the cancel token the in-flight run will observe. Reuse a
        # preset-supplied token (so a token handed in via the preset is the one
        # we trip) and otherwise mint a fresh one, folding it into the same
        # ``replace`` as the other run overrides. The loop polls
        # ``config.cancel_token.is_cancelled`` between turns, so flipping this
        # token from the reader thread stops the run cooperatively without
        # touching ``composable_loop``.
        token = config.cancel_token
        if token is None:
            token = CancelToken()
            changes["cancel_token"] = token
        if changes:
            config = replace(config, **changes)

        state = DefaultState(max_steps=max_steps)

        # Start the stdin reader (idempotent) and a per-run watcher that trips
        # the token when a ``{"cmd":"cancel"}`` frame arrives mid-run.
        self._ensure_reader()
        self._active_cancel_token = token
        self._deferred = []
        stop_watcher = threading.Event()
        watcher = threading.Thread(
            target=self._cancel_watcher,
            args=(token, stop_watcher),
            name="rpc-cancel-watcher",
            daemon=True,
        )
        watcher.start()

        # Checkpoint-frame emission. The loop writes a JSON checkpoint after
        # every step when ``checkpoint_dir`` is set (keyed ``step_N``). We never
        # touch the loop or the on-disk format — we observe the directory and
        # announce each NEW file. Stems present before this run started are
        # skipped so a resume that reuses the same directory does not re-announce
        # the checkpoints the original run already reported.
        ckpt_path = Path(config.checkpoint_dir) if getattr(config, "checkpoint_dir", None) else None
        pre_existing: set[str] = set()
        emitted: set[str] = set()
        if ckpt_path is not None and ckpt_path.exists():
            pre_existing = {p.stem for p in ckpt_path.glob("*.json")}

        def _drain_checkpoints() -> None:
            if ckpt_path is None or not ckpt_path.exists():
                return
            for p in sorted(ckpt_path.glob("*.json")):
                if p.stem in pre_existing or p.stem in emitted:
                    continue
                try:
                    data = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue  # partial write mid-save; picked up on a later drain
                emitted.add(p.stem)
                _emit(
                    self.out_stream,
                    {
                        "event": "checkpoint",
                        "id": p.stem,
                        "step_num": data.get("step_number"),
                        "path": str(p),
                    },
                )

        steps_emitted = 0
        errored = False
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
                _drain_checkpoints()
        except Exception as exc:  # noqa: BLE001
            # Emit the diagnostic error frame, then fall through to the
            # terminal ``done`` frame with ``stop_reason="error"`` so every
            # run ends with exactly one ``done`` an orchestrator can await.
            _emit(
                self.out_stream,
                {
                    "event": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(),
                },
            )
            errored = True
        finally:
            # Wind the watcher down before emitting ``done`` so no late frame
            # can follow the terminal one. The watcher only ever blocks on a
            # short queue poll, so the join returns promptly (no hang/leak).
            stop_watcher.set()
            watcher.join(timeout=2.0)
            self._requeue_deferred()
            self._active_cancel_token = None

        # The final checkpoint is written during loop teardown, after the last
        # step is yielded — drain once more so it is never missed.
        _drain_checkpoints()

        # Map the loop's internal termination signal onto the frozen
        # StopReason enum. The loop stashes it as ``state._stop_reason``
        # (also read by StreamingHook); fall back to a public ``stop_reason``
        # if a custom state exposes one.
        raw_reason = getattr(state, "_stop_reason", None)
        if raw_reason is None:
            raw_reason = getattr(state, "stop_reason", None)

        # The loop swallows LLM failures (retried, then a terminal
        # ``__llm_error__`` step) without raising or setting an error
        # ``stop_reason``. Surface that as an error termination too: emit the
        # diagnostic ``error`` frame (parity with the hard-exception path) and
        # map to StopReason.ERROR.
        if not errored:
            _err_step = _terminal_llm_error_step(state)
            if _err_step is not None:
                _emit(
                    self.out_stream,
                    {
                        "event": "error",
                        "message": getattr(_err_step.tool_result, "error", None)
                        or "LLM call failed after all retry attempts",
                    },
                )
                errored = True

        reason = map_stop_reason(raw_reason, errored=errored)
        # ``output`` is the structured payload of the accepted done() sentinel
        # (None when the loop stopped without an accepted done()).
        output = done_output(state, tool_name=config.done_tool)
        _emit(
            self.out_stream,
            {
                "event": "done",
                "stop_reason": reason.value,
                "steps": steps_emitted,
                "output": output,
            },
        )

    # ── stdin reader + mid-run cancel watcher ────────────────────

    def _ensure_reader(self) -> None:
        """Start the single stdin reader thread once, lazily.

        The reader pushes each inbound line onto :attr:`_inbox` and an
        :data:`_EOF` sentinel at end-of-stream, then exits. It is a daemon so a
        process blocked on stdin never wedges interpreter shutdown.
        """
        if self._reader_thread is not None:
            return
        self._inbox = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._pump_stdin, name="rpc-stdin-reader", daemon=True
        )
        self._reader_thread.start()

    def _pump_stdin(self) -> None:
        inbox = self._inbox
        assert inbox is not None  # set by _ensure_reader before the thread starts
        try:
            for line in self._iter_lines():
                inbox.put(line)
        finally:
            inbox.put(_EOF)

    def _cancel_watcher(self, token: Any, stop_event: threading.Event) -> None:
        """Consume :attr:`_inbox` during a run, tripping ``token`` on a
        ``{"cmd":"cancel"}`` frame.

        Blocks only on a short queue poll, so :meth:`cmd_run` can stop it
        promptly via ``stop_event``. Any non-cancel line pulled here is set
        aside in :attr:`_deferred` and handed back to :meth:`serve_forever`
        after the run; an :data:`_EOF` sentinel is put back so the dispatcher
        still observes end-of-stream.
        """
        inbox = self._inbox
        if inbox is None:
            return
        while not stop_event.is_set():
            try:
                item = inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is _EOF:
                inbox.put(_EOF)  # leave EOF for the dispatcher
                return
            line = item.strip() if isinstance(item, str) else ""
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                self._deferred.append(item)
                continue
            if isinstance(parsed, dict) and parsed.get("cmd") == "cancel":
                token.cancel()
                # Keep draining (cancel is idempotent) until the run ends.
            else:
                self._deferred.append(item)

    def _requeue_deferred(self) -> None:
        """Return watcher-pulled non-cancel lines to the inbox, preserving
        order ahead of anything the reader queued after them."""
        if not self._deferred or self._inbox is None:
            self._deferred = []
            return
        remaining: list[Any] = []
        while True:
            try:
                remaining.append(self._inbox.get_nowait())
            except queue.Empty:
                break
        for item in self._deferred:
            self._inbox.put(item)
        for item in remaining:
            self._inbox.put(item)
        self._deferred = []

    # ── dispatch loop ────────────────────────────────────────────

    def serve_forever(self) -> int:
        """Block reading commands until ``quit`` or EOF.

        Commands are consumed from :attr:`_inbox`, fed by the single stdin
        reader thread (:meth:`_ensure_reader`). During a ``run`` the reader
        keeps filling the inbox and the run's cancel watcher consumes it; any
        non-cancel line the watcher pulls is handed back here afterwards, so
        command ordering is preserved.
        """
        handlers: dict[str, str] = {
            "load_workspace": "cmd_load_workspace",
            "set_backend": "cmd_set_backend",
            "run": "cmd_run",
            "resume": "cmd_resume",
        }
        self._ensure_reader()
        assert self._inbox is not None
        while True:
            item = self._inbox.get()
            if item is _EOF:
                break
            line = item.strip() if isinstance(item, str) else ""
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
            if cmd == "cancel":
                # A cancel observed here has no in-flight run to stop (a run's
                # watcher consumes cancels while one is active). No-op.
                continue
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
