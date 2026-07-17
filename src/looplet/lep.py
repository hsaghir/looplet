"""Loop Effect Protocol (LEP) - run a looplet hook in another process.

This is the productionised host-side bridge first prototyped under
``paper/experiments/cross_runtime_portability/lep_poc``. An
:class:`LEPHookAdapter` is a perfectly ordinary :class:`looplet.loop.LoopHook`
whose authority lives in a separate, possibly non-Python, process. For
every lifecycle slot the adapter:

  1. projects live loop state onto the hook's declared
     :class:`~looplet.hook_view.ViewSpec` (the capability view, §4);
  2. ships the slot tag + that view over line-delimited JSON-RPC;
  3. reconstructs the returned *effect* into a
     :class:`~looplet.hook_decision.HookDecision` via
     :meth:`HookDecision.from_wire`, then maps it to the exact value the
     loop expects from that slot.

The adapter carries **zero decision logic** - it is pure transport plus
the §3 fidelity map. That is what makes an out-of-process hook
behaviourally indistinguishable from the same hook in-process, and hence
what makes a ``kind: lep`` cartridge entry a lossless translation of a
library hook (HOOK_CARTRIDGE_DESIGN.md §5).

:class:`LEPServerBase` is the symmetric convenience for *writing* a
Python policy server: subclass it, declare capabilities, implement
``decide(slot, view) -> HookDecision | dict | None``, and call ``.serve()``.
A server need not import anything else from looplet, mirroring how a Rust
or Go server would speak only the wire protocol.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from looplet.hook_decision import HookDecision
from looplet.hook_view import ViewSpec, extract_view
from looplet.types import ToolResult

logger = logging.getLogger(__name__)

__all__ = ["LEPHookAdapter", "LEPServerBase", "LEPProtocolError"]

LEP_VERSION = "0.1"

#: Failure policies for an authority slot when the server errors or the
#: stream dies. ``fail_closed`` denies/blocks; ``fail_open`` allows;
#: ``continue`` is a no-opinion pass-through (the default for
#: non-authority slots).
_FAILURE_POLICIES = ("fail_closed", "fail_open", "continue")


class LEPProtocolError(RuntimeError):
    """Raised when the LEP transport breaks irrecoverably."""


class LEPHookAdapter:
    """A :class:`LoopHook` whose every slot's authority is a remote process.

    Args:
        server_argv: Command + args to launch the policy server.
        view: The hook's declared capability view (§4). Defaults to the
            ``{tool, args}`` digest view sufficient for permission-style
            policies.
        run_id: Correlation id sent in ``loop/initialize``.
        on_failure: Default failure policy for authority slots when the
            server errors. One of ``fail_closed``/``fail_open``/``continue``.
    """

    def __init__(
        self,
        server_argv: list[str],
        *,
        view: ViewSpec | None = None,
        run_id: str = "lep-run",
        on_failure: str = "fail_closed",
        cartridge_id: str = "lep",
    ) -> None:
        if on_failure not in _FAILURE_POLICIES:
            raise ValueError(f"on_failure must be one of {_FAILURE_POLICIES}, got {on_failure!r}")
        self._argv = list(server_argv)
        self._view = view or ViewSpec(fields=frozenset({"tool", "args"}))
        self._run_id = run_id
        self._on_failure = on_failure
        self._cartridge_id = cartridge_id
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 0
        self.capabilities: dict[str, Any] = {}

    # ── transport ────────────────────────────────────────────────
    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise LEPProtocolError("LEP server process is not running")
        self._next_id += 1
        req = {"id": self._next_id, "method": method, "params": params or {}}
        try:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (BrokenPipeError, OSError, ValueError) as exc:
            # Dead/half-closed server: surface as a protocol error so
            # callers apply ``on_failure`` instead of crashing the host.
            raise LEPProtocolError(f"LEP transport failed: {exc}") from exc
        if not line:
            raise LEPProtocolError("LEP server closed the stream unexpectedly")
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise LEPProtocolError(f"LEP server emitted non-JSON: {line!r}") from exc
        return (parsed or {}).get("result") or {}

    def _decision(
        self,
        slot: str,
        *,
        state: Any = None,
        session_log: Any = None,
        tool_call: Any = None,
        tool_result: Any = None,
        step: int | None = None,
        usage: Any = None,
    ) -> HookDecision | None:
        """Run one slot remotely and reconstruct its effect.

        On any transport/protocol failure, applies ``on_failure`` rather
        than propagating - an out-of-process hook must never crash the
        host loop.
        """
        if self._proc is None:
            return self._failure_decision(slot)
        view = extract_view(
            self._view,
            state=state,
            session_log=session_log,
            tool_call=tool_call,
            tool_result=tool_result,
            step=step,
            usage=usage,
        )
        event = {"slot": slot, "type": slot, "step": step or 0, "payload": view}
        try:
            result = self._rpc("loop/event", event)
        except LEPProtocolError:
            logger.warning("LEP slot %s failed; applying on_failure=%s", slot, self._on_failure)
            return self._failure_decision(slot)
        effect = (result or {}).get("effect")
        try:
            return HookDecision.from_wire(effect)
        except (TypeError, ValueError):
            logger.warning("LEP slot %s returned malformed effect %r", slot, effect)
            return self._failure_decision(slot)

    def _failure_decision(self, slot: str) -> HookDecision | None:
        if self._on_failure == "fail_open" or self._on_failure == "continue":
            return None
        # fail_closed: deny tool calls / block done; no-opinion elsewhere.
        if slot in ("pre_dispatch", "check_permission"):
            return HookDecision(permission="deny", block="policy server unavailable")
        if slot == "check_done":
            return HookDecision(block="policy server unavailable")
        return None

    # ── lifecycle: session open ──────────────────────────────────
    @staticmethod
    def _server_env() -> dict[str, str]:
        """Child env that guarantees a Python LEP server can ``import looplet``.

        A Python policy server typically does
        ``from looplet.lep import LEPServerBase``. When looplet runs from a
        source checkout (not pip-installed into the child's site-packages),
        the package dir is only on the *host's* ``sys.path`` - the spawned
        interpreter would not find it. We prepend the directory that
        contains the ``looplet`` package to the child's ``PYTHONPATH`` so
        the import resolves regardless of how the host obtained looplet.
        Non-Python servers simply ignore the variable.
        """
        env = dict(os.environ)
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        parts = [pkg_parent] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def pre_loop(self, state: Any, session_log: Any, context: Any) -> None:
        self._proc = subprocess.Popen(  # noqa: S603 - argv is operator-supplied
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=self._server_env(),
        )
        try:
            self.capabilities = self._rpc(
                "loop/initialize",
                {
                    "cartridge_id": self._cartridge_id,
                    "run_id": self._run_id,
                    "lep_version": LEP_VERSION,
                    "view": self._view.to_dict(),
                },
            )
        except LEPProtocolError:
            logger.warning("LEP initialize failed; hook will apply on_failure")
            self.capabilities = {}

    # ── slot: pre_prompt → briefing injection ────────────────────
    def pre_prompt(self, state: Any, session_log: Any, context: Any, step_num: int) -> str | None:
        decision = self._decision("pre_prompt", state=state, session_log=session_log, step=step_num)
        return decision.additional_context if decision else None

    # ── slot: pre_dispatch → intercept / cache / deny ────────────
    def pre_dispatch(
        self, state: Any, session_log: Any, tool_call: Any, step_num: int
    ) -> ToolResult | HookDecision | None:
        decision = self._decision(
            "pre_dispatch",
            state=state,
            session_log=session_log,
            tool_call=tool_call,
            step=step_num,
        )
        if decision is None:
            return None
        if decision.updated_result is not None:
            return decision.updated_result
        if decision.is_block():
            return decision
        return None

    # ── slot: check_permission → allow / deny ────────────────────
    def check_permission(self, tool_call: Any, state: Any) -> bool:
        decision = self._decision("check_permission", state=state, tool_call=tool_call)
        if decision is None:
            return True
        return not decision.is_block()

    # ── slot: post_dispatch → follow-up context ──────────────────
    def post_dispatch(
        self, state: Any, session_log: Any, tool_call: Any, tool_result: Any, step_num: int
    ) -> str | None:
        decision = self._decision(
            "post_dispatch",
            state=state,
            session_log=session_log,
            tool_call=tool_call,
            tool_result=tool_result,
            step=step_num,
        )
        return decision.additional_context if decision else None

    # ── slot: check_done → reject premature completion ───────────
    def check_done(
        self, state: Any, session_log: Any, context: Any, step_num: int, tool_call: Any = None
    ) -> str | None:
        decision = self._decision(
            "check_done",
            state=state,
            session_log=session_log,
            tool_call=tool_call,
            step=step_num,
        )
        return decision.block if decision else None

    # ── slot: should_stop → force early termination ──────────────
    def should_stop(self, state: Any, step_num: int, new_entities: int) -> bool:
        decision = self._decision("should_stop", state=state, step=step_num)
        return bool(decision and decision.is_stop())

    # ── lifecycle: session close ─────────────────────────────────
    def on_loop_end(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            self._rpc("loop/shutdown", {"run_id": self._run_id, "outcome": "ok"})
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover - best effort
            proc.kill()
        self._proc = None


def server_argv(server_path: str, *args: str) -> list[str]:
    """Build argv to launch a Python LEP server with this interpreter."""
    return [sys.executable, server_path, *args]


class LEPServerBase:
    """Base class for writing a Python LEP policy server.

    Subclasses set :attr:`slots`/:attr:`effects` (for the
    ``loop/initialize`` capability advertisement) and implement
    :meth:`decide`. Call :meth:`serve` from ``__main__`` to run the
    stdin/stdout JSON-RPC loop. Nothing else from looplet is required -
    a subclass may import :class:`HookDecision` for convenience, or
    return a raw effect dict to stay dependency-free like a Rust server.
    """

    slots: tuple[str, ...] = (
        "pre_prompt",
        "pre_dispatch",
        "check_permission",
        "post_dispatch",
        "check_done",
        "should_stop",
    )
    effects: tuple[str, ...] = (
        "Continue",
        "Allow",
        "Deny",
        "Block",
        "Stop",
        "InjectContext",
        "UpdateArgs",
        "UpdateResult",
        "HookDecision",
    )
    view_fields: tuple[str, ...] = ("tool", "args")
    view_fidelity: str = "digest"
    defaults: dict[str, str] = {"check_permission": "fail_closed", "check_done": "fail_closed"}

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {
            "server_capabilities": {"slots": list(self.slots), "effects": list(self.effects)},
            "view_subscription": {"fields": list(self.view_fields), "fidelity": self.view_fidelity},
            "defaults": dict(self.defaults),
        }

    def decide(self, slot: str, view: dict[str, Any]) -> Any:  # pragma: no cover - abstract
        """Return a :class:`HookDecision`, an effect dict, or ``None``."""
        raise NotImplementedError

    def _effect_dict(self, decision: Any) -> dict[str, Any]:
        if decision is None:
            return {"kind": "Continue"}
        if isinstance(decision, HookDecision):
            return decision.to_wire()
        if isinstance(decision, dict):
            return decision
        raise TypeError(
            f"decide() returned {type(decision).__name__}; expected HookDecision|dict|None"
        )

    def serve(self, stdin: Any = None, stdout: Any = None) -> int:
        stdin = stdin or sys.stdin
        out = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(req, dict):
                continue
            method = req.get("method")
            rid = req.get("id")
            params = req.get("params") or {}
            if method == "loop/initialize":
                result: dict[str, Any] = self.initialize(params)
            elif method == "loop/event":
                result = {
                    "effect": self._effect_dict(
                        self.decide(params.get("slot", ""), params.get("payload") or {})
                    )
                }
            elif method == "loop/shutdown":
                result = {"ok": True}
            else:
                result = {"error": f"unknown method {method!r}"}
            out.write(json.dumps({"id": rid, "result": result}) + "\n")
            out.flush()
            if method == "loop/shutdown":
                break
        return 0
