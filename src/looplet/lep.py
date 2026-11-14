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
    :meth:`HookDecision.from_wire`. The loop consumes this complete
    decision under the same slot contract as a Python hook. Direct calls
    to the adapter's named methods retain their legacy return shapes.

Policy remains in the remote process; the host owns effect applicability and
transport-failure behavior. The process boundary does not discard supported
rewrites, stop reasons, denial reasons, or decision metadata. Legacy LEP
block-only permission decisions remain denials; use ``Deny`` for the portable
cross-transport permission contract.

:class:`LEPServerBase` is the symmetric convenience for *writing* a
Python policy server: subclass it, declare capabilities, implement
``decide(slot, view) -> HookDecision | dict | None``, and call ``.serve()``.
A server need not import anything else from looplet, mirroring how a Rust
or Go server would speak only the wire protocol.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from looplet.hook_decision import HookDecision
from looplet.hook_view import ViewSpec, extract_view
from looplet.protocol_context import remaining_deadline
from looplet.types import RunEnvelope, ToolResult

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
        env: Optional host-owned environment captured for the policy process.
    """

    def __init__(
        self,
        server_argv: list[str],
        *,
        view: ViewSpec | None = None,
        run_id: str = "lep-run",
        on_failure: str = "fail_closed",
        cartridge_id: str = "lep",
        run_envelope: RunEnvelope | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_message_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if on_failure not in _FAILURE_POLICIES:
            raise ValueError(f"on_failure must be one of {_FAILURE_POLICIES}, got {on_failure!r}")
        self._argv = list(server_argv)
        self._view = view or ViewSpec(fields=frozenset({"tool", "args"}))
        self._run_id = run_id
        self._on_failure = on_failure
        self._cartridge_id = cartridge_id
        self._run_envelope = run_envelope
        self._env = dict(env) if env is not None else None
        self._timeout = timeout
        self._max_message_bytes = max_message_bytes
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._rpc_lock = threading.RLock()
        self._stdout_cond = threading.Condition()
        self._stdout_lines: list[str] = []
        self._stdout_eof = False
        self._stdout_error: BaseException | None = None
        self._stdout_reader: threading.Thread | None = None
        self._closing = False
        self.capabilities: dict[str, Any] = {}

    # ── transport ────────────────────────────────────────────────
    def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        expect_response: bool = True,
    ) -> dict[str, Any]:
        with self._rpc_lock:
            return self._rpc_locked(method, params, expect_response=expect_response)

    def _rpc_locked(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        expect_response: bool = True,
    ) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise LEPProtocolError("LEP server process is not running")
        self._next_id += 1
        wire_params = dict(params or {})
        if self._run_envelope is not None:
            wire_params.setdefault("run_envelope", self._run_envelope.to_dict())
        req = {"id": self._next_id, "method": method, "params": wire_params}
        try:
            remaining = remaining_deadline(self._run_envelope)
            if remaining is not None and remaining <= 0:
                raise LEPProtocolError("LEP call exceeded run deadline")
            effective_timeout = (
                min(self._timeout, remaining) if remaining is not None else self._timeout
            )
            payload_str = json.dumps(req) + "\n"
            if len(payload_str.encode("utf-8")) > self._max_message_bytes:
                raise LEPProtocolError("LEP request exceeded max_message_bytes")

            def exchange() -> str:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(payload_str)
                proc.stdin.flush()
                return proc.stdout.readline(self._max_message_bytes + 1) if expect_response else ""

            if self._stdout_reader is not None:
                assert proc.stdin is not None
                proc.stdin.write(payload_str)
                proc.stdin.flush()
                if not expect_response:
                    line = ""
                else:
                    end_ts = (
                        None if effective_timeout is None else time.monotonic() + effective_timeout
                    )
                    with self._stdout_cond:
                        while (
                            not self._stdout_lines
                            and not self._stdout_eof
                            and self._stdout_error is None
                        ):
                            rem = None if end_ts is None else end_ts - time.monotonic()
                            if rem is not None and rem <= 0:
                                self._abort_process(proc)
                                if remaining is not None and effective_timeout == remaining:
                                    raise LEPProtocolError("LEP call exceeded run deadline")
                                raise LEPProtocolError("LEP call timed out")
                            self._stdout_cond.wait(rem)
                        if self._stdout_lines:
                            line = self._stdout_lines.pop(0)
                            self._stdout_cond.notify_all()
                        elif self._stdout_error is not None:
                            err = self._stdout_error
                            self._stdout_error = None
                            raise err
                        else:
                            line = ""
            elif effective_timeout is None:
                line = exchange()
            else:
                done = threading.Event()
                lines: list[str] = []
                errors: list[BaseException] = []

                def communicate() -> None:
                    try:
                        lines.append(exchange())
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=communicate, daemon=True, name="looplet-lep-rpc")
                worker.start()
                if not done.wait(effective_timeout):
                    self._abort_process(proc)
                    worker.join(timeout=1.0)
                    if not worker.is_alive():
                        for stream in (proc.stdin, proc.stdout):
                            try:
                                stream.close()
                            except (OSError, ValueError):
                                pass
                    if remaining is not None and effective_timeout == remaining:
                        raise LEPProtocolError("LEP call exceeded run deadline")
                    raise LEPProtocolError("LEP call timed out")
                if errors:
                    raise errors[0]
                line = lines[0]
            if not expect_response:
                return {}
            if len(line.encode("utf-8")) > self._max_message_bytes:
                raise LEPProtocolError("LEP response exceeded max_message_bytes")
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
        if not isinstance(parsed, dict):
            raise LEPProtocolError("LEP server response must be a JSON object")
        if parsed.get("id") != req["id"]:
            raise LEPProtocolError(
                f"LEP server response id mismatch: expected {req['id']}, got {parsed.get('id')}"
            )
        if "error" in parsed:
            if "result" in parsed:
                raise LEPProtocolError("LEP server response cannot contain both result and error")
            raise LEPProtocolError(f"LEP server returned error: {parsed['error']!r}")
        if "result" not in parsed or not isinstance(parsed["result"], dict):
            raise LEPProtocolError("LEP server response result must be a JSON object")
        return parsed["result"]

    def _abort_process(self, proc: subprocess.Popen[str]) -> None:
        self._proc = None
        reader = self._stdout_reader
        self._stdout_reader = None
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (AttributeError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        with self._stdout_cond:
            self._stdout_eof = True
            self._stdout_cond.notify_all()

    def set_run_envelope(self, envelope: RunEnvelope | None) -> None:
        """Bind host correlation context for subsequent policy requests."""
        self._run_envelope = envelope

    def _exchange_slot_view(
        self, slot: str, step: int, view: dict[str, Any]
    ) -> HookDecision | None:
        event = {"slot": slot, "type": slot, "step": step, "payload": view}
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
        return self._exchange_slot_view(slot, step or 0, view)

    def _failure_decision(self, slot: str) -> HookDecision | None:
        if self._on_failure == "fail_open" or self._on_failure == "continue":
            return None
        # fail_closed: deny tool calls / block done; no-opinion elsewhere.
        if slot in ("pre_dispatch", "check_permission"):
            return HookDecision(permission="deny", block="policy server unavailable")
        if slot == "check_done":
            return HookDecision(block="policy server unavailable")
        return None

    def _invoke_hook_return(self, slot: str, *args: Any, **kwargs: Any) -> Any:
        """Return the complete effect to the loop; direct methods stay legacy."""
        method = getattr(self, slot)
        if getattr(method, "__func__", None) is not getattr(LEPHookAdapter, slot, None):
            return method(*args, **kwargs)
        arguments = inspect.signature(method).bind(*args, **kwargs).arguments
        decision = self._decision(
            slot,
            state=arguments.get("state"),
            session_log=arguments.get("session_log"),
            tool_call=arguments.get("tool_call"),
            tool_result=arguments.get("tool_result"),
            step=arguments.get("step_num"),
        )
        if slot == "check_permission" and decision is not None and decision.is_block():
            decision = replace(decision, permission="deny")
        return decision

    async def _invoke_hook_return_async(self, slot: str, *args: Any, **kwargs: Any) -> Any:
        """Non-blocking async counterpart to :meth:`_invoke_hook_return`."""
        import asyncio

        method = getattr(self, slot)
        if getattr(method, "__func__", None) is not getattr(LEPHookAdapter, slot, None):
            raw = method(*args, **kwargs)
            return await raw if inspect.isawaitable(raw) else raw
        if slot in ("pre_loop", "on_loop_end"):
            return await asyncio.to_thread(method, *args, **kwargs)
        arguments = inspect.signature(method).bind(*args, **kwargs).arguments
        if self._proc is None:
            decision = self._failure_decision(slot)
        else:
            view = extract_view(
                self._view,
                state=arguments.get("state"),
                session_log=arguments.get("session_log"),
                tool_call=arguments.get("tool_call"),
                tool_result=arguments.get("tool_result"),
                step=arguments.get("step_num"),
            )
            step = arguments.get("step_num") or 0
            decision = await asyncio.to_thread(self._exchange_slot_view, slot, step, view)
        if slot == "check_permission" and decision is not None and decision.is_block():
            decision = replace(decision, permission="deny")
        return decision

    # ── lifecycle: session open ──────────────────────────────────
    @staticmethod
    def _server_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
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
        env = dict(os.environ if base_env is None else base_env)
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        parts = [pkg_parent] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def pre_loop(self, state: Any, session_log: Any, context: Any) -> None:
        proc = subprocess.Popen(  # noqa: S603 - argv is operator-supplied
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=self._server_env(self._env),
            start_new_session=os.name == "posix",
        )
        self._proc = proc
        self._closing = False
        if proc.stdout is not None and callable(getattr(proc.stdout, "readline", None)):
            stdout_pipe = proc.stdout
            max_chars = self._max_message_bytes + 1
            max_queued_lines = 256
            with self._stdout_cond:
                self._stdout_lines.clear()
                self._stdout_eof = False
                self._stdout_error = None

            def _drain_lep_stdout() -> None:
                try:
                    while True:
                        raw_line = stdout_pipe.readline(max_chars)
                        with self._stdout_cond:
                            if not raw_line:
                                self._stdout_eof = True
                                self._stdout_cond.notify_all()
                                break
                            while len(self._stdout_lines) >= max_queued_lines and not self._closing:
                                self._stdout_cond.wait()
                            if self._closing:
                                self._stdout_eof = True
                                self._stdout_cond.notify_all()
                                break
                            self._stdout_lines.append(raw_line)
                            self._stdout_cond.notify_all()
                except BaseException as exc:  # noqa: BLE001
                    with self._stdout_cond:
                        self._stdout_error = exc
                        self._stdout_eof = True
                        self._stdout_cond.notify_all()

            reader = threading.Thread(
                target=_drain_lep_stdout, daemon=True, name="looplet-lep-stdout"
            )
            self._stdout_reader = reader
            reader.start()
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
        self._closing = True
        locked = self._rpc_lock.acquire(blocking=False)
        try:
            if locked:
                try:
                    self._rpc_locked(
                        "loop/shutdown",
                        {"run_id": self._run_id, "outcome": "ok"},
                        expect_response=False,
                    )
                except Exception:  # pragma: no cover - best effort
                    pass
        finally:
            if locked:
                self._rpc_lock.release()
        with self._stdout_cond:
            self._stdout_eof = True
            self._stdout_cond.notify_all()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:  # pragma: no cover - best effort
            pass
        if not locked:
            try:
                proc.terminate()
            except Exception:  # pragma: no cover - best effort
                pass
        wait_limit = (
            0.2 if not locked else (min(5.0, max(0.1, self._timeout)) if self._timeout else 1.0)
        )
        try:
            proc.wait(timeout=wait_limit)
        except subprocess.TimeoutExpired:  # pragma: no cover - best effort
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
        except Exception:  # pragma: no cover - best effort
            pass
        finally:
            self._proc = None
            reader = self._stdout_reader
            self._stdout_reader = None
            if reader is not None and reader is not threading.current_thread():
                reader.join(timeout=1.0)
            with self._stdout_cond:
                self._stdout_lines.clear()
                self._stdout_eof = True
                self._stdout_error = None
                self._stdout_cond.notify_all()


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
