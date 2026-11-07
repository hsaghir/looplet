"""Process-group-owning subprocess execution helper for tools.

Unlike ``subprocess.run(..., timeout=...)`` or thread-abandoned timeouts,
:func:`run_subprocess_tool` starts each child in a dedicated OS session /
process group, polls ``CancelToken`` and ``timeout_s``, terminates the
entire process tree (including grandchildren holding stdio pipes) on
cancellation or timeout, and optionally spills oversized output to disk.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from looplet.types import CancelToken, ToolCancelledError, ToolContext

__all__ = ["run_subprocess_tool", "terminate_process_tree"]


def terminate_process_tree(proc: subprocess.Popen[Any], *, wait_timeout: float = 0.5) -> None:
    """Terminate ``proc`` and remaining members of its owned process group."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            proc.wait(timeout=wait_timeout)
            return
        try:
            proc.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=wait_timeout)
        return

    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc.terminate()
        proc.wait(timeout=wait_timeout)
        return
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        proc.kill()
        proc.wait(timeout=wait_timeout)
    except (OSError, subprocess.SubprocessError):
        pass


def run_subprocess_tool(
    command: str | Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 30.0,
    cancel_token: CancelToken | None = None,
    ctx: ToolContext | None = None,
    max_output_chars: int = 50_000,
    spill_dir: str | Path | None = None,
    raise_on_timeout: bool = False,
) -> dict[str, Any]:
    """Execute a subprocess in its own process group with cancellation and output bounds."""
    token = (
        cancel_token
        if cancel_token is not None
        else (ctx.cancel_token if ctx is not None else None)
    )
    if token is not None and token.is_cancelled:
        raise ToolCancelledError("Tool execution cancelled")

    effective_cwd = (
        str(cwd) if cwd is not None else (ctx.cwd if ctx is not None and ctx.cwd else None)
    )
    process_options: dict[str, Any] = {}
    if os.name == "posix":
        process_options["start_new_session"] = True
    elif os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        command,
        shell=isinstance(command, str),
        cwd=effective_cwd,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **process_options,
    )

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def _read_pipe(pipe: Any, sink: list[bytes]) -> None:
        if pipe is None:
            return
        try:
            while True:
                chunk = pipe.read(16384)
                if not chunk:
                    break
                sink.append(chunk)
        except OSError:
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    out_thread = threading.Thread(target=_read_pipe, args=(proc.stdout, stdout_chunks), daemon=True)
    err_thread = threading.Thread(target=_read_pipe, args=(proc.stderr, stderr_chunks), daemon=True)
    out_thread.start()
    err_thread.start()

    deadline = (time.monotonic() + timeout_s) if timeout_s > 0 else None
    timed_out = False
    try:
        while proc.poll() is None or out_thread.is_alive() or err_thread.is_alive():
            if token is not None and token.is_cancelled:
                raise ToolCancelledError("Tool execution cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                terminate_process_tree(proc, wait_timeout=0.2)
                break
            time.sleep(0.01)
    except BaseException:
        terminate_process_tree(proc, wait_timeout=0.2)
        out_thread.join(timeout=0.5)
        err_thread.join(timeout=0.5)
        raise

    out_thread.join(timeout=0.5)
    err_thread.join(timeout=0.5)

    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    if timed_out:
        if raise_on_timeout:
            raise TimeoutError(f"Command timed out after {timeout_s:g}s")
        return {
            "stdout": stdout_text[:max_output_chars],
            "stderr": stderr_text[:max_output_chars]
            + (
                f"\nCommand timed out after {timeout_s:g}s"
                if stderr_text
                else f"Command timed out after {timeout_s:g}s"
            ),
            "returncode": -1,
            "timed_out": True,
            "process_group_terminated": True,
        }

    result: dict[str, Any] = {
        "stdout": stdout_text,
        "stderr": stderr_text[:max_output_chars],
        "returncode": proc.returncode if proc.returncode is not None else 0,
    }
    if len(stdout_text) > max_output_chars:
        if spill_dir is not None:
            s_dir = Path(spill_dir)
            s_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest = hashlib.sha256(stdout_text.encode("utf-8")).hexdigest()[:12]
            spill_path = s_dir / f"subprocess-stdout-{digest}.txt"
            spill_path.write_text(stdout_text, encoding="utf-8")
            result["persisted_output_path"] = str(spill_path)
            result["persisted_output_chars"] = len(stdout_text)
        result["stdout"] = (
            stdout_text[:max_output_chars] + f"\n... [truncated; {len(stdout_text)} chars total]"
        )
        result["stdout_truncated"] = True
    return result
