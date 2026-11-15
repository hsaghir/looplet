"""Process-group-owning subprocess execution helper for tools.

Unlike ``subprocess.run(..., timeout=...)`` or thread-abandoned timeouts,
:func:`run_subprocess_tool` starts each child in a dedicated OS session /
process group, polls ``CancelToken`` and ``timeout_s``, terminates the
entire process tree (including grandchildren holding stdio pipes) on
cancellation or timeout, and optionally spills oversized output to disk.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from looplet.types import CancelToken, ToolCancelledError, ToolContext

__all__ = ["run_subprocess_tool", "terminate_process_tree"]


class _BoundedStreamSink:
    """Incremental UTF-8 pipe consumer with O(max_chars) RAM and streaming spill."""

    def __init__(
        self,
        *,
        max_chars: int,
        spill_dir: Path | None,
        stream_name: str,
    ) -> None:
        self.max_chars = max(0, max_chars)
        self.spill_dir = spill_dir
        self.stream_name = stream_name
        self._lock = threading.Lock()
        self._head_parts: list[str] = []
        self._head_chars = 0
        self._total_chars = 0
        self._spill_handle: Any = None
        self._spill_tmp_path: Path | None = None
        self._hasher: Any = None
        self._persisted_path: Path | None = None

    @property
    def total_chars(self) -> int:
        with self._lock:
            return self._total_chars

    @property
    def persisted_path(self) -> Path | None:
        with self._lock:
            return self._persisted_path

    def _feed_text(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            prev_total = self._total_chars
            self._total_chars += len(text)
            cur_total = self._total_chars
            if self._head_chars < self.max_chars:
                need = self.max_chars - self._head_chars
                take = text[:need]
                self._head_parts.append(take)
                self._head_chars += len(take)
            prefix_snapshot = (
                "".join(self._head_parts)
                if self.spill_dir is not None
                and cur_total > self.max_chars
                and self._spill_handle is None
                else ""
            )
        if self.spill_dir is not None and cur_total > self.max_chars:
            if self._spill_handle is None:
                self.spill_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                fd, tmp_name = tempfile.mkstemp(
                    dir=self.spill_dir,
                    prefix=f".subprocess-{self.stream_name}-",
                    suffix=".tmp",
                )
                self._spill_tmp_path = Path(tmp_name)
                self._spill_handle = os.fdopen(fd, "w", encoding="utf-8")
                self._hasher = hashlib.sha256()
                # Write prior prefix plus any overflow beyond what was already in head_parts
                prior_head_len = min(prev_total, self.max_chars)
                initial_str = prefix_snapshot[:prior_head_len] + text
                self._spill_handle.write(initial_str)
                self._hasher.update(initial_str.encode("utf-8"))
            else:
                self._spill_handle.write(text)
                self._hasher.update(text.encode("utf-8"))

    def consume_pipe(self, pipe: Any) -> None:
        if pipe is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = pipe.read(16384)
                if not chunk:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        self._feed_text(tail)
                    break
                decoded = decoder.decode(chunk, final=False)
                if decoded:
                    self._feed_text(decoded)
        except OSError:
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass
            self._finalize_spill()

    def _finalize_spill(self) -> None:
        if self._spill_handle is None:
            return
        try:
            self._spill_handle.flush()
            self._spill_handle.close()
            if (
                self._spill_tmp_path is not None
                and self._hasher is not None
                and self.spill_dir is not None
            ):
                digest = self._hasher.hexdigest()[:12]
                final_path = self.spill_dir / f"subprocess-{self.stream_name}-{digest}.txt"
                os.replace(self._spill_tmp_path, final_path)
                with self._lock:
                    self._persisted_path = final_path
                self._spill_tmp_path = None
        except OSError:
            if self._spill_tmp_path is not None:
                self._spill_tmp_path.unlink(missing_ok=True)
        finally:
            self._spill_handle = None

    def head_text(self) -> str:
        with self._lock:
            parts = list(self._head_parts)
        return "".join(parts)

    @property
    def truncated(self) -> bool:
        with self._lock:
            return self._total_chars > self.max_chars


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

    resolved_spill = Path(spill_dir) if spill_dir is not None else None
    out_sink = _BoundedStreamSink(
        max_chars=max_output_chars, spill_dir=resolved_spill, stream_name="stdout"
    )
    err_sink = _BoundedStreamSink(
        max_chars=max_output_chars, spill_dir=resolved_spill, stream_name="stderr"
    )

    out_thread = threading.Thread(target=out_sink.consume_pipe, args=(proc.stdout,), daemon=True)
    err_thread = threading.Thread(target=err_sink.consume_pipe, args=(proc.stderr,), daemon=True)
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

    stdout_head = out_sink.head_text()
    stderr_head = err_sink.head_text()

    if timed_out:
        if raise_on_timeout:
            raise TimeoutError(f"Command timed out after {timeout_s:g}s")
        timeout_res: dict[str, Any] = {
            "stdout": stdout_head,
            "stderr": stderr_head
            + (
                f"\nCommand timed out after {timeout_s:g}s"
                if stderr_head
                else f"Command timed out after {timeout_s:g}s"
            ),
            "returncode": -1,
            "timed_out": True,
            "process_group_terminated": True,
            "effect_unknown": True,
        }
        if out_sink.truncated:
            timeout_res["stdout_truncated"] = True
            if out_sink.persisted_path is not None:
                timeout_res["persisted_output_path"] = str(out_sink.persisted_path)
                timeout_res["persisted_output_chars"] = out_sink.total_chars
        if err_sink.truncated:
            timeout_res["stderr_truncated"] = True
            timeout_res["stderr_total_chars"] = err_sink.total_chars
            if err_sink.persisted_path is not None:
                timeout_res["persisted_stderr_path"] = str(err_sink.persisted_path)
                timeout_res["persisted_stderr_chars"] = err_sink.total_chars
        return timeout_res

    result: dict[str, Any] = {
        "stdout": stdout_head,
        "stderr": stderr_head,
        "returncode": proc.returncode if proc.returncode is not None else 0,
    }
    if out_sink.truncated:
        if out_sink.persisted_path is not None:
            result["persisted_output_path"] = str(out_sink.persisted_path)
            result["persisted_output_chars"] = out_sink.total_chars
        result["stdout"] = stdout_head + f"\n... [truncated; {out_sink.total_chars} chars total]"
        result["stdout_truncated"] = True
    if err_sink.truncated:
        if err_sink.persisted_path is not None:
            result["persisted_stderr_path"] = str(err_sink.persisted_path)
            result["persisted_stderr_chars"] = err_sink.total_chars
        result["stderr"] = stderr_head + f"\n... [truncated; {err_sink.total_chars} chars total]"
        result["stderr_truncated"] = True
        result["stderr_total_chars"] = err_sink.total_chars
    return result
