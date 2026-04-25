"""Tool reliability features — timeout enforcement & large-output persistence.

Tests for the platform-level reliability improvements inspired by
Claude Code's tool implementation patterns:
- ToolSpec.timeout_s: framework-enforced execution deadline
- truncate_tool_result persist_dir: large outputs saved to disk
"""

from __future__ import annotations

import os
import tempfile
import time

from looplet import BaseToolRegistry, ErrorKind, ToolCall, ToolSpec
from looplet.scaffolding import truncate_tool_result

# ── ToolSpec.timeout_s ───────────────────────────────────────────


class TestToolTimeout:
    """Framework-level timeout enforcement on ToolSpec."""

    def test_timeout_field_default_none(self) -> None:
        spec = ToolSpec(name="t", description="x", parameters={}, execute=lambda: None)
        assert spec.timeout_s is None

    def test_timeout_field_set(self) -> None:
        spec = ToolSpec(
            name="t", description="x", parameters={}, execute=lambda: None, timeout_s=5.0
        )
        assert spec.timeout_s == 5.0

    def test_slow_tool_times_out(self) -> None:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="slow",
                description="x",
                parameters={"x": "str"},
                execute=lambda x="": time.sleep(10),
                timeout_s=0.1,
            )
        )
        r = reg.dispatch(ToolCall(tool="slow", args={"x": "hi"}))
        assert r.error is not None
        assert r.error_kind == ErrorKind.TIMEOUT
        assert r.error_detail is not None
        assert r.error_detail.retriable is True

    def test_fast_tool_no_timeout(self) -> None:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="fast",
                description="x",
                parameters={"x": "str"},
                execute=lambda x="": {"ok": True},
                timeout_s=10.0,
            )
        )
        r = reg.dispatch(ToolCall(tool="fast", args={"x": "hi"}))
        assert r.error is None
        assert r.data == {"ok": True}

    def test_no_timeout_spec(self) -> None:
        """timeout_s=None means no framework enforcement."""
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="normal",
                description="x",
                parameters={"x": "str"},
                execute=lambda x="": {"ok": True},
            )
        )
        r = reg.dispatch(ToolCall(tool="normal", args={"x": "hi"}))
        assert r.error is None
        assert r.data == {"ok": True}

    def test_timeout_error_message_contains_timeout(self) -> None:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="slow2",
                description="x",
                parameters={},
                execute=lambda: time.sleep(10),
                timeout_s=0.05,
            )
        )
        r = reg.dispatch(ToolCall(tool="slow2", args={}))
        assert r.error is not None
        assert "Timeout" in r.error or "timeout" in r.error

    def test_timeout_tool_exception_propagated(self) -> None:
        """Non-timeout exception in a timeout-wrapped tool still propagates normally."""

        def bad_tool(x: str = "") -> dict:
            raise ValueError("bad input")

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="bad",
                description="x",
                parameters={"x": "str"},
                execute=bad_tool,
                timeout_s=10.0,
            )
        )
        r = reg.dispatch(ToolCall(tool="bad", args={"x": "hi"}))
        assert r.error is not None
        assert r.error_kind == ErrorKind.VALIDATION  # ValueError -> VALIDATION
        assert "bad input" in r.error

    def test_async_tool_respects_timeout(self) -> None:
        """Async tools must also be subject to timeout_s."""
        import asyncio

        async def slow_async(x: str = "") -> dict:
            await asyncio.sleep(10)
            return {"ok": True}

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="async_slow",
                description="x",
                parameters={"x": "str"},
                execute=slow_async,
                timeout_s=0.1,
            )
        )
        t0 = time.time()
        r = reg.dispatch(ToolCall(tool="async_slow", args={"x": "hi"}))
        elapsed = time.time() - t0
        assert r.error is not None
        assert r.error_kind == ErrorKind.TIMEOUT
        assert elapsed < 2.0  # must return promptly, not wait 10s


# ── Persist large outputs to file ────────────────────────────────


class TestPersistLargeOutput:
    """truncate_tool_result with persist_dir for very large outputs."""

    def test_large_output_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = "x" * 20000
            r = truncate_tool_result(big, persist_dir=tmp, persist_threshold=5000)
            assert isinstance(r, dict)
            assert "persisted_output_path" in r
            assert os.path.exists(r["persisted_output_path"])
            assert r["persisted_output_size"] == 20000
            # File contains full content
            content = open(r["persisted_output_path"]).read()
            assert len(content) == 20000

    def test_small_output_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            small = "hello"
            r = truncate_tool_result(small, persist_dir=tmp, persist_threshold=5000)
            assert r == "hello"
            # No files written
            assert len(os.listdir(tmp)) == 0

    def test_dict_output_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big_dict = {"stdout": "x" * 20000, "exit_code": 0}
            r = truncate_tool_result(big_dict, persist_dir=tmp, persist_threshold=5000)
            assert "persisted_output_path" in r
            assert r["persisted_output_size"] > 20000

    def test_persist_dir_none_backward_compat(self) -> None:
        """No persist_dir means old behavior."""
        big = "x" * 10000
        r = truncate_tool_result(big)
        assert isinstance(r, str)
        assert "[truncated" in r
        assert len(r) < 10000

    def test_persist_threshold_zero_no_persist(self) -> None:
        """persist_threshold=0 means no persist."""
        with tempfile.TemporaryDirectory() as tmp:
            big = "x" * 10000
            r = truncate_tool_result(big, persist_dir=tmp, persist_threshold=0)
            assert isinstance(r, str)  # normal truncation
            assert len(os.listdir(tmp)) == 0

    def test_persisted_output_contains_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = truncate_tool_result("x" * 20000, persist_dir=tmp, persist_threshold=5000)
            assert "note" in r
            assert "read_file" in r["note"] or "bash" in r["note"]

    def test_truncated_output_in_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = truncate_tool_result("x" * 20000, persist_dir=tmp, persist_threshold=5000)
            assert "truncated_output" in r
            assert len(r["truncated_output"]) <= 7000  # max_chars + suffix

    def test_persist_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "sub", "dir")
            r = truncate_tool_result("x" * 20000, persist_dir=nested, persist_threshold=5000)
            assert os.path.exists(r["persisted_output_path"])
