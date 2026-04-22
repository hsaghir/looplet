"""Tests for ToolContext — typed per-call context passed to tool execute.

ToolContext carries cwd, cancellation, progress callback, and arbitrary
metadata. It is opt-in: tools that accept a ``ctx`` kwarg receive it;
tools without ``ctx`` in their signature continue to work unchanged.
"""

from __future__ import annotations

import pytest

from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import CancelToken, ToolCall, ToolContext


class TestCancelToken:
    def test_starts_not_cancelled(self):
        tok = CancelToken()
        assert tok.is_cancelled is False

    def test_cancel_sets_flag(self):
        tok = CancelToken()
        tok.cancel()
        assert tok.is_cancelled is True

    def test_raise_if_cancelled(self):
        tok = CancelToken()
        tok.raise_if_cancelled()  # no-op
        tok.cancel()
        with pytest.raises(RuntimeError, match="cancelled"):
            tok.raise_if_cancelled()


class TestToolContext:
    def test_default_construction(self):
        ctx = ToolContext()
        assert ctx.cwd is None
        assert ctx.workspace_root is None
        assert ctx.cancel_token is None
        assert ctx.on_progress is None
        assert ctx.metadata == {}

    def test_full_construction(self):
        progress_calls: list[tuple[str, dict]] = []
        tok = CancelToken()
        ctx = ToolContext(
            cwd="/tmp",
            workspace_root="/home/ws",
            cancel_token=tok,
            on_progress=lambda stage, data: progress_calls.append((stage, data)),
            session_id="sess-1",
            metadata={"user": "x"},
        )
        assert ctx.cwd == "/tmp"
        assert ctx.cancel_token is tok
        ctx.report_progress("downloading", {"bytes": 100})
        assert progress_calls == [("downloading", {"bytes": 100})]

    def test_report_progress_with_no_callback_is_silent(self):
        ctx = ToolContext()
        ctx.report_progress("anything")  # must not raise


class TestDispatchPassesToolContext:
    def test_tool_without_ctx_param_works_unchanged(self):
        """Existing tools without ctx in signature must keep working."""
        registry = BaseToolRegistry()
        registry.register(
            ToolSpec(
                name="echo",
                description="echo",
                parameters={"x": "input"},
                execute=lambda x: {"got": x},
            )
        )
        result = registry.dispatch(ToolCall(tool="echo", args={"x": "hi"}))
        assert result.error is None
        assert result.data == {"got": "hi"}

    def test_tool_with_ctx_param_receives_context(self):
        """Tools that declare ctx in their signature receive it automatically."""
        seen_ctx: list[ToolContext | None] = []

        def run(x: str, ctx: ToolContext | None = None) -> dict:
            seen_ctx.append(ctx)
            return {"x": x, "cwd": ctx.cwd if ctx else None}

        registry = BaseToolRegistry()
        registry.register(
            ToolSpec(
                name="peek",
                description="peek",
                parameters={"x": "input"},
                execute=run,
            )
        )
        ctx = ToolContext(cwd="/tmp/work")
        result = registry.dispatch(ToolCall(tool="peek", args={"x": "hi"}), ctx=ctx)
        assert result.error is None
        assert result.data == {"x": "hi", "cwd": "/tmp/work"}
        assert seen_ctx == [ctx]

    def test_tool_with_ctx_but_no_context_supplied_gets_none(self):
        """Tool may accept optional ctx; dispatch() without ctx passes None."""
        seen_ctx: list[ToolContext | None] = []

        def run(x: str, ctx: ToolContext | None = None) -> dict:
            seen_ctx.append(ctx)
            return {}

        registry = BaseToolRegistry()
        registry.register(
            ToolSpec(
                name="peek",
                description="peek",
                parameters={"x": "input"},
                execute=run,
            )
        )
        registry.dispatch(ToolCall(tool="peek", args={"x": "hi"}))
        assert seen_ctx == [None]

    def test_dispatch_batch_forwards_ctx(self):
        calls_seen: list[ToolContext | None] = []

        def run(x: str, ctx: ToolContext | None = None):
            calls_seen.append(ctx)
            return x

        registry = BaseToolRegistry()
        registry.register(
            ToolSpec(
                name="t",
                description="",
                parameters={"x": "x"},
                execute=run,
                concurrent_safe=True,
            )
        )
        ctx = ToolContext(cwd="/ws")
        registry.dispatch_batch(
            [ToolCall(tool="t", args={"x": "a"}), ToolCall(tool="t", args={"x": "b"})],
            ctx=ctx,
        )
        assert calls_seen == [ctx, ctx]

    def test_cancelled_token_aborts_before_dispatch(self):
        """When ctx.cancel_token is cancelled, dispatch returns a cancellation error
        instead of invoking the tool."""
        ran = []
        registry = BaseToolRegistry()
        registry.register(
            ToolSpec(
                name="slow",
                description="",
                parameters={},
                execute=lambda: ran.append(1) or {"ok": True},
            )
        )
        tok = CancelToken()
        tok.cancel()
        ctx = ToolContext(cancel_token=tok)
        result = registry.dispatch(ToolCall(tool="slow", args={}), ctx=ctx)
        assert ran == []
        assert result.error is not None
        assert "cancel" in (result.error_message or "").lower()
