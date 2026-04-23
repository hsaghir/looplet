"""Tool warnings — soft advisories attached to successful tool results.

Lets tool authors say "here's your data, AND you should know X" without
forcing a failure. Complements ToolValidationError (hard stop) and the
existing error/error_detail (failure). All three coexist: a tool can
emit warnings, then raise ToolValidationError, and the resulting
ToolResult carries both the warnings and the structured error.
"""

from __future__ import annotations

from looplet import (
    BaseToolRegistry,
    ErrorKind,
    ToolCall,
    ToolContext,
    ToolSpec,
    ToolValidationError,
)


def _noctx_tool(execute):
    """Build a ToolSpec whose execute takes only (ctx) — convenience."""
    return ToolSpec(
        name="t",
        description="test tool",
        parameters={},
        execute=execute,
    )


# ── Basic happy path ─────────────────────────────────────────────


def test_tool_can_attach_warning_to_success_result() -> None:
    """ctx.warn(msg) surfaces on ToolResult.warnings with data intact."""

    def tool(ctx: ToolContext):
        ctx.warn("low-confidence pick")
        return {"picked": "ts"}

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    ctx = ToolContext()

    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ctx)
    assert r.error is None
    assert r.data == {"picked": "ts"}
    assert r.warnings == ["low-confidence pick"]


def test_multiple_warnings_preserved_in_order() -> None:
    def tool(ctx: ToolContext):
        ctx.warn("first")
        ctx.warn("second")
        ctx.warn("third")
        return 42

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    assert r.warnings == ["first", "second", "third"]


def test_tool_without_warn_has_empty_warnings() -> None:
    """Backward compat: tools that don't use ctx.warn still work."""

    def tool(ctx: ToolContext):
        return "ok"

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    assert r.warnings == []


def test_no_ctx_supplied_still_works() -> None:
    """Tool that doesn't accept ctx still dispatches cleanly."""
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="t",
            description="",
            parameters={},
            execute=lambda: "ok",
        )
    )
    r = reg.dispatch(ToolCall(tool="t", args={}))
    assert r.error is None
    assert r.warnings == []


# ── Per-call scoping on a shared ctx ─────────────────────────────


def test_warnings_scoped_per_call_not_cumulative() -> None:
    """A ctx shared across multiple dispatches must not bleed warnings
    from call N into ToolResult N+1."""

    def tool(ctx: ToolContext):
        ctx.warn("this call")
        return None

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    ctx = ToolContext()

    r1 = reg.dispatch(ToolCall(tool="t", args={}), ctx=ctx)
    r2 = reg.dispatch(ToolCall(tool="t", args={}), ctx=ctx)

    assert r1.warnings == ["this call"]
    assert r2.warnings == ["this call"]  # only warnings from r2's own call
    # ctx as an observer accumulates full history — useful for audit.
    assert ctx.warnings == ["this call", "this call"]


# ── Warnings survive failures ────────────────────────────────────


def test_warnings_preserved_when_tool_raises() -> None:
    """Warnings emitted before a raise appear alongside the error —
    they usually explain *why* the tool then gave up."""

    def tool(ctx: ToolContext):
        ctx.warn("tried fallback A, failed")
        ctx.warn("tried fallback B, failed")
        raise RuntimeError("all fallbacks exhausted")

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    assert r.error is not None
    assert r.error_kind == ErrorKind.EXECUTION
    assert r.warnings == ["tried fallback A, failed", "tried fallback B, failed"]


def test_warnings_preserved_with_tool_validation_error() -> None:
    """Warnings + ToolValidationError compose cleanly."""

    def tool(ctx: ToolContext):
        ctx.warn("input had suspicious shape")
        raise ToolValidationError("x not found")

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    assert r.error_kind == ErrorKind.VALIDATION
    assert r.warnings == ["input had suspicious shape"]
    assert r.error == "x not found"


# ── to_dict includes warnings (LLM context) ──────────────────────


def test_to_dict_includes_warnings_when_present() -> None:
    def tool(ctx: ToolContext):
        ctx.warn("truncated to 20 of 3345 items")
        return [{"i": i} for i in range(3345)]

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    d = r.to_dict()
    assert "warnings" in d
    assert d["warnings"] == ["truncated to 20 of 3345 items"]
    assert d["total_items"] == 3345  # existing truncation still works


def test_to_dict_omits_warnings_when_empty() -> None:
    """Keep to_dict compact when nothing was emitted."""
    reg = BaseToolRegistry()
    reg.register(_noctx_tool(lambda ctx: "ok"))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    assert "warnings" not in r.to_dict()


# ── Empty / falsy warnings are ignored ───────────────────────────


def test_empty_warn_message_is_ignored() -> None:
    """ctx.warn('') is a no-op so accidental empty strings don't
    pollute the result."""

    def tool(ctx: ToolContext):
        ctx.warn("")
        ctx.warn("real")
        return None

    reg = BaseToolRegistry()
    reg.register(_noctx_tool(tool))
    r = reg.dispatch(ToolCall(tool="t", args={}), ctx=ToolContext())
    assert r.warnings == ["real"]
