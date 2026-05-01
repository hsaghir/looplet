"""Tests for first-class ToolError taxonomy.

``ToolResult.error`` widens from ``str | None`` to
``ToolError | str | None``. String values keep working (back-compat)
but new code emits a typed ``ToolError(kind, message, retriable, context)``
so the loop and downstream consumers can distinguish permission
denials, timeouts, validation failures, etc.

Helper properties on ``ToolResult`` (``error_message``, ``error_kind``,
``error_retriable``) give a single access shape regardless of whether
the error is the legacy string or the new structured type.
"""

from __future__ import annotations

from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import ErrorKind, ToolCall, ToolError, ToolResult


class TestToolErrorType:
    def test_kinds_are_string_enum(self):
        assert ErrorKind.PERMISSION_DENIED.value == "permission_denied"
        assert ErrorKind.TIMEOUT.value == "timeout"
        assert ErrorKind.VALIDATION.value == "validation"
        assert ErrorKind.EXECUTION.value == "execution"
        assert ErrorKind.CANCELLED.value == "cancelled"

    def test_tool_error_construction(self):
        err = ToolError(kind=ErrorKind.TIMEOUT, message="took too long", retriable=True)
        assert err.kind == ErrorKind.TIMEOUT
        assert err.message == "took too long"
        assert err.retriable is True
        assert err.context == {}

    def test_bool_truthiness(self):
        err = ToolError(kind=ErrorKind.EXECUTION, message="x", retriable=False)
        assert bool(err) is True


class TestToolResultAccessors:
    def test_string_error_back_compat(self):
        r = ToolResult(tool="t", args_summary="", data=None, error="boom")
        assert bool(r.error) is True
        assert r.error_message == "boom"
        assert r.error_kind == ErrorKind.EXECUTION
        assert r.error_retriable is False

    def test_structured_error(self):
        te = ToolError(kind=ErrorKind.PERMISSION_DENIED, message="denied", retriable=False)
        r = ToolResult(
            tool="t",
            args_summary="",
            data=None,
            error=te.message,
            error_detail=te,
        )
        assert r.error_message == "denied"
        assert r.error_kind == ErrorKind.PERMISSION_DENIED
        assert r.error_retriable is False

    def test_no_error(self):
        r = ToolResult(tool="t", args_summary="", data={"x": 1})
        assert r.error_message is None
        assert r.error_kind is None
        assert r.error_retriable is False

    def test_to_dict_serialises_structured_error(self):
        te = ToolError(kind=ErrorKind.TIMEOUT, message="slow", retriable=True)
        r = ToolResult(
            tool="t",
            args_summary="",
            data=None,
            error=te.message,
            error_detail=te,
        )
        d = r.to_dict()
        assert d["error"] == "slow"
        assert d["error_kind"] == "timeout"
        assert d["error_retriable"] is True


class TestDispatchMapsExceptions:
    def test_timeout_error_maps_to_timeout_kind(self):
        def _slow(**kw):
            raise TimeoutError("deadline")

        reg = BaseToolRegistry()
        reg.register(ToolSpec(name="t", description="", parameters={}, execute=_slow))
        res = reg.dispatch(ToolCall(tool="t", args={}))
        assert res.error is not None  # plain string
        assert res.error_detail is not None
        assert res.error_detail.kind == ErrorKind.TIMEOUT
        assert res.error_detail.retriable is True

    def test_value_error_maps_to_validation(self):
        def _bad(**kw):
            raise ValueError("bad arg")

        reg = BaseToolRegistry()
        reg.register(ToolSpec(name="t", description="", parameters={}, execute=_bad))
        res = reg.dispatch(ToolCall(tool="t", args={}))
        assert res.error is not None
        assert res.error_detail is not None
        assert res.error_detail.kind == ErrorKind.VALIDATION
        assert res.error_detail.retriable is False

    def test_generic_exception_maps_to_execution(self):
        def _boom(**kw):
            raise RuntimeError("oops")

        reg = BaseToolRegistry()
        reg.register(ToolSpec(name="t", description="", parameters={}, execute=_boom))
        res = reg.dispatch(ToolCall(tool="t", args={}))
        assert res.error_detail is not None
        assert res.error_detail.kind == ErrorKind.EXECUTION

    def test_unknown_tool_maps_to_validation(self):
        reg = BaseToolRegistry()
        res = reg.dispatch(ToolCall(tool="nope", args={}))
        assert res.error_detail is not None
        assert res.error_detail.kind == ErrorKind.VALIDATION


# ── recovery_hint surfaces actionable structure on dispatcher errors ──


def test_unknown_tool_error_carries_recovery_hint() -> None:
    """``ToolError.recovery_hint`` carries a 'did you mean?' suggestion
    for typos. Information-additive: smarter models can self-correct
    from the structured hint without needing to re-discover the
    catalog from prose."""
    from looplet import ToolSpec
    from looplet.tools import BaseToolRegistry
    from looplet.types import ToolCall

    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="search", description="d", parameters={"q": "query"}, execute=lambda *, q: {"q": q}
        )
    )
    result = reg.dispatch(ToolCall(tool="serach", args={"q": "x"}))
    assert result.error_detail is not None
    hint = result.error_detail.recovery_hint
    assert hint is not None
    assert "search" in str(hint)


def test_missing_arg_error_carries_recovery_hint() -> None:
    """Missing required arg → ``recovery_hint`` carries the structured
    ``{missing, provided, expected}`` so the model knows exactly what
    to add on the next call."""
    from looplet import ToolSpec
    from looplet.tools import BaseToolRegistry
    from looplet.types import ToolCall

    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="rank",
            description="d",
            parameters={"column": "col", "choices": "list"},
            execute=lambda *, column, choices: {},
        )
    )
    result = reg.dispatch(ToolCall(tool="rank", args={"column": "x"}))
    assert result.error_detail is not None
    hint = result.error_detail.recovery_hint
    assert isinstance(hint, dict)
    assert "missing" in hint and "choices" in hint["missing"]
    assert "provided" in hint and "column" in hint["provided"]
    assert "expected" in hint and "choices" in hint["expected"]


def test_unexpected_arg_error_carries_recovery_hint() -> None:
    """Unexpected arg → ``recovery_hint`` carries did_you_mean +
    expected so the model can fix the typo or drop the bogus arg."""
    from looplet import ToolSpec
    from looplet.tools import BaseToolRegistry
    from looplet.types import ToolCall

    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="rank", description="d", parameters={"column": "col"}, execute=lambda *, column: {}
        )
    )
    result = reg.dispatch(ToolCall(tool="rank", args={"colmun": "x"}))
    assert result.error_detail is not None
    hint = result.error_detail.recovery_hint
    assert isinstance(hint, dict)
    assert hint["did_you_mean"] == "column"
    assert "colmun" in hint["unexpected"]
    assert "expected" in hint
