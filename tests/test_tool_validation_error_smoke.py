"""Tool-author-friendly validation errors and did-you-mean diagnostics.

These are generalizable ergonomics patterns any agent framework benefits
from: when a tool call fails because of a caller/input mistake (typo'd
tool name, typo'd arg, field-not-found inside the tool body), the
result should carry a clean VALIDATION error with a suggestion - not an
opaque TypeError, a silent empty list, or a pseudo-error sentinel mixed
into the tool's data payload.
"""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    ErrorKind,
    ToolCall,
    ToolSpec,
    ToolValidationError,
    suggest_similar,
)

# ── suggest_similar() helper ─────────────────────────────────────


class TestSuggestSimilar:
    def test_close_match_suggested(self) -> None:
        assert suggest_similar("scann", ["scan", "rank", "pivot"]) == "scan"

    def test_far_match_returns_none(self) -> None:
        assert suggest_similar("xyz", ["scan", "rank", "pivot"]) is None

    def test_empty_inputs_return_none(self) -> None:
        assert suggest_similar("", ["scan"]) is None
        assert suggest_similar("scan", []) is None

    def test_cutoff_threshold_respected(self) -> None:
        # 'scanx' is closeish to 'scan' - strict cutoff rejects it.
        assert suggest_similar("zzzzscan", ["scan"], cutoff=0.9) is None


# ── Unknown tool diagnostics ─────────────────────────────────────


def _registry_with_tools(names: list[str]) -> BaseToolRegistry:
    reg = BaseToolRegistry()
    for n in names:
        reg.register(
            ToolSpec(
                name=n,
                description=f"tool {n}",
                parameters={"x": "str"},
                execute=lambda x="": {"ok": True, "x": x},
            )
        )
    return reg


class TestUnknownToolDidYouMean:
    def test_close_typo_gets_suggestion(self) -> None:
        reg = _registry_with_tools(["scan", "rank", "pivot"])
        result = reg.dispatch(ToolCall(tool="scann", args={"x": "a"}))
        assert result.error is not None
        assert result.error_kind == ErrorKind.VALIDATION
        assert "Did you mean 'scan'" in result.error

    def test_far_typo_no_suggestion_but_lists_available(self) -> None:
        reg = _registry_with_tools(["scan", "rank"])
        result = reg.dispatch(ToolCall(tool="xyz", args={"x": "a"}))
        assert result.error is not None
        assert "Did you mean" not in result.error
        assert "scan" in result.error and "rank" in result.error


# ── Unknown / typo'd argument diagnostics ────────────────────────


class TestUnknownArgumentDidYouMean:
    def test_typo_in_arg_name_gets_suggestion(self) -> None:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="read",
                description="read a file",
                parameters={"file_path": "str"},
                execute=lambda file_path="": {"ok": True, "file_path": file_path},
            )
        )
        # LLM typo: `file_pth` instead of `file_path`.
        result = reg.dispatch(ToolCall(tool="read", args={"file_pth": "/etc/hosts"}))
        assert result.error_kind == ErrorKind.VALIDATION
        assert "file_pth" in result.error
        assert "Did you mean 'file_path'" in result.error

    def test_completely_unknown_arg_lists_expected_schema(self) -> None:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="read",
                description="read a file",
                parameters={"file_path": "str"},
                execute=lambda file_path="": None,
            )
        )
        result = reg.dispatch(ToolCall(tool="read", args={"zzqq": "x"}))
        assert result.error_kind == ErrorKind.VALIDATION
        assert "zzqq" in result.error
        assert "file_path" in result.error  # schema hint

    def test_missing_required_still_rejected(self) -> None:
        """Backward compat: missing-required diagnostic still fires."""
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="read",
                description="read a file",
                parameters={"file_path": "str"},
                execute=lambda file_path="": None,
            )
        )
        result = reg.dispatch(ToolCall(tool="read", args={}))
        assert result.error_kind == ErrorKind.VALIDATION
        assert "missing required" in result.error.lower()
        assert "file_path" in result.error


# ── ToolValidationError raised inside a tool body ────────────────


class TestToolValidationErrorRouting:
    def test_raise_becomes_clean_validation_result(self) -> None:
        """Tool authors raise ToolValidationError; dispatcher routes to a
        clean ToolResult.error with VALIDATION kind and non-retriable."""
        choices = ["apple", "banana", "cherry"]

        def pick(name: str = "") -> dict:
            if name not in choices:
                hint = suggest_similar(name, choices)
                suffix = f" Did you mean {hint!r}?" if hint else ""
                raise ToolValidationError(f"fruit {name!r} not found.{suffix}")
            return {"picked": name}

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="pick",
                description="pick a fruit",
                parameters={"name": "str"},
                execute=pick,
            )
        )
        # Happy path - unchanged.
        ok = reg.dispatch(ToolCall(tool="pick", args={"name": "apple"}))
        assert ok.error is None
        assert ok.data == {"picked": "apple"}

        # Typo path - clean structured error, with did-you-mean baked in.
        bad = reg.dispatch(ToolCall(tool="pick", args={"name": "aple"}))
        assert bad.data is None
        assert bad.error_kind == ErrorKind.VALIDATION
        assert bad.error_retriable is False
        assert "'aple'" in bad.error
        assert "Did you mean 'apple'" in bad.error
        # The message is the author's message - no "ToolValidationError:"
        # type prefix, so it renders cleanly in LLM context.
        assert not bad.error.startswith("ToolValidationError:")

    def test_validation_error_distinct_from_execution_error(self) -> None:
        """Plain exceptions still classify as EXECUTION (or stdlib kinds).

        ToolValidationError is the opt-in semantic signal.
        """

        def boom() -> None:
            raise RuntimeError("uh oh")

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="boom",
                description="explodes",
                parameters={},
                execute=boom,
            )
        )
        r = reg.dispatch(ToolCall(tool="boom", args={}))
        assert r.error_kind == ErrorKind.EXECUTION
        # RuntimeError keeps the type prefix (no opt-in clean message).
        assert r.error.startswith("RuntimeError:")


# ── Explicit: ToolValidationError is importable from package root ─


def test_tool_validation_error_exported_from_package() -> None:
    """The top-level import path is part of the public API."""
    import looplet

    assert looplet.ToolValidationError is ToolValidationError
    assert looplet.suggest_similar is suggest_similar


# ── Does NOT bleed into stdlib-exception classification ──────────


def test_value_error_still_routes_to_validation_for_back_compat() -> None:
    """Prior behaviour: ValueError/TypeError/KeyError inside a tool map
    to VALIDATION. We must not regress that when adding the new
    opt-in ToolValidationError path."""

    def bad(x: str = "") -> None:
        raise ValueError(f"bad input: {x}")

    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="bad",
            description="fail",
            parameters={"x": "str"},
            execute=bad,
        )
    )
    r = reg.dispatch(ToolCall(tool="bad", args={"x": "hi"}))
    assert r.error_kind == ErrorKind.VALIDATION
    assert "ValueError" in r.error  # still type-prefixed since not opt-in


# ── Empty / None values for required parameters ─────────────────


class TestEmptyRequiredArgRejected:
    """LLMs sometimes send {"command": ""} or {"command": null} which
    passes the "key exists" check but produces silent failures (bash
    runs empty command → exit 0, no output). The dispatch layer now
    rejects empty/None values for required string parameters."""

    @pytest.fixture()
    def registry(self) -> BaseToolRegistry:
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="bash",
                description="Run a command",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                execute=lambda *, command: {"exit_code": 0},
            )
        )
        return reg

    def test_empty_string_rejected(self, registry: BaseToolRegistry) -> None:
        r = registry.dispatch(ToolCall(tool="bash", args={"command": ""}))
        assert r.error is not None
        assert r.error_kind == ErrorKind.VALIDATION
        assert "empty" in r.error.lower()
        assert "command" in r.error

    def test_whitespace_only_rejected(self, registry: BaseToolRegistry) -> None:
        r = registry.dispatch(ToolCall(tool="bash", args={"command": "   "}))
        assert r.error is not None
        assert r.error_kind == ErrorKind.VALIDATION

    def test_none_value_rejected(self, registry: BaseToolRegistry) -> None:
        r = registry.dispatch(ToolCall(tool="bash", args={"command": None}))
        assert r.error is not None
        assert r.error_kind == ErrorKind.VALIDATION

    def test_valid_command_accepted(self, registry: BaseToolRegistry) -> None:
        r = registry.dispatch(ToolCall(tool="bash", args={"command": "echo hello"}))
        assert r.error is None
        assert r.data == {"exit_code": 0}

    def test_optional_param_empty_still_allowed(self) -> None:
        """Only required params are validated for emptiness."""
        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="read",
                description="Read file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "encoding": {"type": "string"},
                    },
                    "required": ["path"],
                },
                execute=lambda *, path, encoding="": {"path": path},
            )
        )
        # Empty optional param should NOT be rejected
        r = reg.dispatch(ToolCall(tool="read", args={"path": "x.py", "encoding": ""}))
        assert r.error is None
        assert r.data == {"path": "x.py"}

    def test_missing_arg_still_uses_missing_error(self, registry: BaseToolRegistry) -> None:
        """Missing args should show 'missing required' not 'empty value'."""
        r = registry.dispatch(ToolCall(tool="bash", args={}))
        assert r.error is not None
        assert "missing required" in r.error.lower()

    def test_error_is_retriable(self, registry: BaseToolRegistry) -> None:
        r = registry.dispatch(ToolCall(tool="bash", args={"command": ""}))
        assert r.error_detail is not None
        assert r.error_detail.retriable
