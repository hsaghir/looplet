"""Tests for looplet.parse — JSON and native tool parsing."""

from __future__ import annotations

import pytest

from looplet.parse import parse_multi_tool_calls, parse_native_tool_use
from looplet.types import ToolCall

pytestmark = pytest.mark.smoke


# ── parse_multi_tool_calls ────────────────────────────────────────


class TestSingleToolJSON:
    def test_basic_single_tool(self) -> None:
        raw = '{"tool": "search", "args": {"query": "hello"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "search"
        assert calls[0].args == {"query": "hello"}

    def test_single_tool_with_reasoning(self) -> None:
        raw = '{"tool": "lookup", "args": {}, "reasoning": "need data"}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].reasoning == "need data"

    def test_single_tool_no_args(self) -> None:
        raw = '{"tool": "ping"}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "ping"
        assert calls[0].args == {}

    def test_returns_list_of_tool_calls(self) -> None:
        raw = '{"tool": "x", "args": {}}'
        calls = parse_multi_tool_calls(raw)
        assert isinstance(calls, list)
        assert all(isinstance(c, ToolCall) for c in calls)


class TestMultiToolJSON:
    def test_multi_tool_two_calls(self) -> None:
        raw = '{"tools": [{"tool": "a", "args": {}}, {"tool": "b", "args": {"k": 1}}]}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 2
        assert calls[0].tool == "a"
        assert calls[1].tool == "b"
        assert calls[1].args == {"k": 1}

    def test_multi_tool_with_theory(self) -> None:
        raw = '{"tools": [{"tool": "x", "args": {}}], "theory": "my plan"}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].args.get("__theory__") == "my plan"

    def test_multi_tool_with_reasoning_fallback(self) -> None:
        raw = '{"tools": [{"tool": "x", "args": {}}], "reasoning": "why"}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].reasoning == "why"

    def test_multi_tool_per_item_reasoning(self) -> None:
        raw = '{"tools": [{"tool": "x", "args": {}, "reasoning": "item reason"}]}'
        calls = parse_multi_tool_calls(raw)
        assert calls[0].reasoning == "item reason"

    def test_multi_tool_empty_list(self) -> None:
        raw = '{"tools": []}'
        calls = parse_multi_tool_calls(raw)
        assert calls == []

    def test_multi_tool_skips_invalid_items(self) -> None:
        raw = '{"tools": [{"tool": "valid"}, "not-a-dict", {"no_tool_key": 1}]}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "valid"


class TestMarkdownFencedJSON:
    def test_json_code_fence(self) -> None:
        raw = '```json\n{"tool": "search", "args": {}}\n```'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "search"

    def test_plain_code_fence(self) -> None:
        raw = '```\n{"tool": "calc", "args": {"x": 5}}\n```'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "calc"

    def test_fenced_multi_tool(self) -> None:
        raw = '```json\n{"tools": [{"tool": "a"}, {"tool": "b"}]}\n```'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 2


class TestExtraSurroundingText:
    def test_json_with_preamble(self) -> None:
        raw = 'Sure, here is my answer:\n{"tool": "lookup", "args": {}}\nDone.'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "lookup"

    def test_json_with_trailing_text(self) -> None:
        raw = '{"tool": "run", "args": {}} // execute this'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1


class TestMalformedJSON:
    def test_missing_tool_field(self) -> None:
        raw = '{"action": "search", "args": {}}'
        calls = parse_multi_tool_calls(raw)
        assert calls == []

    def test_completely_invalid_json(self) -> None:
        calls = parse_multi_tool_calls("not json at all")
        assert calls == []

    def test_truncated_json(self) -> None:
        calls = parse_multi_tool_calls('{"tool": "x", "args":')
        assert calls == []

    def test_tools_not_a_list(self) -> None:
        raw = '{"tools": "not-a-list"}'
        calls = parse_multi_tool_calls(raw)
        assert calls == []


class TestEmptyAndNoneInput:
    def test_empty_string(self) -> None:
        assert parse_multi_tool_calls("") == []

    def test_whitespace_only(self) -> None:
        assert parse_multi_tool_calls("   \n  ") == []

    def test_none_input(self) -> None:
        assert parse_multi_tool_calls(None) == []  # type: ignore[arg-type]


# ── parse_native_tool_use ─────────────────────────────────────────


class TestNativeToolUse:
    def test_single_block(self) -> None:
        blocks = [{"type": "tool_use", "id": "abc", "name": "search", "input": {"q": "hi"}}]
        calls = parse_native_tool_use(blocks)
        assert len(calls) == 1
        assert calls[0].tool == "search"
        assert calls[0].args == {"q": "hi"}

    def test_multiple_blocks(self) -> None:
        blocks = [
            {"type": "tool_use", "id": "1", "name": "a", "input": {}},
            {"type": "tool_use", "id": "2", "name": "b", "input": {"x": 1}},
        ]
        calls = parse_native_tool_use(blocks)
        assert len(calls) == 2
        assert calls[0].tool == "a"
        assert calls[1].tool == "b"

    def test_skips_non_tool_use_blocks(self) -> None:
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "1", "name": "x", "input": {}},
        ]
        calls = parse_native_tool_use(blocks)
        assert len(calls) == 1
        assert calls[0].tool == "x"

    def test_skips_non_dict_items(self) -> None:
        blocks = ["string", None, {"type": "tool_use", "id": "1", "name": "y", "input": {}}]
        calls = parse_native_tool_use(blocks)
        assert len(calls) == 1

    def test_skips_block_with_empty_name(self) -> None:
        blocks = [{"type": "tool_use", "id": "1", "name": "", "input": {}}]
        calls = parse_native_tool_use(blocks)
        assert calls == []

    def test_empty_blocks_list(self) -> None:
        assert parse_native_tool_use([]) == []

    def test_reasoning_is_empty_string(self) -> None:
        blocks = [{"type": "tool_use", "id": "1", "name": "x", "input": {}}]
        calls = parse_native_tool_use(blocks)
        assert calls[0].reasoning == ""

    def test_non_dict_input_becomes_empty_dict(self) -> None:
        blocks = [{"type": "tool_use", "id": "1", "name": "x", "input": None}]
        calls = parse_native_tool_use(blocks)
        assert calls[0].args == {}

    def test_returns_tool_call_instances(self) -> None:
        blocks = [{"type": "tool_use", "id": "1", "name": "z", "input": {}}]
        calls = parse_native_tool_use(blocks)
        assert all(isinstance(c, ToolCall) for c in calls)


# ── Robust edit_file parsing ────────────────────────────────────


class TestRobustEditParsing:
    """Parser handles common LLM quirks when calling edit_file."""

    def test_literal_newlines_in_json_strings(self) -> None:
        """LLM puts actual newlines inside old_string/new_string."""
        raw = '{"tool": "edit_file", "args": {"file_path": "t.py", "old_string": "line1\nline2", "new_string": "new1\nnew2"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].args["file_path"] == "t.py"
        assert "\n" in calls[0].args["old_string"]

    def test_flat_args_without_wrapper(self) -> None:
        """LLM puts args as siblings of 'tool' instead of nested under 'args'."""
        raw = '{"tool": "edit_file", "file_path": "t.py", "old_string": "a", "new_string": "b"}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].args["file_path"] == "t.py"
        assert calls[0].args["old_string"] == "a"

    def test_input_key_instead_of_args(self) -> None:
        """LLM uses 'input' (Anthropic style) instead of 'args'."""
        raw = '{"tool": "edit_file", "input": {"file_path": "t.py", "old_string": "a", "new_string": "b"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].args["file_path"] == "t.py"

    def test_parameters_key_instead_of_args(self) -> None:
        """LLM uses 'parameters' (OpenAPI style) instead of 'args'."""
        raw = '{"tool": "edit_file", "parameters": {"file_path": "t.py", "old_string": "a", "new_string": "b"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].args["file_path"] == "t.py"

    def test_name_key_instead_of_tool(self) -> None:
        """LLM uses 'name' instead of 'tool'."""
        raw = '{"name": "bash", "args": {"command": "ls"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool == "bash"

    def test_thinking_key_preserved(self) -> None:
        """LLM uses 'thinking' key for reasoning."""
        raw = '{"tool": "bash", "thinking": "I need to list files", "args": {"command": "ls"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].reasoning == "I need to list files"

    def test_multiline_code_block_in_edit(self) -> None:
        """LLM sends actual multi-line Python code in old/new strings."""
        raw = '{"tool": "edit_file", "args": {"file_path": "app.py", "old_string": "class Foo:\n    def bar(self):\n        return 1", "new_string": "class Foo:\n    def bar(self):\n        return 2\n\n    def baz(self):\n        return 3"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert "class Foo:" in calls[0].args["old_string"]
        assert "def baz" in calls[0].args["new_string"]

    def test_tabs_in_code(self) -> None:
        """LLM sends code with literal tabs."""
        raw = '{"tool": "edit_file", "args": {"file_path": "t.py", "old_string": "def f():\n\treturn 1", "new_string": "def f():\n\treturn 2"}}'
        calls = parse_multi_tool_calls(raw)
        assert len(calls) == 1
        assert "\t" in calls[0].args["old_string"]
