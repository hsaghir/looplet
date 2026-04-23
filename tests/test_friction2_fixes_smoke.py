"""Round-2 friction fixes (2026-04-24).

Discovered by building three more sample agents (cart, broken-eval,
flaky-tool) against looplet:

1. ``build_prompt`` default ``action_prompt`` now includes the
   expected JSON shape (``{"tool": ..., "args": ..., "reasoning": ...}``)
   so the LLM stops burning the first step on empty-arg calls.
2. ``ToolSpec.to_api_schema`` now emits ``required`` on the
   auto-converted simple-format schema, matching ``to_json_schema``.
"""

from __future__ import annotations

import pytest

from looplet import BaseToolRegistry, ToolSpec, preview_prompt

pytestmark = pytest.mark.smoke


# ── Fix 4: action_prompt includes schema hint ──────────────────────


class TestActionPromptSchemaHint:
    def test_default_prompt_mentions_tool_and_args_keys(self) -> None:
        out = preview_prompt(task={"goal": "x"}, tools=BaseToolRegistry())
        assert '"tool"' in out, out
        assert '"args"' in out, out
        assert '"reasoning"' in out, out

    def test_default_prompt_says_json_only(self) -> None:
        out = preview_prompt(task={"goal": "x"}, tools=BaseToolRegistry())
        assert "JSON only" in out, out

    def test_custom_action_prompt_still_respected(self) -> None:
        from looplet.prompts import build_prompt

        out = build_prompt(
            task={"goal": "x"},
            tool_catalog="",
            action_prompt="CUSTOM PROMPT",
        )
        assert "CUSTOM PROMPT" in out
        assert '"tool"' not in out


# ── Fix 5: to_api_schema emits `required` ──────────────────────────


class TestApiSchemaRequired:
    def test_simple_format_api_schema_has_required(self) -> None:
        spec = ToolSpec(
            name="add",
            description="Sum two numbers.",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {},
        )
        schema = spec.to_api_schema()
        assert schema["input_schema"]["required"] == ["a", "b"]

    def test_json_schema_format_api_schema_passes_required_through(self) -> None:
        spec = ToolSpec(
            name="q",
            description="d",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
                "required": ["x"],
            },
            execute=lambda **_: {},
        )
        schema = spec.to_api_schema()
        # JSON Schema path uses parameters directly — required survives.
        assert schema["input_schema"]["required"] == ["x"]

    def test_no_params_api_schema_empty_required(self) -> None:
        spec = ToolSpec(
            name="ping",
            description="d",
            parameters={},
            execute=lambda: {},
        )
        schema = spec.to_api_schema()
        assert schema["input_schema"]["required"] == []
