"""Tests for PersistentMemorySource.

Problem: many agent frameworks inject a memory file on every turn;
it survives all compactions. Open-harness had no equivalent - callers must stuff memory
into the system prompt manually, and it isn't protected from
compaction.

Design: a domain-agnostic ``PersistentMemorySource`` protocol. Callers
attach one or more sources to ``LoopConfig``. They are rendered into a
stable ``═══ MEMORY ═══`` section of the default prompt on every turn.

This file verifies:

1. The protocol shape: any object with ``load(state) -> str`` qualifies.
2. A convenience ``CallableMemorySource`` wraps a plain lambda.
3. A convenience ``StaticMemorySource`` wraps a constant string.
4. ``prompts.build_prompt`` renders ``memory_sources`` above the TASK
   section so the LLM sees them first on every turn.
5. Empty outputs are skipped (no empty section headers).
6. Multiple sources are joined with blank lines, in order.
"""

from __future__ import annotations

from looplet.memory import (
    CallableMemorySource,
    PersistentMemorySource,
    StaticMemorySource,
    render_memory,
)
from looplet.prompts import build_prompt


class TestMemorySourceShape:
    def test_static_memory_source_returns_its_text(self):
        s = StaticMemorySource("you are a helpful security analyst.")
        assert s.load(state=None) == "you are a helpful security analyst."

    def test_callable_memory_source_invokes_fn_with_state(self):
        seen = {}

        def fn(state):
            seen["state"] = state
            return "dynamic memory"

        s = CallableMemorySource(fn)
        out = s.load(state={"k": 1})
        assert out == "dynamic memory"
        assert seen["state"] == {"k": 1}

    def test_any_object_with_load_satisfies_protocol(self):
        class Custom:
            def load(self, state):
                return "x"

        # Structural check
        assert isinstance(Custom(), PersistentMemorySource)


class TestRenderMemory:
    def test_single_source(self):
        out = render_memory([StaticMemorySource("hello")], state=None)
        assert out == "hello"

    def test_multiple_sources_joined_with_blank_line(self):
        out = render_memory(
            [StaticMemorySource("first"), StaticMemorySource("second")],
            state=None,
        )
        assert out == "first\n\nsecond"

    def test_empty_sources_are_skipped(self):
        out = render_memory(
            [StaticMemorySource(""), StaticMemorySource("real"), StaticMemorySource("   ")],
            state=None,
        )
        assert out == "real"

    def test_none_returning_source_is_safe(self):
        out = render_memory(
            [CallableMemorySource(lambda s: None), StaticMemorySource("kept")],
            state=None,
        )
        assert out == "kept"

    def test_no_sources_returns_empty_string(self):
        assert render_memory([], state=None) == ""


class TestPromptIntegration:
    def test_memory_section_appears_before_task(self):
        prompt = build_prompt(
            task={"id": "T-1", "title": "demo"},
            tool_catalog="- noop",
            memory="You must always cite sources.\n\nRubric: strictness=high.",
        )
        assert "═══ MEMORY ═══" in prompt
        # MEMORY must precede TASK
        assert prompt.index("═══ MEMORY ═══") < prompt.index("═══ TASK ═══")
        assert "cite sources" in prompt
        assert "Rubric" in prompt

    def test_no_memory_means_no_header(self):
        prompt = build_prompt(
            task={"id": "T-1"},
            tool_catalog="- noop",
            memory="",
        )
        assert "═══ MEMORY ═══" not in prompt
        # TASK still first
        assert prompt.startswith("═══ TASK ═══")

    def test_none_memory_means_no_header(self):
        prompt = build_prompt(
            task={"id": "T-1"},
            tool_catalog="- noop",
            memory=None,
        )
        assert "═══ MEMORY ═══" not in prompt
