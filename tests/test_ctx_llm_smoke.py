"""ctx.llm — tool-internal LLM access + nested provenance scoping."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    ToolSpec,
    composable_loop,
    register_done_tool,
)
from looplet.provenance import RecordingLLMBackend, TrajectoryRecorder
from looplet.testing import MockLLMBackend
from looplet.types import ToolContext

pytestmark = pytest.mark.smoke


class TestCtxLlmAvailable:
    def test_ctx_llm_is_populated(self):
        """Tools that accept ctx= receive ctx.llm from the loop."""
        received_llm = []

        def my_tool(*, query: str, ctx: ToolContext) -> dict:
            received_llm.append(ctx.llm)
            return {"result": "ok"}

        mock = MockLLMBackend(
            responses=[
                '{"tool": "search", "args": {"query": "x"}, "reasoning": "r"}',
                '{"tool": "done", "args": {"summary": "ok"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(name="search", description="s", parameters={"query": "str"}, execute=my_tool)
        )

        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        assert len(received_llm) == 1
        assert received_llm[0] is not None

    def test_ctx_llm_can_generate(self):
        """ctx.llm.generate() works — the tool can use the LLM internally."""
        internal_results = []

        def summarize_tool(*, text: str, ctx: ToolContext) -> dict:
            if ctx.llm is not None:
                summary = ctx.llm.generate(f"Summarize: {text}")
                internal_results.append(summary)
                return {"summary": summary}
            return {"summary": text[:50]}

        # Mock returns: step 1 → tool call, step 2 → done.
        # The internal ctx.llm.generate call also consumes a mock response,
        # so we need 3 responses total.
        mock = MockLLMBackend(
            responses=[
                '{"tool": "summarize", "args": {"text": "long text here"}, "reasoning": "r"}',
                "A brief summary.",  # internal call response
                '{"tool": "done", "args": {"summary": "done"}, "reasoning": "r"}',
            ]
        )
        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="summarize",
                description="s",
                parameters={"text": "str"},
                execute=summarize_tool,
            )
        )

        list(
            composable_loop(
                llm=mock,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        assert len(internal_results) == 1
        assert internal_results[0] == "A brief summary."


class TestNestedProvenance:
    def test_tool_internal_calls_tagged_with_scope(self):
        """When a RecordingLLMBackend is used, tool-internal calls get scope='tool:<name>'."""

        def classify_tool(*, text: str, ctx: ToolContext) -> dict:
            if ctx.llm is not None:
                label = ctx.llm.generate(f"Classify: {text}")
                return {"label": label}
            return {"label": "unknown"}

        inner_mock = MockLLMBackend(
            responses=[
                '{"tool": "classify", "args": {"text": "hello"}, "reasoning": "r"}',
                "positive",  # internal call
                '{"tool": "done", "args": {"summary": "classified"}, "reasoning": "r"}',
            ]
        )
        recording = RecordingLLMBackend(inner_mock)

        tools = BaseToolRegistry()
        register_done_tool(tools)
        tools.register(
            ToolSpec(
                name="classify",
                description="c",
                parameters={"text": "str"},
                execute=classify_tool,
            )
        )

        list(
            composable_loop(
                llm=recording,
                tools=tools,
                state=DefaultState(max_steps=5),
                config=LoopConfig(max_steps=5),
                task={},
            )
        )

        # Should have 3 calls total: step1 prompt, internal classify, step2 prompt
        assert len(recording.calls) == 3

        # The internal call should be scoped
        scoped = [c for c in recording.calls if c.scope is not None]
        assert len(scoped) == 1
        assert scoped[0].scope == "tool:classify"

        # Loop-level calls should have no scope
        unscoped = [c for c in recording.calls if c.scope is None]
        assert len(unscoped) == 2

    def test_scope_in_to_dict(self):
        """LLMCall.scope appears in to_dict() output."""
        from looplet.provenance import LLMCall

        call = LLMCall(
            index=0,
            timestamp=0.0,
            duration_ms=10.0,
            method="generate",
            prompt="test",
            system_prompt="",
            response="ok",
            scope="tool:search",
        )
        d = call.to_dict()
        assert d["scope"] == "tool:search"

    def test_scope_none_by_default(self):
        from looplet.provenance import LLMCall

        call = LLMCall(
            index=0,
            timestamp=0.0,
            duration_ms=10.0,
            method="generate",
            prompt="test",
            system_prompt="",
            response="ok",
        )
        assert call.scope is None
        assert call.to_dict()["scope"] is None
