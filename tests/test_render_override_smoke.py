"""Smoke tests for :attr:`LoopConfig.render_messages_override`."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    ContextProjection,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="add",
            description="Add",
            parameters={"a": "int", "b": "int"},
            execute=lambda *, a, b: {"sum": a + b},
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="Finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


class TestRenderMessagesOverride:
    def test_override_can_consume_context_projection(self):
        projections: list[ContextProjection] = []

        def project(*, projection):
            projections.append(projection)
            return projection.default_prompt

        list(
            composable_loop(
                llm=MockLLMBackend(responses=['{"tool":"done","args":{"answer":"ok"}}']),
                tools=_tools(),
                state=DefaultState(max_steps=2),
                config=LoopConfig(max_steps=2, render_messages_override=project),
            )
        )

        assert projections
        assert projections[0].messages == ()
        assert projections[0].step_num == 1
        assert projections[0].to_dict()["message_count"] == 0

    def test_projection_isolated_from_nested_input_mutation(self):
        seen = []

        def project(*, projection):
            seen.append(projection)
            projection.task["nested"]["value"] = "renderer-change"
            projection.state_summary["facts"]["value"] = "renderer-change"
            return projection.default_prompt

        task = {"nested": {"value": "original"}}
        state = DefaultState(max_steps=2, metadata={"facts": {"value": "original"}})
        tools = _tools()
        list(
            composable_loop(
                llm=MockLLMBackend(responses=['{"tool":"done","args":{"answer":"ok"}}']),
                tools=tools,
                state=state,
                config=LoopConfig(max_steps=2, render_messages_override=project),
                task=task,
            )
        )

        assert seen
        assert task["nested"]["value"] == "original"
        assert state.metadata["facts"]["value"] == "original"

    def test_override_controls_prompt_bytes(self):
        seen_prompts: list[str] = []
        llm = MockLLMBackend(
            responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]
        )
        # Wrap generate to capture what the backend sees.
        _orig = llm.generate

        def capture(prompt, **kwargs):
            seen_prompts.append(prompt)
            return _orig(prompt, **kwargs)

        llm.generate = capture  # type: ignore[method-assign]

        def rewrite(*, messages, default_prompt, step_num):
            return f"<REWRITTEN step={step_num} msgs={len(messages)}>"

        list(
            composable_loop(
                llm=llm,
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3, render_messages_override=rewrite),
            )
        )
        assert seen_prompts
        assert all(p.startswith("<REWRITTEN") for p in seen_prompts)

    def test_override_sees_live_conversation(self):
        messages_seen: list[int] = []

        def spy(*, messages, default_prompt, step_num):
            messages_seen.append(len(messages))
            return default_prompt

        llm = MockLLMBackend(
            responses=[
                '{"tool":"add","args":{"a":1,"b":2},"reasoning":"r"}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]
        )
        list(
            composable_loop(
                llm=llm,
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3, render_messages_override=spy),
            )
        )
        # Step 1 sees empty conversation; step 2 sees the tool-use exchange.
        assert len(messages_seen) == 2
        assert messages_seen[0] == 0
        assert messages_seen[1] > messages_seen[0]

    def test_no_override_uses_default_prompt(self):
        """Default path unchanged when override is None."""
        llm = MockLLMBackend(
            responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]
        )
        steps = list(
            composable_loop(
                llm=llm,
                tools=_tools(),
                state=DefaultState(max_steps=3),
                config=LoopConfig(max_steps=3),
            )
        )
        assert steps  # loop completes normally
