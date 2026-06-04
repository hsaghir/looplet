"""T9 dogfood: provider usage/cost surfaced onto state + view.

Budget/cost hooks historically read ``backend.last_usage`` directly — a
runtime-local side-channel that an out-of-process (LEP) hook cannot see.
These tests lock the portable path: the loop stamps token usage onto
``state.metadata`` and into the ``POST_LLM_RESPONSE`` event payload, and
:func:`extract_view` projects it through a declared ``usage`` view so the
same hook works in-process or across a runtime boundary.
"""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    HookDecision,
    LifecycleEvent,
    LoopConfig,
    composable_loop,
)
from looplet.events import EventPayload
from looplet.hook_view import ViewSpec, extract_view
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


class _UsageBackend(MockLLMBackend):
    """Mock backend that reports token usage like a real provider."""

    def generate(self, prompt, **kwargs):  # type: ignore[override]
        text = super().generate(prompt, **kwargs)
        self.last_usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        return text


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="done",
            description="Finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


class _UsageRecorder:
    def __init__(self) -> None:
        self.post_llm_usage: list = []

    def on_event(self, payload: EventPayload) -> HookDecision | None:
        if payload.event == LifecycleEvent.POST_LLM_RESPONSE:
            self.post_llm_usage.append(payload.usage)
        return None


def _run(hook):
    state = DefaultState(max_steps=3)
    responses = [
        '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
    ]
    list(
        composable_loop(
            llm=_UsageBackend(responses=responses),
            tools=_tools(),
            state=state,
            hooks=[hook],
            config=LoopConfig(max_steps=3),
        )
    )
    return state


class TestUsageSurface:
    def test_usage_stamped_on_state_metadata(self):
        rec = _UsageRecorder()
        state = _run(rec)
        assert state.metadata.get("last_usage") == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        # Running total accumulates numeric fields across steps.
        assert state.metadata.get("usage_total", {}).get("total_tokens", 0) >= 15

    def test_post_llm_event_carries_usage(self):
        rec = _UsageRecorder()
        _run(rec)
        assert rec.post_llm_usage, "POST_LLM_RESPONSE should have fired"
        assert rec.post_llm_usage[0]["total_tokens"] == 15

    def test_view_reads_usage_from_state_fallback(self):
        # A hook that declares only a `usage` view, with no explicit usage
        # argument, still sees token usage projected from state metadata.
        state = DefaultState(max_steps=3)
        state.metadata["usage_total"] = {"total_tokens": 42}
        spec = ViewSpec(fields=frozenset({"usage"}))
        view = extract_view(spec, state=state)
        assert view == {"usage": {"total_tokens": 42}}

    def test_view_omits_usage_when_not_subscribed(self):
        state = DefaultState(max_steps=3)
        state.metadata["usage_total"] = {"total_tokens": 42}
        spec = ViewSpec(fields=frozenset({"tool"}))
        view = extract_view(spec, state=state)
        assert "usage" not in view
