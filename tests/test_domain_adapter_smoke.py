"""Smoke tests for :class:`DomainAdapter` bundling."""

from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    DomainAdapter,
    LoopConfig,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools():
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="done",
            description="finish",
            parameters={"answer": "str"},
            execute=lambda *, answer: {"answer": answer},
        )
    )
    return reg


class _RecordingBackend:
    def __init__(self, responses):
        self._r = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.prompts.append(prompt)
        return self._r.pop(0)


class TestDomainAdapter:
    def test_adapter_briefing_used_when_config_field_none(self):
        b = _RecordingBackend(['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}'])
        dom = DomainAdapter(build_briefing=lambda *a, **k: "ADAPTER-BRIEF")
        list(
            composable_loop(
                llm=b,
                tools=_tools(),
                state=DefaultState(max_steps=2),
                hooks=[],
                config=LoopConfig(max_steps=2, domain=dom),
            )
        )
        assert "ADAPTER-BRIEF" in b.prompts[0]

    def test_flat_field_overrides_adapter(self):
        b = _RecordingBackend(['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}'])
        dom = DomainAdapter(build_briefing=lambda *a, **k: "ADAPTER-BRIEF")
        cfg = LoopConfig(
            max_steps=2,
            domain=dom,
            build_briefing=lambda *a, **k: "FLAT-WINS",
        )
        list(
            composable_loop(
                llm=b,
                tools=_tools(),
                state=DefaultState(max_steps=2),
                hooks=[],
                config=cfg,
            )
        )
        assert "FLAT-WINS" in b.prompts[0]
        assert "ADAPTER-BRIEF" not in b.prompts[0]

    def test_adapter_build_trace_used(self):
        def _trace(**kw):
            return {"custom": True, "task": kw["task"]}

        dom = DomainAdapter(build_trace=_trace)
        gen = composable_loop(
            llm=_RecordingBackend(['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}']),
            tools=_tools(),
            state=DefaultState(max_steps=2),
            hooks=[],
            config=LoopConfig(max_steps=2, domain=dom),
            task={"goal": "hello"},
        )
        trace = None
        try:
            while True:
                next(gen)
        except StopIteration as s:
            trace = s.value
        assert trace == {"custom": True, "task": {"goal": "hello"}}

    def test_empty_adapter_noop(self):
        b = _RecordingBackend(['{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}'])
        list(
            composable_loop(
                llm=b,
                tools=_tools(),
                state=DefaultState(max_steps=2),
                hooks=[],
                config=LoopConfig(max_steps=2, domain=DomainAdapter()),
            )
        )
        # No crash; default briefing used.
        assert b.prompts
