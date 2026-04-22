"""Smoke tests for ``build_briefing`` / ``build_prompt`` hook slots."""
from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools():
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="done", description="finish",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


class _RecordingBackend:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt, *, max_tokens=2000, system_prompt="", temperature=0.2):
        self.prompts.append(prompt)
        return self._responses.pop(0)


class _BriefingHook:
    def __init__(self, text): self._text = text
    def build_briefing(self, state, session_log, context): return self._text
    def pre_loop(self, *a, **k): return None
    def pre_prompt(self, *a, **k): return None
    def post_dispatch(self, *a, **k): return None
    def check_done(self, *a, **k): return None
    def should_stop(self, *a, **k): return False


class _PromptHook:
    def __init__(self, text): self._text = text
    def build_prompt(self, **kw): return self._text
    def pre_loop(self, *a, **k): return None
    def pre_prompt(self, *a, **k): return None
    def post_dispatch(self, *a, **k): return None
    def check_done(self, *a, **k): return None
    def should_stop(self, *a, **k): return False


class TestHookBuildBriefing:
    def test_hook_briefing_wins_over_config(self):
        b = _RecordingBackend([
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ])
        cfg = LoopConfig(
            max_steps=2,
            build_briefing=lambda *a, **k: "CONFIG-BRIEF",
        )
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[_BriefingHook("HOOK-BRIEF")], config=cfg,
        ))
        assert "HOOK-BRIEF" in b.prompts[0]
        assert "CONFIG-BRIEF" not in b.prompts[0]

    def test_none_falls_back_to_config(self):
        class _NoneHook(_BriefingHook):
            def build_briefing(self, *a, **k): return None
        b = _RecordingBackend([
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ])
        cfg = LoopConfig(
            max_steps=2,
            build_briefing=lambda *a, **k: "CONFIG-BRIEF",
        )
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[_NoneHook("x")], config=cfg,
        ))
        assert "CONFIG-BRIEF" in b.prompts[0]


class TestHookBuildPrompt:
    def test_hook_prompt_wins(self):
        b = _RecordingBackend([
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ])
        cfg = LoopConfig(
            max_steps=2,
            build_prompt=lambda **kw: "CONFIG-PROMPT",
        )
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[_PromptHook("HOOK-PROMPT")], config=cfg,
        ))
        assert b.prompts[0] == "HOOK-PROMPT"

    def test_first_hook_wins(self):
        b = _RecordingBackend([
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ])
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[_PromptHook("FIRST"), _PromptHook("SECOND")],
            config=LoopConfig(max_steps=2),
        ))
        assert b.prompts[0] == "FIRST"

    def test_none_falls_back_to_default(self):
        class _NoneHook(_PromptHook):
            def build_prompt(self, **kw): return None
        b = _RecordingBackend([
            '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
        ])
        list(composable_loop(
            llm=b, tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[_NoneHook("x")], config=LoopConfig(max_steps=2),
        ))
        # Default template has "TASK" section.
        assert "TASK" in b.prompts[0]
