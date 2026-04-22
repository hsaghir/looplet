"""Smoke tests for the proactive ``should_compact`` hook slot."""
from __future__ import annotations

import pytest

from looplet import (
    BaseToolRegistry,
    DefaultState,
    LoopConfig,
    TruncateCompact,
    composable_loop,
)
from looplet.testing import MockLLMBackend
from looplet.tools import ToolSpec

pytestmark = pytest.mark.smoke


def _tools() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="echo", description="echo",
        parameters={"msg": "str"},
        execute=lambda *, msg: {"msg": msg},
    ))
    reg.register(ToolSpec(
        name="done", description="finish",
        parameters={"answer": "str"},
        execute=lambda *, answer: {"answer": answer},
    ))
    return reg


class _Compacter:
    """Counts compact calls triggered by the hook."""
    def __init__(self, on_steps: set[int]):
        self.on_steps = on_steps
        self.calls = 0

    def should_compact(self, state, session_log, conversation, step_num):
        return step_num in self.on_steps

    def pre_loop(self, *a, **k): return None
    def pre_prompt(self, *a, **k): return None
    def post_dispatch(self, *a, **k): return None
    def check_done(self, *a, **k): return None
    def should_stop(self, *a, **k): return False


class _CountingCompact(TruncateCompact):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def compact(self, **kw):
        self.calls += 1
        return super().compact(**kw)


class TestProactiveCompact:
    def test_hook_triggers_compact_before_prompt(self):
        svc = _CountingCompact()
        cfg = LoopConfig(max_steps=3, compact_service=svc)
        h = _Compacter(on_steps={1})
        list(composable_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"echo","args":{"msg":"a"},"reasoning":"r"}',
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(), state=DefaultState(max_steps=3),
            hooks=[h], config=cfg,
        ))
        assert svc.calls == 1

    def test_no_hook_means_no_compact(self):
        svc = _CountingCompact()
        cfg = LoopConfig(max_steps=2, compact_service=svc)
        list(composable_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[], config=cfg,
        ))
        assert svc.calls == 0

    def test_compact_skipped_when_service_unset(self):
        cfg = LoopConfig(max_steps=2)
        h = _Compacter(on_steps={1})
        # No service configured — loop must not crash; hook is just ignored.
        list(composable_loop(
            llm=MockLLMBackend(responses=[
                '{"tool":"done","args":{"answer":"ok"},"reasoning":"r"}',
            ]),
            tools=_tools(), state=DefaultState(max_steps=2),
            hooks=[h], config=cfg,
        ))
