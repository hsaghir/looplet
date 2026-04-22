"""Tests for reactive compaction on prompt-too-long.

Covers two architectural invariants:

   multi-strategy reactive recovery chain as the sync loop when the
   LLM raises a prompt-too-long error. Without this, long async sessions
   silently fail on context overflow.

2. **Reset-on-success** — once any recovery strategy succeeds, the
   ``recovery_state`` ledger must reset so a *later* prompt-too-long
   in the same run can trigger the chain again. Without reset, a
   single successful compaction permanently disables recovery.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from looplet.loop import LoopConfig, composable_loop
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import DefaultState, LLMBackend

# ── Shared helpers ────────────────────────────────────────────────

class _PromptTooLongError(Exception):
    """Mimics Anthropic's prompt-too-long error shape."""

    def __init__(self) -> None:
        super().__init__("prompt is too long: 200000 tokens > 180000 limit")


def _registry() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(ToolSpec(
        name="noop", description="no-op",
        parameters={}, execute=lambda: {"ok": True},
        concurrent_safe=True,
    ))
    reg.register(ToolSpec(
        name="done", description="finish",
        parameters={"summary": "final"},
        execute=lambda summary="": {"done": True, "summary": summary},
    ))
    return reg


# ── Sync: reset-on-success ────────────────────────────────────────

class _FlakyLLM(LLMBackend):
    """Alternates prompt-too-long / success / prompt-too-long / success."""

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    def generate(self, prompt: str, *, max_tokens: int = 2000,
                 system_prompt: str = "", temperature: float = 0.2) -> str:
        self.calls += 1
        if not self._script:
            return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestSyncResetOnSuccess:
    def test_second_prompt_too_long_still_recovers(self):
        """After a successful recovery, a *later* overflow must still
        trigger the chain. Today, recovery_state flags persist forever,
        so the second overflow silently returns None."""
        # Step 1: overflow → recovery succeeds with noop → yields step
        # Step 2: LLM returns done
        # But we want: Step 1 overflow → recovery → success (noop);
        # Step 2 overflow again → recovery should re-fire → success (done)
        good_tool = '```json\n{"tool": "noop", "args": {}}\n```'
        good_done = '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'
        llm = _FlakyLLM([
            _PromptTooLongError(),      # step 1: overflow
            good_tool,                  # step 1: recovery retry succeeds
            _PromptTooLongError(),      # step 2: overflow again
            good_done,                  # step 2: recovery retry succeeds
        ])
        state = DefaultState(max_steps=5)
        reg = _registry()
        steps = list(composable_loop(
            llm=llm, task={"id": "T-1"}, tools=reg,
            config=LoopConfig(max_steps=5), state=state,
        ))
        # 2 real steps completed (noop + done), neither as __llm_error__
        assert len(steps) == 2, f"got {len(steps)} steps"
        tools_called = [s.tool_call.tool for s in steps]
        assert "__llm_error__" not in tools_called, \
            f"second overflow silently failed (tools={tools_called})"
        assert tools_called == ["noop", "done"]


# ── Async: parity ─────────────────────────────────────────────────

class _AsyncFlakyLLM:
    """Async analog of _FlakyLLM."""

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    async def generate(self, prompt: str, *, max_tokens: int = 2000,
                       system_prompt: str = "", temperature: float = 0.2) -> str:
        self.calls += 1
        if not self._script:
            return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

