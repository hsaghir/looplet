"""Tests for reactive compaction on prompt-too-long.

Covers two architectural invariants:

   multi-strategy reactive recovery chain as the sync loop when the
   LLM raises a prompt-too-long error. Without this, long async sessions
   silently fail on context overflow.

2. **Reset-on-success** - once any recovery strategy succeeds, the
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
    reg.register(
        ToolSpec(
            name="noop",
            description="no-op",
            parameters={},
            execute=lambda: {"ok": True},
            concurrent_safe=True,
        )
    )
    reg.register(
        ToolSpec(
            name="done",
            description="finish",
            parameters={"summary": "final"},
            execute=lambda summary="": {"done": True, "summary": summary},
        )
    )
    return reg


# ── Sync: reset-on-success ────────────────────────────────────────


class _FlakyLLM(LLMBackend):
    """Alternates prompt-too-long / success / prompt-too-long / success."""

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
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
        llm = _FlakyLLM(
            [
                _PromptTooLongError(),  # step 1: overflow
                good_tool,  # step 1: recovery retry succeeds
                _PromptTooLongError(),  # step 2: overflow again
                good_done,  # step 2: recovery retry succeeds
            ]
        )
        state = DefaultState(max_steps=5)
        reg = _registry()
        steps = list(
            composable_loop(
                llm=llm,
                task={"id": "T-1"},
                tools=reg,
                config=LoopConfig(max_steps=5, use_native_tools=False),
                state=state,
            )
        )
        # 2 real steps completed (noop + done), neither as __llm_error__
        assert len(steps) == 2, f"got {len(steps)} steps"
        tools_called = [s.tool_call.tool for s in steps]
        assert "__llm_error__" not in tools_called, (
            f"second overflow silently failed (tools={tools_called})"
        )
        assert tools_called == ["noop", "done"]


# ── Async: parity ─────────────────────────────────────────────────


class _AsyncFlakyLLM:
    """Async analog of _FlakyLLM."""

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        if not self._script:
            return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.parametrize("async_mode", [False, True])
async def test_repeated_overflow_recovers_without_repeating_pre_prompt(async_mode):
    from looplet import async_composable_loop

    script = [
        _PromptTooLongError(),
        '{"tool":"noop","args":{}}',
        _PromptTooLongError(),
        '{"tool":"done","args":{"summary":"ok"}}',
    ]
    backend = (_AsyncFlakyLLM if async_mode else _FlakyLLM)(script)
    calls = []

    class Hook:
        def pre_prompt(self, state, session_log, context, step_num):
            calls.append(step_num)
            return "retained briefing"

    kwargs = dict(
        llm=backend,
        tools=_registry(),
        hooks=[Hook()],
        config=LoopConfig(max_steps=3, use_native_tools=False),
    )
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )
    assert [step.tool_call.tool for step in steps] == ["noop", "done"]
    assert backend.calls == 4
    assert calls == [1, 2]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("routed", [False, True])
async def test_parse_recovery_keeps_explicit_text_policy_and_effective_backend(async_mode, routed):
    from looplet import async_composable_loop

    class Backend:
        def __init__(self):
            self.text_calls = 0
            self.native_calls = 0

        def generate(self, prompt, **kwargs):
            self.text_calls += 1
            return (
                "not a tool call"
                if self.text_calls == 1
                else '{"tool":"done","args":{"summary":"ok"}}'
            )

        def generate_with_tools(self, prompt, **kwargs):
            self.native_calls += 1
            return [{"type": "text", "text": "wrong protocol"}]

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

        async def generate_with_tools(self, prompt, **kwargs):
            return super().generate_with_tools(prompt, **kwargs)

    backend = (AsyncBackend if async_mode else Backend)()
    original = (AsyncBackend if async_mode else Backend)()

    class Router:
        def select(self, *, purpose):
            return backend

    kwargs = dict(
        llm=original if routed else backend,
        tools=_registry(),
        config=LoopConfig(max_steps=3, use_native_tools=False, router=Router() if routed else None),
    )
    steps = (
        [step async for step in async_composable_loop(**kwargs)]
        if async_mode
        else list(composable_loop(**kwargs))
    )

    assert [step.tool_call.tool for step in steps] == ["done"]
    assert backend.text_calls == 2
    assert backend.native_calls == original.native_calls == original.text_calls == 0
