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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["messages", "stateful"])
async def test_reactive_compaction_refreshes_typed_input(asynchronous, mode):
    import copy

    from looplet import AssistantTurnDecision, RunResult, async_composable_loop
    from looplet.compact import CompactOutcome
    from looplet.conversation import Message, MessageRole

    requests = []
    resets = []

    class Backend:
        conversation_mode = mode

        def set_conversation_input(self, snapshot):
            self.snapshot = snapshot

        def apply_context_compaction(self, snapshot):
            resets.append(copy.deepcopy(snapshot))
            return True

        def generate(self, prompt, **kwargs):
            requests.append(copy.deepcopy(self.snapshot))
            if len(requests) == 1:
                return "Retained hypothesis"
            if len(requests) < 4:
                raise _PromptTooLongError()
            return '{"tool":"done","args":{"summary":"accepted"}}'

    class AsyncBackend(Backend):
        async def generate(self, prompt, **kwargs):
            return super().generate(prompt, **kwargs)

    class Compact:
        def compact(self, **kwargs):
            kwargs["conversation"].messages[:] = [
                Message(role=MessageRole.USER, content="Reduced evidence")
            ]
            return CompactOutcome(reason=kwargs["reason"])

    state = DefaultState(max_steps=1)
    arguments = dict(
        llm=AsyncBackend() if asynchronous else Backend(),
        tools=_registry(),
        state=state,
        config=LoopConfig(
            max_steps=1,
            max_model_calls=4,
            use_native_tools=False,
            compact_service=Compact(),
            assistant_turn_policy=lambda context: AssistantTurnDecision("continue"),
        ),
    )
    steps = (
        [step async for step in async_composable_loop(**arguments)]
        if asynchronous
        else list(composable_loop(**arguments))
    )
    assert [step.tool_call.tool for step in steps] == ["done"]
    assert RunResult.from_state(state).completed
    assert len(requests) == 4
    assert requests[-1] is not None
    assert "Reduced evidence" in str(requests[-1]["conversation"])
    assert "Retained hypothesis" not in str(requests[-1]["conversation"])
    assert len(resets) == (1 if mode == "stateful" else 0)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("phase", ["pre_compact", "post_compact", "cleanup"])
async def test_compaction_detects_hook_and_cleanup_projection_changes(asynchronous, phase):
    from looplet.assistant_turns import TURN_STATE_KEY, _AssistantTurns
    from looplet.compact import CompactOutcome, run_compact, run_compact_async
    from looplet.conversation import Conversation, Message, MessageRole
    from looplet.session import SessionLog

    state = DefaultState()
    turns = _AssistantTurns(state, LoopConfig(max_model_calls=2))
    conversation = Conversation()
    conversation.append(Message(role=MessageRole.USER, content="Original evidence"))

    def change_projection():
        conversation.messages[0].content = "Reduced evidence"

    class Hook:
        def on_event(self, payload):
            if payload.event.value == phase:
                change_projection()

    class Compact:
        def compact(self, **kwargs):
            return CompactOutcome(
                reason=kwargs["reason"],
                cleanup=change_projection if phase == "cleanup" else None,
            )

    arguments = dict(
        service=Compact(),
        hooks=[Hook()],
        state=state,
        session_log=SessionLog(),
        llm=turns.wrap(_FlakyLLM([])),
        conversation=conversation,
        step_num=1,
        reason="proactive",
    )
    if asynchronous:
        await run_compact_async(**arguments)
    else:
        run_compact(**arguments)
    assert state.metadata[TURN_STATE_KEY]["context_reset_required"]


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
