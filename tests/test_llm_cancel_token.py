"""Tests for LLM-side cancellation via CancelToken threading.

A ``CancelToken`` on ``LoopConfig`` must reach:

1. Every LLM call via ``llm_call_with_retry`` / ``async_llm_call_with_retry``
   when the backend's ``generate`` / ``generate_with_tools`` opts-in by
   declaring a ``cancel_token`` parameter. Backends without the
   parameter keep working unchanged (opt-in, like ``ctx`` on tools).
2. The ``ToolContext`` threaded into tool dispatch so tools and LLM calls
   share the same cancellation signal.
3. Cause an early loop exit when ``is_cancelled`` becomes true between
   turns — without raising, so the trace stays clean.
"""

from __future__ import annotations

import asyncio

from looplet.loop import LoopConfig, composable_loop
from looplet.scaffolding import llm_call_with_retry
from looplet.tools import BaseToolRegistry, ToolSpec
from looplet.types import CancelToken, DefaultState, LLMBackend, ToolContext


class _CancelAwareLLM(LLMBackend):
    """Backend that accepts an optional cancel_token kwarg."""

    def __init__(self) -> None:
        self.cancel_tokens_seen: list[CancelToken | None] = []

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
        cancel_token: CancelToken | None = None,
    ) -> str:
        self.cancel_tokens_seen.append(cancel_token)
        if cancel_token is not None and cancel_token.is_cancelled:
            raise RuntimeError("cancelled")
        return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'


class _VanillaLLM(LLMBackend):
    """Backend without cancel_token param — must keep working."""

    def __init__(self) -> None:
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
        return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'


def _reg() -> BaseToolRegistry:
    reg = BaseToolRegistry()
    reg.register(
        ToolSpec(
            name="done",
            description="finish",
            parameters={"summary": "s"},
            execute=lambda summary="": {"done": True, "summary": summary},
        )
    )
    return reg


class TestScaffoldingPassesCancelToken:
    def test_llm_call_with_retry_forwards_cancel_token_when_accepted(self):
        llm = _CancelAwareLLM()
        tok = CancelToken()
        result = llm_call_with_retry(llm, "hi", cancel_token=tok)
        assert result.ok
        assert llm.cancel_tokens_seen == [tok]

    def test_llm_call_with_retry_skips_kwarg_for_vanilla_backend(self):
        llm = _VanillaLLM()
        tok = CancelToken()
        # Must not raise TypeError despite backend not accepting cancel_token.
        result = llm_call_with_retry(llm, "hi", cancel_token=tok)
        assert result.ok
        assert llm.calls == 1


class TestLoopThreadsCancelToken:
    def test_config_cancel_token_reaches_backend(self):
        llm = _CancelAwareLLM()
        tok = CancelToken()
        cfg = LoopConfig(max_steps=2, cancel_token=tok)
        list(
            composable_loop(
                llm=llm,
                task={"id": "T-1"},
                tools=_reg(),
                config=cfg,
                state=DefaultState(max_steps=2),
            )
        )
        assert tok in llm.cancel_tokens_seen

    def test_loop_exits_early_when_token_cancelled_before_next_turn(self):
        """If the token is flipped between steps the loop must stop
        cleanly rather than call the LLM again."""

        class _CountingLLM(LLMBackend):
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str, **kw) -> str:
                self.calls += 1
                # First turn: a real tool call, not done
                return '```json\n{"tool": "noop", "args": {}}\n```'

        reg = BaseToolRegistry()
        tok = CancelToken()

        def _cancel_once(**kw):
            tok.cancel()
            return {"ok": True}

        reg.register(
            ToolSpec(
                name="noop",
                description="n",
                parameters={},
                execute=_cancel_once,
                concurrent_safe=False,
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"summary": "s"},
                execute=lambda summary="": {"done": True, "summary": summary},
            )
        )
        llm = _CountingLLM()
        cfg = LoopConfig(max_steps=5, cancel_token=tok)
        steps = list(
            composable_loop(
                llm=llm,
                task={"id": "T-1"},
                tools=reg,
                config=cfg,
                state=DefaultState(max_steps=5),
            )
        )
        # Tool ran once, then the loop noticed cancellation and stopped.
        assert len(steps) == 1
        assert llm.calls == 1, f"LLM called {llm.calls} times after cancel"


class TestToolContextSharesCancelToken:
    def test_tool_receives_same_cancel_token_as_llm(self):
        seen: dict[str, CancelToken | None] = {"tool": None}

        def _tool(ctx: ToolContext) -> dict:
            seen["tool"] = ctx.cancel_token
            return {"ok": True}

        reg = BaseToolRegistry()
        reg.register(
            ToolSpec(
                name="probe",
                description="p",
                parameters={},
                execute=_tool,
                concurrent_safe=False,
            )
        )
        reg.register(
            ToolSpec(
                name="done",
                description="d",
                parameters={"summary": "s"},
                execute=lambda summary="": {"done": True, "summary": summary},
            )
        )

        class _LLM(LLMBackend):
            def __init__(self) -> None:
                self.n = 0

            def generate(self, prompt: str, **kw) -> str:
                self.n += 1
                if self.n == 1:
                    return '```json\n{"tool": "probe", "args": {}}\n```'
                return '```json\n{"tool": "done", "args": {"summary": "ok"}}\n```'

        tok = CancelToken()
        cfg = LoopConfig(max_steps=5, cancel_token=tok)
        list(
            composable_loop(
                llm=_LLM(),
                task={"id": "T-1"},
                tools=reg,
                config=cfg,
                state=DefaultState(max_steps=5),
            )
        )
        assert seen["tool"] is tok
