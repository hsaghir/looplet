"""Regression tests for friction surfaced in dogfood round 15
(soc_triage cartridge — production SOC analyst).

Two things came out:

* **bind name collision** — the loop calls ``hook.bind(loop_ctx)``.
  User cartridges naturally name their dependency-injection method
  ``bind(*, my_resource)`` (it's a generic verb). The loop used to
  crash with ``TypeError: bind() takes 1 positional argument but
  2 were given`` whenever a hook defined a kw-only ``bind``.
  Fix: sig-aware dispatch — only call bind(loop_ctx) if the
  signature actually accepts a positional argument.

* **mid-loop PII scrubbing causes hallucinations** — not a loader
  bug; documented as a pitfall in docs/pitfalls.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet import (
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    composable_loop,
    tool,
    tools_from,
)

# ── bind sig-aware dispatch ──────────────────────────────────────


def test_loop_does_not_crash_on_hook_with_kw_only_bind() -> None:
    """A hook whose ``bind`` method takes only keyword args (a common
    user pattern for declaring resource dependencies) must NOT cause
    the loop to crash. The loop should detect the signature shape
    and skip the call rather than passing its own loop_ctx as a
    positional arg.
    """

    bind_calls: list[dict] = []

    class HookWithKwOnlyBind:
        def bind(self, *, my_resource):  # kw-only — DOES NOT match loop's bind protocol
            bind_calls.append({"my_resource": my_resource})

    @tool
    def noop() -> dict:
        return {"ok": True}

    hook = HookWithKwOnlyBind()
    # Don't actually call bind here; the loop should detect the shape.
    tools = tools_from([noop], include_done=True, done_parameters={"summary": "x"})
    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "noop", "args": {}, "reasoning": "", "call_id": "1"}),
            json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "", "call_id": "2"}
            ),
        ]
    )
    cfg = LoopConfig(max_steps=3)
    state = DefaultState(max_steps=3)
    # No exception — even though the hook has bind() that the loop
    # used to call positionally and explode on.
    steps = list(
        composable_loop(
            llm=llm,
            tools=tools,
            state=state,
            config=cfg,
            hooks=[hook],
            task={"goal": "test"},
        )
    )
    assert len(steps) == 2
    assert steps[0].tool_call.tool == "noop"
    assert steps[1].tool_call.tool == "done"
    # The loop did NOT call the user's kw-only bind; setup.py would
    # be the one to invoke it.
    assert bind_calls == []


def test_loop_still_calls_positional_bind_protocol() -> None:
    """Sanity: hooks following the looplet ``bind(loop_ctx)`` protocol
    still get called. Don't regress the original contract."""

    received: list[object] = []

    class HookWithPositionalBind:
        def bind(self, loop_ctx):  # positional — MATCHES loop protocol
            received.append(loop_ctx)

    @tool
    def noop() -> dict:
        return {"ok": True}

    hook = HookWithPositionalBind()
    tools = tools_from([noop], include_done=True, done_parameters={"summary": "x"})
    llm = MockLLMBackend(
        responses=[
            json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "", "call_id": "1"}
            ),
        ]
    )
    cfg = LoopConfig(max_steps=2)
    state = DefaultState(max_steps=2)
    list(
        composable_loop(
            llm=llm,
            tools=tools,
            state=state,
            config=cfg,
            hooks=[hook],
            task={"goal": "test"},
        )
    )
    assert len(received) == 1, (
        "loop must still call hook.bind(loop_ctx) for hooks that follow the original protocol"
    )
    # The loop_ctx must look right (has tools/config/state).
    ctx = received[0]
    assert hasattr(ctx, "tools") or hasattr(ctx, "config") or hasattr(ctx, "state")
