"""Regression tests for dogfood round 16.

Round 16 surfaced two real loader gaps via the soc_triage cartridge:

* **setup.py was unnecessary** - the existing ``@<name>`` ref grammar
  in hook ``kwargs`` already injects resources. The dogfood originally
  wrote a setup.py because the ref grammar wasn't visible enough; it's
  documented now, and the cartridge dropped setup.py.

* **No declarative ``context_window_steps``** - the env-default of
  5 recent steps caused chained-tool-use cartridges (SOC triage:
  step 1 get_alert → step 8 lookup_user) to lose the source-of-truth
  payload before the LLM needed it again, producing hallucinated
  arguments downstream. Cartridges now set
  ``context_window_steps:`` in ``config.yaml`` and the loop honors
  it via :data:`looplet.context_budget._CONTEXT_WINDOW_STEPS_OVERRIDE`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from looplet import (
    DefaultState,
    LoopConfig,
    MockLLMBackend,
    composable_loop,
    tool,
    tools_from,
)


def test_loopconfig_context_window_steps_overrides_default() -> None:
    """Setting ``LoopConfig.context_window_steps`` must override the
    module-level default (5) for the duration of the loop.
    """
    from looplet.context_budget import (
        _CONTEXT_WINDOW_STEPS_OVERRIDE,
        get_context_window_steps,
    )

    # Start from the module default.
    assert _CONTEXT_WINDOW_STEPS_OVERRIDE.get() is None

    captured: list[int] = []

    @tool
    def probe() -> dict:
        captured.append(get_context_window_steps())
        return {"saw": captured[-1]}

    tools = tools_from([probe], include_done=True, done_parameters={"summary": "x"})
    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "probe", "args": {}, "reasoning": "", "call_id": "1"}),
            json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "", "call_id": "2"}
            ),
        ]
    )
    cfg = LoopConfig(max_steps=5, context_window_steps=42)
    state = DefaultState(max_steps=5)
    list(
        composable_loop(
            llm=llm,
            tools=tools,
            state=state,
            config=cfg,
            hooks=[],
            task={"goal": "test"},
        )
    )
    assert captured == [42], f"expected [42] inside the loop, got {captured}"
    # Override is reset at end of loop.
    assert _CONTEXT_WINDOW_STEPS_OVERRIDE.get() is None


def test_loopconfig_context_inline_per_step_chars_override() -> None:
    from looplet.context_budget import get_context_inline_per_step_chars

    captured: list[int] = []

    @tool
    def probe() -> dict:
        captured.append(get_context_inline_per_step_chars())
        return {"ok": True}

    tools = tools_from([probe], include_done=True, done_parameters={"summary": "x"})
    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "probe", "args": {}, "reasoning": "", "call_id": "1"}),
            json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "", "call_id": "2"}
            ),
        ]
    )
    cfg = LoopConfig(max_steps=5, context_inline_per_step_chars=99)
    state = DefaultState(max_steps=5)
    list(
        composable_loop(llm=llm, tools=tools, state=state, config=cfg, hooks=[], task={"goal": "x"})
    )
    assert captured == [99]


def test_loopconfig_context_window_total_chars_override() -> None:
    from looplet.context_budget import get_context_window_total_chars

    captured: list[int] = []

    @tool
    def probe() -> dict:
        captured.append(get_context_window_total_chars())
        return {"ok": True}

    tools = tools_from([probe], include_done=True, done_parameters={"summary": "x"})
    llm = MockLLMBackend(
        responses=[
            json.dumps({"tool": "probe", "args": {}, "reasoning": "", "call_id": "1"}),
            json.dumps(
                {"tool": "done", "args": {"summary": "ok"}, "reasoning": "", "call_id": "2"}
            ),
        ]
    )
    cfg = LoopConfig(max_steps=5, context_window_total_chars=12345)
    state = DefaultState(max_steps=5)
    list(
        composable_loop(llm=llm, tools=tools, state=state, config=cfg, hooks=[], task={"goal": "x"})
    )
    assert captured == [12345]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_context_window_steps_round_trips_via_config_yaml(tmp_path: Path) -> None:
    """A cartridge declaring ``context_window_steps`` in ``config.yaml``
    must produce a ``LoopConfig`` with the override set."""
    from looplet import cartridge_to_preset

    root = tmp_path / "x.cartridge"
    root.mkdir()
    (root / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (root / "config.yaml").write_text("max_steps: 5\n")
    (root / "runtime.yaml").write_text(
        "context_window_steps: 25\ncontext_window_total_chars: 50000\n"
    )
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("test")
    done = root / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )
    preset = cartridge_to_preset(str(root), strict=True)
    assert preset.config.context_window_steps == 25
    assert preset.config.context_window_total_chars == 50000
    assert preset.config.context_inline_per_step_chars is None  # not set


def test_default_state_context_summary_uses_overridden_window() -> None:
    """DefaultState.context_summary must inline the number of steps
    set by the override, not the module default."""
    from looplet.context_budget import _CONTEXT_WINDOW_STEPS_OVERRIDE
    from looplet.types import DefaultState, Step, ToolCall, ToolResult

    state = DefaultState(max_steps=20)
    for i in range(10):
        state.steps.append(
            Step(
                number=i + 1,
                tool_call=ToolCall(tool="t", args={"i": i}, reasoning="", call_id=str(i)),
                tool_result=ToolResult(tool="t", args_summary=f"i={i}", data={"i": i}, error=None),
            )
        )

    # Default: 5 steps inlined → first step (i=0) NOT visible.
    summary_default = state.context_summary()
    assert "i=0" not in summary_default

    # With override: 10 steps → first step (i=0) IS visible.
    token = _CONTEXT_WINDOW_STEPS_OVERRIDE.set(10)
    try:
        summary_wide = state.context_summary()
    finally:
        _CONTEXT_WINDOW_STEPS_OVERRIDE.reset(token)
    assert "i=0" in summary_wide, (
        "with context_window_steps=10, the first step (i=0) must appear "
        "in the rendered context summary"
    )
