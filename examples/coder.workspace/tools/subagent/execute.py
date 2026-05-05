"""subagent tool — workspace wrapper around looplet.subagent.run_sub_loop."""

from __future__ import annotations

from pathlib import Path

from looplet import workspace_to_preset
from looplet.subagent import run_sub_loop
from looplet.types import ToolContext


def execute(ctx: ToolContext, *, prompt: str, max_steps: int = 5, system_prompt: str = "") -> dict:
    if ctx.llm is None:
        return {
            "error": "subagent requires an active ctx.llm from composable_loop; direct dispatch cannot run it.",
            "recovery": "Call subagent from inside a running looplet loop, or run run_sub_loop directly in Python.",
        }
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    coder_workspace = Path(__file__).resolve().parents[2]
    preset = workspace_to_preset(str(coder_workspace), runtime={"workspace": workspace})
    result = run_sub_loop(
        llm=ctx.llm,
        task={"goal": prompt},
        tools=preset.tools,
        max_steps=max(1, int(max_steps or 5)),
        system_prompt=system_prompt
        or "You are a focused coding sub-agent. Investigate the requested task and return concise findings.",
        state_mutating_tools=["done", "subagent"],
    )
    return {
        "summary": result.get("summary", ""),
        "findings": result.get("findings", []),
        "highlights": result.get("highlights", []),
        "step_count": len(result.get("steps", [])),
        "llm_calls": result.get("llm_calls", 0),
        "subagent_id": result.get("subagent_id"),
    }
