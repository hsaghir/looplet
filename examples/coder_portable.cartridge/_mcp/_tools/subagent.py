"""subagent tool - portable twin of the in-process subagent.

The original ``tools/subagent/execute.py`` called
``cartridge_to_preset(<coder cartridge root>)`` to rebuild the parent's
tool set for the sub-loop. In a fully-portable cartridge that would
recursively re-load THIS cartridge (re-spawning its MCP/SSP/MGP
servers), so instead we build an isolated, in-process coding tool set
with :func:`coder_lib_tools.make_tools` (bash/list_dir/read_file/
write_file/edit_file/glob/grep + think + done) bound to a FRESH
``FileCache`` - the sub-agent already runs with isolated state, so it
should not share the parent's cache anyway.

The LLM is reached the portable way: ``ctx.llm`` is a
:class:`looplet.model_gateway.ModelGatewayClient` proxy injected by the
MCP tools server, which forwards generation to the HOST's bound backend
over the Model Gateway Protocol. When no backend is bound (direct
dispatch / no active loop) ``ctx.llm`` is ``None`` and we return the
same actionable error the original did.
"""

from __future__ import annotations

from coder_lib_tools import FileCache, make_tools

from looplet.subagent import run_sub_loop
from looplet.types import ToolContext


def execute(ctx: ToolContext, *, prompt: str, max_steps: int = 5, system_prompt: str = "") -> dict:
    if ctx.llm is None:
        return {
            "error": "subagent requires an active ctx.llm from composable_loop; "
            "direct dispatch cannot run it.",
            "recovery": "Call subagent from inside a running looplet loop (the "
            "host binds an LLM over the Model Gateway), or run run_sub_loop "
            "directly in Python.",
        }
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    tools = make_tools(workspace, FileCache(workspace))
    result = run_sub_loop(
        llm=ctx.llm,
        task={"goal": prompt},
        tools=tools,
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
