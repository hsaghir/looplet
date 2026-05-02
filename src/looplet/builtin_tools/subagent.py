"""``subagent`` built-in tool — invoke another workspace as a sub-loop.

A workspace opts in by listing ``subagent`` in its ``config.yaml``::

    builtin_tools:
      - subagent

The agent can then dispatch a sub-task to any other workspace::

    subagent(
        workspace="./researcher.workspace",
        task="find recent CVEs for the openssl 3.x line",
        max_steps=10,                # OPTIONAL, defaults to remaining parent budget
    )

The sub-loop runs synchronously, sharing the parent's ``llm`` and
``runtime`` (so the same workspace_config / file_cache is in scope).
The result returned to the parent is the sub-loop's final tool result
(typically the ``done`` summary).

## Recursion safety

A small ``LOOPLET_SUBAGENT_DEPTH`` env-var counter increments on each
sub-loop entry and decrements on exit. If it exceeds
``max_depth`` (default 5) the call is refused with a structured error
pointing the agent at the depth budget. This prevents an agent from
spawning itself indefinitely.

## Why no parallel fan-out

We deliberately ship the sequential case only. ``subagent(...)`` calls
can be chained in the parent's reasoning ("dispatch to A, then dispatch
to B"). Parallel execution requires a real use case that motivates the
extra surface area; for now, ``async_composable_loop`` is the right
place to express concurrency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from looplet.tools import ToolSpec
from looplet.types import DefaultState, ToolContext

_DEPTH_ENV = "LOOPLET_SUBAGENT_DEPTH"
DEFAULT_MAX_DEPTH = 5


def _execute(
    ctx: ToolContext,
    *,
    workspace: str,
    task: str,
    max_steps: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict:
    # Recursion guard.
    depth = int(os.environ.get(_DEPTH_ENV, "0"))
    if depth >= max_depth:
        return {
            "error": (
                f"sub-agent depth {depth + 1} would exceed max_depth={max_depth}. "
                "Stop spawning sub-agents recursively."
            ),
            "depth": depth,
            "max_depth": max_depth,
        }

    # Resolve the target workspace path. Allow either absolute or
    # relative-to-the-host-workspace if a workspace_config resource is
    # available; otherwise relative to cwd.
    ws_path = Path(workspace)
    if not ws_path.is_absolute():
        # Try the host workspace if we can find one in resources.
        host_ws = None
        try:
            cfg = ctx.metadata.get("workspace_config") if ctx.metadata else None
            host_ws = getattr(cfg, "path", None) if cfg is not None else None
        except Exception:
            host_ws = None
        if host_ws:
            ws_path = Path(host_ws) / workspace
        else:
            ws_path = Path.cwd() / workspace
    if not ws_path.is_dir():
        return {
            "error": f"sub-agent workspace not found at {ws_path!s}",
            "workspace": str(ws_path),
        }

    # Defer the heavy imports so this module remains cheap to import.
    from looplet import composable_loop, workspace_to_preset  # noqa: PLC0415

    # Inherit runtime from the parent if we can. The standard hand-off
    # is via ctx.metadata["runtime"], which the workspace loader sets.
    runtime = (ctx.metadata or {}).get("runtime", {})
    sub_preset = workspace_to_preset(str(ws_path), runtime=dict(runtime))

    # Sub-loop budget: explicit ``max_steps`` overrides; otherwise
    # inherit from sub_preset's own config.
    if max_steps > 0:
        steps = max_steps
        # Apply the override to the sub-preset's config so the loop
        # honours it and ``DefaultState(max_steps=...)`` matches.
        sub_preset.config.max_steps = steps
    else:
        steps = sub_preset.config.max_steps

    # Bump depth env-var so any nested subagent calls see the updated
    # depth. We restore the old value after the sub-loop returns.
    prev_depth = os.environ.get(_DEPTH_ENV)
    os.environ[_DEPTH_ENV] = str(depth + 1)

    state = DefaultState(max_steps=steps)
    last_step: Any = None
    sub_steps = 0
    try:
        for step in composable_loop(
            llm=ctx.llm,
            config=sub_preset.config,
            tools=sub_preset.tools,
            state=state,
            hooks=sub_preset.hooks,
            task={"goal": task},
        ):
            sub_steps += 1
            last_step = step
    finally:
        # Restore parent's depth (or remove if we set it from absent).
        if prev_depth is None:
            os.environ.pop(_DEPTH_ENV, None)
        else:
            os.environ[_DEPTH_ENV] = prev_depth

    # Surface the final tool result. By convention sub-agents end with
    # ``done(summary=...)``, so we expose the summary at the top level
    # for easy chaining.
    final_data: dict = {}
    final_tool: str | None = None
    if last_step is not None and last_step.tool_call is not None:
        final_tool = last_step.tool_call.tool
        if last_step.tool_result is not None and last_step.tool_result.data:
            final_data = dict(last_step.tool_result.data)

    return {
        "workspace": str(ws_path),
        "steps_used": sub_steps,
        "max_steps": steps,
        "final_tool": final_tool,
        "summary": final_data.get("summary"),
        "result": final_data,
        "depth": depth + 1,
    }


SPEC = ToolSpec(
    name="subagent",
    description=(
        "Invoke another looplet workspace as a sub-agent. The sub-agent "
        "shares this agent's LLM and runtime, runs to its own ``done`` "
        "tool, and returns the final result. Use this for hierarchical "
        "task decomposition: dispatch a focused sub-task to a workspace "
        "that specializes in it, then continue with the result.\n\n"
        "Args:\n"
        "  workspace (str): path to a workspace directory (absolute or "
        "relative to the host workspace root).\n"
        "  task (str): natural-language task to give the sub-agent.\n"
        "  max_steps (int, optional): cap on sub-loop steps. Defaults to "
        "the sub-workspace's own ``max_steps`` from its config.yaml.\n"
        "  max_depth (int, optional): recursion limit (default 5).\n\n"
        "Returns: ``{summary, result, final_tool, steps_used, ...}``."
    ),
    parameters={
        "workspace": "str — path to the sub-agent workspace",
        "task": "str — natural-language task for the sub-agent",
        "max_steps": "int — optional cap on sub-loop steps",
        "max_depth": "int — recursion limit (default 5)",
    },
    execute=_execute,
)
