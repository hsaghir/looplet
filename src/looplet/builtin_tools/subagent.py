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
constructing a fresh ``runtime`` that defaults to the parent's
``workspace_config.path`` (so resource builders like
``resources/file_cache.py`` bind to the same project root). Other
runtime values can be overridden via ``ctx.metadata["runtime"]``.
The result returned to the parent is the sub-loop's final tool
result (typically the ``done`` summary).

## Recursion safety

A small ``contextvars.ContextVar`` counter increments on each sub-loop
entry and decrements on exit. If it exceeds ``max_depth`` (default 5)
the call is refused with a structured error pointing the agent at the
depth budget. Threadsafe and per-async-task — two parallel parent
loops in the same process don't share the counter.

## Why no parallel fan-out

We deliberately ship the sequential case only. ``subagent(...)`` calls
can be chained in the parent's reasoning ("dispatch to A, then dispatch
to B"). Parallel execution requires a real use case that motivates the
extra surface area; for now, ``async_composable_loop`` is the right
place to express concurrency.
"""

from __future__ import annotations

import contextvars
from pathlib import Path
from typing import Any

from looplet.tools import ToolSpec
from looplet.types import DefaultState, ToolContext

# Per-task recursion depth — threadsafe and per-async-task, unlike a
# process-global env var. ``ContextVar.set`` returns a token used by
# ``reset`` so nested sub-loops restore depth precisely on exit.
_DEPTH_VAR: contextvars.ContextVar[int] = contextvars.ContextVar(
    "looplet_subagent_depth", default=0
)
DEFAULT_MAX_DEPTH = 5


def _execute(
    ctx: ToolContext,
    *,
    workspace: str,
    task: str,
    max_steps: int | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict:
    # Recursion guard.
    depth = _DEPTH_VAR.get()
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
    host_ws_str: str | None = None
    if not ws_path.is_absolute():
        cfg = ctx.resources.get("workspace_config") if ctx.resources else None
        host_ws_str = getattr(cfg, "path", None) if cfg is not None else None
        if host_ws_str:
            ws_path = Path(host_ws_str) / workspace
        else:
            ws_path = Path.cwd() / workspace
    if not ws_path.is_dir():
        return {
            "error": f"sub-agent workspace not found at {ws_path!s}",
            "workspace": str(ws_path),
        }

    # Defer the heavy imports so this module remains cheap to import.
    from looplet import composable_loop, workspace_to_preset  # noqa: PLC0415

    # Build a runtime dict for the sub-loop. The parent's
    # ``workspace_config.path`` is the canonical "where am I" value;
    # forward it so the sub-loop's resources/file_cache.py builders
    # (which read ``runtime['workspace']``) bind to the same project
    # root. Caller may override via ctx.metadata['runtime'] when they
    # want the sub-loop to operate on a different workspace.
    if host_ws_str is None:
        cfg = ctx.resources.get("workspace_config") if ctx.resources else None
        host_ws_str = getattr(cfg, "path", None) if cfg is not None else None
    metadata_runtime = (ctx.metadata or {}).get("runtime") if ctx.metadata else None
    runtime: dict[str, Any] = dict(metadata_runtime) if metadata_runtime else {}
    fallback_used = False
    if "workspace" not in runtime:
        if host_ws_str:
            runtime["workspace"] = host_ws_str
        else:
            # No parent workspace_config and no explicit metadata.runtime —
            # fall back to cwd, but make it loud so the host knows the
            # sub-loop is rooted in whichever dir the process happens to
            # be running from.
            runtime["workspace"] = str(Path.cwd())
            fallback_used = True
    sub_preset = workspace_to_preset(str(ws_path), runtime=runtime)

    # Sub-loop budget: explicit ``max_steps`` overrides; otherwise
    # inherit from sub_preset's own config.
    if max_steps is not None and max_steps > 0:
        steps = max_steps
        # Apply the override to the sub-preset's config so the loop
        # honours it and ``DefaultState(max_steps=...)`` matches.
        sub_preset.config.max_steps = steps
    else:
        steps = sub_preset.config.max_steps

    # Bump depth for any nested subagent calls inside this run.
    token = _DEPTH_VAR.set(depth + 1)

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
        _DEPTH_VAR.reset(token)

    # Surface the final tool result. By convention sub-agents end with
    # ``done(summary=...)``, so we expose the summary at the top level
    # for easy chaining.
    final_data: dict = {}
    final_tool: str | None = None
    if last_step is not None and last_step.tool_call is not None:
        final_tool = last_step.tool_call.tool
        if last_step.tool_result is not None and last_step.tool_result.data:
            final_data = dict(last_step.tool_result.data)

    result: dict[str, Any] = {
        "workspace": str(ws_path),
        "steps_used": sub_steps,
        "max_steps": steps,
        "final_tool": final_tool,
        "summary": final_data.get("summary"),
        "result": final_data,
        "depth": depth + 1,
    }
    if fallback_used:
        result["warning"] = (
            "no workspace_config resource on the parent and no "
            "ctx.metadata['runtime'] — sub-loop's runtime['workspace'] "
            f"defaulted to cwd ({runtime['workspace']!r}). Pass "
            "runtime={'workspace': '...'} to workspace_to_preset for "
            "the parent, or set ctx.metadata['runtime'] before calling."
        )
    return result


SPEC = ToolSpec(
    name="subagent",
    description=(
        "Invoke another looplet workspace as a sub-agent. The sub-agent "
        "shares this agent's LLM and inherits the parent's workspace "
        "path, runs to its own ``done`` "
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
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Path to the sub-agent workspace (absolute or relative to host).",
            },
            "task": {
                "type": "string",
                "description": "Natural-language task to give the sub-agent.",
            },
            "max_steps": {
                "type": "integer",
                "description": (
                    "Optional cap on sub-loop steps. Omit to inherit "
                    "the sub-workspace's own ``max_steps`` from its "
                    "config.yaml."
                ),
            },
            "max_depth": {
                "type": "integer",
                "description": "Recursion limit (default 5).",
                "default": 5,
            },
        },
        "required": ["workspace", "task"],
    },
    requires=["workspace_config"],
    execute=_execute,
)
