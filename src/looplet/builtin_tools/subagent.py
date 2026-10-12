"""``subagent`` built-in tool for invoking another cartridge as a sub-loop.

A cartridge opts in by listing ``subagent`` in its ``config.yaml``::

    builtin_tools:
      - subagent

The agent can then dispatch a sub-task to another cartridge::

    subagent(
        workspace="./researcher.cartridge",
        task="find recent CVEs for the openssl 3.x line",
        max_steps=10,                # OPTIONAL, defaults to remaining parent budget
    )

The child shares the parent's backend and project root.
Async parents use the async child driver.
The tool closes its loaded preset on completion, failure, or cancellation.
Results include accepted output, execution status, steps, and child identity.
Parent permission decisions and host limits follow nested delegation.

The recursion counter is local to each task and restores on exit.
The default maximum depth is five.
Python hosts can use ``run_sub_loops_async`` for bounded parallel children.
Use separate child state and mutable hooks for concurrent runs.
State separation does not provide process or filesystem isolation.
"""

from __future__ import annotations

import contextvars
import hashlib
import time
from pathlib import Path
from typing import Any

from looplet.tools import ToolSpec
from looplet.types import ToolContext

# Per-task recursion depth - threadsafe and per-async-task, unlike a
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
) -> Any:
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
    if ctx.parent_budget_remaining is not None and ctx.parent_budget_remaining <= 0:
        return {"error": "parent step budget exhausted", "steps_used": 0}

    from looplet.subagent import (  # noqa: PLC0415
        ChildRunPolicy,
        _is_async_backend,
        _policy_from_tool_context,
        run_sub_loop,
        run_sub_loop_async,
    )

    policy = (ctx.metadata or {}).get("child_run_policy")
    try:
        policy = _policy_from_tool_context(ctx, policy)
    except (ValueError, AttributeError) as exc:
        return {"error": str(exc)}
    if policy is not None:
        if not isinstance(policy, ChildRunPolicy) or not policy.parent_id:
            return {"error": "child_run_policy must have a parent_id"}
        if policy.max_steps is not None and policy.max_steps <= 0:
            return {"error": "child_run_policy max_steps must be positive"}
        if policy.cancel_token is not None and ctx.cancel_token is not None:
            if policy.cancel_token is not ctx.cancel_token:
                return {"error": "child_run_policy must share the parent's cancellation token"}
        if policy.deadline is not None and time.monotonic() >= policy.deadline:
            return {"error": "child run deadline exceeded"}
        if (
            policy.recovery_dir is None
            and policy.model_budget is not None
            and policy.model_budget.remaining <= 0
        ):
            return {"error": "model call budget exhausted"}
    recovery_key = (ctx.metadata or {}).get("child_run_key")
    subagent_id = None
    if policy is not None and policy.recovery_dir is not None:
        if not isinstance(recovery_key, str) or not recovery_key or not policy.recovery_version:
            return {"error": "child recovery requires host child_run_key and recovery_version"}
        subagent_id = (
            "child-"
            + hashlib.sha256(f"{policy.parent_id}\0{recovery_key}".encode()).hexdigest()[:24]
        )

    # Resolve the host workspace path once. Both the path-resolution
    # below (for relative ``workspace=`` args) and the runtime
    # construction further down need it; doing the lookup here keeps
    # the two consumers in sync and avoids a duplicated try/getattr
    # dance.
    cfg = ctx.resources.get("workspace_config") if ctx.resources else None
    host_ws_str: str | None = getattr(cfg, "path", None) if cfg is not None else None

    # Resolve the target workspace path. Allow either absolute or
    # relative-to-the-host-workspace if a workspace_config resource is
    # available; otherwise relative to cwd.
    ws_path = Path(workspace)
    if not ws_path.is_absolute():
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
    from looplet import cartridge_to_preset  # noqa: PLC0415

    # Build a runtime dict for the sub-loop. The parent's
    # ``workspace_config.path`` is the canonical "where am I" value;
    # forward it so the sub-loop's resources (which read the project
    # root via :func:`looplet.cartridge.runtime_helpers.resolve_project_root`)
    # bind to the same project. Caller may override via
    # ctx.metadata['runtime'] when they want the sub-loop to operate
    # on a different project.
    metadata_runtime = (ctx.metadata or {}).get("runtime") if ctx.metadata else None
    runtime: dict[str, Any] = dict(metadata_runtime) if metadata_runtime else {}
    fallback_used = False
    if "project_root" not in runtime and "workspace" not in runtime:
        if host_ws_str:
            runtime["project_root"] = host_ws_str
        else:
            # No parent workspace_config and no explicit metadata.runtime -
            # the resolver in the sub-loop will fall back to git toplevel
            # / cwd. Flag it so the host knows the sub-loop is rooted in
            # whichever dir the process happens to be running from.
            fallback_used = True
    sub_preset = cartridge_to_preset(str(ws_path), runtime=runtime)

    try:
        if max_steps is not None and max_steps > 0:
            steps = max_steps
            sub_preset.config.max_steps = steps
        else:
            steps = sub_preset.config.max_steps
        if ctx.parent_budget_remaining is not None:
            steps = min(steps, ctx.parent_budget_remaining)
            sub_preset.config.max_steps = steps
        sub_preset.config.cancel_token = ctx.cancel_token
        if ctx.run_envelope is not None:
            sub_preset.config.run_envelope = ctx.run_envelope
        if policy is not None:
            if policy.max_steps is not None:
                steps = min(steps, policy.max_steps)
                sub_preset.config.max_steps = steps
        if hasattr(sub_preset.state, "max_steps"):
            sub_preset.state.max_steps = steps
    except BaseException:
        sub_preset.close()
        raise

    def summarize(state: Any, session_log: Any, child_steps: list[dict]) -> dict[str, Any]:
        final = child_steps[-1] if child_steps else {}
        final_data = (final.get("tool_result") or {}).get("data") or {}
        result = {
            "workspace": str(ws_path),
            "steps_used": len(child_steps),
            "max_steps": steps,
            "final_tool": (final.get("tool_call") or {}).get("tool"),
            "summary": final_data.get("summary") if isinstance(final_data, dict) else None,
            "result": final_data,
            "depth": depth + 1,
        }
        if fallback_used:
            result["warning"] = (
                "No parent project root was supplied. The project root will be resolved from host paths."
            )
        return result

    options = {
        "llm": ctx.async_llm if ctx.async_llm is not None else ctx.llm,
        "task": {"goal": task},
        "sub_tools": sub_preset.tools,
        "state": sub_preset.state,
        "config": sub_preset.config,
        "hooks": sub_preset.hooks,
        "parent_hooks": list(ctx.parent_hooks),
        "policy": policy,
        "subagent_id": subagent_id,
        "build_summary": summarize,
    }

    if _is_async_backend(options["llm"]):

        async def execute_async() -> dict[str, Any]:
            token = _DEPTH_VAR.set(depth + 1)
            try:
                return await run_sub_loop_async(**options)
            finally:
                _DEPTH_VAR.reset(token)
                sub_preset.close()

        return execute_async()

    token = _DEPTH_VAR.set(depth + 1)
    try:
        return run_sub_loop(**options)
    finally:
        _DEPTH_VAR.reset(token)
        sub_preset.close()


SPEC = ToolSpec(
    name="subagent",
    description=(
        "Invoke another Looplet cartridge as a sub-agent. The sub-agent "
        "shares this agent's LLM and inherits the parent's project "
        "path, runs to its own ``done`` "
        "tool, and returns the final result. Use this for hierarchical "
        "task decomposition: dispatch a focused sub-task to a cartridge "
        "that specializes in it, then continue with the result.\n\n"
        "Args:\n"
        "  workspace (str): path to a cartridge directory (absolute or "
        "relative to the host cartridge root).\n"
        "  task (str): natural-language task to give the sub-agent.\n"
        "  max_steps (int, optional): cap on sub-loop steps. Defaults to "
        "the child cartridge's own ``max_steps`` from its config.yaml.\n"
        "  max_depth (int, optional): recursion limit (default 5).\n\n"
        "Returns: ``{summary, result, final_tool, steps_used, ...}``."
    ),
    parameters={
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Path to the child cartridge (absolute or relative to host).",
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
