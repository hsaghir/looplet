"""``scaffold_workspace`` built-in tool — agent-callable wrapper.

A workspace opts in via ``builtin_tools: [scaffold_workspace]`` in
its ``config.yaml``. The tool wraps :func:`looplet.scaffold.scaffold_workspace`
so an agent can scaffold a fresh workspace skeleton without leaving
its reasoning loop.

Use this when:

* The agent is generating a new workspace and the host did NOT
  pre-scaffold (so the path / name / tool list is the agent's choice).
* The agent is iterating: scaffold a base, customize, validate, fix,
  scaffold a new variant, etc.

When the host already knows what to build (e.g. ``looplet new
--name=foo --tools=a,b``) it should pre-scaffold via ``runtime``
kwargs instead — that saves the agent a tool turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from looplet.scaffold import scaffold_workspace as _scaffold
from looplet.tools import ToolSpec
from looplet.types import ToolContext


def _execute(
    ctx: ToolContext,
    *,
    path: str,
    name: str,
    tools: list[str],
    overwrite: bool = False,
) -> dict[str, Any]:
    # Resolve relative path against the host workspace if available.
    target_path = path
    cfg = ctx.resources.get("workspace_config") if ctx.resources else None
    host_ws = getattr(cfg, "path", None) if cfg is not None else None
    if host_ws and not Path(path).is_absolute():
        target_path = str(Path(host_ws) / path)

    try:
        result = _scaffold(target_path, name=name, tools=list(tools), overwrite=overwrite)
    except FileExistsError as exc:
        return {
            "error": f"FileExistsError: {exc}",
            "recovery": (
                "The path already has files. Pass overwrite=True to "
                "scaffold into it (existing files are preserved)."
            ),
        }
    except ValueError as exc:
        return {
            "error": f"ValueError: {exc}",
            "recovery": "Use only alphanumeric / underscore tool names.",
        }
    return {
        "scaffolded": True,
        "path": str(result),
        "tools_created": [*tools, "done"] if "done" not in tools else list(tools),
        "next_steps": (
            "Use multi_edit / edit_file to fill in the TODO markers in "
            "prompts/system.md, tools/<name>/tool.yaml, and "
            "tools/<name>/execute.py. Use validate_workspace(workspace_path) "
            "to confirm the result loads cleanly."
        ),
    }


SPEC = ToolSpec(
    name="scaffold_workspace",
    description=(
        "Create a stubbed looplet workspace skeleton at ``path`` in one call. "
        "Generates workspace.json, config.yaml, prompts/system.md, "
        "tools/<name>/{tool.yaml, execute.py} stubs (raise "
        "NotImplementedError) for each requested tool, plus the standard "
        "``done`` tool. Idempotent — re-running preserves files that "
        "already exist.\n\n"
        "Use this FIRST when generating a new workspace, then fill in "
        "the TODO markers via multi_edit / edit_file. Saves the 5-7 "
        "turns spent writing identical boilerplate by hand.\n\n"
        "Args:\n"
        "  path (str): directory to create. Relative paths are resolved "
        "against the host workspace.\n"
        "  name (str): becomes workspace.json.name and the title of the "
        "system prompt.\n"
        "  tools (list[str]): tool names to scaffold. ``done`` is added "
        "automatically.\n"
        "  overwrite (bool): write into a non-empty dir (existing files "
        "preserved). Default False.\n\n"
        "Returns: ``{scaffolded, path, tools_created, next_steps}`` "
        "or ``{error, recovery}``."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to create."},
            "name": {
                "type": "string",
                "description": "Workspace name (becomes workspace.json.name).",
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool names to scaffold (done is added automatically).",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow non-empty target dir (existing files preserved).",
                "default": False,
            },
        },
        "required": ["path", "name", "tools"],
    },
    requires=["workspace_config"],
    execute=_execute,
)
