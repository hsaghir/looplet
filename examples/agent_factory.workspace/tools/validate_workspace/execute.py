"""validate_workspace tool — structural validator for generated workspaces.

Calls ``looplet.workspace_to_preset()`` against the path and reports
either the loaded tool/hook/config inventory (success) or a structured
error that names the failure mode and the likely file (failure). The
factory model uses this tool as a tight feedback loop: it writes files,
validates, fixes, and re-validates without burning steps on shell
gymnastics.
"""

from __future__ import annotations

from coder_lib_tools import _resolve_safe_path

from looplet.types import ToolContext


def execute(
    ctx: ToolContext,
    *,
    workspace_path: str,
    strict: bool = True,
) -> dict:
    cfg = ctx.resources.get("workspace_config")
    workspace = cfg.path if cfg is not None else "."
    abs_path = _resolve_safe_path(workspace, workspace_path)
    if abs_path is None:
        return {
            "error": f"Path {workspace_path!r} is outside the project directory.",
        }
    if not abs_path.is_dir():
        return {
            "error": f"{workspace_path!r} is not a directory.",
            "recovery": "Pass the workspace root (the directory containing workspace.json).",
        }

    # Defer the looplet import to call-time so the tool stays
    # importable in environments where looplet isn't on the path
    # (e.g. test harnesses that mock the dispatcher).
    from looplet import workspace_to_preset  # noqa: PLC0415
    from looplet.workspace import WorkspaceSerializationError  # noqa: PLC0415

    try:
        preset = workspace_to_preset(str(abs_path), strict=strict)
    except FileNotFoundError as exc:
        return {
            "error": f"FileNotFoundError: {exc}",
            "missing": "workspace.json",
            "recovery": (
                f"Create {workspace_path}/workspace.json with content "
                '`{"name": "<agent-name>", "schema_version": 1}`.'
            ),
        }
    except WorkspaceSerializationError as exc:
        return {
            "error": f"WorkspaceSerializationError: {exc}",
            "kind": "serialization",
            "recovery": (
                "The message names the offending file. Read it, fix the "
                "shape (yaml indent / required keys), and re-validate."
            ),
        }
    except ImportError as exc:
        return {
            "error": f"ImportError while loading: {exc}",
            "kind": "import",
            "recovery": ("A tool/hook execute.py has a bad import. Read it and fix."),
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "kind": "other",
        }

    tools = sorted(preset.tools._tools.keys())
    hooks = [type(h).__name__ for h in preset.hooks]
    sys_prompt = preset.config.system_prompt or ""
    sys_prompt_chars = len(sys_prompt)

    # Detect scaffolded-but-unfilled artifacts. The scaffolder writes
    # ``<TODO: ...>`` markers in the system prompt and
    # ``raise NotImplementedError("scaffold: implement <name>")`` in
    # tool bodies; if either survives, the agent has not finished
    # the work and should not declare done.
    todo_in_prompt = "<TODO:" in sys_prompt or "TODO:" in sys_prompt[:600]
    scaffold_stubs: list[str] = []
    tools_dir = abs_path / "tools"
    if tools_dir.is_dir():
        for tool_dir in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
            execute_py = tool_dir / "execute.py"
            if not execute_py.is_file():
                continue
            body = execute_py.read_text(encoding="utf-8", errors="replace")
            if 'NotImplementedError("scaffold: implement' in body:
                scaffold_stubs.append(tool_dir.name)

    return {
        "valid": True,
        "workspace_path": workspace_path,
        "tools": tools,
        "n_tools": len(tools),
        "hooks": hooks,
        "n_hooks": len(hooks),
        "max_steps": preset.config.max_steps,
        "system_prompt_chars": sys_prompt_chars,
        "warnings": [
            warning
            for warning in (
                "system_prompt is empty — add prompts/system.md" if sys_prompt_chars == 0 else None,
                "no `done` tool — every agent must have one" if "done" not in tools else None,
                "system_prompt still has TODO markers — fill them in" if todo_in_prompt else None,
                (
                    f"tools still raise NotImplementedError (unfilled scaffolds): "
                    f"{', '.join(scaffold_stubs)}"
                )
                if scaffold_stubs
                else None,
            )
            if warning is not None
        ],
    }
