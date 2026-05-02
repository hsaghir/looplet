"""Scaffold a minimal looplet workspace skeleton in one call.

Use this from the host before invoking the agent_factory so the agent
starts with the boilerplate already in place and spends LLM turns on
the interesting work (writing tool bodies, system prompt, tests)
instead of writing the structural files that look the same in every
agent.

Programmatic use:

.. code-block:: python

    from looplet.scaffold import scaffold_workspace

    scaffold_workspace(
        path="/tmp/my_project/summarizer.workspace",
        name="summarizer",
        tools=["summarize", "extract_keywords"],
    )

After this returns, ``summarizer.workspace/`` contains a working
(but stubbed) workspace that ``workspace_to_preset()`` can load.
The agent only needs to fill in the TODO markers.

Files created:

* ``workspace.json`` — required metadata (one line of JSON)
* ``config.yaml`` — sensible defaults: max_steps=20, temperature=0.7
* ``prompts/system.md`` — stub with TODO markers
* ``tools/<name>/{tool.yaml,execute.py}`` for each tool requested
* ``tools/done/{tool.yaml,execute.py}`` — the standard finalizer
  (always added; every agent needs it)
"""

from __future__ import annotations

import json
from pathlib import Path

# ── file templates ───────────────────────────────────────────────


# ``{name_json}`` is filled with ``json.dumps(name)`` — produces
# valid JSON (double-quoted), unlike ``{name!r}`` which emits Python
# repr (single-quoted) and breaks ``json.loads(workspace.json)``.
_WORKSPACE_JSON = '{{"name": {name_json}, "schema_version": 1}}\n'

_CONFIG_YAML = """\
max_steps: 20
max_tokens: 2000
temperature: 0.7
done_tool: done
"""

_SYSTEM_MD = """\
# {title} Agent

<TODO: one-paragraph mission statement — what does this agent do?>

## Workflow

<TODO: numbered list of steps the agent should take>
{tool_workflow_hint}
- Always end with `done(summary=...)`.

## Tools

{tool_list_md}
- `done(summary)` — signals task completion.
"""

_TOOL_YAML = """\
name: {name}
description: |-
  TODO: one-line description, then optional Usage / Examples sections.

parameters: {{}}
"""

_TOOL_EXECUTE = '''\
"""TODO: one-line module summary for {name}."""


def execute(ctx, **kwargs) -> dict:
    """TODO: replace ``**kwargs`` with explicit keyword params, fill in body."""
    raise NotImplementedError("scaffold: implement {name}")
'''

_DONE_YAML = """\
name: done
description: |-
  Signal that the task is complete. Pass a one-line summary of what was accomplished.

parameters:
  summary:
    type: string
    description: One-line summary of the result.
"""

_DONE_EXECUTE = '''\
"""done tool — completion sentinel."""


def execute(*, summary: str) -> dict:
    """Mark the task as complete with a summary."""
    return {"summary": summary, "done": True}
'''


# ── main entry point ─────────────────────────────────────────────


def scaffold_workspace(
    path: str | Path,
    *,
    name: str,
    tools: list[str],
    overwrite: bool = False,
) -> Path:
    """Create a workspace skeleton at ``path``.

    Args:
        path: Directory to create (created if missing). Refuses to
            overwrite an existing non-empty directory unless
            ``overwrite=True``.
        name: Workspace name (becomes ``workspace.json`` "name" field
            and the title of the system prompt).
        tools: Tool names to scaffold under ``tools/<name>/``. The
            ``done`` tool is always added even if not listed; if you
            include "done" explicitly it is not duplicated.
        overwrite: When True, write into an existing directory. Files
            already present are NOT clobbered (so any agent edits
            survive a re-scaffold). Default False.

    Returns:
        The absolute path to the workspace.

    Raises:
        FileExistsError: If ``path`` exists and is non-empty and
            ``overwrite=False``.
        ValueError: If ``tools`` contains an empty string or names
            with characters that aren't valid Python identifiers.
    """
    root = Path(path)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"workspace path {root} already exists and is non-empty. "
            f"Pass overwrite=True to scaffold into it (existing files "
            f"will be preserved)."
        )

    # Validate tool names — they become directory names AND yaml
    # ``name:`` fields. Reject anything that wouldn't survive both.
    for t in tools:
        if not t:
            raise ValueError("tool name cannot be empty")
        if not t.replace("_", "").isalnum():
            raise ValueError(
                f"tool name {t!r} must be alphanumeric / underscore "
                f"(used as both a directory name and a yaml key)"
            )

    # Make sure ``done`` is in the set — every agent needs it.
    tool_set = list(dict.fromkeys([*tools, "done"]))

    root.mkdir(parents=True, exist_ok=True)
    _write_if_absent(
        root / "workspace.json",
        _WORKSPACE_JSON.format(name_json=json.dumps(name)),
    )
    _write_if_absent(root / "config.yaml", _CONFIG_YAML)

    prompts_dir = root / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    title = name.replace("_", " ").title()
    tool_workflow_hint = (
        "\n".join(
            f"{i + 1}. <TODO: when to call `{t}`>"
            for i, t in enumerate(t for t in tool_set if t != "done")
        )
        or "1. <TODO: first step>"
    )
    tool_list_md = "\n".join(f"- `{t}(...)` — <TODO: describe>" for t in tool_set if t != "done")
    _write_if_absent(
        prompts_dir / "system.md",
        _SYSTEM_MD.format(
            title=title,
            tool_workflow_hint=tool_workflow_hint,
            tool_list_md=tool_list_md,
        ),
    )

    tools_dir = root / "tools"
    tools_dir.mkdir(exist_ok=True)
    for tool in tool_set:
        tdir = tools_dir / tool
        tdir.mkdir(exist_ok=True)
        if tool == "done":
            _write_if_absent(tdir / "tool.yaml", _DONE_YAML)
            _write_if_absent(tdir / "execute.py", _DONE_EXECUTE)
        else:
            _write_if_absent(tdir / "tool.yaml", _TOOL_YAML.format(name=tool))
            _write_if_absent(tdir / "execute.py", _TOOL_EXECUTE.format(name=tool))

    return root.resolve()


def _write_if_absent(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` only if the file does not yet exist.

    Idempotent: re-running ``scaffold_workspace`` against an existing
    directory preserves any edits the agent (or user) has already made.
    """
    if not path.exists():
        path.write_text(content)


__all__ = ["scaffold_workspace"]
