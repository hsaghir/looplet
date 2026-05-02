"""Built-in tools any looplet workspace can opt into.

A workspace enables built-ins by listing them in ``config.yaml``::

    builtin_tools:
      - subagent
      - scaffold_workspace

The loader looks each name up here at workspace-load time and
registers it in the tool registry alongside the workspace's
own ``tools/<name>/`` directories.

Built-ins live here (rather than in every workspace's ``tools/``)
so they evolve with looplet: a new release ships an improved tool
and every workspace using ``builtin_tools:`` picks it up
immediately, no per-workspace edit needed.

Currently shipped built-ins:

* ``subagent`` — invoke another workspace as a synchronous sub-loop.
* ``scaffold_workspace`` — create a stubbed workspace skeleton in one
  call (agent-callable wrapper around :func:`looplet.scaffold.scaffold_workspace`).

Adding a new built-in: write a small module exposing a ``SPEC``
:class:`looplet.tools.ToolSpec`, then list its ``name`` in
:data:`AVAILABLE` below.
"""

from __future__ import annotations

from looplet.builtin_tools.scaffold_workspace import SPEC as _SCAFFOLD_SPEC
from looplet.builtin_tools.subagent import SPEC as _SUBAGENT_SPEC
from looplet.tools import ToolSpec

AVAILABLE: dict[str, ToolSpec] = {
    _SUBAGENT_SPEC.name: _SUBAGENT_SPEC,
    _SCAFFOLD_SPEC.name: _SCAFFOLD_SPEC,
}


def get_builtin_tool(name: str) -> ToolSpec | None:
    """Return the :class:`ToolSpec` for a built-in tool by name, or None."""
    return AVAILABLE.get(name)


__all__ = ["AVAILABLE", "get_builtin_tool"]
